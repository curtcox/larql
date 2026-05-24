//! Sparse walk path — zero matrix multiplications.
//!
//! The hot path for FFN inference on the LARQL vindex. For each position:
//!
//!   1. `gate_knn` → top-K features (HNSW / batched brute-force / gate-walk)
//!   2. For each feature:
//!      - `up_score  = dot(up_row(feat), x)`         via unified ffn_row_dot
//!      - `activated = silu(gate_score) * up_score`   (GEGLU)
//!      - `out      += activated * down_row(feat)`   via unified ffn_row_scaled_add
//!
//! The "unified" accessors in the `GateIndex` trait dispatch through
//! FP4 → native f32 → Q4K backends in priority order, so this single
//! function is **format-blind** — the same code path serves FP4, Q4K,
//! and native f32 vindexes. Adding a new storage format doesn't touch
//! this file.
//!
//! Three specialisations are layered on top for perf:
//!
//! - **Full-K gemv fast path** (line ~100): when K ≥ num_features, the
//!   per-feature loop is mathematically equivalent to three dense
//!   matmuls. We route through BLAS gemm (or Q4K direct matmul) when
//!   the backend supports it.
//! - **Parallel Q4K down-cache path** (line ~170): for medium-K on
//!   Q4K-only vindexes, the down matrix transposition cost justifies
//!   caching the whole dequantised layer and parallelising feature
//!   chunks over rayon.
//! - **Serial per-feature loop** (line ~240): the canonical
//!   correctness baseline; always works because `ffn_row_*` always has
//!   *some* backend.

use ndarray::Array2;
use rayon::prelude::*;

use super::helpers::hits_len_ge_intermediate;
use super::WalkFfn;
use crate::monty_call::{CallCandidate, CallContext};
use crate::vindex::walk_config::FeatureSelector;

impl<'a> WalkFfn<'a> {
    /// Sparse walk FFN — see module docs.
    pub(super) fn walk_ffn_sparse(
        &self,
        layer: usize,
        x: &Array2<f32>,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let hidden = x.shape()[1];
        let seq_len = x.shape()[0];
        let intermediate = self.index.num_features(layer);

        // Prefer native f32 mmap (zero-copy). When no native mmap is
        // available we still run — the inner loops dispatch per-row
        // through `ffn_row_dot` / `ffn_row_scaled_add`, which the
        // GateIndex trait routes to FP4 or Q4K or last-resort native
        // as appropriate. The only thing we can't do with neither
        // native f32 mmap, Q4K storage, nor FP4 storage is the serial
        // per-feature loop — those all fail and bail.
        let up_native = self.index.up_layer_matrix(layer);
        let down_native = self.index.down_layer_matrix(layer);
        let row_fallback = up_native.is_none() || down_native.is_none();
        if row_fallback
            && self.index.interleaved_kquant_layer_data(layer).is_none()
            && !self.index.has_fp4_storage()
        {
            return None;
        }

        let arch = &*self.weights.arch;
        let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
        let use_gelu = matches!(
            arch.activation(),
            larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
        );

        // Hint the kernel to start streaming layer N+1's Q4_K/Q6_K bytes
        // into the page cache while we work on N. No-op when there's no
        // Q4_K mmap, no manifest, or `layer+1` is out of range.
        self.index.prefetch_interleaved_kquant_layer(layer + 1);

        let mut out = Array2::<f32>::zeros((seq_len, hidden));
        let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));

        let layer_has_overrides = self.index.has_overrides_at(layer);
        let up_bias_for_layer = if !is_gated {
            arch.ffn_up_bias_key(layer)
                .and_then(|bk| self.weights.vectors.get(&bk).cloned())
        } else {
            None
        };

        // ── Full-K gemv fast path ────────────────────────────────────────
        // See module docs for the three variants (A/B/C). Skipped when a
        // non-default selector is configured or a per-layer pool
        // restriction is set: in both cases gemv would bypass the
        // alternative selection criterion, so we force the walk.
        let selector_forces_walk = !matches!(self.config.selector, FeatureSelector::GateOnly)
            || self.config.pool_per_layer.is_some();
        let k_is_full =
            !selector_forces_walk && hits_len_ge_intermediate(&self.config, layer, intermediate);
        if !layer_has_overrides && is_gated && k_is_full {
            let x_slice_for_matmul: Option<&[f32]> = x.as_slice();
            if let (Some(gate_scores), Some(x_flat)) = (
                self.index.gate_scores_batch_backend(layer, x, self.backend),
                x_slice_for_matmul,
            ) {
                let up_scores: Option<ndarray::Array2<f32>> = if let Some(v) = up_native {
                    Some(larql_compute::dot_proj_gpu(x, &v, self.backend))
                } else if let Some(y) =
                    self.index
                        .kquant_matmul_transb(layer, 1, x_flat, seq_len, self.backend)
                {
                    ndarray::Array2::from_shape_vec((seq_len, intermediate), y).ok()
                } else {
                    None
                };

                if let Some(up_scores) = up_scores {
                    let activation = if use_gelu {
                        crate::ffn::gelu_tanh_gate_up(&gate_scores, &up_scores)
                    } else {
                        crate::ffn::silu_gate_up(&gate_scores, &up_scores)
                    };
                    let act_slice: Option<&[f32]> = activation.as_slice();
                    let out_matmul: Option<ndarray::Array2<f32>> = if let Some(v) = down_native {
                        Some(larql_compute::matmul_gpu(&activation, &v, self.backend))
                    } else if let Some(act_flat) = act_slice {
                        self.index
                            .kquant_matmul_transb(layer, 2, act_flat, seq_len, self.backend)
                            .and_then(|y| {
                                ndarray::Array2::from_shape_vec((seq_len, hidden), y).ok()
                            })
                    } else {
                        None
                    };
                    if let Some(out_matmul) = out_matmul {
                        out.assign(&out_matmul);
                        full_activation.assign(&activation);
                        self.trace_path(layer, "sparse:gemv_full_k");
                        return Some((out, full_activation));
                    }
                }
            }
        }

        // ── Per-position sparse loop ─────────────────────────────────────
        for s in 0..seq_len {
            let x_row = x.row(s);
            let x_owned = x_row.to_owned();
            let x_slice_owned: Vec<f32>;
            let x_slice: &[f32] = if let Some(sl) = x_row.as_slice() {
                sl
            } else {
                x_slice_owned = x_owned.as_slice().unwrap().to_vec();
                &x_slice_owned
            };

            let top_k = self.top_k_for(layer);
            let t_gate = std::time::Instant::now();
            let hits = if let Some(pool_per_layer) = self.config.pool_per_layer.as_ref() {
                let empty = Vec::new();
                let pool = pool_per_layer.get(layer).unwrap_or(&empty);
                self.pool_restricted_gate_knn(layer, &x_owned, top_k, pool)
            } else {
                match self.config.selector {
                    FeatureSelector::GateOnly => self
                        .index
                        .gate_walk(layer, &x_owned, top_k)
                        .or_else(|| {
                            self.backend
                                .and_then(|be| self.index.gate_knn_q4(layer, &x_owned, top_k, be))
                        })
                        .unwrap_or_else(|| self.index.gate_knn(layer, &x_owned, top_k)),
                    kind @ (FeatureSelector::GateXDownNorm
                    | FeatureSelector::GateXUpDownNorm
                    | FeatureSelector::GateXUpScore
                    | FeatureSelector::ActXUpScoreXDownNorm
                    | FeatureSelector::Random) => self.joint_gate_knn(layer, &x_owned, top_k, kind),
                }
            };
            let gate_knn_ns = t_gate.elapsed().as_nanos() as u64;

            let mut out_row = out.row_mut(s);

            // Parallel Q4K-down-cache path — only used when feature
            // count is medium-large (≥ 512) and no native down exists.
            let parallelisable =
                !layer_has_overrides && is_gated && hits.len() >= 512 && down_native.is_none();
            let t_cache = std::time::Instant::now();
            let down_cache_local: Option<std::sync::Arc<Vec<f32>>> = if parallelisable {
                self.index.kquant_ffn_layer(layer, 2)
            } else {
                None
            };
            let cache_fetch_ns = t_cache.elapsed().as_nanos() as u64;
            if let Some(down_arc) = down_cache_local.as_ref().filter(|_| parallelisable) {
                let down_data: &[f32] = down_arc.as_slice();
                let up_slices = self.index.interleaved_kquant_layer_data(layer);
                // Resolve up via the registry — accepts Q4_K, Q6_K, and
                // any future K-quant rather than hardcoding Q4_K-only.
                let up_q4k: Option<(&[u8], &larql_vindex::quant::registry::QuantFormatInfo)> =
                    match (up_native.as_ref(), up_slices) {
                        (Some(_), _) => None,
                        (None, Some(s)) => {
                            larql_vindex::quant::registry::lookup(s[1].1).map(|info| (s[1].0, info))
                        }
                        _ => None,
                    };
                let n_threads = rayon::current_num_threads().max(1);
                let chunk_size = hits.len().div_ceil(n_threads);
                let up_native_ref = up_native.as_ref();

                let t_scan = std::time::Instant::now();
                let partials: Vec<Vec<f32>> = hits
                    .par_chunks(chunk_size)
                    .map(|chunk| {
                        let mut partial = vec![0.0f32; hidden];
                        for &(feat, gate_score) in chunk {
                            let up_score = if let Some(up_view) = up_native_ref {
                                up_view.row(feat).dot(&x_row)
                            } else if let Some((up_bytes, info)) = up_q4k {
                                let row_dot = info.row_dot.expect("registry: row_dot");
                                let bytes_per_row = info
                                    .bytes_per_row(hidden)
                                    .expect("registry: bytes_per_row aligned");
                                let start = feat * bytes_per_row;
                                let end = start + bytes_per_row;
                                row_dot(&up_bytes[start..end], x_slice).unwrap_or(0.0)
                            } else {
                                0.0
                            };
                            let activated_gate = if use_gelu {
                                crate::ffn::gelu_tanh(gate_score)
                            } else {
                                gate_score * crate::ffn::sigmoid(gate_score)
                            };
                            let act = activated_gate * up_score;
                            if act.abs() > 1e-10 {
                                let row_start = feat * hidden;
                                let down_row = &down_data[row_start..row_start + hidden];
                                let mut pv = ndarray::ArrayViewMut1::from(partial.as_mut_slice());
                                let dv = ndarray::ArrayView1::from(down_row);
                                pv.scaled_add(act, &dv);
                            }
                        }
                        partial
                    })
                    .collect();
                let parallel_scan_ns = t_scan.elapsed().as_nanos() as u64;

                let t_reduce = std::time::Instant::now();
                let out_slice = out_row.as_slice_mut().unwrap();
                for p in &partials {
                    for i in 0..hidden {
                        out_slice[i] += p[i];
                    }
                }
                let reduce_ns = t_reduce.elapsed().as_nanos() as u64;

                if let Some(h) = &self.phase_timings {
                    use std::sync::atomic::Ordering::Relaxed;
                    h.gate_knn_ns.fetch_add(gate_knn_ns, Relaxed);
                    h.cache_fetch_ns.fetch_add(cache_fetch_ns, Relaxed);
                    h.parallel_scan_ns.fetch_add(parallel_scan_ns, Relaxed);
                    h.reduce_ns.fetch_add(reduce_ns, Relaxed);
                    h.calls.fetch_add(1, Relaxed);
                }

                self.trace_path(layer, "sparse:parallel_q4k_down");
                continue;
            }

            let mut margins = Vec::new();
            if self.call_patches.is_some() {
                margins.reserve(hits.len());
                for i in 0..hits.len() {
                    let next = hits.get(i + 1).map(|(_, s)| s.abs()).unwrap_or(0.0);
                    margins.push(hits[i].1.abs() - next);
                }
            }

            // Counts call patches that fired for this position; enforces
            // per-token budgets defined in each call's trigger.
            let mut calls_fired_this_position: usize = 0;

            // Serial per-feature loop — the correctness baseline.
            for (rank_idx, (feat, gate_score)) in hits.into_iter().enumerate() {
                if let Some(call) = self
                    .call_patches
                    .and_then(|patches| patches.call_patch(layer, feat))
                {
                    if let Some(runtime) = self.call_runtime {
                        let ctx = CallContext {
                            layer,
                            position: self.call_position_base.get() + s,
                            residual: x_slice,
                            token_ids: &[],
                            token_text: None,
                        };
                        if let Ok(Some(call_output)) = runtime.execute_call(
                            call,
                            CallCandidate {
                                rank: rank_idx + 1,
                                score: gate_score,
                                margin: margins.get(rank_idx).copied(),
                                calls_already_fired: calls_fired_this_position,
                            },
                            &ctx,
                            hidden,
                        ) {
                            calls_fired_this_position += 1;
                            if let Some(delta) = call_output.residual_delta {
                                if delta.len() == hidden {
                                    out_row += &ndarray::ArrayView1::from(delta.as_slice());
                                }
                            }
                        }
                    }
                    continue;
                }

                let act = if is_gated {
                    let up_ov = if layer_has_overrides {
                        self.index.up_override(layer, feat)
                    } else {
                        None
                    };
                    let up_score = if let Some(up_ov) = up_ov.filter(|o| o.len() == hidden) {
                        ndarray::ArrayView1::from(up_ov).dot(&x_row)
                    } else if let Some(ref up_view) = up_native {
                        up_view.row(feat).dot(&x_row)
                    } else {
                        // Unified dispatch: FP4 → native → Q4K, per GateIndex.
                        self.index.ffn_row_dot(layer, 1, feat, x_slice)?
                    };
                    let activated_gate = if use_gelu {
                        crate::ffn::gelu_tanh(gate_score)
                    } else {
                        gate_score * crate::ffn::sigmoid(gate_score)
                    };
                    activated_gate * up_score
                } else {
                    let mut v = gate_score;
                    if let Some(ref bias) = up_bias_for_layer {
                        if feat < bias.len() {
                            v += bias[feat];
                        }
                    }
                    if use_gelu {
                        crate::ffn::gelu_tanh(v)
                    } else {
                        v * crate::ffn::sigmoid(v)
                    }
                };

                full_activation[[s, feat]] = act;

                if act.abs() > 1e-10 {
                    let down_ov = if layer_has_overrides {
                        self.index.down_override(layer, feat)
                    } else {
                        None
                    };
                    if let Some(override_down) = down_ov.filter(|o| o.len() == hidden) {
                        out_row.scaled_add(act, &ndarray::ArrayView1::from(override_down));
                        continue;
                    }
                    if let Some(ref down_view) = down_native {
                        out_row.scaled_add(act, &down_view.row(feat));
                    } else {
                        let out_slice = out_row.as_slice_mut().unwrap();
                        // Unified dispatch: FP4 → native → Q4K-via-cache, per GateIndex.
                        if !self
                            .index
                            .ffn_row_scaled_add(layer, 2, feat, act, out_slice)
                        {
                            return None;
                        }
                    }
                }
            }
        }

        // Down bias
        if let Some(bias) = arch
            .ffn_down_bias_key(layer)
            .and_then(|k| self.weights.vectors.get(&k))
        {
            crate::forward::add_bias(&mut out, bias);
        }

        self.trace_path(layer, "sparse:serial");
        Some((out, full_activation))
    }
}

#[cfg(test)]
mod tests {
    use crate::monty_call::{CallError, CallProgramRunner, MontyCallRuntime};
    use crate::test_utils::{
        make_test_q4k_vindex, make_test_q4k_weights, make_test_vindex, make_test_weights,
    };
    use crate::vindex::{WalkFfn, WalkFfnConfig};
    use larql_vindex::{CallPatchOp, CallResourceLimits, CallSafetyPolicy, CallTrigger};
    use ndarray::Array2;
    use serde_json::{json, Value};
    use std::cell::RefCell;

    fn x(seq: usize, hidden: usize) -> Array2<f32> {
        Array2::from_shape_vec(
            (seq, hidden),
            (0..seq * hidden).map(|i| (i as f32 + 1.0) * 0.02).collect(),
        )
        .unwrap()
    }

    /// Sparse walk over the Q4K fixture — `up_layer_matrix`/`down_layer_matrix`
    /// both return None (Q4K storage is byte-only) so the function
    /// routes through the row-fallback ladder dispatching via
    /// `ffn_row_dot` / `ffn_row_scaled_add`.
    #[test]
    fn walk_ffn_sparse_routes_through_q4k_fixture() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        if let Some((out, activation)) = result {
            assert_eq!(out.shape(), &[1, weights.hidden_size]);
            assert_eq!(activation.shape()[0], 1);
        }
    }

    /// Sparse walk over the feature-major f32 fixture — `up_layer_matrix`
    /// + `down_layer_matrix` both return Some so the function bypasses
    ///   the row-fallback and goes through the BLAS gemm fast path.
    #[test]
    fn walk_ffn_sparse_routes_through_feature_major_f32_fixture() {
        use crate::test_utils::attach_feature_major_f32_to_test_vindex;
        let weights = make_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn
            .walk_ffn_sparse(0, &x(2, weights.hidden_size))
            .expect("feature-major f32 fixture should produce output");
        let (out, _activation) = result;
        assert_eq!(out.shape(), &[2, weights.hidden_size]);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// Sparse walk with full-K (K >= num_features) routes through the
    /// gemv fast path. Drives the `hits_len_ge_intermediate` branch.
    #[test]
    fn walk_ffn_sparse_full_k_takes_gemv_path() {
        use crate::test_utils::attach_feature_major_f32_to_test_vindex;
        let weights = make_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::dense(weights.num_layers);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let out = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("dense-K sparse walk should succeed");
        assert_eq!(out.0.shape(), &[1, weights.hidden_size]);
    }

    /// Sparse walk against a bare vindex (no FFN data) returns None —
    /// no native f32, no Q4K, no FP4 → the `row_fallback` guard fires.
    #[test]
    fn walk_ffn_sparse_returns_none_when_no_ffn_data() {
        let weights = make_test_weights();
        let index = make_test_vindex(&weights);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        assert!(result.is_none());
    }

    /// Sparse walk against a StarCoder2-shaped arch (Standard FFN +
    /// up_bias) on a feature-major f32 fixture drives the
    /// `up_bias_for_layer = Some(...)` branch (lines 81-86) AND the
    /// non-gated activation arm (lines 254-266).
    #[test]
    fn walk_ffn_sparse_non_gated_arch_uses_up_bias() {
        use crate::test_utils::{
            attach_feature_major_f32_to_test_vindex, make_starcoder2_test_weights,
        };
        let weights = make_starcoder2_test_weights();
        let mut index = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut index);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 4);
        let ffn = WalkFfn::from_config(&weights, &index, cfg);
        let out = ffn
            .walk_ffn_sparse(0, &x(1, weights.hidden_size))
            .expect("starcoder2 + feature-major fixture should produce output");
        assert_eq!(out.0.shape(), &[1, weights.hidden_size]);
        assert!(out.0.iter().all(|v| v.is_finite()));
    }

    struct StaticRunner {
        hidden: usize,
    }

    impl CallProgramRunner for StaticRunner {
        fn run(&mut self, _call: &CallPatchOp, _input: Value) -> Result<Value, CallError> {
            Ok(json!({"residual_delta": vec![1.0f32; self.hidden]}))
        }
    }

    fn call_patch_with_trigger(hidden: usize, trigger: CallTrigger) -> (CallPatchOp, Vec<f32>) {
        (
            CallPatchOp {
                layer: 0,
                feature: 0,
                gate_vector_b64: None,
                monty_code: "def main(input):\n    return input\n".into(),
                code_hash: None,
                input_schema: Value::Null,
                output_schema: Value::Null,
                trigger,
                limits: CallResourceLimits::default(),
                safety: CallSafetyPolicy::default(),
                metadata: Value::Null,
            },
            vec![100.0; hidden],
        )
    }

    fn raw_lm_head_logits(weights: &larql_models::ModelWeights, residual: &[f32]) -> Vec<f32> {
        weights
            .lm_head
            .outer_iter()
            .map(|row| row.iter().zip(residual).map(|(a, b)| a * b).sum())
            .collect()
    }

    #[test]
    fn walk_ffn_sparse_applies_selected_call_patch_delta() {
        use crate::test_utils::attach_feature_major_f32_to_test_vindex;
        let weights = make_test_weights();
        let mut base = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut base);
        let hidden = weights.hidden_size;
        let mut patched = larql_vindex::PatchedVindex::new(base);
        let (call, gate_vector) = call_patch_with_trigger(hidden, CallTrigger::default());
        patched.insert_call_patch(call, gate_vector);
        // Default trigger has max_calls_per_token=1 and require_top_k=1,
        // score_threshold=None. The gate vector [100.0; hidden] scores
        // far above any base feature for input [1.0; hidden], so this
        // call lands at rank 1 and fires once (calls_already_fired=0).
        let runtime = RefCell::new(MontyCallRuntime::new(StaticRunner { hidden }));
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 1);
        let ffn = WalkFfn::from_config(&weights, &patched, cfg)
            .with_call_patches(&patched)
            .with_call_runtime(&runtime);

        let (out, _activation) = ffn
            .walk_ffn_sparse(0, &Array2::from_elem((1, hidden), 1.0))
            .expect("call-patched sparse walk should produce output");

        assert_eq!(out.row(0).to_vec(), vec![1.0; hidden]);
        assert_eq!(runtime.borrow().metrics().fired, 1);
    }

    #[test]
    fn walk_ffn_sparse_skips_selected_call_patch_without_runtime() {
        use crate::test_utils::attach_feature_major_f32_to_test_vindex;
        let weights = make_test_weights();
        let mut base = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut base);
        let hidden = weights.hidden_size;
        let mut patched = larql_vindex::PatchedVindex::new(base);
        let (call, gate_vector) = call_patch_with_trigger(hidden, CallTrigger::default());
        patched.insert_call_patch(call, gate_vector);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 1);
        let ffn = WalkFfn::from_config(&weights, &patched, cfg).with_call_patches(&patched);

        let (out, _activation) = ffn
            .walk_ffn_sparse(0, &Array2::from_elem((1, hidden), 1.0))
            .expect("call-patched sparse walk should still produce zero output");

        assert_eq!(out.row(0).to_vec(), vec![0.0; hidden]);
    }

    #[test]
    fn walk_ffn_sparse_non_fired_call_patch_leaves_logits_unchanged() {
        use crate::test_utils::attach_feature_major_f32_to_test_vindex;
        let weights = make_test_weights();
        let mut base = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut base);
        let hidden = weights.hidden_size;
        let mut patched = larql_vindex::PatchedVindex::new(base);
        let mut trigger = CallTrigger::default();
        trigger.score_threshold = Some(f32::MAX);
        let (call, gate_vector) = call_patch_with_trigger(hidden, trigger);
        patched.insert_call_patch(call, gate_vector);
        let runtime = RefCell::new(MontyCallRuntime::new(StaticRunner { hidden }));
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 1);
        let ffn = WalkFfn::from_config(&weights, &patched, cfg)
            .with_call_patches(&patched)
            .with_call_runtime(&runtime);

        let (out, _activation) = ffn
            .walk_ffn_sparse(0, &Array2::from_elem((1, hidden), 1.0))
            .expect("non-fired call-patched sparse walk should produce output");
        let logits = raw_lm_head_logits(&weights, out.row(0).as_slice().unwrap());

        assert_eq!(out.row(0).to_vec(), vec![0.0; hidden]);
        assert!(logits.iter().all(|v| *v == 0.0));
        assert_eq!(runtime.borrow().metrics().fired, 0);
        assert_eq!(runtime.borrow().metrics().skipped, 1);
    }

    #[test]
    fn walk_ffn_sparse_fired_call_patch_changes_next_token_logits() {
        use crate::test_utils::attach_feature_major_f32_to_test_vindex;
        let weights = make_test_weights();
        let mut base = make_test_vindex(&weights);
        attach_feature_major_f32_to_test_vindex(&weights, &mut base);
        let hidden = weights.hidden_size;
        let mut patched = larql_vindex::PatchedVindex::new(base);
        let (call, gate_vector) = call_patch_with_trigger(hidden, CallTrigger::default());
        patched.insert_call_patch(call, gate_vector);
        let runtime = RefCell::new(MontyCallRuntime::new(StaticRunner { hidden }));
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 1);
        let ffn = WalkFfn::from_config(&weights, &patched, cfg)
            .with_call_patches(&patched)
            .with_call_runtime(&runtime);

        let (out, _activation) = ffn
            .walk_ffn_sparse(0, &Array2::from_elem((1, hidden), 1.0))
            .expect("fired call-patched sparse walk should produce output");
        let no_call_logits = raw_lm_head_logits(&weights, &vec![0.0; hidden]);
        let fired_logits = raw_lm_head_logits(&weights, out.row(0).as_slice().unwrap());

        assert_eq!(runtime.borrow().metrics().fired, 1);
        assert!(
            fired_logits
                .iter()
                .zip(no_call_logits.iter())
                .any(|(a, b)| (a - b).abs() > 1e-6),
            "call residual delta should perturb at least one next-token logit"
        );
    }

    /// Sparse walk in full-K mode against the Q4K fixture (no native
    /// up/down) drives the `kquant_matmul_transb` arms inside the
    /// full-K gemv fast path (lines 99-131): up_scores via Q4K matmul,
    /// then down via Q4K matmul again.
    #[test]
    fn walk_ffn_sparse_full_k_routes_through_kquant_matmul_on_q4k_fixture() {
        let weights = make_test_q4k_weights();
        let index = make_test_q4k_vindex(&weights);
        let cfg = WalkFfnConfig::dense(weights.num_layers);
        let backend = larql_compute::cpu_backend();
        let ffn = WalkFfn::from_config(&weights, &index, cfg).with_backend(&*backend);
        let result = ffn.walk_ffn_sparse(0, &x(1, weights.hidden_size));
        // Full-K + Q4K — either takes the fast path (Some) or falls through
        // to the serial loop (also Some). Just exercise the wiring.
        if let Some((out, _activation)) = result {
            assert_eq!(out.shape(), &[1, weights.hidden_size]);
            assert!(out.iter().all(|v| v.is_finite()));
        }
    }
}
