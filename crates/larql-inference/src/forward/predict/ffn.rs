//! FFN-backend forward passes (custom backend, router, strategy).

use super::super::embed::{embed_tokens, embed_tokens_pub};
use super::super::layer::{
    apply_layer_scalar, run_attention, run_ffn, run_layer_with_capture, run_layer_with_ffn,
};
use super::super::ple::{apply_per_layer_embedding, precompute_per_layer_inputs};
use super::dense::logits_to_predictions;
use super::types::{LayerAttentionCapture, LayerMode, PredictResult, PredictResultWithAttention};
use crate::attention::{run_attention_block_decode_step_backend, SharedKV};
use crate::ffn::{FfnBackend, LayerFfnRouter};
use crate::layer_graph::generate::{EosConfig, GenerateError};
use crate::model::ModelWeights;
use crate::monty_call::{
    CallProgramRunner, CallTraceEvent, MontyCallMetrics, MontyCallRuntime, MontyVmRunner,
};
use crate::vindex::{WalkFfn, WalkFfnConfig};
use ndarray::Array2;
use std::cell::RefCell;
use std::collections::HashMap;

/// Prediction result plus runtime call-patch counters and per-event trace.
#[derive(Debug, Clone, PartialEq)]
pub struct PredictResultWithCallMetrics {
    pub predictions: Vec<(String, f64)>,
    pub call_metrics: MontyCallMetrics,
    /// Per-event trace. Empty unless the call was made via
    /// [`predict_with_call_patches_runner`] with trace collection enabled via
    /// [`PredictCallPatchesOptions::with_trace`].
    pub trace_events: Vec<CallTraceEvent>,
}

/// Run a full forward pass with a custom FFN backend for all layers.
pub fn predict_with_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    let mut kv_cache: std::collections::HashMap<usize, SharedKV> = std::collections::HashMap::new();

    for layer in 0..num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));

        match run_layer_with_ffn(
            weights,
            &h,
            layer,
            ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            Some((h_new, _, kv_out)) => {
                h = h_new;
                if let Some(kv) = kv_out {
                    kv_cache.insert(layer, kv);
                }
            }
            None => continue,
        }
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

/// Options for [`predict_with_call_patches`] / [`predict_with_call_patches_runner`].
#[derive(Debug, Clone, Default)]
pub struct PredictCallPatchesOptions {
    /// Collect per-event trace. Populated in [`PredictResultWithCallMetrics::trace_events`].
    pub trace: bool,
}

impl PredictCallPatchesOptions {
    pub fn with_trace(mut self) -> Self {
        self.trace = true;
        self
    }
}

/// Run a vindex-backed forward pass with runtime call patches enabled.
///
/// This is the public inference entry point for callers that have a
/// `PatchedVindex` and want both next-token predictions and call-patch
/// observability without manually constructing `WalkFfn`.
pub fn predict_with_call_patches(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    patched: &larql_vindex::PatchedVindex,
    config: WalkFfnConfig,
) -> PredictResultWithCallMetrics {
    predict_with_call_patches_runner(
        weights,
        tokenizer,
        token_ids,
        top_k,
        patched,
        config,
        MontyVmRunner::new(),
        PredictCallPatchesOptions::default(),
        None,
    )
}

/// Run a vindex-backed forward pass with runtime call patches enabled and a
/// caller-supplied runner. This is primarily useful for embedding tests,
/// deterministic fakes, or hosted Monty runners.
///
/// Pass `backend` (e.g. `MetalBackend`) to route dense/Q4/Q4K walk paths through
/// GPU kernels; call patches still execute on CPU after the matmul completes.
pub fn predict_with_call_patches_runner<R: CallProgramRunner>(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    patched: &larql_vindex::PatchedVindex,
    config: WalkFfnConfig,
    runner: R,
    opts: PredictCallPatchesOptions,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> PredictResultWithCallMetrics {
    let base_runtime = MontyCallRuntime::new(runner);
    let base_runtime = if opts.trace {
        base_runtime.with_trace_events()
    } else {
        base_runtime
    };
    let runtime = RefCell::new(base_runtime);
    let mut ffn = WalkFfn::from_config(weights, patched, config)
        .with_call_patches(patched)
        .with_call_runtime(&runtime);
    if let Some(be) = backend {
        ffn = ffn.with_backend(be);
    }
    let result = predict_with_ffn(weights, tokenizer, token_ids, top_k, &ffn);
    drop(ffn);
    let call_metrics = runtime.borrow().metrics();
    let trace_events = runtime.borrow_mut().take_trace_events();
    PredictResultWithCallMetrics {
        predictions: result.predictions,
        call_metrics,
        trace_events,
    }
}

/// Result of multi-token generation with call-patch observability.
#[derive(Debug)]
pub struct GenerateResultWithCallMetrics {
    pub tokens: Vec<(String, f64)>,
    pub prefill_ms: f64,
    pub decode_ms: Vec<f64>,
    pub call_metrics: MontyCallMetrics,
    /// Per-event trace. Empty unless trace collection was enabled via
    /// [`PredictCallPatchesOptions::with_trace`].
    pub trace_events: Vec<CallTraceEvent>,
    pub error: Option<GenerateError>,
}

impl GenerateResultWithCallMetrics {
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    pub fn text(&self) -> String {
        self.tokens
            .iter()
            .map(|(t, _)| t.as_str())
            .collect::<Vec<_>>()
            .join("")
    }
}

/// Run multi-token generation with runtime call patches enabled.
///
/// Convenience wrapper over [`generate_with_call_patches_runner`] using the
/// default [`MontyVmRunner`] and no trace collection.
pub fn generate_with_call_patches(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    patched: &larql_vindex::PatchedVindex,
    config: WalkFfnConfig,
    eos: &EosConfig,
) -> GenerateResultWithCallMetrics {
    generate_with_call_patches_runner(
        weights,
        tokenizer,
        token_ids,
        max_tokens,
        patched,
        config,
        MontyVmRunner::new(),
        PredictCallPatchesOptions::default(),
        None,
        eos,
    )
}

/// Run multi-token generation with runtime call patches enabled and a
/// caller-supplied runner.
///
/// Threads a `PatchedVindex` and `MontyCallRuntime` through `WalkFfn` at every
/// prefill and decode step, resets per-sequence state once at the start, then
/// accumulates call-patch metrics and optional trace events across the full
/// sequence.
///
/// Uses a production KV-cached loop (prefill once, then single-token decode
/// steps). When the model and vindex support Q4K cached decode
/// ([`crate::vindex::supports_kquant_cached_custom_ffn`]), prefill and decode
/// run through that driver with [`WalkFfn`] so batched prompt positions and
/// decode steps both fire call patches. Otherwise falls back to the generic
/// layer loop in [`kv_prefill_with_call_ffn`].
///
/// Pass `backend` to enable Metal/GPU matmul paths; call patches execute on
/// CPU after the accelerated FFN completes.
pub fn generate_with_call_patches_runner<R: CallProgramRunner>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    max_tokens: usize,
    patched: &larql_vindex::PatchedVindex,
    config: WalkFfnConfig,
    runner: R,
    opts: PredictCallPatchesOptions,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    eos: &EosConfig,
) -> GenerateResultWithCallMetrics {
    if max_tokens == 0 {
        return GenerateResultWithCallMetrics {
            tokens: Vec::new(),
            prefill_ms: 0.0,
            decode_ms: Vec::new(),
            call_metrics: MontyCallMetrics::default(),
            trace_events: Vec::new(),
            error: None,
        };
    }
    if token_ids.is_empty() {
        return GenerateResultWithCallMetrics {
            tokens: Vec::new(),
            prefill_ms: 0.0,
            decode_ms: Vec::new(),
            call_metrics: MontyCallMetrics::default(),
            trace_events: Vec::new(),
            error: None,
        };
    }

    let base_runtime = MontyCallRuntime::new(runner);
    let base_runtime = if opts.trace {
        base_runtime.with_trace_events()
    } else {
        base_runtime
    };
    let runtime = RefCell::new(base_runtime);
    runtime.borrow_mut().reset_sequence_state();

    let use_kquant_cached =
        crate::vindex::supports_kquant_cached_custom_ffn(weights, patched.base());

    let mut tokens: Vec<(String, f64)> = Vec::with_capacity(max_tokens);
    let mut decode_ms: Vec<f64> = Vec::with_capacity(max_tokens);

    if use_kquant_cached {
        let tensor_index = patched.base();
        let prefill_start = std::time::Instant::now();
        let kquant_ctx = crate::vindex::KquantCallPatchCtx {
            tensor_index,
            patched,
            config: config.clone(),
            runtime: &runtime,
            call_position_base: 0,
            matmul_backend: backend,
        };
        let (h_prompt, q4_cache) = {
            let (h_prompt, q4_cache, _) = crate::vindex::predict_kquant_prefill_with_call_patches(
                weights,
                token_ids,
                &kquant_ctx,
            );
            (h_prompt, q4_cache)
        };
        let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
        let decode_state = KquantCallDecodeState { cache: q4_cache };
        match sample_first_and_decode_loop_kquant(
            weights,
            tokenizer,
            patched,
            config,
            &runtime,
            backend,
            eos,
            max_tokens,
            &last_row_as_2d(&h_prompt),
            token_ids.len(),
            decode_state,
            &mut tokens,
            &mut decode_ms,
            &kquant_ctx,
        ) {
            Ok(()) => {
                let call_metrics = runtime.borrow().metrics();
                let trace_events = runtime.borrow_mut().take_trace_events();
                return GenerateResultWithCallMetrics {
                    tokens,
                    prefill_ms,
                    decode_ms,
                    call_metrics,
                    trace_events,
                    error: None,
                };
            }
            Err(err) => {
                let call_metrics = runtime.borrow().metrics();
                let trace_events = runtime.borrow_mut().take_trace_events();
                return GenerateResultWithCallMetrics {
                    tokens,
                    prefill_ms,
                    decode_ms,
                    call_metrics,
                    trace_events,
                    error: Some(err),
                };
            }
        }
    }

    let mut ffn = WalkFfn::from_config(weights, patched, config)
        .with_call_patches(patched)
        .with_call_runtime(&runtime);
    if let Some(be) = backend {
        ffn = ffn.with_backend(be);
    }

    let prefill_start = std::time::Instant::now();
    match kv_prefill_with_call_ffn(weights, token_ids, &ffn) {
        Some((last_hidden, mut kv_cache, next_position)) => {
            let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
            finish_generate_with_call_legacy(
                weights,
                tokenizer,
                &ffn,
                eos,
                max_tokens,
                prefill_ms,
                last_hidden,
                &mut kv_cache,
                next_position,
                &runtime,
                &mut tokens,
                &mut decode_ms,
            )
        }
        None => {
            let call_metrics = runtime.borrow().metrics();
            let trace_events = runtime.borrow_mut().take_trace_events();
            GenerateResultWithCallMetrics {
                tokens,
                prefill_ms: 0.0,
                decode_ms,
                call_metrics,
                trace_events,
                error: Some(GenerateError::empty_output(
                    "generate_with_call_patches: prefill failed",
                )),
            }
        }
    }
}

struct KquantCallDecodeState {
    cache: crate::vindex::CpuKvCache,
}

/// After Q4K cached prefill: sample first token and run decode loop.
fn sample_first_and_decode_loop_kquant<R: CallProgramRunner>(
    weights: &mut ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    _patched: &larql_vindex::PatchedVindex,
    config: WalkFfnConfig,
    _runtime: &RefCell<MontyCallRuntime<R>>,
    _backend: Option<&dyn larql_compute::ComputeBackend>,
    eos: &EosConfig,
    max_tokens: usize,
    first_hidden: &Array2<f32>,
    mut next_position: usize,
    mut decode_state: KquantCallDecodeState,
    tokens: &mut Vec<(String, f64)>,
    decode_ms: &mut Vec<f64>,
    kquant_ctx_template: &crate::vindex::KquantCallPatchCtx<'_, R>,
) -> Result<(), GenerateError> {
    let first = logits_to_predictions(weights, first_hidden, tokenizer, 1, 1.0);
    let first_stop = match (first.token_ids.first(), first.predictions.first()) {
        (Some(&id), Some(pred)) => {
            let stop = eos.is_eos_with_tokenizer(id, &pred.0, tokenizer);
            tokens.push((pred.0.clone(), 1.0));
            stop
        }
        _ => {
            return Err(GenerateError::empty_output(
                "generate_with_call_patches: no first token",
            ));
        }
    };
    if first_stop || max_tokens == 1 {
        return Ok(());
    }

    let mut current_id = first.token_ids[0];
    for _step in 1..max_tokens {
        let step_start = std::time::Instant::now();
        let step_ctx = crate::vindex::KquantCallPatchCtx {
            tensor_index: kquant_ctx_template.tensor_index,
            patched: kquant_ctx_template.patched,
            config: config.clone(),
            runtime: kquant_ctx_template.runtime,
            call_position_base: next_position,
            matmul_backend: kquant_ctx_template.matmul_backend,
        };
        let h_step = match crate::vindex::predict_kquant_decode_step_with_call_patches(
            weights,
            current_id,
            &mut decode_state.cache,
            next_position,
            &step_ctx,
        ) {
            Some((h, _timings)) => h,
            None => break,
        };
        next_position += 1;
        decode_ms.push(step_start.elapsed().as_secs_f64() * 1000.0);

        let result = logits_to_predictions(weights, &h_step, tokenizer, 1, 1.0);
        match (result.token_ids.first(), result.predictions.first()) {
            (Some(&id), Some(pred)) => {
                let stop = eos.is_eos_with_tokenizer(id, &pred.0, tokenizer);
                tokens.push((pred.0.clone(), 1.0));
                current_id = id;
                if stop {
                    break;
                }
            }
            _ => break,
        }
    }
    Ok(())
}

fn finish_generate_with_call_legacy<R: CallProgramRunner>(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    ffn: &WalkFfn<'_>,
    eos: &EosConfig,
    max_tokens: usize,
    prefill_ms: f64,
    last_hidden: Array2<f32>,
    kv_cache: &mut HashMap<usize, SharedKV>,
    mut next_position: usize,
    runtime: &RefCell<MontyCallRuntime<R>>,
    tokens: &mut Vec<(String, f64)>,
    decode_ms: &mut Vec<f64>,
) -> GenerateResultWithCallMetrics {
    let first = logits_to_predictions(weights, &last_hidden, tokenizer, 1, 1.0);
    let first_stop = match (first.token_ids.first(), first.predictions.first()) {
        (Some(&id), Some(pred)) => {
            let stop = eos.is_eos_with_tokenizer(id, &pred.0, tokenizer);
            tokens.push((pred.0.clone(), 1.0));
            stop
        }
        _ => {
            let call_metrics = runtime.borrow().metrics();
            let trace_events = runtime.borrow_mut().take_trace_events();
            return GenerateResultWithCallMetrics {
                tokens: tokens.clone(),
                prefill_ms,
                decode_ms: decode_ms.clone(),
                call_metrics,
                trace_events,
                error: Some(GenerateError::empty_output(
                    "generate_with_call_patches: no first token",
                )),
            };
        }
    };
    if first_stop || max_tokens == 1 {
        let call_metrics = runtime.borrow().metrics();
        let trace_events = runtime.borrow_mut().take_trace_events();
        return GenerateResultWithCallMetrics {
            tokens: tokens.clone(),
            prefill_ms,
            decode_ms: decode_ms.clone(),
            call_metrics,
            trace_events,
            error: None,
        };
    }

    let mut current_id = first.token_ids[0];
    for _step in 1..max_tokens {
        let step_start = std::time::Instant::now();
        ffn.set_call_position_base(next_position);
        let h_step = match kv_decode_step_with_call_ffn(
            weights,
            ffn,
            kv_cache,
            current_id,
            next_position,
        ) {
            Some(h) => h,
            None => break,
        };
        next_position += 1;
        decode_ms.push(step_start.elapsed().as_secs_f64() * 1000.0);

        let result = logits_to_predictions(weights, &h_step, tokenizer, 1, 1.0);
        match (result.token_ids.first(), result.predictions.first()) {
            (Some(&id), Some(pred)) => {
                let stop = eos.is_eos_with_tokenizer(id, &pred.0, tokenizer);
                tokens.push((pred.0.clone(), 1.0));
                current_id = id;
                if stop {
                    break;
                }
            }
            _ => break,
        }
    }

    let call_metrics = runtime.borrow().metrics();
    let trace_events = runtime.borrow_mut().take_trace_events();
    GenerateResultWithCallMetrics {
        tokens: tokens.clone(),
        prefill_ms,
        decode_ms: decode_ms.clone(),
        call_metrics,
        trace_events,
        error: None,
    }
}

/// KV-cache prefill for call-patch generation. Returns the last prompt hidden
/// state, per-layer K/V, and the next absolute token position.
fn kv_prefill_with_call_ffn(
    weights: &ModelWeights,
    prompt_ids: &[u32],
    ffn: &WalkFfn<'_>,
) -> Option<(Array2<f32>, HashMap<usize, SharedKV>, usize)> {
    ffn.set_call_position_base(0);
    let num_layers = weights.num_layers;
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::new();
    let mut h = embed_tokens_pub(weights, prompt_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, prompt_ids);
    for layer in 0..num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        let (h_new, _, kv_out) = run_layer_with_ffn(
            weights,
            &h,
            layer,
            ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        )?;
        h = h_new;
        if let Some(kv) = kv_out {
            kv_cache.insert(layer, kv);
        }
    }
    let next_position = prompt_ids.len();
    Some((last_row_as_2d(&h), kv_cache, next_position))
}

/// Single-token KV decode step for call-patch generation.
///
/// Mirrors `larql_kv::generation::kv_decode_step_run` but lives in inference
/// to avoid a circular dependency. Architectures with cross-layer KV sharing
/// are not supported on this path (same constraint as `supports_cached_decode`).
fn kv_decode_step_with_call_ffn(
    weights: &ModelWeights,
    ffn: &WalkFfn<'_>,
    kv_cache: &mut HashMap<usize, SharedKV>,
    token_id: u32,
    abs_position: usize,
) -> Option<Array2<f32>> {
    let num_layers = weights.num_layers;
    let h_new = embed_tokens_pub(weights, &[token_id]);
    let ple_inputs = precompute_per_layer_inputs(weights, &h_new, &[token_id]);
    let mut h_step = h_new;
    for layer in 0..num_layers {
        let h_post_attn = if let Some(src) = weights.arch.kv_shared_source_layer(layer) {
            let shared = kv_cache.get(&src)?;
            larql_compute::attention::run_attention_block_decode_step_shared(
                weights,
                &h_step,
                layer,
                shared,
                abs_position,
                ffn.backend,
            )?
        } else {
            let prior_kv = kv_cache.get(&layer);
            let (h_post_attn, new_kv) = run_attention_block_decode_step_backend(
                weights,
                &h_step,
                layer,
                prior_kv,
                abs_position,
                ffn.backend,
            )?;
            kv_cache.insert(layer, new_kv);
            h_post_attn
        };
        let (h_post_ffn, _) = run_ffn(weights, &h_post_attn, layer, ffn, false);
        let mut h_out =
            apply_per_layer_embedding(weights, &h_post_ffn, layer, ple_inputs.get(layer));
        apply_layer_scalar(weights, &mut h_out, layer);
        h_step = h_out;
    }
    Some(h_step)
}

fn last_row_as_2d(h: &Array2<f32>) -> Array2<f32> {
    let seq_len = h.shape()[0];
    let hidden = h.shape()[1];
    let mut out = Array2::<f32>::zeros((1, hidden));
    out.row_mut(0).assign(&h.row(seq_len - 1));
    out
}

/// Run a full forward pass with a custom FFN backend, capturing attention weights
/// and per-layer residuals for logit lens.
pub fn predict_with_ffn_attention(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    ffn: &dyn FfnBackend,
) -> PredictResultWithAttention {
    let num_layers = weights.num_layers;
    let seq_len = token_ids.len();
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);
    let mut attention = Vec::with_capacity(num_layers);
    let mut residuals = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        match run_layer_with_capture(
            weights,
            &h,
            layer,
            ffn,
            false,
            true,
            ple_inputs.get(layer),
            None,
        ) {
            Some((h_new, _, attn_weights, _)) => {
                h = h_new;
                residuals.push((layer, h.row(seq_len - 1).to_vec()));
                if let Some(w) = attn_weights {
                    attention.push(LayerAttentionCapture { layer, weights: w });
                }
            }
            None => continue,
        }
    }

    let result = logits_to_predictions(weights, &h, tokenizer, top_k, 1.0);
    PredictResultWithAttention {
        predictions: result.predictions,
        attention,
        residuals,
    }
}

/// Run a full forward pass with per-layer FFN backend selection.
pub fn predict_with_router(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    router: &LayerFfnRouter,
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for layer in 0..num_layers {
        let ffn = router.get(layer);
        h = match run_layer_with_ffn(weights, &h, layer, ffn, false, ple_inputs.get(layer), None) {
            Some((h_new, _, _)) => h_new,
            None => continue,
        };
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

/// Run a forward pass with per-layer strategy: full compute or scalar gain bypass.
pub fn predict_with_strategy(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    strategy: &[LayerMode],
) -> PredictResult {
    let num_layers = weights.num_layers;
    let mut h = embed_tokens(weights, token_ids);
    let ple_inputs = precompute_per_layer_inputs(weights, &h, token_ids);

    for (layer, mode) in strategy.iter().enumerate().take(num_layers) {
        match mode {
            LayerMode::Compute(ffn) => {
                h = match run_layer_with_ffn(
                    weights,
                    &h,
                    layer,
                    *ffn,
                    false,
                    ple_inputs.get(layer),
                    None,
                ) {
                    Some((h_new, _, _)) => h_new,
                    None => continue,
                };
            }
            LayerMode::ScalarGain(gain) => {
                h *= *gain;
            }
            LayerMode::AttentionOnly => {
                if let Some(h_post_attn) = run_attention(weights, &h, layer) {
                    h = h_post_attn;
                }
            }
        }
    }

    logits_to_predictions(weights, &h, tokenizer, top_k, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffn::{LayerFfnRouter, WeightFfn};
    use crate::monty_call::{CallError, CallProgramRunner};
    use crate::test_utils::{attach_feature_major_f32_to_test_vindex, TestFixtures};
    use larql_vindex::{CallPatchOp, CallResourceLimits, CallSafetyPolicy, CallTrigger};
    use serde_json::{json, Value};

    struct StaticRunner {
        hidden: usize,
    }

    impl CallProgramRunner for StaticRunner {
        fn run(&mut self, _call: &CallPatchOp, _input: Value) -> Result<Value, CallError> {
            Ok(json!({"residual_delta": vec![1.0f32; self.hidden]}))
        }
    }

    #[test]
    fn predict_with_ffn_attention_returns_attention_and_residuals() {
        let fx = TestFixtures::build();
        let ffn = WeightFfn {
            weights: &fx.weights,
        };
        let result = predict_with_ffn_attention(&fx.weights, &fx.tokenizer, &[0u32, 1], 3, &ffn);
        assert!(result.predictions.len() <= 3);
        assert_eq!(result.residuals.len(), fx.weights.num_layers);
        // Attention captured at every layer.
        assert_eq!(result.attention.len(), fx.weights.num_layers);
        for cap in &result.attention {
            assert!(cap.layer < fx.weights.num_layers);
        }
    }

    #[test]
    fn predict_with_router_routes_per_layer() {
        let fx = TestFixtures::build();
        let ffn = WeightFfn {
            weights: &fx.weights,
        };
        let router = LayerFfnRouter::uniform(&ffn, fx.weights.num_layers);
        let result = predict_with_router(&fx.weights, &fx.tokenizer, &[0u32, 1], 3, &router);
        assert!(result.predictions.len() <= 3);
    }

    #[test]
    fn predict_with_call_patches_runner_exposes_metrics() {
        let mut fx = TestFixtures::build();
        attach_feature_major_f32_to_test_vindex(&fx.weights, &mut fx.index);
        let hidden = fx.weights.hidden_size;
        let first_hidden = embed_tokens(&fx.weights, &[0u32]).row(0).to_vec();
        let mut patched = larql_vindex::PatchedVindex::new(fx.index);
        patched.insert_call_patch(
            CallPatchOp {
                layer: 0,
                feature: 0,
                gate_vector_b64: None,
                monty_code: "def main(input):\n    return input\n".into(),
                code_hash: None,
                input_schema: Value::Null,
                output_schema: Value::Null,
                trigger: CallTrigger::default(),
                limits: CallResourceLimits::default(),
                safety: CallSafetyPolicy::default(),
                metadata: Value::Null,
            },
            first_hidden.iter().map(|v| v * 100.0).collect(),
        );
        let result = predict_with_call_patches_runner(
            &fx.weights,
            &fx.tokenizer,
            &[0u32],
            3,
            &patched,
            WalkFfnConfig::sparse(fx.weights.num_layers, 1),
            StaticRunner { hidden },
            PredictCallPatchesOptions::default(),
            None,
        );

        assert!(result.predictions.len() <= 3);
        assert_eq!(result.call_metrics.attempted, 1);
        assert_eq!(result.call_metrics.fired, 1);
        assert!(result.trace_events.is_empty(), "trace disabled by default");
    }

    #[test]
    fn predict_with_call_patches_runner_collects_trace_when_enabled() {
        use crate::monty_call::CallOutcome;
        let mut fx = TestFixtures::build();
        attach_feature_major_f32_to_test_vindex(&fx.weights, &mut fx.index);
        let hidden = fx.weights.hidden_size;
        let first_hidden = embed_tokens(&fx.weights, &[0u32]).row(0).to_vec();
        let mut patched = larql_vindex::PatchedVindex::new(fx.index);
        patched.insert_call_patch(
            CallPatchOp {
                layer: 0,
                feature: 0,
                gate_vector_b64: None,
                monty_code: "def main(input):\n    return input\n".into(),
                code_hash: None,
                input_schema: Value::Null,
                output_schema: Value::Null,
                trigger: CallTrigger::default(),
                limits: larql_vindex::CallResourceLimits::default(),
                safety: larql_vindex::CallSafetyPolicy::default(),
                metadata: Value::Null,
            },
            first_hidden.iter().map(|v| v * 100.0).collect(),
        );
        let result = predict_with_call_patches_runner(
            &fx.weights,
            &fx.tokenizer,
            &[0u32],
            3,
            &patched,
            WalkFfnConfig::sparse(fx.weights.num_layers, 1),
            StaticRunner { hidden },
            PredictCallPatchesOptions::default().with_trace(),
            None,
        );

        assert_eq!(result.call_metrics.fired, 1);
        assert_eq!(result.trace_events.len(), 1, "one fired trace event expected");
        assert!(
            matches!(result.trace_events[0].outcome, CallOutcome::Fired),
            "event should be Fired"
        );
        assert_eq!(result.trace_events[0].layer, 0);
    }

    #[test]
    fn generate_with_call_patches_runner_accumulates_metrics_across_tokens() {
        use crate::layer_graph::generate::EosConfig;

        let mut fx = TestFixtures::build();
        attach_feature_major_f32_to_test_vindex(&fx.weights, &mut fx.index);
        let hidden = fx.weights.hidden_size;
        let first_hidden = embed_tokens(&fx.weights, &[0u32]).row(0).to_vec();
        let mut patched = larql_vindex::PatchedVindex::new(fx.index);
        patched.insert_call_patch(
            CallPatchOp {
                layer: 0,
                feature: 0,
                gate_vector_b64: None,
                monty_code: "def main(input):\n    return input\n".into(),
                code_hash: None,
                input_schema: Value::Null,
                output_schema: Value::Null,
                trigger: CallTrigger::default(),
                limits: CallResourceLimits::default(),
                safety: CallSafetyPolicy::default(),
                metadata: Value::Null,
            },
            first_hidden.iter().map(|v| v * 100.0).collect(),
        );
        let num_layers = fx.weights.num_layers;
        let result = generate_with_call_patches_runner(
            &mut fx.weights,
            &fx.tokenizer,
            &[0u32],
            3,
            &patched,
            WalkFfnConfig::sparse(num_layers, 1),
            StaticRunner { hidden },
            PredictCallPatchesOptions::default(),
            None,
            &EosConfig::empty(),
        );

        assert!(!result.is_error(), "generate should succeed; err: {:?}", result.error);
        assert!(!result.tokens.is_empty(), "should produce at least one token");
        // Call patch fires on layer 0 at least once across the sequence.
        assert!(
            result.call_metrics.attempted >= 1,
            "expected at least one call attempt"
        );
    }

    #[test]
    fn generate_with_call_patches_runner_zero_max_tokens_returns_empty() {
        use crate::layer_graph::generate::EosConfig;

        let mut fx = TestFixtures::build();
        attach_feature_major_f32_to_test_vindex(&fx.weights, &mut fx.index);
        let hidden = fx.weights.hidden_size;
        let patched = larql_vindex::PatchedVindex::new(fx.index);
        let num_layers = fx.weights.num_layers;
        let result = generate_with_call_patches_runner(
            &mut fx.weights,
            &fx.tokenizer,
            &[0u32],
            0,
            &patched,
            WalkFfnConfig::sparse(num_layers, 1),
            StaticRunner { hidden },
            PredictCallPatchesOptions::default(),
            None,
            &EosConfig::empty(),
        );

        assert!(result.tokens.is_empty());
        assert!(!result.is_error());
        assert_eq!(result.call_metrics.attempted, 0);
    }

    #[test]
    fn generate_with_call_patches_runner_collects_trace_across_steps() {
        use crate::layer_graph::generate::EosConfig;
        use crate::monty_call::CallOutcome;

        let mut fx = TestFixtures::build();
        attach_feature_major_f32_to_test_vindex(&fx.weights, &mut fx.index);
        let hidden = fx.weights.hidden_size;
        let first_hidden = embed_tokens(&fx.weights, &[0u32]).row(0).to_vec();
        let mut patched = larql_vindex::PatchedVindex::new(fx.index);
        patched.insert_call_patch(
            CallPatchOp {
                layer: 0,
                feature: 0,
                gate_vector_b64: None,
                monty_code: "def main(input):\n    return input\n".into(),
                code_hash: None,
                input_schema: Value::Null,
                output_schema: Value::Null,
                trigger: CallTrigger::default(),
                limits: CallResourceLimits::default(),
                safety: CallSafetyPolicy::default(),
                metadata: Value::Null,
            },
            first_hidden.iter().map(|v| v * 100.0).collect(),
        );
        let num_layers = fx.weights.num_layers;
        let result = generate_with_call_patches_runner(
            &mut fx.weights,
            &fx.tokenizer,
            &[0u32],
            2,
            &patched,
            WalkFfnConfig::sparse(num_layers, 1),
            StaticRunner { hidden },
            PredictCallPatchesOptions::default().with_trace(),
            None,
            &EosConfig::empty(),
        );

        assert!(!result.trace_events.is_empty(), "trace events should be collected");
        assert!(
            result.trace_events.iter().any(|e| matches!(e.outcome, CallOutcome::Fired)),
            "at least one Fired event expected"
        );
    }

    #[test]
    fn kv_decode_step_with_call_ffn_handles_kv_shared_layers() {
        use larql_models::test_fixtures::make_synthetic_e2b_like_weights;

        let weights = make_synthetic_e2b_like_weights();
        let hidden = weights.hidden_size;
        let index = larql_vindex::VectorIndex::new(
            vec![None; weights.num_layers],
            vec![None; weights.num_layers],
            weights.num_layers,
            hidden,
        );
        let patched = larql_vindex::PatchedVindex::new(index);
        let ffn = WalkFfn::from_config(&weights, &patched, WalkFfnConfig::sparse(4, 1));

        let (_, mut kv_cache, _) =
            kv_prefill_with_call_ffn(&weights, &[0u32, 1], &ffn).expect("prefill");
        let h_step = kv_decode_step_with_call_ffn(&weights, &ffn, &mut kv_cache, 2, 2)
            .expect("decode on kv-shared arch");
        assert_eq!(h_step.shape(), &[1, hidden]);
        assert!(h_step.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn predict_with_strategy_compute_mode_runs_layer_normally() {
        let fx = TestFixtures::build();
        let ffn = WeightFfn {
            weights: &fx.weights,
        };
        // Every layer = compute mode (same as predict_with_ffn).
        let strategy: Vec<LayerMode> = (0..fx.weights.num_layers)
            .map(|_| LayerMode::Compute(&ffn as &dyn FfnBackend))
            .collect();
        let result = predict_with_strategy(&fx.weights, &fx.tokenizer, &[0u32, 1], 3, &strategy);
        assert!(result.predictions.len() <= 3);
    }

    #[test]
    fn predict_with_strategy_scalar_gain_skips_compute() {
        let fx = TestFixtures::build();
        let ffn = WeightFfn {
            weights: &fx.weights,
        };
        // First layer = compute, rest = scalar gain (skip layers via *=).
        let mut strategy: Vec<LayerMode> = vec![LayerMode::Compute(&ffn as &dyn FfnBackend)];
        for _ in 1..fx.weights.num_layers {
            strategy.push(LayerMode::ScalarGain(1.0));
        }
        let result = predict_with_strategy(&fx.weights, &fx.tokenizer, &[0u32], 3, &strategy);
        assert!(result.predictions.len() <= 3);
    }

    #[test]
    fn predict_with_strategy_attention_only_skips_ffn() {
        let fx = TestFixtures::build();
        let ffn = WeightFfn {
            weights: &fx.weights,
        };
        // Mix of compute and attention-only layers.
        let mut strategy: Vec<LayerMode> = Vec::with_capacity(fx.weights.num_layers);
        for layer in 0..fx.weights.num_layers {
            if layer == 0 {
                strategy.push(LayerMode::Compute(&ffn as &dyn FfnBackend));
            } else {
                strategy.push(LayerMode::AttentionOnly);
            }
        }
        let result = predict_with_strategy(&fx.weights, &fx.tokenizer, &[0u32, 1], 3, &strategy);
        assert!(result.predictions.len() <= 3);
    }
}
