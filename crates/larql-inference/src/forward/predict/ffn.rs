//! FFN-backend forward passes (custom backend, router, strategy).

use super::super::embed::embed_tokens;
use super::super::layer::{run_attention, run_layer_with_capture, run_layer_with_ffn};
use super::super::ple::precompute_per_layer_inputs;
use super::dense::logits_to_predictions;
use super::types::{LayerAttentionCapture, LayerMode, PredictResult, PredictResultWithAttention};
use crate::attention::SharedKV;
use crate::ffn::{FfnBackend, LayerFfnRouter};
use crate::model::ModelWeights;
use crate::monty_call::{
    CallProgramRunner, CallTraceEvent, MontyCallMetrics, MontyCallRuntime, MontyVmRunner,
};
use crate::vindex::{WalkFfn, WalkFfnConfig};
use std::cell::RefCell;

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
    )
}

/// Run a vindex-backed forward pass with runtime call patches enabled and a
/// caller-supplied runner. This is primarily useful for embedding tests,
/// deterministic fakes, or hosted Monty runners.
pub fn predict_with_call_patches_runner<R: CallProgramRunner>(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    patched: &larql_vindex::PatchedVindex,
    config: WalkFfnConfig,
    runner: R,
    opts: PredictCallPatchesOptions,
) -> PredictResultWithCallMetrics {
    let base_runtime = MontyCallRuntime::new(runner);
    let base_runtime = if opts.trace {
        base_runtime.with_trace_events()
    } else {
        base_runtime
    };
    let runtime = RefCell::new(base_runtime);
    let ffn = WalkFfn::from_config(weights, patched, config)
        .with_call_patches(patched)
        .with_call_runtime(&runtime);
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
