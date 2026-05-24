//! Runtime call-patch support.
//!
//! This module intentionally stops at the deterministic boundary around a
//! call: trigger checks, JSON-ish input encoding, output decoding, and safety
//! clamps. The actual Monty VM integration can sit behind this surface without
//! changing the vindex patch format or the inference hook contract.

use larql_vindex::{CallPatchOp, CallSafetyPolicy, CallTrigger};
use serde::Serialize;
use serde_json::{json, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum CallError {
    #[error("missing output key `{0}`")]
    MissingKey(String),
    #[error("expected `{0}` to be an array")]
    ExpectedArray(String),
    #[error("expected `{key}` to have length {expected}, got {actual}")]
    WrongLength {
        key: String,
        expected: usize,
        actual: usize,
    },
    #[error("expected numeric value in `{0}`")]
    ExpectedNumber(String),
    #[error("invalid sparse logit bias: {0}")]
    InvalidLogitBias(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CallCandidate {
    pub rank: usize,
    pub score: f32,
    pub margin: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct CallContext<'a> {
    pub layer: usize,
    pub position: usize,
    pub residual: &'a [f32],
    #[serde(default)]
    pub token_ids: &'a [u32],
    #[serde(default)]
    pub token_text: Option<&'a str>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SparseLogitBias {
    pub token_ids: Vec<u32>,
    pub biases: Vec<f32>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct CallOutput {
    pub residual_delta: Option<Vec<f32>>,
    pub logit_bias: Option<SparseLogitBias>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MontyCallMetrics {
    pub attempted: u64,
    pub fired: u64,
    pub skipped: u64,
    pub failed: u64,
}

pub trait CallProgramRunner {
    fn run(&mut self, call: &CallPatchOp, input: Value) -> Result<Value, CallError>;
}

#[derive(Debug, Default)]
pub struct MontyCallRuntime<R> {
    runner: R,
    metrics: MontyCallMetrics,
}

impl<R> MontyCallRuntime<R> {
    pub fn new(runner: R) -> Self {
        Self {
            runner,
            metrics: MontyCallMetrics::default(),
        }
    }

    pub fn metrics(&self) -> MontyCallMetrics {
        self.metrics
    }

    pub fn into_runner(self) -> R {
        self.runner
    }
}

impl<R: CallProgramRunner> MontyCallRuntime<R> {
    pub fn execute_call(
        &mut self,
        call: &CallPatchOp,
        candidate: CallCandidate,
        ctx: &CallContext<'_>,
        hidden_size: usize,
    ) -> Result<Option<CallOutput>, CallError> {
        self.metrics.attempted += 1;
        if !should_fire(&call.trigger, candidate) {
            self.metrics.skipped += 1;
            return Ok(None);
        }

        let input = encode_input(call, ctx);
        let raw = match self.runner.run(call, input) {
            Ok(raw) => raw,
            Err(err) => {
                self.metrics.failed += 1;
                return Err(err);
            }
        };

        let mut output = match decode_output(call, &raw, hidden_size) {
            Ok(output) => output,
            Err(err) => {
                self.metrics.failed += 1;
                return Err(err);
            }
        };
        apply_safety(&mut output, &call.safety);
        self.metrics.fired += 1;
        Ok(Some(output))
    }
}

pub fn should_fire(trigger: &CallTrigger, candidate: CallCandidate) -> bool {
    if candidate.rank == 0 || candidate.rank > trigger.require_top_k {
        return false;
    }
    if let Some(threshold) = trigger.score_threshold {
        if candidate.score.abs() < threshold {
            return false;
        }
    }
    if let Some(threshold) = trigger.margin_threshold {
        if candidate.margin.unwrap_or(f32::NEG_INFINITY) < threshold {
            return false;
        }
    }
    trigger.max_calls_per_token > 0
}

pub fn encode_input(call: &CallPatchOp, ctx: &CallContext<'_>) -> Value {
    let include_tokens = call
        .input_schema
        .get("include_token_ids")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    let include_text = call
        .input_schema
        .get("include_token_text")
        .and_then(Value::as_bool)
        .unwrap_or(ctx.token_text.is_some());

    let mut obj = serde_json::Map::new();
    obj.insert("layer".into(), json!(ctx.layer));
    obj.insert("position".into(), json!(ctx.position));
    obj.insert("residual".into(), json!(ctx.residual));
    if include_tokens {
        obj.insert("token_ids".into(), json!(ctx.token_ids));
    }
    if include_text {
        if let Some(text) = ctx.token_text {
            obj.insert("text".into(), json!(text));
        }
    }
    Value::Object(obj)
}

pub fn decode_output(
    call: &CallPatchOp,
    obj: &Value,
    hidden_size: usize,
) -> Result<CallOutput, CallError> {
    let residual_key = output_key(call, "residual_delta", "residual_delta");
    let logit_key = output_key(call, "logit_bias", "logit_bias");

    let residual_delta = match obj.get(&residual_key) {
        Some(value) => Some(decode_f32_array(&residual_key, value, Some(hidden_size))?),
        None => None,
    };

    let logit_bias = match obj.get(&logit_key) {
        Some(value) => Some(decode_sparse_logit_bias(value)?),
        None => None,
    };

    Ok(CallOutput {
        residual_delta,
        logit_bias,
    })
}

pub fn apply_safety(output: &mut CallOutput, safety: &CallSafetyPolicy) {
    let Some(limit) = safety.residual_clamp_norm else {
        return;
    };
    if limit <= 0.0 {
        output.residual_delta = None;
        return;
    }
    let Some(delta) = output.residual_delta.as_mut() else {
        return;
    };
    let norm = delta
        .iter()
        .map(|v| (*v as f64) * (*v as f64))
        .sum::<f64>()
        .sqrt() as f32;
    if norm > limit && norm > 0.0 {
        let scale = limit / norm;
        for v in delta {
            *v *= scale;
        }
    }
}

fn output_key(call: &CallPatchOp, sink: &str, default: &str) -> String {
    call.output_schema
        .get("keys")
        .and_then(|keys| keys.get(sink))
        .and_then(Value::as_str)
        .unwrap_or(default)
        .to_string()
}

fn decode_f32_array(
    key: &str,
    value: &Value,
    expected_len: Option<usize>,
) -> Result<Vec<f32>, CallError> {
    let arr = value
        .as_array()
        .ok_or_else(|| CallError::ExpectedArray(key.to_string()))?;
    if let Some(expected) = expected_len {
        if arr.len() != expected {
            return Err(CallError::WrongLength {
                key: key.to_string(),
                expected,
                actual: arr.len(),
            });
        }
    }
    arr.iter()
        .map(|v| {
            v.as_f64()
                .map(|n| n as f32)
                .ok_or_else(|| CallError::ExpectedNumber(key.to_string()))
        })
        .collect()
}

fn decode_sparse_logit_bias(value: &Value) -> Result<SparseLogitBias, CallError> {
    let token_ids = value
        .get("token_ids")
        .ok_or_else(|| CallError::InvalidLogitBias("missing token_ids".into()))?;
    let biases = value
        .get("biases")
        .ok_or_else(|| CallError::InvalidLogitBias("missing biases".into()))?;

    let token_ids_arr = token_ids
        .as_array()
        .ok_or_else(|| CallError::InvalidLogitBias("token_ids must be an array".into()))?;
    let token_ids = token_ids_arr
        .iter()
        .map(|v| {
            v.as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| CallError::InvalidLogitBias("token_ids must be u32 values".into()))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let biases = decode_f32_array("logit_bias.biases", biases, None)?;
    if token_ids.len() != biases.len() {
        return Err(CallError::InvalidLogitBias(format!(
            "token_ids length {} != biases length {}",
            token_ids.len(),
            biases.len()
        )));
    }
    Ok(SparseLogitBias { token_ids, biases })
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_vindex::{CallResourceLimits, CallSafetyPolicy};

    struct StaticRunner {
        output: Value,
    }

    impl CallProgramRunner for StaticRunner {
        fn run(&mut self, _call: &CallPatchOp, input: Value) -> Result<Value, CallError> {
            assert_eq!(input["residual"], json!([1.0, 2.0, 3.0]));
            Ok(self.output.clone())
        }
    }

    fn call() -> CallPatchOp {
        CallPatchOp {
            layer: 2,
            feature: 7,
            gate_vector_b64: None,
            monty_code: "def main(input):\n    return input\n".into(),
            code_hash: None,
            input_schema: json!({"include_token_ids": true, "include_token_text": true}),
            output_schema: json!({"keys": {"residual_delta": "delta", "logit_bias": "bias"}}),
            trigger: CallTrigger {
                score_threshold: Some(3.0),
                margin_threshold: Some(0.5),
                require_top_k: 2,
                ..Default::default()
            },
            limits: CallResourceLimits::default(),
            safety: CallSafetyPolicy::default(),
            metadata: Value::Null,
        }
    }

    #[test]
    fn trigger_requires_rank_score_and_margin() {
        let trigger = call().trigger;
        assert!(should_fire(
            &trigger,
            CallCandidate {
                rank: 1,
                score: -3.5,
                margin: Some(0.6),
            },
        ));
        assert!(!should_fire(
            &trigger,
            CallCandidate {
                rank: 3,
                score: 9.0,
                margin: Some(9.0),
            },
        ));
        assert!(!should_fire(
            &trigger,
            CallCandidate {
                rank: 1,
                score: 2.0,
                margin: Some(9.0),
            },
        ));
        assert!(!should_fire(
            &trigger,
            CallCandidate {
                rank: 1,
                score: 9.0,
                margin: Some(0.1),
            },
        ));
    }

    #[test]
    fn encode_input_includes_residual_and_tokens() {
        let call = call();
        let ctx = CallContext {
            layer: 2,
            position: 4,
            residual: &[1.0, -2.0],
            token_ids: &[10, 11],
            token_text: Some("hi"),
        };
        let encoded = encode_input(&call, &ctx);
        assert_eq!(encoded["layer"], 2);
        assert_eq!(encoded["position"], 4);
        assert_eq!(encoded["residual"], json!([1.0, -2.0]));
        assert_eq!(encoded["token_ids"], json!([10, 11]));
        assert_eq!(encoded["text"], "hi");
    }

    #[test]
    fn decode_output_reads_custom_keys() {
        let call = call();
        let output = decode_output(
            &call,
            &json!({
                "delta": [0.1, 0.2, 0.3],
                "bias": {"token_ids": [42, 43], "biases": [1.5, -0.5]}
            }),
            3,
        )
        .unwrap();
        assert_eq!(output.residual_delta.unwrap(), vec![0.1, 0.2, 0.3]);
        let bias = output.logit_bias.unwrap();
        assert_eq!(bias.token_ids, vec![42, 43]);
        assert_eq!(bias.biases, vec![1.5, -0.5]);
    }

    #[test]
    fn decode_output_rejects_wrong_residual_width() {
        let err = decode_output(&call(), &json!({"delta": [1.0, 2.0]}), 3).unwrap_err();
        assert!(matches!(
            err,
            CallError::WrongLength {
                expected: 3,
                actual: 2,
                ..
            }
        ));
    }

    #[test]
    fn safety_clamps_residual_delta_norm() {
        let mut output = CallOutput {
            residual_delta: Some(vec![3.0, 4.0]),
            logit_bias: None,
        };
        apply_safety(
            &mut output,
            &CallSafetyPolicy {
                residual_clamp_norm: Some(2.5),
                ..Default::default()
            },
        );
        let delta = output.residual_delta.unwrap();
        assert!((delta[0] - 1.5).abs() < 1e-6);
        assert!((delta[1] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn runtime_runs_full_lifecycle_and_counts_metrics() {
        let mut call = call();
        call.safety.residual_clamp_norm = Some(1.0);
        let ctx = CallContext {
            layer: 2,
            position: 0,
            residual: &[1.0, 2.0, 3.0],
            token_ids: &[],
            token_text: None,
        };
        let mut runtime = MontyCallRuntime::new(StaticRunner {
            output: json!({"delta": [0.0, 3.0, 4.0]}),
        });

        let output = runtime
            .execute_call(
                &call,
                CallCandidate {
                    rank: 1,
                    score: 3.5,
                    margin: Some(0.6),
                },
                &ctx,
                3,
            )
            .unwrap()
            .unwrap();

        assert_eq!(runtime.metrics().attempted, 1);
        assert_eq!(runtime.metrics().fired, 1);
        let delta = output.residual_delta.unwrap();
        assert!((delta[1] - 0.6).abs() < 1e-6);
        assert!((delta[2] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn runtime_skips_when_trigger_fails() {
        let ctx = CallContext {
            layer: 2,
            position: 0,
            residual: &[1.0, 2.0, 3.0],
            token_ids: &[],
            token_text: None,
        };
        let mut runtime = MontyCallRuntime::new(StaticRunner {
            output: json!({"delta": [1.0, 1.0, 1.0]}),
        });

        let output = runtime
            .execute_call(
                &call(),
                CallCandidate {
                    rank: 99,
                    score: 100.0,
                    margin: Some(100.0),
                },
                &ctx,
                3,
            )
            .unwrap();

        assert!(output.is_none());
        assert_eq!(runtime.metrics().attempted, 1);
        assert_eq!(runtime.metrics().skipped, 1);
        assert_eq!(runtime.metrics().fired, 0);
    }
}
