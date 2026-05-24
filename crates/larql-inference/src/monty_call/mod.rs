//! Runtime call-patch support.
//!
//! This module intentionally stops at the deterministic boundary around a
//! call: trigger checks, JSON-ish input encoding, output decoding, and safety
//! clamps. The actual Monty VM integration can sit behind this surface without
//! changing the vindex patch format or the inference hook contract.

use larql_vindex::{
    patch::core::{decode_gate_vector, encode_gate_vector},
    CallPatchOp, CallSafetyPolicy, CallTrigger,
};
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
    /// How many call patches have already fired for this token position.
    /// Checked against `CallTrigger::max_calls_per_token` before executing.
    pub calls_already_fired: usize,
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
    /// Calls that completed but exceeded their `time_us` budget.
    /// The output is discarded and the position is treated as if the
    /// call was not fired (output zeroed, metric incremented).
    pub timed_out: u64,
}

pub trait CallProgramRunner {
    fn run(&mut self, call: &CallPatchOp, input: Value) -> Result<Value, CallError>;
}

pub trait CallPatchLookup {
    fn call_patch(&self, layer: usize, feature: usize) -> Option<&CallPatchOp>;
}

impl CallPatchLookup for larql_vindex::PatchedVindex {
    fn call_patch(&self, layer: usize, feature: usize) -> Option<&CallPatchOp> {
        self.call_patch(layer, feature)
    }
}

pub trait WalkCallRuntime {
    fn execute_call(
        &self,
        call: &CallPatchOp,
        candidate: CallCandidate,
        ctx: &CallContext<'_>,
        hidden_size: usize,
    ) -> Result<Option<CallOutput>, CallError>;
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
        let t0 = std::time::Instant::now();
        let raw = match self.runner.run(call, input) {
            Ok(raw) => raw,
            Err(err) => {
                self.metrics.failed += 1;
                return Err(err);
            }
        };
        let elapsed_us = t0.elapsed().as_micros() as u64;
        if call.limits.time_us > 0 && elapsed_us > call.limits.time_us {
            self.metrics.timed_out += 1;
            self.metrics.skipped += 1;
            return Ok(None);
        }

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

impl<R: CallProgramRunner> WalkCallRuntime for std::cell::RefCell<MontyCallRuntime<R>> {
    fn execute_call(
        &self,
        call: &CallPatchOp,
        candidate: CallCandidate,
        ctx: &CallContext<'_>,
        hidden_size: usize,
    ) -> Result<Option<CallOutput>, CallError> {
        self.borrow_mut()
            .execute_call(call, candidate, ctx, hidden_size)
    }
}

pub fn should_fire(trigger: &CallTrigger, candidate: CallCandidate) -> bool {
    if candidate.rank == 0 || candidate.rank > trigger.require_top_k {
        return false;
    }
    if trigger.max_calls_per_token == 0
        || candidate.calls_already_fired >= trigger.max_calls_per_token
    {
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
    true
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

    // Encode the residual according to the source codec specified in input_schema.
    // The structured schema form is: {"sources": [{"kind": "current_residual", "key": "K", "codec": {...}}]}
    // Legacy flat form omits sources and defaults to raw_f32 under key "residual".
    let residual_encoded = encode_residual_source(call, ctx.residual);
    for (k, v) in residual_encoded {
        obj.insert(k, v);
    }

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

/// Encode the residual source(s) from a call's input_schema.
/// Returns a vec of (key, encoded_value) pairs to insert into the input dict.
fn encode_residual_source(call: &CallPatchOp, residual: &[f32]) -> Vec<(String, Value)> {
    if let Some(sources) = call.input_schema.get("sources").and_then(|s| s.as_array()) {
        let mut out = Vec::new();
        for src in sources {
            let kind = src.get("kind").and_then(|k| k.as_str()).unwrap_or("current_residual");
            if kind != "current_residual" {
                continue;
            }
            let key = src.get("key").and_then(|k| k.as_str()).unwrap_or("residual");
            let codec_kind = src
                .get("codec")
                .and_then(|c| c.get("kind"))
                .and_then(|k| k.as_str())
                .unwrap_or("raw_f32");
            let encoded = match codec_kind {
                "top_k_basis" => {
                    let k = src
                        .get("codec")
                        .and_then(|c| c.get("k"))
                        .and_then(|v| v.as_u64())
                        .unwrap_or(8) as usize;
                    if let Some(basis) = extract_basis_from_schema(src.get("codec").unwrap_or(&Value::Null)) {
                        encode_topk_basis(residual, &basis, k)
                    } else {
                        json!(residual) // fallback to raw if basis missing
                    }
                }
                _ => json!(residual), // raw_f32 default
            };
            out.push((key.to_string(), encoded));
        }
        if !out.is_empty() {
            return out;
        }
    }
    // Legacy flat schema: always raw residual under "residual"
    vec![("residual".to_string(), json!(residual))]
}

pub fn decode_output(
    call: &CallPatchOp,
    obj: &Value,
    hidden_size: usize,
) -> Result<CallOutput, CallError> {
    let residual_delta = decode_residual_delta_sink(call, obj, hidden_size)?;

    let logit_key = output_key(call, "logit_bias", "logit_bias");
    let logit_bias = match obj.get(&logit_key) {
        Some(value) => Some(decode_sparse_logit_bias(value)?),
        None => None,
    };

    Ok(CallOutput {
        residual_delta,
        logit_bias,
    })
}

/// Decode the residual_delta sink from a Monty output object.
/// Checks the output_schema sinks array for codec type; defaults to raw_f32.
fn decode_residual_delta_sink(
    call: &CallPatchOp,
    obj: &Value,
    hidden_size: usize,
) -> Result<Option<Vec<f32>>, CallError> {
    if let Some(sinks) = call.output_schema.get("sinks").and_then(|s| s.as_array()) {
        for sink in sinks {
            let kind = sink.get("kind").and_then(|k| k.as_str()).unwrap_or("residual_delta");
            if kind != "residual_delta" {
                continue;
            }
            let key = sink.get("key").and_then(|k| k.as_str()).unwrap_or("residual_delta");
            let codec_kind = sink
                .get("codec")
                .and_then(|c| c.get("kind"))
                .and_then(|k| k.as_str())
                .unwrap_or("raw_f32");

            return match obj.get(key) {
                None => Ok(None),
                Some(value) => match codec_kind {
                    "sparse_basis_delta" => {
                        if let Some(basis) = extract_basis_from_schema(sink.get("codec").unwrap_or(&Value::Null)) {
                            decode_sparse_basis_delta(key, value, &basis, hidden_size).map(Some)
                        } else {
                            decode_f32_array(key, value, Some(hidden_size)).map(Some)
                        }
                    }
                    _ => decode_f32_array(key, value, Some(hidden_size)).map(Some),
                },
            };
        }
    }
    // Legacy flat output_schema: use keys map for residual_delta
    let residual_key = output_key(call, "residual_delta", "residual_delta");
    match obj.get(&residual_key) {
        Some(value) => decode_f32_array(&residual_key, value, Some(hidden_size)).map(Some),
        None => Ok(None),
    }
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

/// Encode a residual vector using the TopK-basis codec.
///
/// Projects `residual` onto each row of `basis` (dot products), keeps the top-k
/// by absolute magnitude, and returns `{"indices": [...], "values": [...]}`.
///
/// The basis is stored column-major as a flat f32 vector with shape
/// `(num_basis_vectors, hidden_size)`. Each row is one basis vector.
pub fn encode_topk_basis(residual: &[f32], basis: &[Vec<f32>], k: usize) -> Value {
    let mut scores: Vec<(usize, f32)> = basis
        .iter()
        .enumerate()
        .map(|(i, bvec)| {
            let dot: f32 = bvec.iter().zip(residual.iter()).map(|(b, r)| b * r).sum();
            (i, dot)
        })
        .collect();
    scores.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);
    scores.sort_by_key(|(i, _)| *i); // stable order for determinism

    let indices: Vec<usize> = scores.iter().map(|(i, _)| *i).collect();
    let values: Vec<f32> = scores.iter().map(|(_, v)| *v).collect();
    json!({"indices": indices, "values": values})
}

/// Decode a sparse basis delta back into a hidden-size residual vector.
///
/// Expects `{"basis_indices": [...], "values": [...]}`. Reconstructs
/// `sum(values[i] * basis[basis_indices[i]])`.
pub fn decode_sparse_basis_delta(
    key: &str,
    value: &Value,
    basis: &[Vec<f32>],
    hidden_size: usize,
) -> Result<Vec<f32>, CallError> {
    let indices = value
        .get("basis_indices")
        .and_then(|v| v.as_array())
        .ok_or_else(|| CallError::ExpectedArray(format!("{key}.basis_indices")))?;
    let values = value
        .get("values")
        .and_then(|v| v.as_array())
        .ok_or_else(|| CallError::ExpectedArray(format!("{key}.values")))?;

    if indices.len() != values.len() {
        return Err(CallError::WrongLength {
            key: format!("{key}.basis_indices/values"),
            expected: indices.len(),
            actual: values.len(),
        });
    }

    let mut out = vec![0.0f32; hidden_size];
    for (idx_val, coeff_val) in indices.iter().zip(values.iter()) {
        let idx = idx_val
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| CallError::ExpectedNumber(format!("{key}.basis_indices")))?;
        let coeff = coeff_val
            .as_f64()
            .map(|f| f as f32)
            .ok_or_else(|| CallError::ExpectedNumber(format!("{key}.values")))?;
        if idx >= basis.len() {
            return Err(CallError::InvalidLogitBias(format!(
                "{key}: basis index {idx} out of range (basis has {} vectors)",
                basis.len()
            )));
        }
        for (o, b) in out.iter_mut().zip(basis[idx].iter()) {
            *o += coeff * b;
        }
    }
    Ok(out)
}

/// Extract a basis matrix stored as a JSON array of base64-encoded f32 rows
/// inside `schema["basis"]`. Returns `None` if the key is absent.
pub fn extract_basis_from_schema(schema: &Value) -> Option<Vec<Vec<f32>>> {
    let rows = schema.get("basis")?.as_array()?;
    rows.iter()
        .map(|row| {
            let b64 = row.as_str()?;
            decode_gate_vector(b64).ok()
        })
        .collect()
}

/// Encode a basis matrix as a JSON array of base64-encoded f32 rows.
pub fn encode_basis_to_schema(basis: &[Vec<f32>]) -> Value {
    let rows: Vec<Value> = basis
        .iter()
        .map(|row| Value::String(encode_gate_vector(row)))
        .collect();
    Value::Array(rows)
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

    struct SlowRunner {
        sleep_us: u64,
        output: Value,
    }

    impl CallProgramRunner for SlowRunner {
        fn run(&mut self, _call: &CallPatchOp, _input: Value) -> Result<Value, CallError> {
            std::thread::sleep(std::time::Duration::from_micros(self.sleep_us));
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
                calls_already_fired: 0,
            },
        ));
        assert!(!should_fire(
            &trigger,
            CallCandidate {
                rank: 3,
                score: 9.0,
                margin: Some(9.0),
                calls_already_fired: 0,
            },
        ));
        assert!(!should_fire(
            &trigger,
            CallCandidate {
                rank: 1,
                score: 2.0,
                margin: Some(9.0),
                calls_already_fired: 0,
            },
        ));
        assert!(!should_fire(
            &trigger,
            CallCandidate {
                rank: 1,
                score: 9.0,
                margin: Some(0.1),
                calls_already_fired: 0,
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
        call.limits.time_us = 0;
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
                    calls_already_fired: 0,
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
                    calls_already_fired: 0,
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

    #[test]
    fn should_fire_blocked_when_budget_exhausted() {
        let trigger = call().trigger; // max_calls_per_token = 2
        let mut t = call().trigger;
        t.score_threshold = None;
        t.margin_threshold = None;
        t.max_calls_per_token = 2;
        t.require_top_k = 10;
        assert!(should_fire(
            &t,
            CallCandidate {
                rank: 1,
                score: 1.0,
                margin: None,
                calls_already_fired: 0,
            }
        ));
        assert!(should_fire(
            &t,
            CallCandidate {
                rank: 2,
                score: 1.0,
                margin: None,
                calls_already_fired: 1,
            }
        ));
        assert!(!should_fire(
            &t,
            CallCandidate {
                rank: 3,
                score: 1.0,
                margin: None,
                calls_already_fired: 2,
            }
        ));
        // max_calls_per_token = 0 always blocks
        t.max_calls_per_token = 0;
        assert!(!should_fire(
            &t,
            CallCandidate {
                rank: 1,
                score: 1.0,
                margin: None,
                calls_already_fired: 0,
            }
        ));
        drop(trigger);
    }

    #[test]
    fn runtime_counts_timeout_when_call_exceeds_time_budget() {
        let mut call = call();
        call.trigger.score_threshold = None;
        call.trigger.margin_threshold = None;
        // 1 µs budget — any real sleep will exceed this.
        call.limits.time_us = 1;

        let ctx = CallContext {
            layer: 2,
            position: 0,
            residual: &[1.0, 2.0, 3.0],
            token_ids: &[],
            token_text: None,
        };
        let mut runtime = MontyCallRuntime::new(SlowRunner {
            sleep_us: 2_000, // 2 ms, well over 1 µs budget
            output: json!({"delta": [0.1, 0.2, 0.3]}),
        });

        let output = runtime
            .execute_call(
                &call,
                CallCandidate {
                    rank: 1,
                    score: 5.0,
                    margin: Some(1.0),
                    calls_already_fired: 0,
                },
                &ctx,
                3,
            )
            .unwrap();

        assert!(output.is_none(), "timed-out call should return None");
        assert_eq!(runtime.metrics().timed_out, 1);
        assert_eq!(runtime.metrics().skipped, 1);
        assert_eq!(runtime.metrics().fired, 0);
    }

    #[test]
    fn runtime_fires_when_within_time_budget() {
        let mut call = call();
        call.trigger.score_threshold = None;
        call.trigger.margin_threshold = None;
        // Very generous budget — should never time out.
        call.limits.time_us = 1_000_000;

        let ctx = CallContext {
            layer: 2,
            position: 0,
            residual: &[1.0, 2.0, 3.0],
            token_ids: &[],
            token_text: None,
        };
        let mut runtime = MontyCallRuntime::new(StaticRunner {
            output: json!({"delta": [0.1, 0.2, 0.3]}),
        });

        let output = runtime
            .execute_call(
                &call,
                CallCandidate {
                    rank: 1,
                    score: 5.0,
                    margin: Some(1.0),
                    calls_already_fired: 0,
                },
                &ctx,
                3,
            )
            .unwrap()
            .expect("should fire within budget");

        assert!(output.residual_delta.is_some());
        assert_eq!(runtime.metrics().fired, 1);
        assert_eq!(runtime.metrics().timed_out, 0);
    }

    // ── TopK basis codec tests ────────────────────────────────────────────────

    fn identity_basis(n: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect())
            .collect()
    }

    #[test]
    fn topk_basis_encode_returns_top_k_by_abs_magnitude() {
        // residual = [3, -5, 1, 2], identity basis → scores = residual components
        let residual = vec![3.0f32, -5.0, 1.0, 2.0];
        let basis = identity_basis(4);
        let encoded = encode_topk_basis(&residual, &basis, 2);
        let indices: Vec<usize> = encoded["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        let values: Vec<f32> = encoded["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect();
        // Top-2 by abs: -5 (idx 1), 3 (idx 0); sorted by index → [0, 1]
        assert_eq!(indices, vec![0, 1]);
        assert!((values[0] - 3.0).abs() < 1e-6, "value at idx 0 = 3.0");
        assert!((values[1] - (-5.0)).abs() < 1e-6, "value at idx 1 = -5.0");
    }

    #[test]
    fn topk_basis_encode_decode_roundtrip_with_identity_basis() {
        let residual = vec![1.0f32, 0.0, -2.0, 0.5];
        let basis = identity_basis(4);
        let encoded = encode_topk_basis(&residual, &basis, 4);
        // Re-encode as sparse_basis_delta form
        let as_sparse = json!({
            "basis_indices": encoded["indices"],
            "values": encoded["values"],
        });
        let decoded = decode_sparse_basis_delta("test", &as_sparse, &basis, 4).unwrap();
        for (a, b) in residual.iter().zip(decoded.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn topk_basis_encode_decode_roundtrip_partial_k() {
        // k=2 keeps only the 2 largest; reconstructed vector is a 2-component approximation
        let residual = vec![10.0f32, 0.001, 0.001, 0.001];
        let basis = identity_basis(4);
        let encoded = encode_topk_basis(&residual, &basis, 2);
        let as_sparse = json!({
            "basis_indices": encoded["indices"],
            "values": encoded["values"],
        });
        let decoded = decode_sparse_basis_delta("test", &as_sparse, &basis, 4).unwrap();
        // Should recover the dominant component
        assert!((decoded[0] - 10.0).abs() < 1e-5, "dominant component preserved");
    }

    #[test]
    fn decode_sparse_basis_delta_rejects_out_of_range_index() {
        let basis = identity_basis(3);
        let sparse = json!({ "basis_indices": [5], "values": [1.0] });
        let err = decode_sparse_basis_delta("key", &sparse, &basis, 3).unwrap_err();
        assert!(
            matches!(err, CallError::InvalidLogitBias(_)),
            "expected InvalidLogitBias for out-of-range index"
        );
    }

    #[test]
    fn basis_schema_roundtrip() {
        let basis = vec![vec![1.0f32, 0.0], vec![0.0, 1.0]];
        let encoded_schema = encode_basis_to_schema(&basis);
        let schema = json!({ "basis": encoded_schema });
        let recovered = extract_basis_from_schema(&schema).unwrap();
        assert_eq!(recovered.len(), 2);
        assert!((recovered[0][0] - 1.0).abs() < 1e-6);
        assert!((recovered[1][1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn encode_input_uses_topk_basis_codec_when_specified() {
        let basis = identity_basis(3);
        let basis_schema = encode_basis_to_schema(&basis);
        let call_op = CallPatchOp {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            monty_code: "".into(),
            code_hash: None,
            input_schema: json!({
                "sources": [{
                    "kind": "current_residual",
                    "key": "compressed",
                    "codec": { "kind": "top_k_basis", "k": 2, "basis": basis_schema }
                }]
            }),
            output_schema: Value::Null,
            trigger: CallTrigger::default(),
            limits: larql_vindex::CallResourceLimits::default(),
            safety: larql_vindex::CallSafetyPolicy::default(),
            metadata: Value::Null,
        };
        let residual = vec![5.0f32, -3.0, 1.0];
        let ctx = CallContext {
            layer: 0,
            position: 0,
            residual: &residual,
            token_ids: &[],
            token_text: None,
        };
        let encoded = encode_input(&call_op, &ctx);
        let compressed = &encoded["compressed"];
        assert!(compressed.get("indices").is_some(), "should have indices");
        assert!(compressed.get("values").is_some(), "should have values");
        // top-2: abs([5, -3, 1]) → [5, 3] → idx 0 and 1
        let indices: Vec<usize> = compressed["indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as usize)
            .collect();
        assert!(indices.contains(&0), "idx 0 (score 5) should be in top-2");
        assert!(indices.contains(&1), "idx 1 (score -3) should be in top-2");
        assert!(!indices.contains(&2), "idx 2 (score 1) should not be in top-2");
    }
}
