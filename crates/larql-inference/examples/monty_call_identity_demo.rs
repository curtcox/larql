//! Identity call-patch demo — exercises MontyCallRuntime end-to-end
//! without any model weights or real Monty VM.
//!
//! Demonstrates:
//!   1. Building a CallPatchOp with trigger / limits / safety.
//!   2. A stub CallProgramRunner that returns a known residual delta.
//!   3. MontyCallRuntime::execute_call — trigger check, encode, decode, safety.
//!   4. Metrics (attempted / fired / skipped / timed_out / failed).
//!   5. Trigger blocking: budget exhaustion, score threshold, rank guard.
//!
//! Run: cargo run -p larql-inference --example monty_call_identity_demo

use larql_inference::monty_call::{
    CallCandidate, CallContext, CallError, CallProgramRunner, MontyCallRuntime, MontyCallMetrics,
};
use larql_vindex::{
    CallPatchOp, CallResourceLimits, CallSafetyPolicy, CallTrigger,
};
use serde_json::{json, Value};

// ── Stub runner ──────────────────────────────────────────────────────────────

/// Returns a residual_delta that scales the input residual by a fixed factor.
struct ScaleRunner {
    scale: f32,
}

impl CallProgramRunner for ScaleRunner {
    fn run(&mut self, call: &CallPatchOp, input: Value) -> Result<Value, CallError> {
        let residual_key = call
            .input_schema
            .get("sources")
            .and_then(|s| s.as_array())
            .and_then(|arr| arr.first())
            .and_then(|src| src.get("key"))
            .and_then(|k| k.as_str())
            .unwrap_or("residual");

        let residual = input
            .get(residual_key)
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_f64().map(|f| f as f32 * self.scale))
                    .collect::<Vec<f32>>()
            })
            .unwrap_or_default();

        Ok(json!({ "residual_delta": residual }))
    }
}

/// Always sleeps past the time budget.
struct SlowRunner;

impl CallProgramRunner for SlowRunner {
    fn run(&mut self, _call: &CallPatchOp, _input: Value) -> Result<Value, CallError> {
        std::thread::sleep(std::time::Duration::from_millis(10));
        Ok(json!({ "residual_delta": [0.0] }))
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn default_patch(_hidden: usize) -> CallPatchOp {
    CallPatchOp {
        layer: 0,
        feature: 7,
        gate_vector_b64: None,
        monty_code: "def main(input): return {'residual_delta': input['residual']}".into(),
        code_hash: None,
        input_schema: json!({
            "sources": [{"kind": "current_residual", "key": "residual"}]
        }),
        output_schema: json!({
            "sinks": [{"kind": "residual_delta", "key": "residual_delta"}]
        }),
        trigger: CallTrigger {
            score_threshold: Some(0.5),
            margin_threshold: None,
            max_calls_per_token: 2,
            require_top_k: 3,
            ..Default::default()
        },
        limits: CallResourceLimits {
            time_us: 1_000_000, // 1s — generous for the identity demo
            memory_bytes: 1_048_576,
            steps: 10_000,
        },
        safety: CallSafetyPolicy {
            residual_clamp_norm: Some(10.0),
            ..Default::default()
        },
        metadata: json!({"description": "identity scale demo"}),
    }
}

fn ctx_at(residual: &[f32], position: usize) -> CallContext<'_> {
    CallContext {
        layer: 0,
        position,
        residual,
        token_ids: &[],
        token_text: None,
    }
}

fn assert_metrics(m: MontyCallMetrics, label: &str, attempted: u64, fired: u64, skipped: u64, timed_out: u64, failed: u64) {
    assert_eq!(m.attempted, attempted, "{label}: attempted");
    assert_eq!(m.fired, fired, "{label}: fired");
    assert_eq!(m.skipped, skipped, "{label}: skipped");
    assert_eq!(m.timed_out, timed_out, "{label}: timed_out");
    assert_eq!(m.failed, failed, "{label}: failed");
}

// ── Demo ─────────────────────────────────────────────────────────────────────

fn main() {
    let hidden = 4;
    println!("=== MontyCall Identity Demo ===\n");

    // ── 1. Happy path: scale residual by 0.5 ────────────────────────────────
    {
        let call = default_patch(hidden);
        let residual = vec![1.0f32, 2.0, 3.0, 4.0];
        let ctx = ctx_at(&residual, 0);
        let candidate = CallCandidate {
            rank: 1,
            score: 0.9,
            margin: Some(0.4),
            calls_already_fired: 0,
        };

        let mut rt = MontyCallRuntime::new(ScaleRunner { scale: 0.5 });
        let output = rt
            .execute_call(&call, candidate, &ctx, hidden)
            .expect("execute_call")
            .expect("output present");
        let delta = output.residual_delta.as_deref().unwrap();
        println!("1. Scale 0.5x residual [1,2,3,4] → delta = {delta:?}");
        assert_eq!(delta, &[0.5, 1.0, 1.5, 2.0], "delta should be residual * 0.5");
        assert_metrics(rt.metrics(), "happy path", 1, 1, 0, 0, 0);
        println!("   metrics: {:?}", rt.metrics());
        println!("   ✓ fired, delta correct\n");
    }

    // ── 2. Trigger blocked by score threshold ───────────────────────────────
    {
        let call = default_patch(hidden);
        let residual = vec![1.0f32; hidden];
        let ctx = ctx_at(&residual, 0);
        let candidate = CallCandidate {
            rank: 1,
            score: 0.1, // below threshold 0.5
            margin: Some(0.4),
            calls_already_fired: 0,
        };
        let mut rt = MontyCallRuntime::new(ScaleRunner { scale: 1.0 });
        let output = rt.execute_call(&call, candidate, &ctx, hidden).expect("no error");
        assert!(output.is_none(), "should be blocked by score threshold");
        assert_metrics(rt.metrics(), "score block", 1, 0, 1, 0, 0);
        println!("2. Score 0.1 < threshold 0.5 → blocked (skipped=1)");
        println!("   ✓\n");
    }

    // ── 3. Trigger blocked by rank guard ────────────────────────────────────
    {
        let call = default_patch(hidden);
        let residual = vec![1.0f32; hidden];
        let ctx = ctx_at(&residual, 0);
        let candidate = CallCandidate {
            rank: 5, // require_top_k = 3
            score: 0.9,
            margin: Some(0.4),
            calls_already_fired: 0,
        };
        let mut rt = MontyCallRuntime::new(ScaleRunner { scale: 1.0 });
        let output = rt.execute_call(&call, candidate, &ctx, hidden).expect("no error");
        assert!(output.is_none(), "rank 5 > require_top_k 3 should block");
        assert_metrics(rt.metrics(), "rank block", 1, 0, 1, 0, 0);
        println!("3. Rank 5 > require_top_k 3 → blocked (skipped=1)");
        println!("   ✓\n");
    }

    // ── 4. Trigger blocked by budget exhaustion ──────────────────────────────
    {
        let call = default_patch(hidden); // max_calls_per_token = 2
        let residual = vec![1.0f32; hidden];
        let ctx = ctx_at(&residual, 0);
        let candidate = CallCandidate {
            rank: 1,
            score: 0.9,
            margin: Some(0.4),
            calls_already_fired: 2, // already at max
        };
        let mut rt = MontyCallRuntime::new(ScaleRunner { scale: 1.0 });
        let output = rt.execute_call(&call, candidate, &ctx, hidden).expect("no error");
        assert!(output.is_none(), "budget exhausted should block");
        assert_metrics(rt.metrics(), "budget block", 1, 0, 1, 0, 0);
        println!("4. calls_already_fired=2 = max_calls_per_token → blocked (skipped=1)");
        println!("   ✓\n");
    }

    // ── 5. Time budget exceeded → timed_out, output discarded ───────────────
    {
        let mut call = default_patch(hidden);
        call.limits.time_us = 1; // 1 µs — will always exceed
        let residual = vec![1.0f32; hidden];
        let ctx = ctx_at(&residual, 0);
        let candidate = CallCandidate {
            rank: 1,
            score: 0.9,
            margin: Some(0.4),
            calls_already_fired: 0,
        };
        let mut rt = MontyCallRuntime::new(SlowRunner);
        let output = rt.execute_call(&call, candidate, &ctx, hidden).expect("no error");
        assert!(output.is_none(), "timed out → output should be None");
        assert_metrics(rt.metrics(), "timeout", 1, 0, 1, 1, 0);
        println!("5. time_us=1 exceeded → timed_out=1, output discarded");
        println!("   ✓\n");
    }

    // ── 6. Safety clamp: large delta clamped to norm limit ──────────────────
    {
        let mut call = default_patch(hidden);
        call.safety.residual_clamp_norm = Some(1.0);
        // ScaleRunner(1.0) passes residual [10, 0, 0, 0] → delta [10, 0, 0, 0]; norm = 10
        let residual = vec![10.0f32, 0.0, 0.0, 0.0];
        let ctx = ctx_at(&residual, 0);
        let candidate = CallCandidate {
            rank: 1,
            score: 0.9,
            margin: Some(0.4),
            calls_already_fired: 0,
        };
        let mut rt = MontyCallRuntime::new(ScaleRunner { scale: 1.0 });
        let output = rt
            .execute_call(&call, candidate, &ctx, hidden)
            .expect("execute_call")
            .expect("output present");
        let delta = output.residual_delta.as_deref().unwrap();
        let norm = delta.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("6. Safety clamp norm=1.0 on delta [10,0,0,0] → norm = {norm:.4}");
        assert!((norm - 1.0).abs() < 1e-5, "norm should be clamped to 1.0, got {norm}");
        println!("   ✓\n");
    }

    // ── 7. Two sequential calls respect max_calls_per_token ─────────────────
    {
        let call = default_patch(hidden); // max_calls_per_token = 2
        let residual = vec![1.0f32; hidden];
        let ctx = ctx_at(&residual, 0);
        let mut rt = MontyCallRuntime::new(ScaleRunner { scale: 0.1 });

        let c1 = CallCandidate { rank: 1, score: 0.9, margin: Some(0.4), calls_already_fired: 0 };
        let c2 = CallCandidate { rank: 2, score: 0.8, margin: Some(0.3), calls_already_fired: 1 };
        let c3 = CallCandidate { rank: 3, score: 0.7, margin: Some(0.2), calls_already_fired: 2 };

        let o1 = rt.execute_call(&call, c1, &ctx, hidden).unwrap();
        let o2 = rt.execute_call(&call, c2, &ctx, hidden).unwrap();
        let o3 = rt.execute_call(&call, c3, &ctx, hidden).unwrap();

        assert!(o1.is_some(), "first call should fire");
        assert!(o2.is_some(), "second call should fire (budget=2)");
        assert!(o3.is_none(), "third call should be blocked (budget exhausted)");
        assert_metrics(rt.metrics(), "budget 2", 3, 2, 1, 0, 0);
        println!("7. max_calls_per_token=2: calls 1+2 fire, call 3 blocked");
        println!("   metrics: {:?}", rt.metrics());
        println!("   ✓\n");
    }

    println!("=== All checks passed ===");
}
