//! Criterion benchmarks for the Monty call-patch runtime.
//!
//! Covers three scenarios from the FUNCTIONS.md latency budget:
//!
//! * `no_calls`            — sparse WalkFfn with no call patches loaded.
//!   Validates that loading zero call patches adds no measurable overhead.
//! * `loaded_not_fired`    — one call patch loaded but its score-threshold
//!   is set so high that it never fires. The gate KNN cost is unchanged;
//!   only the trigger check adds a tiny amount of work.
//! * `one_call_per_token`  — one call patch fires on every token. Measures
//!   the full encode → run → decode → apply path using a zero-overhead
//!   in-process runner (no subprocess spawn).
//! * `encoder_raw_vs_topk` — compares raw-f32 encoding against the top-k
//!   basis codec at typical hidden sizes.
//!
//! Run:
//!   cargo bench -p larql-inference --bench monty_call
//!
//! Interpret results:
//! * `no_calls` is the baseline.
//! * `loaded_not_fired` overhead vs `no_calls` should be <1%.
//! * `one_call_per_token` overhead is the cost of one in-process call;
//!   it is NOT representative of a real subprocess-launched MontyVmRunner.
//! * `encoder_raw_vs_topk` shows codec overhead independently of runtime.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use larql_compute::FfnBackend;
use larql_inference::{
    monty_call::{
        encode_input, encode_topk_basis, CallCandidate, CallContext, CallError, CallOutput,
        CallProgramRunner, MontyCallRuntime,
    },
    test_utils::{attach_feature_major_f32_to_test_vindex, make_test_vindex, make_test_weights},
    vindex::{WalkFfn, WalkFfnConfig},
};
use larql_vindex::{CallPatchOp, CallResourceLimits, CallSafetyPolicy, CallTrigger, PatchedVindex};
use ndarray::Array2;
use serde_json::{json, Value};
use std::cell::RefCell;

// ── Shared fixtures ───────────────────────────────────────────────────────────

fn residual_row(hidden: usize) -> Array2<f32> {
    Array2::from_shape_vec(
        (1, hidden),
        (0..hidden).map(|i| (i as f32 + 1.0) * 0.01).collect(),
    )
    .unwrap()
}

fn make_always_fire_patch(hidden: usize) -> (CallPatchOp, Vec<f32>) {
    let patch = CallPatchOp {
        layer: 0,
        feature: 0,
        gate_vector_b64: None,
        monty_code: "def main(input):\n    return input\n".into(),
        code_hash: None,
        input_schema: json!({"include_token_ids": false, "include_token_text": false}),
        output_schema: json!({"keys": {"residual_delta": "residual_delta"}}),
        trigger: CallTrigger {
            score_threshold: None,
            margin_threshold: None,
            max_calls_per_token: 1,
            max_calls_per_sequence: None,
            cooldown_tokens: None,
            require_top_k: 1,
        },
        limits: CallResourceLimits {
            time_us: 0,
            ..Default::default()
        },
        safety: CallSafetyPolicy {
            residual_clamp_norm: Some(10.0),
            ..Default::default()
        },
        metadata: Value::Null,
    };
    let gate_vector = vec![100.0f32; hidden];
    (patch, gate_vector)
}

fn make_never_fire_patch(hidden: usize) -> (CallPatchOp, Vec<f32>) {
    let (mut patch, gate_vector) = make_always_fire_patch(hidden);
    // Score threshold high enough that the synthetic gate score never reaches it.
    patch.trigger.score_threshold = Some(f32::MAX);
    (patch, gate_vector)
}

// ── In-process runner (no subprocess, measures pure Rust call path) ───────────

struct ZeroDeltaRunner {
    hidden: usize,
}

impl CallProgramRunner for ZeroDeltaRunner {
    fn run(&mut self, _call: &CallPatchOp, _input: Value) -> Result<Value, CallError> {
        Ok(json!({"residual_delta": vec![0.0f32; self.hidden]}))
    }
}

// ── Benchmark: no_calls vs loaded_not_fired vs one_call_per_token ─────────────

fn bench_walk_ffn_call_paths(c: &mut Criterion) {
    let weights = make_test_weights();
    let hidden = weights.hidden_size;
    let mut base = make_test_vindex(&weights);
    attach_feature_major_f32_to_test_vindex(&weights, &mut base);
    let x = residual_row(hidden);

    let mut group = c.benchmark_group("walk_ffn_call_paths");
    group.throughput(Throughput::Elements(1));

    // Baseline: no call patches.
    {
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8);
        let ffn = WalkFfn::from_config(&weights, &base, cfg);
        group.bench_function("no_calls", |b| {
            b.iter(|| ffn.forward(0, &x))
        });
    }

    // Loaded but never fires (trigger score_threshold = f32::MAX).
    {
        let mut patched = PatchedVindex::new(base.clone());
        let (patch, gate_vec) = make_never_fire_patch(hidden);
        patched.insert_call_patch(patch, gate_vec);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 8);
        let ffn =
            WalkFfn::from_config(&weights, &patched, cfg).with_call_patches(&patched);
        group.bench_function("loaded_not_fired", |b| {
            b.iter(|| ffn.forward(0, &x))
        });
    }

    // One call fires per token (in-process zero-delta runner).
    {
        let mut patched = PatchedVindex::new(base.clone());
        let (patch, gate_vec) = make_always_fire_patch(hidden);
        patched.insert_call_patch(patch, gate_vec);
        let cfg = WalkFfnConfig::sparse(weights.num_layers, 1);
        let runtime = RefCell::new(MontyCallRuntime::new(ZeroDeltaRunner { hidden }));
        let ffn = WalkFfn::from_config(&weights, &patched, cfg)
            .with_call_patches(&patched)
            .with_call_runtime(&runtime);
        group.bench_function("one_call_per_token", |b| {
            b.iter(|| ffn.forward(0, &x))
        });
    }

    group.finish();
}

// ── Benchmark: encoder_raw_vs_topk ───────────────────────────────────────────

fn bench_encoder_raw_vs_topk(c: &mut Criterion) {
    // Typical hidden sizes (architecture-agnostic labels).
    let hidden_sizes: &[(usize, &str)] = &[
        (256, "h256"),   // small test fixture
        (2560, "h2560"), // typical 4B dense
        (4096, "h4096"), // typical 8B / MoE shared
    ];
    // k values for top-k codec.
    let k_values: &[(usize, &str)] = &[(8, "k8"), (32, "k32")];

    let mut group = c.benchmark_group("encoder_raw_vs_topk");

    for &(hidden, hlabel) in hidden_sizes {
        let residual: Vec<f32> = (0..hidden).map(|i| (i as f32) * 0.001 - 0.5).collect();
        group.throughput(Throughput::Elements(hidden as u64));

        // Raw f32 encode via encode_input with flat schema.
        {
            let patch = CallPatchOp {
                layer: 0,
                feature: 0,
                gate_vector_b64: None,
                monty_code: String::new(),
                code_hash: None,
                input_schema: json!({"include_token_ids": false, "include_token_text": false}),
                output_schema: Value::Null,
                trigger: CallTrigger::default(),
                limits: CallResourceLimits::default(),
                safety: CallSafetyPolicy::default(),
                metadata: Value::Null,
            };
            let ctx = CallContext {
                layer: 0,
                position: 0,
                residual: &residual,
                token_ids: &[],
                token_text: None,
            };
            group.bench_with_input(
                BenchmarkId::new("raw_f32", hlabel),
                &hidden,
                |b, _| b.iter(|| encode_input(&patch, &ctx, None).unwrap()),
            );
        }

        // Top-k basis encode: small basis (k rows × hidden cols) of unit vectors.
        for &(k, klabel) in k_values {
            if k > hidden {
                continue;
            }
            let basis: Vec<Vec<f32>> = (0..k)
                .map(|i| {
                    let mut row = vec![0.0f32; hidden];
                    row[i] = 1.0;
                    row
                })
                .collect();
            let label = format!("{hlabel}/{klabel}");
            group.bench_with_input(
                BenchmarkId::new("topk_basis", &label),
                &(k, hidden),
                |b, _| b.iter(|| encode_topk_basis(&residual, &basis, k)),
            );
        }
    }

    group.finish();
}

// ── Benchmark: MontyCallRuntime execute_call overhead ────────────────────────

fn bench_monty_call_runtime(c: &mut Criterion) {
    let hidden = 256usize; // small fixture to keep bench fast
    let patch = CallPatchOp {
        layer: 0,
        feature: 0,
        gate_vector_b64: None,
        monty_code: "def main(input):\n    return input\n".into(),
        code_hash: None,
        input_schema: json!({"include_token_ids": false, "include_token_text": false}),
        output_schema: json!({"keys": {"residual_delta": "residual_delta"}}),
        trigger: CallTrigger {
            score_threshold: None,
            margin_threshold: None,
            max_calls_per_token: 1,
            max_calls_per_sequence: None,
            cooldown_tokens: None,
            require_top_k: 1,
        },
        limits: CallResourceLimits {
            time_us: 0,
            ..Default::default()
        },
        safety: CallSafetyPolicy::default(),
        metadata: Value::Null,
    };
    let residual: Vec<f32> = vec![0.1f32; hidden];
    let ctx = CallContext {
        layer: 0,
        position: 0,
        residual: &residual,
        token_ids: &[],
        token_text: None,
    };
    let candidate = CallCandidate {
        rank: 1,
        score: 5.0,
        margin: Some(1.0),
        calls_already_fired: 0,
    };

    let mut group = c.benchmark_group("monty_call_runtime");
    group.throughput(Throughput::Elements(1));

    // Measures trigger check + encode + in-process run + decode path.
    group.bench_function("in_process_runner", |b| {
        let mut runtime = MontyCallRuntime::new(ZeroDeltaRunner { hidden });
        b.iter(|| {
            let _ = runtime.execute_call(&patch, candidate, &ctx, hidden);
        })
    });

    // Measures skipped-trigger path (score below threshold = fast early exit).
    {
        let mut no_fire_patch = patch.clone();
        no_fire_patch.trigger.score_threshold = Some(f32::MAX);
        group.bench_function("skipped_trigger", |b| {
            let mut runtime = MontyCallRuntime::new(ZeroDeltaRunner { hidden });
            b.iter(|| {
                let _ = runtime.execute_call(&no_fire_patch, candidate, &ctx, hidden);
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_walk_ffn_call_paths,
    bench_encoder_raw_vs_topk,
    bench_monty_call_runtime,
);
criterion_main!(benches);
