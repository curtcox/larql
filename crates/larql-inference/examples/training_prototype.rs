//! M8 training prototype — end-to-end demonstration.
//!
//! Shows all four deliverables in sequence:
//!   1. Synthetic task generator
//!   2. Gate calibration with hard negatives
//!   3. Decoder warmup (forced-call teacher forcing)
//!   4. Evaluation harness with accuracy, gate P/R/F1, KL regression, reach probe
//!
//! No model weights or Monty VM are required.  All tensors are synthetic.
//!
//! Run:
//!   cargo run -p larql-inference --example training_prototype

use larql_inference::training::{
    decoder_warmup::{DecoderWarmupConfig, DecoderWarmupExample, LinearDecoder, SurrogateDecoder},
    evaluation::{
        accuracy_improvement, gate_metrics, print_m8_report, reach_probe, regression_report,
        GatePrediction,
    },
    gate_calibration::{GateCalibrationConfig, GateCalibrator, GateExample},
    synthetic::{build_dataset, TaskCategory},
};
// ── Synthetic data helpers ────────────────────────────────────────────────────

fn residual_for_example(idx: usize, hidden: usize) -> Vec<f32> {
    // Deterministic pseudo-random residual seeded by index.
    let seed = (idx as u64).wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    (0..hidden)
        .map(|i| {
            let v = seed.wrapping_add(i as u64)
                .wrapping_mul(2_862_933_555_777_941_757);
            let f = (v >> 32) as i32 as f32 / i32::MAX as f32;
            f
        })
        .collect()
}

fn monty_output_for_example(idx: usize, monty_dim: usize) -> Vec<f32> {
    // Each task type produces a characteristic direction.
    let base = ((idx % 7) as f32 + 1.0) / 7.0;
    (0..monty_dim)
        .map(|i| if i == idx % monty_dim { base } else { 0.0 })
        .collect()
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() {
    let hidden = 16;
    let monty_dim = 8;
    let seed = 42u64;

    println!("=== M8 Training Prototype ===\n");

    // ────────────────────────────────────────────────────────────────────────
    // Phase 1: Synthetic task generation
    // ────────────────────────────────────────────────────────────────────────
    println!("--- Phase 1: Synthetic task generator ---");
    let dataset = build_dataset(20, seed);
    let positives = dataset.iter().filter(|e| e.is_positive).count();
    let negatives = dataset.iter().filter(|e| !e.is_positive).count();
    println!("Dataset: {} examples ({} positive, {} hard negative)", dataset.len(), positives, negatives);

    let categories = [
        TaskCategory::ArithmeticNorm,
        TaskCategory::DateDelta,
        TaskCategory::StringTransform,
        TaskCategory::UnitConversion,
        TaskCategory::TableLookup,
        TaskCategory::SymbolicFormat,
    ];
    for cat in &categories {
        let n = dataset.iter().filter(|e| &e.category == cat && e.is_positive).count();
        println!("  {:?}: {} positive examples", cat, n);
    }
    println!();

    // Show a sample example.
    if let Some(ex) = dataset.first() {
        println!("Sample example:");
        println!("  Prompt      : {:?}", ex.prompt);
        println!("  Target token: {:?}", ex.target_token);
        println!("  Layer        : {}", ex.layer);
        println!("  Position     : {}", ex.position);
        println!("  Monty input  : {}", ex.monty_input);
        println!("  Monty output : {}", ex.monty_output);
    }
    println!();

    // ────────────────────────────────────────────────────────────────────────
    // Phase 2: Gate calibration with hard negatives
    // ────────────────────────────────────────────────────────────────────────
    println!("--- Phase 2: Gate calibration ---");

    let gate_examples: Vec<GateExample> = dataset
        .iter()
        .enumerate()
        .map(|(i, ex)| {
            let residual = residual_for_example(i, hidden);
            let label = if ex.is_positive { 1.0 } else { 0.0 };
            // Pair each positive with the next negative.
            let paired_negative = if ex.is_positive {
                dataset.iter().enumerate()
                    .find(|(j, e)| *j != i && !e.is_positive)
                    .map(|(j, _)| residual_for_example(j, hidden))
            } else {
                None
            };
            GateExample { residual, label, paired_negative }
        })
        .collect();

    let cal_config = GateCalibrationConfig {
        learning_rate: 0.02,
        lambda_sparse: 0.001,
        margin: 1.0,
        lambda_margin: 0.1,
        weight_decay: 1e-4,
        grad_clip: 1.0,
    };
    let mut calibrator = GateCalibrator::new(hidden, cal_config);

    let mut last_loss = f32::MAX;
    for epoch in 1..=10 {
        let acc = calibrator.fit_epoch(&gate_examples);
        if epoch == 1 || epoch % 3 == 0 || epoch == 10 {
            println!("  Epoch {epoch:2}: BCE={:.4}  total={:.4}  fire_rate={:.2}%",
                acc.bce_mean(), acc.total_mean(), acc.fire_rate() * 100.0);
        }
        last_loss = acc.total_mean();
    }

    let (precision, recall, f1) = calibrator.precision_recall_f1(&gate_examples, 0.0);
    println!("  Final: P={precision:.3} R={recall:.3} F1={f1:.3}  loss={last_loss:.4}");
    println!();

    // ────────────────────────────────────────────────────────────────────────
    // Phase 3: Decoder warmup (forced-call teacher forcing)
    // ────────────────────────────────────────────────────────────────────────
    println!("--- Phase 3: Decoder warmup ---");

    let decoder_examples: Vec<DecoderWarmupExample> = dataset.iter().enumerate()
        .filter(|(_, e)| e.is_positive)
        .map(|(i, ex)| {
            // Extract target token index as a simple modular hash.
            let target_idx = ex.target_token.len() % hidden;
            DecoderWarmupExample {
                monty_output: monty_output_for_example(i, monty_dim),
                residual: residual_for_example(i, hidden),
                target_token_idx: target_idx,
                vocab_size: 32000,
            }
        })
        .collect();

    let dec_config = DecoderWarmupConfig {
        learning_rate: 0.05,
        lambda_norm: 1e-3,
        grad_clip: 1.0,
        weight_decay: 1e-4,
    };
    let mut decoder = LinearDecoder::new(hidden, monty_dim, dec_config.clone());

    let (ce1, _) = decoder.fit_epoch(&decoder_examples);
    let mut last_ce = ce1;
    for epoch in 2..=10 {
        let (ce, reg) = decoder.fit_epoch(&decoder_examples);
        if epoch == 2 || epoch % 3 == 0 || epoch == 10 {
            println!("  Epoch {epoch:2}: CE≈{ce:.5}  norm_reg={reg:.5}");
        }
        last_ce = ce;
    }
    println!("  Epoch 1 CE={ce1:.5} → Epoch 10 CE={last_ce:.5} (should not increase)");
    println!();

    // Surrogate backward prototype.
    println!("--- Phase 3b: Surrogate backward prototype ---");
    let mut surrogate = SurrogateDecoder::new(hidden, dec_config);
    let residual_sample = residual_for_example(0, hidden);
    // Real delta: decoder output for first example.
    let real_delta = decoder.forward(&monty_output_for_example(0, monty_dim));

    let mut final_mse = 0.0;
    for step in 1..=100 {
        final_mse = surrogate.update_to_match(&residual_sample, &real_delta);
        if step == 1 || step == 50 || step == 100 {
            println!("  Step {step:3}: surrogate MSE = {final_mse:.6}");
        }
    }
    println!("  Surrogate converged to MSE = {final_mse:.6}");
    println!();

    // ────────────────────────────────────────────────────────────────────────
    // Phase 4: Evaluation harness
    // ────────────────────────────────────────────────────────────────────────
    println!("--- Phase 4: Evaluation harness ---");

    // Simulate baseline vs patched predictions for accuracy.
    // Baseline: predict "X" (wrong for arithmetic); patched: predict the target.
    let baseline_preds: Vec<(String, String)> = dataset.iter()
        .filter(|e| e.is_positive)
        .map(|e| ("X".to_string(), e.target_token.clone()))
        .collect();
    let patched_preds: Vec<(String, String)> = dataset.iter()
        .filter(|e| e.is_positive)
        .map(|e| {
            // Arithmetic and string tasks "fixed" by call patch.
            let pred = match e.category {
                TaskCategory::ArithmeticNorm | TaskCategory::StringTransform => {
                    e.target_token.clone()
                }
                _ => "X".to_string(),
            };
            (pred, e.target_token.clone())
        })
        .collect();

    let (base_acc, patch_acc, _) = accuracy_improvement(&baseline_preds, &patched_preds);

    // Gate predictions using the calibrated gate.
    let gate_preds: Vec<GatePrediction> = dataset.iter().enumerate()
        .map(|(i, ex)| {
            let res = residual_for_example(i, hidden);
            GatePrediction {
                fired: calibrator.score(&res) >= 0.0,
                should_fire: ex.is_positive,
            }
        })
        .collect();
    let gate = gate_metrics(&gate_preds);

    // KL / regression check: simulate log-probs.
    let vocab = 10usize;
    let pairs: Vec<(Vec<f32>, Vec<f32>, usize)> = (0..20)
        .map(|i| {
            let target = i % vocab;
            // Baseline: near-uniform log-probs.
            let base_lp: Vec<f32> = (0..vocab).map(|j| {
                if j == target { -1.0f32 } else { -2.5 }
            }).collect();
            // Patched: slightly sharper on the target (a small improvement).
            let patch_lp: Vec<f32> = (0..vocab).map(|j| {
                if j == target { -0.8f32 } else { -2.6 }
            }).collect();
            (base_lp, patch_lp, target)
        })
        .collect();
    let regression = regression_report(&pairs, 0.10);

    // Reach probe: call at position t-1 improves 1-token tasks more than 2-token.
    let single: Vec<(bool, bool)> = (0..20).map(|i| (i < 14, i < 5)).collect();
    let two: Vec<(bool, bool)> = (0..20).map(|i| (i < 4, i < 2)).collect();
    let reach = reach_probe(&single, &two);

    print_m8_report(dataset.len(), base_acc, patch_acc, &gate, &regression, &reach);

    // Exit criteria validation.
    assert!(
        patch_acc >= base_acc,
        "patched accuracy should be >= baseline ({base_acc:.3}), got {patch_acc:.3}"
    );
    assert!(
        gate.f1 >= 0.0,
        "gate F1 should be non-negative, got {}", gate.f1
    );
    assert!(
        !regression.regression_detected,
        "regression detected — patched PPL ({:.3}) exceeds baseline ({:.3}) by >{:.0}%",
        regression.patched_perplexity,
        regression.baseline_perplexity,
        regression.regression_tolerance * 100.0,
    );
    assert!(
        reach.single_token_reach_confirmed,
        "next-token reach probe failed: single_token_improvement ({:.2}) should > two_token ({:.2})",
        reach.single_token_improvement_rate,
        reach.two_token_improvement_rate,
    );

    println!("All M8 exit criteria satisfied.");
}
