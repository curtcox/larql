//! Evaluation harness for the call-patch training prototype (Phase 5.6).
//!
//! Covers M8 exit criteria:
//! - Synthetic exact-task accuracy
//! - Gate precision / recall / F1 at (layer, position)
//! - No-regression KL/perplexity check (mock logit distributions)
//! - Next-token reach probe: confirms one call only improves the immediate
//!   next-token distribution, not multi-token spans without recurrence.
//!
//! All computations are pure Rust — no model weights required.  Real
//! evaluations would pipe actual logits through these functions.

use serde::{Deserialize, Serialize};

// ── Types ─────────────────────────────────────────────────────────────────────

/// Prediction outcome for a single synthetic example.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Correct,
    Incorrect,
}

/// Gate prediction outcome at one (layer, position).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatePrediction {
    pub fired: bool,
    pub should_fire: bool,
}

impl GatePrediction {
    pub fn is_tp(&self) -> bool {
        self.fired && self.should_fire
    }
    pub fn is_fp(&self) -> bool {
        self.fired && !self.should_fire
    }
    pub fn is_fn(&self) -> bool {
        !self.fired && self.should_fire
    }
    pub fn is_tn(&self) -> bool {
        !self.fired && !self.should_fire
    }
}

/// Summary of gate precision/recall/F1 over a dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateMetrics {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub tp: u64,
    pub fp: u64,
    pub fn_: u64,
    pub tn: u64,
    pub fire_rate: f32,
}

/// Perplexity / KL regression report.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    /// KL(p_with_call || p_base) over a sample of positions.
    pub mean_kl: f32,
    /// Mean perplexity without call patches.
    pub baseline_perplexity: f32,
    /// Mean perplexity with call patches.
    pub patched_perplexity: f32,
    /// True if patched perplexity increased above the allowed tolerance.
    pub regression_detected: bool,
    pub regression_tolerance: f32,
}

/// Next-token reach probe result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReachProbeResult {
    /// Improvement fraction for one-token tasks (call at token t-1).
    pub single_token_improvement_rate: f32,
    /// Improvement fraction for two-token tasks (call at token t-1).
    pub two_token_improvement_rate: f32,
    /// Confirms the design invariant: single > two for most tasks.
    pub single_token_reach_confirmed: bool,
}

// ── Synthetic accuracy ────────────────────────────────────────────────────────

/// Evaluate task accuracy: for each example, compare the model's top-1
/// prediction against the target token.
pub fn synthetic_accuracy(predictions: &[(String, String)]) -> f32 {
    if predictions.is_empty() {
        return 0.0;
    }
    let correct = predictions
        .iter()
        .filter(|(pred, target)| pred == target)
        .count();
    correct as f32 / predictions.len() as f32
}

/// Measure accuracy improvement from applying call-patch outputs.
///
/// Returns `(baseline_accuracy, patched_accuracy, delta)`.
pub fn accuracy_improvement(
    baseline: &[(String, String)],
    patched: &[(String, String)],
) -> (f32, f32, f32) {
    let base_acc = synthetic_accuracy(baseline);
    let patch_acc = synthetic_accuracy(patched);
    (base_acc, patch_acc, patch_acc - base_acc)
}

// ── Gate metrics ──────────────────────────────────────────────────────────────

/// Aggregate gate prediction outcomes into precision/recall/F1.
pub fn gate_metrics(predictions: &[GatePrediction]) -> GateMetrics {
    let (mut tp, mut fp, mut fn_, mut tn) = (0u64, 0u64, 0u64, 0u64);
    for p in predictions {
        match (p.fired, p.should_fire) {
            (true, true) => tp += 1,
            (true, false) => fp += 1,
            (false, true) => fn_ += 1,
            (false, false) => tn += 1,
        }
    }
    let total = predictions.len() as u64;
    let fired = tp + fp;

    let precision = if fired == 0 {
        0.0
    } else {
        tp as f32 / fired as f32
    };
    let recall = if tp + fn_ == 0 {
        0.0
    } else {
        tp as f32 / (tp + fn_) as f32
    };
    let f1 = if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    };
    let fire_rate = if total == 0 {
        0.0
    } else {
        fired as f32 / total as f32
    };

    GateMetrics {
        precision,
        recall,
        f1,
        tp,
        fp,
        fn_,
        tn,
        fire_rate,
    }
}

// ── KL / perplexity regression check ─────────────────────────────────────────

/// KL divergence D_KL(p || q) where p and q are log-probability vectors.
///
/// Both `log_p` and `log_q` must be normalised (log-softmax output).
pub fn kl_divergence(log_p: &[f32], log_q: &[f32]) -> f32 {
    log_p
        .iter()
        .zip(log_q.iter())
        .map(|(lp, lq)| {
            let p = lp.exp();
            if p < 1e-9 {
                0.0
            } else {
                p * (lp - lq)
            }
        })
        .sum()
}

/// Cross-entropy: -sum(p * log_q).  `log_q` should be log-softmax of model
/// output; `p` is the target distribution (one-hot: target_idx).
pub fn cross_entropy_one_hot(log_q: &[f32], target_idx: usize) -> f32 {
    -log_q[target_idx]
}

/// Perplexity for a sequence: exp(mean CE).
pub fn perplexity(log_q_per_position: &[f32], targets: &[usize]) -> f32 {
    if targets.is_empty() {
        return 1.0;
    }
    let mean_ce: f32 = log_q_per_position
        .iter()
        .zip(targets.iter())
        .map(|(log_q, &t)| {
            -log_q.min(0.0)
                + if *log_q_per_position.get(t).unwrap_or(log_q) < 0.0 {
                    0.0
                } else {
                    0.0
                }
        })
        .sum::<f32>()
        / targets.len() as f32;
    mean_ce.exp()
}

/// Build a regression report from baseline and patched log-probability samples.
///
/// `pairs` is a slice of `(baseline_log_probs, patched_log_probs, target_idx)`.
pub fn regression_report(
    pairs: &[(Vec<f32>, Vec<f32>, usize)],
    tolerance: f32,
) -> RegressionReport {
    if pairs.is_empty() {
        return RegressionReport {
            mean_kl: 0.0,
            baseline_perplexity: 1.0,
            patched_perplexity: 1.0,
            regression_detected: false,
            regression_tolerance: tolerance,
        };
    }

    let mut kl_sum = 0.0f32;
    let mut base_ce_sum = 0.0f32;
    let mut patch_ce_sum = 0.0f32;

    for (base_lp, patch_lp, target) in pairs {
        kl_sum += kl_divergence(patch_lp, base_lp);
        base_ce_sum += cross_entropy_one_hot(base_lp, *target);
        patch_ce_sum += cross_entropy_one_hot(patch_lp, *target);
    }

    let n = pairs.len() as f32;
    let mean_kl = kl_sum / n;
    let base_ppl = (base_ce_sum / n).exp();
    let patch_ppl = (patch_ce_sum / n).exp();
    let regression_detected = patch_ppl > base_ppl * (1.0 + tolerance);

    RegressionReport {
        mean_kl,
        baseline_perplexity: base_ppl,
        patched_perplexity: patch_ppl,
        regression_detected,
        regression_tolerance: tolerance,
    }
}

// ── Next-token reach probe ────────────────────────────────────────────────────

/// Probe whether a call at position t-1 improves the next token vs. the token
/// two steps later.
///
/// `single_token_pairs`: (patched_correct, baseline_correct) for 1-token tasks.
/// `two_token_pairs`: same for 2-token tasks (second generated token).
pub fn reach_probe(
    single_token_pairs: &[(bool, bool)],
    two_token_pairs: &[(bool, bool)],
) -> ReachProbeResult {
    let improve = |pairs: &[(bool, bool)]| -> f32 {
        if pairs.is_empty() {
            return 0.0;
        }
        pairs.iter().filter(|(p, b)| *p && !b).count() as f32 / pairs.len() as f32
    };

    let single = improve(single_token_pairs);
    let two = improve(two_token_pairs);

    ReachProbeResult {
        single_token_improvement_rate: single,
        two_token_improvement_rate: two,
        // Confirmed if single-token improvement > two-token improvement.
        single_token_reach_confirmed: single > two,
    }
}

// ── Reporting ─────────────────────────────────────────────────────────────────

/// Print an M8 evaluation summary to stdout.
pub fn print_m8_report(
    dataset_size: usize,
    base_acc: f32,
    patch_acc: f32,
    gate: &GateMetrics,
    regression: &RegressionReport,
    reach: &ReachProbeResult,
) {
    println!("=== M8 Training Prototype Evaluation ===");
    println!("Dataset size: {dataset_size}");
    println!();
    println!("Synthetic task accuracy:");
    println!("  Baseline : {:.1}%", base_acc * 100.0);
    println!("  Patched  : {:.1}%", patch_acc * 100.0);
    println!("  Δ        : {:+.1}%", (patch_acc - base_acc) * 100.0);
    println!();
    println!("Gate calibration:");
    println!("  Precision: {:.3}", gate.precision);
    println!("  Recall   : {:.3}", gate.recall);
    println!("  F1       : {:.3}", gate.f1);
    println!("  Fire rate: {:.2}%", gate.fire_rate * 100.0);
    println!(
        "  TP={} FP={} FN={} TN={}",
        gate.tp, gate.fp, gate.fn_, gate.tn
    );
    println!();
    println!(
        "No-regression check (tolerance={:.0}%):",
        regression.regression_tolerance * 100.0
    );
    println!("  Baseline PPL : {:.3}", regression.baseline_perplexity);
    println!("  Patched PPL  : {:.3}", regression.patched_perplexity);
    println!("  Mean KL      : {:.4}", regression.mean_kl);
    println!(
        "  Regression   : {}",
        if regression.regression_detected {
            "DETECTED"
        } else {
            "none"
        }
    );
    println!();
    println!("Next-token reach probe:");
    println!(
        "  Single-token improvement rate: {:.1}%",
        reach.single_token_improvement_rate * 100.0
    );
    println!(
        "  Two-token improvement rate   : {:.1}%",
        reach.two_token_improvement_rate * 100.0
    );
    println!(
        "  Single-token reach confirmed : {}",
        reach.single_token_reach_confirmed
    );
    println!();
    println!("=== End of report ===");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synthetic_accuracy_perfect() {
        let preds: Vec<(String, String)> = vec![("a".into(), "a".into()), ("b".into(), "b".into())];
        assert_eq!(synthetic_accuracy(&preds), 1.0);
    }

    #[test]
    fn synthetic_accuracy_half() {
        let preds: Vec<(String, String)> = vec![("a".into(), "a".into()), ("x".into(), "b".into())];
        assert_eq!(synthetic_accuracy(&preds), 0.5);
    }

    #[test]
    fn gate_metrics_all_correct() {
        let preds: Vec<GatePrediction> = vec![
            GatePrediction {
                fired: true,
                should_fire: true,
            },
            GatePrediction {
                fired: false,
                should_fire: false,
            },
        ];
        let m = gate_metrics(&preds);
        assert_eq!(m.precision, 1.0);
        assert_eq!(m.recall, 1.0);
        assert_eq!(m.f1, 1.0);
    }

    #[test]
    fn gate_metrics_all_false_positives() {
        let preds: Vec<GatePrediction> = vec![
            GatePrediction {
                fired: true,
                should_fire: false,
            },
            GatePrediction {
                fired: true,
                should_fire: false,
            },
        ];
        let m = gate_metrics(&preds);
        assert_eq!(m.precision, 0.0);
        assert_eq!(m.recall, 0.0);
        assert_eq!(m.f1, 0.0);
    }

    #[test]
    fn kl_divergence_identical_distributions_is_zero() {
        let log_p: Vec<f32> = vec![-1.0, -2.0, -3.0];
        let kl = kl_divergence(&log_p, &log_p);
        assert!(
            kl.abs() < 1e-5,
            "KL of identical distributions should be 0, got {kl}"
        );
    }

    #[test]
    fn regression_report_detects_large_increase() {
        let vocab = 4;
        // Baseline: uniform; patched: concentrate mass on wrong token.
        let base_lp: Vec<f32> = vec![-1.386; vocab]; // log(0.25)
        let bad_lp: Vec<f32> = vec![-0.1, -3.0, -3.0, -3.0]; // mass on token 0
                                                             // Target is token 1 (not token 0), so patched has higher CE.
        let pairs = vec![(base_lp.clone(), bad_lp.clone(), 1usize)];
        let rep = regression_report(&pairs, 0.05);
        assert!(
            rep.regression_detected,
            "should detect regression when patched CE is higher"
        );
    }

    #[test]
    fn reach_probe_confirms_single_token_dominance() {
        // Call helps 80% of 1-token tasks but only 20% of 2-token tasks.
        let single: Vec<(bool, bool)> = (0..10).map(|i| (i < 8, false)).collect();
        let two: Vec<(bool, bool)> = (0..10).map(|i| (i < 2, false)).collect();
        let probe = reach_probe(&single, &two);
        assert!(probe.single_token_reach_confirmed);
        assert!((probe.single_token_improvement_rate - 0.8).abs() < 1e-5);
        assert!((probe.two_token_improvement_rate - 0.2).abs() < 1e-5);
    }

    #[test]
    fn reach_probe_no_improvement() {
        let single: Vec<(bool, bool)> = vec![(false, true), (false, true)];
        let two: Vec<(bool, bool)> = vec![(false, true)];
        let probe = reach_probe(&single, &two);
        assert_eq!(probe.single_token_improvement_rate, 0.0);
        assert_eq!(probe.two_token_improvement_rate, 0.0);
        // single == two == 0; confirmed = false (not strict >)
        assert!(!probe.single_token_reach_confirmed);
    }
}
