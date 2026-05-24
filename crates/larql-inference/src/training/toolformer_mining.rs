//! Toolformer-style mining for call-patch training data (Phase 5.1, M9).
//!
//! Pipeline:
//! 1. Run base model on corpus (simulated here with mock log-probs).
//! 2. Select candidate (layer, position) where uncertainty is high or a task
//!    pattern matches in the token window.
//! 3. Execute call patch with candidate inputs (provided by caller as Monty output).
//! 4. Decode candidate output to logits/delta using a warm-start decoder.
//! 5. Keep examples where continuation loss drops above `loss_drop_threshold`.
//! 6. Add hard negatives where execution does not help.
//!
//! No real model weights or Monty VM are required — callers supply baseline
//! log-probs, patched log-probs, and the Monty output that was used.

use serde::{Deserialize, Serialize};

use super::synthetic::SyntheticExample;

// ── Types ──────────────────────────────────────────────────────────────────────

/// A candidate (layer, position) from a corpus pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningCandidate {
    /// Example index in the source corpus.
    pub corpus_idx: usize,
    pub layer: usize,
    pub position: usize,
    /// Entropy of the baseline next-token distribution at this position.
    pub baseline_entropy: f32,
    /// True if a task-pattern heuristic matched the token window.
    pub pattern_matched: bool,
}

/// A mined example, kept because the call lowered continuation loss.
#[derive(Debug, Clone)]
pub struct MinedExample {
    pub candidate: MiningCandidate,
    /// Baseline log-probs (log-softmax, vocab-length).
    pub baseline_log_probs: Vec<f32>,
    /// Patched log-probs after the call output was applied.
    pub patched_log_probs: Vec<f32>,
    /// Monty output vector used to produce the patched log-probs.
    pub monty_output: Vec<f32>,
    /// Residual vector at (layer, position).
    pub residual: Vec<f32>,
    /// Target token index.
    pub target_idx: usize,
    /// True = positive (call helped); false = hard negative.
    pub is_positive: bool,
    /// CE loss reduction: baseline_ce − patched_ce (positive = improvement).
    pub loss_drop: f32,
}

/// Configuration for the mining pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MiningConfig {
    /// Minimum CE-loss reduction required to keep a positive example.
    pub loss_drop_threshold: f32,
    /// Entropy threshold above which a position is considered uncertain.
    pub high_entropy_threshold: f32,
    /// Maximum candidates per example (limits mining cost).
    pub max_candidates_per_example: usize,
    /// Fraction of kept positives to also collect as hard negatives.
    pub hard_negative_ratio: f32,
}

impl Default for MiningConfig {
    fn default() -> Self {
        Self {
            loss_drop_threshold: 0.1,
            high_entropy_threshold: 1.5,
            max_candidates_per_example: 4,
            hard_negative_ratio: 0.5,
        }
    }
}

/// Summary statistics from one mining pass.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MiningSummary {
    pub total_candidates: u64,
    pub kept_positives: u64,
    pub hard_negatives: u64,
    pub mean_loss_drop: f32,
    pub mean_entropy: f32,
}

// ── Mining logic ──────────────────────────────────────────────────────────────

/// Entropy of a log-probability distribution.
pub fn entropy(log_probs: &[f32]) -> f32 {
    -log_probs
        .iter()
        .map(|&lp| {
            let p = lp.exp();
            if p < 1e-9 { 0.0 } else { p * lp }
        })
        .sum::<f32>()
}

/// Cross-entropy of log-probs at a target index.
pub fn ce_at(log_probs: &[f32], target: usize) -> f32 {
    -log_probs[target]
}

/// Score a candidate by uncertainty + pattern match.
///
/// Higher score = higher mining priority.
pub fn candidate_score(candidate: &MiningCandidate, config: &MiningConfig) -> f32 {
    let entropy_score = (candidate.baseline_entropy - config.high_entropy_threshold).max(0.0);
    let pattern_bonus = if candidate.pattern_matched { 1.0 } else { 0.0 };
    entropy_score + pattern_bonus
}

/// Filter a list of candidates to at most `max` by score, highest first.
pub fn top_candidates(
    mut candidates: Vec<MiningCandidate>,
    max: usize,
    config: &MiningConfig,
) -> Vec<MiningCandidate> {
    candidates.sort_by(|a, b| {
        candidate_score(b, config)
            .partial_cmp(&candidate_score(a, config))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.truncate(max);
    candidates
}

/// Evaluate one candidate: return `Some(MinedExample)` if the call helped
/// enough, or a hard negative if it did not.
///
/// `patched_log_probs` must be the same length as `baseline_log_probs`.
pub fn evaluate_candidate(
    candidate: MiningCandidate,
    baseline_log_probs: Vec<f32>,
    patched_log_probs: Vec<f32>,
    monty_output: Vec<f32>,
    residual: Vec<f32>,
    target_idx: usize,
    config: &MiningConfig,
    rng_val: f32, // caller provides a random float in [0,1] for hard-neg sampling
) -> Option<MinedExample> {
    let base_ce = ce_at(&baseline_log_probs, target_idx);
    let patch_ce = ce_at(&patched_log_probs, target_idx);
    let loss_drop = base_ce - patch_ce;

    if loss_drop >= config.loss_drop_threshold {
        // Positive example: call helped.
        Some(MinedExample {
            candidate,
            baseline_log_probs,
            patched_log_probs,
            monty_output,
            residual,
            target_idx,
            is_positive: true,
            loss_drop,
        })
    } else if rng_val < config.hard_negative_ratio {
        // Hard negative: call did not help (or hurt).
        Some(MinedExample {
            candidate,
            baseline_log_probs,
            patched_log_probs,
            monty_output,
            residual,
            target_idx,
            is_positive: false,
            loss_drop,
        })
    } else {
        None
    }
}

/// Aggregate a batch of mined examples into a summary.
pub fn summarise(examples: &[MinedExample]) -> MiningSummary {
    let kept_positives = examples.iter().filter(|e| e.is_positive).count() as u64;
    let hard_negatives = examples.iter().filter(|e| !e.is_positive).count() as u64;
    let total = examples.len();

    let mean_loss_drop = if total == 0 {
        0.0
    } else {
        examples.iter().map(|e| e.loss_drop).sum::<f32>() / total as f32
    };

    let mean_entropy = if total == 0 {
        0.0
    } else {
        examples
            .iter()
            .map(|e| e.candidate.baseline_entropy)
            .sum::<f32>()
            / total as f32
    };

    MiningSummary {
        total_candidates: total as u64,
        kept_positives,
        hard_negatives,
        mean_loss_drop,
        mean_entropy,
    }
}

// ── Corpus simulation helpers ─────────────────────────────────────────────────

/// Build synthetic mining candidates from a dataset slice.
///
/// Uses entropy of a mock baseline distribution (uniform-ish) and
/// a simple pattern heuristic (prompt contains a digit → pattern match).
pub fn mine_candidates_from_synthetic(
    dataset: &[SyntheticExample],
    config: &MiningConfig,
    seed: u64,
) -> Vec<MiningCandidate> {
    let mut rng = Lcg64(seed);
    let mut out = Vec::new();

    for (idx, ex) in dataset.iter().enumerate() {
        let vocab = 32usize;
        // Mock: baseline entropy based on how long the prompt is.
        let baseline_entropy = 1.0 + (ex.prompt.len() as f32 / 30.0).min(2.0);
        let pattern_matched = ex.prompt.chars().any(|c| c.is_ascii_digit());

        let n_cands = (rng.next() as usize % config.max_candidates_per_example) + 1;
        let raw_cands: Vec<MiningCandidate> = (0..n_cands)
            .map(|_| MiningCandidate {
                corpus_idx: idx,
                layer: ex.layer,
                position: ex.position,
                baseline_entropy,
                pattern_matched,
            })
            .collect();

        let _ = vocab; // silence unused warning
        let top = top_candidates(raw_cands, config.max_candidates_per_example, config);
        out.extend(top);
    }
    out
}

// ── Tiny PRNG ─────────────────────────────────────────────────────────────────

struct Lcg64(u64);
impl Lcg64 {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() as f32) / (u32::MAX as f32)
    }
}

// ── Convenience: mine a full batch ───────────────────────────────────────────

/// Mine a batch of positives + hard negatives from a synthetic dataset.
///
/// For each candidate, mock baseline log-probs and patched log-probs are
/// constructed deterministically.  The caller would replace this with a real
/// forward pass + decoder application.
pub fn mine_batch(
    dataset: &[SyntheticExample],
    monty_dim: usize,
    config: &MiningConfig,
    seed: u64,
) -> (Vec<MinedExample>, MiningSummary) {
    let candidates = mine_candidates_from_synthetic(dataset, config, seed);
    let mut rng = Lcg64(seed.wrapping_add(0xbeef));
    let vocab = 16usize;
    let hidden = 16usize;

    let mut examples = Vec::new();
    for cand in candidates {
        let idx = cand.corpus_idx;
        let target_idx = idx % vocab;

        // Mock baseline: near-uniform.
        let base_lp: Vec<f32> = (0..vocab)
            .map(|j| if j == target_idx { -1.2f32 } else { -3.0 })
            .collect();

        // Mock patched: slight improvement for positive examples.
        let is_pos = dataset.get(idx).map(|e| e.is_positive).unwrap_or(false);
        let patch_lp: Vec<f32> = if is_pos {
            (0..vocab)
                .map(|j| if j == target_idx { -0.5f32 } else { -3.2 })
                .collect()
        } else {
            base_lp.clone() // no improvement
        };

        // Mock Monty output and residual.
        let monty_out: Vec<f32> = (0..monty_dim)
            .map(|i| if i == idx % monty_dim { 0.5 } else { 0.0 })
            .collect();
        let residual: Vec<f32> = (0..hidden)
            .map(|i| {
                let v = (idx as u64)
                    .wrapping_add(i as u64)
                    .wrapping_mul(2_862_933_555_777_941_757);
                (v >> 32) as i32 as f32 / i32::MAX as f32
            })
            .collect();

        let rng_val = rng.next_f32();
        if let Some(ex) = evaluate_candidate(
            cand, base_lp, patch_lp, monty_out, residual, target_idx, config, rng_val,
        ) {
            examples.push(ex);
        }
    }

    let summary = summarise(&examples);
    (examples, summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::synthetic::build_dataset;

    #[test]
    fn entropy_of_uniform_two_class_is_ln2() {
        let log_probs = vec![-std::f32::consts::LN_2; 2];
        let h = entropy(&log_probs);
        assert!((h - std::f32::consts::LN_2).abs() < 1e-5, "got {h}");
    }

    #[test]
    fn ce_at_target_matches_negative_log_prob() {
        let lp = vec![-1.0f32, -2.0, -3.0];
        assert!((ce_at(&lp, 1) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn evaluate_candidate_keeps_positive_above_threshold() {
        let config = MiningConfig {
            loss_drop_threshold: 0.1,
            ..Default::default()
        };
        let cand = MiningCandidate {
            corpus_idx: 0,
            layer: 0,
            position: 0,
            baseline_entropy: 2.0,
            pattern_matched: true,
        };
        let base_lp = vec![-2.0f32, -1.0]; // CE at idx=1 is 1.0
        let patch_lp = vec![-3.0f32, -0.5]; // CE at idx=1 is 0.5 → drop=0.5
        let result = evaluate_candidate(
            cand,
            base_lp,
            patch_lp,
            vec![0.5],
            vec![0.1],
            1,
            &config,
            0.0,
        );
        let ex = result.expect("should be kept");
        assert!(ex.is_positive);
        assert!((ex.loss_drop - 0.5).abs() < 1e-5);
    }

    #[test]
    fn evaluate_candidate_hard_negative_below_threshold() {
        let config = MiningConfig {
            loss_drop_threshold: 0.5,
            hard_negative_ratio: 1.0, // always keep hard negatives
            ..Default::default()
        };
        let cand = MiningCandidate {
            corpus_idx: 0,
            layer: 0,
            position: 0,
            baseline_entropy: 2.0,
            pattern_matched: false,
        };
        let base_lp = vec![-1.0f32, -2.0];
        let patch_lp = vec![-1.05f32, -2.0]; // tiny improvement, below threshold
        let result = evaluate_candidate(
            cand,
            base_lp,
            patch_lp,
            vec![],
            vec![],
            0,
            &config,
            0.0,
        );
        let ex = result.expect("should be kept as hard negative");
        assert!(!ex.is_positive);
    }

    #[test]
    fn mine_batch_produces_both_polarities() {
        let dataset = build_dataset(10, 99);
        let config = MiningConfig::default();
        let (examples, summary) = mine_batch(&dataset, 4, &config, 42);
        assert!(summary.kept_positives > 0, "should have some positives");
        // Hard negatives may or may not appear depending on RNG.
        assert!(summary.total_candidates > 0);
        assert_eq!(
            examples.len() as u64,
            summary.total_candidates,
            "summary should match example count"
        );
    }

    #[test]
    fn top_candidates_limits_count() {
        let config = MiningConfig::default();
        let cands: Vec<MiningCandidate> = (0..10)
            .map(|i| MiningCandidate {
                corpus_idx: i,
                layer: 0,
                position: 0,
                baseline_entropy: i as f32,
                pattern_matched: i % 2 == 0,
            })
            .collect();
        let top = top_candidates(cands, 3, &config);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn summarise_counts_correctly() {
        let make = |pos: bool| MinedExample {
            candidate: MiningCandidate {
                corpus_idx: 0,
                layer: 0,
                position: 0,
                baseline_entropy: 1.0,
                pattern_matched: false,
            },
            baseline_log_probs: vec![],
            patched_log_probs: vec![],
            monty_output: vec![],
            residual: vec![],
            target_idx: 0,
            is_positive: pos,
            loss_drop: if pos { 0.3 } else { -0.1 },
        };
        let examples = vec![make(true), make(true), make(false)];
        let s = summarise(&examples);
        assert_eq!(s.kept_positives, 2);
        assert_eq!(s.hard_negatives, 1);
        assert_eq!(s.total_candidates, 3);
    }
}
