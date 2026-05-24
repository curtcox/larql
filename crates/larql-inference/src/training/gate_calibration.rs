//! Gate calibration with hard negatives (Phase 5.4).
//!
//! Trains a gate direction vector `g ∈ R^hidden` as a retrieval classifier:
//!
//! ```text
//! score = g · residual
//! fire  = score ≥ threshold  AND  margin ≥ margin_threshold
//! ```
//!
//! Loss (one gradient step per call to `update`):
//!
//! ```text
//! L = BCE(σ(score), label)
//!   + λ_sparse * expected_fire_rate        (sparsity)
//!   + λ_margin * max(0, m - pos + neg)    (margin, if pair provided)
//! ```
//!
//! No external ML framework is needed — only dot products and scalar ops.

use serde::{Deserialize, Serialize};

/// Training example for gate calibration.
#[derive(Debug, Clone)]
pub struct GateExample {
    /// Residual vector at the candidate (layer, position).
    pub residual: Vec<f32>,
    /// 1.0 = call should fire (positive), 0.0 = hard negative.
    pub label: f32,
    /// Optional paired negative residual for margin loss.
    pub paired_negative: Option<Vec<f32>>,
}

/// Hyper-parameters for gate calibration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateCalibrationConfig {
    pub learning_rate: f32,
    /// Weight of fire-rate sparsity penalty.
    pub lambda_sparse: f32,
    /// Margin for positive/negative separation.
    pub margin: f32,
    /// Weight of margin loss.
    pub lambda_margin: f32,
    /// L2 weight decay on gate vector.
    pub weight_decay: f32,
    /// Clip gradient norm to this value (0 = no clip).
    pub grad_clip: f32,
}

impl Default for GateCalibrationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            lambda_sparse: 0.001,
            margin: 1.0,
            lambda_margin: 0.1,
            weight_decay: 1e-4,
            grad_clip: 1.0,
        }
    }
}

/// Running loss accumulator for one calibration pass.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GateLossAccumulator {
    pub bce_sum: f64,
    pub margin_sum: f64,
    pub sparsity_sum: f64,
    pub total_sum: f64,
    pub n_steps: u64,
    pub n_fired: u64,
    pub n_total: u64,
}

impl GateLossAccumulator {
    pub fn bce_mean(&self) -> f32 {
        if self.n_steps == 0 { 0.0 } else { (self.bce_sum / self.n_steps as f64) as f32 }
    }
    pub fn total_mean(&self) -> f32 {
        if self.n_steps == 0 { 0.0 } else { (self.total_sum / self.n_steps as f64) as f32 }
    }
    pub fn fire_rate(&self) -> f32 {
        if self.n_total == 0 { 0.0 } else { self.n_fired as f32 / self.n_total as f32 }
    }
}

/// Gate calibration state — holds the gate vector and optimizer state.
#[derive(Debug, Clone)]
pub struct GateCalibrator {
    pub gate: Vec<f32>,
    pub config: GateCalibrationConfig,
    /// AdaGrad-style accumulated squared gradients.
    sq_grad: Vec<f32>,
}

impl GateCalibrator {
    /// Create a new calibrator with a zero-initialised gate of `hidden_size`.
    pub fn new(hidden_size: usize, config: GateCalibrationConfig) -> Self {
        Self {
            gate: vec![0.0; hidden_size],
            sq_grad: vec![1e-8; hidden_size],
            config,
        }
    }

    /// Create from an existing gate vector (e.g., warm-start from a prior patch).
    pub fn from_gate(gate: Vec<f32>, config: GateCalibrationConfig) -> Self {
        let n = gate.len();
        Self { gate, sq_grad: vec![1e-8; n], config }
    }

    /// Score a residual: dot product with the gate.
    pub fn score(&self, residual: &[f32]) -> f32 {
        dot(&self.gate, residual)
    }

    /// Sigmoid score (firing probability).
    pub fn prob(&self, residual: &[f32]) -> f32 {
        sigmoid(self.score(residual))
    }

    /// One gradient step on a single example.
    ///
    /// Returns the scalar losses `(bce, margin, sparsity, total)`.
    pub fn update(&mut self, example: &GateExample) -> (f32, f32, f32, f32) {
        let score = dot(&self.gate, &example.residual);
        let p = sigmoid(score);
        let y = example.label;

        // Binary cross-entropy gradient w.r.t. score: (p - y)
        let bce_grad = p - y;
        let bce_loss = bce(p, y);

        // Sparsity gradient: λ_sparse * p  (pushes score towards −∞)
        let sparse_loss = self.config.lambda_sparse * p;
        let sparse_grad = self.config.lambda_sparse * p * (1.0 - p);

        // Margin loss (only if paired negative provided and example is positive)
        let (margin_loss, margin_grad_pos, margin_grad_neg) =
            if let (Some(neg), true) = (example.paired_negative.as_deref(), y > 0.5) {
                let neg_score = dot(&self.gate, neg);
                let slack = self.config.margin - score + neg_score;
                if slack > 0.0 {
                    let loss = self.config.lambda_margin * slack;
                    // ∂/∂g: -λ * residual_pos + λ * residual_neg
                    (loss, -self.config.lambda_margin, self.config.lambda_margin)
                } else {
                    (0.0, 0.0, 0.0)
                }
            } else {
                (0.0, 0.0, 0.0)
            };

        // Aggregate gradient w.r.t. gate
        let total_loss = bce_loss + sparse_loss + margin_loss;
        let n = self.gate.len();
        let mut grad = vec![0.0f32; n];
        let residual = &example.residual;
        let lr = self.config.learning_rate;

        for i in 0..n {
            let g_bce = bce_grad * residual[i];
            let g_sparse = sparse_grad * residual[i];
            let g_margin = if y > 0.5 {
                margin_grad_pos * residual[i]
                    + margin_grad_neg
                        * example
                            .paired_negative
                            .as_deref()
                            .map(|neg| neg[i])
                            .unwrap_or(0.0)
            } else {
                0.0
            };
            grad[i] = g_bce + g_sparse + g_margin + self.config.weight_decay * self.gate[i];
        }

        // Gradient clipping
        if self.config.grad_clip > 0.0 {
            let norm = grad.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > self.config.grad_clip {
                let scale = self.config.grad_clip / norm;
                for v in &mut grad {
                    *v *= scale;
                }
            }
        }

        // AdaGrad update
        for i in 0..n {
            self.sq_grad[i] += grad[i] * grad[i];
            self.gate[i] -= lr * grad[i] / (self.sq_grad[i].sqrt() + 1e-8);
        }

        (bce_loss, margin_loss, sparse_loss, total_loss)
    }

    /// Run one epoch over a batch of examples; returns accumulated loss.
    pub fn fit_epoch(&mut self, examples: &[GateExample]) -> GateLossAccumulator {
        let mut acc = GateLossAccumulator::default();
        for ex in examples {
            let (bce, margin, sparse, total) = self.update(ex);
            acc.bce_sum += bce as f64;
            acc.margin_sum += margin as f64;
            acc.sparsity_sum += sparse as f64;
            acc.total_sum += total as f64;
            acc.n_steps += 1;
            acc.n_total += 1;
            if self.score(&ex.residual) >= 0.0 {
                acc.n_fired += 1;
            }
        }
        acc
    }

    /// Evaluate precision, recall, and F1 at a given score threshold.
    pub fn precision_recall_f1(
        &self,
        examples: &[GateExample],
        threshold: f32,
    ) -> (f32, f32, f32) {
        let mut tp = 0u32;
        let mut fp = 0u32;
        let mut fn_ = 0u32;

        for ex in examples {
            let fires = self.score(&ex.residual) >= threshold;
            let positive = ex.label > 0.5;
            match (fires, positive) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => {}
            }
        }

        let precision = if tp + fp == 0 { 0.0 } else { tp as f32 / (tp + fp) as f32 };
        let recall = if tp + fn_ == 0 { 0.0 } else { tp as f32 / (tp + fn_) as f32 };
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        (precision, recall, f1)
    }
}

// ── Math helpers ─────────────────────────────────────────────────────────────

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Numerically-stable binary cross-entropy.
fn bce(p: f32, y: f32) -> f32 {
    let p = p.clamp(1e-7, 1.0 - 1e-7);
    -(y * p.ln() + (1.0 - y) * (1.0 - p).ln())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_example(residual: Vec<f32>, label: f32) -> GateExample {
        GateExample { residual, label, paired_negative: None }
    }

    #[test]
    fn gate_score_is_dot_product() {
        let mut cal = GateCalibrator::new(3, GateCalibrationConfig::default());
        cal.gate = vec![1.0, 2.0, 3.0];
        assert!((cal.score(&[1.0, 0.0, 1.0]) - 4.0).abs() < 1e-6);
    }

    #[test]
    fn bce_loss_is_zero_for_perfect_prediction() {
        let p_close_to_one = 1.0 - 1e-6;
        let loss = bce(p_close_to_one, 1.0);
        assert!(loss < 0.01, "bce should be near zero for perfect prediction, got {loss}");
    }

    #[test]
    fn fit_epoch_reduces_loss_on_linearly_separable_data() {
        let hidden = 4;
        let config = GateCalibrationConfig {
            learning_rate: 0.05,
            lambda_sparse: 0.0,
            lambda_margin: 0.0,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut cal = GateCalibrator::new(hidden, config);

        // Positive: residual = [1,1,1,1], Negative: residual = [-1,-1,-1,-1]
        let examples: Vec<GateExample> = (0..100)
            .map(|i| {
                if i % 2 == 0 {
                    make_example(vec![1.0, 1.0, 1.0, 1.0], 1.0)
                } else {
                    make_example(vec![-1.0, -1.0, -1.0, -1.0], 0.0)
                }
            })
            .collect();

        let before = cal.fit_epoch(&examples).bce_mean();
        // Second epoch should improve.
        let after = cal.fit_epoch(&examples).bce_mean();
        assert!(
            after <= before + 0.05,
            "loss should not increase significantly: before={before:.4} after={after:.4}"
        );
    }

    #[test]
    fn precision_recall_f1_perfect_separator() {
        let hidden = 2;
        let config = GateCalibrationConfig {
            learning_rate: 0.1,
            lambda_sparse: 0.0,
            lambda_margin: 0.0,
            weight_decay: 0.0,
            ..Default::default()
        };
        let mut cal = GateCalibrator::new(hidden, config);
        cal.gate = vec![1.0, 0.0]; // fires when residual[0] > 0

        let examples = vec![
            make_example(vec![1.0, 0.0], 1.0),
            make_example(vec![2.0, 0.0], 1.0),
            make_example(vec![-1.0, 0.0], 0.0),
            make_example(vec![-2.0, 0.0], 0.0),
        ];

        let (p, r, f1) = cal.precision_recall_f1(&examples, 0.0);
        assert_eq!(p, 1.0, "precision should be 1.0");
        assert_eq!(r, 1.0, "recall should be 1.0");
        assert_eq!(f1, 1.0, "F1 should be 1.0");
    }

    #[test]
    fn margin_loss_pushes_negatives_below_positives() {
        let hidden = 2;
        let config = GateCalibrationConfig {
            learning_rate: 0.05,
            lambda_sparse: 0.0,
            lambda_margin: 1.0,
            margin: 2.0,
            weight_decay: 0.0,
            grad_clip: 0.0,
        };
        let mut cal = GateCalibrator::new(hidden, config);
        // Positive residual points in +x direction, negative in -x.
        let pos = vec![1.0, 0.0];
        let neg = vec![-1.0, 0.0];
        let ex_with_pair = GateExample {
            residual: pos.clone(),
            label: 1.0,
            paired_negative: Some(neg.clone()),
        };
        for _ in 0..50 {
            cal.update(&ex_with_pair);
        }
        let pos_score = cal.score(&pos);
        let neg_score = cal.score(&neg);
        assert!(
            pos_score > neg_score,
            "positive score {pos_score:.3} should exceed negative {neg_score:.3}"
        );
    }
}
