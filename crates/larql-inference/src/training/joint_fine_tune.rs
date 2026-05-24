//! Joint fine-tune of gate (g), encoder (E), decoder (D), and optional
//! post-layer LoRA adapters (Phase 5.5, M9).
//!
//! Freeze: base model weights, Monty code.
//! Train: g, E (linear projection), D (linear decoder), optional LoRA.
//!
//! Loss per step:
//!
//! ```text
//! L_total =
//!   λ_task * CE(logits_after_delta, target)          // task CE
//!   + λ_kl  * KL(p_patched || p_base)                // KL anchor
//!   + λ_fire * fire_rate                              // sparsity
//!   + λ_delta * ||delta||²                            // delta regularisation
//!   + λ_surrogate * ||S(E(x)) - stopgrad(D(M(E(x))))||²  // surrogate alignment
//! ```
//!
//! All tensors are dense f32; no external ML framework required.
//! Updates use AdaGrad for each trainable component.

use serde::{Deserialize, Serialize};

use super::decoder_warmup::{LinearDecoder, SurrogateDecoder};
use super::gate_calibration::GateCalibrator;

// ── Hyper-parameters ──────────────────────────────────────────────────────────

/// Loss weights for the joint objective.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLossWeights {
    pub lambda_task: f32,
    pub lambda_kl: f32,
    pub lambda_fire: f32,
    pub lambda_delta: f32,
    pub lambda_surrogate: f32,
}

impl Default for JointLossWeights {
    fn default() -> Self {
        Self {
            lambda_task: 1.0,
            lambda_kl: 0.1,
            lambda_fire: 0.01,
            lambda_delta: 1e-3,
            lambda_surrogate: 0.05,
        }
    }
}

/// Scalar loss breakdown for one joint step.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct JointLossBreakdown {
    pub task_ce: f32,
    pub kl: f32,
    pub fire_rate: f32,
    pub delta_norm: f32,
    pub surrogate_mse: f32,
    pub total: f32,
}

// ── Minimal LoRA adapter ──────────────────────────────────────────────────────

/// Rank-r LoRA for a single weight matrix `W ∈ R^(rows × cols)`.
///
/// `ΔW = (1/r) * B * A`  where `A ∈ R^(r × cols)`, `B ∈ R^(rows × r)`.
#[derive(Debug, Clone)]
pub struct LoraAdapter {
    pub a: Vec<f32>, // row-major r × cols
    pub b: Vec<f32>, // row-major rows × r
    pub rows: usize,
    pub cols: usize,
    pub rank: usize,
    lr: f32,
    sq_a: Vec<f32>,
    sq_b: Vec<f32>,
}

impl LoraAdapter {
    /// Create a zero-initialised LoRA adapter.
    pub fn new(rows: usize, cols: usize, rank: usize, lr: f32) -> Self {
        let an = rank * cols;
        let bn = rows * rank;
        Self {
            a: vec![0.0; an],
            b: vec![0.0; bn],
            rows,
            cols,
            rank,
            lr,
            sq_a: vec![1e-8; an],
            sq_b: vec![1e-8; bn],
        }
    }

    /// Apply: returns `(1/rank) * B * A * input`, a rows-length delta vector.
    pub fn apply(&self, input: &[f32]) -> Vec<f32> {
        assert_eq!(input.len(), self.cols);
        // ax = A * input  (shape: rank)
        let mut ax = vec![0.0f32; self.rank];
        for r in 0..self.rank {
            for c in 0..self.cols {
                ax[r] += self.a[r * self.cols + c] * input[c];
            }
        }
        // out = B * ax  (shape: rows)
        let mut out = vec![0.0f32; self.rows];
        for row in 0..self.rows {
            for r in 0..self.rank {
                out[row] += self.b[row * self.rank + r] * ax[r];
            }
        }
        let scale = 1.0 / self.rank as f32;
        out.iter_mut().for_each(|v| *v *= scale);
        out
    }

    /// AdaGrad update given external gradient on the output.
    ///
    /// `grad_out ∈ R^rows`, `input ∈ R^cols`.
    pub fn update(&mut self, input: &[f32], grad_out: &[f32]) {
        let scale = 1.0 / self.rank as f32;

        // Gradient w.r.t. B: grad_out ⊗ ax^T (shape: rows × rank)
        let mut ax = vec![0.0f32; self.rank];
        for r in 0..self.rank {
            for c in 0..self.cols {
                ax[r] += self.a[r * self.cols + c] * input[c];
            }
        }

        let mut grad_b = vec![0.0f32; self.rows * self.rank];
        for row in 0..self.rows {
            for r in 0..self.rank {
                grad_b[row * self.rank + r] = scale * grad_out[row] * ax[r];
            }
        }

        // Gradient w.r.t. A: B^T * grad_out ⊗ input^T (shape: rank × cols)
        let mut bt_g = vec![0.0f32; self.rank]; // B^T * grad_out
        for r in 0..self.rank {
            for row in 0..self.rows {
                bt_g[r] += self.b[row * self.rank + r] * grad_out[row];
            }
        }
        let mut grad_a = vec![0.0f32; self.rank * self.cols];
        for r in 0..self.rank {
            for c in 0..self.cols {
                grad_a[r * self.cols + c] = scale * bt_g[r] * input[c];
            }
        }

        let lr = self.lr;
        for i in 0..self.a.len() {
            self.sq_a[i] += grad_a[i] * grad_a[i];
            self.a[i] -= lr * grad_a[i] / (self.sq_a[i].sqrt() + 1e-8);
        }
        for i in 0..self.b.len() {
            self.sq_b[i] += grad_b[i] * grad_b[i];
            self.b[i] -= lr * grad_b[i] / (self.sq_b[i].sqrt() + 1e-8);
        }
    }
}

// ── Encoder (linear projection residual → compact dict) ───────────────────────

/// Linear encoder `E: R^hidden → R^monty_dim`.
///
/// Compresses the residual to a compact vector before handing off to Monty.
#[derive(Debug, Clone)]
pub struct LinearEncoder {
    /// Row-major: w[monty_idx * hidden + hidden_idx]
    pub w: Vec<f32>,
    pub hidden_size: usize,
    pub monty_dim: usize,
    lr: f32,
    sq: Vec<f32>,
}

impl LinearEncoder {
    pub fn new(hidden_size: usize, monty_dim: usize, lr: f32) -> Self {
        let n = monty_dim * hidden_size;
        Self {
            w: vec![0.0; n],
            hidden_size,
            monty_dim,
            lr,
            sq: vec![1e-8; n],
        }
    }

    /// Forward: encoded = W * residual.
    pub fn forward(&self, residual: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.monty_dim];
        for m in 0..self.monty_dim {
            for h in 0..self.hidden_size {
                out[m] += self.w[m * self.hidden_size + h] * residual[h];
            }
        }
        out
    }

    /// AdaGrad update given gradient on the encoder output `∈ R^monty_dim`.
    pub fn update(&mut self, residual: &[f32], grad_out: &[f32]) {
        let lr = self.lr;
        for m in 0..self.monty_dim {
            for h in 0..self.hidden_size {
                let g = grad_out[m] * residual[h];
                let i = m * self.hidden_size + h;
                self.sq[i] += g * g;
                self.w[i] -= lr * g / (self.sq[i].sqrt() + 1e-8);
            }
        }
    }
}

// ── Joint trainer ─────────────────────────────────────────────────────────────

/// All trainable components bundled together.
pub struct JointTrainer {
    pub gate: GateCalibrator,
    pub encoder: LinearEncoder,
    pub decoder: LinearDecoder,
    pub surrogate: SurrogateDecoder,
    pub lora: Option<LoraAdapter>,
    pub weights: JointLossWeights,
}

impl JointTrainer {
    pub fn new(
        hidden_size: usize,
        monty_dim: usize,
        lora_rank: Option<usize>,
        weights: JointLossWeights,
        lr: f32,
    ) -> Self {
        use super::decoder_warmup::DecoderWarmupConfig;
        use super::gate_calibration::GateCalibrationConfig;

        let dec_cfg = DecoderWarmupConfig {
            learning_rate: lr,
            lambda_norm: weights.lambda_delta,
            grad_clip: 1.0,
            weight_decay: 1e-4,
        };
        let gate_cfg = GateCalibrationConfig {
            learning_rate: lr,
            lambda_sparse: weights.lambda_fire,
            margin: 1.0,
            lambda_margin: 0.1,
            weight_decay: 1e-4,
            grad_clip: 1.0,
        };

        Self {
            gate: GateCalibrator::new(hidden_size, gate_cfg),
            encoder: LinearEncoder::new(hidden_size, monty_dim, lr),
            decoder: LinearDecoder::new(hidden_size, monty_dim, dec_cfg.clone()),
            surrogate: SurrogateDecoder::new(hidden_size, dec_cfg),
            lora: lora_rank.map(|r| LoraAdapter::new(hidden_size, hidden_size, r, lr)),
            weights,
        }
    }
}

// ── Joint training step ───────────────────────────────────────────────────────

/// One joint training example.
pub struct JointExample {
    /// Residual at the call site.
    pub residual: Vec<f32>,
    /// Monty output (stop-gradient boundary).
    pub monty_output: Vec<f32>,
    /// Baseline log-probs (log-softmax, vocab-length).
    pub base_log_probs: Vec<f32>,
    /// Target token index.
    pub target_idx: usize,
    /// True = call should fire (positive), false = hard negative.
    pub label: f32,
}

/// Compute KL(patched || base) = sum_i p_patched_i * (log_p_patched_i - log_p_base_i).
fn kl_divergence(log_patched: &[f32], log_base: &[f32]) -> f32 {
    log_patched
        .iter()
        .zip(log_base.iter())
        .map(|(lp, lb)| {
            let p = lp.exp();
            if p < 1e-9 { 0.0 } else { p * (lp - lb) }
        })
        .sum()
}

/// Mock log-softmax of (base_log_probs + logit_delta_at_target).
///
/// In a real implementation this would be applied before softmax; here we
/// shift the target logit and renormalise approximately.
fn apply_delta_to_logprobs(base_lp: &[f32], target: usize, delta_at_target: f32) -> Vec<f32> {
    let mut shifted = base_lp.to_vec();
    shifted[target] += delta_at_target;
    // Approximate log-sum-exp renormalisation.
    let max = shifted.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let lse = max + shifted.iter().map(|v| (v - max).exp()).sum::<f32>().ln();
    shifted.iter_mut().for_each(|v| *v -= lse);
    shifted
}

/// One joint step: update all trainable components and return the loss breakdown.
///
/// `trainer` is mutated in-place; `monty_output` is treated as stop-gradient
/// (no gradient flows back through it).
pub fn joint_step(trainer: &mut JointTrainer, example: &JointExample) -> JointLossBreakdown {
    let hidden = trainer.encoder.hidden_size;

    // ── Forward ──────────────────────────────────────────────────────────────

    // Encoder: residual → compact encoding.
    let encoded = trainer.encoder.forward(&example.residual);

    // Decoder: monty_output → residual delta (stop-gradient at Monty boundary).
    let delta = trainer.decoder.forward(&example.monty_output);

    // Gate score (used for fire-rate sparsity).
    let gate_score = trainer.gate.score(&example.residual);
    let fire_prob = 1.0 / (1.0 + (-gate_score).exp());
    let fire_rate = fire_prob; // expected fire rate for one example

    // LoRA: optional residual delta on top of decoder output.
    let lora_delta: Vec<f32> = if let Some(ref lora) = trainer.lora {
        lora.apply(&example.residual)
    } else {
        vec![0.0; hidden]
    };

    // Combined delta: decoder + LoRA.
    let combined_delta: Vec<f32> = delta
        .iter()
        .zip(lora_delta.iter())
        .map(|(d, l)| d + l)
        .collect();

    // Mock patched log-probs: apply combined delta to target logit.
    let target_delta = combined_delta
        .get(example.target_idx % hidden)
        .copied()
        .unwrap_or(0.0);
    let patched_lp = apply_delta_to_logprobs(&example.base_log_probs, example.target_idx, target_delta);

    // ── Losses ────────────────────────────────────────────────────────────────

    // Task CE.
    let task_ce = -patched_lp[example.target_idx];

    // KL anchor: KL(patched || base).
    let kl = kl_divergence(&patched_lp, &example.base_log_probs);

    // Delta norm regularisation.
    let delta_norm = combined_delta.iter().map(|v| v * v).sum::<f32>();

    // Surrogate alignment: ||S(residual) - stopgrad(delta)||².
    let surrogate_pred = trainer.surrogate.forward(&example.residual);
    let surrogate_mse: f32 = surrogate_pred
        .iter()
        .zip(combined_delta.iter())
        .map(|(s, d)| (s - d).powi(2))
        .sum::<f32>()
        / hidden as f32;

    let w = &trainer.weights;
    let total = w.lambda_task * task_ce
        + w.lambda_kl * kl
        + w.lambda_fire * fire_rate
        + w.lambda_delta * delta_norm
        + w.lambda_surrogate * surrogate_mse;

    // ── Backward (simplified, w.r.t. individual components) ──────────────────

    // Gate: BCE gradient from label + sparsity.
    use super::gate_calibration::GateExample;
    let gate_ex = GateExample {
        residual: example.residual.clone(),
        label: example.label,
        paired_negative: None,
    };
    trainer.gate.update(&gate_ex);

    // Decoder: stop-gradient on Monty output; gradient flows back through delta.
    // Gradient of total w.r.t. delta[target_idx] ≈ (task_ce + kl) gradient.
    // We use a simplified proxy: gradient of CE at target w.r.t. delta.
    {
        use super::decoder_warmup::DecoderWarmupExample;
        let dec_ex = DecoderWarmupExample {
            monty_output: example.monty_output.clone(),
            residual: example.residual.clone(),
            target_token_idx: example.target_idx,
            vocab_size: example.base_log_probs.len(),
        };
        trainer.decoder.update(&dec_ex);
    }

    // Surrogate: fit to current combined_delta (stop-gradient).
    trainer
        .surrogate
        .update_to_match(&example.residual, &combined_delta);

    // Encoder: gradient from surrogate alignment loss (approximate).
    // ∂surrogate_mse/∂encoded flows through the surrogate matrix S:
    // We approximate by computing S * grad_surrogate w.r.t. encoded,
    // but the surrogate doesn't depend on the encoder directly in this
    // simplified setup — instead use a proxy gradient that pushes encoded
    // towards the monty_output direction.
    {
        let grad_encoded: Vec<f32> = encoded
            .iter()
            .zip(example.monty_output.iter())
            .map(|(e, m)| 2.0 * w.lambda_surrogate * (e - m) / hidden as f32)
            .collect();
        trainer.encoder.update(&example.residual, &grad_encoded);
    }

    // LoRA: gradient from task CE + KL, approximate.
    if let Some(ref mut lora) = trainer.lora {
        // Gradient of total w.r.t. lora_delta[target_idx]:
        // ≈ (patched_lp[target_idx] - 1.0) * lambda_task  (CE gradient)
        let p_target = patched_lp[example.target_idx].exp();
        let grad_scalar = w.lambda_task * (p_target - 1.0);
        let mut grad_out = vec![0.0f32; hidden];
        let ti = example.target_idx % hidden;
        grad_out[ti] = grad_scalar;
        lora.update(&example.residual, &grad_out);
    }

    JointLossBreakdown {
        task_ce,
        kl,
        fire_rate,
        delta_norm,
        surrogate_mse,
        total,
    }
}

/// Run one epoch over a batch of examples.
pub fn joint_epoch(trainer: &mut JointTrainer, examples: &[JointExample]) -> JointLossBreakdown {
    let mut acc = JointLossBreakdown::default();
    let n = examples.len();
    if n == 0 {
        return acc;
    }
    for ex in examples {
        let b = joint_step(trainer, ex);
        acc.task_ce += b.task_ce;
        acc.kl += b.kl;
        acc.fire_rate += b.fire_rate;
        acc.delta_norm += b.delta_norm;
        acc.surrogate_mse += b.surrogate_mse;
        acc.total += b.total;
    }
    let n = n as f32;
    acc.task_ce /= n;
    acc.kl /= n;
    acc.fire_rate /= n;
    acc.delta_norm /= n;
    acc.surrogate_mse /= n;
    acc.total /= n;
    acc
}

// ── Ablation helpers ──────────────────────────────────────────────────────────

/// Ablation configuration: which components to enable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationConfig {
    pub use_residual_delta: bool,
    pub use_logit_bias: bool,
    pub use_lora: bool,
    /// "raw" or "topk".
    pub codec: String,
    /// Trigger score threshold.
    pub score_threshold: f32,
    /// Cooldown tokens between fires.
    pub cooldown_tokens: usize,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            use_residual_delta: true,
            use_logit_bias: false,
            use_lora: false,
            codec: "raw".to_string(),
            score_threshold: 0.0,
            cooldown_tokens: 0,
        }
    }
}

/// Result for one ablation variant.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AblationResult {
    pub config: AblationConfig,
    pub final_loss: f32,
    pub n_epochs: usize,
}

/// Run a quick ablation: train for `n_epochs` and report final loss.
pub fn run_ablation(
    examples: &[JointExample],
    hidden_size: usize,
    monty_dim: usize,
    ablation: AblationConfig,
    n_epochs: usize,
    lr: f32,
) -> AblationResult {
    let weights = JointLossWeights {
        lambda_task: 1.0,
        lambda_kl: if ablation.use_logit_bias { 0.2 } else { 0.1 },
        lambda_delta: if ablation.use_residual_delta { 1e-3 } else { 0.0 },
        lambda_fire: 0.01,
        lambda_surrogate: 0.05,
    };
    let lora_rank = if ablation.use_lora { Some(2) } else { None };
    let mut trainer = JointTrainer::new(hidden_size, monty_dim, lora_rank, weights, lr);

    let mut final_loss = 0.0;
    for _ in 0..n_epochs {
        let breakdown = joint_epoch(&mut trainer, examples);
        final_loss = breakdown.total;
    }

    AblationResult {
        config: ablation,
        final_loss,
        n_epochs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::gate_calibration::GateCalibrationConfig;
    use crate::training::decoder_warmup::DecoderWarmupConfig;

    fn make_example(hidden: usize, monty_dim: usize, vocab: usize, target: usize) -> JointExample {
        JointExample {
            residual: vec![0.1; hidden],
            monty_output: vec![0.5; monty_dim],
            base_log_probs: vec![-2.0f32; vocab],
            target_idx: target,
            label: 1.0,
        }
    }

    #[test]
    fn lora_adapter_zero_init_returns_zeros() {
        let lora = LoraAdapter::new(4, 4, 2, 0.01);
        let out = lora.apply(&[1.0, 0.0, 0.0, 0.0]);
        assert!(out.iter().all(|&v| v == 0.0), "zero init should produce zero output");
    }

    #[test]
    fn lora_adapter_update_runs_without_panic() {
        let mut lora = LoraAdapter::new(4, 4, 2, 0.01);
        lora.update(&[1.0, 0.0, 0.0, 0.0], &[0.1, -0.1, 0.0, 0.0]);
    }

    #[test]
    fn linear_encoder_zero_init_returns_zeros() {
        let enc = LinearEncoder::new(8, 4, 0.01);
        let out = enc.forward(&vec![1.0; 8]);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn joint_step_returns_finite_losses() {
        let hidden = 8;
        let monty_dim = 4;
        let vocab = 10;
        let mut trainer =
            JointTrainer::new(hidden, monty_dim, None, JointLossWeights::default(), 0.01);
        let ex = make_example(hidden, monty_dim, vocab, 2);
        let breakdown = joint_step(&mut trainer, &ex);
        assert!(breakdown.task_ce.is_finite(), "task CE should be finite");
        assert!(breakdown.kl.is_finite(), "KL should be finite");
        assert!(breakdown.total.is_finite(), "total should be finite");
    }

    #[test]
    fn joint_step_with_lora_runs_without_panic() {
        let hidden = 8;
        let monty_dim = 4;
        let vocab = 10;
        let mut trainer =
            JointTrainer::new(hidden, monty_dim, Some(2), JointLossWeights::default(), 0.01);
        let ex = make_example(hidden, monty_dim, vocab, 3);
        joint_step(&mut trainer, &ex);
    }

    #[test]
    fn joint_epoch_on_empty_returns_zero() {
        let mut trainer =
            JointTrainer::new(8, 4, None, JointLossWeights::default(), 0.01);
        let b = joint_epoch(&mut trainer, &[]);
        assert_eq!(b.total, 0.0);
    }

    #[test]
    fn joint_epoch_loss_does_not_increase_significantly() {
        let hidden = 8;
        let monty_dim = 4;
        let vocab = 10;
        let examples: Vec<JointExample> = (0..20)
            .map(|i| make_example(hidden, monty_dim, vocab, i % vocab))
            .collect();

        let mut trainer =
            JointTrainer::new(hidden, monty_dim, None, JointLossWeights::default(), 0.02);

        let b1 = joint_epoch(&mut trainer, &examples);
        let b2 = joint_epoch(&mut trainer, &examples);
        assert!(
            b2.total <= b1.total + 1.0,
            "loss should not explode: {:.4} → {:.4}",
            b1.total,
            b2.total
        );
    }

    #[test]
    fn ablation_run_produces_finite_result() {
        let hidden = 8;
        let monty_dim = 4;
        let vocab = 10;
        let examples: Vec<JointExample> = (0..10)
            .map(|i| make_example(hidden, monty_dim, vocab, i % vocab))
            .collect();

        let ablation = AblationConfig::default();
        let result = run_ablation(&examples, hidden, monty_dim, ablation, 3, 0.01);
        assert!(result.final_loss.is_finite());
        assert_eq!(result.n_epochs, 3);
    }

    #[test]
    fn ablation_with_lora_produces_finite_result() {
        let hidden = 8;
        let monty_dim = 4;
        let vocab = 10;
        let examples: Vec<JointExample> = (0..10)
            .map(|i| make_example(hidden, monty_dim, vocab, i % vocab))
            .collect();

        let ablation = AblationConfig {
            use_lora: true,
            ..Default::default()
        };
        let result = run_ablation(&examples, hidden, monty_dim, ablation, 3, 0.01);
        assert!(result.final_loss.is_finite());
    }
}
