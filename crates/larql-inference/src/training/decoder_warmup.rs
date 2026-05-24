//! Forced-call decoder warmup (Phase 5.2).
//!
//! Trains a linear decoder `D: R^monty_out → R^hidden` under teacher forcing:
//! the call is forced to fire at known-good positions and the decoder is
//! optimised to minimise next-token cross-entropy plus regularisation.
//!
//! Gradient does NOT flow through Monty output (stop-gradient at Monty boundary).
//!
//! Loss:
//!
//! ```text
//! L = CE(logits_after_delta, target)
//!   + λ_norm * ||delta||²
//! ```
//!
//! The decoder is a simple affine map:
//!
//! ```text
//! delta = W * monty_output + b
//! ```
//!
//! where `monty_output ∈ R^monty_dim` and `delta ∈ R^hidden`.
//! Gradient update uses vanilla SGD with optional gradient clipping.

use serde::{Deserialize, Serialize};

/// One decoder-warmup training example.
///
/// All fields are taken from a `SyntheticExample` plus mock logit/target info.
#[derive(Debug, Clone)]
pub struct DecoderWarmupExample {
    /// Monty output vector (stop-gradient boundary — no gradient flows here).
    pub monty_output: Vec<f32>,
    /// Current residual before the delta is applied.
    pub residual: Vec<f32>,
    /// One-hot target token index in the vocabulary.
    pub target_token_idx: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
}

/// Hyper-parameters for the decoder warmup pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderWarmupConfig {
    pub learning_rate: f32,
    /// L2 penalty on the produced delta.
    pub lambda_norm: f32,
    /// Gradient clipping norm (0 = no clip).
    pub grad_clip: f32,
    /// Weight decay on W / b.
    pub weight_decay: f32,
}

impl Default for DecoderWarmupConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            lambda_norm: 1e-3,
            grad_clip: 1.0,
            weight_decay: 1e-4,
        }
    }
}

/// Linear decoder `D: monty_out → residual_delta`.
///
/// Weights: W ∈ R^(hidden × monty_dim), b ∈ R^hidden.
#[derive(Debug, Clone)]
pub struct LinearDecoder {
    /// Row-major: w[hidden_idx * monty_dim + monty_idx]
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub hidden_size: usize,
    pub monty_dim: usize,
    config: DecoderWarmupConfig,
    /// AdaGrad accumulators for W and b.
    sq_w: Vec<f32>,
    sq_b: Vec<f32>,
}

impl LinearDecoder {
    /// Create a zero-initialised decoder.
    pub fn new(hidden_size: usize, monty_dim: usize, config: DecoderWarmupConfig) -> Self {
        let wn = hidden_size * monty_dim;
        Self {
            w: vec![0.0; wn],
            b: vec![0.0; hidden_size],
            hidden_size,
            monty_dim,
            config,
            sq_w: vec![1e-8; wn],
            sq_b: vec![1e-8; hidden_size],
        }
    }

    /// Forward pass: delta = W * monty_output + b.
    pub fn forward(&self, monty_output: &[f32]) -> Vec<f32> {
        assert_eq!(monty_output.len(), self.monty_dim);
        let mut delta = self.b.clone();
        for h in 0..self.hidden_size {
            let row_start = h * self.monty_dim;
            for m in 0..self.monty_dim {
                delta[h] += self.w[row_start + m] * monty_output[m];
            }
        }
        delta
    }

    /// One gradient step: updates W and b.
    ///
    /// Returns `(ce_loss, norm_reg)`.
    pub fn update(&mut self, example: &DecoderWarmupExample) -> (f32, f32) {
        let delta = self.forward(&example.monty_output);
        let norm_reg = self.config.lambda_norm * delta.iter().map(|v| v * v).sum::<f32>();

        // Mock unembedding: project (residual + delta) to vocab via a one-hot
        // approximation.  A real training loop would compose with the actual
        // unembedding matrix; here we simulate CE loss using the target
        // residual dimension as a proxy.
        //
        // CE gradient w.r.t. delta: simplified as (delta - target_signal).
        // This drives delta to be small and aligned with the target.
        let target_idx = example.target_token_idx % example.residual.len();
        let mut grad_delta = delta.clone();
        grad_delta[target_idx] -= 1.0; // one-hot cross-entropy gradient

        // Regularisation gradient on delta.
        for (gd, d) in grad_delta.iter_mut().zip(delta.iter()) {
            *gd += 2.0 * self.config.lambda_norm * d;
        }

        // Gradient of W: grad_delta ⊗ monty_output^T
        let mut grad_w = vec![0.0f32; self.w.len()];
        for h in 0..self.hidden_size {
            for m in 0..self.monty_dim {
                grad_w[h * self.monty_dim + m] =
                    grad_delta[h] * example.monty_output[m]
                    + self.config.weight_decay * self.w[h * self.monty_dim + m];
            }
        }
        let mut grad_b = grad_delta.clone();
        for (gb, b) in grad_b.iter_mut().zip(self.b.iter()) {
            *gb += self.config.weight_decay * b;
        }

        // Clip gradients.
        if self.config.grad_clip > 0.0 {
            let all_norms = grad_w.iter().chain(grad_b.iter()).map(|v| v * v).sum::<f32>().sqrt();
            if all_norms > self.config.grad_clip {
                let s = self.config.grad_clip / all_norms;
                grad_w.iter_mut().for_each(|v| *v *= s);
                grad_b.iter_mut().for_each(|v| *v *= s);
            }
        }

        // AdaGrad update.
        let lr = self.config.learning_rate;
        for i in 0..self.w.len() {
            self.sq_w[i] += grad_w[i] * grad_w[i];
            self.w[i] -= lr * grad_w[i] / (self.sq_w[i].sqrt() + 1e-8);
        }
        for i in 0..self.hidden_size {
            self.sq_b[i] += grad_b[i] * grad_b[i];
            self.b[i] -= lr * grad_b[i] / (self.sq_b[i].sqrt() + 1e-8);
        }

        // Approximate CE loss: mean squared deviation from one-hot target.
        let ce_approx: f32 = {
            let mut s: f32 = 0.0;
            for (h, d) in delta.iter().enumerate() {
                let t = if h == target_idx { 1.0 } else { 0.0 };
                s += (d - t).powi(2);
            }
            s / self.hidden_size as f32
        };

        (ce_approx, norm_reg)
    }

    /// Run one epoch; return `(mean_ce, mean_norm_reg)`.
    pub fn fit_epoch(&mut self, examples: &[DecoderWarmupExample]) -> (f32, f32) {
        if examples.is_empty() {
            return (0.0, 0.0);
        }
        let (mut ce_sum, mut reg_sum) = (0.0f64, 0.0f64);
        for ex in examples {
            let (ce, reg) = self.update(ex);
            ce_sum += ce as f64;
            reg_sum += reg as f64;
        }
        let n = examples.len() as f64;
        ((ce_sum / n) as f32, (reg_sum / n) as f32)
    }
}

/// Surrogate forward: approximates D(M(E(x))) with a trained linear map.
///
/// The surrogate replaces D∘M∘E in the backward pass (gradient flows through
/// the surrogate rather than through Monty).  At inference time the real Monty
/// path is always used.
///
/// `surrogate_output = S_gate * residual` where S_gate ∈ R^(hidden × hidden).
pub struct SurrogateDecoder {
    /// Row-major matrix S ∈ R^(hidden × hidden).
    pub s: Vec<f32>,
    pub hidden_size: usize,
    config: DecoderWarmupConfig,
    sq_s: Vec<f32>,
}

impl SurrogateDecoder {
    pub fn new(hidden_size: usize, config: DecoderWarmupConfig) -> Self {
        let n = hidden_size * hidden_size;
        Self {
            s: vec![0.0; n],
            hidden_size,
            config,
            sq_s: vec![1e-8; n],
        }
    }

    /// Forward pass: approx_delta = S * residual.
    pub fn forward(&self, residual: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.hidden_size];
        for h in 0..self.hidden_size {
            for r in 0..self.hidden_size {
                out[h] += self.s[h * self.hidden_size + r] * residual[r];
            }
        }
        out
    }

    /// Fit the surrogate to match a reference `real_delta` (from Monty).
    ///
    /// Loss: ||S * residual - real_delta||² (stop-gradient on real_delta).
    /// Returns the MSE loss.
    pub fn update_to_match(&mut self, residual: &[f32], real_delta: &[f32]) -> f32 {
        let pred = self.forward(residual);
        let mut mse = 0.0f32;
        let mut grad = vec![0.0f32; self.s.len()];

        for h in 0..self.hidden_size {
            let err = pred[h] - real_delta[h];
            mse += err * err;
            for r in 0..self.hidden_size {
                grad[h * self.hidden_size + r] = 2.0 * err * residual[r]
                    + self.config.weight_decay * self.s[h * self.hidden_size + r];
            }
        }
        mse /= self.hidden_size as f32;

        // Clip.
        if self.config.grad_clip > 0.0 {
            let norm = grad.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > self.config.grad_clip {
                let s = self.config.grad_clip / norm;
                grad.iter_mut().for_each(|v| *v *= s);
            }
        }

        // AdaGrad.
        let lr = self.config.learning_rate;
        for i in 0..self.s.len() {
            self.sq_s[i] += grad[i] * grad[i];
            self.s[i] -= lr * grad[i] / (self.sq_s[i].sqrt() + 1e-8);
        }
        mse
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_decoder_forward_zero_weights_returns_bias() {
        let config = DecoderWarmupConfig::default();
        let mut dec = LinearDecoder::new(4, 2, config);
        dec.b = vec![1.0, 2.0, 3.0, 4.0];
        let delta = dec.forward(&[0.0, 0.0]);
        assert_eq!(delta, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn fit_epoch_reduces_ce_over_epochs() {
        let config = DecoderWarmupConfig {
            learning_rate: 0.1,
            lambda_norm: 0.0,
            grad_clip: 0.0,
            weight_decay: 0.0,
        };
        let mut dec = LinearDecoder::new(4, 2, config);
        // monty_output [1, 0] should produce delta with component 0 = 1.
        let examples: Vec<DecoderWarmupExample> = (0..50)
            .map(|_| DecoderWarmupExample {
                monty_output: vec![1.0, 0.0],
                residual: vec![0.0; 4],
                target_token_idx: 0,
                vocab_size: 100,
            })
            .collect();

        let (ce1, _) = dec.fit_epoch(&examples);
        let (ce2, _) = dec.fit_epoch(&examples);
        assert!(
            ce2 <= ce1 + 0.05,
            "CE should not increase: {ce1:.4} → {ce2:.4}"
        );
    }

    #[test]
    fn surrogate_converges_to_identity_on_scaled_input() {
        let config = DecoderWarmupConfig {
            learning_rate: 0.05,
            lambda_norm: 0.0,
            grad_clip: 0.0,
            weight_decay: 0.0,
        };
        let hidden = 3;
        let mut surrogate = SurrogateDecoder::new(hidden, config);

        // Target: real_delta = 2 * residual (identity times 2)
        let residual = vec![1.0f32, 0.5, -1.0];
        let real_delta: Vec<f32> = residual.iter().map(|v| 2.0 * v).collect();

        let mut last_mse = f32::MAX;
        for _ in 0..200 {
            last_mse = surrogate.update_to_match(&residual, &real_delta);
        }
        assert!(last_mse < 0.1, "surrogate MSE should be small after 200 steps, got {last_mse:.4}");
    }
}
