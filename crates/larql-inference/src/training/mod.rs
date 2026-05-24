//! Training prototype for Monty call patches (M8, Phase 5).
//!
//! Modules:
//! - [`synthetic`]: generates training examples for tasks where Monty has exact
//!   algorithmic advantage (arithmetic, date, string, unit, table, format).
//! - [`gate_calibration`]: logistic-regression gate calibration with hard
//!   negatives and margin loss.
//! - [`decoder_warmup`]: forced-call linear decoder warmup (teacher forcing);
//!   also exposes the surrogate-backward prototype.
//! - [`evaluation`]: accuracy, gate P/R/F1, KL regression, reach probe.

pub mod decoder_warmup;
pub mod evaluation;
pub mod gate_calibration;
pub mod synthetic;
