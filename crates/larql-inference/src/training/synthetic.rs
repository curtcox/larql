//! Synthetic task generator for call-patch training data (Phase 5.1).
//!
//! Produces examples where a Monty call has exact algorithmic advantage over
//! the base LM: arithmetic, date arithmetic, string transforms, unit
//! conversions, table lookup, and symbolic formatting.
//!
//! Each `SyntheticExample` captures:
//! - a text prompt
//! - the target next-token string
//! - a suggested (layer, position) site where a call patch should fire
//! - the expected Monty input dict
//! - the expected Monty output dict (residual_delta or logit_bias keys)
//! - a task category label
//!
//! None of these require a real model; they are used by training harnesses to
//! build gate-calibration and decoder-warmup datasets.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

/// Category of a synthetic training task.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskCategory {
    ArithmeticNorm,
    DateDelta,
    StringTransform,
    UnitConversion,
    TableLookup,
    SymbolicFormat,
}

/// One synthetic training example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticExample {
    pub category: TaskCategory,
    /// Human-readable prompt given to the model (prefix up to the answer).
    pub prompt: String,
    /// The target next token (string form).
    pub target_token: String,
    /// Suggested layer and token-position for the call-patch site.
    pub layer: usize,
    pub position: usize,
    /// Input dict the Monty code should receive.
    pub monty_input: Value,
    /// Output dict the Monty code should produce.
    pub monty_output: Value,
    /// True → this is a positive example (call helps); false → hard negative.
    pub is_positive: bool,
}

impl SyntheticExample {
    pub fn is_hard_negative(&self) -> bool {
        !self.is_positive
    }
}

// ── Arithmetic normalization ─────────────────────────────────────────────────

/// Generate `n` arithmetic-normalization examples.
///
/// Task: given an integer sum in the prompt, produce the normalised decimal
/// string (e.g., `"0.25"` for `1/4`).
///
/// The call patch receives `{"a": a, "b": b}` and returns `{"result": "..."}`.
pub fn arithmetic_norm_examples(n: usize, seed: u64) -> Vec<SyntheticExample> {
    let mut rng = Lcg64(seed);
    (0..n)
        .map(|_| {
            let a = (rng.next() % 99 + 1) as i64;
            let b = (rng.next() % 99 + 1) as i64;
            let sum = a + b;
            SyntheticExample {
                category: TaskCategory::ArithmeticNorm,
                prompt: format!("What is {a} + {b}? Answer: "),
                target_token: sum.to_string(),
                layer: 12,
                position: 8,
                monty_input: json!({"a": a, "b": b}),
                monty_output: json!({"result": sum.to_string()}),
                is_positive: true,
            }
        })
        .collect()
}

/// Hard negatives for arithmetic: same prompt, wrong position or unrelated sum.
pub fn arithmetic_norm_hard_negatives(n: usize, seed: u64) -> Vec<SyntheticExample> {
    let mut rng = Lcg64(seed.wrapping_add(0xdead));
    (0..n)
        .map(|_| {
            let a = (rng.next() % 99 + 1) as i64;
            let b = (rng.next() % 99 + 1) as i64;
            let sum = a + b;
            // Wrong position — call fires too early in the prompt.
            SyntheticExample {
                category: TaskCategory::ArithmeticNorm,
                prompt: format!("What is {a} + {b}? Answer: "),
                target_token: sum.to_string(),
                layer: 12,
                position: 2, // too early
                monty_input: json!({"a": a, "b": b}),
                monty_output: json!({"result": "wrong"}),
                is_positive: false,
            }
        })
        .collect()
}

// ── Date arithmetic ──────────────────────────────────────────────────────────

/// Generate date-delta examples: "How many days from {year1}-01-01 to {year2}-01-01?"
pub fn date_delta_examples(n: usize, seed: u64) -> Vec<SyntheticExample> {
    let mut rng = Lcg64(seed.wrapping_add(1));
    (0..n)
        .map(|_| {
            let year1 = 2000 + (rng.next() % 30) as u32;
            let year2 = year1 + 1 + (rng.next() % 5) as u32;
            let days = (year2 - year1) * 365; // simplified (no leap year handling)
            SyntheticExample {
                category: TaskCategory::DateDelta,
                prompt: format!("Days from {year1}-01-01 to {year2}-01-01: "),
                target_token: days.to_string(),
                layer: 14,
                position: 10,
                monty_input: json!({"year1": year1, "year2": year2}),
                monty_output: json!({"days": days}),
                is_positive: true,
            }
        })
        .collect()
}

// ── String transforms ────────────────────────────────────────────────────────

/// String-reverse examples: "Reverse 'hello': "
pub fn string_transform_examples(n: usize, seed: u64) -> Vec<SyntheticExample> {
    let words = [
        "hello", "world", "rust", "larql", "patch", "gate", "monty", "call",
    ];
    let mut rng = Lcg64(seed.wrapping_add(2));
    (0..n)
        .map(|_| {
            let word = words[(rng.next() as usize) % words.len()];
            let reversed: String = word.chars().rev().collect();
            SyntheticExample {
                category: TaskCategory::StringTransform,
                prompt: format!("Reverse \"{word}\": "),
                target_token: reversed.clone(),
                layer: 8,
                position: 6,
                monty_input: json!({"text": word}),
                monty_output: json!({"result": reversed}),
                is_positive: true,
            }
        })
        .collect()
}

// ── Unit conversions ─────────────────────────────────────────────────────────

/// Celsius-to-Fahrenheit examples: "20°C in °F is "
pub fn unit_conversion_examples(n: usize, seed: u64) -> Vec<SyntheticExample> {
    let mut rng = Lcg64(seed.wrapping_add(3));
    (0..n)
        .map(|_| {
            let celsius = (rng.next() % 100) as i64 - 20; // -20..80
            let fahrenheit = celsius * 9 / 5 + 32;
            SyntheticExample {
                category: TaskCategory::UnitConversion,
                prompt: format!("{celsius}°C in °F is "),
                target_token: fahrenheit.to_string(),
                layer: 10,
                position: 5,
                monty_input: json!({"celsius": celsius}),
                monty_output: json!({"fahrenheit": fahrenheit}),
                is_positive: true,
            }
        })
        .collect()
}

// ── Table lookup ─────────────────────────────────────────────────────────────

/// Key→value table lookup examples.
pub fn table_lookup_examples(n: usize, seed: u64) -> Vec<SyntheticExample> {
    let table: &[(&str, &str)] = &[
        ("Paris", "France"),
        ("Berlin", "Germany"),
        ("Tokyo", "Japan"),
        ("Rome", "Italy"),
        ("Madrid", "Spain"),
        ("Lisbon", "Portugal"),
        ("Vienna", "Austria"),
        ("Warsaw", "Poland"),
    ];
    let mut rng = Lcg64(seed.wrapping_add(4));
    (0..n)
        .map(|_| {
            let (city, country) = table[(rng.next() as usize) % table.len()];
            SyntheticExample {
                category: TaskCategory::TableLookup,
                prompt: format!("Capital city {city} is in "),
                target_token: country.to_string(),
                layer: 16,
                position: 9,
                monty_input: json!({"city": city}),
                monty_output: json!({"country": country}),
                is_positive: true,
            }
        })
        .collect()
}

// ── Symbolic formatting ──────────────────────────────────────────────────────

/// Symbolic formatting: pad an integer to fixed width, e.g. "0007".
pub fn symbolic_format_examples(n: usize, seed: u64) -> Vec<SyntheticExample> {
    let mut rng = Lcg64(seed.wrapping_add(5));
    (0..n)
        .map(|_| {
            let num = rng.next() % 1000;
            let formatted = format!("{num:04}");
            SyntheticExample {
                category: TaskCategory::SymbolicFormat,
                prompt: format!("Zero-pad {num} to 4 digits: "),
                target_token: formatted.clone(),
                layer: 11,
                position: 7,
                monty_input: json!({"number": num, "width": 4}),
                monty_output: json!({"formatted": formatted}),
                is_positive: true,
            }
        })
        .collect()
}

// ── Convenience: full dataset ─────────────────────────────────────────────────

/// Build a balanced dataset with positives and hard negatives across all categories.
pub fn build_dataset(examples_per_category: usize, seed: u64) -> Vec<SyntheticExample> {
    let n = examples_per_category;
    let mut dataset = Vec::new();
    dataset.extend(arithmetic_norm_examples(n, seed));
    dataset.extend(arithmetic_norm_hard_negatives(n / 2, seed));
    dataset.extend(date_delta_examples(n, seed));
    dataset.extend(string_transform_examples(n, seed));
    dataset.extend(unit_conversion_examples(n, seed));
    dataset.extend(table_lookup_examples(n, seed));
    dataset.extend(symbolic_format_examples(n, seed));
    dataset
}

// ── Minimal deterministic PRNG (no deps) ─────────────────────────────────────

struct Lcg64(u64);
impl Lcg64 {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arithmetic_examples_are_correct() {
        let examples = arithmetic_norm_examples(10, 42);
        for ex in &examples {
            assert!(ex.is_positive);
            let a: i64 = ex.monty_input["a"].as_i64().unwrap();
            let b: i64 = ex.monty_input["b"].as_i64().unwrap();
            let expected = (a + b).to_string();
            assert_eq!(ex.target_token, expected);
            assert_eq!(ex.monty_output["result"].as_str().unwrap(), expected);
        }
    }

    #[test]
    fn arithmetic_hard_negatives_are_flagged() {
        let negs = arithmetic_norm_hard_negatives(5, 42);
        for ex in &negs {
            assert!(!ex.is_positive);
        }
    }

    #[test]
    fn build_dataset_contains_both_polarities() {
        let ds = build_dataset(8, 0);
        assert!(ds.iter().any(|e| e.is_positive));
        assert!(ds.iter().any(|e| !e.is_positive));
    }

    #[test]
    fn all_categories_present_in_dataset() {
        let ds = build_dataset(4, 1);
        use TaskCategory::*;
        for cat in [
            ArithmeticNorm,
            DateDelta,
            StringTransform,
            UnitConversion,
            TableLookup,
            SymbolicFormat,
        ] {
            assert!(ds.iter().any(|e| e.category == cat), "missing {:?}", cat);
        }
    }

    #[test]
    fn string_transform_reversal_is_correct() {
        let examples = string_transform_examples(20, 7);
        for ex in &examples {
            let word = ex.monty_input["text"].as_str().unwrap();
            let reversed: String = word.chars().rev().collect();
            assert_eq!(ex.target_token, reversed);
        }
    }
}
