//! Learned linear codec artifacts for Monty call patches (M6).
//!
//! Artifacts live under `<vindex>/call_codecs/<artifact_id>.json` and
//! store a row-major weight matrix `W[output_dim × input_dim]` plus an
//! optional bias. Input encoding projects a residual to a compact vector
//! for Monty; output decoding expands Monty's return value back to
//! hidden size.

use larql_vindex::patch::core::{decode_gate_vector, encode_gate_vector};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use thiserror::Error;

use super::CallError;

#[derive(Debug, Error)]
pub enum CodecError {
    #[error("missing codec artifact `{0}`")]
    MissingArtifact(String),
    #[error("codec `{artifact_id}` input dim mismatch: expected {expected}, got {actual}")]
    InputDimMismatch {
        artifact_id: String,
        expected: usize,
        actual: usize,
    },
    #[error("codec `{artifact_id}` output dim mismatch: expected {expected}, got {actual}")]
    OutputDimMismatch {
        artifact_id: String,
        expected: usize,
        actual: usize,
    },
    #[error("invalid codec artifact `{artifact_id}`: {detail}")]
    Invalid { artifact_id: String, detail: String },
}

impl From<CodecError> for CallError {
    fn from(err: CodecError) -> Self {
        CallError::ProgramRun(err.to_string())
    }
}

/// Row-major linear map `y = W @ x + b` with shape `[output_dim × input_dim]`.
#[derive(Debug, Clone, PartialEq)]
pub struct LearnedLinearCodec {
    pub artifact_id: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights: Vec<f32>,
    pub bias: Vec<f32>,
}

impl LearnedLinearCodec {
    pub fn new(
        artifact_id: impl Into<String>,
        input_dim: usize,
        output_dim: usize,
        weights: Vec<f32>,
        bias: Vec<f32>,
    ) -> Result<Self, CodecError> {
        let artifact_id = artifact_id.into();
        let expected_weights = input_dim.checked_mul(output_dim).ok_or_else(|| {
            CodecError::Invalid {
                artifact_id: artifact_id.clone(),
                detail: "weight matrix dimensions overflow".into(),
            }
        })?;
        if weights.len() != expected_weights {
            return Err(CodecError::Invalid {
                artifact_id,
                detail: format!(
                    "weights length {} != output_dim * input_dim ({expected_weights})",
                    weights.len()
                ),
            });
        }
        if bias.len() != output_dim {
            return Err(CodecError::Invalid {
                artifact_id,
                detail: format!(
                    "bias length {} != output_dim ({output_dim})",
                    bias.len()
                ),
            });
        }
        Ok(Self {
            artifact_id,
            input_dim,
            output_dim,
            weights,
            bias,
        })
    }

    pub fn apply(&self, input: &[f32]) -> Result<Vec<f32>, CodecError> {
        if input.len() != self.input_dim {
            return Err(CodecError::InputDimMismatch {
                artifact_id: self.artifact_id.clone(),
                expected: self.input_dim,
                actual: input.len(),
            });
        }
        let mut out = self.bias.clone();
        for (row, slot) in out.iter_mut().enumerate().take(self.output_dim) {
            let base = row * self.input_dim;
            let dot: f32 = self.weights[base..base + self.input_dim]
                .iter()
                .zip(input.iter())
                .map(|(w, x)| w * x)
                .sum();
            *slot += dot;
        }
        Ok(out)
    }
}

/// On-disk JSON artifact for a learned linear codec.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LearnedLinearCodecArtifact {
    pub version: u32,
    pub artifact_id: String,
    pub input_dim: usize,
    pub output_dim: usize,
    pub weights_b64: String,
    #[serde(default)]
    pub bias_b64: Option<String>,
}

impl LearnedLinearCodecArtifact {
    pub fn to_codec(&self) -> Result<LearnedLinearCodec, CodecError> {
        let weights = decode_gate_vector(&self.weights_b64).map_err(|e| CodecError::Invalid {
            artifact_id: self.artifact_id.clone(),
            detail: format!("decode weights: {e}"),
        })?;
        let bias = match &self.bias_b64 {
            Some(b64) => decode_gate_vector(b64).map_err(|e| CodecError::Invalid {
                artifact_id: self.artifact_id.clone(),
                detail: format!("decode bias: {e}"),
            })?,
            None => vec![0.0f32; self.output_dim],
        };
        LearnedLinearCodec::new(
            self.artifact_id.clone(),
            self.input_dim,
            self.output_dim,
            weights,
            bias,
        )
    }

    pub fn from_codec(codec: &LearnedLinearCodec) -> Self {
        Self {
            version: 1,
            artifact_id: codec.artifact_id.clone(),
            input_dim: codec.input_dim,
            output_dim: codec.output_dim,
            weights_b64: encode_gate_vector(&codec.weights),
            bias_b64: Some(encode_gate_vector(&codec.bias)),
        }
    }

    pub fn save(&self, path: &Path) -> Result<(), CodecError> {
        let json = serde_json::to_string_pretty(self).map_err(|e| CodecError::Invalid {
            artifact_id: self.artifact_id.clone(),
            detail: format!("serialize: {e}"),
        })?;
        std::fs::write(path, json).map_err(|e| CodecError::Invalid {
            artifact_id: self.artifact_id.clone(),
            detail: format!("write {}: {e}", path.display()),
        })
    }

    pub fn load(path: &Path) -> Result<Self, CodecError> {
        let text = std::fs::read_to_string(path).map_err(|e| CodecError::Invalid {
            artifact_id: path.display().to_string(),
            detail: format!("read: {e}"),
        })?;
        serde_json::from_str(&text).map_err(|e| CodecError::Invalid {
            artifact_id: path.display().to_string(),
            detail: format!("parse JSON: {e}"),
        })
    }
}

#[derive(Debug, Default, Clone)]
pub struct CodecRegistry {
    codecs: HashMap<String, LearnedLinearCodec>,
}

impl CodecRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, codec: LearnedLinearCodec) {
        self.codecs.insert(codec.artifact_id.clone(), codec);
    }

    pub fn get(&self, artifact_id: &str) -> Option<&LearnedLinearCodec> {
        self.codecs.get(artifact_id)
    }

    pub fn len(&self) -> usize {
        self.codecs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.codecs.is_empty()
    }
}

/// Load every `call_codecs/*.json` artifact under a vindex directory.
pub fn load_codec_registry_from_dir(vindex_dir: &Path) -> CodecRegistry {
    use larql_vindex::format::filenames::CALL_CODECS_DIR;

    let mut registry = CodecRegistry::new();
    let dir = vindex_dir.join(CALL_CODECS_DIR);
    let entries = match std::fs::read_dir(&dir) {
        Ok(entries) => entries,
        Err(_) => return registry,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        match LearnedLinearCodecArtifact::load(&path).and_then(|artifact| artifact.to_codec()) {
            Ok(codec) => registry.insert(codec),
            Err(err) => {
                eprintln!(
                    "warning: failed to load call codec {}: {err}",
                    path.display()
                );
            }
        }
    }
    registry
}

/// Encode a residual with a `learned_linear` codec from `input_schema`.
pub fn encode_learned_linear(
    residual: &[f32],
    codec_spec: &Value,
    registry: &CodecRegistry,
) -> Result<Value, CodecError> {
    let artifact_id = codec_spec
        .get("artifact_id")
        .and_then(Value::as_str)
        .ok_or_else(|| CodecError::Invalid {
            artifact_id: "<unknown>".into(),
            detail: "learned_linear codec missing artifact_id".into(),
        })?;
    let expected_input = codec_spec
        .get("input_dim")
        .and_then(Value::as_u64)
        .map(|n| n as usize);
    let codec = registry.get(artifact_id).ok_or_else(|| {
        CodecError::MissingArtifact(artifact_id.to_string())
    })?;
    if let Some(expected) = expected_input {
        if expected != codec.input_dim {
            return Err(CodecError::InputDimMismatch {
                artifact_id: artifact_id.to_string(),
                expected,
                actual: codec.input_dim,
            });
        }
    }
    if residual.len() != codec.input_dim {
        return Err(CodecError::InputDimMismatch {
            artifact_id: artifact_id.to_string(),
            expected: codec.input_dim,
            actual: residual.len(),
        });
    }
    let encoded = codec.apply(residual)?;
    Ok(json!(encoded))
}

/// Decode a Monty output value with a `learned_linear` codec from `output_schema`.
pub fn decode_learned_linear(
    key: &str,
    value: &Value,
    codec_spec: &Value,
    registry: &CodecRegistry,
    hidden_size: usize,
) -> Result<Vec<f32>, CodecError> {
    let artifact_id = codec_spec
        .get("artifact_id")
        .and_then(Value::as_str)
        .ok_or_else(|| CodecError::Invalid {
            artifact_id: "<unknown>".into(),
            detail: format!("{key}: learned_linear codec missing artifact_id"),
        })?;
    let expected_output = codec_spec
        .get("output_dim")
        .and_then(Value::as_u64)
        .map(|n| n as usize);
    let codec = registry.get(artifact_id).ok_or_else(|| {
        CodecError::MissingArtifact(artifact_id.to_string())
    })?;
    if let Some(expected) = expected_output {
        if expected != hidden_size {
            return Err(CodecError::OutputDimMismatch {
                artifact_id: artifact_id.to_string(),
                expected,
                actual: hidden_size,
            });
        }
    }
    if codec.output_dim != hidden_size {
        return Err(CodecError::OutputDimMismatch {
            artifact_id: artifact_id.to_string(),
            expected: hidden_size,
            actual: codec.output_dim,
        });
    }
    let arr = value
        .as_array()
        .ok_or_else(|| CodecError::Invalid {
            artifact_id: artifact_id.to_string(),
            detail: format!("{key}: expected array"),
        })?;
    if arr.len() != codec.input_dim {
        return Err(CodecError::InputDimMismatch {
            artifact_id: artifact_id.to_string(),
            expected: codec.input_dim,
            actual: arr.len(),
        });
    }
    let mut input = Vec::with_capacity(arr.len());
    for item in arr {
        let n = item.as_f64().ok_or_else(|| CodecError::Invalid {
            artifact_id: artifact_id.to_string(),
            detail: format!("{key}: expected numeric array"),
        })?;
        input.push(n as f32);
    }
    codec.apply(&input)
}

/// Write a codec artifact to `<vindex>/call_codecs/<artifact_id>.json`.
pub fn save_codec_artifact(vindex_dir: &Path, artifact: &LearnedLinearCodecArtifact) -> Result<PathBuf, CodecError> {
    use larql_vindex::format::filenames::CALL_CODECS_DIR;

    let dir = vindex_dir.join(CALL_CODECS_DIR);
    std::fs::create_dir_all(&dir).map_err(|e| CodecError::Invalid {
        artifact_id: artifact.artifact_id.clone(),
        detail: format!("create {}: {e}", dir.display()),
    })?;
    let path = dir.join(format!("{}.json", artifact.artifact_id));
    artifact.save(&path)?;
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn identity_codec(id: &str, dim: usize) -> LearnedLinearCodec {
        let mut weights = vec![0.0f32; dim * dim];
        for i in 0..dim {
            weights[i * dim + i] = 1.0;
        }
        LearnedLinearCodec::new(id, dim, dim, weights, vec![0.0; dim]).unwrap()
    }

    fn compress_codec(id: &str, input_dim: usize, output_dim: usize) -> LearnedLinearCodec {
        let mut weights = vec![0.0f32; output_dim * input_dim];
        for row in 0..output_dim {
            weights[row * input_dim + row] = 1.0;
        }
        LearnedLinearCodec::new(id, input_dim, output_dim, weights, vec![0.0; output_dim]).unwrap()
    }

    #[test]
    fn learned_linear_apply_identity() {
        let codec = identity_codec("id", 3);
        let out = codec.apply(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn artifact_roundtrip() {
        let codec = compress_codec("enc_v1", 4, 2);
        let artifact = LearnedLinearCodecArtifact::from_codec(&codec);
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("enc_v1.json");
        artifact.save(&path).unwrap();
        let loaded = LearnedLinearCodecArtifact::load(&path).unwrap().to_codec().unwrap();
        assert_eq!(loaded, codec);
    }

    #[test]
    fn registry_load_from_dir() {
        let dir = TempDir::new().unwrap();
        let codec = compress_codec("enc_v1", 4, 2);
        save_codec_artifact(dir.path(), &LearnedLinearCodecArtifact::from_codec(&codec)).unwrap();
        let registry = load_codec_registry_from_dir(dir.path());
        assert_eq!(registry.len(), 1);
        assert!(registry.get("enc_v1").is_some());
    }

    #[test]
    fn encode_decode_learned_linear_roundtrip() {
        let mut registry = CodecRegistry::new();
        let enc = compress_codec("enc_v1", 4, 2);
        let dec = identity_codec("dec_v1", 4);
        registry.insert(enc);
        registry.insert(dec);

        let residual = vec![5.0f32, -3.0, 1.0, 2.0];
        let spec = json!({
            "kind": "learned_linear",
            "artifact_id": "enc_v1",
            "input_dim": 4,
            "output_dim": 2
        });
        let encoded = encode_learned_linear(&residual, &spec, &registry).unwrap();
        assert_eq!(encoded.as_array().unwrap().len(), 2);

        let out_spec = json!({
            "kind": "learned_linear",
            "artifact_id": "dec_v1",
            "input_dim": 4,
            "output_dim": 4
        });
        let monty_out = json!([0.1, 0.2, 0.3, 0.4]);
        let decoded = decode_learned_linear("delta", &monty_out, &out_spec, &registry, 4).unwrap();
        assert_eq!(decoded, vec![0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn encode_missing_artifact_errors() {
        let registry = CodecRegistry::new();
        let spec = json!({"artifact_id": "missing", "input_dim": 2, "output_dim": 1});
        let err = encode_learned_linear(&[1.0, 2.0], &spec, &registry).unwrap_err();
        assert!(matches!(err, CodecError::MissingArtifact(_)));
    }
}
