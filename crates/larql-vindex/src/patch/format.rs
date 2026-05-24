//! Patch file format — `.vlp` JSON diffs that overlay an immutable
//! base vindex without modifying its files on disk.
//!
//! This module owns the on-the-wire representation: `VindexPatch`,
//! `PatchOp` (Insert/Update/Delete + arch-B InsertKnn/DeleteKnn),
//! `PatchDownMeta`, save/load, and the base64 helpers used to embed
//! gate/key/up/down vectors inside the JSON.
//!
//! `Insert` / `Update` carry up to three optional component vectors —
//! `gate_vector_b64`, `up_vector_b64`, `down_vector_b64`. Compose-mode
//! `INSERT` writes all three so the round-trip
//! `apply_patch` → `COMPILE INTO VINDEX` reproduces the install. The
//! up / down fields are `#[serde(default)]`, so `.vlp` files written
//! before they were introduced still parse with both defaulting to
//! `None`.
//!
//! Runtime application of patches lives in `super::overlay`
//! (`PatchedVindex`).

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::VindexError;

// ═══════════════════════════════════════════════════════════════
// Patch data types
// ═══════════════════════════════════════════════════════════════

/// A vindex patch — a set of operations to apply to a base vindex.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VindexPatch {
    pub version: u32,
    pub base_model: String,
    #[serde(default)]
    pub base_checksum: Option<String>,
    pub created_at: String,
    #[serde(default)]
    pub description: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub operations: Vec<PatchOp>,
}

/// Summary counts for a patch file.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PatchCounts {
    pub inserts: usize,
    pub updates: usize,
    pub deletes: usize,
    pub calls: usize,
}

/// A single patch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum PatchOp {
    Insert {
        layer: usize,
        feature: usize,
        #[serde(default)]
        relation: Option<String>,
        entity: String,
        target: String,
        #[serde(default)]
        confidence: Option<f32>,
        /// Base64-encoded f32 gate vector.
        #[serde(default)]
        gate_vector_b64: Option<String>,
        /// Base64-encoded f32 up vector. Compose-mode INSERT writes a
        /// norm-matched up override alongside gate; persisting it here
        /// lets `apply_patch` reconstruct the install when the .vlp is
        /// reapplied (without it `COMPILE INTO VINDEX` baked nothing).
        #[serde(default)]
        up_vector_b64: Option<String>,
        /// Base64-encoded f32 down vector (column at the inserted slot).
        /// Same rationale as `up_vector_b64`.
        #[serde(default)]
        down_vector_b64: Option<String>,
        #[serde(default)]
        down_meta: Option<PatchDownMeta>,
    },
    Update {
        layer: usize,
        feature: usize,
        #[serde(default)]
        gate_vector_b64: Option<String>,
        #[serde(default)]
        up_vector_b64: Option<String>,
        #[serde(default)]
        down_vector_b64: Option<String>,
        #[serde(default)]
        down_meta: Option<PatchDownMeta>,
    },
    Delete {
        layer: usize,
        feature: usize,
        #[serde(default)]
        reason: Option<String>,
    },
    /// Runtime call patch. The gate vector participates in ordinary
    /// `gate_knn`; the call metadata is retrieved separately by inference
    /// after candidate selection.
    Call(CallPatchOp),
    /// Architecture B: residual-key KNN insert.
    #[serde(rename = "insert_knn")]
    InsertKnn {
        layer: usize,
        entity: String,
        relation: String,
        target: String,
        target_id: u32,
        #[serde(default)]
        confidence: Option<f32>,
        /// Base64-encoded f32 residual key (L2-normalized).
        key_vector_b64: String,
    },
    /// Architecture B: remove all KNN entries for an entity.
    #[serde(rename = "delete_knn")]
    DeleteKnn { entity: String },
}

/// Compact down_meta for a patch operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatchDownMeta {
    #[serde(rename = "t")]
    pub top_token: String,
    #[serde(rename = "i")]
    pub top_token_id: u32,
    #[serde(rename = "c")]
    pub c_score: f32,
}

impl PatchOp {
    /// The (layer, feature) this operation targets. KNN ops return None.
    pub fn key(&self) -> Option<(usize, usize)> {
        match self {
            PatchOp::Insert { layer, feature, .. } => Some((*layer, *feature)),
            PatchOp::Update { layer, feature, .. } => Some((*layer, *feature)),
            PatchOp::Delete { layer, feature, .. } => Some((*layer, *feature)),
            PatchOp::Call(call) => Some((call.layer, call.feature)),
            PatchOp::InsertKnn { .. } | PatchOp::DeleteKnn { .. } => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallPatchOp {
    pub layer: usize,
    pub feature: usize,
    #[serde(default)]
    pub gate_vector_b64: Option<String>,
    pub monty_code: String,
    #[serde(default)]
    pub code_hash: Option<String>,
    #[serde(default)]
    pub input_schema: serde_json::Value,
    #[serde(default)]
    pub output_schema: serde_json::Value,
    #[serde(default)]
    pub trigger: CallTrigger,
    #[serde(default)]
    pub limits: CallResourceLimits,
    #[serde(default)]
    pub safety: CallSafetyPolicy,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CallTrigger {
    #[serde(default)]
    pub score_threshold: Option<f32>,
    #[serde(default)]
    pub margin_threshold: Option<f32>,
    #[serde(default = "default_max_calls_per_token")]
    pub max_calls_per_token: usize,
    #[serde(default)]
    pub max_calls_per_sequence: Option<usize>,
    #[serde(default)]
    pub cooldown_tokens: Option<usize>,
    #[serde(default = "default_require_top_k")]
    pub require_top_k: usize,
}

impl Default for CallTrigger {
    fn default() -> Self {
        Self {
            score_threshold: None,
            margin_threshold: None,
            max_calls_per_token: default_max_calls_per_token(),
            max_calls_per_sequence: None,
            cooldown_tokens: None,
            require_top_k: default_require_top_k(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CallResourceLimits {
    #[serde(default = "default_time_us")]
    pub time_us: u64,
    #[serde(default = "default_memory_bytes")]
    pub memory_bytes: u64,
    #[serde(default = "default_steps")]
    pub steps: u64,
}

impl Default for CallResourceLimits {
    fn default() -> Self {
        Self {
            time_us: default_time_us(),
            memory_bytes: default_memory_bytes(),
            steps: default_steps(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CallSafetyPolicy {
    #[serde(default)]
    pub residual_clamp_norm: Option<f32>,
    #[serde(default = "default_failure_policy")]
    pub failure_policy: String,
}

impl Default for CallSafetyPolicy {
    fn default() -> Self {
        Self {
            residual_clamp_norm: None,
            failure_policy: default_failure_policy(),
        }
    }
}

impl CallPatchOp {
    /// Fill `code_hash` with a stable SHA-256 digest when the patch JSON
    /// omitted it. Existing hashes are preserved so externally signed or
    /// precomputed metadata round-trips unchanged.
    pub fn ensure_code_hash(&mut self) {
        if self.code_hash.is_some() {
            return;
        }
        use sha2::{Digest, Sha256};
        let digest = Sha256::digest(self.monty_code.as_bytes());
        let mut hex = String::with_capacity(digest.len() * 2);
        for byte in digest {
            hex.push_str(&format!("{byte:02x}"));
        }
        self.code_hash = Some(format!("sha256:{hex}"));
    }
}

fn default_max_calls_per_token() -> usize {
    1
}

fn default_require_top_k() -> usize {
    1
}

fn default_time_us() -> u64 {
    250
}

fn default_memory_bytes() -> u64 {
    1_048_576
}

fn default_steps() -> u64 {
    10_000
}

fn default_failure_policy() -> String {
    "ignore_and_continue".into()
}

// ═══════════════════════════════════════════════════════════════
// Patch file I/O
// ═══════════════════════════════════════════════════════════════

impl VindexPatch {
    /// Write patch to a .vlp file.
    pub fn save(&self, path: &Path) -> Result<(), VindexError> {
        let json =
            serde_json::to_string_pretty(self).map_err(|e| VindexError::Parse(e.to_string()))?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load patch from a .vlp file.
    pub fn load(path: &Path) -> Result<Self, VindexError> {
        let text = std::fs::read_to_string(path)?;
        let patch: VindexPatch =
            serde_json::from_str(&text).map_err(|e| VindexError::Parse(e.to_string()))?;
        Ok(patch)
    }

    /// Number of operations in this patch.
    pub fn len(&self) -> usize {
        self.operations.len()
    }

    /// Whether this patch has no operations.
    pub fn is_empty(&self) -> bool {
        self.operations.is_empty()
    }

    /// Summary counts: (inserts, updates, deletes).
    pub fn counts(&self) -> (usize, usize, usize) {
        let counts = self.counts_detailed();
        (counts.inserts, counts.updates, counts.deletes)
    }

    /// Summary counts including runtime call patches.
    pub fn counts_detailed(&self) -> PatchCounts {
        let mut ins = 0;
        let mut upd = 0;
        let mut del = 0;
        let mut calls = 0;
        for op in &self.operations {
            match op {
                PatchOp::Insert { .. } | PatchOp::InsertKnn { .. } => ins += 1,
                PatchOp::Update { .. } => upd += 1,
                PatchOp::Delete { .. } | PatchOp::DeleteKnn { .. } => del += 1,
                PatchOp::Call(_) => calls += 1,
            }
        }
        PatchCounts {
            inserts: ins,
            updates: upd,
            deletes: del,
            calls,
        }
    }
}

/// Load the runtime call-patch sidecar emitted by `COMPILE INTO VINDEX`.
///
/// Returns `Ok(None)` when the sidecar file is absent — compiled vindexes
/// without call patches do not carry one.
pub fn load_runtime_patches_sidecar(vindex_dir: &Path) -> Result<Option<VindexPatch>, VindexError> {
    use crate::format::filenames::RUNTIME_PATCHES_VLP;

    let path = vindex_dir.join(RUNTIME_PATCHES_VLP);
    if !path.exists() {
        return Ok(None);
    }
    VindexPatch::load(&path).map(Some)
}

// ═══════════════════════════════════════════════════════════════
// Base64 gate vector encoding
// ═══════════════════════════════════════════════════════════════

/// Encode a gate vector (f32 slice) as base64 string.
pub fn encode_gate_vector(vec: &[f32]) -> String {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(vec.as_ptr() as *const u8, vec.len() * 4) };
    base64_encode(bytes)
}

/// Decode a base64 string back to f32 vector.
pub fn decode_gate_vector(b64: &str) -> Result<Vec<f32>, VindexError> {
    let bytes = base64_decode(b64)?;
    if bytes.len() % 4 != 0 {
        return Err(VindexError::Parse(
            "gate vector bytes not aligned to f32".into(),
        ));
    }
    let floats: Vec<f32> =
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4) }
            .to_vec();
    Ok(floats)
}

// Simple base64 (no external dependency). Used by `encode_gate_vector`
// and indirectly by patch save / DIFF INTO PATCH.
fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

fn base64_decode(input: &str) -> Result<Vec<u8>, VindexError> {
    fn val(c: u8) -> Result<u32, VindexError> {
        match c {
            b'A'..=b'Z' => Ok((c - b'A') as u32),
            b'a'..=b'z' => Ok((c - b'a' + 26) as u32),
            b'0'..=b'9' => Ok((c - b'0' + 52) as u32),
            b'+' => Ok(62),
            b'/' => Ok(63),
            b'=' => Ok(0),
            _ => Err(VindexError::Parse(format!("invalid base64 char: {c}"))),
        }
    }
    let input = input.as_bytes();
    let mut result = Vec::with_capacity(input.len() * 3 / 4);
    for chunk in input.chunks(4) {
        if chunk.len() < 4 {
            break;
        }
        let a = val(chunk[0])?;
        let b = val(chunk[1])?;
        let c = val(chunk[2])?;
        let d = val(chunk[3])?;
        let triple = (a << 18) | (b << 12) | (c << 6) | d;
        result.push(((triple >> 16) & 0xFF) as u8);
        if chunk[2] != b'=' {
            result.push(((triple >> 8) & 0xFF) as u8);
        }
        if chunk[3] != b'=' {
            result.push((triple & 0xFF) as u8);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    // ── base64 encoding ─────────────────────────────────────────────────

    #[test]
    fn encode_decode_round_trip_single_float() {
        let vec = vec![1.0f32];
        let b64 = encode_gate_vector(&vec);
        let back = decode_gate_vector(&b64).unwrap();
        assert_eq!(back, vec);
    }

    #[test]
    fn encode_decode_round_trip_multi_float() {
        let vec: Vec<f32> = vec![0.0, 1.0, -1.0, 3.25, f32::MAX, f32::MIN_POSITIVE];
        let b64 = encode_gate_vector(&vec);
        let back = decode_gate_vector(&b64).unwrap();
        for (a, b) in vec.iter().zip(back.iter()) {
            assert_eq!(a.to_bits(), b.to_bits(), "bit-exact round-trip required");
        }
    }

    #[test]
    fn decode_rejects_unaligned_bytes() {
        // "YWJj" is base64 for the 3 bytes b"abc".
        // 3 bytes % 4 != 0, so decode_gate_vector must reject it.
        let result = decode_gate_vector("YWJj");
        assert!(
            result.is_err(),
            "3-byte payload should fail alignment check"
        );
    }

    #[test]
    fn decode_rejects_invalid_char() {
        let result = decode_gate_vector("!!!!");
        assert!(result.is_err());
    }

    // ── PatchOp::key ─────────────────────────────────────────────────────

    #[test]
    fn patch_op_key_insert() {
        let op = PatchOp::Insert {
            layer: 3,
            feature: 42,
            relation: None,
            entity: "France".into(),
            target: "Paris".into(),
            confidence: None,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        };
        assert_eq!(op.key(), Some((3, 42)));
    }

    #[test]
    fn patch_op_key_update() {
        let op = PatchOp::Update {
            layer: 5,
            feature: 7,
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        };
        assert_eq!(op.key(), Some((5, 7)));
    }

    #[test]
    fn patch_op_key_delete() {
        let op = PatchOp::Delete {
            layer: 1,
            feature: 0,
            reason: None,
        };
        assert_eq!(op.key(), Some((1, 0)));
    }

    #[test]
    fn patch_op_key_insert_knn_is_none() {
        let op = PatchOp::InsertKnn {
            layer: 0,
            entity: "e".into(),
            relation: "r".into(),
            target: "t".into(),
            target_id: 1,
            confidence: None,
            key_vector_b64: encode_gate_vector(&[1.0, 0.0]),
        };
        assert_eq!(op.key(), None);
    }

    #[test]
    fn patch_op_key_delete_knn_is_none() {
        let op = PatchOp::DeleteKnn { entity: "e".into() };
        assert_eq!(op.key(), None);
    }

    // ── VindexPatch counts / len / is_empty ──────────────────────────────

    fn make_patch(ops: Vec<PatchOp>) -> VindexPatch {
        VindexPatch {
            version: 1,
            base_model: "test".into(),
            base_checksum: None,
            created_at: "2026-01-01T00:00:00Z".into(),
            description: None,
            author: None,
            tags: vec![],
            operations: ops,
        }
    }

    #[test]
    fn empty_patch_counts() {
        let p = make_patch(vec![]);
        assert_eq!(p.len(), 0);
        assert!(p.is_empty());
        assert_eq!(p.counts(), (0, 0, 0));
    }

    #[test]
    fn patch_counts_mixed_ops() {
        let ops = vec![
            PatchOp::Insert {
                layer: 0,
                feature: 0,
                relation: None,
                entity: "A".into(),
                target: "B".into(),
                confidence: None,
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
            PatchOp::Insert {
                layer: 0,
                feature: 1,
                relation: None,
                entity: "C".into(),
                target: "D".into(),
                confidence: None,
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
            PatchOp::Update {
                layer: 0,
                feature: 2,
                gate_vector_b64: None,
                up_vector_b64: None,
                down_vector_b64: None,
                down_meta: None,
            },
            PatchOp::Delete {
                layer: 0,
                feature: 3,
                reason: None,
            },
        ];
        let p = make_patch(ops);
        assert_eq!(p.len(), 4);
        assert!(!p.is_empty());
        assert_eq!(p.counts(), (2, 1, 1));
    }

    #[test]
    fn patch_counts_knn_ops() {
        let kv = encode_gate_vector(&[1.0]);
        let ops = vec![
            PatchOp::InsertKnn {
                layer: 0,
                entity: "e".into(),
                relation: "r".into(),
                target: "t".into(),
                target_id: 1,
                confidence: None,
                key_vector_b64: kv,
            },
            PatchOp::DeleteKnn { entity: "e".into() },
        ];
        let p = make_patch(ops);
        // InsertKnn → insert counter, DeleteKnn → delete counter
        assert_eq!(p.counts(), (1, 0, 1));
    }

    #[test]
    fn patch_counts_track_calls_separately() {
        let p = make_patch(vec![PatchOp::Call(CallPatchOp {
            layer: 0,
            feature: 1,
            gate_vector_b64: None,
            monty_code: "def main(input):\n    return input\n".into(),
            code_hash: None,
            input_schema: serde_json::Value::Null,
            output_schema: serde_json::Value::Null,
            trigger: CallTrigger::default(),
            limits: CallResourceLimits::default(),
            safety: CallSafetyPolicy::default(),
            metadata: serde_json::Value::Null,
        })]);
        assert_eq!(p.counts(), (0, 0, 0));
        assert_eq!(
            p.counts_detailed(),
            PatchCounts {
                inserts: 0,
                updates: 0,
                deletes: 0,
                calls: 1,
            }
        );
    }

    // ── Save / load round-trip ────────────────────────────────────────────

    #[test]
    fn save_load_round_trip() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.vlp");

        let ops = vec![PatchOp::Insert {
            layer: 2,
            feature: 100,
            relation: Some("capital".into()),
            entity: "France".into(),
            target: "Paris".into(),
            confidence: Some(0.95),
            gate_vector_b64: None,
            up_vector_b64: None,
            down_vector_b64: None,
            down_meta: None,
        }];
        let patch = VindexPatch {
            version: 1,
            base_model: "gemma3-4b".into(),
            base_checksum: Some("abc123".into()),
            created_at: "2026-01-01T00:00:00Z".into(),
            description: Some("test patch".into()),
            author: Some("test".into()),
            tags: vec!["geography".into()],
            operations: ops,
        };

        patch.save(&path).unwrap();
        let loaded = VindexPatch::load(&path).unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.base_model, "gemma3-4b");
        assert_eq!(loaded.tags, vec!["geography"]);
        assert_eq!(loaded.operations.len(), 1);
    }

    #[test]
    fn load_missing_file_returns_error() {
        let result = VindexPatch::load(std::path::Path::new("/nonexistent/path.vlp"));
        assert!(result.is_err());
    }

    #[test]
    fn save_load_round_trip_preserves_gate_up_down_vectors() {
        // Compose-mode INSERT writes all three vectors; the .vlp must
        // round-trip them. Regression for the lossy-patch bug where only
        // gate was serialised and re-applying the file dropped up + down.
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("with_vectors.vlp");

        let gate = vec![1.0f32, 0.5, -0.5];
        let up = vec![0.1f32, 0.2, 0.3];
        let down = vec![-1.0f32, 0.0, 1.0];
        let ops = vec![PatchOp::Insert {
            layer: 7,
            feature: 13,
            relation: Some("capital".into()),
            entity: "France".into(),
            target: "Paris".into(),
            confidence: Some(0.9),
            gate_vector_b64: Some(encode_gate_vector(&gate)),
            up_vector_b64: Some(encode_gate_vector(&up)),
            down_vector_b64: Some(encode_gate_vector(&down)),
            down_meta: None,
        }];
        let patch = VindexPatch {
            version: 1,
            base_model: "test".into(),
            base_checksum: None,
            created_at: String::new(),
            description: None,
            author: None,
            tags: vec![],
            operations: ops,
        };
        patch.save(&path).unwrap();
        let loaded = VindexPatch::load(&path).unwrap();
        match &loaded.operations[0] {
            PatchOp::Insert {
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } => {
                assert_eq!(
                    decode_gate_vector(gate_vector_b64.as_ref().unwrap()).unwrap(),
                    gate
                );
                assert_eq!(
                    decode_gate_vector(up_vector_b64.as_ref().unwrap()).unwrap(),
                    up
                );
                assert_eq!(
                    decode_gate_vector(down_vector_b64.as_ref().unwrap()).unwrap(),
                    down
                );
            }
            _ => panic!("expected Insert"),
        }
    }

    #[test]
    fn load_legacy_patch_without_up_down_vectors() {
        // .vlp files written before up_vector_b64 / down_vector_b64 were
        // added must still parse — both fields default to None. This
        // pins the backward-compatibility contract: removing
        // `#[serde(default)]` on either field would silently break
        // existing patch files.
        let json = r#"{
          "version": 1,
          "base_model": "test",
          "created_at": "2026-01-01",
          "operations": [
            {
              "op": "insert",
              "layer": 0,
              "feature": 1,
              "entity": "France",
              "target": "Paris",
              "gate_vector_b64": null
            }
          ]
        }"#;
        let patch: VindexPatch = serde_json::from_str(json).unwrap();
        match &patch.operations[0] {
            PatchOp::Insert {
                gate_vector_b64,
                up_vector_b64,
                down_vector_b64,
                ..
            } => {
                assert!(gate_vector_b64.is_none());
                assert!(
                    up_vector_b64.is_none(),
                    "missing up_vector_b64 should default to None"
                );
                assert!(
                    down_vector_b64.is_none(),
                    "missing down_vector_b64 should default to None"
                );
            }
            _ => panic!("expected Insert"),
        }
    }

    #[test]
    fn call_patch_json_round_trips() {
        let gate = vec![1.0f32, -2.0, 3.5];
        let patch = VindexPatch {
            version: 1,
            base_model: "test".into(),
            base_checksum: None,
            created_at: "2026-01-01".into(),
            description: None,
            author: None,
            tags: vec![],
            operations: vec![PatchOp::Call(CallPatchOp {
                layer: 2,
                feature: 7,
                gate_vector_b64: Some(encode_gate_vector(&gate)),
                monty_code: "def main(input):\n    return input\n".into(),
                code_hash: Some("sha256:test".into()),
                input_schema: serde_json::json!({"sources": ["current_residual"]}),
                output_schema: serde_json::json!({"sinks": ["residual_delta"]}),
                trigger: CallTrigger {
                    score_threshold: Some(12.0),
                    ..Default::default()
                },
                limits: CallResourceLimits::default(),
                safety: CallSafetyPolicy {
                    failure_policy: "ignore".into(),
                    ..Default::default()
                },
                metadata: serde_json::json!({"name": "identity"}),
            })],
        };

        let json = serde_json::to_string(&patch).unwrap();
        let loaded: VindexPatch = serde_json::from_str(&json).unwrap();
        match &loaded.operations[0] {
            PatchOp::Call(call) => {
                assert_eq!(call.layer, 2);
                assert_eq!(call.feature, 7);
                assert_eq!(
                    decode_gate_vector(call.gate_vector_b64.as_ref().unwrap()).unwrap(),
                    gate
                );
                assert_eq!(call.trigger.score_threshold, Some(12.0));
                assert_eq!(call.limits.time_us, 250);
            }
            _ => panic!("expected Call"),
        }
    }

    #[test]
    fn call_patch_ensure_code_hash_is_stable() {
        let mut call = CallPatchOp {
            layer: 0,
            feature: 0,
            gate_vector_b64: None,
            monty_code: "def main(input):\n    return input\n".into(),
            code_hash: None,
            input_schema: serde_json::Value::Null,
            output_schema: serde_json::Value::Null,
            trigger: CallTrigger::default(),
            limits: CallResourceLimits::default(),
            safety: CallSafetyPolicy::default(),
            metadata: serde_json::Value::Null,
        };

        call.ensure_code_hash();
        let first = call.code_hash.clone().unwrap();
        call.ensure_code_hash();
        assert_eq!(call.code_hash.as_deref(), Some(first.as_str()));
        assert!(first.starts_with("sha256:"));
        assert_eq!(first.len(), "sha256:".len() + 64);
    }

    #[test]
    fn load_runtime_patches_sidecar_missing_returns_none() {
        let dir = TempDir::new().unwrap();
        let loaded = load_runtime_patches_sidecar(dir.path()).unwrap();
        assert!(loaded.is_none());
    }

    #[test]
    fn load_runtime_patches_sidecar_loads_present_file() {
        let dir = TempDir::new().unwrap();
        let sidecar_path = dir.path().join(crate::format::filenames::RUNTIME_PATCHES_VLP);
        let patch = VindexPatch {
            version: 1,
            base_model: String::new(),
            base_checksum: None,
            created_at: "compiled".into(),
            description: None,
            author: None,
            tags: vec![],
            operations: vec![PatchOp::Call(CallPatchOp {
                layer: 1,
                feature: 2,
                gate_vector_b64: None,
                monty_code: "def main(input):\n    return input\n".into(),
                code_hash: None,
                input_schema: serde_json::Value::Null,
                output_schema: serde_json::Value::Null,
                trigger: CallTrigger::default(),
                limits: CallResourceLimits::default(),
                safety: CallSafetyPolicy::default(),
                metadata: serde_json::Value::Null,
            })],
        };
        patch.save(&sidecar_path).unwrap();
        let loaded = load_runtime_patches_sidecar(dir.path())
            .unwrap()
            .expect("sidecar should load");
        assert_eq!(loaded.counts_detailed().calls, 1);
    }
}
