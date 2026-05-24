//! Call-patch round-trip — encode a CallPatchOp into .vlp, reload, and verify.
//!
//! Demonstrates the full call-patch data-layer lifecycle without any LQL or
//! inference engine involvement:
//!
//!   1. Build a PatchedVindex with an in-memory call patch.
//!   2. Serialize to a VindexPatch (.vlp JSON) and save to disk.
//!   3. Load the file back and verify the Call op round-trips.
//!   4. Apply the loaded patch to a fresh PatchedVindex and verify the
//!      overlay state (gate_knn includes the call gate; call_patch returns
//!      the correct op).
//!
//! Run: cargo run -p larql-vindex --example call_patch_roundtrip

use larql_vindex::{
    patch::core::{decode_gate_vector, encode_gate_vector},
    CallPatchOp, CallResourceLimits, CallSafetyPolicy, CallTrigger, PatchOp, PatchedVindex,
    VectorIndex, VindexPatch,
};
use ndarray::Array1;
use tempfile::TempDir;

fn main() {
    let dir = TempDir::new().expect("temp dir");
    println!("=== Call-Patch Round-Trip Demo ===\n");

    // ── 1. Build a base vindex and attach a call patch ───────────────────────
    let hidden = 8;
    let base = VectorIndex::new(vec![None, None], vec![None, None], 2, hidden);
    let mut patched = PatchedVindex::new(base);

    let gate_vec: Vec<f32> = (0..hidden).map(|i| if i == 0 { 1.0 } else { 0.0 }).collect();
    let call = CallPatchOp {
        layer: 0,
        feature: 42,
        gate_vector_b64: Some(encode_gate_vector(&gate_vec)),
        monty_code: "def main(input):\n    delta = [x * 0.1 for x in input['residual']]\n    return {'residual_delta': delta}\n".into(),
        code_hash: None,
        input_schema: serde_json::json!({"sources": [{"kind": "current_residual", "key": "residual"}]}),
        output_schema: serde_json::json!({"sinks": [{"kind": "residual_delta", "key": "residual_delta", "scale": 1.0}]}),
        trigger: CallTrigger {
            score_threshold: Some(0.5),
            margin_threshold: Some(0.1),
            max_calls_per_token: 1,
            require_top_k: 2,
            ..Default::default()
        },
        limits: CallResourceLimits {
            time_us: 250,
            memory_bytes: 1_048_576,
            steps: 10_000,
        },
        safety: CallSafetyPolicy {
            residual_clamp_norm: Some(2.0),
            ..Default::default()
        },
        metadata: serde_json::json!({"description": "scale residual by 0.1", "author": "demo"}),
    };

    patched.insert_call_patch(call.clone(), gate_vec.clone());

    // Verify the overlay before serialization.
    assert!(patched.call_patch(0, 42).is_some(), "call patch registered");
    assert!(
        patched.overrides_gate_at(0, 42).is_some(),
        "gate vector registered"
    );
    println!("✓ Call patch inserted into overlay at L0 F42");

    // gate_knn should surface the call feature.
    let q = Array1::from_vec(gate_vec.clone());
    let hits = patched.gate_knn(0, &q, 1);
    assert_eq!(hits[0].0, 42, "call gate should score at rank 1");
    println!("✓ gate_knn returns call feature at rank 1 (score={:.3})", hits[0].1);

    // ── 2. Serialize the overlay into a VindexPatch and save ─────────────────
    let mut call_with_hash = call.clone();
    call_with_hash.ensure_code_hash();
    let patch = VindexPatch {
        version: 1,
        base_model: "test/roundtrip-demo".into(),
        base_checksum: None,
        created_at: "2026-01-01T00:00:00Z".into(),
        description: Some("call patch demo".into()),
        author: Some("demo".into()),
        tags: vec!["call".into()],
        operations: vec![PatchOp::Call(call_with_hash)],
    };

    let vlp_path = dir.path().join("tools.vlp");
    patch.save(&vlp_path).expect("save .vlp");
    println!("✓ Saved .vlp to {}", vlp_path.display());

    // ── 3. Load the .vlp and verify round-trip ───────────────────────────────
    let loaded = VindexPatch::load(&vlp_path).expect("load .vlp");
    let counts = loaded.counts_detailed();
    assert_eq!(counts.calls, 1, "loaded patch should have 1 call op");
    println!("✓ Loaded .vlp: {} op(s), calls={}", loaded.len(), counts.calls);

    let PatchOp::Call(ref loaded_call) = loaded.operations[0] else {
        panic!("expected Call op");
    };
    assert_eq!(loaded_call.layer, 0);
    assert_eq!(loaded_call.feature, 42);
    assert_eq!(
        loaded_call.trigger.score_threshold,
        Some(0.5),
        "trigger round-trips"
    );
    assert_eq!(
        loaded_call.limits.time_us, 250,
        "limits round-trip"
    );
    assert_eq!(
        loaded_call.safety.residual_clamp_norm,
        Some(2.0),
        "safety round-trips"
    );
    let code_hash = loaded_call.code_hash.as_deref().unwrap_or("");
    assert!(code_hash.starts_with("sha256:"), "code_hash present");
    println!("✓ All fields round-trip correctly (code_hash={code_hash:.20}…)");

    let loaded_gate = decode_gate_vector(loaded_call.gate_vector_b64.as_ref().unwrap())
        .expect("decode gate");
    assert_eq!(
        loaded_gate.len(),
        hidden,
        "gate vector width preserved"
    );
    assert!(
        (loaded_gate[0] - 1.0).abs() < 1e-6,
        "gate vector values preserved"
    );
    println!("✓ gate_vector_b64 decodes to the original vector");

    // ── 4. Apply the loaded patch to a fresh PatchedVindex ───────────────────
    let base2 = VectorIndex::new(vec![None, None], vec![None, None], 2, hidden);
    let mut patched2 = PatchedVindex::new(base2);
    patched2.apply_patch(loaded);

    let call2 = patched2.call_patch(0, 42).expect("call rehydrated");
    assert_eq!(call2.layer, 0);
    assert_eq!(call2.feature, 42);

    let hits2 = patched2.gate_knn(0, &q, 1);
    assert_eq!(hits2[0].0, 42, "gate participates in KNN after apply_patch");
    println!("✓ APPLY round-trip: call_patch at L0 F42, gate_knn rank 1 = F{}", hits2[0].0);

    println!("\n=== Round-trip complete ===");
}
