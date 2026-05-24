//! ATTACH CALL demo — attaches a trivial runtime call patch, saves and re-applies it.
//!
//! Shows the complete call-patch lifecycle in LQL:
//!   1. Create a synthetic vindex and open a session.
//!   2. Write a minimal call-patch JSON file.
//!   3. ATTACH CALL FROM FILE — registers the call at (layer=0, feature=1).
//!   4. Verify the patch is accessible via the overlay.
//!   5. BEGIN PATCH → SAVE PATCH → verify the saved .vlp contains op="call".
//!   6. APPLY PATCH from the saved file to a fresh session — verify round-trip.
//!   7. Show that COMPILE INTO MODEL rejects the call patch.
//!
//! Run: cargo run -p larql-lql --example call_patch_demo

use larql_lql::{parse, Session};
use larql_vindex::patch::core::encode_gate_vector;
use std::path::Path;
use tempfile::TempDir;

fn main() {
    let dir = TempDir::new().expect("temp dir");
    println!("=== ATTACH CALL Demo ===\n");

    // ── 1. Build a synthetic vindex ──────────────────────────────────────────
    let vindex_dir = dir.path().join("demo.vindex");
    build_synthetic_vindex(&vindex_dir);

    // ── 2. Write a call-patch JSON ──────────────────────────────────────────
    let patch_path = dir.path().join("normalize.json");
    write_call_patch(&patch_path);
    println!("Wrote call patch to: {}", patch_path.display());

    // ── 3. Open a session and attach the call patch ─────────────────────────
    let mut session = Session::new();
    exec(
        &mut session,
        &format!(r#"USE "{}";"#, vindex_dir.display()),
        "USE vindex",
    );
    exec(
        &mut session,
        &format!(r#"ATTACH CALL FROM FILE "{}";"#, patch_path.display()),
        "ATTACH CALL",
    );

    // ── 4. Verify the overlay has the call patch ────────────────────────────
    {
        let overlay = session
            .patched_overlay_mut()
            .expect("patched overlay accessible");
        let call = overlay
            .call_patch(0, 1)
            .expect("call patch registered at L0 F1");
        println!(
            "\nCall patch at L{} F{}: {} bytes of monty code, code_hash={:?}",
            call.layer,
            call.feature,
            call.monty_code.len(),
            call.code_hash
        );
        assert!(
            overlay.overrides_gate_at(0, 1).is_some(),
            "gate vector registered"
        );
    }

    // ── 5. Save and reload ──────────────────────────────────────────────────
    // Start a new session for the save/apply cycle so the auto-patch from
    // step 3 doesn't interfere with the recording.
    let mut save_session = Session::new();
    exec(
        &mut save_session,
        &format!(r#"USE "{}";"#, vindex_dir.display()),
        "USE vindex (save session)",
    );
    let vlp_path = dir.path().join("tools.vlp");
    exec(
        &mut save_session,
        &format!(r#"BEGIN PATCH "{}";"#, vlp_path.display()),
        "BEGIN PATCH",
    );
    exec(
        &mut save_session,
        &format!(r#"ATTACH CALL FROM FILE "{}";"#, patch_path.display()),
        "ATTACH CALL (recording)",
    );
    exec(&mut save_session, "SAVE PATCH;", "SAVE PATCH");

    let saved = larql_vindex::VindexPatch::load(&vlp_path).expect("load saved patch");
    println!("\nSaved patch: {} operations", saved.operations.len());
    let counts = saved.counts_detailed();
    println!(
        "  inserts={}, updates={}, deletes={}, calls={}",
        counts.inserts, counts.updates, counts.deletes, counts.calls
    );
    assert_eq!(counts.calls, 1, "saved patch should contain 1 call op");
    println!("  ✓ saved .vlp contains op=\"call\"");

    // ── 6. Apply the saved patch in a fresh session ─────────────────────────
    let mut session2 = Session::new();
    exec(
        &mut session2,
        &format!(r#"USE "{}";"#, vindex_dir.display()),
        "USE vindex (fresh session)",
    );
    exec(
        &mut session2,
        &format!(r#"APPLY PATCH "{}";"#, vlp_path.display()),
        "APPLY PATCH",
    );
    {
        let overlay2 = session2.patched_overlay_mut().expect("overlay");
        assert!(
            overlay2.call_patch(0, 1).is_some(),
            "APPLY PATCH should rehydrate call at L0 F1"
        );
        println!("  ✓ APPLY PATCH round-trip: call patch rehydrated");
    }

    // ── 7. COMPILE INTO MODEL rejects the call patch ────────────────────────
    let model_out = dir.path().join("out.safetensors");
    let stmt = parse(&format!(
        r#"COMPILE CURRENT INTO MODEL "{}";"#,
        model_out.display()
    ))
    .expect("parse COMPILE INTO MODEL");
    let result = session.execute(&stmt);
    assert!(
        result.is_err(),
        "COMPILE INTO MODEL should fail when call patches are present"
    );
    println!("  ✓ COMPILE INTO MODEL correctly rejected call patch");

    println!("\n=== Demo complete ===");
}

fn build_synthetic_vindex(dir: &Path) {
    use larql_models::TopKEntry;
    use larql_vindex::{FeatureMeta, VectorIndex, VindexConfig};
    use ndarray::Array2;

    let hidden = 4;
    let num_features = 3;
    let num_layers = 2;

    let gate = Array2::<f32>::from_shape_vec(
        (num_features, hidden),
        vec![
            1.0, 0.0, 0.0, 0.0, // feature 0
            0.0, 1.0, 0.0, 0.0, // feature 1
            0.0, 0.0, 1.0, 0.0, // feature 2
        ],
    )
    .unwrap();

    let meta_entry = |tok: &str, id: u32| {
        Some(FeatureMeta {
            top_token: tok.into(),
            top_token_id: id,
            c_score: 0.9,
            top_k: vec![TopKEntry {
                token: tok.into(),
                token_id: id,
                logit: 0.9,
            }],
        })
    };

    let index = VectorIndex::new(
        vec![Some(gate.clone()), Some(gate.clone())],
        vec![
            Some(vec![
                meta_entry("A", 1),
                meta_entry("B", 2),
                meta_entry("C", 3),
            ]),
            Some(vec![
                meta_entry("D", 4),
                meta_entry("E", 5),
                meta_entry("F", 6),
            ]),
        ],
        num_layers,
        hidden,
    );

    let vocab_size = 32;
    let mut config = VindexConfig {
        version: 1,
        model: "test/call-patch-demo".into(),
        family: "test".into(),
        source: None,
        checksums: None,
        num_layers,
        hidden_size: hidden,
        intermediate_size: num_features,
        vocab_size,
        embed_scale: 1.0,
        extract_level: larql_vindex::ExtractLevel::Browse,
        dtype: larql_vindex::StorageDtype::F32,
        quant: larql_vindex::QuantFormat::None,
        layer_bands: None,
        layers: Vec::new(),
        down_top_k: 5,
        has_model_weights: false,
        model_config: None,
        fp4: None,
        ffn_layout: None,
    };
    std::fs::create_dir_all(dir).expect("create vindex dir");
    index.save_vindex(dir, &mut config).unwrap();

    let embed_bytes = vec![0u8; vocab_size * hidden * 4];
    std::fs::write(dir.join("embeddings.bin"), embed_bytes).unwrap();
    let tok_json =
        r#"{"version":"1.0","model":{"type":"BPE","vocab":{},"merges":[]},"added_tokens":[]}"#;
    std::fs::write(dir.join("tokenizer.json"), tok_json).unwrap();
}

fn write_call_patch(path: &Path) {
    let gate_b64 = encode_gate_vector(&[0.0f32, 1.0, 0.0, 0.0]);
    let json = serde_json::json!({
        "op": "call",
        "layer": 0,
        "feature": 1,
        "gate_vector_b64": gate_b64,
        "monty_code": "def main(input):\n    delta = [x * 0.01 for x in input['residual']]\n    return {'residual_delta': delta}\n",
        "trigger": {
            "score_threshold": null,
            "max_calls_per_token": 1,
            "require_top_k": 1
        },
        "limits": {"time_us": 250, "memory_bytes": 1048576, "steps": 10000},
        "safety": {"residual_clamp_norm": 2.0},
        "metadata": {"description": "scale residual by 0.01"}
    });
    std::fs::write(path, serde_json::to_string_pretty(&json).unwrap()).unwrap();
}

fn exec(session: &mut Session, lql: &str, label: &str) {
    let stmt = parse(lql).unwrap_or_else(|e| panic!("parse {label}: {e}"));
    match session.execute(&stmt) {
        Ok(out) => {
            for line in &out {
                println!("  {line}");
            }
        }
        Err(e) => eprintln!("  [error] {label}: {e}"),
    }
}
