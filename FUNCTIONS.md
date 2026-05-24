Plan: Monty-call patches for LARQL

Current status snapshot

This document is now split into two readings:

* **Already done**: the repository has the schema, patch overlay, docs, LQL
  file-backed attach flow, and an opt-in inference-side execution seam for
  call patches.
* **Remaining**: the actual Monty VM runner, broader inference-path coverage,
  production safety hardening, public metrics/trace surfacing, benchmarks, and
  the training pipeline.

Already implemented in-tree

* Documentation:
    * [docs/monty-call-patches.md](docs/monty-call-patches.md) exists and
      states the runtime contract, patch JSON shape, LQL surface, compile
      behavior, and current implementation status.
    * [docs/specs.md](docs/specs.md), [docs/inference-engine.md](docs/inference-engine.md),
      [docs/lql-guide.md](docs/lql-guide.md), and
      [crates/larql-vindex/docs/operations-spec.md](crates/larql-vindex/docs/operations-spec.md)
      reference call patches.
* `larql-vindex`:
    * `PatchOp::Call(CallPatchOp)` is implemented in
      [crates/larql-vindex/src/patch/format.rs](crates/larql-vindex/src/patch/format.rs).
    * `CallPatchOp`, `CallTrigger`, `CallResourceLimits`, `CallSafetyPolicy`,
      and `PatchCounts.calls` are implemented with serde support.
    * `PatchedVindex` stores call metadata, inserts call gate vectors into the
      overlay, exposes `call_patch(layer, feature)` / `call_patches_for_layer`,
      and applies call ops through the patch overlay.
    * `crates/larql-vindex/examples/call_patch_roundtrip.rs` demonstrates
      call patch save/load and overlay lookup.
* `larql-lql`:
    * `ATTACH CALL FROM FILE "..."` is implemented and tested.
    * Inline `ATTACH CALL ... GATE VECTOR FROM FILE ... MONTY CODE FROM FILE ...`
      is implemented and tested for the current vertical slice.
    * File-backed attach auto-starts a patch session, validates the gate vector
      against hidden size, fills `code_hash` when absent, and persists the call
      op in `.vlp`.
    * `COMPILE INTO MODEL` rejects runtime call patches.
    * `COMPILE INTO VINDEX` writes call patches to `runtime_patches.vlp` by
      default, and `COMPILE INTO VINDEX STATIC_ONLY` rejects them.
* `larql-inference`:
    * [crates/larql-inference/src/monty_call/mod.rs](crates/larql-inference/src/monty_call/mod.rs)
      implements trigger checks, JSON input encoding, output decoding, residual
      clamping, sparse logit-bias decoding, metrics, and
      `MontyCallRuntime<R: CallProgramRunner>`.
    * `MontyVmRunner` executes call patch code through the `pydantic_monty`
      Python package, with `LARQL_MONTY_PYTHON` selecting the interpreter when
      needed. The unit smoke test runs when `pydantic_monty` is importable and
      skips cleanly otherwise.
    * `WalkFfn` has opt-in `with_call_patches(...)` and
      `with_call_runtime(...)` hooks. Selected call features are looked up
      after gate selection, executed through the installed runtime, and skipped
      as static FFN rows.
    * Sparse CPU `WalkFfn` now has synthetic coverage for selected call patches
      firing, non-fired triggers leaving next-token logits unchanged, and
      loaded-without-runtime call patches being skipped instead of treated as
      static rows.
    * Public prediction helpers `predict_with_call_patches(...)` and
      `predict_with_call_patches_runner(...)` thread `PatchedVindex` +
      Monty call runtime hooks into `WalkFfn` and return
      `PredictResultWithCallMetrics` so callers can observe attempted/fired/
      skipped/failed/timeout counters without constructing `WalkFfn` directly.

Still remaining

* Extend execution beyond the sparse CPU `WalkFfn` path:
    * dense/static FFN paths
    * Metal/GPU paths
    * full mmap/kquant paths where applicable
    * batched prefill and generation loop integration
* Expand inline LQL beyond the current file-backed/gate-code vertical slice if
  richer `INPUT (...)`, `OUTPUT (...)`, and policy grammar proves necessary.
* Harden runtime policy:
    * cooldowns and per-sequence budgets
    * end-to-end timeout/memory limits using Monty resource tracking
    * trace/metrics surfacing through public inference APIs
    * false-fire and residual-explosion tests
* Benchmark the no-call, loaded-but-not-fired, and fired-call paths.
* Build the codec roadmap beyond the current raw residual / top-k basis /
  sparse basis delta / sparse logit-bias support.
* Implement the training prototype and later research-grade training loop.

Recommended next milestone

The next highest-leverage step is to thread call-patch lookup/runtime hooks
through a public inference entry point so callers can observe call metrics and
trace events without constructing `WalkFfn` directly.

Reading note

The detailed phase plan below is preserved as the original implementation map.
Where it says "proposed" for items now listed above as implemented, treat the
status snapshot as authoritative.

Source verification status

Confirmed from current public repo/docs:

LARQL is Apache-2.0 licensed; Monty is MIT licensed. MIT code can be incorporated as a dependency in an Apache-2.0 project with normal notice/license preservation.  ￼  ￼

LARQL’s documented architecture has larql-vindex, larql-lql, larql-inference, larql-kv, larql-server, larql-cli, plus portable model-compute. larql-vindex owns vindex lifecycle and patch overlay; larql-lql owns parser/executor/REPL; larql-inference owns the forward pass and WalkFfn.  ￼

Confirmed LARQL symbols / paths from docs or source browser:

* crates/larql-vindex/src/
    * directories: clustering, config, engine, extract, format, index, patch, quant, vindexfile
    * files: describe.rs, error.rs, lib.rs, mmap_util.rs  ￼
* crates/larql-lql/src/
    * directories: executor, parser
    * files: ast.rs, error.rs, lexer.rs, lib.rs, relations.rs, repl.rs  ￼
* crates/larql-inference/src/
    * directories: attention, ffn, forward, trace, vindex, etc.
    * files include forward_overrides.rs, model.rs, tokenizer.rs, residual.rs, lib.rs  ￼
* VectorIndex::load_vindex, PatchedVindex::new, PatchedVindex::insert_feature, apply_patch, gate_knn, walk, GateIndex, FeatureMeta, WalkTrace, WalkHit, bake_down are documented.  ￼
* .vlp is documented as JSON with version, base metadata, and operations containing op, layer, feature, gate_vector_b64, and down_meta.  ￼
* COMPILE INTO VINDEX flattens patches into a standalone vindex; COMPILE INTO MODEL exports plain model weights via MEMIT-style edits and emits standard safetensors/GGUF with no special loader.  ￼
* larql-inference has WalkFfn, predict_with_ffn(), trace_forward(), trace_residuals(), and a LayerHook surface with on_post_attention, on_ffn_activation, and on_post_layer.  ￼  ￼

Confirmed Monty symbols / paths:

* repo contains Rust crate at crates/monty and Python package usage as pydantic_monty; top-level language mix is Rust/Python/TypeScript.  ￼
* crates/monty/src/ contains lib.rs, run.rs, run_progress.rs, object.rs, object_json.rs, resource.rs, etc.  ￼
* public exports include MontyRun, MontyObject, DictPairs, JsonMontyObject, JsonMontyPairs, ResourceLimits, LimitedTracker, NoLimitTracker, PrintWriter, RunProgress, FunctionCall, etc.  ￼
* MontyRun::new, run, run_no_limits, dump, load, and start are confirmed.  ￼
* MontyObject includes Dict(DictPairs), List, Tuple, Int, Float, String, Bytes, etc.; JSON-facing wrappers exist in object_json.  ￼
* Monty supports deterministic sandboxing/resource limits, no host FS/network/env except controlled external calls, snapshot/resume, and microsecond startup claims in README.  ￼

Locally confirmed in this repository since the original plan was written:

* `PatchOp`, `VindexPatch`, `CallPatchOp`, `CallTrigger`, `CallResourceLimits`,
  and `CallSafetyPolicy` live in
  [crates/larql-vindex/src/patch/format.rs](crates/larql-vindex/src/patch/format.rs).
* `PatchedVindex` and call-patch overlay storage live in
  [crates/larql-vindex/src/patch/overlay.rs](crates/larql-vindex/src/patch/overlay.rs),
  with apply behavior in
  [crates/larql-vindex/src/patch/overlay_apply.rs](crates/larql-vindex/src/patch/overlay_apply.rs).
* `GateIndex` is defined in
  [crates/larql-vindex/src/index/types/ffn_row/mod.rs](crates/larql-vindex/src/index/types/ffn_row/mod.rs).
* `WalkFfn` is defined in
  [crates/larql-inference/src/vindex/walk_ffn/mod.rs](crates/larql-inference/src/vindex/walk_ffn/mod.rs)
  and now carries optional call-patch lookup/runtime hooks.
* The inference call runtime seam lives in
  [crates/larql-inference/src/monty_call/mod.rs](crates/larql-inference/src/monty_call/mod.rs).
* `LayerHook` is defined under `larql-inference/src/forward/hooks`.
* `ATTACH CALL` executor tests live in
  [crates/larql-lql/src/executor/tests.rs](crates/larql-lql/src/executor/tests.rs).

Recently completed (M5 safety hardening, partial):

* `MontyCallRuntime` now enforces `max_calls_per_sequence` and
  `cooldown_tokens` from `CallTrigger` — both fields were previously defined
  in the schema but ignored at runtime.
* `MontyCallMetrics` gained `skipped_sequence_budget` and `skipped_cooldown`
  counters.
* Per-event trace surface: `CallOutcome` + `CallTraceEvent` types; enable with
  `MontyCallRuntime::with_trace_events()`, drain with `take_trace_events()`.
* `reset_sequence_state()` on `MontyCallRuntime` resets per-sequence fire
  counts and cooldown positions between inference passes / generation sequences.
* New tests: sequence budget enforcement, budget reset, cooldown enforcement,
  trace event emission (Fired / SkippedTrigger / SkippedSequenceBudget /
  SkippedCooldown), drain semantics, tracing-disabled no-op.

Still incomplete:

* Dense/static FFN paths, Metal/GPU paths, full mmap/kquant paths, and batched
  prefill/generation loop integration — call patches only execute on the
  sparse CPU `WalkFfn` path today.
* False-fire and residual-explosion tests (residual clamping is already
  tested; dedicated named tests for these safety scenarios are not yet written).

⸻

Phase 0 — Define the feature contract before touching runtime code

Milestone 0.1: Call patch semantics

Add a new dynamic patch operation conceptually equivalent to:

PatchOp::Call {
    layer: usize,
    feature: usize,
    gate_vector: Vec<f32>,
    monty_code: String,
    input_schema: CallInputSchema,
    output_schema: CallOutputSchema,
    trigger: CallTrigger,
    limits: CallResourceLimits,
    safety: CallSafetyPolicy,
    metadata: CallPatchMetadata,
}

This is a proposed interface, not confirmed existing code.

Semantics:

1. gate_knn(layer, residual, k) still computes gate candidates from the base and overlay.
2. If a selected candidate is a Call op, inference does not read down_meta as a static token direction.
3. Instead, larql-inference builds an input dict from residual/context sources, runs Monty, decodes the output dict, and applies one or more sinks:
    * residual additive delta at (layer, position)
    * optional FFN/pre-down activation modification
    * optional logit bias for current position
    * optional engine scratch/state for outer-loop multi-token execution

Milestone 0.2: State hard constraints explicitly in the design doc

Add docs/specs/monty-call-patches.md and update docs/specs/vindex-operations-spec.md, docs/specs/lql-spec.md, and docs/inference-engine.md.

The design doc must state:

* Non-differentiable oracle: Monty execution is opaque. No gradient passes through Python code.
* Next-token reach: A layer/position call can directly affect only the current forward pass and therefore the current next-token distribution. Multi-token effects require the generation loop to re-run calls at later positions.
* Non-portability: Call patches are runtime overlay artifacts. They are not representable in vanilla safetensors and must be rejected or stripped by COMPILE INTO MODEL.
* Firing safety: gate thresholding, cooldowns, execution budgets, residual norm clamps, and hard-negative calibration are required.
* dict↔vector impedance: input/output encoders are explicit learned or hand-authored adapters, not magic serialization.
* Latency/purity: gate_knn remains a pure candidate selector; actual calls happen after candidate selection in a hookable FFN path, breaking pure batched BLAS assumptions only for fired features.

Tests for Phase 0

* Spec examples validate as JSON.
* .vlp compatibility doc test: legacy insert/update/delete patch examples still parse.
* Compile behavior spec tests:
    * COMPILE INTO MODEL with Call returns an explicit unsupported error.
    * COMPILE INTO VINDEX either preserves Call as sidecar overlay metadata or rejects unless --include-runtime-patches is used.

⸻

A. Architecture & code changes, per crate

Phase 1 — larql-vindex: patch schema and overlay representation

Confirmed integration point

larql-vindex owns load/query/mutate/patch/save, and both VectorIndex and PatchedVindex implement GateIndex; docs show PatchedVindex carries patches: Vec<VindexPatch> and overrides: HashMap<(usize, usize), PatchOp>.  ￼

File-level changes

Confirmed directories:

* crates/larql-vindex/src/patch/
* crates/larql-vindex/src/index/
* crates/larql-vindex/src/lib.rs
* docs/specs/vindex-operations-spec.md  ￼

Proposed files/modules:

* crates/larql-vindex/src/patch/call.rs
* crates/larql-vindex/src/patch/schema.rs
* update existing patch module that defines PatchOp and VindexPatch
* update existing PatchedVindex implementation module
* update lib.rs re-exports

Exact current file that defines PatchOp is unverified.

Proposed public types

pub enum PatchOp {
    Insert { /* existing fields */ },
    Update { /* existing fields */ },
    Delete { /* existing fields */ },
    Call(CallPatchOp),
}
pub struct CallPatchOp {
    pub layer: usize,
    pub feature: usize,
    pub gate_vector: Vec<f32>,
    pub monty_code: String,
    pub code_hash: String,
    pub input_schema: CallInputSchema,
    pub output_schema: CallOutputSchema,
    pub trigger: CallTrigger,
    pub limits: CallResourceLimits,
    pub safety: CallSafetyPolicy,
    pub metadata: CallPatchMetadata,
}
pub struct CallTrigger {
    pub score_threshold: f32,
    pub margin_threshold: Option<f32>,
    pub max_calls_per_token: usize,
    pub max_calls_per_sequence: Option<usize>,
    pub cooldown_tokens: Option<usize>,
    pub require_top_k: usize,
}
pub enum CallSource {
    Residual { layer: usize, position: PositionSpec },
    CurrentResidual,
    GateActivations,
    UpActivations,
    PriorLayerResiduals { layers: Vec<usize> },
    TokenWindow { before: usize, after: usize },
    WalkTrace { top_k: usize },
    EngineScratch { key: String },
}
pub enum CallSink {
    ResidualDelta { scale: f32, clamp_norm: Option<f32> },
    PreDownActivationDelta { scale: f32 },
    LogitBias { scale: f32, top_k_limit: Option<usize> },
    EngineScratch { key: String },
}
pub struct CallInputSchema {
    pub sources: Vec<CallSource>,
    pub vector_codec: VectorCodecSpec,
    pub include_token_text: bool,
    pub include_token_ids: bool,
}
pub struct CallOutputSchema {
    pub sinks: Vec<CallSink>,
    pub vector_codec: VectorCodecSpec,
    pub required_keys: Vec<String>,
}
pub enum VectorCodecSpec {
    RawF32 { dim: usize },
    TopK { k: usize, basis: BasisSpec },
    LearnedLinear { artifact_id: String, input_dim: usize, output_dim: usize },
    TokenLogitBias { vocab_size: usize, sparse: bool },
}

New overlay query interface

Keep GateIndex::gate_knn backward compatible. Add a separate capability to retrieve call metadata only after candidate selection:

pub trait CallPatchIndex {
    fn call_patch(&self, layer: usize, feature: usize) -> Option<&CallPatchOp>;
    fn call_patches_for_layer(&self, layer: usize) -> &[CallPatchOp];
}

Rationale: gate_knn should not execute Monty. It should remain a BLAS-backed candidate selector. Docs describe gate_knn as gate_matrix @ residual and emphasize its speed, so preserving that purity is the least invasive path.  ￼

Data behavior

* Call contributes a gate vector to the patched overlay exactly like static inserts.
* feature_meta(layer, feature) for Call should either:
    * return None, and inference checks CallPatchIndex; or
    * return a sentinel FeatureMeta with a kind = call extension.
* Prefer the first option to avoid overloading static down_meta.

Phase 1 tests

Unit tests in larql-vindex:

* PatchOp::Call JSON round trip.
* Legacy .vlp round trip unchanged.
* PatchedVindex::gate_knn includes a Call gate vector.
* CallPatchIndex::call_patch(layer, feature) returns the correct op.
* Delete/update collision behavior:
    * delete over call
    * call over insert
    * duplicate call at same (layer, feature) with last-wins / fail policy

Bench:

* Existing gate_knn benchmark with zero calls must remain within noise.
* gate_knn with 1, 10, 100 call overlay vectors should measure overlay merge overhead separately.

⸻

Phase 2 — larql-lql: LQL surface for Call patches

Confirmed integration point

larql-lql has parser/executor/REPL, mutation statements, patch lifecycle, and compile statements. Source browser confirms src/parser, src/executor, ast.rs, lexer.rs, repl.rs.  ￼

Proposed LQL syntax

Minimal hand-authored vertical slice:

BEGIN PATCH "tools.vlp";
ATTACH CALL
    AT LAYER 12 FEATURE 9001
    GATE VECTOR FROM FILE "gate.f32"
    MONTY CODE FROM FILE "normalize.py"
    INPUT (
        RESIDUAL CURRENT AS "residual",
        TOKENS WINDOW 32 AS "tokens"
    )
    OUTPUT (
        RESIDUAL_DELTA KEY "residual_delta" SCALE 0.05 CLAMP_NORM 2.0,
        LOGIT_BIAS KEY "logit_bias" SCALE 1.0 TOP_K 16
    )
    TRIGGER SCORE >= 12.0 MARGIN >= 2.0 MAX_CALLS_PER_TOKEN 1
    LIMITS TIME_US 250 MEMORY_BYTES 1048576 STEPS 10000;
SAVE PATCH;

Lower-level JSON escape hatch:

ATTACH CALL FROM FILE "call_patch.json";

Preferred early implementation: support ATTACH CALL FROM FILE first. It avoids expanding parser complexity before runtime semantics are proven.

AST changes

Exact AST enum names are unverified. Proposed additions:

pub enum Statement {
    /* existing variants */
    AttachCall(AttachCallStatement),
}
pub struct AttachCallStatement {
    pub layer: usize,
    pub feature: Option<usize>,
    pub gate_vector: GateVectorInput,
    pub monty_code: CodeInput,
    pub input_schema: CallInputSchema,
    pub output_schema: CallOutputSchema,
    pub trigger: CallTrigger,
    pub limits: CallResourceLimits,
}

Executor behavior

* If no patch session exists, match current auto-patch mutation behavior and start an anonymous patch session.
* If feature omitted, call PatchedVindex::find_free_feature(layer) if confirmed available; docs confirm find_free_feature(layer).  ￼
* Validate gate vector length against hidden size.
* Validate Monty code by constructing MontyRun::new(code, script_name, input_names) during attach, not first inference, so bad code fails early. MontyRun::new is confirmed.  ￼
* Persist source code plus hash in .vlp; optionally persist serialized Monty runner bytes later, but start with source only for readability/debuggability.

Parser changes

Confirmed files/directories to touch:

* crates/larql-lql/src/lexer.rs
* crates/larql-lql/src/parser/
* crates/larql-lql/src/ast.rs
* crates/larql-lql/src/executor/
* docs/specs/lql-spec.md  ￼

Exact parser functions are unverified.

Phase 2 tests

Parser tests:

* parse ATTACH CALL FROM FILE
* parse full inline syntax
* reject missing MONTY CODE
* reject unknown source/sink
* reject missing trigger threshold unless default policy explicitly defined

Executor tests:

* attach call creates patch session
* SAVE PATCH writes .vlp with op: "call"
* APPLY PATCH rehydrates call
* SHOW PATCHES counts calls separately from inserts/updates/deletes
* invalid Monty code fails during attach

Example:

* crates/larql-lql/examples/call_patch_demo.rs
    * load synthetic vindex
    * attach trivial call
    * save/apply patch
    * verify call metadata accessible

⸻

Phase 3 — larql-inference: forward-pass hook and source/sink plumbing

Confirmed integration points

larql-inference contains ffn, forward, trace, vindex, tokenizer.rs, residual.rs, and forward_overrides.rs.  ￼

Docs identify WalkFfn as the sparse FFN via vindex gate KNN; predict_with_ffn() uses fused attention plus WalkFfn; trace_residuals() uses caller-provided FfnBackend.  ￼

Docs also confirm a LayerHook system with mutating residual callbacks, though production Metal path is hook-free and CPU path is used for hooks.  ￼

Runtime placement

Do not execute Monty inside gate_knn. Execute immediately after the FFN path identifies active features for (layer, position).

Preferred hook site:

1. WalkFfn calls gate_knn(layer, residual_row, top_k).
2. It applies static down-vector contributions as today.
3. It checks each selected (feature, score) against CallPatchIndex.
4. For matching calls passing thresholds, it invokes MontyCallRuntime.
5. Apply decoded sink outputs before returning the layer FFN residual contribution, or at on_post_layer if the existing hook system gives safer residual mutation.

Exact current WalkFfn signature and FfnBackend trait are unverified.

New modules

Proposed files:

* crates/larql-inference/src/monty_call/mod.rs
* crates/larql-inference/src/monty_call/runtime.rs
* crates/larql-inference/src/monty_call/codec.rs
* crates/larql-inference/src/monty_call/context.rs
* crates/larql-inference/src/monty_call/error.rs
* crates/larql-inference/src/monty_call/metrics.rs

Expose behind a Cargo feature:

[features]
monty-call = ["monty", "serde_json"]

Proposed runtime interfaces

pub struct MontyCallRuntime {
    cache: MontyProgramCache,
    limits: DefaultCallLimits,
    policy: CallFailurePolicy,
    metrics: MontyCallMetrics,
}
pub trait CallExecutor {
    fn execute_call(
        &mut self,
        call: &CallPatchOp,
        ctx: &CallContext<'_>,
    ) -> Result<CallOutput, CallError>;
}
pub struct CallContext<'a> {
    pub layer: usize,
    pub position: usize,
    pub residual: &'a [f32],
    pub residual_history: Option<&'a ResidualHistoryView<'a>>,
    pub gate_activations: Option<&'a [f32]>,
    pub up_activations: Option<&'a [f32]>,
    pub token_ids: &'a [u32],
    pub token_text: Option<&'a str>,
    pub tokenizer: Option<&'a TokenizerView<'a>>,
    pub walk_trace: Option<&'a WalkTraceView<'a>>,
    pub scratch: &'a mut EngineScratch,
}
pub struct CallOutput {
    pub residual_delta: Option<Vec<f32>>,
    pub pre_down_delta: Option<Vec<f32>>,
    pub logit_bias: Option<SparseLogitBias>,
    pub scratch_writes: Vec<(String, MontyValue)>,
}
pub enum CallFailurePolicy {
    IgnoreAndContinue,
    ZeroOutputAndLog,
    AbortForward,
}

Monty integration

Use confirmed Monty API:

* Compile/cache: MontyRun::new(code, script_name, input_names) and optionally dump/load for caching parsed programs.  ￼
* Run to completion: MontyRun::run(inputs, tracker, PrintWriter) returning MontyObject.  ￼
* Iterative external-call mode: MontyRun::start(...) -> RunProgress later; not needed in the first vertical slice.
* Values: encode/decode via MontyObject::Dict(DictPairs); JSON wrappers are available for natural JSON.  ￼  ￼
* Limits: use LimitedTracker / ResourceLimits rather than NoLimitTracker for production. Confirmed exports exist.  ￼

Encoder design

Start with deterministic encoders:

1. RawF32: residual vector becomes {"residual": [float, ...]}.
2. TopKBasis: residual projected onto a fixed basis and serialized as sparse {indices, values}.
3. Token context: {"token_ids": [...], "text": "..."}.
4. Walk context: {"walk": [{"layer": L, "feature": F, "score": S}, ...]}.

Then add learned encoders:

pub trait VectorEncoder {
    fn encode(&self, ctx: &CallContext<'_>, schema: &CallInputSchema)
        -> Result<MontyObject, CodecError>;
}

Learned encoder E should live outside Monty as a small Rust-side matrix/module. Monty gets compressed dict values, not the whole hidden vector unless explicitly configured.

Decoder design

pub trait CallDecoder {
    fn decode(
        &self,
        obj: MontyObject,
        ctx: &CallContext<'_>,
        schema: &CallOutputSchema,
    ) -> Result<CallOutput, CodecError>;
}

Decoder options:

* RawResidualDelta: expects {"residual_delta": [f32; hidden_size]}.
* SparseBasisDelta: expects {basis_indices, values}, then Rust decodes to hidden vector.
* SparseLogitBias: expects {token_ids, biases}.
* ScalarControls: expects simple values consumed by a learned decoder D.

Source/sink buffer access

Initial minimal supported sources/sinks:

* Source: current residual at (L, t).
* Source: token ids/window.
* Sink: residual additive delta.
* Sink: sparse final logit bias only if inference already exposes logits post-forward.

Defer these until after vertical slice:

* pre-down activation sink
* gate/up activations source
* prior-layer residual history source
* engine scratch
* snapshot/resume external functions

Latency budget

Baseline from docs: vindex gate KNN is documented at 0.008ms/layer in one spec and production Gemma gate KNN benchmark around 2.78ms/layer; WalkFfn/mmap path is already optimized and full FFN is hundreds of ms for 34 layers in the documented profile.  ￼  ￼  ￼

Budget policy:

* Default max_calls_per_token = 1.
* Default time_us = 250 per fired call for interactive decode.
* Hard cap total call time per generated token at 1–2 ms.
* If fired calls exceed budget: skip remaining calls, mark metric.
* Batch/pure BLAS path remains unchanged when no Call patches are loaded.
* CPU hook path only at first; no Metal support until stable, matching existing hook-vs-Metal split.  ￼

Phase 3 tests

Unit tests:

* encode/decode raw residual dict.
* Monty identity-ish function returns known delta.
* invalid output schema produces CallError::Decode.
* resource timeout produces controlled failure.
* residual clamp prevents norm explosion.
* call metrics count fired/skipped/failed.

Integration tests:

* synthetic model/vindex:
    * call gate fires only on known residual
    * Monty returns residual delta
    * final next-token logit changes in expected direction
* no-call path produces byte-identical output to baseline.
* call patch loaded but not fired produces byte-identical output to baseline.
* call failure with IgnoreAndContinue leaves output unchanged except metrics.

Bench:

* bench_monty_call_cold_compile
* bench_monty_call_cached_run
* bench_walk_ffn_no_calls
* bench_walk_ffn_one_call_per_token
* bench_encoder_raw_vs_topk

⸻

Phase 4 — Monty lifecycle, sandbox policy, and shipped artifact model

Lifecycle

Use three levels:

1. Attach-time validation
    * parse/compile monty_code
    * validate input variable names
    * run optional type-check path only if Monty exposes it in Rust; Python README confirms type-checking support but Rust API details for type checking were not confirmed.  ￼
2. Load-time preparation
    * create MontyRun per unique code_hash
    * cache MontyRun or serialized dump() bytes; dump/load are confirmed.  ￼
3. Forward-time execution
    * clone/use cached runner
    * pass a single dict input
    * enforce ResourceLimits
    * collect stdout/stderr into internal trace only; do not print during decode

External functions

Do not allow external functions in v1. Monty supports external call pause/resume, but that creates non-local latency and purity concerns.  ￼

V2 may support a fixed whitelist:

pub trait MontyExternalFunctionHost {
    fn call(&mut self, name: &str, args: &[MontyObject]) -> Result<MontyObject, CallError>;
}

Error handling

Default: fail closed for unsafe output, fail soft for runtime exceptions.

* parse error at attach: reject patch
* resource limit at inference: skip output, record metric
* output wrong type: skip output, record metric
* NaN/Inf in decoded vector: reject output
* residual delta norm above clamp: clamp or reject depending policy
* Monty panic: abort forward only in debug/test; production converts to call failure if safely catchable

Shipped artifact

Define three artifact classes:

1. Static compiled model
    * COMPILE INTO MODEL
    * no Call support
    * standard safetensors/GGUF only
    * any Call patches cause explicit error:
        * Call patches are runtime-only and cannot be compiled into plain model weights
2. Standalone vindex with runtime patches
    * directory containing base vindex plus runtime_patches.vlp
    * requires LARQL loader/inference engine with monty-call feature
    * not vanilla Transformers-compatible
3. Overlay patch package
    * .vlp containing Call ops
    * applied to compatible base vindex
    * shipped with license notices for embedded Monty dependency and patch author metadata

COMPILE INTO VINDEX decision:

* Recommended: preserve static inserts by baking them, but copy Call ops into an overlay sidecar rather than pretending they are baked into weights.
* Command surface:
    * COMPILE CURRENT INTO VINDEX "out.vindex": succeeds only if no Call, or writes out.vindex/runtime_patches.vlp with warning.
    * COMPILE CURRENT INTO VINDEX "out.vindex" STATIC_ONLY: drops/rejects calls.
    * COMPILE CURRENT INTO MODEL: rejects calls.

⸻

B. Data structures & formats

.vlp extension

Current .vlp is a JSON object with operations list; operations use "op": "insert" | "update" | "delete".  ￼

Add:

{
  "op": "call",
  "layer": 12,
  "feature": 9001,
  "gate_vector_b64": "<base64 f32 x hidden_size>",
  "monty": {
    "code": "def f(x): ...\nf(input)",
    "code_hash": "sha256:...",
    "entry": "module_return",
    "inputs": ["input"]
  },
  "input_schema": {
    "sources": [
      {"kind": "current_residual", "key": "residual", "codec": {"kind": "raw_f32"}},
      {"kind": "token_window", "key": "tokens", "before": 32, "after": 0}
    ]
  },
  "output_schema": {
    "sinks": [
      {"kind": "residual_delta", "key": "residual_delta", "scale": 0.05, "clamp_norm": 2.0}
    ]
  },
  "trigger": {
    "score_threshold": 12.0,
    "margin_threshold": 2.0,
    "max_calls_per_token": 1
  },
  "limits": {
    "time_us": 250,
    "memory_bytes": 1048576,
    "max_allocations": 10000,
    "max_stack_depth": 64
  },
  "safety": {
    "on_error": "ignore_and_continue",
    "reject_nan": true,
    "max_delta_norm": 2.0
  },
  "metadata": {
    "description": "normalizes numeric span",
    "author": "...",
    "created_by": "larql-lql ATTACH CALL"
  }
}

Representation by compile targets

* APPLY PATCH: full support.
* SAVE PATCH: full support.
* DIFF INTO PATCH: should not infer Call from weight diffs. Only emits calls if comparing two patched sessions with runtime patch metadata preserved.
* COMPILE INTO VINDEX: static patches baked; call patches copied as runtime overlay sidecar or rejected under strict mode.
* COMPILE INTO MODEL: unsupported by design.
* Vindexfile: add CALL_PATCH ./tools.vlp or allow PATCH to include dynamic ops but require runtime-capable target.

⸻

C. Training pipeline

Phase 5 — Training surface

Trainable components:

1. Frozen base LM.
2. Gate direction g for each Call.
3. Encoder E: residual/context → compact dict fields.
4. Decoder D: Monty dict output → residual delta / logit bias.
5. Optional thin LoRA adapters on layers after call layer L.

Do not train Monty code through gradients. Treat it as an opaque deterministic oracle.

Recommended parameterization

For a call patch c:

* g_c ∈ R^hidden
* E_c(x) = small MLP/linear/top-k projection → structured dict
* Monty: y = M(code_c, E_c(x))
* D_c(y, x) = residual_delta ∈ R^hidden or sparse logit bias
* optional LoRA_{>L} absorbs distributional mismatch

Use a straight-through-style runtime model for training:

* forward pass executes Monty
* backward pass uses a learned differentiable surrogate S_c(E_c(x)) ≈ D_c(M(E_c(x)))

Phase 5.1 — Supervision data

Synthetic exact tasks

Generate tasks where the Monty function has exact algorithmic advantage:

* arithmetic normalization
* date arithmetic
* string transforms
* JSON field extraction
* unit conversions
* table lookup from supplied dict
* symbolic formatting

For each example:

* prompt
* target continuation
* candidate layer/position
* expected Monty input dict
* expected Monty output dict
* expected token/logit effect

Toolformer-style mining

Pipeline:

1. Run base model on corpus.
2. Select candidate (layer, position) where uncertainty is high or task pattern matches.
3. Execute Monty with candidate inputs.
4. Decode candidate output to logits/delta using warm-start D.
5. Keep examples where continuation loss drops above threshold.
6. Add hard negatives where execution does not help.

Feasibility: Monty is deterministic and designed for low startup/resource-controlled execution, so broad candidate mining is practical compared with containerized Python.  ￼

Phase 5.2 — Decoder warmup under teacher forcing

Goal: train D before optimizing gate.

Procedure:

1. Freeze base LM and gate selection.
2. Force call execution at gold positions.
3. Feed gold or Monty-produced dict output.
4. Train D to minimize next-token cross entropy plus delta regularization.
5. Stop gradient at Monty output.

Loss:

L = CE(logits_after_delta, target)
  + λ_norm ||delta||²
  + λ_kl KL(p_after || p_base) on non-target examples

Phase 5.3 — Gradient past the oracle

Option A: Differentiable surrogate backward pass — recommended first

Train S_c to approximate D(M(E(x))).

Forward:

y = M(E(x))
delta = D(y)

Backward:

∂loss/∂E uses S_c(E(x)) as surrogate

Pros:

* lower variance than policy gradient
* works with standard minibatch training
* stable enough for encoder/gate calibration
* can be validated by surrogate error

Cons:

* biased gradients
* surrogate can learn wrong local geometry
* must monitor mismatch between surrogate-predicted and real Monty outputs

Recommendation: use this for E, D, and optional LoRA.

Option B: REINFORCE / policy gradient

Treat firing and possibly encoder discretization as policy actions.

Reward:

r = loss_base - loss_with_call

or task success delta.

Pros:

* unbiased for discrete firing decisions
* directly optimizes actual Monty behavior

Cons:

* high variance
* expensive if many candidate positions
* requires baselines/control variates
* harder to debug

Recommendation: use only for final gate threshold/firing policy tuning, not initial encoder/decoder training.

Phase 5.4 — Gate calibration

Train g as a retrieval classifier.

Positive examples:

* positions where Monty lowers continuation loss
* synthetic gold positions

Hard negatives:

* same prompt wrong positions
* same entity/task form without needing the function
* near-neighbor residuals where code output would be harmful
* contexts where output parses but should not be injected

Loss:

L_gate =
  BCE(sigmoid(g·r), fire_label)
  + λ_sparse E[fire_rate]
  + λ_margin max(0, m - score_pos + score_neg)

Runtime thresholds:

* require absolute score threshold
* require margin over next feature
* cap per-token fires
* require call type allowlist per layer
* residual norm guard before and after sink

Phase 5.5 — Joint fine-tune

Train:

* g
* E
* D
* optional post-L LoRA

Freeze:

* base model weights
* Monty code

Use mixed batches:

1. function-needed examples
2. hard negatives
3. general LM preservation examples

Loss:

L_total =
  CE_task
  + λ_kl KL(p_model || p_base)
  + λ_fire fire_rate
  + λ_delta ||delta||²
  + λ_surrogate ||S(E(x)) - stopgrad(D(M(E(x))))||²

KL anchor is mandatory to preserve general LM quality.

Phase 5.6 — Evaluation

Metrics:

* gate precision/recall/F1 at (layer, position)
* fire rate per 1k generated tokens
* Monty execution count per generated token
* timeout/error rate
* residual delta norm distribution
* task accuracy on synthetic exact tasks
* continuation loss drop on mined examples
* no-regression perplexity / next-token accuracy on general corpus
* latency p50/p95/p99 per token
* next-token reach probe:
    * tasks requiring one-token answer
    * tasks requiring multi-token answer
    * compare single injection vs outer-loop repeated injection

Expected finding: single (L,t) call improves only the immediate next-token distribution; multi-token answers need recurrent invocation in generation.

⸻

D. Testing & validation strategy

LARQL’s docs emphasize extensive per-crate tests, examples, benches, parser tests, vindex tests, and inference tests.  ￼

Per-crate unit tests

larql-vindex

* patch schema serde
* legacy .vlp compatibility
* call overlay lookup
* gate KNN with call overlay
* patch conflict behavior
* compile-mode rejection/preservation policy

larql-lql

* lexer/parser for ATTACH CALL
* AST round trip
* executor attach/save/apply
* invalid code rejection
* invalid vector length rejection
* SHOW PATCHES call counts
* Vindexfile support if added

larql-inference

* dict encoder/decoder
* Monty runtime cache
* resource-limit failures
* residual/logit sink application
* no-call parity
* non-fired-call parity
* synthetic fired-call behavior
* metrics

Monty integration tests

* MontyRun::new compile cache
* MontyRun::run with MontyObject::Dict
* malformed output
* timeout/resource limit
* snapshot serialization only if used

Examples

* crates/larql-vindex/examples/call_patch_roundtrip.rs
* crates/larql-lql/examples/call_patch_demo.rs
* crates/larql-inference/examples/monty_call_identity_demo.rs
* crates/larql-inference/examples/bench_monty_call.rs

Round-trip / compile tests

* .vlp with call loads/saves exactly.
* COMPILE INTO MODEL with call fails with stable error text.
* COMPILE INTO VINDEX with call emits sidecar overlay or strict rejection.
* fresh USE of compiled runtime vindex plus sidecar reproduces call behavior.
* fresh USE without sidecar does not silently claim behavior.

Bench gates

Required budget checks:

* no call patches: no measurable regression
* call patch loaded but not fired: <1% overhead target
* one cached call fired: target <1–2 ms extra per token depending encoder
* raw hidden vector dict encoding measured separately; likely too expensive for frequent calls
* top-k/learned codec preferred for production

⸻

E. Milestones & sequencing

M1 — Source-local verification and design doc

Deliverables:

* local grep-confirmed file/symbol map
* docs/specs/monty-call-patches.md
* updates to LQL/vindex/inference specs
* compile-target policy documented

Exit criteria:

* every touched symbol/file confirmed locally
* no runtime code yet
* reviewers agree on artifact semantics

M2 — .vlp schema only

Deliverables:

* PatchOp::Call serde support
* CallPatchOp data model
* legacy patch compatibility
* CallPatchIndex trait or equivalent accessor

Exit criteria:

* cargo test -p larql-vindex
* call patch round trip
* no inference behavior yet

M3 — LQL attach/save/apply

Deliverables:

* ATTACH CALL FROM FILE
* save/apply/show support
* attach-time Monty parse validation

Exit criteria:

* cargo test -p larql-lql
* example creates .vlp with call
* no forward execution yet

M4 — Smallest end-to-end vertical slice

Deliverables:

* one hand-written call patch
* one synthetic vindex/model
* WalkFfn detects fired call
* Monty code receives {"residual": [...]} and returns {"residual_delta": [...]}
* residual delta applied
* output logits change in expected direction

Exit criteria:

* no-call and non-fired parity tests pass
* fired-call test passes
* resource failure test passes
* p50/p95 call latency printed by bench

This is the first meaningful demo.

M5 — Safety hardening

Deliverables:

* trigger thresholds and margins
* norm clamps
* call budgets
* timeout/memory limits
* metrics and tracing
* failure policies

Exit criteria:

* false-fire tests
* residual explosion tests
* timeout tests
* bench budget tests

M6 — Encoder/decoder codecs

Deliverables:

* raw f32 codec
* top-k basis codec
* sparse logit-bias codec
* learned linear codec artifact format

Exit criteria:

* codec round trips
* codec bench
* schema validation

M7 — Compile/artifact behavior

Deliverables:

* COMPILE INTO MODEL hard rejection
* COMPILE INTO VINDEX strict/runtime-sidecar modes
* docs and examples

Exit criteria:

* compile tests
* fresh load behavior verified
* no silent portability footgun

M8 — Training prototype

Deliverables:

* synthetic task generator
* forced-call decoder warmup
* gate calibration with hard negatives
* surrogate backward prototype
* evaluation harness

Exit criteria:

* synthetic exact-task accuracy improves
* gate precision/recall reported
* no-regression KL/perplexity check
* next-token reach probe confirms limitation

M9 — Research-grade training

Deliverables:

* Toolformer-style mining
* joint g/E/D/LoRA fine-tune
* surrogate-vs-REINFORCE comparison
* ablations:
    * residual delta vs logit bias
    * raw vector vs top-k codec
    * no LoRA vs post-layer LoRA
    * thresholds/cooldowns

Exit criteria:

* stable improvement on held-out algorithmic tasks
* bounded degradation on general LM eval
* latency within production budget

⸻

F. Risks & open research questions

Weekend wiring

Likely straightforward:

* .vlp schema extension
* patch overlay metadata
* LQL ATTACH CALL FROM FILE
* Monty parse/run from Rust using MontyRun
* residual delta sink in CPU WalkFfn
* compile-target rejection policy
* unit tests and synthetic demo

Engineering-hard but bounded

* cleanly threading source buffers through WalkFfn
* preserving zero-overhead no-call path
* dict/vector codecs that are fast enough
* safe residual/logit clamps
* deterministic metrics and trace output
* sidecar artifact packaging

Research-grade

Hardest parts:

1. Learning when to fire
    * false positives are costly because they both burn latency and perturb unrelated predictions.
    * gate calibration needs aggressive hard negatives and sparse fire-rate penalties.
2. Learning useful E and D
    * raw residuals are not a natural dict interface.
    * learned codecs are necessary for nontrivial functions.
3. Gradient through Monty
    * no true backprop through interpreter execution.
    * surrogate backward is practical but biased.
    * REINFORCE is principled but high variance.
4. Multi-token behavior
    * one call affects the immediate next-token distribution.
    * multi-token algorithmic outputs require repeated outer-loop calls, scratch state, or a generation controller.
5. Portability expectations
    * users will expect COMPILE INTO MODEL to “just work”; it cannot for executable hooks.
    * docs and errors must be blunt: shipped artifact is LARQL runtime + vindex + call patch, not vanilla Transformers.
6. Latency collapse under false fires
    * even microsecond-start Monty calls are expensive relative to pure BLAS KNN when many fire.
    * production default must be conservative: one call per token, strict thresholds, compact codecs, cached compiled programs, and hard budgets.
7. Security review
    * Monty is sandboxed and resource-limited, but executing user-authored code inside inference still changes the threat model.
    * no external functions in v1; no filesystem/network/env; audit dependency and license notices.

Final recommendation

Build in this order:

1. .vlp schema + overlay metadata.
2. LQL attach/save/apply.
3. One CPU WalkFfn synthetic identity-ish call.
4. Safety/budgeting.
5. Codecs.
6. Compile/artifact semantics.
7. Training prototype.

Do not start with training. The first proof point should be a single hand-written Call patch firing once inside larql-inference, producing a bounded residual delta, with the no-call path provably unchanged.
