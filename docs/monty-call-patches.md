# Monty Call Patches

Monty call patches are runtime overlay operations that attach a bounded
function call to a vindex gate slot. They are not static knowledge edges:
`gate_knn(layer, residual, k)` still selects candidates from gate vectors, then
inference may inspect selected candidates for `op: "call"` metadata and run a
sandboxed program.

## Patch Shape

Minimal `.vlp` operation:

```json
{
  "op": "call",
  "layer": 12,
  "feature": 9001,
  "gate_vector_b64": "<base64 encoded f32 x hidden_size>",
  "monty_code": "def main(input):\n    return {\"residual_delta\": input[\"residual\"]}\n",
  "code_hash": "sha256:<64 hex chars>",
  "input_schema": {"sources": ["current_residual"]},
  "output_schema": {"sinks": ["residual_delta"]},
  "trigger": {
    "score_threshold": 12.0,
    "margin_threshold": 2.0,
    "max_calls_per_token": 1,
    "require_top_k": 1
  },
  "limits": {
    "time_us": 250,
    "memory_bytes": 1048576,
    "steps": 10000
  },
  "safety": {
    "residual_clamp_norm": 2.0,
    "failure_policy": "ignore_and_continue"
  },
  "metadata": {"name": "normalize"}
}
```

`code_hash` is filled at attach time when omitted. Trigger, limits, and safety
have typed defaults. Input and output schemas remain JSON so early adapters can
evolve without another patch-format break.

## LQL Surface

The implemented vertical slice supports the file-backed form:

```sql
BEGIN PATCH "tools.vlp";
ATTACH CALL FROM FILE "normalize-call.json";
SAVE PATCH;
```

`ATTACH CALL` requires a local vindex backend. The gate vector must decode to
the active vindex hidden size. The operation is recorded in the current patch
session, auto-starting an anonymous patch session like `INSERT` when needed.

## Runtime Contract

- Non-differentiable oracle: the function call is opaque. No gradient passes
  through the program.
- Next-token reach: a layer/position call can directly affect only the current
  forward pass and therefore the current next-token distribution. Multi-token
  effects require the generation loop to re-run calls at later positions.
- Non-portability: call patches are runtime overlay artifacts. They are not
  representable in vanilla safetensors or GGUF.
- Firing safety: gate thresholding, top-k requirements, cooldowns, execution
  budgets, residual norm clamps, and hard-negative calibration are required
  before a call can be enabled in an inference path.
- Dict/vector impedance: input/output encoders are explicit adapters. Residual
  vectors are not magically serialized into useful structured objects.
- Latency/purity: `gate_knn` remains a pure candidate selector. Function calls
  happen only after candidate selection in a hookable FFN/inference path.

## Compile Behavior

`COMPILE INTO MODEL` rejects runtime call patches. `COMPILE INTO VINDEX`
bakes static patch material into the output vindex and writes runtime call
patches to `runtime_patches.vlp` as an explicit sidecar. `STATIC_ONLY` rejects
call patches instead of writing the sidecar.

Fresh `USE "compiled.vindex"` auto-applies `runtime_patches.vlp` when present
and reports the call count. A compiled vindex without the sidecar does not
silently claim runtime call behavior.

## Codec Artifacts (M6)

Learned linear codecs live under `<vindex>/call_codecs/<artifact_id>.json`:

```json
{
  "version": 1,
  "artifact_id": "residual_compress_v1",
  "input_dim": 2560,
  "output_dim": 64,
  "weights_b64": "<base64 row-major f32[output_dim × input_dim]>",
  "bias_b64": "<optional base64 f32[output_dim]>"
}
```

Reference a codec from input/output schema:

```json
"codec": {
  "kind": "learned_linear",
  "artifact_id": "residual_compress_v1",
  "input_dim": 2560,
  "output_dim": 64
}
```

`USE` loads `call_codecs/` automatically. `COMPILE INTO VINDEX` copies the
directory when present.

## Current Implementation Status

Implemented:

- `.vlp` JSON round-trip for `op: "call"`.
- `PatchedVindex` overlay storage and lookup by `(layer, feature)`.
- Call gate vectors participate in ordinary patched `gate_knn`.
- `ATTACH CALL FROM FILE` and inline `ATTACH CALL ... GATE VECTOR ... MONTY CODE ...`.
- Explicit compile rejection and `runtime_patches.vlp` sidecar on `COMPILE INTO VINDEX`.
- Fresh `USE` auto-applies the runtime sidecar.
- Inference-side trigger evaluation, raw residual and top-k basis input
  encoding, raw residual and sparse basis delta output decoding, learned
  linear codec artifacts, residual norm clamping, sparse logit-bias decoding,
  and an injectable `MontyCallRuntime<R: CallProgramRunner>` lifecycle.
- Concrete `MontyVmRunner` execution through the `pydantic_monty` Python
  package, selected with `LARQL_MONTY_PYTHON` when a non-default interpreter is
  needed.
- Sparse and dense CPU walk paths, Metal/Q4/Q4K GPU paths, KV-cached generation,
  and fused GPU prefill/decode with call patches.
- Public helpers `predict_with_call_patches_runner` /
  `generate_with_call_patches_runner` with optional GPU backend, trace events,
  and codec registry.

Not implemented yet:

- Training-time codec learning (M8+).
- Rich inline `INPUT (...)`, `OUTPUT (...)`, and policy grammar beyond the
  current vertical slice.
