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
patches to `runtime_patches.vlp` as an explicit sidecar. This prevents
producing artifacts that appear self-contained while silently dropping runtime
behavior.

## Current Implementation Status

Implemented:

- `.vlp` JSON round-trip for `op: "call"`.
- `PatchedVindex` overlay storage and lookup by `(layer, feature)`.
- Call gate vectors participate in ordinary patched `gate_knn`.
- `ATTACH CALL FROM FILE`.
- Explicit compile rejection.
- Inference-side trigger evaluation, raw residual and top-k basis input
  encoding, raw residual and sparse basis delta output decoding, residual norm
  clamping, sparse logit-bias decoding, and an injectable
  `MontyCallRuntime<R: CallProgramRunner>` lifecycle.
- Concrete `MontyVmRunner` execution through the `pydantic_monty` Python
  package, selected with `LARQL_MONTY_PYTHON` when a non-default interpreter is
  needed.
- Opt-in sparse `WalkFfn` wiring for selected call patches. Call features are
  looked up after gate selection, executed through the installed runtime, and
  skipped as static FFN rows.
- Dense and Metal/Q4 walk paths post-process FFN output with
  `apply_call_patches_dense` after accelerated matmul completes.
- Sparse full-K gemv and parallel Q4K down fast paths apply call patches per
  position instead of bypassing the serial call loop.
- Public helpers `predict_with_call_patches_runner` /
  `generate_with_call_patches_runner` accept an optional GPU backend.

Not implemented yet:

- Fused GPU `prefill_kquant` with call patches (CPU Q4K KV-cached batched prefill is wired).
- Cross-layer KV sharing on the call-patch generation path.
- Inline `ATTACH CALL ... INPUT ... OUTPUT ... TRIGGER ...` grammar.
