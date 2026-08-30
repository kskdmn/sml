# V2 Artifact Residual Corrections Design

## Status

Approved in conversation on 2026-08-30. This design authorizes a new correction
wave after the artifact plan's final scoped re-review found five Important and
three Minor residuals at `27ff11b`.

The binding specifications remain:

- `docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md`
- `docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md`

This document narrows the corrections needed to make the artifact component
eligible for a new zero-finding review. It does not authorize advancing to the
final-acceptance plan before that review is clean.

## Context

The artifact implementation now has retained descriptor ownership, strict
canonical manifests, recursive owner verification, descriptor-bound NPY and
safetensors consumption, safe public SWAG leases, bounded pretraining staging,
metadata/chunk-bounded FULL verification, and neutral semantic validators used
by inference and standalone verification. Fresh gates at `27ff11b` passed all
1,523 V2 tests and full Ruff/format checks.

The scoped re-review nevertheless found production-facing gaps:

1. FULL run progress assumes every optimizer update used a complete gradient
   accumulation window.
2. The counter-addressed RNG attempt overwrites the key returned by forward and
   resume can consume an invalid restored key before validation.
3. Two manifest readers close before semantic parsing and postcheck finish.
4. Safetensors header allocation is not capped before reading.
5. Inference and recursive verification disagree about ambiguous root kinds.
6. SWAG stream ownership transfer is unsafe with a live public lease.
7. Selected checkpoints perform redundant structural payload opens.
8. Several mapped-data reductions are constant-scratch but not fixed-chunk.

## Goals

- Preserve the binding explicit-randomness rule: active dropout consumes keys
  in canonical forward order, forward returns the terminal next-unused key,
  and checkpoint trainer state persists that returned key.
- Make the persisted RNG state independently and boundedly verifiable without
  replaying training history.
- Accept every internally valid complete or partial accumulation boundary and
  reject impossible progress.
- Give every manifest consumer one opened → semantic parse/validation →
  stability postcheck → close lifecycle with semantic-primary exception
  precedence.
- Use one retained-root candidate dispatcher for inference and recursive
  verification.
- Bound all artifact-controlled allocation and mapped-data reductions.
- Close the remaining lease-transfer and duplicate-open lifetime/performance
  gaps with lasting tests.

## Non-Goals

- Compatibility with artifacts or APIs produced by earlier pre-final V2
  commits. The umbrella design explicitly excludes that compatibility.
- A legacy RNG reader, optional schedule field, warning fallback, conversion
  tool, or dual verification algorithm.
- A new checkpoint history or external data-geometry dependency.
- Changes to model/loss/optimizer mathematics, dropout probability, canonical
  dropout-site order, dataset ordering, or inference sampling.
- Top-level dependency or lockfile changes.

## Counter-Addressed Forward-Key Contract

### Persisted schedule identity

Both pretraining and LoRA run checkpoint projections gain one exact required
field:

```text
rng_schedule: "counter-addressed-forward-terminal-v1"
```

The field is semantic run metadata and therefore participates in `run.json`
identity and every checkpoint's owning-run binding. Strict readers reject a
missing, unknown, or different value. It is not user-configurable and has no
fallback interpretation.

The existing scalar `microsteps` value is the persisted counter. No duplicate
counter field is added. `trainer.next_key` remains the actual terminal key
returned by the latest completed forward, not the counter itself.

### Per-microstep state transition

Let `i` be the zero-based microstep about to execute. The runtime derives the
starting MLX key in O(1):

```text
start_key = counter_random_key(seed, i)
```

For a configuration with at least one active dropout site, the host installs
`start_key` into trainer state immediately before the compiled/eager microstep.
The model and LoRA layers then consume one split per active site in their
already specified canonical execution order. The runtime persists the terminal
key returned by that forward unchanged. It must not replace the returned key
after forward.

If model and LoRA dropout are both disabled, no site consumes a key. The host
does not replace or advance trainer key state; the forward returns the same key
it received. The initial and every later zero-dropout checkpoint therefore keep
the initial counter key unchanged.

This creates independent, counter-addressed microstep streams while retaining
the binding split-and-return contract within every forward. Adding or removing
an active dropout site changes the terminal key and is detected; canonical site
order remains enforced by the existing runtime/equivalence contract rather
than inferred from a terminal key when the site count is unchanged.

### Bounded expected-key derivation

The shared semantic validator counts active sites from exact model and LoRA
configuration:

```text
model_sites = num_layers if hidden_dropout > 0 else 0
lora_sites = num_layers * len(target_modules) if lora.dropout > 0 else 0
active_sites = model_sites + lora_sites
```

Expected trainer state is:

- zero completed microsteps: `counter_random_key(seed, 0)`;
- any history with `active_sites == 0`: the same initial key;
- otherwise: derive `counter_random_key(seed, microsteps - 1)`, perform exactly
  `active_sites` canonical split-and-return advances, and compare the resulting
  terminal key with `trainer.next_key`.

Work is O(active dropout sites), bounded by the configured architecture, and
independent of training history length.

### Resume validation

Resume validates the restored trainer key against the shared expected-key
derivation while the retained checkpoint reader is live and before the key can
reach a forward. A re-signed wrong key fails closed without changing model,
optimizer, cursor, pruning, or publication state.

The next microstep then follows the same pre-forward transition as an
uninterrupted run. Tests compare every uninterrupted/resumed checkpoint group,
including the terminal key, with model dropout, LoRA dropout, mixed active
sites, and zero-dropout configurations.

## Progress Semantics

Each completed optimizer step contains at least one and at most
`gradient_accumulation_steps` microsteps. FULL verification therefore requires:

```text
step <= microsteps <= step * gradient_accumulation_steps
```

Step zero still requires zero microsteps through the same bound. Pretraining
retains exact `rows == microsteps * microbatch_size` validation. LoRA retains:

```text
microsteps <= examples <= microsteps * microbatch_size
```

These are all relationships derivable without external dataset geometry. The
accepted cursor/data-geometry ruling remains unchanged.

Tests cover complete windows, partial end-of-epoch windows, LoRA microbatch
sizes greater than one, corrupted lower/upper bounds, FULL standalone verify,
FULL inference, and resume.

## Stable Manifest Consumption

One shared stable JSON payload primitive owns:

1. the opened descriptor and acquisition stat;
2. the consumed stat after reading bytes;
3. UTF-8/JSON parsing and duplicate/non-finite rejection;
4. manifest type dispatch and exact schema parsing;
5. identity and canonical-byte validation;
6. final stability postcheck and close.

Recursive candidate dispatch and prepared-data nested tokenizer parsing use
this primitive instead of raw `ArtifactRoot.open_payload`. The handle remains
open through all semantic work. If semantic parsing and postcheck/close both
fail, the exact semantic exception remains primary and cleanup failure is its
cause. Mutation tests inject during the real parser callbacks, not before open.

## Exact Retained-Root Kind Dispatch

A neutral artifact-dispatch module exposes this exact interface:

```text
open_dispatched_artifact(
  path: Path,
  root: ArtifactRoot,
  verification: VerificationLevel,
) -> OpenedArtifact[ArtifactManifest]
```

It probes the retained root for the two legal candidate names in deterministic
order:

```text
run.json
manifest.json
```

Exactly one candidate must exist as a strict regular, single-link entry. Zero
or two candidates are errors at both MANIFEST_TRUSTED and FULL levels. The
dispatcher opens the selected manifest once, retains that descriptor through
the common stable parse, then re-probes the candidate set and selected named
inode before transferring the supplied root into the returned owner. A
candidate added/replaced during dispatch therefore fails closed. The function
owns the supplied root on entry: success transfers it to `OpenedArtifact`, and
failure closes it with semantic-primary precedence. It neither parses a second
time nor reopens the root path.

Recursive verification and inference call this same dispatcher. Diagnostic
`Path` values may still describe results, but never choose or reopen the owner.

## Bounded Safetensors Headers

`read_safetensors_layout` rejects a claimed header length greater than
`100_000_000` bytes immediately after decoding the fixed eight-byte prefix,
before header-sized allocation/read or downstream size arithmetic. It also
retains the existing payload-size, exact-key/dtype/shape/range, contiguous
coverage, and same-descriptor checks.

The limit matches the upstream safetensors parser. Valid leading JSON
whitespace accepted by current MLX and normal trailing space padding remain
accepted. A regression supplies a re-signed small file with an oversized
claimed length and proves the maximum requested read and traced memory stay
bounded.

## Ownership and Reduction Cleanup

### SWAG stream transfer

An owning `SwagBatchStream` refuses construction while the supplied bundle has
any public bucket lease. Failure occurs before stream ownership or closed state
changes. Borrowing streams retain their existing internal contract. Tests prove
the lease and bundle remain usable after rejection and that normal later
lease-close/bundle-close teardown remains deterministic.

### Checkpoint closed-world proof

Closed-world verification separates directory namespace enumeration and
regular/single-link/alias/size validation from payload hash/use. When the
semantic pass will open every checkpoint payload, the structural pass does not
open those payloads a second time. The semantic pass remains the sole proof/use
open for the selected checkpoint. Tests count actual descriptor opens, not only
semantic helper calls, in FULL and MANIFEST_TRUSTED requested-group paths.

### Fixed-size mapped reductions

Prepared-data token bounds and row identity use the same fixed row chunks.
SWAG first-position, label, mask, boundary, and identity checks are folded into
the existing 1,024-row scan. No `min`, `max`, `any`, `all`, or equivalent
reduction receives a complete mapped dataset/bucket when its first dimension
can exceed the configured chunk.

Instrumentation records reduction operand shapes/bytes for multi-chunk sparse
fixtures and proves the configured cap. No permanent test-only hook is added to
production code.

## Error Handling and Ownership

- All new strict failures use `SMLArtifactError`, `SMLDataError`, or the
  existing boundary-specific project error type.
- Semantic errors remain primary over postcheck and close failures.
- Failed SWAG stream ownership transfer leaves the supplied bundle live and
  retryable; dispatcher failure instead closes the root it accepted ownership
  of on entry.
- FULL and MANIFEST_TRUSTED retain their current strength distinction; trusted
  mode performs no tensor value reductions.
- The diagnostic data locator is never opened by portable verification.

## Testing and Review

Implementation uses strict TDD. Every finding first receives a focused RED
that fails for the reviewed reason, followed by the smallest shared GREEN.
Required lasting coverage includes:

- partial pretraining and LoRA progress across verify/inference/resume;
- active/mixed/disabled dropout terminal keys, huge microstep counters,
  uninterrupted/resumed equality, and re-signed wrong-key rejection before use;
- recursive and prepared nested manifest mutation plus dual failure precedence;
- oversized safetensors headers with bounded read/allocation;
- ambiguous root rejection at both verification levels and inference;
- active public lease stream-transfer rejection and retryable ownership;
- exact selected-checkpoint payload open counts;
- fixed-size prepared/SWAG reduction operands.

Each independently reviewable task receives a scoped review and up to five
normal fix rounds. After all tasks pass, one whole-component review and at most
one combined final correction wave plus one scoped re-review determine whether
the artifact phase is clean.

Before completion, run outside the sandbox where required:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
```

Also require `git diff --check`, an unchanged `pyproject.toml`/`uv.lock`, the
forbidden path-based consumer scan, and the tracked handoff as the only inherited
worktree modification.
