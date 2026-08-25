# V2 Descriptor Ownership and Recursive Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every artifact consumer prove and consume the same retained inode, give mapped SWAG arrays an explicit lifetime owner, and make FULL verification recursively enforce each artifact kind's semantic contract.

**Architecture:** Pathnames are accepted only at outer boundaries. An `ArtifactRoot` is retained while its strict manifest and payloads are consumed; nested roots are opened relative to that descriptor. One `VerifiedPayload` object owns each opened file, performs the selected proof, exposes that same stream to the semantic loader, and rejects mutation at close. Safetensors are evaluated before close, NPY arrays retain descriptor-backed read-only mappings, and verification dispatches to owner-specific loaders with deterministic child results.

**Tech Stack:** Python 3.12.13, POSIX `openat`/`fstat`, NumPy NPY header parsing and `mmap`, MLX safetensors loading, strict dataclass manifests, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md`

## Global Constraints

- Continue in the current checkout; do not create a worktree.
- Use `uv run`; run every pytest command outside the sandbox.
- Do not edit top-level project files or `uv.lock`.
- Never re-open an artifact pathname after proof, use `/dev/fd`, or add a path-based fallback.
- FULL proof, semantic consumption, and post-consumption mutation checks use one payload descriptor.
- MANIFEST_TRUSTED skips content hashing only; it retains structural, size, descriptor, and post-consumption checks.
- Keep model verification metadata-only: do not allocate a complete live model or optimizer merely to validate safetensors.
- Preserve the existing prepared-data descriptor/mmap owner and extend its swap coverage without regressing it.

---

## File Structure

- Modify `v2/src/sml/artifacts/manifest.py`: descriptor-relative child roots, retained manifest parsing, verified payload ownership.
- Create `v2/src/sml/artifacts/arrays.py`: descriptor-bound safetensors loading and exact array-contract validation.
- Modify `v2/src/sml/artifacts/checkpoint.py`: expose retained run/checkpoint ownership needed by semantic run verification.
- Modify `v2/src/sml/artifacts/verify.py`: per-kind semantic dispatch and deterministic recursive child results.
- Modify `v2/src/sml/artifacts/__init__.py`: export only public primitives needed outside the package.
- Modify `v2/src/sml/data/pretraining.py`: retain prepared-data roots and verified shard payloads through mapped-array use.
- Modify `v2/src/sml/data/swag.py`: `_OwnedNpyMapping`, complete `SwagDataBundle` ownership, semantic bundle validation.
- Modify `v2/src/sml/data/tokenizer.py`: descriptor-bound tokenizer bundle loading.
- Modify `v2/src/sml/model/language_model.py`: pure plain-model parameter metadata projection.
- Modify `v2/src/sml/training/lora.py`: pure adapter parameter metadata projection.
- Modify `v2/src/sml/training/swag.py`: descriptor-bound base/export/resume loads and deterministic bundle cleanup.
- Modify `v2/src/sml/inference.py`: descriptor-bound export/base/tokenizer loads.
- Modify focused unit, integration, equivalence, and CLI tests listed in the tasks below.

## Frozen Interfaces

Add these internal ownership contracts in `artifacts/manifest.py`:

```text
ArtifactRoot.open_child(logical_path: str) -> ArtifactRoot

OpenedArtifact:
  path: Path (diagnostic only)
  root: ArtifactRoot
  manifest: one exact strict manifest dataclass
  verification: VerificationLevel
  closed: bool
  open_payload(reference: PayloadRef) -> VerifiedPayload
  open_child(logical_path, manifest_types) -> OpenedArtifact
  detach_root() -> ArtifactRoot
  close() -> None
  context-manager entry and exit

VerifiedPayload:
  reference: PayloadRef
  verification: VerificationLevel
  stream: BinaryIO
  opened_stat: stat_result
  closed: bool
  close() -> None
  __enter__() -> VerifiedPayload
  __exit__(exception_type, exception, traceback) -> None

open_artifact(
  path: Path,
  manifest_types: immutable tuple of supported manifest types,
  verification: VerificationLevel,
) -> OpenedArtifact
```

`open_artifact` opens one root, parses and identity-checks the strict manifest
through that descriptor, and transfers root ownership to `OpenedArtifact`.
`OpenedArtifact.open_child` opens the nested directory descriptor-relatively
and parses its manifest without constructing a child pathname for access.
Neither operation pre-hashes and closes semantic payloads: callers consume each
declared payload once through `open_payload` so proof and use cannot separate.

`OpenedArtifact.open_payload(reference)` returns a context-managed
`VerifiedPayload`. It opens once, checks regular-file,
single-link, alias, declared size, and the selected hash policy, rewinds, then
exposes that same stream. `close()` compares device, inode, size, mtime-ns, and
ctime-ns with `opened_stat` before closing, even when semantic loading raised.

Add this array-loading contract in `artifacts/arrays.py`:

```text
load_safetensors_payload(
  artifact: OpenedArtifact,
  reference: ArrayPayloadRef,
) -> dict[str, mx.array]
```

It validates the exact leaf-name set and every `ArraySpec` dtype/shape, calls
`mx.eval` on every value, and only then closes the verified payload.

The complete SWAG owners are:

```text
_OwnedNpyMapping:
  logical_path: str
  payload: VerifiedPayload
  mapping: mmap.mmap
  array: np.ndarray

SwagDataBundle:
  path: Path
  manifest: SwagDataManifest
  verification: VerificationLevel
  buckets: immutable tuple of SwagBucket
  _root: ArtifactRoot
  _mappings: immutable tuple of _OwnedNpyMapping
  _closed: bool
```

Both owners implement idempotent `close()` and context-manager methods.
`SwagDataBundle.close()` releases bucket/array views, mappings, payloads, then
the root. No bucket array remains usable after close.

Add pure metadata projections:

```text
model_parameter_specs(config: ModelConfig) -> dict[str, ArraySpec]
lora_parameter_specs(
  model_config: ModelConfig,
  lora_config: LoRAConfig,
) -> dict[str, ArraySpec]
```

They return exact flattened leaf paths, shapes, and persisted dtypes without
constructing a full live model. Focused parity tests compare them with tiny real
`model_state_dict` and `lora_state_dict` results.

Extend checkpoint ownership with these internal interfaces:

```text
CheckpointReader:
  existing fields and methods
  _run_descriptor: retained descriptor for the run root
  open_run_child(logical_path, manifest_types) -> OpenedArtifact

open_latest_checkpoint_reader(
  run: Path,
  expected_checkpoint_identity: str | None,
  verification: VerificationLevel,
  load_array_groups: frozenset[str] | None,
) -> context-managed CheckpointReader
```

Latest-index resolution, run/checkpoint parsing, array loading, and child opens
all occur while one run-access lock and the retained run descriptor are held.

## Task 1: Introduce Retained Root and Payload Primitives

**Files:**
- Modify: `v2/src/sml/artifacts/manifest.py`
- Modify: `v2/tests/unit/artifacts/test_artifact_root.py`
- Modify: `v2/tests/unit/artifacts/test_manifest.py`

**Interfaces:**
- Consumes: strict `PayloadRef`, `VerificationLevel`, existing no-follow traversal and inode-alias checks.
- Produces: `OpenedArtifact`, descriptor-relative child roots, retained-root manifest parsing, same-descriptor proof/consume lifecycle.

- [ ] **Step 1: Write failing child-root, same-FD, and mutation tests**

Add tests that:

1. Open a root, replace its pathname with another directory, then prove
   `open_child("tokenizer")` still reaches the child under the retained root.
2. Open a FULL verified payload, replace the payload pathname after hashing,
   and prove reads still return bytes from the proven inode.
3. Mutate the open inode in place after proof and assert close raises
   `SMLArtifactError` for changed size, mtime, or ctime.
4. Inject a semantic-reader exception and assert the payload descriptor still
   closes and the mutation postcheck is not skipped.
5. Exercise symlink, internal/external hard-link, and duplicate-inode-alias
   rejection through the new API.

Use a spy around `os.open`/`os.fstat` to assert proof and consumer share one
final payload file descriptor rather than checking path call counts alone.

- [ ] **Step 2: Run focused artifact-root tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/artifacts/test_artifact_root.py v2/tests/unit/artifacts/test_manifest.py -q
```

Expected: `OpenedArtifact`, `open_child`, retained parsing, and verified payload
APIs are absent.

- [ ] **Step 3: Implement descriptor-relative roots**

Factor the existing traversal into a private directory opener that walks
logical components with `_DIRECTORY_OPEN_FLAGS` and `dir_fd`. Implement
`ArtifactRoot.open_child` by duplicating the retained root, walking every child
component, validating each result is a directory, and transferring ownership
of the final descriptor into a new `ArtifactRoot`. Do not construct a `Path` for
the open operation.

Preserve the root's local-APFS property and closed-root errors. Add a descriptor
duplication helper only if manifest/checkpoint owners require independent
lifetimes; each duplicate must have one explicit closer.

- [ ] **Step 4: Implement verified payload ownership**

Move size/hash verification from `verify_payloads` into
`OpenedArtifact.open_payload`. Snapshot these exact fields immediately after
open:

```text
st_dev, st_ino, st_size, st_mtime_ns, st_ctime_ns
```

For FULL, hash the stream and compare `reference.identity`; for
MANIFEST_TRUSTED, do not hash. In both modes compare declared byte size and
rewind. At close, compare the five fields before closing. If semantic loading
and postcheck both fail, preserve the semantic exception and chain/report the
mutation failure without leaking the descriptor.

Reimplement `verify_payloads` as a compatibility loop over the same verified
payload primitive. It must not contain a second verification algorithm.

- [ ] **Step 5: Implement the retained artifact owner**

Extract the strict decode, type dispatch, identity recomputation, and payload
reference extraction into private descriptor-based helpers. `open_artifact`
opens the root, parses `manifest.json` or `run.json` through that root with
pre/post `fstat` stability checks, validates the exact manifest type and
structured identity, and returns `OpenedArtifact`. It does not open ordinary
semantic payloads before the owner requests them.

`OpenedArtifact.open_child` calls `root.open_child`, parses the child manifest,
and transfers the child descriptor to an independent owner. Callers use an
`ExitStack` so payload and child owners close before their parent.
`OpenedArtifact.close()` closes its root and is idempotent; `detach_root()` is a
one-time ownership transfer used by `SwagDataBundle` after all mappings have
successfully opened. Constructor failure closes every partially acquired
handle.

Keep `read_manifest`, `read_run_manifest`, and `read_checkpoint_manifest` as
compatibility wrappers: open an `OpenedArtifact`, verify every declared payload
through its same primitive, return `Verified`, and close. Migrated semantic
consumers must use `OpenedArtifact` directly, never these proof-only wrappers.

- [ ] **Step 6: Run the primitive gate and commit**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/artifacts/test_artifact_root.py v2/tests/unit/artifacts/test_manifest.py -q
uv run pytest v2/tests/unit/artifacts/test_checkpoint.py -q
uv run ruff check v2/src/sml/artifacts/manifest.py v2/tests/unit/artifacts/test_artifact_root.py v2/tests/unit/artifacts/test_manifest.py
uv run ruff format --check v2/src/sml/artifacts/manifest.py v2/tests/unit/artifacts/test_artifact_root.py v2/tests/unit/artifacts/test_manifest.py
git add v2/src/sml/artifacts/manifest.py v2/tests/unit/artifacts/test_artifact_root.py v2/tests/unit/artifacts/test_manifest.py
git commit -m "refactor(v2): retain artifact payload descriptors"
```

## Task 2: Load and Validate Safetensors through the Proven Descriptor

**Files:**
- Create: `v2/src/sml/artifacts/arrays.py`
- Modify: `v2/src/sml/artifacts/__init__.py`
- Create: `v2/tests/unit/artifacts/test_arrays.py`
- Modify: `v2/tests/unit/artifacts/test_checkpoint_semantics.py`

**Interfaces:**
- Consumes: Task 1 `OpenedArtifact.open_payload`, `ArrayPayloadRef` and exact `ArraySpec` entries.
- Produces: eagerly materialized, exactly validated MLX array dictionaries.

- [ ] **Step 1: Write failing safetensors ownership/contract tests**

Create tiny safetensors payloads and signed references. Cover:

- exact leaf names succeed and are returned in sorted-key order;
- missing or extra leaf, wrong dtype, and wrong shape fail;
- pathname replacement after FULL hash still loads the proven inode;
- in-place mutation during `mx.load` fails the postcheck;
- a spy array that would touch the stream lazily fails unless `mx.eval` occurs
  before the payload closes;
- injected `mx.load`, validation, and `mx.eval` failures close the stream/root.

- [ ] **Step 2: Run array tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/artifacts/test_arrays.py -q
```

Expected: shared descriptor-bound safetensors loader is absent.

- [ ] **Step 3: Implement exact eager loading**

Inside the verified-payload context call:

```python
arrays = mx.load(payload.stream, format="safetensors")
```

Require a string-keyed mapping. Compare the exact key set against
`reference.arrays`, compare normalized MLX dtype names and tuple shapes for
every entry, evaluate values in deterministic key order, then return a new
sorted dictionary. Do not accept a path or expose the stream to callers.

- [ ] **Step 4: Prove checkpoint loading already satisfies the same contract**

Add parity tests showing checkpoint safetensors hash, load, exact spec
validation, and `mx.eval` all use the already opened step payload before close,
with named-step inode checks before and after. Keep the existing checkpoint
implementation in this task; Task 4 extends run-root ownership without
replacing its already compliant step/payload proof path.

- [ ] **Step 5: Run checkpoint/array tests and commit**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/artifacts/test_arrays.py v2/tests/unit/artifacts/test_checkpoint.py v2/tests/unit/artifacts/test_checkpoint_semantics.py -q
uv run ruff check v2/src/sml/artifacts/arrays.py v2/tests/unit/artifacts/test_arrays.py v2/tests/unit/artifacts/test_checkpoint_semantics.py
uv run ruff format --check v2/src/sml/artifacts/arrays.py v2/tests/unit/artifacts/test_arrays.py v2/tests/unit/artifacts/test_checkpoint_semantics.py
git add v2/src/sml/artifacts/arrays.py v2/src/sml/artifacts/__init__.py v2/tests/unit/artifacts/test_arrays.py v2/tests/unit/artifacts/test_checkpoint_semantics.py
git commit -m "refactor(v2): load arrays through verified descriptors"
```

## Task 3: Give Prepared and SWAG Mappings Deterministic Ownership

**Files:**
- Modify: `v2/src/sml/data/pretraining.py`
- Modify: `v2/src/sml/data/swag.py`
- Modify: `v2/src/sml/training/swag.py`
- Modify: `v2/tests/unit/data/test_pretraining.py`
- Modify: `v2/tests/unit/data/test_swag.py`
- Modify: `v2/tests/integration/test_pretraining_data_workflow.py`
- Modify: `v2/tests/integration/test_swag_workflow.py`
- Modify: `v2/tests/integration/test_artifact_integrity.py`

**Interfaces:**
- Consumes: retained SWAG root/manifest and verified NPY payload descriptors.
- Produces: read-only descriptor-backed arrays whose lifetime is exactly the `SwagDataBundle` lifetime.

- [ ] **Step 1: Write failing NPY descriptor and cleanup tests**

Add focused tests that replace the SWAG root and each bucket payload pathname
after manifest/hash proof. Assert semantic reads use the proven inode. Add
rejections for NPY version, Fortran order, dtype, shape, data offset, trailing
bytes, short bytes, writable arrays, and in-place mutation.

Add cleanup spies for normal iteration, exhausted stream, early return, stream
construction failure, training failure, and explicit double close. Assert this
order exactly:

```text
bucket views -> mmap objects -> verified payloads -> artifact root
```

Add the same root/payload replacement, in-place mutation, and close-order tests
to the existing prepared-data stream. Its descriptor-backed NPY mapping remains
the reference implementation, but retained raw streams become
`VerifiedPayload` owners so close performs the common stability postcheck.

- [ ] **Step 2: Run SWAG tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/data/test_pretraining.py v2/tests/unit/data/test_swag.py v2/tests/integration/test_pretraining_data_workflow.py v2/tests/integration/test_swag_workflow.py -q
```

Expected: `_open_buckets` still uses path-based `np.load(path, mmap_mode="r")`
and `SwagDataBundle` has no deterministic owner/close protocol.

- [ ] **Step 3: Implement `_OwnedNpyMapping` on the verified stream**

Use `numpy.lib.format.read_magic` and the matching versioned header reader on
`payload.stream`; reject unsupported versions and non-C arrays. Record the data
offset, compute expected data bytes from exact shape/dtype, and require:

```text
data_offset + product(shape) * dtype.itemsize == opened_stat.st_size
```

Create `mmap.mmap(payload.stream.fileno(), 0, access=mmap.ACCESS_READ)` and then
an `np.ndarray` view with the parsed offset. Mark it non-writeable. The mapping
owns the still-open `VerifiedPayload`; closing it releases its array view, mmap,
then payload and runs the payload postcheck.

- [ ] **Step 4: Implement complete `SwagDataBundle` ownership**

Open one `OpenedArtifact`, validate recorded projections, open every declared
NPY once, and construct buckets from `_OwnedNpyMapping` views. Transfer its root
and mapping ownership to the bundle only after all validation succeeds. A
construction failure closes partial mappings in reverse order and then the
artifact owner/root.

Implement `close`, `__enter__`, and `__exit__`. Because the dataclass is frozen,
use `object.__setattr__` only for `_closed` and the released empty tuples. Any
array-access API checks `_closed` and fails clearly after close.

- [ ] **Step 5: Retain the bundle across streams and training**

Every SWAG stream or training run that can touch bucket arrays owns the bundle
for its complete lifetime. Put `close()` in `finally` paths covering success,
early return, user exception, and setup failure. Do not close at the end of
`load_swag_bundle` and do not rely on garbage collection.

Update `_open_validated_prepared_resources` to open one `OpenedArtifact`,
compare its strict manifest with the supplied `PreparedDataBundle`, and open
tokenizer references and shards through that owner. Retain each shard
`VerifiedPayload` beside its mapping until the pretraining stream closes. The
stream closes array views, mappings, payload owners, then the detached root on
success, early stop, and failure; remove the initial proof-only `read_manifest`
plus second-root reopen.

- [ ] **Step 6: Run SWAG gates and commit**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/data/test_pretraining.py v2/tests/unit/data/test_swag.py -q
uv run pytest v2/tests/integration/test_pretraining_data_workflow.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_artifact_integrity.py -q
uv run pytest v2/tests/equivalence/test_swag_equivalence.py -q
uv run ruff check v2/src/sml/data/pretraining.py v2/src/sml/data/swag.py v2/src/sml/training/swag.py v2/tests/unit/data/test_pretraining.py v2/tests/unit/data/test_swag.py v2/tests/integration/test_pretraining_data_workflow.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_artifact_integrity.py
uv run ruff format --check v2/src/sml/data/pretraining.py v2/src/sml/data/swag.py v2/src/sml/training/swag.py v2/tests/unit/data/test_pretraining.py v2/tests/unit/data/test_swag.py v2/tests/integration/test_pretraining_data_workflow.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_artifact_integrity.py
git add v2/src/sml/data/pretraining.py v2/src/sml/data/swag.py v2/src/sml/training/swag.py v2/tests/unit/data/test_pretraining.py v2/tests/unit/data/test_swag.py v2/tests/integration/test_pretraining_data_workflow.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_artifact_integrity.py
git commit -m "fix(v2): own mapped data by verified descriptors"
```

## Task 4: Migrate Tokenizer, Base, Export, Resume, and Inference Owners

**Files:**
- Modify: `v2/src/sml/data/tokenizer.py`
- Modify: `v2/src/sml/model/language_model.py`
- Modify: `v2/src/sml/training/lora.py`
- Modify: `v2/src/sml/training/swag.py`
- Modify: `v2/src/sml/inference.py`
- Modify: `v2/src/sml/artifacts/checkpoint.py`
- Modify: `v2/tests/unit/data/test_tokenizer.py`
- Modify: `v2/tests/unit/model/test_language_model.py`
- Modify: `v2/tests/unit/training/test_lora.py`
- Modify: `v2/tests/unit/artifacts/test_lora_base_snapshot.py`
- Modify: `v2/tests/integration/test_inference_workflow.py`
- Modify: `v2/tests/integration/test_swag_workflow.py`
- Modify: `v2/tests/integration/test_part1_workflow.py`

**Interfaces:**
- Consumes: retained roots, descriptor-bound safetensors, descriptor-owned checkpoints, strict model/LoRA configs.
- Produces: no path reopen across tokenizer, copied base, run resume, merged export, inference, or training.

- [ ] **Step 1: Write failing model and adapter metadata parity tests**

For a tiny valid `ModelConfig`, instantiate one real model only in the test and
compare flattened `model_parameter_specs(config)` keys/shapes/dtypes with its
plain state dictionary. Cover tied/untied embedding behavior and invalid
config rejection.

For representative LoRA target sets and ranks, compare
`lora_parameter_specs(model_config, lora_config)` with a tiny transformed
model's `lora_state_dict`. Assert exact `*.lora_a`/`*.lora_b` names, FP32 dtype,
rank axes, target coverage, and no base-model leaves.

- [ ] **Step 2: Run metadata tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/model/test_language_model.py v2/tests/unit/training/test_lora.py -q
```

Expected: pure expected-parameter projections are absent.

- [ ] **Step 3: Implement pure expected-parameter projections**

Build names and shapes from validated configuration integers and module naming
rules; do not instantiate `LanguageModel`, `Linear`, or an optimizer. Reuse a
small shared leaf-spec helper if both projections need identical validation.
The persisted plain-model dtype is BF16 and adapter leaves are FP32. Keep static
LoRA forward policy outside both projections.

- [ ] **Step 4: Write failing root/child/payload swap tests for every owner**

Add adversarial hooks after manifest parsing or FULL hashing and before semantic
construction for:

- tokenizer bundle construction;
- LoRA copied-base loading and resume;
- merged-export inference;
- pretraining-run resume/inference checkpoint access;
- SWAG training export creation.

Replace the outer root name, `tokenizer`/`base` child name, or safetensors
payload name. Each test must prove the already opened inode is consumed or the
operation fails closed. Add injected tokenizer construction, base load,
checkpoint load, merge, and inference failures and assert every root/payload
descriptor closes.

- [ ] **Step 5: Run owner tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/data/test_tokenizer.py v2/tests/unit/artifacts/test_lora_base_snapshot.py -q
uv run pytest v2/tests/integration/test_part1_workflow.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_inference_workflow.py -q
```

Expected: current owners call `read_manifest` with a path, construct child
paths, or call a local path-based `_load_safetensors` after proof.

- [ ] **Step 6: Convert tokenizer loading to a retained owner**

Add an internal loader that accepts an `OpenedArtifact`; open model and vocab
references through that owner and retain their verified payloads through
SentencePiece construction,
vocabulary readability, special-token/config checks, and model/vocab binding.
Materialize any required bytes before close.

The public `load_tokenizer_bundle(path, verification)` remains the outer
boundary: it opens exactly one artifact owner, delegates, and closes after the
loaded tokenizer is independent of the files. Parent owners call the internal
loader with `parent_owner.open_child("tokenizer", tokenizer manifest types)`,
never `path / "tokenizer"`.

- [ ] **Step 7: Convert base/export/training/inference loads**

For each owner, open its outer root once and retain it until all child and
payload work finishes:

- LoRA run: run root, descriptor-relative tokenizer child, base child, then
  descriptor-owned latest checkpoint;
- base snapshot: strict manifest and `load_safetensors_payload` from the same
  base root;
- export: outer manifest, descriptor-relative tokenizer child, exact nested
  payload binding, and model safetensors from the outer root;
- pretraining run: descriptor-relative tokenizer and checkpoint ownership;
- SWAG training/resume/merge: retain the SWAG bundle and all run/base owners
  through last array access.

Delete both local `_load_safetensors(root: Path, logical_path: str)` helpers.
Extend `CheckpointReader` with `_run_descriptor` and `open_run_child`, and add
`open_latest_checkpoint_reader`. Move the existing latest resolution into the
same run-access lock used by the reader. LoRA/pretraining run consumers enter
that reader first, then open tokenizer/base children through the reader and use
its already loaded latest checkpoint contents. Keep all named-step checks.

- [ ] **Step 8: Run migrated-owner gates and commit**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/data/test_tokenizer.py v2/tests/unit/model/test_language_model.py v2/tests/unit/training/test_lora.py v2/tests/unit/artifacts/test_lora_base_snapshot.py -q
uv run pytest v2/tests/integration/test_part1_workflow.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_inference_workflow.py v2/tests/integration/test_artifact_integrity.py -q
uv run pytest v2/tests/equivalence/test_lora_equivalence.py -q
uv run ruff check v2/src/sml/artifacts v2/src/sml/data/tokenizer.py v2/src/sml/model/language_model.py v2/src/sml/training/lora.py v2/src/sml/training/swag.py v2/src/sml/inference.py
uv run ruff format --check v2/src/sml/artifacts v2/src/sml/data/tokenizer.py v2/src/sml/model/language_model.py v2/src/sml/training/lora.py v2/src/sml/training/swag.py v2/src/sml/inference.py
git add v2/src/sml v2/tests/unit/data/test_tokenizer.py v2/tests/unit/model/test_language_model.py v2/tests/unit/training/test_lora.py v2/tests/unit/artifacts/test_lora_base_snapshot.py v2/tests/integration/test_part1_workflow.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_inference_workflow.py v2/tests/integration/test_artifact_integrity.py
git commit -m "fix(v2): bind model artifact loads to descriptors"
```

## Task 5: Build Owner-Specific Recursive FULL Verification

**Files:**
- Modify: `v2/src/sml/artifacts/verify.py`
- Modify: `v2/src/sml/artifacts/checkpoint.py`
- Modify: `v2/src/sml/data/pretraining.py`
- Modify: `v2/src/sml/data/swag.py`
- Modify: `v2/src/sml/data/tokenizer.py`
- Create: `v2/tests/unit/artifacts/test_recursive_verify.py`
- Modify: `v2/tests/integration/test_artifact_integrity.py`
- Modify: `v2/tests/integration/test_cli_workflows.py`

**Interfaces:**
- Consumes: strict kind dispatch, descriptor-bound semantic owners, pure expected-array specs.
- Produces: `VerificationResult` with complete semantic validation and deterministic ordered FULL children.

- [ ] **Step 1: Freeze expected result trees in failing tests**

Build minimal artifacts of every portable kind and both run kinds. Assert FULL
result trees exactly:

```text
tokenizer -> ()
pretraining-data -> (tokenizer,)
swag-data -> ()
base-snapshot -> ()
export -> (tokenizer,)
pretraining-run -> (tokenizer, latest-checkpoint)
lora-run -> (tokenizer, base-snapshot, latest-checkpoint)
```

Assert every FULL child reports `VerificationLevel.FULL`, paths are diagnostic
only, and order never depends on dictionary/filesystem order. For `full=False`,
assert the same structurally owned child ordering where applicable, every node
reports MANIFEST_TRUSTED, and no result claims semantic reductions.

- [ ] **Step 2: Write re-signed semantic corruption tests**

For each mutation, update payload hashes and recompute every affected manifest
identity so structural/FULL byte proof succeeds, then assert `verify_artifact`
still rejects the semantic defect:

- tokenizer unreadable model, invalid vocabulary, special-token disagreement;
- prepared-data wrong row identity or out-of-vocabulary token;
- SWAG invalid mask nesting, label, BOS/EOS, boundary, example count, source
  projection, base identity, or tokenizer identity field;
- base missing/extra leaf, wrong BF16 dtype, wrong shape, or invalid config;
- export outer/inner tokenizer mismatch, adapter-only leaf, wrong dtype/shape;
- pretraining run/checkpoint/tokenizer/config/current-state disagreement;
- LoRA base/tokenizer/LoRA/precision/data mismatch, wrong adapter leaf/rank/dtype,
  invalid optimizer/trainer/cursor state, or incorrect next RNG key.

Also run outer-root, child-name, and payload-name swaps during recursion to
prove the verifier uses the owner loaders rather than new path reads.

- [ ] **Step 3: Run recursive tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/artifacts/test_recursive_verify.py v2/tests/integration/test_artifact_integrity.py -q
```

Expected: current verifier mainly validates signed bytes, only recurses through
some tokenizer children, and omits owner-specific semantics.

- [ ] **Step 4: Separate structural kind dispatch from FULL semantic dispatch**

Open the candidate root once and probe `run.json` and `manifest.json` through
that descriptor, requiring exactly one regular no-follow manifest entry; do not
use `Path.exists()`. Parse kind and the strict manifest from that retained root. For
MANIFEST_TRUSTED, run structural reference/size checks, recursively verify
declared child owners at the same level, and return without semantic loaders or
reductions. For FULL, dispatch the parsed strict manifest to one kind-specific
validator that consumes the supplied root. Never call
`_manifest_kind(path)` and then reopen `path`.

Each semantic validator returns its manifest plus already verified child
results. Construct `VerificationResult` only after the owner and all children
finish successfully.

- [ ] **Step 5: Implement bundle semantic validators**

Require these exact FULL behaviors:

- tokenizer: descriptor-bound tokenizer loader, readable vocabulary,
  special-token/config agreement, model/vocab binding;
- pretraining data: descriptor-relative tokenizer recursion, NPY header and
  dtype/shape/order checks, row-content identity, vocabulary bounds, and the
  existing batch-size-one preflight without starting a producer thread;
- SWAG: open/close `SwagDataBundle`, validate complete bucket layout,
  dtype/shape, vocabulary bounds, BOS/EOS, valid/score masks, labels, declared
  boundaries/counts, recorded projections, and strict identity fields without
  resolving external base/tokenizer artifacts;
- base: strict config/precision plus exact plain-model specs and BF16 leaves;
- export: tokenizer recursion and exact outer/inner payload binding plus exact
  plain-model specs and no adapter leaves.

Use bounded reductions over mapped data and array metadata. Do not allocate a
live complete model.

- [ ] **Step 6: Implement run/checkpoint semantic validators**

Retain the run root and access lock across child and latest-checkpoint proof.
For pretraining runs, compare run, tokenizer, latest index, checkpoint, model,
precision, data, cursor, RNG, and current run-state identities. For LoRA runs,
also recurse into base after tokenizer, validate exact adapter specs and
optimizer/trainer state, and compare base/tokenizer/LoRA/precision/data
compatibility fields across all three owners.

Checkpoint directories remain run-owned result children. Use the checkpoint
reader's proven contents; do not make them independently path-portable.

- [ ] **Step 7: Test CLI FULL verification**

Add CLI cases where `sml verify --full` rejects re-signed semantic corruption
and reports success for every valid kind. Assert MANIFEST_TRUSTED behavior stays
explicitly weaker and deterministic. Exercise failure cleanup under the CLI.

- [ ] **Step 8: Run recursive verification gates and commit**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/artifacts/test_recursive_verify.py v2/tests/unit/artifacts/test_checkpoint_semantics.py -q
uv run pytest v2/tests/integration/test_artifact_integrity.py v2/tests/integration/test_cli_workflows.py -q
uv run pytest v2/tests/integration/test_part1_workflow.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_inference_workflow.py -q
uv run ruff check v2/src/sml/artifacts v2/src/sml/data v2/tests/unit/artifacts/test_recursive_verify.py v2/tests/integration/test_artifact_integrity.py v2/tests/integration/test_cli_workflows.py
uv run ruff format --check v2/src/sml/artifacts v2/src/sml/data v2/tests/unit/artifacts/test_recursive_verify.py v2/tests/integration/test_artifact_integrity.py v2/tests/integration/test_cli_workflows.py
git add v2/src/sml/artifacts v2/src/sml/data v2/tests/unit/artifacts/test_recursive_verify.py v2/tests/unit/artifacts/test_checkpoint_semantics.py v2/tests/integration/test_artifact_integrity.py v2/tests/integration/test_cli_workflows.py v2/tests/integration/test_part1_workflow.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_inference_workflow.py
git commit -m "fix(v2): verify artifact semantics recursively"
```

## Plan Completion Gate

- [ ] **Step 1: Scan for forbidden path-based consumer fallbacks**

```bash
rg -n 'np\.load\([^\n]*mmap_mode|/dev/fd|def _load_safetensors\(root: Path|path / "(tokenizer|base|checkpoints)"|read_manifest\(' v2/src/sml
```

Expected: no migrated consumer proves one pathname and consumes another. Any
remaining path joining is a write/publish boundary or diagnostic only and
is reviewed individually.

- [ ] **Step 2: Run the artifact component gate**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/artifacts v2/tests/unit/data v2/tests/unit/model/test_language_model.py v2/tests/unit/training/test_lora.py -q
uv run pytest v2/tests/integration/test_artifact_integrity.py v2/tests/integration/test_cli_workflows.py v2/tests/integration/test_part1_workflow.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_inference_workflow.py -q
uv run pytest v2/tests/equivalence/test_lora_equivalence.py v2/tests/equivalence/test_swag_equivalence.py -q
```

- [ ] **Step 3: Review the complete artifact contract**

Confirm one retained outer root per owner; descriptor-relative nested roots;
same-FD proof, semantic load, and postcheck; safetensors evaluation before
close; deterministic SWAG cleanup; semantic validation for every artifact kind;
complete ordered children; and no live-model allocation in verification.

Expected: no Critical, Important, or Minor findings before final acceptance.
