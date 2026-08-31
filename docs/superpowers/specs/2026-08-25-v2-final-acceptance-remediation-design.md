# V2 Final-Acceptance Remediation Design

## Status

The repair direction was approved in conversation on 2026-08-25. The
remediation, Task 6.4, Part 2, and the umbrella refactor are complete as of
2026-09-01. The final production source/test commit is
`24a6627d386f7230a1ef23ec988909e7a326d69d`; clean SWAG source/harness
retirement is `5c54baa017b04fddc5a31cd958facbb47f2ec65d`; reviewed
pre-documentation evidence HEAD is
`6647282ca90cb4e1354f3dccea6406ce382acc10`; and the completion documentation
commit is `V2_FINAL_ACCEPTANCE_COMPLETION_COMMIT_SHA_TO_BE_RECORDED`.

The final Task 5 scoped architecture re-review at `6647282`, after two fix
rounds, reported Critical 0, Important 0, Minor 0. It does not stand in for or
claim completion of the later SDD final branch review.

Final evidence: full V2 `1592 passed in 101.92s`; integration `252 passed in
23.75s`; CLI workflows `31 passed`; CLI config `13 passed`; source/package `9
passed`; Ruff clean and `104 files already formatted`; pretraining validator
`pass`; SWAG validator `pass`. SWAG source/harness is
`5c54baa017b04fddc5a31cd958facbb47f2ec65d` with `harness_clean=true`; its
manifest/raw/report SHA-256 values are respectively
`76ed3446282054471dc813da860b6cd30ae90501e0a01c481618c23e2a773ab2`,
`885e62e96fab950031b8185bc916878cfbd0885d898a26255785f65c7d29aa93`, and
`a30639ff20f68974d546e9d809de4ba37f915cc30486f8809a0cc9c9b88996bb`.

All seven protected hashes stayed exact: pretraining manifest
`17a346df8e0ded255cb50e40a568517b3a1c72c0ccbc1828a044c3f3dac12763`, raw
`e80197de96c2733a5f6790bb85a3f6f142c2a8475436772dee521656b2248beb`, report
`b64f13920d1ffc78754070c6a5107635b2507d50ba54348f7c1d6462b6b30bd2`, train
fixture `6de8260bb5c060c2391ab69df4baae500d1b549e812233321f46843a156e33aa`,
validation fixture
`b5fd31a7de28084a916290c2c89ce87b320b6aca4519d0cf6f125d20c3f14d49`, SWAG
train fixture
`ff5fb55512e02fd3a4a5b6eb9e72aa2bc747e4bbdb64107d183230dc94750d60`, and
SWAG validation fixture
`a82fe60cc118ffc68119f4b99e8cf04d859fa39997d7df5b797e5b676323101a`.

`uv.lock` is unchanged from `4225c54`; no flat `v2/src/*.py` or forbidden
legacy bridge string remains; help lists `tokenize`, `prepare`, `train`,
`infer`, `evaluate`, `finetune`, `export`, and `verify`; and the accepted
checkout was clean. Performance measurement remains optional and is not an
acceptance gate.

The approved
[`2026-07-31-v2-performance-first-refactor-design.md`](2026-07-31-v2-performance-first-refactor-design.md)
remains the binding umbrella specification. This document refines how the final
implementation will meet that specification after the whole-scope review of
commit `807137d`. It does not relax the umbrella contract or authorize another
compatibility layer.

## Context

Task 6.4 acceptance at `807137d` established that the tracked checkout was
clean, Ruff passed, all 1,104 v2 tests passed, the unified CLI and integration
workflows passed, both controlled-quality validators passed, canonical evidence
was unchanged, and `uv.lock` was unchanged. A subsequent whole-scope
architecture review compared the completed tree with the umbrella design and
found six conformance gaps that the existing tests did not expose:

1. Evaluation discards the value returned by `lm_eval.simple_evaluate` and
   publishes only model, task-name, and package-version metadata. It therefore
   loses the provider metrics and the required task/request/dataset provenance.
2. The pure-array SWAG training path converts a frozen LoRA scale to BF16 and
   does not apply configured LoRA dropout. That path does not satisfy the
   specified FP32 adapter formula or explicit-randomness contract.
3. Generation selects one bucket from prompt length plus generation allowance
   and uses it for both prefill and cache capacity. Short prompts can therefore
   perform capacity-sized prefill work.
4. Evaluation scoring materializes a vocabulary-sized log-probability tensor
   before gathering the target values.
5. Several FULL-verified consumers verify through one open and then reload the
   payload through a new ordinary path or artifact root. A namespace or inode
   replacement between proof and consumption can substitute different bytes.
6. `sml verify --full` hashes all declared bytes but recursively performs
   owner-level semantic validation only for part of the artifact graph. SWAG,
   base-snapshot, export, and LoRA-run semantics are incomplete.

Passing tests were evidence about the implemented behavior; they did not
override these stricter architectural requirements. At that review point,
Task 6.4 was incomplete pending the repairs, quality recapture, gates, and
whole-scope review described here. Those requirements are now complete under
the final status above.

## Goals

- Publish strict, immutable, self-identifying evaluation results containing
  complete provider output and reproducible task provenance.
- Make live and pure-array LoRA execution obey the same FP32 scale, FP32 adapter,
  and explicit-key dropout formula.
- Select prefill and cache-capacity shapes independently.
- Gather target logits before log-sum-exp scoring without allocating complete
  log probabilities.
- Make each verification proof and its corresponding payload consumption use
  the same retained artifact-root and payload descriptors.
- Make FULL standalone verification recursively exercise the semantic loader
  for every owned artifact kind.
- Preserve all behavior outside these corrections, then repeat the full v2
  acceptance and review gates.

## Non-Goals

- No legacy readers, compatibility aliases, conversion tools, or old flat
  entrypoints are restored.
- No model, loss, optimizer, generation, or scoring change is authorized beyond
  behavior already required by the umbrella design.
- No top-level project or dependency edit is planned, and `uv.lock` must remain
  unchanged.
- No performance comparison or threshold is an acceptance requirement.
- Pretraining's numerical contract is unaffected, so its canonical quality
  evidence is validated but not regenerated.
- This remediation does not add checkpoint history, historical step selection,
  pluggable artifact backends, or a generic training framework.

## Cross-Cutting Rules

Every repaired reader fails closed. There are no warning-based fallbacks when a
strict field, authoritative identity, semantic array property, or descriptor
invariant cannot be established.

All persisted JSON uses the existing canonical `sml-json-v1` encoding. Strict
readers reject missing or extra fields, duplicate object keys, non-finite
numbers, non-JSON values, a non-canonical byte representation, and an identity
that does not match its declared projection. Absolute runtime paths are
diagnostic only and never enter a persisted semantic identity.

All numerical tests exercise both eager and compiled execution where a compiled
owner exists. All randomness is explicit state. Disabled dropout consumes no
key; enabled dropout consumes exactly one subkey at its position in canonical
forward order and returns the next unused key.

## Evaluation Result Contract

### JSON value boundary

Provider-owned values are normalized immediately at the evaluation boundary to
this closed recursive type:

```text
JsonScalar = null | bool | int | finite float | string
JsonValue = JsonScalar | tuple[JsonValue, ...] | mapping[string, JsonValue]
```

Mappings are emitted with canonical key ordering. Provider tuples and lists are
normalized to JSON arrays. Supported NumPy scalar values are converted to their
corresponding Python scalar only after finite/range validation. Arbitrary
objects, byte strings, arrays, path objects, and non-string mapping keys are
errors rather than stringified diagnostics. This guarantees that the complete
captured provider result is stable and can be strictly read without provider
code.

### Source and provider records

The result schema defines these frozen value records:

```text
EvaluationSourceIdentity
  logical_name: string
  content_identity: sha256 identity

EvaluationProviderVersion
  name: nonempty string
  version: nonempty string

EvaluationTaskRecord
  task_name: nonempty string
  task_identity: structured identity
  task_yaml: EvaluationSourceIdentity
  include_template_closure: ordered tuple[EvaluationSourceIdentity, ...]
  task_metadata_version: nonempty string
  prompt_config: JsonObject
  few_shot_config: JsonObject
  generation_config: JsonObject
  metric_normalization_config: JsonObject
  seeds: JsonObject
  limit: positive int | null
  ordered_request_identity: structured identity
  lm_eval_package_version: nonempty string
  lm_eval_source_commit: nonempty string | null
  dataset_revision: nonempty string
  dataset_fingerprint: nonempty string
  provider_versions: sorted tuple[EvaluationProviderVersion, ...]
  metric_payload: JsonObject

EvaluationResult
  kind: exactly "evaluation-result"
  version: exactly 1 or exactly 2
  identity: structured identity
  model: ModelIdentity
  tasks: ordered nonempty tuple[EvaluationTaskRecord, ...]
  provider_result: JsonObject
```

`JsonObject` means a `JsonValue` mapping, not an unvalidated `dict[str,
object]`. Provider names are unique and sorted by name. Source logical names are
portable logical identifiers; local filesystem locations may be exposed by an
in-memory diagnostic result but are not serialized in this schema.

The persisted v1 field set and `sml-evaluation-result-v1` identity domain remain
frozen. Its `ModelIdentity` has the already frozen artifact kind, optional run
identity, optional step, optional checkpoint identity, optional current
run-state identity, tokenizer identity, and verification level. A legacy v1
read reports recovery state as unavailable rather than inventing values.

Strict v2 adds Boolean `latest_recovered` and `pruning_pending` fields to
`ModelIdentity` and uses the `sml-evaluation-result-v2` identity domain. The
current writer emits v2, and readers dispatch strictly by version. Read-only
resolution reports whether latest selection recovered from an invalid newer
candidate and whether superseded checkpoints remain pending pruning, without
performing a write or prune. The output destination is not part of either
`EvaluationResult` version and is not serialized.

### Identity projections

Each task identity is a domain-separated structured identity over every task
field above except `task_identity` and `metric_payload`. It therefore identifies
the definition, configuration, provider code, resolved dataset, and exact
ordered requests independently of the observed scores.

The result identity is a domain-separated structured identity over the exact
schema kind/version, resolved model identity, complete ordered task records
including metric payloads, and complete normalized top-level provider result.
Changing a metric, request order, task/include/template byte, dataset revision
or fingerprint, provider version, seed, limit, normalization rule, resolved
model, or verification level must change the result identity.

### Resolution and publication

Evaluation resolves the model once, then resolves each requested lm-eval task
before publication. The resolver follows the complete ordered YAML
include/template closure, hashes the authoritative source bytes, records the
task metadata version and effective prompt/few-shot/generation/normalization
configuration, resolves provider package versions and source commit when the
installed source exposes one, and obtains the immutable dataset revision and
fingerprint from the actual dataset used.

The adapter records the canonical semantic request representation, in provider
execution order, before satisfying each request. One ordered request identity is
computed per task. Merely counting request objects is insufficient. Evaluation
fails before publication if the authoritative task source, closure, effective
configuration, dataset revision/fingerprint, or ordered request identity cannot
be resolved. A source commit is the sole nullable provenance field because the
umbrella contract permits it only when available.

The completed implementation captures each YAML source through a retained,
nonblocking, no-follow descriptor, rejects non-regular files before reading,
and verifies descriptor stability around the byte read. Stable pre/post source
snapshots bracket provider parsing and task construction, and the retained
snapshot is compared again before publication. Evaluation provenance is thus
bound to the exact load-adjacent bytes without introducing a second YAML
parser.

The return from `simple_evaluate` is mandatory input. The owner normalizes and
preserves its complete top-level payload, extracts each supported task's
complete metric payload without dropping provider-defined metrics, and proves
that every requested task has exactly one corresponding task record. Missing,
extra, or ambiguous provider task results fail closed.

Publication uses a temporary file in the destination directory, file `fsync`,
no-replace rename/link publication, and parent-directory `fsync`. If a target
already exists, the strict reader fully validates it. Byte-identical semantic
content is an idempotent success; any different valid or invalid content is a
collision and is never overwritten.

### Evaluation tests

Tests first fail against the metadata-only implementation, then prove:

- complete provider output and per-task metrics survive a write/read round trip;
- absolute output locations do not enter persisted bytes or identities;
- every schema field is exact and duplicate/non-canonical/non-finite input is
  rejected;
- model, task source/closure, task configuration, request order, dataset,
  provider, metric, seed, and limit changes affect the correct identity;
- unresolved or ambiguous provenance prevents publication;
- identical publication is idempotent and a differing target is a collision;
- the base-run, LoRA-run, and export evaluation CLI paths publish the same
  strict contract; and
- unsupported lm-eval request methods still fail explicitly.

## LoRA Precision and Randomness Contract

### Static forward policy

One immutable `LoRAForwardPolicy` is derived from the strict `LoRAConfig` and
the canonical adapted-module traversal. Its complete field contract is:

```text
LoRAAdapterSpec
  module_path: canonical model parameter path
  scale: finite positive scalar derived from alpha/rank/scaling_mode
  dropout: finite float in [0, 1)

LoRAForwardPolicy
  adapters: ordered nonempty tuple[LoRAAdapterSpec, ...]
```

Module paths are unique and appear in model execution order. The policy is
static semantic configuration, not a model parameter tree. The trainable tree
contains only FP32 `lora_a` and `lora_b` leaves. The frozen array tree contains
only the BF16 base-model leaves. No `scale` leaf is cast to BF16, saved as base
state, reconstructed from BF16, or optimized. At each use, the configured scale
is materialized explicitly as an FP32 MLX scalar.

Checkpoint compatibility remains based on the complete serialized LoRA config
and canonical adapter leaf set. Checkpoints continue to store only trainable
adapter arrays and trainer/optimizer state; no schema migration or fallback
scale interpretation is introduced.

### Shared adapter formula

Live module execution, pure-array training, merge/export, and their test oracles
share one canonical adapter calculation:

```text
adapter_input_fp32 = cast_fp32(input_bf16)
if training and dropout > 0:
    adapter_input_fp32, next_key = keyed_dropout(
        adapter_input_fp32, dropout, current_key
    )
else:
    next_key = current_key

adapter_fp32 = scale_fp32 * (
    (adapter_input_fp32 @ lora_a_fp32.T) @ lora_b_fp32.T
)
output_bf16 = cast_bf16(base_output_bf16 + cast_bf16(adapter_fp32))
```

The pure-array linear dispatch receives the canonical module path and policy; it
must not infer adapter behavior from an incidental parameter-tree leaf. During
training, one key cursor is threaded through the entire forward. Existing base
dropout sites retain their execution order, and an enabled LoRA dropout site is
visited immediately before its corresponding adapter matmuls in canonical
module order. Adding the previously missing LoRA sites intentionally changes
which subkeys reach later active sites. The returned key is the first unused
key after all active sites and is the only key persisted in checkpoint trainer
state.

Inference and export run with `training=False`; LoRA dropout consumes no key.
The merge formula remains FP32 `scale * (B @ A)` added to an FP32 cast of the
base weight followed by the specified BF16 cast.

### LoRA tests and evidence

Tests use nonzero adapter dropout and a scale whose FP32 value is not exactly
representable in BF16. They prove:

- adapter leaves stay FP32, base leaves stay BF16, and scale multiplication is
  FP32 in live, pure-array, and merge paths;
- live and pure-array paths follow the same formula and key progression;
- disabled or inference dropout consumes no subkey;
- two eager steps and two compiled steps agree under the pinned tolerance;
- uninterrupted and checkpoint-resumed training are exact for adapter,
  optimizer, cursor, and next-key state;
- changing the input key changes dropout output while replaying the key exactly
  reproduces it; and
- live-adapter and merged-export outputs retain their existing BF16 tolerance
  contract.

This correction changes the controlled SWAG training trajectory. After all
repair code and tests are committed and the tracked source checkout is clean,
the canonical SWAG quality result is regenerated against that exact source
commit and then independently validated. The pretraining quality result remains
unchanged and is only revalidated.

## Inference Shape and Scoring Contract

### Independent generation shapes

Generation uses the configured length ladder independently in two domains:

- `prefill_length_bucket` is the smallest bucket that fits the real encoded
  prompt and bounds only prefill token, mask, and position work.
- `cache_capacity_bucket` is the smallest bucket that fits real prompt length
  plus `max_new_tokens` and bounds token storage, KV storage, and decode state.

The amended internal records have these exact fields:

```text
_PreparedRequest
  caller_index: int
  prompt_ids: tuple[int, ...]
  prefill_length_bucket: positive int
  cache_capacity_bucket: positive int
  kernel_key: GenerationKernelKey
  seed: nonnegative int
  key: MLX PRNG-key array
  max_new_tokens: nonnegative int
  include_prompt: bool

GenerationBucket
  prefill_length_bucket: positive int
  cache_capacity_bucket: positive int
  batch_size_bucket: positive int
  kernel_key: GenerationKernelKey
  keys: MLX PRNG-key array
  request_mask: Boolean MLX array
  prompt_ids: tuple[tuple[int, ...], ...]
  prompt_lengths: int32 MLX array
  max_new_tokens: int32 MLX array
  seeds: tuple[int, ...]
  caller_indices: tuple[int, ...]
  include_prompt: tuple[bool, ...]
  host_max_new: nonnegative int
```

The grouping/compile key contains both bucket values, the selected batch-size
bucket, and the generation-kernel policy. The buffer lease allocates token and
KV state at cache capacity; prefill arrays are padded only to the prefill
bucket. Prefill returns cache state backed by the capacity-sized lease while
executing over the prompt-sized input. Decode writes at each request's logical
position and is bounded by its capacity and individual maximum-new-token count.

The two buckets may be equal, but no code may recover one by aliasing or
copying the other. A short prompt with a large generation allowance must compile
and run a short prefill against a large cache. Caller-order restoration and
serial/batched random-stream equivalence remain unchanged.

### Target-logit scoring

For every scoring batch, the owner computes:

```text
target_logits = take_along_axis(
    predictor_logits_fp32, target_token_ids[..., None], axis=-1
).squeeze(-1)
target_log_prob = target_logits - logsumexp(
    predictor_logits_fp32, axis=-1
)
```

Valid continuation masks are applied only to the gathered token-shaped result.
The implementation never constructs `predictor_logits - logsumexp(...)` with
the full vocabulary shape. Greedy-match calculation may perform `argmax` on the
original logits but does not create another vocabulary-sized probability
tensor.

### Inference tests

Tests prove:

- a short-prompt/long-generation request leases capacity-sized KV/token state
  but sends only the prompt bucket into the prefill function;
- generation grouping and compile caches distinguish either bucket changing;
- heterogeneous serial and batched generation retain existing deterministic
  equivalence contracts;
- scoring equals the direct FP32 mathematical oracle for both padding layouts;
- a shape spy observes only gathered target-shaped log probabilities; and
- no source path computes or retains a full log-probability tensor.

## Descriptor-Bound Artifact Ownership

### Retained root and payload handles

An artifact-consuming owner accepts a `Path` only at its outer boundary. It then
opens one `ArtifactRoot` descriptor, parses the canonical manifest from that
root, and retains the root until all owned payload consumption is complete.
Nested artifact roots are opened descriptor-relatively from the retained parent
root, never by constructing and reopening `parent_path / child_path`.

The internal open result is an owned context-managed value containing the root,
strict manifest, and verification level. Opening a declared payload performs
path traversal, regular-file/link-count/inode-alias checks, size checking, and
the selected verification on one payload descriptor:

- FULL hashes that descriptor, compares the content identity, rewinds it, and
  gives the same descriptor to the semantic loader.
- MANIFEST_TRUSTED validates the same descriptor's structural metadata and
  declared byte size, rewinds it, and gives it to the semantic loader without a
  content hash.

The owner snapshots device, inode, size, modification time, and change time
from `fstat` before proof/consumption and rechecks them after semantic loading.
Any change is corruption. This closes the supported reader window for in-place
mutation as well as namespace replacement; external mutation remains outside
the cooperating-writer protocol.

No consumer calls a path-based `np.load`, reopens a safetensors path under a new
`ArtifactRoot`, or performs a check-then-open sequence. Whole-root, child-entry,
payload-entry, symlink, hard-link, and inode changes are either made irrelevant
by retained descriptors or rejected by the descriptor checks.

### Safetensors lifetime

Safetensors are loaded by passing the already verified open payload object to
`mx.load(..., format="safetensors")`. All returned arrays are validated against
the manifest's exact name/dtype/shape contract and evaluated before the payload
or root descriptor closes. A materialized MLX array may then outlive the file;
no lazy load may first touch the file after close.

This rule applies to checkpoint groups, copied LoRA base weights, export model
weights, and any verification-only semantic safetensors load.

### Owned NPY mappings

SWAG uses an internal `_OwnedNpyMapping` for each declared NPY payload. Its
complete field contract is:

```text
_OwnedNpyMapping
  logical_path: portable logical path
  payload: open binary payload object
  mapping: read-only mmap of payload descriptor
  array: read-only ndarray backed by mapping
```

The NPY magic/header is parsed from the verified descriptor, and the mapping is
created from that same file descriptor at the parsed offset. Header version,
dtype, C order, exact shape, data offset, and exact file size are checked before
the array is exposed. It does not use `/dev/fd`, an ordinary path, or
`np.load(..., mmap_mode="r")`.

`SwagDataBundle` has the complete fields `path: Path` (diagnostic only),
`manifest: SwagDataManifest`, `verification: VerificationLevel`, `buckets:
tuple[SwagBucket, ...]`, `_root: ArtifactRoot`, `_mappings:
tuple[_OwnedNpyMapping, ...]`, and `_closed: bool`. It is the deterministic owner
of its retained root, payload handles, mappings, and bucket arrays and
implements `close()` and the context manager protocol. Close first releases
bucket/array views, then mappings, then payload descriptors, then the root; it
is idempotent. Streams and training runs retain the bundle for the complete
period in which arrays can be accessed and close it in success, early return,
and failure paths. Garbage collection is not the resource-management protocol.

The existing prepared-data descriptor/mmap pattern remains the reference owner
and is covered by the same swap regressions; no path-reopen regression is
introduced there.

### Descriptor tests

Adversarial tests replace a root name, child directory, or payload path after
manifest parsing or FULL hashing but before semantic loading. They prove that
SWAG load and resume, LoRA copied-base resolution, merged-export inference, and
recursive verification either consume the already proved inode or fail closed.
The tests also cover symlinks, internal/external hard links, duplicate inode
aliases, close ordering, early-return cleanup, and injected loader failure.

## Recursive FULL Verification

`verify_artifact(path, full=True)` dispatches by strict manifest kind to an
owner-specific semantic validator. Hash success alone is not semantic success.
Each validator uses the descriptor-bound ownership rules above and returns a
`VerificationResult` whose ordered `children` contain every independently owned
child artifact it verified.

FULL validation by kind is:

- **Tokenizer:** strict manifest and content verification, SentencePiece model
  construction, vocabulary readability, special-token/config agreement, and
  model/vocab binding.
- **Pretraining data:** tokenizer-child recursion, NPY header/dtype/shape/order
  validation, semantic row-content identity, vocabulary bounds, and the existing
  prepared-data preflight without starting a producer thread.
- **SWAG data:** descriptor-bound NPY opening plus complete bucket layout,
  dtype/shape, vocabulary bounds, BOS/EOS, valid-mask, score-mask, label,
  declared-boundary, example-count, recorded source-projection
  self-consistency, and strict base/tokenizer identity fields. The standalone
  SWAG bundle does not resolve external base or tokenizer artifacts.
- **Base snapshot:** strict model/precision config construction, complete
  safetensors leaf-name/dtype/shape validation against the configured model,
  BF16 frozen-weight enforcement, and a strict tokenizer-identity field. The
  owning LoRA run performs the comparison with its tokenizer child.
- **Export:** tokenizer-child recursion and exact outer/inner tokenizer payload
  binding, strict model/precision construction, complete plain-model leaf set,
  dtype/shape checks, and rejection of adapter-only leaves.
- **Pretraining run:** tokenizer-child recursion, latest-index recovery and
  checkpoint payload semantics, run/checkpoint/tokenizer/config identity
  agreement, and current run-state identity validation.
- **LoRA run:** tokenizer-child recursion, base-snapshot-child recursion, latest
  checkpoint payload semantics, exact adapter leaf/rank/dtype/shape validation,
  optimizer/trainer/cursor/next-key semantics, and equality of the run's base,
  tokenizer, LoRA, precision, and data compatibility fields with its children.

Merged export is also a correctness-sensitive FULL consumer. Before merging it
validates the complete LoRA source-run semantics, including progress bounds,
recovered latest/checkpoint identities, adapter/trainer state, and the
seed-derived terminal key; it does not treat a payload hash alone as sufficient
authorization to export.

Checkpoint directories remain owned children of runs rather than standalone
portable roots. A run result orders children as tokenizer, then base snapshot
for LoRA runs, then latest checkpoint. Bundle results order children by their
manifest-defined logical ownership. Every child's reported verification level
is FULL.

`full=False` remains manifest-trusted structural validation; it does not pretend
to provide the semantic guarantees above. Read-only inference/evaluation may
still default to that level, but their actual payload loads remain
descriptor-bound. FULL verification avoids allocating a complete live model or
training runtime when strict array metadata and bounded semantic reductions are
sufficient.

Tests re-sign structurally valid manifests around semantically corrupt payloads
so byte hashes alone pass. `sml verify --full` must reject corrupt SWAG masks or
labels, base/export leaf shape or dtype errors, tokenizer mismatches, adapter
rank/leaf errors, and LoRA run/base/checkpoint incompatibility. Focused tests
also assert the complete and deterministic `VerificationResult.children` tree.

## Implementation Boundaries and Sequence

The remediation is implemented as a new reviewed plan, not appended as an
unstructured final-acceptance patch. The plan will order work as follows:

1. Freeze and implement the evaluation result/provenance schema and publication
   contract.
2. Introduce the static LoRA forward policy, then repair FP32 scale and
   explicit-key dropout across live, pure-array, resume, and merge paths.
3. Separate generation prefill/capacity shapes and change scoring to gather
   target logits first.
4. Introduce retained descriptor-bound artifact ownership and migrate SWAG,
   base-snapshot, export, inference, and fine-tuning consumers.
5. Build recursive semantic FULL verification on those owner loaders.
6. Run focused integration tests, commit the repair source, regenerate only the
   canonical SWAG quality evidence from a clean checkout, and close the final
   gates and whole-scope review.

Tests are written RED before each production change. Shared artifact primitives
may be introduced before the first consumer, but each commit must leave all
already migrated owners runnable and must not create a path-based fallback.

## Acceptance

Completion requires fresh evidence for all of the following at the final source
commit:

- focused tests for every conformance gap and adversarial regression above;
- `uv run ruff check v2`;
- `uv run ruff format --check v2`;
- `uv run pytest v2/tests` outside the sandbox so MLX/Metal is available;
- all unified CLI workflow and integration suites;
- unchanged canonical pretraining evidence plus a passing validator;
- regenerated canonical SWAG evidence bound to the repaired clean source commit
  plus a passing independent validator;
- unchanged `uv.lock` and no unauthorized top-level edits;
- a clean tracked checkout after evidence/documentation commits; and
- a new whole-scope review against the umbrella design and this remediation
  design with no unresolved Critical, Important, or Minor findings.

The umbrella plan/status documentation is updated only after these gates pass.
No previous green run or review substitutes for fresh final-commit evidence.
