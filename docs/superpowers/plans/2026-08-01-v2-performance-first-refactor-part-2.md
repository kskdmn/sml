# V2 Performance-First Refactor Part 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Final execution status (2026-09-01):** Part 2, Task 6.4, the
final-acceptance remediation, and the umbrella refactor are complete. The final
production source/test commit is
`24a6627d386f7230a1ef23ec988909e7a326d69d`; clean SWAG source/harness
retirement is `5c54baa017b04fddc5a31cd958facbb47f2ec65d`; reviewed
pre-documentation evidence HEAD is
`6647282ca90cb4e1354f3dccea6406ce382acc10`; and the completion documentation
commit is `V2_FINAL_ACCEPTANCE_COMPLETION_COMMIT_SHA_TO_BE_RECORDED`.

The final Task 5 scoped architecture re-review at `6647282` followed two fix
rounds and reported Critical 0, Important 0, Minor 0. This does not claim that
the later SDD final branch review has occurred.

Final acceptance evidence: full V2 `1592 passed in 101.92s`; integration `252
passed in 23.75s`; CLI workflows `31 passed`; CLI config `13 passed`;
source/package `9 passed`; Ruff clean and `104 files already formatted`; both
pretraining and SWAG validators `pass`. SWAG source/harness is
`5c54baa017b04fddc5a31cd958facbb47f2ec65d`, `harness_clean=true`; final
manifest/raw/report hashes are
`76ed3446282054471dc813da860b6cd30ae90501e0a01c481618c23e2a773ab2`,
`885e62e96fab950031b8185bc916878cfbd0885d898a26255785f65c7d29aa93`, and
`a30639ff20f68974d546e9d809de4ba37f915cc30486f8809a0cc9c9b88996bb`.

The unchanged protected hashes are pretraining manifest
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
legacy bridge string remains; CLI help lists `tokenize`, `prepare`, `train`,
`infer`, `evaluate`, `finetune`, `export`, and `verify`; and the accepted
checkout was clean. Performance measurement remains optional and is not an
acceptance gate. Unchecked boxes in earlier task bodies are retained only as
the historical execution procedure; the final Task 6.4 boxes below record the
completed acceptance.

**Goal:** Complete persistent inference/evaluation, cached SWAG LoRA fine-tuning/export, the unified CLI, and the clean removal of every replaced flat v2 path while meeting the final correctness and runnable-workflow gates.

**Architecture:** This part consumes the model, artifact, prepared-data, and pretraining-run contracts completed in Part 1. It adds persistent non-reentrant inference sessions and batched evaluation, then self-contained LoRA runs backed by immutable encoded SWAG data, and finally exposes all workflows through one typed CLI before deleting migration scaffolding and legacy modules.

**Tech Stack:** Python 3.12.13, MLX 0.32+, NumPy 2.4+, SentencePiece 0.2+, datasets 5, lm-eval 0.4, `tomllib`, `uv run`, pytest 9, Ruff 0.15+

## Global Constraints

**Superseding acceptance policy (2026-08-22):** Phase progression is gated by
Ruff, the full v2 test suite, controlled correctness/quality checks, and
relevant end-to-end/CLI smoke workflows. Baseline comparisons, statistical
thresholds, thermal launch gates, and performance evidence files are optional
diagnostics and cannot block a phase.

- Complete `docs/superpowers/plans/2026-08-01-v2-performance-first-refactor-part-1.md` through its Phase 3 gate first, including recorded-source quality validation and deeply immutable checkpoint-reader contents.
- The approved source of truth is `docs/superpowers/specs/2026-07-31-v2-performance-first-refactor-design.md`.
- This is a clean break: provide no readers, aliases, conversions, warnings, or fallback interpretations for existing v2 imports, CLI flags, tokenizer inputs, prepared datasets, manifests, checkpoints, metadata, or resume state.
- Preserve Transformer, YaRN/RoPE, GQA, RMSNorm, SwiGLU, causal-loss, KV-cache, greedy/configured sampling, and SentencePiece BPE mathematics.
- The only corrections are canonical summed tied-embedding gradients, FP32 mean continuation-token SWAG scores including EOS, and authoritative FP32 base-model master parameters with FP32 Adam state/arithmetic plus an explicitly derived BF16 working tree.
- Live LoRA copies and freezes the selected BF16 working base from a fully verified FP32-master/BF16-working pretraining checkpoint, uses FP32 adapter state/math, and casts the adapter contribution to BF16 at the base-output addition. Merged export uses the exact FP32 delta plus BF16 cast formula and contains no training-only master state.
- Every base-pretraining run consumed here authoritatively records `rope_scaling_factor=1.0`, and the SWAG LoRA flow preserves that value. A separate future long-context fine-tuning workflow will create a distinct run with `rope_scaling_factor > 1.0`; this plan does not change the factor during inference, evaluation, SWAG fine-tuning, resume, or export.
- MLX is the only model, training, inference, and evaluation backend, and Apple Silicon is the target.
- Throughput wins over peak memory only while the default workload still fits the Apple M5 10-core CPU, 10-core GPU, 24 GB target without critical memory pressure.
- Optional performance investigations may use source commit `3687f8b` and the committed harness/workload identities from Part 1. Historical phase-screen and final thresholds are report-only.
- Correctness-sensitive fine-tuning, resume, export, recovery, and retention fully rehash inputs before GPU initialization or deletion. Read-only inference/evaluation default to `manifest-trusted` and accept `full_verify=True` / `--full`.
- Writable artifact operations require local APFS and preserve descriptor-relative no-follow traversal, publication/writer/access locks, fsync/rename publication, recovery, and retention guarantees.
- Use Python 3.12.13 through `uv run`. Run every MLX pytest command and every benchmark outside the sandbox so Metal is available.
- Do not add dependencies or edit `uv.lock`. Do not change top-level files beyond the already-authorized `pyproject.toml` source mapping from Part 1.
- Before each phase commit, run `uv run ruff check v2`, `uv run ruff format --check v2`, and `uv run pytest v2/tests` outside the sandbox.
- Keep inference and training loops direct. Do not add generic pipelines, callbacks, registries, plugins, per-token Python conversion, or mutable array state captured only through compiled closures.
- Compiled cores accept and return only built-in MLX array trees. Host dataclasses and session owners unwrap before a compiled call and wrap returned arrays afterward without synchronizing.

---

## Phase Links

- [Master index and Phases 1–3](2026-08-01-v2-performance-first-refactor-part-1.md#master-phase-index)
- [Phase 4: Inference and evaluation](#phase-4-inference-and-evaluation)
- [Phase 5: LoRA and SWAG](#phase-5-lora-and-swag)
- [Phase 6: Unified CLI and final cutover](#phase-6-unified-cli-and-final-cutover)

## File Structure for Part 2

Create or complete:

- `v2/src/sml/inference.py` — artifact/model resolution, persistent session, prefill/decode, result metadata
- `v2/src/sml/evaluation.py` — batched log-likelihood/generation adapter and atomic result writing
- `v2/src/sml/data/swag.py` — provider identity, immutable encoded buckets, deterministic batch/cursor logic
- `v2/src/sml/training/lora.py` — FP32 adapters, strict state, deterministic merge/export weights
- `v2/src/sml/training/swag.py` — compiled ranking/update and fine-tuning run orchestration
- `v2/src/sml/cli.py` — TOML/config precedence, lazy workflow dispatch, expected-error rendering
- `v2/src/sml/__main__.py` — final unified entrypoint
- `v2/README.md` — only the replacement package/artifact/CLI workflow

Create mirrored unit/equivalence/integration tests. Phase 6 deletes every flat `v2/src/*.py` implementation and every replaced flat `v2/tests/test_*.py` test after their package replacements pass.

Test snippets omit routine imports only. Reuse the Part 1 fixtures for MLX availability, numeric/tree assertions, tiny tokenizer/prepared-data/base-run artifacts, and temporary APFS roots. Put inference request builders beside `test_inference.py`, provider fakes beside `test_swag.py` until subprocess stubs are shared through `v2/tests/fixtures/provider_stubs/`, and lm-eval fakes beside `test_evaluation.py`; every named helper in a snippet must be defined above its first use or supplied by `v2/tests/conftest.py` in the same task.

## Phase 4: Inference and Evaluation

### Task 4.1: Resolve Model Artifacts into a Persistent Non-Reentrant Session

**Files:**
- Create: `v2/src/sml/inference.py`
- Create: `v2/tests/unit/test_inference.py`
- Create: `v2/tests/integration/test_inference_workflow.py`
- Modify: `v2/src/sml/artifacts/checkpoint.py`

**Interfaces:**
- Frozen `ResolvedModel` records artifact kind, run identity, resolved step, checkpoint identity, run-step identity, verification level, model config, a fully loaded `LoadedTokenizer`, and owned BF16 inference-weight arrays. For a pretraining checkpoint, resolution validates the complete checkpoint schema including FP32 master metadata but loads only `model.safetensors` into the inference session; `full_verify=True` additionally rehashes both parameter payloads and proves the BF16 model tree is the exact cast of the FP32 master tree before returning owned inference weights. Its `tokenizer_identity` property delegates to the loaded tokenizer manifest, so inference and SWAG preparation use the same verified processor rather than reopening or accepting a second tokenizer path.
- `resolve_model_artifact(path, *, full_verify: bool) -> ResolvedModel` supports a pretraining run in this task; Phase 5 extends it for latest-only LoRA runs and self-contained exports. Direct `step-*` paths and historical step selectors are rejected.
- Frozen `InferenceRuntimeConfig(batch_size_buckets=(1, 2, 4, 8, 16), decode_chunk_size=8)` validates strictly increasing positive batch buckets. Length buckets are powers of two capped by and including the loaded model's authoritative effective context length; a request's bucket is selected from its required capacity `prompt_token_count + max_new_tokens`, not prompt length alone.
- `InferenceSession.from_checkpoint(path, *, full_verify=False, runtime=InferenceRuntimeConfig()) -> InferenceSession` loads the run's recovered latest tokenizer/model once and caches compiled functions by complete stable shape/policy key.
- The session owns immutable model state and a buffer pool but no request token/KV/logical-length/finished/key state between calls.
- A nonblocking call guard rejects overlapping `generate`, `generate_batch`, or evaluation scoring before leasing mutable buffers.
- A base-pretraining `ResolvedModel.model_config.rope_scaling_factor` must be exactly `1.0`, matching its run manifest. Resolution never substitutes a larger inference factor. LoRA/export resolution added in Phase 5 likewise preserves its source run's authoritative value.

- [ ] **Step 1: Write resolution, reuse, cleanup, and call-guard tests**

```python
def test_session_loads_latest_once_and_pins_identity(tiny_pretraining_run, monkeypatch):
    calls = 0
    real_loader = inference.load_owned_model_arrays

    def counted_loader(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_loader(*args, **kwargs)

    monkeypatch.setattr(inference, "load_owned_model_arrays", counted_loader)
    session = InferenceSession.from_checkpoint(tiny_pretraining_run)
    first_identity = session.model_identity
    assert session.resolved_model.model_config.rope_scaling_factor == 1.0
    publish_new_valid_step(tiny_pretraining_run, step=4)
    assert session.model_identity == first_identity
    assert calls == 1


def test_failed_call_cannot_contaminate_next_call(tiny_session, monkeypatch):
    monkeypatch.setattr(tiny_session, "_decode_chunk", raise_after_one_token)
    with pytest.raises(SMLRuntimeError, match="decode"):
        tiny_session.generate("first", GenerationRequest(max_new_tokens=2))
    monkeypatch.setattr(tiny_session, "_decode_chunk", deterministic_decode)
    assert tiny_session.generate("second", GenerationRequest(max_new_tokens=2)).token_ids == expected_second_tokens()


def test_overlapping_call_fails_before_state_mutation(tiny_session):
    with tiny_session._call_guard.acquire():
        with pytest.raises(SMLRuntimeError, match="non-reentrant"):
            tiny_session.generate("blocked", GenerationRequest(max_new_tokens=1))
    assert tiny_session.buffer_pool.active_leases == 0
```

Also test read-only stale-latest recovery without persistence, direct
`step-*` path and historical-selector rejection, shared access lock held through
owned-array evaluation, immutable scalar and nested array-group mappings from
`VerifiedCheckpointContents`, manifest-trusted versus full metadata, full
verification rejecting a master/working cast mismatch, inference loading no
training-only master or optimizer arrays into session ownership, prompt
overflow, and empty text without usable BOS.

- [ ] **Step 2: Run session tests and verify RED**

```bash
uv run pytest v2/tests/unit/test_inference.py v2/tests/integration/test_inference_workflow.py -k "resolve or session or guard or failed or latest" -v
```

Expected: FAIL because persistent inference APIs do not exist.

- [ ] **Step 3: Implement model resolution, owned loading, and buffer leasing**

Resolve artifacts once, hold the shared access lock until required safetensors
are validated and the BF16 working arrays have been evaluated into owned MLX
storage, then release it. Before the model resolver consumes the Part 1 reader,
copy/freeze its scalar mapping and outer/inner array-group mappings so assignment
or alias mutation fails. `full_verify=False` must still validate the complete
checkpoint schema/path/array metadata and report `manifest-trusted`;
`full_verify=True` rehashes both `master.safetensors` and
`model.safetensors` and checks the exact FP32-master-to-BF16-working cast
relationship before discarding the temporary master load. Construct the
tokenizer/model from copied run state only and preserve its authoritative
`rope_scaling_factor=1.0`. Do not retain master, optimizer, or trainer arrays in
`InferenceSession`. Implement the call guard with
`threading.Lock.acquire(blocking=False)` and a `finally` block that
clears/discards leased request storage before releasing the guard.
Session-owned cache wrappers pass only their `KVArrayState` tuple into compiled
functions and install returned state after evaluation; no `KVCache` object
crosses compilation.

Use these result types:

```python
@dataclass(frozen=True, slots=True)
class ModelIdentity:
    artifact_kind: str
    run_identity: str | None
    step: int | None
    checkpoint_identity: str | None
    run_step_identity: str | None
    tokenizer_identity: str
    verification: VerificationLevel
    latest_recovered: bool | None
    pruning_pending: bool | None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    token_ids: tuple[int, ...]
    seed: int | None
    model: ModelIdentity
```

- [ ] **Step 4: Verify artifact-resolution and session-lifecycle tests**

```bash
uv run pytest v2/tests/unit/test_inference.py v2/tests/integration/test_inference_workflow.py -k "resolve or session or guard or failed or latest" -v
```

Expected: all focused tests pass; sequential calls begin with empty token/KV
state, read-only latest recovery never persists repairs, direct step paths and
historical selectors fail, and verified checkpoint mappings cannot be mutated.

- [ ] **Step 5: Commit persistent session foundation**

```bash
git add v2/src/sml/inference.py v2/src/sml/artifacts/checkpoint.py v2/tests/unit/test_inference.py v2/tests/integration/test_inference_workflow.py
git commit -m "feat(v2): load persistent inference sessions"
```

### Task 4.2: Compile Unequal-Length Batched Prefill and Decode

**Files:**
- Modify: `v2/src/sml/inference.py`
- Modify: `v2/src/sml/model/cache.py`
- Modify: `v2/tests/unit/test_inference.py`
- Create: `v2/tests/equivalence/test_inference_equivalence.py`

**Interfaces:**
- Frozen `GenerationRequest(max_new_tokens: int, config: GenerationConfig = GenerationConfig(), include_prompt: bool = False)` rejects negative token counts; the CLI DTO defaults omitted `--max-new-tokens` to `128` before constructing it, while library callers choose the limit explicitly. `max_new_tokens` counts only generated continuation tokens. `GenerationResult.token_ids` and `.text` contain only that continuation unless `include_prompt=True`, in which case both contain prompt plus continuation.
- Frozen `InferenceConfig(checkpoint: Path, prompt: str, request: GenerationRequest, full_verify: bool = False, runtime: InferenceRuntimeConfig = InferenceRuntimeConfig())` is the typed one-shot CLI domain configuration. `infer(config: InferenceConfig) -> GenerationResult` constructs one latest-only session and delegates exactly once to `generate`; persistent library callers construct `InferenceSession` directly.
- `InferenceSession.generate(text, request) -> GenerationResult` delegates to the same batch engine as one request.
- `InferenceSession.generate_batch(items: Sequence[tuple[str, GenerationRequest]]) -> tuple[GenerationResult, ...]` restores caller order.
- An empty batch returns `()` before taking the call guard. For an omitted seed, the host allocator calls `secrets.randbits(32)` once per real request in caller order; tests monkeypatch that allocator, and the concrete value is returned in `GenerationResult` for replay.
- Frozen `GenerationKernelKey(temperature, top_p, repetition_penalty, no_repeat_ngram_size)` contains every sampling/processor field that controls compiled branches or constants; it excludes `seed`, `max_new_tokens`, and `include_prompt`, which are request state or host result formatting.
- Stable buckets are keyed by `(length_bucket, batch_size_bucket, GenerationKernelKey)`. `length_bucket` is the smallest configured power of two that fits `prompt_token_count + max_new_tokens`; reject a request whose required capacity exceeds the authoritative effective context length before leasing buffers. `batch_size_bucket` is the smallest configured value that fits the group; groups larger than the maximum are stably chunked. Every device array, including token/KV storage, masks, logical positions, lengths, finished flags, max-new-token limits, and PRNG keys, uses those fixed dimensions. Synthetic batch slots carry `request_mask=False`; their finite selector computation is masked afterward and has no result or persistent-state contribution.
- Per-request native MLX keys have shape `(2,)`; the bucket stores them as `(batch_size_bucket, 2)` only as vmap input. A compiled `mx.vmap(select_one_token, in_axes=(0, 0))` invokes categorical sampling with one `(2,)` key per real request and returns stacked token IDs plus next keys. The stacked key array is never passed directly to `mx.random.categorical`.
- Prefill/decode compile functions accept and return all mutable arrays explicitly and synchronize once per configured decode chunk.

- [ ] **Step 1: Write serial/batch, seed, padding, EOS, and processor equivalence tests**

```python
@pytest.mark.parametrize("mode", ["greedy", "sampled", "repetition", "no-repeat-ngram", "eos"])
def test_heterogeneous_batch_matches_serial_with_margin_rule(tiny_session, requests_for_mode, mode):
    serial = tuple(tiny_session.generate(text, request) for text, request in requests_for_mode)
    batched = tiny_session.generate_batch(requests_for_mode)
    assert_results_margin_aware_equal(serial, batched)


def test_seed_stream_is_invariant_to_bucket_neighbors(tiny_session):
    target = ("short", GenerationRequest(max_new_tokens=4, config=GenerationConfig(temperature=0.8, top_p=0.9, seed=17)))
    alone = tiny_session.generate_batch([target])[0]
    mixed = tiny_session.generate_batch([long_seeded_request(99), target, medium_seeded_request(33)])[1]
    assert mixed.seed == alone.seed == 17
    assert mixed.token_ids == alone.token_ids


def test_omitted_seed_is_allocated_before_reordering(tiny_session):
    result = tiny_session.generate_batch([unseeded_request("a"), unseeded_request("b")])
    assert all(item.seed is not None for item in result)
    replay = tiny_session.generate("a", replace_request_seed(unseeded_request("a")[1], result[0].seed))
    assert replay.token_ids == result[0].token_ids


def test_one_batch_preserves_distinct_generation_policies(tiny_session):
    prompt = "policy-sensitive repeated prompt"
    items = [
        (prompt, GenerationRequest(4, GenerationConfig(temperature=0.0, seed=1))),
        (prompt, GenerationRequest(4, GenerationConfig(temperature=0.8, top_p=0.9, seed=2))),
        (prompt, GenerationRequest(4, GenerationConfig(temperature=0.0, repetition_penalty=1.2, seed=3))),
        (prompt, GenerationRequest(4, GenerationConfig(temperature=0.0, no_repeat_ngram_size=2, seed=4))),
    ]
    serial = tuple(tiny_session.generate(text, request) for text, request in items)
    assert len({result.token_ids for result in serial}) == len(items)
    batched = tiny_session.generate_batch(items)
    assert_results_margin_aware_equal(serial, batched)


def test_batch_cardinality_reuses_fixed_compiled_bucket(tiny_session, compile_spy):
    tiny_session.generate_batch(seed_requests(3))
    compiled_after_three = set(compile_spy.keys)
    tiny_session.generate_batch(seed_requests(4))
    assert set(compile_spy.keys) == compiled_after_three
    assert len(compiled_after_three) == 1
    assert next(iter(compiled_after_three)).batch_size_bucket == 4


def test_sampling_vmaps_scalar_keys_and_ignores_synthetic_slots(tiny_session):
    bucket = tiny_session._bucketize(seed_requests(3))[0]
    assert bucket.keys.shape == (4, 2)
    assert bucket.request_mask.tolist() == [True, True, True, False]
    selected, next_keys = vmapped_select_one_token(
        fixed_bucket_logits(batch_size=4),
        bucket.keys,
        bucket.request_mask,
        bucket.kernel_key,
    )
    mx.eval(selected, next_keys)
    assert selected.shape == (4,)
    assert next_keys.shape == (4, 2)
    assert source_contains(inference_module, "mx.vmap(select_one_token")
    assert source_has_none_of(inference_module, ["mx.random.categorical(logits, key=keys)"])
```

The mixed-policy fixture must use safe-margin logits chosen so greedy, seeded top-p, repetition penalty, and no-repeat n-gram each produce the distinct continuation asserted above; a coincidental equal-output fixture is invalid.

- [ ] **Step 2: Run generation-equivalence tests and verify RED**

```bash
uv run pytest v2/tests/unit/test_inference.py v2/tests/equivalence/test_inference_equivalence.py -v
```

Expected: FAIL because batching/chunked decode is absent.

- [ ] **Step 3: Implement stable buckets and explicit generation state**

Return immediately for an empty batch. Otherwise encode and apply BOS before bucketing; reject zero-token prompts and any `prompt_token_count + max_new_tokens` beyond effective context. Normalize and validate each request's `GenerationKernelKey`, allocate omitted concrete seeds with the host allocator in caller order, then stably group/chunk by `(length_bucket, batch_size_bucket, kernel_key)` without changing caller-order seed allocation. Create each MLX key directly from that seed and carry it through reordering. Pad every group to its batch-size bucket with finite synthetic request state plus `request_mask=False`. Compile/cache prefill and decode for the complete bucket key and pass max-new-token limits and stacked PRNG keys as explicit arrays. Build one policy-specific `select_one_token(logits_row, key)` closure per `GenerationKernelKey` around Part 1's array-only selector and apply `mx.vmap(select_one_token, in_axes=(0, 0))`; mask synthetic selected tokens and next keys afterward so each categorical call receives exactly one `(2,)` key. Gather prefill logits at each last real prefix token. Exclude padding cache slots and synthetic requests from attention, write/rotate at logical positions, keep finished/EOS state on device, preallocate token/cache capacity, decode in compiled chunks, and restore only real results to original order. Do not concatenate full prefixes per token or convert device arrays to Python inside the loop.

- [ ] **Step 4: Verify equivalence, state reset, and compiled-source constraints**

```bash
uv run pytest v2/tests/unit/test_inference.py v2/tests/equivalence/test_inference_equivalence.py v2/tests/integration/test_inference_workflow.py -v
```

Expected: tests pass; token equality is enforced for safe-margin fixtures, boundary fixtures match logits/processor masks, bucket order cannot change a seeded stream, cardinalities within one batch-size bucket cause no recompile, each categorical call sees one `(2,)` key, synthetic slots are inert, and a mixed-policy caller batch matches four independently configured serial calls.

- [ ] **Step 5: Commit batched generation**

```bash
git add v2/src/sml/inference.py v2/src/sml/model/cache.py v2/tests/unit/test_inference.py v2/tests/equivalence/test_inference_equivalence.py v2/tests/integration/test_inference_workflow.py
git commit -m "perf(v2): batch compiled prefill and decode"
```

### Task 4.3: Add Batched lm-eval Scoring and Pinned Result Metadata

**Files:**
- Create: `v2/src/sml/evaluation.py`
- Create: `v2/tests/unit/test_evaluation.py`
- Create: `v2/tests/equivalence/test_evaluation_equivalence.py`
- Create: `v2/tests/integration/test_evaluation_workflow.py`

**Interfaces:**
- `LoglikelihoodRequest(context: str, continuation: str)` and `LoglikelihoodResult(log_likelihood: float, greedy_match: bool)`.
- `score_loglikelihood_batch(session, requests, *, padding) -> tuple[LoglikelihoodResult, ...]` reuses the session's length and batch-size buckets, pads with masked finite synthetic rows, and synchronizes once per fixed-shape batch.
- `SMLEvalLM` implements only `loglikelihood` and `generate_until`; unsupported request methods raise `SMLRuntimeError`.
- Frozen `EvaluationConfig(checkpoint: Path, tasks: tuple[Literal["hellaswag", "winogrande"], ...], output: Path, full_verify: bool = False, padding: Literal["left", "right"] = "right", runtime: InferenceRuntimeConfig = InferenceRuntimeConfig(), limit: int | None = None)` requires at least one task and a positive optional limit.
- `evaluate(config)` returns the current strict `EvaluationResult` and supports
  repeated tasks from exactly `hellaswag` and `winogrande`. The historical
  persisted v1 exact field set and `sml-evaluation-result-v1` identity domain
  remain frozen; strict readers expose its recovery state as unavailable.
  Current writes are strict v2 in the `sml-evaluation-result-v2` domain, adding
  Boolean `latest_recovered` and `pruning_pending` to the pinned
  `ModelIdentity`. Publication preserves the complete provider result and
  sorted provider versions atomically at the required output path.

- [ ] **Step 1: Write serial/batched score and adapter contract tests**

```python
@pytest.mark.parametrize("padding", ["left", "right"])
def test_padded_loglikelihood_matches_serial(tiny_session, heterogeneous_scoring_requests, padding):
    serial = tuple(score_loglikelihood_batch(tiny_session, [request], padding=padding)[0] for request in heterogeneous_scoring_requests)
    batched = score_loglikelihood_batch(tiny_session, heterogeneous_scoring_requests, padding=padding)
    assert_loglikelihood_results_close(serial, batched, atol=2e-2, rtol=2e-2)


def test_evaluation_result_pins_resolved_identity(tiny_pretraining_run, fake_lm_eval):
    result = evaluate(tiny_evaluation_config(tiny_pretraining_run, tasks=("hellaswag",)))
    pinned = result.model
    publish_new_valid_step(tiny_pretraining_run, step=pinned.step + 1)
    persisted = read_evaluation_result(result.output)
    assert persisted.model == pinned
    assert persisted.tasks == ("hellaswag",)


def test_score_cardinality_reuses_fixed_compiled_bucket(tiny_session, compile_spy):
    score_loglikelihood_batch(tiny_session, scoring_requests(3), padding="right")
    compiled_after_three = set(compile_spy.scoring_keys)
    score_loglikelihood_batch(tiny_session, scoring_requests(4), padding="right")
    assert set(compile_spy.scoring_keys) == compiled_after_three
    assert len(compiled_after_three) == 1
    assert next(iter(compiled_after_three)).batch_size_bucket == 4
```

Also test continuation-only scoring, greedy-match flag, empty-context prefix token, no-prefix failure before bucketing, stop strings, `generate_until` batching, unsupported methods, task rejection, and no network in the normal suite.

- [ ] **Step 2: Run evaluation tests and verify RED**

```bash
uv run pytest v2/tests/unit/test_evaluation.py v2/tests/equivalence/test_evaluation_equivalence.py v2/tests/integration/test_evaluation_workflow.py -v
```

Expected: FAIL because the package evaluation adapter is absent.

- [ ] **Step 3: Implement on-device target gathering and lazy provider adapter**

Tokenize context/continuation according to the preserved joint-boundary contract, insert the configured empty-context prefix token, and group by fixed `(length_bucket, batch_size_bucket, padding)` scoring keys using the session runtime buckets. Pad with finite masked synthetic rows, create Boolean attention/score/request masks, gather target logits from `logits[..., :-1, :]`, compute FP32 log likelihood, and exclude padding/synthetic rows. Call `generate_batch` for generation requests. Lazy-import `lm_eval` only inside evaluation construction. Serialize results through atomic immutable publication or atomic file replacement with the resolved `ModelIdentity`, task config, and relevant provider versions.

- [ ] **Step 4: Verify equivalence and integration**

```bash
uv run pytest v2/tests/unit/test_evaluation.py v2/tests/equivalence/test_evaluation_equivalence.py v2/tests/integration/test_evaluation_workflow.py -v
```

Expected: all tests pass; left/right padding layouts are inert within tolerance and result identity cannot change after `latest.json` advances.

- [ ] **Step 5: Commit evaluation runtime**

```bash
git add v2/src/sml/evaluation.py v2/tests/unit/test_evaluation.py v2/tests/equivalence/test_evaluation_equivalence.py v2/tests/integration/test_evaluation_workflow.py
git commit -m "perf(v2): batch evaluation requests"
```

### Task 4.4: Verify Inference and Evaluation Workflows

**Interfaces:**
- Consumes the functionally verified Phase 3 package and artifact contracts.
- Proves persistent latest-only inference, batched evaluation, and
  read-only/full verification behavior through production workflows.

- [ ] **Step 1: Run full correctness/format verification**

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git status --short
```

- [ ] **Step 2: Run focused production workflow verification**

```bash
uv run pytest v2/tests/integration/test_inference_workflow.py v2/tests/integration/test_evaluation_workflow.py -v
```

Expected: persistent-session generation, unequal-length batching, latest-only
artifact resolution, direct step-path rejection, evaluation scoring, and both
verification levels pass.

- [ ] **Step 3: Run CLI smoke coverage available at this phase**

```bash
uv run python -m sml --help
```

- [ ] **Step 4: Optionally collect inference performance diagnostics**

The historical five-pair inference comparison may be run when useful. Its
absence, noise, thermal rejection, or ratios do not block Phase 5 and no
`phase-4.json` evidence commit is required.

## Phase 5: LoRA and SWAG

### Task 5.1: Implement Strict FP32 LoRA Layers and Deterministic Merge

**Files:**
- Create: `v2/src/sml/training/lora.py`
- Create: `v2/tests/unit/training/test_lora.py`
- Create: `v2/tests/equivalence/test_lora_equivalence.py`

**Interfaces:**
- LoRA configuration has these exact fields/defaults:

```python
@dataclass(frozen=True, slots=True)
class LoRAInitializerConfig:
    lora_a: float = 0.01
    lora_b: float = 0.0


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    scaling_mode: Literal["lora", "rslora"] = "rslora"
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    initializer: LoRAInitializerConfig = field(default_factory=LoRAInitializerConfig)


@dataclass(frozen=True, slots=True)
class LoRAPrecisionConfig:
    frozen_base_dtype: Literal["bfloat16"] = "bfloat16"
    adapter_parameter_dtype: Literal["float32"] = "float32"
    gradient_accumulator_dtype: Literal["float32"] = "float32"
    optimizer_state_dtype: Literal["float32"] = "float32"
    update_dtype: Literal["float32"] = "float32"
    dynamic_loss_scaling: Literal[False] = False
```

- Target names must be unique and drawn from `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`; empty targets are rejected. Initializer standard deviations are finite/nonnegative. Rank, alpha, scaling, and dropout retain the captured legacy defaults while replacing the legacy target Boolean aliases with one canonical tuple.
- `LoRAPrecisionConfig` describes only the copied frozen BF16 working base and trainable FP32 adapter runtime. It is distinct from Part 1's `PrecisionConfig`, whose `master_weights=True` contract remains attached to the source pretraining run/base snapshot provenance and is never misapplied to adapter state.
- `LoRALinear(linear, config, *, key)` freezes BF16 base and registers FP32 `lora_a` / `lora_b`.
- `apply_lora(model, config, *, key)`, `lora_state_dict`, `load_lora_state_dict`, and `merged_model_weights`.
- Live adapter formula is `scale_fp32 * ((input_fp32 @ A_fp32.T) @ B_fp32.T)`, cast to BF16 before adding to BF16 base output.
- `scale_fp32` is exactly `alpha / rank` for `scaling_mode="lora"` and `alpha / sqrt(rank)` for `scaling_mode="rslora"`, computed in FP32.
- Merged formula is `cast_bf16(cast_fp32(base_bf16) + scale_fp32 * (B_fp32 @ A_fp32))` without mutating/deep-copying the model.

- [ ] **Step 1: Write strict state, dtype, forward, merge, and key tests**

```python
def test_lora_dtype_boundaries_and_live_formula(tiny_model):
    adapted = apply_lora(tiny_model, tiny_lora_config(dropout=0.0), key=mx.random.key(4))
    layer = adapted.layers[0].self_attn.q_proj
    x = mx.ones((1, 2, layer.base.weight.shape[1]), dtype=mx.bfloat16)
    actual, _next_key = layer(x, key=mx.random.key(8), training=False)
    expected_adapter = layer.scale.astype(mx.float32) * ((x.astype(mx.float32) @ layer.lora_a.T) @ layer.lora_b.T)
    expected = layer.base(x) + expected_adapter.astype(mx.bfloat16)
    mx.eval(actual, expected)
    assert layer.base.weight.dtype == mx.bfloat16
    assert layer.lora_a.dtype == mx.float32
    assert layer.lora_b.dtype == mx.float32
    assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_merged_weight_matches_exact_array_formula_without_mutation(tiny_adapted_model):
    before = lora_state_dict(tiny_adapted_model)
    merged = merged_model_weights(tiny_adapted_model)
    module = tiny_adapted_model.layers[0].self_attn.q_proj
    expected = (module.base.weight.astype(mx.float32) + module.scale.astype(mx.float32) * (module.lora_b @ module.lora_a)).astype(mx.bfloat16)
    assert mx.array_equal(merged["layers.0.self_attn.q_proj.weight"], expected).item()
    assert_lora_state_equal(lora_state_dict(tiny_adapted_model), before)


def test_lora_forward_matches_weight_pinned_legacy_reference(
    tiny_model_config,
    tiny_lora_config,
    legacy_arrays,
    legacy_control,
):
    model = SMLLanguageModel(tiny_model_config, key=mx.random.key(4))
    load_legacy_model_state(model, legacy_arrays, legacy_control)
    adapted = apply_lora(model, tiny_lora_config(dropout=0.0), key=mx.random.key(5))
    load_legacy_lora_state(adapted, legacy_arrays, legacy_control)
    actual, _next_key = adapted.layers[0].self_attn.q_proj(
        legacy_arrays["lora.input"],
        key=mx.random.key(8),
        training=False,
    )
    mx.eval(actual)
    assert_close(actual, legacy_arrays["lora.output"], atol=2e-2, rtol=2e-2)
```

Also reject missing/additional/wrong-shape/wrong-dtype adapter keys, test explicit dropout-key replay, target classification, live versus merged output BF16 tolerance, and plain inference parameter names. Every comparison in `test_lora_equivalence.py` must load both captured base and adapter state before evaluating the legacy output; seed-only parameter reconstruction is forbidden.

- [ ] **Step 2: Run LoRA tests and verify RED**

```bash
uv run pytest v2/tests/unit/training/test_lora.py v2/tests/equivalence/test_lora_equivalence.py -v
```

Expected: FAIL because package LoRA runtime is absent.

- [ ] **Step 3: Implement keyed FP32 adapters and direct merged dictionary**

Wrap only configured `nn.Linear` targets, freeze all base leaves, initialize A/B from `config.initializer` with explicit split keys, use keyed dropout rather than `nn.Dropout`, and unfreeze only adapter arrays. Strict loading compares exact flattened names/shapes/dtypes before assignment. Build merged weights by traversing named parameters, replacing wrapped base weight names with plain inference names and excluding all adapter arrays.

- [ ] **Step 4: Verify LoRA formula/equivalence tests**

```bash
uv run pytest v2/tests/unit/training/test_lora.py v2/tests/equivalence/test_lora_equivalence.py -v
```

Expected: tests pass; merge arrays match exactly, output comparison uses pinned BF16 tolerance, and source adapters remain intact.

- [ ] **Step 5: Commit LoRA primitives**

```bash
git add v2/src/sml/training/lora.py v2/tests/unit/training/test_lora.py v2/tests/equivalence/test_lora_equivalence.py
git commit -m "feat(v2): add strict fp32 lora adapters"
```

### Task 5.2: Build Immutable Encoded SWAG Buckets

**Files:**
- Create: `v2/src/sml/data/swag.py`
- Create: `v2/tests/unit/data/test_swag.py`
- Create: `v2/tests/integration/test_swag_data_workflow.py`

**Interfaces:**
- SWAG source/preparation configuration is exact:

```python
class SwagProvider(Protocol):
    def resolve(self, source: SwagSourceConfig) -> ResolvedSwagSource: ...
    def iter_rows(self, resolved: ResolvedSwagSource) -> Iterator[Mapping[str, object]]: ...


@dataclass(frozen=True, slots=True)
class SwagSourceConfig:
    revision: str
    backend: Literal["huggingface-datasets"] = "huggingface-datasets"
    namespace: str = "allenai"
    name: str = "swag"
    dataset_config: str = "regular"
    split: str = "train"


@dataclass(frozen=True, slots=True)
class SwagPreparationConfig:
    provider: SwagProvider = field(repr=False, compare=False)
    source: SwagSourceConfig
    preprocessing_schema_version: int = 1
    join_policy: Literal["separate-context-ending-v1"] = "separate-context-ending-v1"
    overlength_policy: Literal["drop-complete-row-v1"] = "drop-complete-row-v1"
    bos_policy: Literal["context-bos-v1"] = "context-bos-v1"
    eos_policy: Literal["scored-ending-eos-v1"] = "scored-ending-eos-v1"
    maximum_length: int = 256
    bucket_boundaries: tuple[int, ...] = (64, 128, 256)
    maximum_examples: int | None = None
```

- `source.revision` must resolve to an immutable provider commit. `ResolvedSwagSource` records the normalized source fields, resolved commit, provider fingerprint, and provider package/version. The runtime provider object is never serialized or included in artifact identity; all resolved source fields are. Tokenizer/model identity and special IDs come from the fully verified `base`, not duplicate user-supplied config fields.
- `prepare_swag_bundle(config, base: ResolvedModel, output: Path) -> SwagDataBundle` writes memory-mappable arrays under `buckets/length-NNNN/`.
- Per bucket: `input_ids:int32`, `valid_token_mask:bool`, `score_mask:bool` with shape `(examples, 4, bucket_length)`; `labels:int32` with `(examples,)`.
- `load_swag_bundle(path, verification) -> SwagDataBundle` validates all preprocessing/base compatibility before base/adapters/optimizer allocation.

- [ ] **Step 1: Write cache identity, encoding, mask, drop, and bucket tests**

```python
def test_context_and_endings_are_encoded_separately_and_eos_is_scored(fake_provider, base_model_with_recording_tokenizer, tmp_path):
    recording_tokenizer = base_model_with_recording_tokenizer.tokenizer
    bundle = prepare_swag_bundle(tiny_swag_config(fake_provider), base_model_with_recording_tokenizer, tmp_path / "swag")
    assert recording_tokenizer.calls[0] == fake_provider.rows[0]["context"]
    assert recording_tokenizer.calls[1:5] == tuple(fake_provider.rows[0]["endings"])
    bucket = bundle.buckets[0]
    assert not bucket.score_mask[:, :, 0].any()
    assert bucket.score_mask[bucket.valid_token_mask].any()
    assert_eos_positions_are_scored(bucket, eos_token_id=2)


@pytest.mark.parametrize("field", SWAG_IDENTITY_FIELDS)
def test_every_preprocessing_field_changes_bundle_identity(field, swag_config, base_model, tmp_path):
    first = prepare_swag_bundle(swag_config, base_model, tmp_path / "first")
    changed = prepare_swag_bundle(change_identity_field(swag_config, field), base_model, tmp_path / field)
    assert changed.manifest.identity != first.manifest.identity


def test_overlength_candidate_drops_complete_row(fake_provider_with_valid_and_long_row, base_model, tmp_path):
    bundle = prepare_swag_bundle(tiny_swag_config(fake_provider_with_valid_and_long_row), base_model, tmp_path / "swag")
    assert bundle.manifest.dropped_overlength_rows == 1
    assert bundle.manifest.example_count == 1
    assert all(0 <= label < 4 for bucket in bundle.buckets for label in bucket.labels)
```

Add failures for zero usable examples, not-four candidates, label/token range, missing scored token, wrong tokenizer identity/vocab/special IDs/preprocessing, cache maximum beyond effective context, and an unavailable uncached provider. The provider failure must name its namespace/config/revision and computed cache key.

- [ ] **Step 2: Run SWAG-data tests and verify RED**

```bash
uv run pytest v2/tests/unit/data/test_swag.py v2/tests/integration/test_swag_data_workflow.py -v
```

Expected: FAIL because encoded SWAG bundles do not exist.

- [ ] **Step 3: Implement batched source/tokenizer ingestion and atomic NPY buckets**

If the target already exists, compare its authoritative source/preprocessing/base projections to the requested config and fully verify it before returning, without resolving/importing the provider. For a new target, resolve and record the immutable provider commit/fingerprint/version before rows are processed. The production provider lazy-imports `datasets`; batch source/tokenizer calls, encode context and each normalized ending separately, concatenate, append EOS, drop the entire row when any candidate is too long, and place each row in the smallest finite bucket that fits its longest candidate. Construct masks exactly as the interfaces specify, validate finite usable data, hash arrays while writing, and publish manifest last.

- [ ] **Step 4: Verify data contracts and offline reuse**

```bash
uv run pytest v2/tests/unit/data/test_swag.py v2/tests/integration/test_swag_data_workflow.py -v
```

Expected: all tests pass; reopening a valid bundle performs no provider/network call, and every identity field causes a cache miss.

- [ ] **Step 5: Commit encoded SWAG bundles**

```bash
git add v2/src/sml/data/swag.py v2/tests/unit/data/test_swag.py v2/tests/integration/test_swag_data_workflow.py
git commit -m "feat(v2): cache encoded swag buckets"
```

### Task 5.3: Compile Mean-Normalized SWAG Ranking and Fixed-Shape Batches

**Files:**
- Create: `v2/src/sml/training/swag.py`
- Create: `v2/tests/unit/training/test_swag.py`
- Create: `v2/tests/equivalence/test_swag_equivalence.py`
- Modify: `v2/src/sml/data/swag.py`
- Modify: `v2/src/sml/training/common.py`
- Modify: `v2/tests/unit/training/test_common.py`

**Interfaces:**
- `SwagCursor(epoch, bucket_order_position, row_offset)` names the next real example and never a padded slot.
- `SwagBatchEnvelope` owns read-only NumPy input/mask/label/example-mask arrays plus normalized `cursor_after`, with the same idempotent release/context-manager contract as the Part 1 pretraining envelope. `SwagBatch.from_envelope(envelope)` performs the main-thread MLX transfers and returns on-device input/masks/labels, Boolean `example_mask`, and the cursor.
- `SwagBatchStream` is a context-managed bounded CPU prefetcher using `LoaderConfig.prefetch_depth`. It deterministically permutes buckets/rows through a local `np.random.Generator(np.random.PCG64(np.random.SeedSequence([LoaderConfig.epoch_seed, epoch])))`, pads each bucket tail to configured batch size with finite synthetic examples, and consumes every real example once. Every queued padded-tail/staged array has distinct leased storage until its envelope is transferred and released; producer threads never call MLX. `SwagTrainingConfig.seed` is reserved for adapter/dropout PRNG state.
- `score_candidates(logits, input_ids, score_mask) -> mx.array` returns FP32 mean continuation-token log likelihood.
- Per real example, ranking loss is exactly `mx.logsumexp(candidate_scores) - candidate_scores[label]` in FP32 and accuracy is `argmax(candidate_scores) == label`. Ranking kernels return the masked additive loss sum, correct count, and valid-example count; accumulation divides gradients/loss once by the total valid-example count.
- `adamw_fp32_update(parameters, gradients, state, config, weight_decay_tree) -> tuple[dict, AdamState]` performs the same internal configured FP32 AdamW formula as base training but preserves every FP32 adapter parameter as FP32; it never calls the base-only `adamw_mixed_precision_update` wrapper, never creates a second master tree for adapters, and never casts an adapter leaf to BF16.
- `default_swag_optimizer_config() -> OptimizerConfig` returns `OptimizerConfig(learning_rate=1e-4, schedule_steps=8_192)`; the shared weight-decay defaults already keep `lora_a`/`lora_b` at zero.
- Frozen `SwagKernelConfig(accumulation_steps, gradient_clip_norm, compile)` contains only stable kernel policy and is derived exactly once from `loader.gradient_accumulation_steps`, `optimizer.gradient_clip_norm`, and `config.compile`. `build_swag_kernels(model, config, weight_decay_tree) -> SwagKernels` returns host wrappers around eager/compiled ranking microsteps and an optimizer step. Private compiled cores accept and return only built-in dict/tuple trees containing the frozen BF16 base arrays, FP32 adapters, FP32 moments/accumulators, int32 counters, and explicit key. `AdamState` and SWAG trainer-state dataclasses are unwrapped before each compiled call and reconstructed afterward without synchronization; they never cross `mx.compile`. Frozen BF16 base arrays remain unchanged and are never optimizer leaves.

- [ ] **Step 1: Write formula, padding, weighting, cursor, and compiled-state tests**

```python
def test_candidate_score_is_fp32_mean_including_eos():
    logits = fixed_logits(dtype=mx.bfloat16)
    input_ids, score_mask = score_fixture_with_unequal_lengths_and_eos()
    scores = score_candidates(logits, input_ids, score_mask)
    expected = direct_fp32_target_logit_minus_logsumexp_mean(logits, input_ids, score_mask)
    mx.eval(scores, expected)
    assert scores.dtype == mx.float32
    assert_close(scores, expected, atol=1e-6, rtol=1e-6)


def test_padded_tail_matches_unpadded_example_weighted_update(tiny_swag_runtime):
    padded = tiny_swag_runtime.train_one_epoch(fixed_batch_size=4)
    eager = tiny_swag_runtime.train_one_epoch_unpadded_reference()
    assert padded.real_examples == eager.real_examples
    assert_tree_close(padded.adapters, eager.adapters, atol=1e-5, rtol=1e-5)
    assert padded.cursor == eager.cursor == SwagCursor(epoch=1, bucket_order_position=0, row_offset=0)


def test_fp32_adam_preserves_adapter_dtype_and_formula():
    parameters = {"lora_a": mx.array([[1.0, -2.0]], dtype=mx.float32)}
    gradients = {"lora_a": mx.array([[0.25, -0.5]], dtype=mx.float32)}
    state = initialize_adam_state(parameters)
    config = optimizer_config(weight_decay=0.1)
    weight_decay_tree = {"lora_a": True}
    updated, next_state = adamw_fp32_update(
        parameters,
        gradients,
        state,
        config,
        weight_decay_tree,
    )
    expected, expected_state = direct_fp32_adamw_oracle(
        parameters,
        gradients,
        state,
        config,
        weight_decay_tree,
    )
    mx.eval(updated, next_state.to_tree(), expected, expected_state.to_tree())
    assert updated["lora_a"].dtype == mx.float32
    assert next_state.first_moments["lora_a"].dtype == mx.float32
    assert next_state.second_moments["lora_a"].dtype == mx.float32
    assert_tree_close(updated, expected, atol=1e-6, rtol=1e-6)


def test_default_swag_optimizer_preserves_legacy_learning_rate_and_zero_adapter_decay():
    config = default_swag_optimizer_config()
    assert config.learning_rate == 1e-4
    assert config.schedule_steps == 8_192
    assert config.weight_decay.lora_a == 0.0
    assert config.weight_decay.lora_b == 0.0


def test_two_compiled_swag_updates_keep_base_bf16_and_all_optimizer_state_fp32(tiny_swag_runtime):
    before_base = tree_map(lambda value: mx.array(value), tiny_swag_runtime.base_parameters)
    eager = tiny_swag_runtime.run_two_updates(compiled=False)
    compiled = tiny_swag_runtime.run_two_updates(compiled=True)
    mx.eval(
        eager.adapters,
        eager.optimizer.to_tree(),
        compiled.base_parameters,
        compiled.adapters,
        compiled.optimizer.to_tree(),
        before_base,
    )
    assert_tree_dtypes(compiled.base_parameters, mx.bfloat16)
    assert_tree_equal(compiled.base_parameters, before_base)
    assert_tree_dtypes(compiled.adapters, mx.float32)
    assert_tree_dtypes(compiled.optimizer.first_moments, mx.float32)
    assert_tree_dtypes(compiled.optimizer.second_moments, mx.float32)
    assert_tree_close(compiled.adapters, eager.adapters, atol=1e-5, rtol=1e-5)


def test_compiled_swag_cores_accept_only_builtin_array_trees(tiny_swag_runtime):
    kernels = tiny_swag_runtime.kernels(compiled=True)
    core_inputs = tiny_swag_runtime.core_inputs()
    assert_builtin_array_tree(core_inputs)
    core_outputs = kernels.compiled_microstep_core(*core_inputs)
    mx.eval(core_outputs)
    assert_builtin_array_tree(core_outputs)
    assert source_has_none_of(
        swag_module,
        ["mx.compile(lambda state", "mx.compile(lambda batch", "mx.compile(lambda config"],
    )


def test_swag_value_and_grad_targets_adapters_only():
    assert source_contains(swag_module, "mx.value_and_grad(adapter_loss, argnums=0)")
    assert source_has_none_of(
        swag_module,
        ["mx.value_and_grad(combined_parameters", "nn.value_and_grad(model"],
    )
```

Place `test_fp32_adam_preserves_adapter_dtype_and_formula` and `direct_fp32_adamw_oracle` in `test_common.py`; place the ranking, padded-tail, prefetch, and compiled SWAG tests in `test_swag.py`. Define the oracle above its first test; it independently evaluates the saved optimizer configuration and weight-decay Boolean tree in NumPy/explicit scalar arithmetic and must not call either production update helper. Also prove finite per-slot losses before masking, synthetic slots have no loss/accuracy/gradient/progress/cursor contribution, a full prefetch queue owns distinct read-only storage and cannot advance the committed cursor, bucket-tail compilation shape is fixed, epoch transition commits with update, and second compiled step sees returned adapter/optimizer/key state.

- [ ] **Step 2: Run SWAG runtime tests and verify RED**

```bash
uv run pytest v2/tests/unit/training/test_common.py v2/tests/unit/training/test_swag.py v2/tests/equivalence/test_swag_equivalence.py -v
```

Expected: FAIL because SWAG stream/ranking kernels are absent.

- [ ] **Step 3: Implement fixed-shape bucket iteration and target-only log likelihood**

Create finite synthetic candidates with at least one valid scored token and set their example mask false. Prefetch only NumPy envelopes; the main loop converts each stored array to MLX, releases the envelope immediately after those non-aliasing transfers, and passes only its built-in device array tree into kernels. Flatten batch/candidate only inside compiled kernels. Compute `target_logit - mx.logsumexp(logits, axis=-1)` without materializing full log probabilities, mask continuation positions including EOS, sum then divide per candidate by its score-token count. Return loss numerator/count; accumulate FP32 adapter-gradient numerators and divide by total real-example count at the optimizer step. Implement `adamw_fp32_update` beside the base update by sharing only an internal FP32 AdamW formula helper; the public mixed-precision-base and FP32-adapter update functions have explicit post-update dtype assertions and distinct result contracts. Implement private `ranking_microstep_core` and `swag_optimizer_step_core` functions over built-in array trees, compile those functions directly, and keep model/config/batch/state dataclasses in the host wrappers. Define `adapter_loss(adapters, frozen_base, batch_arrays, key)`, merge the two parameter trees only for the pure `model.forward_arrays` call, and construct its transform with `mx.value_and_grad(adapter_loss, argnums=0)` so the frozen base stays an explicit dependency but MLX differentiates only adapters. The SWAG optimizer step must call only `adamw_fp32_update`, pass/return all adapter and optimizer state explicitly, and prove frozen BF16 base leaves are byte-unchanged.

- [ ] **Step 4: Verify formula oracle, cursor, and eager/compiled equivalence**

```bash
uv run pytest v2/tests/unit/training/test_common.py v2/tests/unit/training/test_swag.py v2/tests/equivalence/test_swag_equivalence.py -v
```

Expected: tests pass; legacy summed fixture differs where intended while the direct FP32 mean oracle matches, two eager/compiled updates agree, adapters/moments remain FP32, and the BF16 base is unchanged.

- [ ] **Step 5: Commit compiled SWAG runtime**

```bash
git add v2/src/sml/data/swag.py v2/src/sml/training/common.py v2/src/sml/training/swag.py v2/tests/unit/training/test_common.py v2/tests/unit/training/test_swag.py v2/tests/equivalence/test_swag_equivalence.py
git commit -m "perf(v2): compile mean-normalized swag ranking"
```

### Task 5.4: Create Self-Contained LoRA Runs, Exact Resume, and Merged Export

**Files:**
- Modify: `v2/src/sml/training/swag.py`
- Modify: `v2/src/sml/training/lora.py`
- Modify: `v2/src/sml/artifacts/checkpoint.py`
- Modify: `v2/src/sml/inference.py`
- Create: `v2/tests/integration/test_swag_workflow.py`

**Interfaces:**
- Frozen `SwagTrainingConfig` composes the shared policies with these exact fields/defaults:

```python
@dataclass(frozen=True, slots=True)
class SwagTrainingConfig:
    base_checkpoint: Path
    data: Path
    output_run: Path
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    optimizer: OptimizerConfig = field(default_factory=default_swag_optimizer_config)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    checkpoint: CheckpointPolicy = field(
        default_factory=lambda: CheckpointPolicy(interval=500),
    )
    precision: LoRAPrecisionConfig = field(default_factory=LoRAPrecisionConfig)
    maximum_steps: int | None = 8_192
    maximum_epochs: int | None = 5
    log_interval: int = 10
    seed: int = 42
    compile: bool = True
```

- It validates that at least one termination limit exists, stops at the first limit reached when both are present, and requires BF16 frozen base plus FP32 adapters/optimizer state. It contains no provider object, runtime hook, or test-only dependency. `WeightDecayPolicy.lora_a` and `.lora_b` default to `0.0`, preserving the legacy adapter-decay policy; SWAG classifies adapter leaves by those fields rather than `other`.
- Frozen `SwagTrainingResult(run: Path, step: int, epoch: int, examples: int)` is the result type for fresh and resumed fine-tuning.
- `finetune(config: SwagTrainingConfig) -> SwagTrainingResult` creates a new LoRA run only.
- `resume_finetune(run, *, data, overrides) -> SwagTrainingResult` accepts only termination/observability overrides and identity-matching relocated data.
- Run root copies tokenizer and selected base weights once; periodic steps store adapters/optimizer/trainer/scalar state only.
- `export_merged(checkpoint, output) -> ExportResult` publishes a portable immutable export from the run's recovered latest checkpoint without mutating live state.
- Extend `resolve_model_artifact` for latest-only LoRA runs and merged exports; direct checkpoint-step directories and historical selectors remain invalid.

- [ ] **Step 1: Write step-zero, resume, source-removal, export, and resolution tests**

```python
def test_lora_run_is_self_contained_after_source_run_removal(tiny_base_run, tiny_swag_bundle, tmp_path):
    trained = finetune(tiny_swag_training_config(tiny_base_run, tiny_swag_bundle, tmp_path / "lora-run", maximum_steps=1))
    move_to_unavailable_location(tiny_base_run)
    resumed = resume_finetune(trained.run, data=tiny_swag_bundle.path, overrides=ResumeOverrides(maximum_steps=2))
    session = InferenceSession.from_checkpoint(resumed.run, full_verify=True)
    exported = export_merged(resumed.run, tmp_path / "export")
    export_session = InferenceSession.from_checkpoint(exported.path, full_verify=True)
    assert resumed.step == 2
    assert session.model_identity.artifact_kind == "lora-run"
    assert session.resolved_model.model_config.rope_scaling_factor == 1.0
    assert export_session.model_identity.artifact_kind == "export"
    assert export_session.resolved_model.model_config.rope_scaling_factor == 1.0


def test_lora_checkpoint_omits_frozen_base(tiny_lora_run):
    step = resolve_latest_step(tiny_lora_run, writable=False, verification=VerificationLevel.FULL)
    assert {path.name for path in step.step_directory.iterdir()} == {
        "adapters.safetensors", "optimizer.safetensors", "trainer.safetensors", "state.json", "checkpoint.json"
    }
    assert (tiny_lora_run / "base/model.safetensors").is_file()
    assert_checkpoint_array_dtypes(
        step,
        adapters=mx.float32,
        optimizer_moments=mx.float32,
        trainer_accumulators=mx.float32,
    )
    assert_base_snapshot_array_dtypes(tiny_lora_run / "base", model=mx.bfloat16)
```

Also compare uninterrupted/interrupted adapter/optimizer/cursor/step/PRNG state,
reject every data/base/tokenizer/precision mismatch before allocation, ensure
limit-satisfied resume returns before iterator/kernel construction, prove
latest-only export uses recovered latest state and rejects direct step paths,
and reject any LoRA run, checkpoint, or export whose model configuration changes
the copied base's `rope_scaling_factor=1.0`.

- [ ] **Step 2: Run LoRA workflow tests and verify RED**

```bash
uv run pytest v2/tests/integration/test_swag_workflow.py -v
```

Expected: FAIL because LoRA run orchestration/export/resolution is incomplete.

- [ ] **Step 3: Implement copied-base run creation, strict resume, and atomic export**

Fully verify the recovered latest base checkpoint, tokenizer, and SWAG bundle
before allocation, including both pretraining parameter payloads and the exact
FP32-master-to-BF16-working cast relationship. Require the copied base model
configuration to record `rope_scaling_factor=1.0`; copy that complete
configuration unchanged into the LoRA run, every adapter checkpoint, and the
merged export. Construct `SwagKernelConfig` only from the validated composed
`SwagTrainingConfig`; do not flatten shared loader/optimizer/checkpoint fields
into a second configuration source. Under the writer lock, copy only the
recovered latest checkpoint's exact BF16 `model.safetensors` bytes into the
frozen base snapshot, not `master.safetensors` or base optimizer/trainer state;
create `base/manifest.json` with complete source model/precision configuration,
tokenizer identity, copied-working-weight ref, and diagnostic source run/step
identity. Copy tokenizer and atomically publish adapter step zero, immutable run
manifest, and latest. The LoRA run manifest records copied-base and authoritative
encoded-SWAG identities plus only a diagnostic data locator. Resume strictly
loads FP32 adapter/moment/accumulator state, calls the FP32 SWAG optimizer path,
restores the explicit PRNG key, and rejects any BF16 adapter or optimizer leaf
or changed RoPE factor before constructing kernels. Export resolves recovered
latest state, computes `merged_model_weights`, saves BF16 plain names without
base masters, copies tokenizer, and builds `ExportManifest` with the unchanged
complete model/precision configuration, tokenizer/payload identities, and
diagnostic source run/step identity before immutable publication. Long-context
fine-tuning with a factor above `1.0` will use a future, separately specified
workflow and distinct artifact identity; it is not an export-time or
inference-time override.

- [ ] **Step 4: Verify end-to-end LoRA portability/resume/export**

```bash
uv run pytest v2/tests/unit/training/test_lora.py v2/tests/unit/training/test_swag.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_inference_workflow.py -v
```

Expected: all tests pass after deleting the source pretraining run; only matching encoded data is needed to resume.

- [ ] **Step 5: Commit self-contained fine-tuning runs**

```bash
git add v2/src/sml/training v2/src/sml/artifacts/checkpoint.py v2/src/sml/inference.py v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_inference_workflow.py
git commit -m "feat(v2): persist portable lora runs and exports"
```

### Task 5.5: Verify the Complete SWAG Flow and Controlled Quality

**Files:**
- Create: `v2/tests/integration/test_part2_swag_flow.py`
- Create: `v2/benchmarks/swag_quality.py`
- Create: `v2/tests/unit/test_swag_quality.py`
- Create: `v2/benchmarks/manifests/swag-quality-v1.json`
- Create: `v2/benchmarks/results/swag-quality-v1.jsonl`
- Create: `v2/benchmarks/results/swag-quality-v1.json`

**Interfaces:**
- Produces one offline tokenizer/base/SWAG/LoRA/export/inference/evaluation proof and controlled SWAG quality evidence.
- `SwagQualityWorkload` pins a frozen BF16 base identity originating from a fully verified FP32-master pretraining checkpoint, fixed source-train and disjoint validation example identities, initial FP32 adapter identity, ordered batches, optimizer configuration, mean-score policy, seeds, and exactly 256 optimizer steps.
- `SwagQualityReport(candidate_validation_loss: float, oracle_validation_loss: float, candidate_accuracy: float, oracle_accuracy: float, candidate_examples: int, oracle_examples: int, candidate_finite: bool, oracle_finite: bool)` is reconstructed from the raw candidate/oracle evidence.
- `decide_swag_quality(report: SwagQualityReport) -> Literal["pass", "fail"]` requires finite candidate/oracle state, identical real-example counts, candidate validation loss within 1 percent relative of the eager FP32-adapter oracle, and candidate validation accuracy within one percentage point of the oracle.

- [ ] **Step 1: Write the complete offline SWAG flow test**

```python
def test_encoded_swag_to_exported_evaluation(tiny_base_run, fake_swag_provider, fake_lm_eval, tmp_path):
    data = prepare_swag_bundle(tiny_swag_preparation_config(fake_swag_provider), resolve_model_artifact(tiny_base_run, full_verify=True), tmp_path / "swag")
    tuned = finetune(tiny_swag_training_config(tiny_base_run, data, tmp_path / "run", maximum_steps=1))
    resumed = resume_finetune(tuned.run, data=data.path, overrides=ResumeOverrides(maximum_steps=2))
    exported = export_merged(resumed.run, tmp_path / "export")
    inference_result = InferenceSession.from_checkpoint(exported.path, full_verify=True).generate("prompt", GenerationRequest(max_new_tokens=2))
    evaluation_result = evaluate(tiny_evaluation_config(exported.path, tasks=("hellaswag",)))
    assert inference_result.model.verification is VerificationLevel.FULL
    assert evaluation_result.model.artifact_kind == "export"
    assert read_export_manifest(exported.path).model_config.rope_scaling_factor == 1.0
```

- [ ] **Step 2: Run all correctness tests and commit the integration proof**

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
git add v2/tests/integration/test_part2_swag_flow.py
git commit -m "test(v2): prove encoded swag to portable export"
```

Expected: correctness passes and `git status --short` is empty before recording controlled quality evidence.

- [ ] **Step 3: Write deterministic SWAG quality-decision tests**

```python
def test_swag_quality_gate_enforces_loss_accuracy_and_example_count():
    passing = SwagQualityReport(
        candidate_validation_loss=1.005,
        oracle_validation_loss=1.0,
        candidate_accuracy=0.795,
        oracle_accuracy=0.80,
        candidate_examples=512,
        oracle_examples=512,
        candidate_finite=True,
        oracle_finite=True,
    )
    assert decide_swag_quality(passing) == "pass"
    assert decide_swag_quality(
        replace(passing, candidate_validation_loss=1.011),
    ) == "fail"
    assert decide_swag_quality(
        replace(passing, candidate_accuracy=0.789),
    ) == "fail"
    assert decide_swag_quality(
        replace(passing, candidate_examples=511),
    ) == "fail"
```

- [ ] **Step 4: Run the SWAG quality test and verify RED**

```bash
uv run pytest v2/tests/unit/test_swag_quality.py -v
```

Expected: FAIL because the controlled SWAG quality harness does not exist.

- [ ] **Step 5: Implement the fixed 256-step candidate/oracle harness**

Build fixed source-train and disjoint validation examples from checked-in
encoded arrays. Fully verify the source pretraining checkpoint and record both
its FP32-master identity and selected BF16-working identity before copying only
the frozen BF16 base into each quality run. Initialize identical FP32 adapters,
then run the compiled candidate and eager FP32-adapter oracle for exactly 256
optimizer steps using the same ordered real examples, synthetic-tail masks,
mean continuation-token score including EOS, AdamW formula, schedule, and keys.
Record raw train/validation loss, validation accuracy, finite-state flags,
real-example counts, base byte identity, adapter state identity, and workload
identity. Reject missing/extra records, changed base bytes, mismatched
real-example counts, or any nonfinite value before applying the 1-percent loss
and one-percentage-point accuracy decisions. The harness identity hashes the
ordered bytes of `swag_quality.py` and `test_swag_quality.py`. As with
pretraining quality, the manifest records an authoritative source commit and
execution dependency boundary; standalone validation reconstructs those
recorded bytes rather than deriving an expected workload from a later checkout.

- [ ] **Step 6: Verify and commit the SWAG quality harness before evidence**

```bash
uv run pytest v2/tests/unit/test_swag_quality.py -v
git add v2/benchmarks/swag_quality.py v2/tests/unit/test_swag_quality.py
git commit -m "test(v2): version swag quality harness"
```

Expected: decision tests pass and neither the workload manifest nor results exist in the harness commit.

- [ ] **Step 7: Record and validate controlled SWAG quality evidence**

```bash
uv run python -m v2.benchmarks.swag_quality record --steps 256 --manifest v2/benchmarks/manifests/swag-quality-v1.json --raw-output v2/benchmarks/results/swag-quality-v1.jsonl --output v2/benchmarks/results/swag-quality-v1.json
uv run python -m v2.benchmarks.swag_quality validate --manifest v2/benchmarks/manifests/swag-quality-v1.json --raw-input v2/benchmarks/results/swag-quality-v1.jsonl --report v2/benchmarks/results/swag-quality-v1.json
```

Expected: PASS with finite compiled/oracle state, identical real-example counts, relative validation-loss difference at most 1 percent, and validation-accuracy difference at most one percentage point.

- [ ] **Step 8: Commit controlled SWAG quality evidence**

```bash
git add v2/benchmarks/manifests/swag-quality-v1.json v2/benchmarks/results/swag-quality-v1.jsonl v2/benchmarks/results/swag-quality-v1.json
git commit -m "test(v2): record swag quality evidence"
```

- [ ] **Step 9: Revalidate quality and the complete portability workflow**

```bash
uv run python -m v2.benchmarks.swag_quality validate --manifest v2/benchmarks/manifests/swag-quality-v1.json --raw-input v2/benchmarks/results/swag-quality-v1.jsonl --report v2/benchmarks/results/swag-quality-v1.json
uv run pytest v2/tests/integration/test_part2_swag_flow.py v2/tests/integration/test_swag_workflow.py -v
```

- [ ] **Step 10: Optionally collect SWAG performance diagnostics**

The historical five-pair SWAG comparison may be run when useful. Its absence,
noise, thermal rejection, or ratios do not block Phase 6 and no `phase-5.json`
evidence commit is required.

## Phase 6: Unified CLI and Final Cutover

### Task 6.1: Implement Frozen TOML Configuration and Unified Lazy CLI

**Files:**
- Modify: `v2/src/sml/cli.py`
- Modify: `v2/src/sml/__main__.py`
- Create: `v2/tests/unit/test_cli.py`
- Create: `v2/tests/integration/test_cli_config.py`

**Interfaces:**
- Commands are exactly `tokenize`, `prepare pretraining`, `prepare swag`, `train`, `infer`, `evaluate`, `finetune`, `export`, and `verify`.
- Every command accepts optional `--config PATH`; precedence is frozen dataclass defaults, TOML values, then explicit CLI values.
- The only accepted TOML root tables are exactly `[tokenize]`, `[prepare.pretraining]`, `[prepare.swag]`, `[train]`, `[infer]`, `[evaluate]`, `[finetune]`, `[export]`, and `[verify]`. A file passed to one command must contain exactly that command table and its documented nested policy tables; sibling command tables and unknown keys fail.
- CLI functions parse to frozen command-specific DTOs, construct the owner module's domain dataclasses, import the owner workflow functions only after validation, print typed results, and map focused domain exceptions to stable nonzero exit codes. A DTO may contain CLI-only fields such as `resume`, `full`, or `output`, but neither CLI nor domain APIs accept historical step selection, and domain dataclasses contain only the fields defined by their owner tasks.
- `argparse.Namespace` never enters a domain API.
- Resume accepts only maximum steps/epochs, logging interval, checkpoint interval, and an optional identity-matching `--data` location; checkpoint storage is always pruned to latest-only.
- The overlay mapper uses the exact domain paths below; it never relies on same-named flat attributes:

| TOML/CLI field | Domain destination |
| --- | --- |
| `train.microbatch_size`, `finetune.microbatch_size` / `--microbatch-size` | `loader.microbatch_size` |
| `*.gradient_accumulation_steps` / `--gradient-accumulation-steps` | `loader.gradient_accumulation_steps` |
| `*.prefetch_depth` / `--prefetch-depth` | `loader.prefetch_depth` |
| `*.learning_rate` / `--learning-rate` | `optimizer.learning_rate` |
| `*.checkpoint_interval` / `--checkpoint-interval` | `checkpoint.interval` on fresh runs; `ResumeOverrides.checkpoint_interval` on resume |
| `*.maximum_steps`, `*.maximum_epochs`, `*.log_interval` | the same field on fresh training configs or `ResumeOverrides` on resume |
| `prepare.swag.revision` / `--revision` | `source.revision` |

- All remaining structured settings use recursively validated nested tables that match their domain field names, such as `[train.model]`, `[train.optimizer.weight_decay]`, `[train.precision]`, `[prepare.swag.source]`, `[finetune.lora]`, and `[finetune.lora.initializer]`. Fields listed in the flat mapping table are accepted only in their flat form and are rejected if repeated in a nested policy table, so one invocation can never supply two TOML sources for the same domain path. `rope_scaling_factor` in `[train.model]` must be `1.0`; the CLI provides no inference/export override and the current `[finetune]` command preserves the base value.
- Command DTO defaults not owned by a domain dataclass are exact: `infer.max_new_tokens=128`; `infer` and `evaluate` use `full_verify=False`; `prepare swag` and `export` always perform full correctness-sensitive verification of recovered latest state; and evaluation requires an explicit output path rather than writing into a model artifact or implicit project directory.
- Dispatch is exact: `tokenize` calls `train_tokenizer_bundle(config, output)`; `prepare pretraining` calls `prepare_pretraining_bundle(config, output)`; `prepare swag` calls `resolve_model_artifact(checkpoint, full_verify=True)` and then `prepare_swag_bundle(config, base, output)`; fresh/resumed `train` call `train(config)` / `resume(run, data=data, overrides=overrides)`; `infer` calls `infer(InferenceConfig(...))`; `evaluate` calls `evaluate(EvaluationConfig(...))`; fresh/resumed `finetune` call `finetune(config)` / `resume_finetune(run, data=data, overrides=overrides)`; `export` calls `export_merged(checkpoint, output)` and that workflow fully verifies its source; and `verify` calls `verify_artifact(path, full=full)`. `prepare swag` is allowed to import and call both resolution and preparation functions from their owner modules; “lazy” means unrelated workflows and optional providers stay unimported.

- [ ] **Step 1: Write parser/config/error tests for every command**

```python
def test_cli_precedence_is_defaults_then_toml_then_explicit(tmp_path):
    config = tmp_path / "train.toml"
    config.write_text(
        '[train]\nmicrobatch_size = 2\nmaximum_steps = 8\n'
        '[train.model]\nrope_scaling_factor = 1.0\n',
        encoding="utf-8",
    )
    parsed = parse_command(["train", "--config", str(config), "--data", "data", "--maximum-steps", "3"])
    domain = parsed.to_domain()
    assert domain.loader.microbatch_size == 2
    assert domain.maximum_steps == 3


@pytest.mark.parametrize(
    ("argv", "workflow"),
    [
        (["tokenize", "--output", "tok"], "train_tokenizer_bundle"),
        (["prepare", "pretraining", "--tokenizer", "tok", "--output", "data"], "prepare_pretraining_bundle"),
        (
            [
                "prepare",
                "swag",
                "--checkpoint",
                "base",
                "--revision",
                "0123456789abcdef",
                "--output",
                "swag",
            ],
            "prepare_swag_bundle",
        ),
        (["train", "--data", "data", "--output", "run"], "train"),
        (["infer", "--checkpoint", "run", "prompt"], "infer"),
        (["evaluate", "--checkpoint", "run", "--task", "hellaswag", "--output", "eval.json"], "evaluate"),
        (["finetune", "--checkpoint", "run", "--data", "swag", "--output", "ft"], "finetune"),
        (["export", "--checkpoint", "ft", "--output", "export"], "export_merged"),
        (["verify", "--full", "run"], "verify_artifact"),
    ],
)
def test_each_subcommand_lazy_dispatches_typed_config(argv, workflow, cli_spies):
    assert main(argv) == 0
    assert cli_spies[workflow].call_count == 1
    assert not isinstance(cli_spies[workflow].call_args.args[0], argparse.Namespace)
```

`cli_spies` replaces every owner workflow plus `resolve_model_artifact` with typed-result fakes, so this unit test covers dispatch without touching the named placeholder paths; subprocess integration in Task 6.2 uses real local artifacts.

Also test repeated supported evaluation tasks, wrong command/root tables
(including `[prepare]` instead of `[prepare.pretraining]`), unknown nested TOML
keys, invalid task, unexpected exceptions retaining traceback, concise domain
errors, rejection of `--step` and direct `step-*` paths, `--full`, fresh-run
collision, resume semantic override rejection, absent resume data locator,
nested DTO-to-domain mappings, and rejection of any base-pretraining
`rope_scaling_factor` other than `1.0`.

- [ ] **Step 2: Run CLI tests and verify RED**

```bash
uv run pytest v2/tests/unit/test_cli.py v2/tests/integration/test_cli_config.py -v
```

Expected: FAIL because the temporary CLI only exposes help.

- [ ] **Step 3: Implement strict TOML overlays and lazy dispatch**

Use `tomllib`, select the exact command table path, reject unknown tables/fields recursively, and distinguish omitted CLI options from explicit values with `argparse.SUPPRESS`. Apply the mapping table above to immutable nested dataclass replacements, then convert paths and enum values while constructing the final frozen domain dataclasses. Import `datasets` only when `prepare_swag_bundle` actually needs the provider, and import `lm_eval` only inside evaluation construction. Render `SMLConfigurationError`, `SMLArtifactError`, `SMLDataError`, and `SMLRuntimeError` without traceback and let unexpected exceptions propagate.

- [ ] **Step 4: Verify CLI/config behavior and optional-import isolation**

```bash
uv run pytest v2/tests/unit/test_cli.py v2/tests/integration/test_cli_config.py v2/tests/unit/test_package.py -v
uv run python -m sml --help
git diff --exit-code -- uv.lock
```

Expected: all tests pass, help lists the exact commands, package help imports no heavy optional provider, and `uv.lock` is unchanged.

- [ ] **Step 5: Commit unified CLI**

```bash
git add v2/src/sml/cli.py v2/src/sml/__main__.py v2/tests/unit/test_cli.py v2/tests/integration/test_cli_config.py
git commit -m "feat(v2): unify typed workflow cli"
```

### Task 6.2: Exercise Every Unified CLI Workflow Locally

**Files:**
- Create: `v2/tests/integration/test_cli_workflows.py`
- Create: `v2/tests/fixtures/provider_stubs/datasets/__init__.py`
- Create: `v2/tests/fixtures/provider_stubs/lm_eval/__init__.py`

**Interfaces:**
- One integration fixture provides tiny local raw corpus, deterministic SWAG rows, and deterministic lm-eval requests/results without network. Subprocesses prepend `v2/tests/fixtures/provider_stubs` to `PYTHONPATH`, so lazy production imports receive test-owned provider modules without a production test switch.
- Smoke tests invoke `python -m sml` in subprocesses rather than calling workflow functions directly.

- [ ] **Step 1: Write end-to-end subprocess workflow tests**

```python
def test_all_cli_workflows(cli_workspace):
    cli_workspace.run("tokenize", "--config", cli_workspace.tokenizer_toml)
    cli_workspace.run("prepare", "pretraining", "--config", cli_workspace.pretraining_toml)
    cli_workspace.run("train", "--config", cli_workspace.train_toml)
    cli_workspace.run("infer", "--checkpoint", cli_workspace.base_run, "--max-new-tokens", "2", "Once upon a time")
    cli_workspace.run(
        "evaluate",
        "--checkpoint",
        cli_workspace.base_run,
        "--task",
        "hellaswag",
        "--output",
        cli_workspace.base_evaluation,
    )
    cli_workspace.run("prepare", "swag", "--config", cli_workspace.swag_data_toml)
    cli_workspace.run("finetune", "--config", cli_workspace.finetune_toml)
    cli_workspace.run("export", "--checkpoint", cli_workspace.lora_run, "--output", cli_workspace.export)
    cli_workspace.run("infer", "--checkpoint", cli_workspace.export, "--max-new-tokens", "2", "Once upon a time")
    cli_workspace.run(
        "evaluate",
        "--checkpoint",
        cli_workspace.export,
        "--task",
        "winogrande",
        "--output",
        cli_workspace.export_evaluation,
    )
    cli_workspace.run("verify", "--full", cli_workspace.export)
```

Add `train --resume`, `finetune --resume`, historical `--step` and direct
`step-*` path rejection, moved run with relocated data, read-only default versus
`--full`, and expected error exit-code tests.

- [ ] **Step 2: Run the smoke tests and verify RED or missing wiring**

```bash
uv run pytest v2/tests/integration/test_cli_workflows.py -v
```

Expected: initial failures identify any incomplete result serialization, config threading, or lazy-provider injection; no test may use live network.

- [ ] **Step 3: Complete only missing CLI/workflow wiring**

Keep all domain behavior in owner modules. CLI additions may translate arguments, construct configs, print typed result summaries, and map known errors; they must not duplicate artifact, tokenizer, loader, training, generation, evaluation, or export logic.

- [ ] **Step 4: Verify all CLI flows and strict verification levels**

```bash
uv run pytest v2/tests/integration/test_cli_workflows.py v2/tests/integration/test_cli_config.py -v
```

Expected: every listed flow passes offline and persisted inference/evaluation results contain the exact resolved identity and verification level.

- [ ] **Step 5: Commit workflow smoke coverage**

```bash
git add v2/src/sml v2/tests/integration/test_cli_workflows.py v2/tests/fixtures/provider_stubs
git commit -m "test(v2): cover unified cli workflows"
```

### Task 6.3: Delete the Flat Implementation and Migration Bridge

**Files:**
- Delete: `v2/src/config.py`
- Delete: `v2/src/evaluate_sml.py`
- Delete: `v2/src/ft_swag.py`
- Delete: `v2/src/infer_sml.py`
- Delete: `v2/src/lora.py`
- Delete: `v2/src/prepare_pretraining_data.py`
- Delete: `v2/src/pretraining_format.py`
- Delete: `v2/src/sml.py`
- Delete: `v2/src/tokenizer.py`
- Delete: `v2/src/train_sml.py`
- Delete: `v2/src/train_tokenizer.py`
- Delete: `v2/src/utils.py`
- Delete: replaced flat tests in `v2/tests/test_*.py` and `v2/tests/helpers.py`
- Delete: `common/scripts/peek_npz.py`
- Modify: `v2/src/sml/__init__.py`
- Create: `v2/tests/unit/test_source_contract.py`
- Modify: `v2/README.md`

**Interfaces:**
- Final `sml.__init__` exports only supported package configuration/result/session/domain-error types; it contains no importlib bridge.
- Final source tree matches the approved package architecture and contains no flat compatibility entrypoint.
- README documents only directory artifacts and `uv run python -m sml` commands.

- [ ] **Step 1: Write final source/layout and documentation tests before deletion**

```python
def test_final_v2_source_has_only_package_modules():
    src = Path(__file__).resolve().parents[2] / "src"
    assert sorted(path.name for path in src.iterdir() if path.suffix == ".py") == []
    assert (src / "sml/__main__.py").is_file()


def test_final_source_has_no_bridge_or_legacy_imports():
    source = "\n".join(path.read_text(encoding="utf-8") for path in (PROJECT / "src/sml").rglob("*.py"))
    for forbidden in (
        "spec_from_file_location",
        "sml._legacy",
        "LEGACY_BRIDGE_EXPORTS",
        "train_sml",
        "infer_sml",
        "evaluate_sml",
        "ft_swag",
        "from config import",
    ):
        assert forbidden not in source


def test_readme_names_only_unified_commands():
    readme = (PROJECT / "README.md").read_text(encoding="utf-8")
    assert "uv run python -m sml train" in readme
    assert "python v2/src/train_sml.py" not in readme
    assert ".npz" not in readme
```

- [ ] **Step 2: Run final-contract tests and verify RED**

```bash
uv run pytest v2/tests/unit/test_source_contract.py -v
```

Expected: FAIL while flat modules and the bridge remain.

- [ ] **Step 3: Remove replaced source/tests, narrow exports, and rewrite README**

Delete the listed files only after mapping each lasting behavioral assertion to
its new mirrored test. Remove the obsolete repository-level NPZ inspector
because replacement prepared data is a strict NPY directory artifact and NPZ
compatibility is explicitly excluded. Remove bridge loading and export new
names directly from owner modules. Document tokenizer/pretraining/SWAG/run/
export layouts, full versus manifest-trusted verification, resume override
rules, latest-only model selection, rejection of direct step paths, and every
unified command. Do not document any legacy path or artifact.

- [ ] **Step 4: Verify clean layout, all behavior, formatting, and unchanged lock**

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
uv run python -m sml --help
git diff --exit-code -- uv.lock
```

Expected: all tests pass with only mirrored package tests; the legacy flat files are absent and no compatibility scaffold remains.

- [ ] **Step 5: Commit final cutover**

```bash
git add -A v2/src v2/tests v2/README.md common/scripts/peek_npz.py
git commit -m "refactor(v2): cut over to unified sml package"
```

### Task 6.4: Run Final Correctness and CLI Acceptance

**Interfaces:**
- Proves the final package, controlled quality evidence, artifact workflows,
  unified CLI, resume paths, inference/evaluation, LoRA/SWAG, export, and clean
  cutover all run correctly from the final tree.

The completed result projections report recovered-latest selection and pending
pruning read-only. Evaluation binds provenance to retained, regular-file YAML
bytes captured next to provider load and rechecks them before publication.
Merged export performs FULL LoRA-run semantic validation, including progress
bounds and the seed-derived terminal key, before producing portable BF16 plain
weights.

- [x] **Step 1: Run all final static, correctness, and quality checks**

```bash
git status --short
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests
uv run python -m v2.benchmarks.quality validate --manifest v2/benchmarks/manifests/pretraining-quality-v1.json --raw-input v2/benchmarks/results/pretraining-quality-v1.jsonl --report v2/benchmarks/results/pretraining-quality-v1.json
uv run python -m v2.benchmarks.swag_quality validate --manifest v2/benchmarks/manifests/swag-quality-v1.json --raw-input v2/benchmarks/results/swag-quality-v1.jsonl --report v2/benchmarks/results/swag-quality-v1.json
```

Expected: the worktree is clean; all correctness/equivalence/integration tests
pass outside the sandbox; and both controlled quality reports revalidate
against their committed raw evidence, harness identities, and recorded source
commits without rebuilding expected workloads from final-tree bytes.

- [x] **Step 2: Run every unified CLI smoke from the final tree**

```bash
uv run pytest v2/tests/integration/test_cli_workflows.py -v
uv run python -m sml --help
```

Expected: all commands and fresh/resume/latest-only/full-verification paths
pass, and historical selectors/direct step paths fail.

- [x] **Step 3: Run the final end-to-end workflow set**

Run the integration tests covering tokenizer preparation, prepared data,
pretraining fresh/resume/recovery, inference/evaluation, SWAG preparation,
LoRA fresh/resume/export, and the unified CLI. Use tiny deterministic fixtures
where the full production workload would make correctness verification
impractical.

- [x] **Step 4: Verify repository invariants**

```bash
git diff --exit-code -- uv.lock
git status --short
```

Expected: `uv.lock` is unchanged and the final accepted implementation is
committed. Optional benchmark output may remain diagnostic and is not required
for completion.

- [x] **Step 5: Resolve optional final performance diagnostics**

No performance capture was required or claimed for final acceptance. The
historical ten-pair comparison may still be run to characterize the completed
refactor. It creates no required acceptance artifact, and any missing,
inconclusive, noisy, thermally rejected, or below-target result does not change
the correctness acceptance decision.

## Part 2 Completion Gate — Met

The conditions below were met at the final accepted evidence boundary:

- every new package unit, equivalence, artifact-safety, resume, integration, and CLI test passes;
- pretraining checkpoints retain authoritative FP32 masters plus exact BF16 working casts, while inference, LoRA base snapshots, and merged exports own only the BF16 model state they require;
- the committed pretraining and SWAG controlled quality reports pass their finite-state, master-update, validation-loss, validation-accuracy, and real-work identity gates;
- each controlled-quality validator verifies its manifest's recorded source
  boundary; unrelated later source edits and the completed migration-bridge
  deletion do not invalidate earlier accepted evidence;
- inference/evaluation pin resolved model identity and correctly report `manifest-trusted` or `full`;
- encoded SWAG data is immutable/reusable offline, LoRA resume is exact, the copied-base run survives source-run deletion, and export is a portable BF16 plain-weight artifact;
- the current inference, evaluation, SWAG LoRA, resume, and export paths preserve the base run's authoritative `rope_scaling_factor=1.0`; a future factor-above-`1.0` long-context fine-tuning creates a distinct run and is outside this plan;
- the Phase 4, 5, and 6 functional workflow gates pass;
- the final static, correctness, controlled-quality, integration, and CLI checks pass;
- all flat implementations, legacy tests, bridge code, and old README commands are gone;
- `uv.lock` remains byte-identical and no unapproved dependency/top-level change exists.
