# V2 Inference Shape and Target-Logit Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use independent prompt-prefill and cache-capacity buckets for generation and compute evaluation token log probabilities without a vocabulary-sized log-probability tensor.

**Architecture:** Generation request/bucket records carry two explicit length domains and compile caches key both shapes. The buffer pool still leases capacity-sized token/KV state, while prefill views and masks stop at the prompt bucket. A small array helper gathers target logits first and is called by the compiled scoring kernel.

**Tech Stack:** Python 3.12.13, MLX compiled prefill/decode/scoring kernels, pooled KV/token arrays, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md`

## Global Constraints

- Continue in the current checkout; do not create a worktree.
- Use `uv run`; run every pytest command outside the sandbox.
- Do not edit top-level project files or `uv.lock`.
- Preserve seeded serial/batched equivalence, caller-order restoration, EOS, repetition penalty, no-repeat n-gram, and non-reentrant session behavior.
- Keep all token/KV allocations bounded by `cache_capacity_bucket` and all initial prompt token/mask/position work bounded by `prefill_length_bucket`.
- Scoring remains FP32 mean/sum behavior already specified by its callers; only the allocation order changes.

---

## File Structure

- Modify `v2/src/sml/inference.py`: dual bucket fields, grouping/compile keys, capacity lease with prompt-sized prefill, target-logit helper.
- Modify `v2/tests/unit/test_inference.py`: bucket selection, compile-key, storage/prefill spy, pooled state tests.
- Modify `v2/tests/unit/test_evaluation.py`: target-logit formula/allocation spy and both padding layouts.
- Modify `v2/tests/integration/test_inference_workflow.py`: short-prompt/long-generation and serial/batched behavior.
- Modify `v2/tests/integration/test_evaluation_workflow.py`: scoring/evaluation regression coverage.

## Frozen Interfaces

Replace the single generation length with these exact fields:

```text
_PreparedRequest:
  caller_index, prompt_ids, prefill_length_bucket, cache_capacity_bucket,
  kernel_key, seed, key, max_new_tokens, include_prompt

GenerationBucket:
  prefill_length_bucket, cache_capacity_bucket, batch_size_bucket, kernel_key,
  keys, request_mask, prompt_ids, prompt_lengths, max_new_tokens, seeds,
  caller_indices, include_prompt, host_max_new
```

Generation compile-cache keys are exact tuples:

```text
(prefill_length_bucket, cache_capacity_bucket, batch_size_bucket, kernel_key)
```

Add this internal scoring helper:

```text
_target_log_probabilities(
    predictor_logits_fp32: mx.array,
    target_token_ids: mx.array,
) -> mx.array
```

It returns shape `predictor_logits_fp32.shape[:-1]` and dtype FP32.

## Task 1: Select and Carry Two Independent Generation Buckets

**Files:**
- Modify: `v2/src/sml/inference.py`
- Modify: `v2/tests/unit/test_inference.py`

**Interfaces:**
- Consumes: existing power-of-two `length_buckets`, `GenerationKernelKey`, batch-size buckets, prompt encoding.
- Produces: dual-bucket `_PreparedRequest`/`GenerationBucket`, stable grouping, four-part compile key.

- [ ] **Step 1: Write failing dual-bucket and compile-key tests**

```python
def test_short_prompt_and_long_generation_select_independent_buckets(tiny_session) -> None:
    prompt = "alpha"
    request = GenerationRequest(max_new_tokens=16)
    prompt_ids = tiny_session._encode_prompt(prompt)
    bucket = tiny_session._bucketize(((prompt, request),))[0]
    assert bucket.prefill_length_bucket == tiny_session._select_length_bucket(len(prompt_ids))
    assert bucket.cache_capacity_bucket == tiny_session._select_length_bucket(
        len(prompt_ids) + request.max_new_tokens
    )
    assert bucket.prefill_length_bucket < bucket.cache_capacity_bucket


def test_generation_compile_cache_keys_both_length_domains(tiny_session) -> None:
    prompt = "alpha"
    request = GenerationRequest(max_new_tokens=16)
    bucket = tiny_session._bucketize(((prompt, request),))[0]
    tiny_session.generate(prompt, request)
    assert (
        bucket.prefill_length_bucket,
        bucket.cache_capacity_bucket,
        bucket.batch_size_bucket,
        bucket.kernel_key,
    ) in tiny_session._compiled
```

Update `_CompileSpyKey` to carry both lengths. Add one test where prompt length
changes but capacity stays in the same bucket and one where capacity changes but
prompt bucket stays fixed; each change must produce a distinct compile entry.

- [ ] **Step 2: Run generation unit tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_inference.py -q
```

Expected: records expose only `length_bucket` and compile keys have three parts.

- [ ] **Step 3: Calculate both lengths at the request boundary**

In `_bucketize`, select:

```python
prefill_length_bucket = self._select_length_bucket(len(prompt_ids))
cache_capacity_bucket = self._select_length_bucket(
    len(prompt_ids) + request.max_new_tokens
)
```

Store both values in `_PreparedRequest`. Group by
`(prefill_length_bucket, cache_capacity_bucket, kernel_key)` in first-seen order.
Pass both values into `_pad_bucket`, and copy them without deriving one from the
other. Keep seed allocation before grouping.

- [ ] **Step 4: Key compiled kernels by both shapes**

Change `_compiled_kernels` to accept both lengths and cache by the exact
four-part tuple. Keep batch size and generation policy unchanged. The prefill
function receives a prompt-shaped token input plus capacity-shaped cache state;
the decode core continues to receive capacity-shaped token/cache state.

- [ ] **Step 5: Run unit tests and commit record/key changes**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/test_inference.py -q
uv run ruff check v2/src/sml/inference.py v2/tests/unit/test_inference.py
uv run ruff format --check v2/src/sml/inference.py v2/tests/unit/test_inference.py
git add v2/src/sml/inference.py v2/tests/unit/test_inference.py
git commit -m "refactor(v2): separate generation bucket identities"
```

Expected: selection/grouping/cache-key tests pass; generation behavior remains
green.

## Task 2: Execute Prompt-Sized Prefill against Capacity-Sized State

**Files:**
- Modify: `v2/src/sml/inference.py`
- Modify: `v2/tests/unit/test_inference.py`
- Modify: `v2/tests/integration/test_inference_workflow.py`

**Interfaces:**
- Consumes: Task 1 dual buckets and existing `BufferPool.lease`.
- Produces: capacity-sized lease, prompt-sized prefill arrays/masks/positions, capacity-backed returned cache.

- [ ] **Step 1: Write a failing prefill/storage shape spy**

```python
def test_prefill_uses_prompt_bucket_while_lease_uses_capacity_bucket(
    tiny_session, monkeypatch
) -> None:
    captured: dict[str, int] = {}
    real_lease = tiny_session.buffer_pool.lease
    real_compiled = tiny_session._compiled_kernels

    def lease_spy(*, batch_size, capacity, config):
        captured["leased_capacity"] = capacity
        return real_lease(batch_size=batch_size, capacity=capacity, config=config)

    def compiled_spy(prefill_length, cache_capacity, batch_size, kernel_key):
        prefill, decode = real_compiled(
            prefill_length, cache_capacity, batch_size, kernel_key
        )
        def prefill_shape_spy(parameters, input_ids, attention_mask, positions, cache_state):
            captured["prefill_tokens"] = int(input_ids.shape[1])
            captured["prefill_mask"] = int(attention_mask.shape[1])
            captured["cache_capacity"] = int(cache_state[0][0].shape[2])
            return prefill(parameters, input_ids, attention_mask, positions, cache_state)
        return prefill_shape_spy, decode

    monkeypatch.setattr(tiny_session.buffer_pool, "lease", lease_spy)
    monkeypatch.setattr(tiny_session, "_compiled_kernels", compiled_spy)
    tiny_session.generate("alpha", GenerationRequest(max_new_tokens=16))
    assert captured["prefill_tokens"] == captured["prefill_mask"]
    assert captured["prefill_tokens"] < captured["leased_capacity"]
    assert captured["cache_capacity"] == captured["leased_capacity"]
```

- [ ] **Step 2: Run the shape spy and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_inference.py::test_prefill_uses_prompt_bucket_while_lease_uses_capacity_bucket -q
```

Expected: prefill width equals the old capacity bucket.

- [ ] **Step 3: Lease capacity but slice prompt inputs**

In `_generate_batch`, lease with `bucket.cache_capacity_bucket`. In
`_decode_chunk`, fill only the real prompt prefix of capacity-sized token
storage, then construct:

```python
prefill_tokens = lease.token_storage[:, : bucket.prefill_length_bucket]
token_range = mx.arange(bucket.prefill_length_bucket, dtype=mx.int32)[None, :]
real_mask = (
    token_range < bucket.prompt_lengths[:, None]
) & bucket.request_mask[:, None]
synthetic_mask = (~bucket.request_mask)[:, None] & (token_range == 0)
attention_mask = real_mask | synthetic_mask
positions = mx.where(
    attention_mask,
    mx.broadcast_to(token_range, attention_mask.shape),
    mx.zeros(attention_mask.shape, dtype=mx.int32),
)
```

Call prefill with `prefill_tokens`, prompt-shaped masks/positions, and the full
capacity cache state. Gather next logits using prompt lengths against prefill
logits. Decode continues with full `lease.token_storage` and full cache state.

- [ ] **Step 4: Run generation equivalence and pooling tests**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_inference.py -q
uv run pytest v2/tests/equivalence/test_inference_equivalence.py -q
uv run pytest v2/tests/integration/test_inference_workflow.py -q
```

Expected: shape spy passes; greedy/seeded serial-batched equivalence, storage
reuse, EOS, processor policies, and caller order remain green.

- [ ] **Step 5: Commit prompt-sized prefill**

```bash
git add v2/src/sml/inference.py v2/tests/unit/test_inference.py v2/tests/integration/test_inference_workflow.py
git commit -m "fix(v2): bound prefill by prompt length"
```

## Task 3: Gather Target Logits before Log-Sum-Exp Subtraction

**Files:**
- Modify: `v2/src/sml/inference.py`
- Modify: `v2/tests/unit/test_evaluation.py`
- Modify: `v2/tests/integration/test_evaluation_workflow.py`

**Interfaces:**
- Consumes: FP32 predictor logits `(batch, tokens, vocab)` and int32 targets `(batch, tokens)`.
- Produces: `_target_log_probabilities` with FP32 `(batch, tokens)` output and no predictor-shaped log-probability allocation.

- [ ] **Step 1: Write failing formula, shape, and source tests**

```python
def test_target_log_probabilities_gather_before_subtraction(monkeypatch) -> None:
    predictor = mx.array(
        [[[1.0, 2.0, 3.0], [3.0, 1.0, -1.0]]],
        dtype=mx.float32,
    )
    targets = mx.array([[2, 0]], dtype=mx.int32)
    calls = []
    original = inference.mx.take_along_axis

    def spy(array, indices, *, axis):
        result = original(array, indices, axis=axis)
        calls.append((tuple(array.shape), tuple(result.shape)))
        return result

    monkeypatch.setattr(inference.mx, "take_along_axis", spy)
    actual = inference._target_log_probabilities(predictor, targets)
    expected = mx.array([[3.0, 3.0]], dtype=mx.float32) - mx.logsumexp(
        predictor, axis=-1
    )
    mx.eval(actual, expected)
    assert tuple(actual.shape) == (1, 2)
    assert calls == [((1, 2, 3), (1, 2, 1))]
    assert bool(mx.allclose(actual, expected, atol=0.0, rtol=0.0).item())
    source = inspect.getsource(inference._target_log_probabilities)
    assert "take_along_axis" in source
    assert "predictor_logits_fp32 -" not in source
```

Add a scoring-kernel source assertion that `log_probs` is absent and a numeric
batch test for both left and right padding.

- [ ] **Step 2: Run evaluation tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_evaluation.py -q
```

Expected: helper is absent and the compiled kernel still creates `log_probs` at
full vocabulary shape.

- [ ] **Step 3: Implement and call the target-logit helper**

```python
def _target_log_probabilities(
    predictor_logits_fp32: mx.array,
    target_token_ids: mx.array,
) -> mx.array:
    target_logits = mx.take_along_axis(
        predictor_logits_fp32,
        mx.expand_dims(target_token_ids, axis=-1),
        axis=-1,
    )
    target_logits = mx.squeeze(target_logits, axis=-1)
    return (
        target_logits - mx.logsumexp(predictor_logits_fp32, axis=-1)
    ).astype(mx.float32)
```

Replace `log_z`, `log_probs`, and gather-from-log-probs in the compiled scoring
kernel with this helper. Apply continuation/request masks only after it returns.
Keep greedy `argmax` on predictor logits.

- [ ] **Step 4: Run the inference/evaluation component gate**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/test_inference.py v2/tests/unit/test_evaluation.py -q
uv run pytest v2/tests/equivalence/test_inference_equivalence.py -q
uv run pytest v2/tests/integration/test_inference_workflow.py v2/tests/integration/test_evaluation_workflow.py -q
uv run ruff check v2/src/sml/inference.py v2/tests/unit/test_inference.py v2/tests/unit/test_evaluation.py v2/tests/integration/test_inference_workflow.py v2/tests/integration/test_evaluation_workflow.py
uv run ruff format --check v2/src/sml/inference.py v2/tests/unit/test_inference.py v2/tests/unit/test_evaluation.py v2/tests/integration/test_inference_workflow.py v2/tests/integration/test_evaluation_workflow.py
```

Expected: dual-shape generation and target-logit scoring pass all unit,
equivalence, and integration tests.

- [ ] **Step 5: Commit scoring and complete the component**

```bash
git add v2/src/sml/inference.py v2/tests/unit/test_inference.py v2/tests/unit/test_evaluation.py v2/tests/integration/test_inference_workflow.py v2/tests/integration/test_evaluation_workflow.py
git commit -m "fix(v2): gather target logits before normalization"
```

## Plan Completion Gate

- [ ] **Step 1: Inspect removed single-domain/log-probability paths**

```bash
rg -n "bucket\.length_bucket|item\.length_bucket|log_probs = predictor" v2/src/sml/inference.py
git status --short
```

Expected: no generation reference aliases one length across both domains, no
full log-probability assignment remains, and the tracked checkout is clean.

- [ ] **Step 2: Review against the inference remediation contract**

Confirm compile keys include both shapes, prefill arrays are prompt-sized,
leases/caches are capacity-sized, logical positions remain per request,
target-logit subtraction is token-shaped, and all prior deterministic policies
remain intact.

Expected: no Critical, Important, or Minor findings before starting the
artifact plan.
