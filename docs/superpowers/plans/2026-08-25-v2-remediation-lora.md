# V2 LoRA Precision and Explicit-Dropout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep LoRA scale and adapter arithmetic FP32 and apply configured explicit-key LoRA dropout in the pure-array SWAG training path with exact eager/compiled/resume replay.

**Architecture:** Static adapter behavior moves out of the MLX parameter tree into an immutable `LoRAForwardPolicy` ordered by canonical module execution. Model linear dispatch receives that policy and threads one key cursor through every adapted projection and existing base-dropout site; live wrappers, pure-array training, and merge/export share the same scale/formula source.

**Tech Stack:** Python 3.12.13, MLX array trees/PRNG/compile/autodiff, pytest, Ruff, safetensors checkpoint integration.

**Spec:** `docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md`

## Global Constraints

- Continue in the current checkout; do not create a worktree.
- Use `uv run`; run every pytest command outside the sandbox.
- Do not edit top-level project files or `uv.lock`.
- The model package must not import `sml.training`; static policy types therefore live in `sml.model.layers` and are constructed by `sml.training.lora`.
- Adapter parameter trees contain only FP32 `lora_a` and `lora_b`; frozen trees contain only BF16 base-model arrays.
- Disabled/inference dropout consumes no key. Every enabled adapter dropout consumes one split at its exact execution site.
- Legacy equivalence fixtures use dropout zero and retain their existing tolerances.
- Do not regenerate SWAG quality evidence in this plan; final acceptance records it only after every repair source commit is clean.

---

## File Structure

- Modify `v2/src/sml/model/layers.py`: static policy records, policy lookup, shared linear/adapter formula, key-threaded attention/MLP/block array forwards.
- Modify `v2/src/sml/model/language_model.py`: initialized optional policy and policy propagation through pure-array forward.
- Modify `v2/src/sml/model/__init__.py`: expose policy records to the training owner.
- Modify `v2/src/sml/training/lora.py`: derive policy, remove array scale state, wrap targets with canonical paths, use policy in live and merge paths.
- Modify `v2/src/sml/training/swag.py`: assert and preserve the static policy while compiled kernels merge only adapter/base arrays.
- Modify `v2/tests/unit/model/test_layers.py`: shared linear formula and key consumption tests.
- Modify `v2/tests/unit/model/test_language_model.py`: base-model no-policy behavior and adapted pure-array policy propagation.
- Modify `v2/tests/unit/training/test_lora.py`: static scale, parameter split, live/pure-array/merge formula, key replay.
- Modify `v2/tests/unit/training/test_swag.py`: nonzero adapter-dropout eager/compiled two-step behavior.
- Modify `v2/tests/equivalence/test_lora_equivalence.py`: preserve dropout-zero captured equivalence and live/merged tolerance.
- Modify `v2/tests/equivalence/test_swag_equivalence.py`: retain the mean-score formula under adapted pure-array forwarding.
- Modify `v2/tests/integration/test_swag_workflow.py`: nonzero-dropout exact interrupted resume and export.

## Frozen Interfaces

`sml.model.layers` adds:

```text
LoRAAdapterSpec(module_path: str, scale: float, dropout: float)
LoRAForwardPolicy(adapters: immutable tuple of LoRAAdapterSpec)
LoRAForwardPolicy.for_module(module_path: str) -> LoRAAdapterSpec | None
```

`_linear` becomes:

```text
_linear(
    x: mx.array,
    parameters: dict[str, object],
    *,
    module_path: str,
    lora_policy: LoRAForwardPolicy | None,
    training: bool,
    key: mx.array | None,
) -> tuple[mx.array, mx.array | None]
```

`SMLLanguageModel.lora_forward_policy` is `LoRAForwardPolicy | None` and is
static Python configuration. All array forward methods accept/return the same
key cursor they already use; attention additionally returns the cursor after
q/k/v/o projections.

## Task 1: Remove Scale from the Parameter Tree

**Files:**
- Modify: `v2/src/sml/model/layers.py`
- Modify: `v2/src/sml/model/language_model.py`
- Modify: `v2/src/sml/model/__init__.py`
- Modify: `v2/src/sml/training/lora.py`
- Modify: `v2/tests/unit/training/test_lora.py`

**Interfaces:**
- Consumes: strict `LoRAConfig`, canonical `model.named_modules()` order, `_Linear`, FP32 A/B initialization.
- Produces: `LoRAAdapterSpec`, `LoRAForwardPolicy`, static `LoRALinear.spec`, and a scale-free parameter split.

- [ ] **Step 1: Write failing policy and precision tests**

```python
def test_lora_policy_is_canonical_static_and_scale_is_not_a_parameter(tiny_model) -> None:
    config = tiny_lora_config(
        rank=3,
        alpha=1.0,
        scaling_mode="lora",
        dropout=0.25,
        target_modules=("q_proj", "v_proj", "down_proj"),
    )
    adapted = apply_lora(tiny_model, config, key=mx.random.key(4))
    policy = adapted.lora_forward_policy
    assert isinstance(policy, LoRAForwardPolicy)
    assert tuple(spec.module_path for spec in policy.adapters) == (
        "layers.0.self_attn.q_proj",
        "layers.0.self_attn.v_proj",
        "layers.0.mlp.down_proj",
    )
    assert all(spec.dropout == 0.25 for spec in policy.adapters)
    parameters = dict(tree_flatten(adapted.parameters()))
    trainable = dict(tree_flatten(adapted.trainable_parameters()))
    assert not any(name.endswith(".scale") for name in parameters)
    assert set(trainable) == {
        name for name in parameters if name.endswith((".lora_a", ".lora_b"))
    }
    scale_fp32 = mx.array(policy.adapters[0].scale, dtype=mx.float32)
    scale_bf16_round_trip = scale_fp32.astype(mx.bfloat16).astype(mx.float32)
    mx.eval(scale_fp32, scale_bf16_round_trip)
    assert not bool(mx.array_equal(scale_fp32, scale_bf16_round_trip).item())


def test_split_adapter_parameters_contains_only_fp32_adapters_and_bf16_base(tiny_model) -> None:
    adapted = apply_lora(tiny_model, tiny_lora_config(), key=mx.random.key(5))
    adapters, frozen = split_adapter_parameters(adapted.parameters())
    adapter_leaves = dict(tree_flatten(adapters))
    frozen_leaves = dict(tree_flatten(frozen))
    assert adapter_leaves
    assert frozen_leaves
    assert all(array.dtype == mx.float32 for array in adapter_leaves.values())
    assert all(array.dtype == mx.bfloat16 for array in frozen_leaves.values())
    assert not any(name.endswith("scale") for name in (*adapter_leaves, *frozen_leaves))
```

- [ ] **Step 2: Run the focused LoRA tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/training/test_lora.py -q
```

Expected: policy attributes are absent and `.scale` remains an FP32 array leaf.

- [ ] **Step 3: Add strict static policy records**

Implement in `model/layers.py`:

```python
@dataclass(frozen=True, slots=True)
class LoRAAdapterSpec:
    module_path: str
    scale: float
    dropout: float

    def __post_init__(self) -> None:
        if not self.module_path:
            raise ValueError("LoRA module_path must be nonempty")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("LoRA scale must be finite and positive")
        if not math.isfinite(self.dropout) or not 0.0 <= self.dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class LoRAForwardPolicy:
    adapters: tuple

    def __post_init__(self) -> None:
        paths = tuple(spec.module_path for spec in self.adapters)
        if not paths or len(paths) != len(set(paths)):
            raise ValueError("LoRA policy paths must be nonempty and unique")

    def for_module(self, module_path: str) -> LoRAAdapterSpec | None:
        return next(
            (spec for spec in self.adapters if spec.module_path == module_path),
            None,
        )
```

Initialize `SMLLanguageModel.lora_forward_policy = None` before its modules.
Re-export the two records from `sml.model` but not from the root `sml` API.

- [ ] **Step 4: Derive policy during wrapping and remove scale arrays**

Change `LoRALinear` to store `module_path` and `spec`, not `self.scale` or a
frozen scale parameter. Derive one Python scale value as `alpha / rank` for
`lora` or `alpha / math.sqrt(rank)` for `rslora`; `_linear` will materialize it
as `mx.array(spec.scale, dtype=mx.float32)` at use.

`apply_lora` collects targets in existing `named_modules()` order, constructs
the complete policy before mutation, wraps every target with its matching spec,
sets `model.lora_forward_policy`, freezes the model, and unfreezes only A/B.
Delete the `key == "scale"` branch from `split_adapter_parameters`.

- [ ] **Step 5: Run focused tests and commit the static policy**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/training/test_lora.py -q
uv run ruff check v2/src/sml/model/layers.py v2/src/sml/model/language_model.py v2/src/sml/model/__init__.py v2/src/sml/training/lora.py v2/tests/unit/training/test_lora.py
uv run ruff format --check v2/src/sml/model/layers.py v2/src/sml/model/language_model.py v2/src/sml/model/__init__.py v2/src/sml/training/lora.py v2/tests/unit/training/test_lora.py
git add v2/src/sml/model/layers.py v2/src/sml/model/language_model.py v2/src/sml/model/__init__.py v2/src/sml/training/lora.py v2/tests/unit/training/test_lora.py
git commit -m "refactor(v2): make lora forward policy static"
```

Expected: static-policy/parameter tests pass and no model parameter name ends in
`.scale`.

## Task 2: Thread Adapter Dropout through Pure-Array Forward

**Files:**
- Modify: `v2/src/sml/model/layers.py`
- Modify: `v2/src/sml/model/language_model.py`
- Modify: `v2/src/sml/training/lora.py`
- Modify: `v2/tests/unit/model/test_layers.py`
- Modify: `v2/tests/unit/model/test_language_model.py`
- Modify: `v2/tests/unit/training/test_lora.py`

**Interfaces:**
- Consumes: Task 1 policy, parameter trees with base/A/B leaves, existing `keyed_dropout`.
- Produces: one shared `_linear` result/key formula used by live wrappers and `forward_arrays`.

- [ ] **Step 1: Write failing shared-formula and replay tests**

```python
def test_pure_array_lora_dropout_replays_and_advances_key(tiny_model) -> None:
    adapted = apply_lora(
        tiny_model,
        tiny_lora_config(
            dropout=0.5,
            initializer=LoRAInitializerConfig(lora_a=0.05, lora_b=0.05),
        ),
        key=mx.random.key(4),
    )
    input_ids = mx.array([[1, 2, 3, 4]], dtype=mx.int32)
    key = mx.random.key(19)
    first, first_cache, first_key = adapted.forward_arrays(
        adapted.parameters(), input_ids,
        attention_mask=None, positions=None, cache_state=None,
        training=True, key=key,
    )
    replay, replay_cache, replay_key = adapted.forward_arrays(
        adapted.parameters(), input_ids,
        attention_mask=None, positions=None, cache_state=None,
        training=True, key=key,
    )
    mx.eval(first, replay, first_key, replay_key)
    assert first_cache is replay_cache is None
    assert bool(mx.array_equal(first, replay).item())
    assert bool(mx.array_equal(first_key, replay_key).item())
    assert not bool(mx.array_equal(first_key, key).item())


def test_inference_and_zero_dropout_consume_no_adapter_key(tiny_model) -> None:
    adapted = apply_lora(
        tiny_model,
        tiny_lora_config(dropout=0.0),
        key=mx.random.key(4),
    )
    key = mx.random.key(23)
    input_ids = mx.array([[1, 2]], dtype=mx.int32)
    _, _, returned = adapted.forward_arrays(
        adapted.parameters(), input_ids,
        attention_mask=None, positions=None, cache_state=None,
        training=True, key=key,
    )
    _, _, inference_returned = adapted.forward_arrays(
        adapted.parameters(), input_ids,
        attention_mask=None, positions=None, cache_state=None,
        training=False, key=key,
    )
    mx.eval(returned, inference_returned, key)
    assert bool(mx.array_equal(returned, key).item())
    assert bool(mx.array_equal(inference_returned, key).item())
```

Add a direct `_linear` oracle using scale `1/3`, nonzero A/B, and a manually
replayed `keyed_dropout` call. Assert live `LoRALinear.__call__` and `_linear`
return identical BF16 outputs and next keys.

- [ ] **Step 2: Run model/LoRA tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/model/test_layers.py v2/tests/unit/model/test_language_model.py v2/tests/unit/training/test_lora.py -q
```

Expected: the pure-array path either ignores policy/dropout or cannot accept the
new `_linear` contract.

- [ ] **Step 3: Implement the shared linear formula**

For a base parameter mapping containing `weight`, return the BF16 base result
and unchanged key. For an adapted mapping, require a policy/spec match and use:

```python
adapter_input = x.astype(mx.float32)
if training and spec.dropout > 0.0:
    if key is None:
        raise ValueError("training with LoRA dropout requires an explicit key")
    adapter_input, key = keyed_dropout(adapter_input, spec.dropout, key)
scale = mx.array(spec.scale, dtype=mx.float32)
adapter = scale * (
    (adapter_input @ parameters["lora_a"].T) @ parameters["lora_b"].T
)
output = (
    x @ parameters["base"]["weight"].T
    + adapter.astype(mx.bfloat16)
).astype(mx.bfloat16)
return output, key
```

Reject adapted parameters with no matching spec and policy entries applied to a
plain-weight mapping. `LoRALinear.__call__` delegates to this function with its
stored module path and a one-entry policy; it contains no second formula.

- [ ] **Step 4: Thread policy and key in canonical execution order**

Pass `lora_policy`, `training`, and `key` from `SMLLanguageModel.forward_arrays`
through each block. In each layer visit active sites in this exact order:

```text
self_attn.q_proj
self_attn.k_proj
self_attn.v_proj
self_attn.o_proj
mlp.gate_proj
mlp.up_proj
mlp.down_proj
mlp.hidden_dropout
```

`GroupedQueryAttention.forward_arrays` now returns `(output, cache_state, key)`.
`SwiGLUFeedForward.forward_arrays` threads the key through gate/up/down before
its existing hidden dropout. Base `__call__` wrappers pass policy `None`,
training `False`, and key `None` unless their existing public arguments specify
training/key.

- [ ] **Step 5: Run focused tests and commit pure-array dropout**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/model/test_layers.py v2/tests/unit/model/test_language_model.py v2/tests/unit/training/test_lora.py -q
uv run pytest v2/tests/equivalence/test_model_math.py v2/tests/equivalence/test_lora_equivalence.py -q
uv run ruff check v2/src/sml/model v2/src/sml/training/lora.py v2/tests/unit/model v2/tests/unit/training/test_lora.py
uv run ruff format --check v2/src/sml/model v2/src/sml/training/lora.py v2/tests/unit/model v2/tests/unit/training/test_lora.py
git add v2/src/sml/model v2/src/sml/training/lora.py v2/tests/unit/model v2/tests/unit/training/test_lora.py v2/tests/equivalence/test_lora_equivalence.py
git commit -m "fix(v2): thread explicit lora dropout keys"
```

Expected: replay, no-consumption, formula, model-math, and dropout-zero legacy
equivalence tests pass.

## Task 3: Prove Eager and Compiled SWAG State Progression

**Files:**
- Modify: `v2/src/sml/training/swag.py`
- Modify: `v2/tests/unit/training/test_swag.py`
- Modify: `v2/tests/equivalence/test_swag_equivalence.py`

**Interfaces:**
- Consumes: adapted model with static `lora_forward_policy`, A/B-only adapter tree, BF16-only frozen tree.
- Produces: `build_swag_kernels` whose eager/compiled microsteps consume identical adapter dropout subkeys and return the first unused key.

- [ ] **Step 1: Change the two-step fixture to exercise adapter dropout**

Construct the test model by applying LoRA with nonzero B initialization:

```python
model = tiny_language_model(hidden_dropout=0.1)
apply_lora(
    model,
    LoRAConfig(
        rank=2,
        alpha=1.0,
        scaling_mode="lora",
        dropout=0.5,
        target_modules=("q_proj", "v_proj", "down_proj"),
        initializer=LoRAInitializerConfig(lora_a=0.05, lora_b=0.05),
    ),
    key=mx.random.key(5),
)
adapters, frozen_base = split_adapter_parameters(model.parameters())
```

Reuse the same two real batches and initial key for eager and compiled runs.
Assert both next keys advance at both steps, eager/compiled keys are exactly
equal, adapter/optimizer trees match within `1e-5`, and frozen leaves remain
exact BF16 copies.

- [ ] **Step 2: Run the SWAG unit/equivalence tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/training/test_swag.py v2/tests/equivalence/test_swag_equivalence.py -q
```

Expected: the existing fixture/model setup does not yet install or assert the
static adapter policy through the compiled path.

- [ ] **Step 3: Preserve policy in kernel construction**

At `build_swag_kernels` entry, require a nonempty
`model.lora_forward_policy`. Do not merge policy into `adapters`, `frozen_base`,
or any compiled array input. The bound model captures this immutable static
configuration, while `adapter_loss` continues to pass only built-in array trees
and its explicit key to `model.forward_arrays`.

Keep `mx.value_and_grad(adapter_loss, argnums=0)` unchanged so gradients target
only A/B. Extend dtype/source assertions to reject `scale` in either array tree
and to require no global/random module state.

- [ ] **Step 4: Run focused tests and commit SWAG kernel integration**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/training/test_swag.py v2/tests/equivalence/test_swag_equivalence.py -q
uv run ruff check v2/src/sml/training/swag.py v2/tests/unit/training/test_swag.py v2/tests/equivalence/test_swag_equivalence.py
uv run ruff format --check v2/src/sml/training/swag.py v2/tests/unit/training/test_swag.py v2/tests/equivalence/test_swag_equivalence.py
git add v2/src/sml/training/swag.py v2/tests/unit/training/test_swag.py v2/tests/equivalence/test_swag_equivalence.py
git commit -m "fix(v2): apply lora dropout in swag kernels"
```

Expected: eager and compiled two-step adapter, optimizer, and next-key state
agree with nonzero base and adapter dropout.

## Task 4: Prove Exact Resume and FP32 Merge

**Files:**
- Modify: `v2/src/sml/training/lora.py`
- Modify: `v2/tests/unit/training/test_lora.py`
- Modify: `v2/tests/equivalence/test_lora_equivalence.py`
- Modify: `v2/tests/integration/test_swag_workflow.py`

**Interfaces:**
- Consumes: static module specs and existing A/B-only checkpoint schema.
- Produces: FP32 merge from `spec.scale`, exact nonzero-dropout checkpoint replay, and unchanged plain BF16 export names.

- [ ] **Step 1: Write failing exact merge and resume assertions**

Change the merge oracle to:

```python
spec = tiny_adapted_model.layers[0].self_attn.q_proj.spec
scale = mx.array(spec.scale, dtype=mx.float32)
expected = (
    module.base.weight.astype(mx.float32)
    + scale * (module.lora_b @ module.lora_a)
).astype(mx.bfloat16)
assert bool(mx.array_equal(merged["layers.0.self_attn.q_proj.weight"], expected).item())
```

Change `test_uninterrupted_and_interrupted_adapter_state_match` to run with
`dropout=0.5`, nonzero A/B initialization, and `compile=True`. Retain exact
equality for adapters, optimizer arrays, trainer arrays (including `next_key`),
cursor, and step after two optimizer steps.

- [ ] **Step 2: Run merge/resume tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/training/test_lora.py v2/tests/integration/test_swag_workflow.py::test_uninterrupted_and_interrupted_adapter_state_match -q
```

Expected: merge still reads an array scale or resume diverges because pure-array
dropout/key state is incomplete.

- [ ] **Step 3: Use the static FP32 scale in merge and state loading**

`merged_model_weights` obtains each module's `LoRAAdapterSpec`, creates one
FP32 scale scalar, computes FP32 `scale * (B @ A)`, adds it to an FP32 cast of
the base, and casts once to BF16. It skips only `.lora_a` and `.lora_b` leaves;
`.scale` no longer exists.

`load_lora_state_dict` continues to require the exact A/B name, shape, and FP32
dtype set. Run construction and resume reconstruct the identical policy only
from strict run `LoRAConfig`; they never infer scale/dropout from checkpoint
arrays.

- [ ] **Step 4: Run the full LoRA/SWAG component gate**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/training/test_lora.py v2/tests/unit/training/test_swag.py -q
uv run pytest v2/tests/equivalence/test_lora_equivalence.py v2/tests/equivalence/test_swag_equivalence.py -q
uv run pytest v2/tests/integration/test_swag_workflow.py v2/tests/integration/test_part2_swag_flow.py -q
uv run ruff check v2
uv run ruff format --check v2
```

Expected: all precision, replay, compiled, resume, live/merged, export, and
offline flow tests pass. The committed historical SWAG quality validator may
still validate its historical source commit; no new evidence is recorded here.

- [ ] **Step 5: Commit the completed numerical correction**

```bash
git add v2/src/sml/training/lora.py v2/tests/unit/training/test_lora.py v2/tests/equivalence/test_lora_equivalence.py v2/tests/integration/test_swag_workflow.py
git commit -m "fix(v2): preserve fp32 lora scale across resume and merge"
```

## Plan Completion Gate

- [ ] **Step 1: Inspect parameter names and tracked state**

```bash
rg -n '"scale"|\.scale' v2/src/sml/model v2/src/sml/training/lora.py v2/src/sml/training/swag.py
git status --short
```

Expected: scale references are static spec access or explicit FP32 scalar
materialization only; no parameter-tree branch or checkpoint leaf remains; the
tracked checkout is clean.

- [ ] **Step 2: Review the LoRA diff against the approved numerical contract**

Confirm canonical site order, one-key-per-enabled-site behavior, no-key disabled
behavior, FP32 A/B/scale arithmetic, BF16 base/output boundaries, A/B-only
autodiff/checkpoints, exact resume, and unchanged dropout-zero equivalence.

Expected: no Critical, Important, or Minor findings before starting the
inference plan.
