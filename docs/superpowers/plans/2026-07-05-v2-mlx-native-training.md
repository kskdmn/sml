# V2 MLX-Native Training Implementation Plan

**Execution status (2026-08-23):** Implemented in the legacy flat v2 tree and
superseded by the package refactor. The unchecked boxes below are historical,
not remaining tasks.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a v2 MLX-native training script whose hot loop uses MLX arrays, gradients, optimizer updates, and checkpoints.

**Architecture:** Add `v2/src/train_sml_mlx.py` as a sibling trainer that reuses cold-path helpers from `train_sml.py` but owns MLX batch creation, gradient accumulation, gradient clipping, optimizer scheduling, checkpoint save/load, and CLI. Keep the existing PyTorch trainer unchanged.

**Tech Stack:** Python 3.12, MLX (`mlx.core`, `mlx.nn`, `mlx.optimizers`), SentencePiece, zstandard, pytest.

---

### Task 1: MLX Batch Iterator

**Files:**
- Create: `v2/tests/test_train_mlx.py`
- Create: `v2/src/train_sml_mlx.py`

- [ ] **Step 1: Write failing tests**

```python
def test_mlx_batch_iterator_emits_mx_arrays():
    import mlx.core as mx
    from train_sml import TokenBlockDataset
    from train_sml_mlx import iter_mlx_batches

    dataset = TokenBlockDataset(["4 5 6 7 8"], FakeTokenizer(), sequence_length=3)
    batches = list(iter_mlx_batches(dataset, batch_size=2))

    assert batches[0]["input_ids"].shape == (2, 3)
    assert batches[0]["labels"].shape == (2, 3)
    assert isinstance(batches[0]["input_ids"], mx.array)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `env UV_CACHE_DIR=.uv-cache uv run pytest v2/tests/test_train_mlx.py::test_mlx_batch_iterator_emits_mx_arrays -q`

Expected: FAIL because `train_sml_mlx` does not exist.

- [ ] **Step 3: Implement minimal batch iterator**

Create `iter_mlx_batches(dataset, batch_size)` that collects examples from the existing token-block dataset, converts Python lists to `mx.array(..., dtype=mx.int32)`, and yields dictionaries with `input_ids` and `labels`.

- [ ] **Step 4: Run test to verify it passes**

Run: `env UV_CACHE_DIR=.uv-cache uv run pytest v2/tests/test_train_mlx.py::test_mlx_batch_iterator_emits_mx_arrays -q`

Expected: PASS.

### Task 2: MLX Optimizer Helpers

**Files:**
- Modify: `v2/tests/test_train_mlx.py`
- Modify: `v2/src/train_sml_mlx.py`

- [ ] **Step 1: Write failing helper tests**

```python
def test_mlx_lr_schedule_matches_torch_helper():
    from train_sml import lr_lambda
    from train_sml_mlx import build_lr_schedule

    schedule = build_lr_schedule(learning_rate=0.01, total_steps=10, warmup_steps=2, min_lr_ratio=0.1)

    assert float(schedule(mx.array(0)).item()) == pytest.approx(0.01 * lr_lambda(0, 10, 2, 0.1))
    assert float(schedule(mx.array(5)).item()) == pytest.approx(0.01 * lr_lambda(5, 10, 2, 0.1))

def test_clip_gradients_by_global_norm_scales_large_grads():
    from train_sml_mlx import clip_gradients_by_global_norm

    grads = {"w": mx.array([3.0, 4.0])}
    clipped, norm = clip_gradients_by_global_norm(grads, max_norm=1.0)

    assert float(norm.item()) == pytest.approx(5.0)
    assert float(mx.sqrt(mx.sum(clipped["w"] * clipped["w"])).item()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env UV_CACHE_DIR=.uv-cache uv run pytest v2/tests/test_train_mlx.py -q`

Expected: FAIL for missing helper functions.

- [ ] **Step 3: Implement helpers**

Add `build_lr_schedule`, `tree_add`, `tree_scale`, `zero_like_tree`, and `clip_gradients_by_global_norm` using MLX and `mlx.utils.tree_map/tree_reduce`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `env UV_CACHE_DIR=.uv-cache uv run pytest v2/tests/test_train_mlx.py -q`

Expected: PASS for helper tests.

### Task 3: Checkpoint Save/Load

**Files:**
- Modify: `v2/tests/test_train_mlx.py`
- Modify: `v2/src/train_sml_mlx.py`

- [ ] **Step 1: Write failing checkpoint tests**

```python
def test_resolve_mlx_checkpoint_path_avoids_pt_suffix(tmp_path):
    from train_sml import TrainingConfig
    from train_sml_mlx import resolve_mlx_checkpoint_path

    path = resolve_mlx_checkpoint_path(TrainingConfig(output_dir=tmp_path, checkpoint_name="sml.pt"))

    assert path == tmp_path / "sml_mlx"

def test_missing_explicit_resume_checkpoint_is_rejected(tmp_path):
    from train_sml import TrainingConfig
    from train_sml_mlx import load_training_checkpoint

    with pytest.raises(FileNotFoundError):
        load_training_checkpoint(tmp_path / "missing", object(), object())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env UV_CACHE_DIR=.uv-cache uv run pytest v2/tests/test_train_mlx.py -q`

Expected: FAIL for missing checkpoint helpers.

- [ ] **Step 3: Implement checkpoint helpers**

Add `resolve_mlx_checkpoint_path`, JSON metadata serialization, flattened optimizer state save/load with `mx.savez`, and model weights via `model.save_weights`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `env UV_CACHE_DIR=.uv-cache uv run pytest v2/tests/test_train_mlx.py -q`

Expected: PASS.

### Task 4: MLX Train Script

**Files:**
- Modify: `v2/tests/test_train_mlx.py`
- Modify: `v2/src/train_sml_mlx.py`

- [ ] **Step 1: Write failing script tests**

```python
def test_main_passes_resume_flag_to_train_model(monkeypatch):
    import train_sml_mlx

    train_model = Spy(return_value=Path("/tmp/sml_mlx"))
    monkeypatch.setattr(train_sml_mlx, "train_model", train_model)

    assert train_sml_mlx.SUCCESS_RETURN_CODE == train_sml_mlx.main(["--resume"])
    assert train_model.call_args.kwargs["resume_from_checkpoint"]

def test_tiny_mlx_training_run_writes_checkpoint(tmp_path, monkeypatch):
    import train_sml_mlx
    from train_sml import TrainingConfig

    monkeypatch.setattr(train_sml_mlx, "discover_input_files", Spy(return_value=(tmp_path / "data.jsonl.zst",)))
    monkeypatch.setattr(train_sml_mlx, "load_tokenizer", Spy(return_value=FakeTokenizer()))
    monkeypatch.setattr(train_sml_mlx, "iter_texts", lambda *args, **kwargs: iter(["4 5 6 7 8 9 10 11"]))

    checkpoint_path = train_sml_mlx.train_model(
        TrainingConfig(output_dir=tmp_path / "out", tokenizer_model_path=tmp_path / "tokenizer.model", sequence_length=4, batch_size=1, max_steps=1, gradient_accumulation_steps=1, save_every=1),
        model_config=tiny_config(),
    )

    assert (checkpoint_path / "model.safetensors").exists()
    assert (checkpoint_path / "optimizer.npz").exists()
    assert (checkpoint_path / "metadata.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `env UV_CACHE_DIR=.uv-cache uv run pytest v2/tests/test_train_mlx.py -q`

Expected: FAIL for missing `train_model`, `parse_args`, or checkpoint behavior.

- [ ] **Step 3: Implement trainer and CLI**

Implement `train_model`, `parse_args`, and `main`. The loop must call MLX `nn.value_and_grad`, accumulate MLX gradients, clip with MLX helpers, update with `mlx.optimizers.AdamW`, and evaluate model/optimizer state with `mx.eval`.

- [ ] **Step 4: Run focused verification**

Run: `env UV_CACHE_DIR=.uv-cache uv run pytest v2/tests/test_train_mlx.py -q`

Expected: PASS.

### Task 5: Full Verification

**Files:**
- Modify: none

- [ ] **Step 1: Run MLX-specific tests**

Run: `env UV_CACHE_DIR=.uv-cache uv run pytest v2/tests/test_sml_mlx.py v2/tests/test_train_mlx.py -q`

Expected: PASS.

- [ ] **Step 2: Run full v2 suite**

Run: `env UV_CACHE_DIR=.uv-cache uv run pytest v2/tests -q`

Expected: PASS.
