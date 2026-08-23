# Model Training Input File Shuffle Implementation Plan

**Execution status (2026-08-23):** Implemented in v1. The unchecked boxes
below are the original TDD procedure, not remaining work.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Shuffle model-training input files deterministically while leaving tokenizer input ordering unchanged.

**Architecture:** Keep file discovery sorted and side-effect free. Add a model-training-only helper in `v1/src/train_sml.py` that shuffles a copied sequence with a local `random.Random(seed)`, then wire `train_model()` to call it when `TrainingConfig.shuffle_input_files` is true.

**Tech Stack:** Python 3.12, `uv run`, `unittest`, `unittest.mock`, `pathlib.Path`, Python standard-library `random`.

---

## File Structure

- Modify `v1/tests/test_train.py`: add focused unit tests for deterministic shuffling, non-mutating behavior, the config default, and `train_model()` wiring.
- Modify `v1/src/train_sml.py`: add `shuffle_input_files()` near `discover_input_files()` and call it in `train_model()` after discovery succeeds.
- Modify `v1/src/sml_config.py`: add `shuffle_input_files: bool = True` to `TrainingConfig`.
- Do not modify `v1/src/train_tokenizer.py`; tokenizer discovery stays sorted.

### Task 1: Add Failing Shuffle Helper Tests

**Files:**
- Modify: `v1/tests/test_train.py:1-68`
- Test: `v1/tests/test_train.py`

- [ ] **Step 1: Write the failing tests**

Add `from unittest import mock` after the existing `import unittest`, then add these tests immediately after `test_discover_input_files_uses_supplied_regex_and_sorts_matches`:

```python
    def test_shuffle_input_files_uses_seeded_deterministic_order(self):
        import train_sml

        files = tuple(
            Path(name)
            for name in (
                "pile-0000.jsonl.zst",
                "pile-0001.jsonl.zst",
                "pile-0002.jsonl.zst",
                "pile-0003.jsonl.zst",
            )
        )

        first_shuffle = train_sml.shuffle_input_files(files, seed=42)
        second_shuffle = train_sml.shuffle_input_files(files, seed=42)

        self.assertEqual(
            [
                "pile-0002.jsonl.zst",
                "pile-0001.jsonl.zst",
                "pile-0003.jsonl.zst",
                "pile-0000.jsonl.zst",
            ],
            [path.name for path in first_shuffle],
        )
        self.assertEqual(first_shuffle, second_shuffle)

    def test_shuffle_input_files_uses_seed_to_change_order(self):
        import train_sml

        files = tuple(
            Path(name)
            for name in (
                "pile-0000.jsonl.zst",
                "pile-0001.jsonl.zst",
                "pile-0002.jsonl.zst",
                "pile-0003.jsonl.zst",
            )
        )

        self.assertEqual(
            [
                "pile-0002.jsonl.zst",
                "pile-0000.jsonl.zst",
                "pile-0001.jsonl.zst",
                "pile-0003.jsonl.zst",
            ],
            [path.name for path in train_sml.shuffle_input_files(files, seed=99)],
        )

    def test_shuffle_input_files_returns_tuple_without_mutating_input(self):
        import train_sml

        files = [
            Path("pile-0000.jsonl.zst"),
            Path("pile-0001.jsonl.zst"),
            Path("pile-0002.jsonl.zst"),
            Path("pile-0003.jsonl.zst"),
        ]
        original_names = [path.name for path in files]

        shuffled = train_sml.shuffle_input_files(files, seed=42)

        self.assertIsInstance(shuffled, tuple)
        self.assertEqual(original_names, [path.name for path in files])
```

- [ ] **Step 2: Run helper tests to verify they fail**

Run:

```bash
env UV_CACHE_DIR=.uv-cache uv run python v1/tests/test_train.py -k shuffle_input_files
```

Expected: FAIL with `AttributeError: module 'train_sml' has no attribute 'shuffle_input_files'`.

- [ ] **Step 3: Implement the minimal helper**

Add this function to `v1/src/train_sml.py` immediately after `discover_input_files()`:

```python
def shuffle_input_files(input_files: Iterable[Path], seed: int) -> tuple[Path, ...]:
    shuffled_files = list(input_files)
    random.Random(seed).shuffle(shuffled_files)
    return tuple(shuffled_files)
```

- [ ] **Step 4: Run helper tests to verify they pass**

Run:

```bash
env UV_CACHE_DIR=.uv-cache uv run python v1/tests/test_train.py -k shuffle_input_files
```

Expected: PASS.

- [ ] **Step 5: Commit helper changes**

```bash
git add v1/src/train_sml.py v1/tests/test_train.py
git commit -m "Add deterministic input file shuffle helper"
```

### Task 2: Add Training Config and Wire `train_model()`

**Files:**
- Modify: `v1/tests/test_train.py:1-130`
- Modify: `v1/src/sml_config.py:127-150`
- Modify: `v1/src/train_sml.py:397-406`
- Test: `v1/tests/test_train.py`

- [ ] **Step 1: Write failing config and wiring tests**

Add these tests after the helper tests from Task 1:

```python
    def test_training_config_shuffles_input_files_by_default(self):
        from sml_config import TrainingConfig

        self.assertIs(True, TrainingConfig().shuffle_input_files)

    def test_train_model_shuffles_discovered_input_files_before_loading_tokenizer(self):
        import train_sml
        from sml_config import TrainingConfig

        discovered = (
            Path("pile-0000.jsonl.zst"),
            Path("pile-0001.jsonl.zst"),
        )
        shuffled = tuple(reversed(discovered))

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_config = TrainingConfig(
                input_dir=Path(tmp_dir),
                output_dir=Path(tmp_dir) / "output",
                tokenizer_model_path=Path(tmp_dir) / "tokenizer.model",
                shuffle_input_files=True,
                seed=123,
            )

            with (
                mock.patch.object(
                    train_sml,
                    "discover_input_files",
                    return_value=discovered,
                ),
                mock.patch.object(
                    train_sml,
                    "shuffle_input_files",
                    return_value=shuffled,
                ) as shuffle,
                mock.patch.object(
                    train_sml,
                    "load_tokenizer",
                    side_effect=RuntimeError("stop after shuffle"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after shuffle"):
                    train_sml.train_model(training_config)

        shuffle.assert_called_once_with(discovered, seed=123)

    def test_train_model_can_keep_discovered_input_file_order(self):
        import train_sml
        from sml_config import TrainingConfig

        discovered = (
            Path("pile-0000.jsonl.zst"),
            Path("pile-0001.jsonl.zst"),
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            training_config = TrainingConfig(
                input_dir=Path(tmp_dir),
                output_dir=Path(tmp_dir) / "output",
                tokenizer_model_path=Path(tmp_dir) / "tokenizer.model",
                shuffle_input_files=False,
                seed=123,
            )

            with (
                mock.patch.object(
                    train_sml,
                    "discover_input_files",
                    return_value=discovered,
                ),
                mock.patch.object(
                    train_sml,
                    "shuffle_input_files",
                    side_effect=AssertionError("shuffle should be skipped"),
                ),
                mock.patch.object(
                    train_sml,
                    "load_tokenizer",
                    side_effect=RuntimeError("stop after input order"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop after input order"):
                    train_sml.train_model(training_config)
```

- [ ] **Step 2: Run config and wiring tests to verify they fail**

Run:

```bash
env UV_CACHE_DIR=.uv-cache uv run python v1/tests/test_train.py -k shuffle_input_files -k train_model
```

Expected: FAIL with `AttributeError: 'TrainingConfig' object has no attribute 'shuffle_input_files'` or `TypeError: TrainingConfig.__init__() got an unexpected keyword argument 'shuffle_input_files'`.

- [ ] **Step 3: Add the config field**

Add this field to `TrainingConfig` in `v1/src/sml_config.py` immediately after `max_rows_per_file`:

```python
    shuffle_input_files: bool = True  # Shuffle model-training shards deterministically with seed.
```

- [ ] **Step 4: Wire the shuffle into `train_model()`**

In `v1/src/train_sml.py`, add this block immediately after the empty-input-files check:

```python
    if training_config.shuffle_input_files:
        input_files = shuffle_input_files(input_files, seed=training_config.seed)
```

The surrounding code should become:

```python
    input_files = discover_input_files(
        training_config.input_dir,
        training_config.input_file_name_regex,
    )
    if not input_files:
        raise FileNotFoundError(
            f"No supported input files found in {resolve_path(training_config.input_dir)}"
        )
    if training_config.shuffle_input_files:
        input_files = shuffle_input_files(input_files, seed=training_config.seed)

    tokenizer = load_tokenizer(training_config.tokenizer_model_path)
```

- [ ] **Step 5: Run config and wiring tests to verify they pass**

Run:

```bash
env UV_CACHE_DIR=.uv-cache uv run python v1/tests/test_train.py -k shuffle_input_files -k train_model
```

Expected: PASS.

- [ ] **Step 6: Commit config and wiring changes**

```bash
git add v1/src/sml_config.py v1/src/train_sml.py v1/tests/test_train.py
git commit -m "Shuffle model training input files"
```

### Task 3: Verify the Training Test Suite

**Files:**
- Test: `v1/tests/test_train.py`
- Test: `v1/tests/test_train_tokenizer.py`

- [ ] **Step 1: Run model-training tests**

Run:

```bash
env UV_CACHE_DIR=.uv-cache uv run python v1/tests/test_train.py
```

Expected: PASS.

- [ ] **Step 2: Run tokenizer-training tests to confirm untouched ordering**

Run:

```bash
env UV_CACHE_DIR=.uv-cache uv run python v1/tests/test_train_tokenizer.py
```

Expected: PASS.

- [ ] **Step 3: Inspect git diff**

Run:

```bash
git diff -- v1/src/train_sml.py v1/src/sml_config.py v1/tests/test_train.py v1/src/train_tokenizer.py
```

Expected: `v1/src/train_tokenizer.py` has no diff; the other diffs only cover helper tests, the shuffle helper, the config field, and `train_model()` wiring.

- [ ] **Step 4: Commit any verification-only cleanup**

If formatting or tiny cleanup was needed after verification, commit it:

```bash
git add v1/src/train_sml.py v1/src/sml_config.py v1/tests/test_train.py
git commit -m "Clean up input file shuffle tests"
```

If no cleanup was needed, do not create an empty commit.
