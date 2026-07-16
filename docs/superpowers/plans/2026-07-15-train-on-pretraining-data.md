# Train on Prepared Pretraining Data Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `v2/src/train_sml.py` train from prepared `pretraining_data` shards (`manifest.json` + `train-*.npz`), with shared format helpers used by prepare, train, and `peek_npz`.

**Architecture:** Extract on-disk format constants and manifest helpers into `v2/src/pretraining_format.py`. Replace online JSONL tokenization in `train_sml.py` with shard/block iteration and exact resume via `PretrainingDataState(shard_index, block_index)`. Keep existing `TrainingDataState` for `ft_swag.py`, which still resumes by `line_number` / `token_buffer`.

**Tech Stack:** Python 3.12, NumPy, MLX, SentencePiece, `uv run pytest`, `uv run ruff`

---

## File Structure

- Create: `v2/src/pretraining_format.py` — format constants, manifest load/validate, shard path resolution, default prepared-data dir
- Create: `v2/tests/test_pretraining_format.py` — unit tests for the shared module
- Modify: `v2/src/prepare_pretraining_data.py` — import shared format constants / default output dir
- Modify: `common/scripts/peek_npz.py` — import shared constants + `load_manifest` (add `v2/src` to `sys.path`)
- Modify: `v2/src/train_sml.py` — prepared-data loader, `PretrainingDataState`, config cleanup, train loop wiring
- Modify: `v2/tests/test_train.py` — replace JSONL/online-tokenization tests with prepared-shard tests
- Modify: `v2/README.md` — prepare-then-train docs
- Do not change `ft_swag.py` data-state shape; it keeps importing `TrainingDataState`

**Naming note vs design doc:** The approved design renamed `TrainingDataState` to shard/block fields. `ft_swag.py` still needs `line_number` / `token_buffer`, so this plan introduces `PretrainingDataState` for base training and leaves `TrainingDataState` for SWAG.

---

### Task 1: Shared `pretraining_format` Module

**Files:**
- Create: `v2/tests/test_pretraining_format.py`
- Create: `v2/src/pretraining_format.py`

- [ ] **Step 1: Write the failing tests**

```python
import json
from pathlib import Path

import pytest


def load_module():
    import importlib.util
    import sys

    project_dir = Path(__file__).resolve().parents[1]
    src_dir = project_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    spec = importlib.util.spec_from_file_location(
        "pretraining_format", src_dir / "pretraining_format.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_manifest_reads_json_object(tmp_path):
    fmt = load_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format": "sml-pretokenized-blocks-v1",
                "sequence_length": 4,
                "tokens_per_block": 5,
                "shards": [{"path": "train-000000.npz", "blocks": 1}],
            }
        ),
        encoding="utf-8",
    )

    manifest = fmt.load_manifest(manifest_path)

    assert manifest["format"] == "sml-pretokenized-blocks-v1"
    assert manifest["sequence_length"] == 4


def test_load_manifest_rejects_non_object(tmp_path):
    fmt = load_module()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON object"):
        fmt.load_manifest(manifest_path)


def test_validate_manifest_for_training_checks_format_and_sequence_length(tmp_path):
    fmt = load_module()
    manifest = {
        "format": "sml-pretokenized-blocks-v1",
        "sequence_length": 4,
        "tokens_per_block": 5,
        "tokenizer_vocab_size": 128,
        "shards": [{"path": "train-000000.npz", "blocks": 2}],
    }
    (tmp_path / "train-000000.npz").write_bytes(b"placeholder")

    shards = fmt.validate_manifest_for_training(
        manifest,
        manifest_dir=tmp_path,
        sequence_length=4,
        tokenizer_vocab_size=128,
    )

    assert shards == (tmp_path / "train-000000.npz",)


def test_validate_manifest_for_training_rejects_sequence_length_mismatch(tmp_path):
    fmt = load_module()
    manifest = {
        "format": "sml-pretokenized-blocks-v1",
        "sequence_length": 8,
        "tokens_per_block": 9,
        "tokenizer_vocab_size": 128,
        "shards": [{"path": "train-000000.npz", "blocks": 1}],
    }
    (tmp_path / "train-000000.npz").write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="sequence_length"):
        fmt.validate_manifest_for_training(
            manifest,
            manifest_dir=tmp_path,
            sequence_length=4,
            tokenizer_vocab_size=128,
        )


def test_validate_manifest_for_training_rejects_vocab_mismatch(tmp_path):
    fmt = load_module()
    manifest = {
        "format": "sml-pretokenized-blocks-v1",
        "sequence_length": 4,
        "tokens_per_block": 5,
        "tokenizer_vocab_size": 128,
        "shards": [{"path": "train-000000.npz", "blocks": 1}],
    }
    (tmp_path / "train-000000.npz").write_bytes(b"placeholder")

    with pytest.raises(ValueError, match="tokenizer_vocab_size"):
        fmt.validate_manifest_for_training(
            manifest,
            manifest_dir=tmp_path,
            sequence_length=4,
            tokenizer_vocab_size=256,
        )


def test_validate_manifest_for_training_rejects_missing_shard(tmp_path):
    fmt = load_module()
    manifest = {
        "format": "sml-pretokenized-blocks-v1",
        "sequence_length": 4,
        "tokens_per_block": 5,
        "tokenizer_vocab_size": 128,
        "shards": [{"path": "missing.npz", "blocks": 1}],
    }

    with pytest.raises(FileNotFoundError, match="missing.npz"):
        fmt.validate_manifest_for_training(
            manifest,
            manifest_dir=tmp_path,
            sequence_length=4,
            tokenizer_vocab_size=128,
        )


def test_shard_name_matches_prepare_convention():
    fmt = load_module()
    assert fmt.shard_name(0) == "train-000000.npz"
    assert fmt.shard_name(12) == "train-000012.npz"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest v2/tests/test_pretraining_format.py -v`

Expected: FAIL (module / symbols missing)

- [ ] **Step 3: Implement `pretraining_format.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

from config import OUTPUT_DIR, resolve_path


FORMAT_NAME = "sml-pretokenized-blocks-v1"
TOKENS_ARRAY_NAME = "tokens"
TOKEN_DTYPE_NAME = "uint16"
MANIFEST_NAME = "manifest.json"
SHARD_NAME_PREFIX = "train"
SHARD_NAME_SUFFIX = ".npz"
DEFAULT_PRETRAINING_DATA_DIR = OUTPUT_DIR / "pretraining_data"


def shard_name(shard_index: int) -> str:
    return f"{SHARD_NAME_PREFIX}-{shard_index:06d}{SHARD_NAME_SUFFIX}"


def load_manifest(manifest_path: Path) -> dict[str, object]:
    path = resolve_path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    return manifest


def validate_manifest_for_training(
    manifest: dict[str, object],
    *,
    manifest_dir: Path,
    sequence_length: int,
    tokenizer_vocab_size: int,
) -> tuple[Path, ...]:
    format_name = manifest.get("format")
    if format_name != FORMAT_NAME:
        raise ValueError(
            f"Unsupported pretraining format {format_name!r}; expected {FORMAT_NAME!r}"
        )

    manifest_sequence_length = manifest.get("sequence_length")
    if manifest_sequence_length != sequence_length:
        raise ValueError(
            "TrainingConfig.sequence_length "
            f"({sequence_length}) does not match manifest sequence_length "
            f"({manifest_sequence_length})"
        )

    expected_tokens_per_block = sequence_length + 1
    tokens_per_block = manifest.get("tokens_per_block")
    if tokens_per_block != expected_tokens_per_block:
        raise ValueError(
            f"manifest tokens_per_block ({tokens_per_block}) does not match "
            f"sequence_length + 1 ({expected_tokens_per_block})"
        )

    manifest_vocab_size = manifest.get("tokenizer_vocab_size")
    if manifest_vocab_size != tokenizer_vocab_size:
        raise ValueError(
            "tokenizer vocab size "
            f"({tokenizer_vocab_size}) does not match manifest tokenizer_vocab_size "
            f"({manifest_vocab_size})"
        )

    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("manifest shards must be a non-empty list")

    resolved: list[Path] = []
    root = resolve_path(manifest_dir)
    for entry in shards:
        if not isinstance(entry, dict):
            raise ValueError("manifest shard entries must be objects")
        relative = entry.get("path")
        if not isinstance(relative, str) or not relative:
            raise ValueError("manifest shard path must be a non-empty string")
        shard_path = root / relative
        if not shard_path.is_file():
            raise FileNotFoundError(f"Pretraining shard does not exist: {shard_path}")
        resolved.append(shard_path)
    return tuple(resolved)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest v2/tests/test_pretraining_format.py -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v2/src/pretraining_format.py v2/tests/test_pretraining_format.py
git commit -m "$(cat <<'EOF'
Add shared pretraining shard format helpers.

EOF
)"
```

---

### Task 2: Point Prepare + Peek at Shared Format Module

**Files:**
- Modify: `v2/src/prepare_pretraining_data.py`
- Modify: `common/scripts/peek_npz.py`
- Test: `v2/tests/test_prepare_pretraining_data.py`, `v2/tests/test_peek_npz.py`

- [ ] **Step 1: Update `prepare_pretraining_data.py` imports**

Remove local duplicates of `FORMAT_NAME`, `TOKENS_ARRAY_NAME`, `TOKEN_DTYPE_NAME`, `DEFAULT_OUTPUT_DIR`, `MANIFEST_NAME`, `SHARD_NAME_PREFIX`, `SHARD_NAME_SUFFIX`, and local `shard_name`.

Add:

```python
from pretraining_format import (
    DEFAULT_PRETRAINING_DATA_DIR as DEFAULT_OUTPUT_DIR,
    FORMAT_NAME,
    MANIFEST_NAME,
    TOKEN_DTYPE_NAME,
    TOKENS_ARRAY_NAME,
    shard_name,
)
```

Keep `UINT16_*` and preparation-only defaults in prepare. Drop unused `OUTPUT_DIR` import from `config` if it is only used for `DEFAULT_OUTPUT_DIR`.

- [ ] **Step 2: Update `peek_npz.py` to import shared constants**

Near the top, after `REPO_ROOT`:

```python
V2_SRC = REPO_ROOT / "v2" / "src"
if str(V2_SRC) not in sys.path:
    sys.path.insert(0, str(V2_SRC))

from pretraining_format import (  # noqa: E402
    MANIFEST_NAME,
    TOKENS_ARRAY_NAME,
    load_manifest as load_manifest_file,
)
```

Replace local `MANIFEST_NAME` / `TOKENS_ARRAY_NAME` constants.

Change `load_manifest(shard_path)` to:

```python
def load_manifest(shard_path: Path) -> dict[str, object] | None:
    manifest_path = shard_path.parent / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    return load_manifest_file(manifest_path)
```

- [ ] **Step 3: Run prepare + peek tests**

Run:

```bash
uv run pytest v2/tests/test_prepare_pretraining_data.py v2/tests/test_peek_npz.py v2/tests/test_pretraining_format.py -v
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add v2/src/prepare_pretraining_data.py common/scripts/peek_npz.py
git commit -m "$(cat <<'EOF'
Reuse shared pretraining format helpers in prepare and peek.

EOF
)"
```

---

### Task 3: Prepared-Block Iterator + Resume State

**Files:**
- Modify: `v2/src/train_sml.py`
- Modify: `v2/tests/test_train.py`

- [ ] **Step 1: Write failing tests for prepared-block iteration**

Add helpers and tests in `v2/tests/test_train.py` (near `TestCanonicalMlxTraining`). Keep `TrainingDataState` tests used by SWAG; add new `PretrainingDataState` coverage.

```python
def write_token_shard(path: Path, blocks: list[list[int]]) -> None:
    import numpy as np

    tokens = np.asarray(blocks, dtype=np.uint16)
    np.savez_compressed(path, tokens=tokens)


def write_pretraining_fixture(
    root: Path,
    *,
    sequence_length: int,
    vocab_size: int,
    shards: list[list[list[int]]],
) -> Path:
    import json

    shard_entries = []
    for index, blocks in enumerate(shards):
        relative = f"train-{index:06d}.npz"
        write_token_shard(root / relative, blocks)
        shard_entries.append({"path": relative, "blocks": len(blocks)})
    manifest = {
        "format": "sml-pretokenized-blocks-v1",
        "array_name": "tokens",
        "dtype": "uint16",
        "sequence_length": sequence_length,
        "tokens_per_block": sequence_length + 1,
        "tokenizer_vocab_size": vocab_size,
        "shards": shard_entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


class TestPreparedPretrainingBlocks:
    def test_iter_prepared_token_blocks_splits_input_ids_and_labels(self, tmp_path):
        import train_sml

        root = write_pretraining_fixture(
            tmp_path,
            sequence_length=4,
            vocab_size=128,
            shards=[
                [
                    [1, 10, 11, 12, 2],
                    [2, 1, 20, 21, 2],
                ]
            ],
        )
        data_state = train_sml.PretrainingDataState()
        progress = train_sml.ReadingProgress()

        blocks = list(
            train_sml.iter_prepared_token_blocks(
                shard_paths=(root / "train-000000.npz",),
                sequence_length=4,
                data_state=data_state,
                progress=progress,
            )
        )

        assert [
            {"input_ids": [1, 10, 11, 12], "labels": [10, 11, 12, 2]},
            {"input_ids": [2, 1, 20, 21], "labels": [1, 20, 21, 2]},
        ] == blocks
        assert data_state.shard_index == 0
        assert data_state.block_index == 2
        assert progress.input_file == "train-000000.npz"
        assert progress.line_number == 1

    def test_iter_prepared_token_blocks_resumes_from_shard_and_block(self, tmp_path):
        import train_sml

        root = write_pretraining_fixture(
            tmp_path,
            sequence_length=4,
            vocab_size=128,
            shards=[
                [[1, 10, 11, 12, 2], [2, 1, 20, 21, 2]],
                [[3, 30, 31, 32, 2]],
            ],
        )
        data_state = train_sml.PretrainingDataState(shard_index=1, block_index=0)

        blocks = list(
            train_sml.iter_prepared_token_blocks(
                shard_paths=(
                    root / "train-000000.npz",
                    root / "train-000001.npz",
                ),
                sequence_length=4,
                data_state=data_state,
            )
        )

        assert [
            {"input_ids": [3, 30, 31, 32], "labels": [30, 31, 32, 2]}
        ] == blocks

    def test_iter_prepared_token_blocks_rejects_wrong_block_width(self, tmp_path):
        import train_sml
        import numpy as np

        path = tmp_path / "train-000000.npz"
        np.savez_compressed(
            path,
            tokens=np.asarray([[1, 2, 3]], dtype=np.uint16),
        )

        with pytest.raises(ValueError, match="tokens_per_block"):
            list(
                train_sml.iter_prepared_token_blocks(
                    shard_paths=(path,),
                    sequence_length=4,
                )
            )

    def test_parse_checkpoint_pretraining_data_state_restores_fields(self):
        import train_sml

        data_state = train_sml.parse_checkpoint_pretraining_data_state(
            {"epoch": 2, "shard_index": 1, "block_index": 7}
        )

        assert data_state == train_sml.PretrainingDataState(
            epoch=2,
            shard_index=1,
            block_index=7,
        )

    def test_reset_pretraining_data_state_starts_new_epoch(self):
        import train_sml

        data_state = train_sml.PretrainingDataState(
            epoch=1,
            shard_index=2,
            block_index=9,
        )
        train_sml.reset_pretraining_data_state(data_state, epoch=3)

        assert data_state == train_sml.PretrainingDataState(epoch=3)
```

Also delete or stop calling tests that only cover removed train-path APIs once Task 4 removes them:

- `test_iter_texts_*`
- `test_mlx_token_blocks_*`
- `test_shuffle_input_files_*` in `test_train.py` (helper remains in `utils.py` for prepare/tokenizer; train no longer re-exports it unless still imported — remove train-specific shuffle tests that import via `train_sml`)
- `test_count_resume_batches_*` and `test_iter_unseen_batches_*` if those helpers are removed from `train_model`

Keep SWAG-facing `TrainingDataState` / `parse_checkpoint_data_state` tests unchanged.

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
uv run pytest v2/tests/test_train.py::TestPreparedPretrainingBlocks -v
```

Expected: FAIL (`PretrainingDataState` / `iter_prepared_token_blocks` missing)

- [ ] **Step 3: Implement prepared-data helpers in `train_sml.py`**

Add imports:

```python
import numpy as np

from pretraining_format import (
    DEFAULT_PRETRAINING_DATA_DIR,
    MANIFEST_NAME,
    TOKENS_ARRAY_NAME,
    load_manifest,
    validate_manifest_for_training,
)
```

Add:

```python
@dataclass(slots=True)
class PretrainingDataState:
    epoch: int = 0
    shard_index: int = 0
    block_index: int = 0


@dataclass(slots=True)
class TrainingResumeState:
    step: int = 0
    input_files: tuple[Path, ...] = ()
    data_state: PretrainingDataState | None = None
```

Keep the existing `TrainingDataState` dataclass and `parse_checkpoint_data_state` for `ft_swag`.

Add:

```python
def parse_checkpoint_pretraining_data_state(
    data_state: object,
) -> PretrainingDataState | None:
    if data_state is None:
        return None
    if not isinstance(data_state, dict):
        raise ValueError("Checkpoint data_state must be a dictionary")
    shard_index = data_state.get("shard_index", 0)
    block_index = data_state.get("block_index", 0)
    if not isinstance(shard_index, int):
        raise ValueError("Checkpoint data_state shard_index must be an integer")
    if not isinstance(block_index, int):
        raise ValueError("Checkpoint data_state block_index must be an integer")
    return PretrainingDataState(
        epoch=int(data_state.get("epoch", 0)),
        shard_index=shard_index,
        block_index=block_index,
    )


def reset_pretraining_data_state(data_state: PretrainingDataState, epoch: int) -> None:
    data_state.epoch = epoch
    data_state.shard_index = 0
    data_state.block_index = 0


def iter_prepared_token_blocks(
    shard_paths: Sequence[Path],
    sequence_length: int,
    data_state: PretrainingDataState | None = None,
    progress: ReadingProgress | None = None,
) -> Iterator[dict[str, list[int]]]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    tokens_per_block = sequence_length + 1
    start_shard = 0 if data_state is None else data_state.shard_index
    start_block = 0 if data_state is None else data_state.block_index
    if start_shard < 0 or start_shard > len(shard_paths):
        raise ValueError(
            f"shard_index must be in [0, {len(shard_paths)}]; got {start_shard}"
        )

    for shard_index, shard_path in enumerate(shard_paths):
        if shard_index < start_shard:
            continue
        with np.load(shard_path) as archive:
            if TOKENS_ARRAY_NAME not in archive:
                raise ValueError(
                    f"Expected array '{TOKENS_ARRAY_NAME}' in {shard_path}"
                )
            tokens = archive[TOKENS_ARRAY_NAME]
            if tokens.ndim != 2 or tokens.shape[1] != tokens_per_block:
                raise ValueError(
                    f"Shard {shard_path} tokens_per_block must be {tokens_per_block}; "
                    f"got shape {tokens.shape}"
                )
            if tokens.dtype != np.uint16:
                raise ValueError(
                    f"Shard {shard_path} tokens dtype must be uint16; got {tokens.dtype}"
                )
            block_start = start_block if shard_index == start_shard else 0
            if block_start < 0 or block_start > tokens.shape[0]:
                raise ValueError(
                    f"block_index must be in [0, {tokens.shape[0]}]; got {block_start}"
                )
            for block_index in range(block_start, tokens.shape[0]):
                block = [int(token_id) for token_id in tokens[block_index]]
                if progress is not None:
                    progress.input_file = shard_path.name
                    progress.line_number = block_index
                    progress.example_index = block_index
                if data_state is not None:
                    data_state.shard_index = shard_index
                    data_state.block_index = block_index + 1
                yield {
                    "input_ids": block[:-1],
                    "labels": block[1:],
                }
        start_block = 0
```

Update `load_training_checkpoint` to parse with `parse_checkpoint_pretraining_data_state`.

Update `save_checkpoint` type hints to accept `PretrainingDataState | None`.

Extend `format_training_log` so existing `input=` / `line=` fields continue to work (reuse `ReadingProgress.input_file` / `line_number` for shard name / block index as set above).

- [ ] **Step 4: Run prepared-block tests**

Run:

```bash
uv run pytest v2/tests/test_train.py::TestPreparedPretrainingBlocks -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add v2/src/train_sml.py v2/tests/test_train.py
git commit -m "$(cat <<'EOF'
Add prepared pretraining block iteration and resume state.

EOF
)"
```

---

### Task 4: Wire `train_model` to Prepared Data + Config Cleanup

**Files:**
- Modify: `v2/src/train_sml.py`
- Modify: `v2/tests/test_train.py`
- Modify: `v2/README.md`

- [ ] **Step 1: Update `TrainingConfig`**

Replace:

```python
    input_dir: Path = PRETRAINING_INPUT_DIR
    input_file_name_regex: str = PRETRAINING_INPUT_FILE_NAME_REGEX
```

and remove `max_rows_per_file` / `shuffle_input_files` with:

```python
    input_dir: Path = DEFAULT_PRETRAINING_DATA_DIR
```

Remove unused imports of `PRETRAINING_INPUT_DIR`, `PRETRAINING_INPUT_FILE_NAME_REGEX`, `discover_input_files`, `shuffle_input_files`, `filter_text`, `TEXT_COLUMN`, `iter_jsonl_records` if nothing else in the file needs them.

Remove train-path helpers that become unused:

- `iter_texts`
- `iter_mlx_token_blocks`
- `count_resume_batches`
- `iter_unseen_batches` / `ResumeProgress` if unused after wiring
- `TextTokenizer` protocol if unused

Keep `TrainingDataState`, `parse_checkpoint_data_state`, and `reset_training_data_state` for `ft_swag`.

- [ ] **Step 2: Rewrite the data-loading section of `train_model`**

Replace discover/shuffle/JSONL iteration with:

```python
    input_dir = resolve_path(training_config.input_dir)
    manifest_path = input_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Pretraining manifest does not exist: {manifest_path}")
    manifest = load_manifest(manifest_path)

    tokenizer = load_tokenizer(training_config.tokenizer_model_path)
    vocab_size = int(tokenizer.get_piece_size())
    input_files = validate_manifest_for_training(
        manifest,
        manifest_dir=input_dir,
        sequence_length=training_config.sequence_length,
        tokenizer_vocab_size=vocab_size,
    )
```

On resume, if `resume_state.input_files` is non-empty, use that shard tuple (same as today).

Replace the epoch loop body data path:

```python
        if epoch != data_state.epoch:
            reset_pretraining_data_state(data_state, epoch)
        reset_gradient_accumulation_window(accumulation)
        blocks = iter_prepared_token_blocks(
            shard_paths=input_files,
            sequence_length=training_config.sequence_length,
            data_state=data_state,
            progress=reading_progress,
        )
        batches = iter_mlx_batches(blocks, batch_size=training_config.batch_size)
        for batch in batches:
            ...
```

Initialize:

```python
    data_state = resume_state.data_state or PretrainingDataState()
```

Remove legacy `legacy_batches_to_skip` / `ResumeProgress` wiring.

Print `Input shards: {len(input_files)}` instead of (or in addition to renaming) `Input files`.

- [ ] **Step 3: Rewrite tiny training / resume integration tests**

Replace monkeypatched `iter_texts` / `discover_input_files` fixtures with real tiny prepared shards:

```python
    def test_tiny_mlx_training_run_writes_checkpoint(self, tmp_path, monkeypatch):
        require_mlx()
        import train_sml
        from train_sml import TrainingConfig

        data_dir = write_pretraining_fixture(
            tmp_path / "data",
            sequence_length=4,
            vocab_size=128,
            shards=[[[1, 4, 5, 6, 2], [2, 7, 8, 9, 2], [1, 10, 11, 12, 2]]],
        )
        tokenizer_path = tmp_path / "tokenizer.model"
        training_config = TrainingConfig(
            input_dir=data_dir,
            output_dir=tmp_path / "out",
            tokenizer_model_path=tokenizer_path,
            sequence_length=4,
            batch_size=1,
            max_steps=1,
            lr_total_steps=1,
            epochs=1,
            learning_rate=1e-4,
            gradient_accumulation_steps=1,
            log_every=1,
            save_every=1,
        )
        monkeypatch.setattr(
            train_sml, "load_tokenizer", Spy(return_value=FakeTokenizer())
        )

        checkpoint_path = train_sml.train_model(
            training_config=training_config,
            model_config=tiny_config(),
        )

        assert checkpoint_path == tmp_path / "out" / "sml"
        assert (checkpoint_path / train_sml.MODEL_WEIGHTS_NAME).exists()
        metadata = json.loads(
            (checkpoint_path / train_sml.METADATA_NAME).read_text(encoding="utf-8")
        )
        assert metadata["data_state"]["shard_index"] >= 0
        assert "block_index" in metadata["data_state"]
```

Similarly rewrite `test_tiny_mlx_training_run_can_resume_checkpoint` with the same fixture style and enough blocks for 2 steps.

Replace `test_train_model_uses_checkpoint_input_file_order_when_resume_is_enabled` so resume restores the checkpoint shard tuple into `iter_prepared_token_blocks` (monkeypatch that iterator and assert `shard_paths`).

Delete obsolete JSONL-only tests listed in Task 3.

- [ ] **Step 4: Update README train section**

In `v2/README.md`, under “Train the base model”, document that preparation must run first and default input is `v2/output/pretraining_data/`:

```markdown
### Train the base model

Trains the MLX SML language model from prepared pretraining shards under
`v2/output/pretraining_data/` (`manifest.json` + `train-*.npz`). Run
`prepare_pretraining_data.py` first (and tokenizer training unless you pass an
existing model with `--tokenizer-model`).

```sh
uv run python v2/src/prepare_pretraining_data.py
uv run python v2/src/train_sml.py
```
```

Also fix the outdated training-data path bullet at the top if it still implies `train_sml` reads JSONL directly — keep JSONL as the prepare/tokenizer input, and state prepared shards as the trainer input.

- [ ] **Step 5: Run focused + full v2 verification**

Run:

```bash
uv run ruff check v2
uv run ruff format --check v2
uv run pytest v2/tests -v
```

Expected: all PASS. Request non-sandbox / GPU permissions for pytest per `AGENTS.md`.

- [ ] **Step 6: Commit**

```bash
git add v2/src/train_sml.py v2/tests/test_train.py v2/README.md
git commit -m "$(cat <<'EOF'
Train SML from prepared pretraining shards.

EOF
)"
```

---

## Spec Coverage Checklist

| Spec requirement | Task |
| --- | --- |
| Shared `pretraining_format.py` | Task 1 |
| Prepare + peek reuse shared helpers | Task 2 |
| Replace JSONL train path with npz blocks | Tasks 3–4 |
| Exact resume `shard_index` / `block_index` | Task 3–4 (`PretrainingDataState`) |
| Manifest wins on `sequence_length` | Task 1 validate + Task 4 wire |
| Tokenizer vocab must match manifest | Task 1 validate + Task 4 wire |
| Remove JSONL-only TrainingConfig fields | Task 4 |
| Default input `pretraining_data` | Task 4 |
| Manifest shard order | Task 3–4 |
| Fail-fast errors | Tasks 1, 3 |
| Tests + README | Tasks 1–4 |
| No train-time shard shuffle / dual path / mmap / schema change | Out of scope (not implemented) |
| Design named `TrainingDataState` rename | Adapted: `PretrainingDataState` so `ft_swag` keeps `TrainingDataState` |

## Self-Review Notes

- No TBD / placeholder steps remain.
- `TrainingResumeState.data_state` type is `PretrainingDataState | None` for train; SWAG keeps its own resume parsing via existing `TrainingDataState` helpers.
- Progress logging reuses `ReadingProgress.input_file` / `line_number` as shard name / block index to avoid breaking `ft_swag`’s use of the same progress type.
