# V2 Checkpoint and SWAG Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Introduce a shared v2 checkpoint contract and correct SWAG resume safety, batching cost, LoRA precision, candidate scoring, merged-weight serialization, and inference tokenizer validation.

**Architecture:** Add `v2/src/checkpoint_io.py` as the only parser, identity, compatibility, and named-array serialization boundary for pretraining, SWAG fine-tuning, and inference. Keep LoRA transformation logic in `lora.py`, SWAG data/ranking logic in `ft_swag.py`, and generation logic in `infer_sml.py`; migrate each owner to the shared contract through focused TDD tasks.

**Tech Stack:** Python 3.12.13, MLX 0.32, SentencePiece 0.2, pytest, Ruff, safetensors through `mlx.core.save_safetensors`

## Global Constraints

- Use Python 3.12.13 and `uv run` for Python commands.
- Run every `uv run pytest` command outside the sandbox so MLX/Metal can access the Apple GPU.
- Update only `v2` and shared documentation; do not edit top-level project files such as `pyproject.toml` or `uv.lock`.
- Preserve compatibility with the current outputs of `v2/src/train_tokenizer.py` and `v2/src/train_sml.py`.
- Do not preserve compatibility with existing SWAG fine-tuning checkpoints; reject them clearly.
- Keep the SentencePiece `.model` and `.vocab` output format unchanged and require no tokenizer sidecar.
- Add no external dependencies and keep the v2 source MLX-only.
- Write each regression test first, run it and observe the intended failure, then implement the smallest production change that passes it.
- Keep training, fine-tuning, and inference performance ahead of test-only convenience; add no permanent test hooks to production code.
- Before completion run `uv run ruff check v2`, `uv run ruff format --check v2`, and `uv run pytest v2/tests`.

---

## File Structure

- Create: `v2/src/checkpoint_io.py` — versioned/legacy metadata parsing, artifact identities, tokenizer validation, strict named-array validation, and MLX serialization.
- Create: `v2/tests/test_checkpoint_io.py` — unit and MLX-backed tests for the shared checkpoint contract.
- Modify: `v2/src/train_sml.py` — write versioned pretraining metadata and resume current legacy checkpoints through `checkpoint_io`.
- Modify: `v2/tests/test_train.py` — cover new metadata and current legacy resume compatibility.
- Modify: `v2/src/infer_sml.py` — load parsed metadata and validate tokenizer identity before prompt encoding.
- Modify: `v2/tests/test_infer.py` — cover legacy/current and versioned tokenizer compatibility.
- Modify: `v2/src/lora.py` — strict adapter loading, FP32 adapter casting, and flat merged-weight export.
- Modify: `v2/tests/test_lora.py` — cover exact state contracts, precision, merged export, and source-model immutability.
- Modify: `v2/src/ft_swag.py` — dynamic padding, FP32 normalized scoring, shared checkpoint serialization, strict resume, and pre-loop step-limit handling.
- Modify: `v2/tests/test_ft_swag.py` — cover all SWAG behavior and checkpoint regressions.
- Modify: `v2/tests/test_module_layout.py` — assert the shared checkpoint module owns the new contract.
- Modify: `v2/README.md` — document checkpoint compatibility and strict SWAG resume requirements.

---

### Task 1: Shared Checkpoint Format and File Identities

**Files:**
- Create: `v2/tests/test_checkpoint_io.py`
- Create: `v2/src/checkpoint_io.py`
- Modify: `v2/tests/test_module_layout.py:36-53`

**Interfaces:**
- Consumes: `config.METADATA_NAME`, `config.MODEL_WEIGHTS_NAME`, and `config.resolve_path`.
- Produces: `ParsedCheckpointMetadata`, `read_checkpoint_metadata`, `write_checkpoint_metadata`, `build_checkpoint_metadata`, `build_file_identity`, `validate_file_identity`, and checkpoint format constants.

- [ ] **Step 1: Write failing metadata and identity tests**

Create `v2/tests/test_checkpoint_io.py` with source-path setup matching the other v2 tests and these initial tests:

```python
import json
import sys
from pathlib import Path

import pytest


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))


def legacy_pretraining_metadata() -> dict[str, object]:
    return {
        "step": 7,
        "model_config": {"vocab_size": 16},
        "training_config": {"tokenizer_model_path": "/tmp/tokenizer.model"},
        "data_state": {"epoch": 0, "shard_index": 1, "block_index": 2},
        "input_files": ["train-000000.npz"],
    }


def test_build_checkpoint_metadata_adds_versioned_common_fields():
    import checkpoint_io

    metadata = checkpoint_io.build_checkpoint_metadata(
        checkpoint_kind=checkpoint_io.PRETRAINING_CHECKPOINT_KIND,
        step=3,
        model_config={"vocab_size": 16},
        training_config={"sequence_length": 4},
        data_state={"epoch": 0},
        tokenizer={"sha256": "abc", "size": 3, "vocab_size": 16},
        extra={"input_files": ["train-000000.npz"]},
    )

    assert metadata["format"] == checkpoint_io.CHECKPOINT_FORMAT
    assert metadata["format_version"] == checkpoint_io.CHECKPOINT_FORMAT_VERSION
    assert metadata["checkpoint_kind"] == checkpoint_io.PRETRAINING_CHECKPOINT_KIND
    assert metadata["input_files"] == ["train-000000.npz"]


def test_read_checkpoint_metadata_normalizes_current_legacy_pretraining(tmp_path):
    import checkpoint_io

    checkpoint = tmp_path / "sml"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps(legacy_pretraining_metadata()),
        encoding="utf-8",
    )

    parsed = checkpoint_io.read_checkpoint_metadata(
        checkpoint,
        expected_kind=checkpoint_io.PRETRAINING_CHECKPOINT_KIND,
        allow_legacy_pretraining=True,
    )

    assert parsed.legacy
    assert parsed.format_version is None
    assert parsed.checkpoint_kind == checkpoint_io.PRETRAINING_CHECKPOINT_KIND
    assert parsed.step == 7
    assert parsed.model_config == {"vocab_size": 16}


def test_write_checkpoint_metadata_round_trips_versioned_metadata(tmp_path):
    import checkpoint_io

    checkpoint = tmp_path / "sml"
    metadata = checkpoint_io.build_checkpoint_metadata(
        checkpoint_kind=checkpoint_io.PRETRAINING_CHECKPOINT_KIND,
        step=3,
        model_config={"vocab_size": 16},
        training_config={"sequence_length": 4},
        data_state={"epoch": 0},
        tokenizer={"sha256": "a" * 64, "size": 3, "vocab_size": 16},
    )

    checkpoint_io.write_checkpoint_metadata(checkpoint, metadata)
    parsed = checkpoint_io.read_checkpoint_metadata(
        checkpoint,
        expected_kind=checkpoint_io.PRETRAINING_CHECKPOINT_KIND,
    )

    assert parsed.raw == metadata


def test_read_checkpoint_metadata_rejects_legacy_swag_resume(tmp_path):
    import checkpoint_io

    checkpoint = tmp_path / "sml-swag"
    checkpoint.mkdir()
    (checkpoint / "metadata.json").write_text(
        json.dumps(legacy_pretraining_metadata()),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Legacy SWAG checkpoints are unsupported"):
        checkpoint_io.read_checkpoint_metadata(
            checkpoint,
            expected_kind=checkpoint_io.SWAG_LORA_CHECKPOINT_KIND,
        )


def test_read_checkpoint_metadata_parses_versioned_swag(tmp_path):
    import checkpoint_io

    checkpoint = tmp_path / "sml-swag"
    metadata = checkpoint_io.build_checkpoint_metadata(
        checkpoint_kind=checkpoint_io.SWAG_LORA_CHECKPOINT_KIND,
        step=9,
        model_config={"vocab_size": 16},
        training_config={"max_steps": 10},
        data_state=None,
        tokenizer={"sha256": "a" * 64, "size": 3, "vocab_size": 16},
        extra={
            "lora_config": {"rank": 2},
            "base_checkpoint": {"sha256": "b" * 64, "size": 4},
        },
    )
    checkpoint_io.write_checkpoint_metadata(checkpoint, metadata)

    parsed = checkpoint_io.read_checkpoint_metadata(
        checkpoint,
        expected_kind=checkpoint_io.SWAG_LORA_CHECKPOINT_KIND,
    )

    assert not parsed.legacy
    assert parsed.checkpoint_kind == checkpoint_io.SWAG_LORA_CHECKPOINT_KIND
    assert parsed.step == 9


@pytest.mark.parametrize(
    ("field_name", "value", "message"),
    [
        ("format", "other", "Unsupported checkpoint format"),
        ("format_version", 99, "Unsupported checkpoint format_version"),
        ("checkpoint_kind", "other", "Unsupported checkpoint_kind"),
        ("tokenizer", None, "missing tokenizer"),
    ],
)
def test_read_checkpoint_metadata_rejects_malformed_versioned_metadata(
    tmp_path, field_name, value, message
):
    import checkpoint_io

    checkpoint = tmp_path / field_name
    metadata = checkpoint_io.build_checkpoint_metadata(
        checkpoint_kind=checkpoint_io.PRETRAINING_CHECKPOINT_KIND,
        step=1,
        model_config={"vocab_size": 16},
        training_config={},
        data_state=None,
        tokenizer={"sha256": "a" * 64, "size": 3, "vocab_size": 16},
    )
    metadata[field_name] = value
    checkpoint_io.write_checkpoint_metadata(checkpoint, metadata)

    with pytest.raises(ValueError, match=message):
        checkpoint_io.read_checkpoint_metadata(checkpoint)


def test_read_checkpoint_metadata_rejects_wrong_expected_kind(tmp_path):
    import checkpoint_io

    checkpoint = tmp_path / "pretraining"
    metadata = checkpoint_io.build_checkpoint_metadata(
        checkpoint_kind=checkpoint_io.PRETRAINING_CHECKPOINT_KIND,
        step=1,
        model_config={"vocab_size": 16},
        training_config={},
        data_state=None,
        tokenizer={"sha256": "a" * 64, "size": 3, "vocab_size": 16},
    )
    checkpoint_io.write_checkpoint_metadata(checkpoint, metadata)

    with pytest.raises(ValueError, match="Expected checkpoint_kind"):
        checkpoint_io.read_checkpoint_metadata(
            checkpoint,
            expected_kind=checkpoint_io.SWAG_LORA_CHECKPOINT_KIND,
        )


def test_file_identity_uses_content_instead_of_path(tmp_path):
    import checkpoint_io

    first = tmp_path / "first.bin"
    second = tmp_path / "moved.bin"
    first.write_bytes(b"same checkpoint bytes")
    second.write_bytes(first.read_bytes())

    first_identity = checkpoint_io.build_file_identity(first)
    second_identity = checkpoint_io.build_file_identity(second)

    assert first_identity["path"] != second_identity["path"]
    assert first_identity["sha256"] == second_identity["sha256"]
    assert first_identity["size"] == second_identity["size"]
    checkpoint_io.validate_file_identity(first_identity, second_identity, "model")


def test_file_identity_rejects_modified_content(tmp_path):
    import checkpoint_io

    expected_path = tmp_path / "expected.bin"
    actual_path = tmp_path / "actual.bin"
    expected_path.write_bytes(b"base A")
    actual_path.write_bytes(b"base B")

    with pytest.raises(ValueError, match="model fingerprint mismatch"):
        checkpoint_io.validate_file_identity(
            checkpoint_io.build_file_identity(expected_path),
            checkpoint_io.build_file_identity(actual_path),
            "model",
        )
```

Also extend `test_model_training_and_inference_configs_live_with_owners` before
creating the module:

```python
checkpoint_io = importlib.import_module("checkpoint_io")
assert hasattr(checkpoint_io, "ParsedCheckpointMetadata")
assert hasattr(checkpoint_io, "read_checkpoint_metadata")
```

- [ ] **Step 2: Run the new test module and observe RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_checkpoint_io.py v2/tests/test_module_layout.py -v
```

Expected: FAIL because `checkpoint_io` does not exist.

- [ ] **Step 3: Implement the format parser and file identities**

Create `v2/src/checkpoint_io.py` with the following public shape and validation logic:

```python
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import METADATA_NAME, resolve_path


CHECKPOINT_FORMAT = "sml-checkpoint"
CHECKPOINT_FORMAT_VERSION = 1
PRETRAINING_CHECKPOINT_KIND = "pretraining"
SWAG_LORA_CHECKPOINT_KIND = "swag_lora"
CHECKPOINT_KINDS = (PRETRAINING_CHECKPOINT_KIND, SWAG_LORA_CHECKPOINT_KIND)
HASH_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ParsedCheckpointMetadata:
    raw: dict[str, Any]
    checkpoint_kind: str
    format_version: int | None
    legacy: bool
    step: int
    model_config: dict[str, Any]
    training_config: dict[str, Any]
    data_state: object
    tokenizer: dict[str, Any] | None


def _require_dict(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Checkpoint metadata {field_name} must be a dictionary")
    return value


def _parse_common(
    raw: dict[str, Any],
    *,
    checkpoint_kind: str,
    format_version: int | None,
    legacy: bool,
) -> ParsedCheckpointMetadata:
    step = raw.get("step")
    if not isinstance(step, int):
        raise ValueError("Checkpoint metadata is missing integer step")
    return ParsedCheckpointMetadata(
        raw=raw,
        checkpoint_kind=checkpoint_kind,
        format_version=format_version,
        legacy=legacy,
        step=step,
        model_config=_require_dict(raw.get("model_config"), "model_config"),
        training_config=_require_dict(raw.get("training_config"), "training_config"),
        data_state=raw.get("data_state"),
        tokenizer=(
            None
            if raw.get("tokenizer") is None
            else _require_dict(raw.get("tokenizer"), "tokenizer")
        ),
    )


def read_checkpoint_metadata(
    checkpoint_path: Path,
    *,
    expected_kind: str | None = None,
    allow_legacy_pretraining: bool = False,
) -> ParsedCheckpointMetadata:
    metadata_path = resolve_path(checkpoint_path) / METADATA_NAME
    if not metadata_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata does not exist: {metadata_path}")
    raw = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint metadata must contain a dictionary: {metadata_path}")

    format_name = raw.get("format")
    format_version = raw.get("format_version")
    if format_name is None and format_version is None:
        if expected_kind == SWAG_LORA_CHECKPOINT_KIND:
            raise ValueError("Legacy SWAG checkpoints are unsupported")
        if not allow_legacy_pretraining:
            raise ValueError("Legacy pretraining checkpoint is not allowed here")
        return _parse_common(
            raw,
            checkpoint_kind=PRETRAINING_CHECKPOINT_KIND,
            format_version=None,
            legacy=True,
        )

    if format_name != CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported checkpoint format: {format_name!r}")
    if format_version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format_version: {format_version!r}")
    checkpoint_kind = raw.get("checkpoint_kind")
    if checkpoint_kind not in CHECKPOINT_KINDS:
        raise ValueError(f"Unsupported checkpoint_kind: {checkpoint_kind!r}")
    if expected_kind is not None and checkpoint_kind != expected_kind:
        raise ValueError(
            f"Expected checkpoint_kind {expected_kind!r}; got {checkpoint_kind!r}"
        )
    parsed = _parse_common(
        raw,
        checkpoint_kind=checkpoint_kind,
        format_version=format_version,
        legacy=False,
    )
    if parsed.tokenizer is None:
        raise ValueError("Versioned checkpoint metadata is missing tokenizer")
    if checkpoint_kind == SWAG_LORA_CHECKPOINT_KIND:
        _require_dict(raw.get("lora_config"), "lora_config")
        _require_dict(raw.get("base_checkpoint"), "base_checkpoint")
    return parsed


def build_checkpoint_metadata(
    *,
    checkpoint_kind: str,
    step: int,
    model_config: Mapping[str, object],
    training_config: Mapping[str, object],
    data_state: object,
    tokenizer: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if checkpoint_kind not in CHECKPOINT_KINDS:
        raise ValueError(f"Unsupported checkpoint_kind: {checkpoint_kind!r}")
    metadata: dict[str, object] = {
        "format": CHECKPOINT_FORMAT,
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_kind": checkpoint_kind,
        "step": step,
        "model_config": dict(model_config),
        "training_config": dict(training_config),
        "data_state": data_state,
        "tokenizer": dict(tokenizer),
    }
    if extra is not None:
        metadata.update(extra)
    return metadata


def write_checkpoint_metadata(
    checkpoint_path: Path,
    metadata: Mapping[str, object],
) -> None:
    resolved = resolve_path(checkpoint_path)
    resolved.mkdir(parents=True, exist_ok=True)
    (resolved / METADATA_NAME).write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_file_identity(path: Path) -> dict[str, object]:
    resolved = resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as artifact_file:
        while chunk := artifact_file.read(HASH_CHUNK_SIZE):
            digest.update(chunk)
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size": resolved.stat().st_size,
    }


def validate_file_identity(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    artifact_name: str,
) -> None:
    for field_name in ("sha256", "size"):
        if expected.get(field_name) != actual.get(field_name):
            raise ValueError(f"{artifact_name} fingerprint mismatch ({field_name})")
```

- [ ] **Step 4: Run the checkpoint tests and observe GREEN**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_checkpoint_io.py v2/tests/test_module_layout.py -v
```

Expected: all Task 1 tests PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add v2/src/checkpoint_io.py v2/tests/test_checkpoint_io.py v2/tests/test_module_layout.py
git commit -m "feat(v2): add shared checkpoint metadata format"
```

---

### Task 2: Tokenizer Identities and Strict MLX Array State

**Files:**
- Modify: `v2/tests/test_checkpoint_io.py`
- Modify: `v2/src/checkpoint_io.py`

**Interfaces:**
- Consumes: Task 1 `build_file_identity` and `validate_file_identity`; `utils.get_special_token_id`.
- Produces: `build_tokenizer_identity`, `validate_tokenizer_identity`, `build_model_checkpoint_identity`, `load_array_state`, `save_array_state`, `save_safetensors`, and `validate_array_state`.

- [ ] **Step 1: Add failing tokenizer and state-contract tests**

Append tests that use a real temporary tokenizer artifact and MLX arrays:

```python
class FakeTokenizer:
    def get_piece_size(self):
        return 16

    def bos_id(self):
        return 1

    def eos_id(self):
        return 2

    def pad_id(self):
        return 3


def test_tokenizer_identity_records_content_and_structure(tmp_path):
    import checkpoint_io

    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"sentencepiece model")

    identity = checkpoint_io.build_tokenizer_identity(
        tokenizer_path,
        FakeTokenizer(),
        fallback_bos_token_id=1,
        fallback_eos_token_id=2,
        fallback_pad_token_id=3,
    )

    assert identity["vocab_size"] == 16
    assert identity["bos_token_id"] == 1
    assert identity["eos_token_id"] == 2
    assert identity["pad_token_id"] == 3
    assert len(identity["sha256"]) == 64


@pytest.mark.parametrize(
    ("field_name", "wrong_value", "message"),
    [
        ("vocab_size", 17, "vocabulary size mismatch"),
        ("bos_token_id", 9, "BOS token mismatch"),
        ("eos_token_id", 9, "EOS token mismatch"),
        ("pad_token_id", 9, "PAD token mismatch"),
    ],
)
def test_tokenizer_identity_rejects_structural_mismatch(
    tmp_path, field_name, wrong_value, message
):
    import checkpoint_io

    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"sentencepiece model")
    actual = checkpoint_io.build_tokenizer_identity(
        tokenizer_path,
        FakeTokenizer(),
        fallback_bos_token_id=1,
        fallback_eos_token_id=2,
        fallback_pad_token_id=3,
    )
    expected = dict(actual)
    expected[field_name] = wrong_value

    with pytest.raises(ValueError, match=message):
        checkpoint_io.validate_tokenizer_identity(expected, actual)


def test_validate_array_state_rejects_missing_and_unexpected_keys():
    mx = pytest.importorskip("mlx.core")
    import checkpoint_io

    expected = {"layer.weight": mx.zeros((2, 2))}
    actual = {"other.weight": mx.zeros((2, 2))}

    with pytest.raises(ValueError, match="missing.*layer.weight.*unexpected.*other.weight"):
        checkpoint_io.validate_array_state(expected, actual, "optimizer")


def test_validate_array_state_rejects_shape_and_dtype_mismatch():
    mx = pytest.importorskip("mlx.core")
    import checkpoint_io

    expected = {"state": mx.zeros((2, 2), dtype=mx.float32)}
    with pytest.raises(ValueError, match="wrong shape"):
        checkpoint_io.validate_array_state(
            expected,
            {"state": mx.zeros((3, 2), dtype=mx.float32)},
            "adapter",
        )
    with pytest.raises(ValueError, match="wrong dtype"):
        checkpoint_io.validate_array_state(
            expected,
            {"state": mx.zeros((2, 2), dtype=mx.bfloat16)},
            "adapter",
        )
```

- [ ] **Step 2: Run the new tests and observe RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_checkpoint_io.py -v
```

Expected: FAIL because the tokenizer and array-state functions are missing.

- [ ] **Step 3: Implement tokenizer and strict state helpers**

Add these imports and functions to `checkpoint_io.py`:

```python
import mlx.core as mx

from config import MODEL_WEIGHTS_NAME
from utils import get_special_token_id


def build_tokenizer_identity(
    tokenizer_model_path: Path,
    tokenizer: object,
    *,
    fallback_bos_token_id: int | None,
    fallback_eos_token_id: int | None,
    fallback_pad_token_id: int | None,
) -> dict[str, object]:
    identity = build_file_identity(tokenizer_model_path)
    identity.update(
        {
            "vocab_size": int(tokenizer.get_piece_size()),
            "bos_token_id": get_special_token_id(
                tokenizer, "bos_id", fallback_bos_token_id
            ),
            "eos_token_id": get_special_token_id(
                tokenizer, "eos_id", fallback_eos_token_id
            ),
            "pad_token_id": get_special_token_id(
                tokenizer, "pad_id", fallback_pad_token_id
            ),
        }
    )
    return identity


def validate_tokenizer_identity(
    expected: Mapping[str, object],
    actual: Mapping[str, object],
    *,
    require_fingerprint: bool = True,
) -> None:
    field_messages = {
        "vocab_size": "tokenizer vocabulary size mismatch",
        "bos_token_id": "tokenizer BOS token mismatch",
        "eos_token_id": "tokenizer EOS token mismatch",
        "pad_token_id": "tokenizer PAD token mismatch",
    }
    for field_name, message in field_messages.items():
        if expected.get(field_name) != actual.get(field_name):
            raise ValueError(message)
    if require_fingerprint:
        validate_file_identity(expected, actual, "tokenizer")


def build_model_checkpoint_identity(
    checkpoint_path: Path,
    model_config: Mapping[str, object],
) -> dict[str, object]:
    identity = build_file_identity(resolve_path(checkpoint_path) / MODEL_WEIGHTS_NAME)
    identity["model_config"] = dict(model_config)
    return identity


def validate_array_state(
    expected: Mapping[str, mx.array],
    actual: Mapping[str, mx.array],
    state_name: str,
) -> None:
    expected_keys = set(expected)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"{state_name} state keys mismatch: missing={missing}, unexpected={unexpected}"
        )
    for key in sorted(expected_keys):
        if tuple(expected[key].shape) != tuple(actual[key].shape):
            raise ValueError(f"{state_name} state has wrong shape for {key}")
        if expected[key].dtype != actual[key].dtype:
            raise ValueError(f"{state_name} state has wrong dtype for {key}")


def load_array_state(path: Path, state_name: str) -> dict[str, mx.array]:
    arrays = mx.load(str(resolve_path(path)))
    if not isinstance(arrays, dict):
        raise ValueError(f"{state_name} checkpoint must contain a state dictionary")
    return arrays


def save_array_state(path: Path, arrays: Mapping[str, mx.array]) -> None:
    mx.savez(str(resolve_path(path)), **dict(arrays))


def save_safetensors(path: Path, arrays: Mapping[str, mx.array]) -> None:
    mx.save_safetensors(str(resolve_path(path)), dict(arrays))
```

- [ ] **Step 4: Run Task 2 tests and observe GREEN**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_checkpoint_io.py -v
```

Expected: all checkpoint I/O tests PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add v2/src/checkpoint_io.py v2/tests/test_checkpoint_io.py
git commit -m "feat(v2): validate checkpoint artifacts and array state"
```

---

### Task 3: Pretraining Checkpoint Producer and Legacy Resume

**Files:**
- Modify: `v2/src/train_sml.py:627-694,697-888`
- Modify: `v2/tests/test_train.py:857-1147`

**Interfaces:**
- Consumes: `checkpoint_io.build_checkpoint_metadata`, `write_checkpoint_metadata`, `build_tokenizer_identity`, `read_checkpoint_metadata`, `load_array_state`, `save_array_state`, `save_safetensors`, and `validate_array_state`.
- Produces: versioned pretraining checkpoint output while retaining `save_checkpoint(...)` and `load_training_checkpoint(...)` as training-owned APIs.

- [ ] **Step 1: Add failing versioned-save and legacy-resume tests**

Update the tiny training fixture so `tokenizer_path.write_bytes(b"tokenizer")` exists even when `load_tokenizer` is monkeypatched. Extend `test_tiny_mlx_training_run_writes_checkpoint`:

```python
assert metadata["format"] == "sml-checkpoint"
assert metadata["format_version"] == 1
assert metadata["checkpoint_kind"] == "pretraining"
assert metadata["tokenizer"]["vocab_size"] == 16
assert metadata["tokenizer"]["bos_token_id"] == 1
assert len(metadata["tokenizer"]["sha256"]) == 64
```

Add a direct legacy round-trip regression that writes metadata in the exact current `train_sml.py` shape, saves model/optimizer arrays, and resumes it:

```python
def test_load_training_checkpoint_accepts_current_legacy_output(tmp_path):
    mx = require_mlx()
    import json
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    import train_sml
    from sml import SMLLanguageModel

    checkpoint = tmp_path / "legacy"
    checkpoint.mkdir()
    model = SMLLanguageModel(train_sml.model_config_for_training(tiny_config()))
    train_sml.apply_model_dtype(model, "bfloat16")
    optimizer = optim.AdamW(learning_rate=1e-4)
    optimizer.init(model.trainable_parameters())
    model.save_weights(str(checkpoint / train_sml.MODEL_WEIGHTS_NAME))
    mx.savez(
        str(checkpoint / train_sml.OPTIMIZER_STATE_NAME),
        **tree_flatten(optimizer.state, destination={}),
    )
    (checkpoint / train_sml.METADATA_NAME).write_text(
        json.dumps(
            {
                "step": 4,
                "model_config": asdict(tiny_config()),
                "training_config": asdict(train_sml.TrainingConfig()),
                "input_files": [],
                "data_state": {"epoch": 0, "shard_index": 0, "block_index": 2},
            },
            default=str,
        ),
        encoding="utf-8",
    )

    state = train_sml.load_training_checkpoint(checkpoint, model, optimizer)

    assert state.step == 4
    assert state.data_state == train_sml.PretrainingDataState(block_index=2)
```

- [ ] **Step 2: Run the focused training tests and observe RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_train.py::TestCanonicalMlxTraining -v
```

Expected: the new metadata assertions fail because the save path still writes legacy metadata.

- [ ] **Step 3: Route pretraining save/load through `checkpoint_io`**

In `train_sml.py`:

1. Import shared checkpoint helpers and `MODEL_WEIGHTS_NAME`/`OPTIMIZER_STATE_NAME` as today.
2. Add required `tokenizer_identity: Mapping[str, object]` to `save_checkpoint`.
3. Replace direct model/optimizer serialization with flat-state serialization:

```python
model_weights = dict(tree_flatten(model.parameters()))
save_safetensors(checkpoint_path / MODEL_WEIGHTS_NAME, model_weights)
optimizer_state = tree_flatten(optimizer.state, destination={})
save_array_state(checkpoint_path / OPTIMIZER_STATE_NAME, optimizer_state)
metadata = build_checkpoint_metadata(
    checkpoint_kind=PRETRAINING_CHECKPOINT_KIND,
    step=step,
    model_config=json_ready(asdict(model_config)),
    training_config=json_ready(asdict(training_config)),
    data_state=None if data_state is None else json_ready(asdict(data_state)),
    tokenizer=tokenizer_identity,
    extra={
        "input_files": [str(input_file) for input_file in input_files],
        "stochastic_resume": "not_guaranteed",
        "resume_note": STOCHASTIC_RESUME_NOTE,
    },
)
write_checkpoint_metadata(checkpoint_path, metadata)
```

4. Make `load_training_checkpoint` call:

```python
parsed = read_checkpoint_metadata(
    checkpoint_path,
    expected_kind=PRETRAINING_CHECKPOINT_KIND,
    allow_legacy_pretraining=True,
)
model.load_weights(str(checkpoint_path / MODEL_WEIGHTS_NAME))
optimizer_arrays = load_array_state(
    checkpoint_path / OPTIMIZER_STATE_NAME,
    "optimizer",
)
expected_optimizer = tree_flatten(optimizer.state, destination={})
validate_array_state(expected_optimizer, optimizer_arrays, "optimizer")
optimizer.state = tree_unflatten(optimizer_arrays)
```

5. In `train_model`, immediately after loading the tokenizer and creating `checkpoint_model_config`, compute the identity once:

```python
tokenizer_identity = build_tokenizer_identity(
    training_config.tokenizer_model_path,
    tokenizer,
    fallback_bos_token_id=checkpoint_model_config.bos_token_id,
    fallback_eos_token_id=checkpoint_model_config.eos_token_id,
    fallback_pad_token_id=checkpoint_model_config.pad_token_id,
)
```

6. Pass the same `tokenizer_identity` into every `save_checkpoint` call.

- [ ] **Step 4: Run all training tests and observe GREEN**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_train.py -v
```

Expected: all training tests PASS, including legacy resume and versioned output.

- [ ] **Step 5: Commit Task 3**

```bash
git add v2/src/train_sml.py v2/tests/test_train.py
git commit -m "feat(v2): version pretraining checkpoints"
```

---

### Task 4: Inference Tokenizer Compatibility

**Files:**
- Modify: `v2/src/infer_sml.py:46-76,133-184`
- Modify: `v2/tests/test_infer.py:38-177`

**Interfaces:**
- Consumes: `ParsedCheckpointMetadata`, `read_checkpoint_metadata`, `build_tokenizer_identity`, and `validate_tokenizer_identity`.
- Produces: `validate_inference_tokenizer(metadata, tokenizer_path, tokenizer, model_config) -> None`; preserves `load_model(checkpoint_path)` and `generate_text(...)` public behavior.

- [ ] **Step 1: Add failing versioned and legacy inference tests**

Extend `FakeTokenizer` with constructor defaults for vocabulary size and all
three special-token IDs, plus `get_piece_size`, `bos_id`, `eos_id`, and
`pad_id` methods. Existing no-argument uses retain values 16, 1, 2, and 3.
Add:

First update the metadata fixture used by the existing
`test_load_model_restores_checkpoint_weights` test to the exact supported legacy
pretraining shape: retain `model_config`, and add integer `step`, dictionary
`training_config`, `data_state`, and `input_files`. This is necessary because the
shared parser intentionally accepts legacy `train_sml.py` output, not arbitrary
partial metadata.

```python
def test_validate_inference_tokenizer_accepts_moved_versioned_tokenizer(tmp_path):
    import checkpoint_io
    import infer_sml

    first = tmp_path / "original.model"
    moved = tmp_path / "moved.model"
    first.write_bytes(b"same tokenizer")
    moved.write_bytes(first.read_bytes())
    expected = checkpoint_io.build_tokenizer_identity(
        first,
        FakeTokenizer(),
        fallback_bos_token_id=1,
        fallback_eos_token_id=2,
        fallback_pad_token_id=3,
    )
    metadata = checkpoint_io.ParsedCheckpointMetadata(
        raw={},
        checkpoint_kind="pretraining",
        format_version=1,
        legacy=False,
        step=0,
        model_config=asdict(self.tiny_config()),
        training_config={},
        data_state=None,
        tokenizer=expected,
    )

    infer_sml.validate_inference_tokenizer(
        metadata,
        moved,
        FakeTokenizer(),
        self.tiny_config(),
    )


def test_validate_inference_tokenizer_rejects_same_size_different_content(tmp_path):
    import checkpoint_io
    import infer_sml

    expected_path = tmp_path / "expected.model"
    actual_path = tmp_path / "actual.model"
    expected_path.write_bytes(b"tokenizer A")
    actual_path.write_bytes(b"tokenizer B")
    expected = checkpoint_io.build_tokenizer_identity(
        expected_path,
        FakeTokenizer(),
        fallback_bos_token_id=1,
        fallback_eos_token_id=2,
        fallback_pad_token_id=3,
    )
    metadata = checkpoint_io.ParsedCheckpointMetadata(
        raw={}, checkpoint_kind="pretraining", format_version=1, legacy=False,
        step=0, model_config=asdict(self.tiny_config()), training_config={},
        data_state=None, tokenizer=expected,
    )

    with pytest.raises(ValueError, match="tokenizer fingerprint mismatch"):
        infer_sml.validate_inference_tokenizer(
            metadata, actual_path, FakeTokenizer(), self.tiny_config()
        )


@pytest.mark.parametrize(
    ("tokenizer", "message"),
    [
        (FakeTokenizer(vocab_size=17), "vocabulary size mismatch"),
        (FakeTokenizer(bos_token_id=9), "BOS token mismatch"),
        (FakeTokenizer(eos_token_id=9), "EOS token mismatch"),
        (FakeTokenizer(pad_token_id=9), "PAD token mismatch"),
    ],
)
def test_validate_inference_tokenizer_rejects_structural_mismatch(
    tmp_path, tokenizer, message
):
    import checkpoint_io
    import infer_sml

    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"tokenizer")
    expected = checkpoint_io.build_tokenizer_identity(
        tokenizer_path,
        FakeTokenizer(),
        fallback_bos_token_id=1,
        fallback_eos_token_id=2,
        fallback_pad_token_id=3,
    )
    metadata = checkpoint_io.ParsedCheckpointMetadata(
        raw={}, checkpoint_kind="pretraining", format_version=1, legacy=False,
        step=0, model_config=asdict(self.tiny_config()), training_config={},
        data_state=None, tokenizer=expected,
    )

    with pytest.raises(ValueError, match=message):
        infer_sml.validate_inference_tokenizer(
            metadata, tokenizer_path, tokenizer, self.tiny_config()
        )


def test_validate_inference_tokenizer_accepts_current_legacy_pretraining_output(
    tmp_path,
):
    import checkpoint_io
    import infer_sml

    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"legacy tokenizer")
    metadata = checkpoint_io.ParsedCheckpointMetadata(
        raw={"training_config": {"tokenizer_model_path": str(tokenizer_path)}},
        checkpoint_kind="pretraining",
        format_version=None,
        legacy=True,
        step=1,
        model_config=asdict(self.tiny_config()),
        training_config={"tokenizer_model_path": str(tokenizer_path)},
        data_state=None,
        tokenizer=None,
    )

    infer_sml.validate_inference_tokenizer(
        metadata, tokenizer_path, FakeTokenizer(), self.tiny_config()
    )
```

Rename `test_load_checkpoint_metadata_requires_dictionary` to
`test_shared_checkpoint_metadata_requires_dictionary`, import `checkpoint_io`,
and call `checkpoint_io.read_checkpoint_metadata(...,
allow_legacy_pretraining=True)` so it continues to test the parsing owner after
the local inference wrapper is deleted.

The existing generation-only tests should not need real checkpoint/tokenizer
files. In both tests that currently monkeypatch `load_model`, instead monkeypatch
`read_checkpoint_metadata` to return a `SimpleNamespace` whose `model_config` is
`asdict(self.tiny_config())`, monkeypatch `validate_inference_tokenizer` to a
no-op `Spy`, and monkeypatch `_load_model_from_metadata` to return the existing
`FakeGenerationModel`. These tests remain focused on generation length and
decoding; the new tests above own compatibility behavior.

- [ ] **Step 2: Run inference tests and observe RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_infer.py -v
```

Expected: FAIL because `validate_inference_tokenizer` is missing and `load_model` still uses the local metadata parser.

- [ ] **Step 3: Implement shared metadata loading and validation**

In `infer_sml.py`:

1. Remove JSON parsing and import shared checkpoint APIs.
   Extend `InferenceTokenizer` with `get_piece_size`, `bos_id`, `eos_id`, and
   `pad_id` protocol methods so the new compatibility boundary is explicit.
2. Add a private `_load_model_from_metadata(checkpoint_dir, metadata)` that
   constructs `SMLConfig(**metadata.model_config)`, strictly loads the model
   weights, evaluates them, and returns the model. Make public `load_model` read
   metadata with `allow_legacy_pretraining=True` and delegate to that helper.
3. Add:

```python
def validate_inference_tokenizer(
    metadata: ParsedCheckpointMetadata,
    tokenizer_model_path: Path,
    tokenizer: InferenceTokenizer,
    model_config: SMLConfig,
) -> None:
    actual = build_tokenizer_identity(
        tokenizer_model_path,
        tokenizer,
        fallback_bos_token_id=model_config.bos_token_id,
        fallback_eos_token_id=model_config.eos_token_id,
        fallback_pad_token_id=model_config.pad_token_id,
    )
    if metadata.tokenizer is not None:
        validate_tokenizer_identity(metadata.tokenizer, actual)
        return

    expected_structure = {
        "vocab_size": model_config.vocab_size,
        "bos_token_id": model_config.bos_token_id,
        "eos_token_id": model_config.eos_token_id,
        "pad_token_id": model_config.pad_token_id,
    }
    validate_tokenizer_identity(
        expected_structure,
        actual,
        require_fingerprint=False,
    )
    historical_path = metadata.training_config.get("tokenizer_model_path")
    if historical_path is None:
        return
    historical = resolve_path(Path(str(historical_path)))
    if historical.is_file():
        validate_file_identity(
            build_file_identity(historical),
            actual,
            "tokenizer",
        )
```

4. In `generate_text`, read parsed metadata once before the tokenizer, construct
   the `SMLConfig`, validate the tokenizer, then call
   `_load_model_from_metadata`. Keep validation before `encode_prompt`; do not
   call public `load_model` here because that would read metadata twice.
5. Delete the local `load_checkpoint_metadata` implementation. Task 9 changes
   `ft_swag.py` to import `read_checkpoint_metadata` directly, and inference
   tests should import shared parsing from `checkpoint_io` when they need to
   exercise metadata parsing independently.

- [ ] **Step 4: Run inference and checkpoint tests and observe GREEN**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_checkpoint_io.py v2/tests/test_infer.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Commit Task 4**

```bash
git add v2/src/infer_sml.py v2/tests/test_infer.py
git commit -m "fix(v2): validate inference tokenizer identity"
```

---

### Task 5: Strict LoRA State and FP32 Adapters

**Files:**
- Modify: `v2/src/lora.py:93-161,251-293`
- Modify: `v2/tests/test_lora.py:98-315,339-371`

**Interfaces:**
- Consumes: `checkpoint_io.validate_array_state`.
- Produces: `cast_lora_parameters(model, dtype=mx.float32) -> None`; strengthens `load_lora_state_dict` to exact key/shape/dtype matching.

- [ ] **Step 1: Add failing strict-state and precision tests**

Add:

```python
def test_load_lora_state_dict_rejects_unexpected_keys():
    mx = require_mlx_runtime()
    from lora import apply_lora, load_lora_state_dict, lora_state_dict
    from sml import SMLLanguageModel

    config, lora_config = self.tiny_config()
    model = SMLLanguageModel(config)
    apply_lora(model, lora_config)
    state = lora_state_dict(model)
    state["unexpected.lora_A"] = mx.zeros((1, 1))

    with pytest.raises(ValueError, match="unexpected.*unexpected.lora_A"):
        load_lora_state_dict(model, state)


def test_cast_lora_parameters_restores_fp32_after_base_bfloat16_cast():
    mx = require_mlx_runtime()
    from lora import apply_lora, cast_lora_parameters
    from sml import SMLLanguageModel
    from train_sml import apply_model_dtype

    config, lora_config = self.tiny_config()
    model = SMLLanguageModel(config)
    apply_lora(model, lora_config)
    apply_model_dtype(model, "bfloat16")

    cast_lora_parameters(model, mx.float32)

    module = model.layers[0].self_attn.q_proj
    assert module.linear.weight.dtype == mx.bfloat16
    assert module.lora_A.dtype == mx.float32
    assert module.lora_B.dtype == mx.float32


def test_fp32_lora_parameters_create_fp32_adam_moments():
    mx = require_mlx_runtime()
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
    from lora import apply_lora, cast_lora_parameters
    from sml import SMLLanguageModel
    from train_sml import apply_model_dtype

    config, lora_config = self.tiny_config()
    model = SMLLanguageModel(config)
    apply_lora(model, lora_config)
    apply_model_dtype(model, "bfloat16")
    cast_lora_parameters(model, mx.float32)
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.0)
    optimizer.init(model.trainable_parameters())

    moments = [
        value
        for name, value in tree_flatten(optimizer.state)
        if name.endswith((".m", ".v"))
    ]
    assert moments
    assert all(moment.dtype == mx.float32 for moment in moments)
```

- [ ] **Step 2: Run LoRA tests and observe RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_lora.py -v
```

Expected: unexpected keys are silently accepted and `cast_lora_parameters` is missing.

- [ ] **Step 3: Implement exact state validation and adapter casting**

In `lora.py`:

```python
def cast_lora_parameters(model, dtype=mx.float32) -> None:
    for module in iter_lora_modules(model):
        module.lora_A = module.lora_A.astype(dtype)
        module.lora_B = module.lora_B.astype(dtype)
    parameters = lora_parameters(model)
    if parameters:
        mx.eval(*parameters)


def load_lora_state_dict(model, state: dict[str, mx.array]) -> None:
    expected = lora_state_dict(model)
    validate_array_state(expected, state, "LoRA adapter")
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    for name, module in modules.items():
        module.lora_A = state[f"{name}.{LORA_A_SUFFIX}"]
        module.lora_B = state[f"{name}.{LORA_B_SUFFIX}"]
```

Do not cast loaded arrays to fit the destination: a new-format checkpoint with the wrong dtype is incompatible by design.

- [ ] **Step 4: Run LoRA tests and observe GREEN**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_lora.py -v
```

Expected: all LoRA tests PASS.

- [ ] **Step 5: Commit Task 5**

```bash
git add v2/src/lora.py v2/tests/test_lora.py
git commit -m "fix(v2): keep strict fp32 LoRA state"
```

---

### Task 6: Direct Merged LoRA Weight Export

**Files:**
- Modify: `v2/src/lora.py:153-161,296-305`
- Modify: `v2/tests/test_lora.py:316-371`

**Interfaces:**
- Consumes: `LoRALinear`, `mlx.utils.tree_flatten`.
- Produces: `LoRALinear.merged_weight() -> mx.array` and `merged_lora_weights(model) -> dict[str, mx.array]`.

- [ ] **Step 1: Add failing merged-export tests**

Add one structural and one behavioral test:

```python
def test_merged_lora_weights_use_base_names_without_adapter_arrays():
    mx = require_mlx_runtime()
    from lora import apply_lora, merged_lora_weights
    from sml import SMLLanguageModel

    config, lora_config = self.tiny_config()
    model = SMLLanguageModel(config)
    apply_lora(model, lora_config)
    weights = merged_lora_weights(model)

    assert "layers.0.self_attn.q_proj.weight" in weights
    assert "layers.0.self_attn.q_proj.linear.weight" not in weights
    assert not any(name.endswith(("lora_A", "lora_B")) for name in weights)
    assert model.layers[0].self_attn.q_proj.lora_A.dtype == mx.float32


def test_merged_lora_weights_load_strictly_and_preserve_output_without_mutation():
    mx = require_mlx_runtime()
    import mlx.nn as nn
    from lora import LoRALinear, apply_lora, merged_lora_weights
    from sml import SMLLanguageModel

    config, lora_config = self.tiny_config()
    adapted = SMLLanguageModel(config)
    apply_lora(adapted, lora_config)
    adapted.layers[0].self_attn.q_proj.lora_B = mx.full(
        adapted.layers[0].self_attn.q_proj.lora_B.shape, 0.05
    )
    adapted.eval()
    input_ids = mx.array([[1, 4, 5]], dtype=mx.int32)
    expected = adapted(input_ids).logits

    merged = SMLLanguageModel(config)
    merged.load_weights(list(merged_lora_weights(adapted).items()), strict=True)
    merged.eval()
    actual = merged(input_ids).logits
    mx.eval(expected, actual)

    assert bool(mx.allclose(expected, actual, atol=1e-5, rtol=1e-5).item())
    assert isinstance(adapted.layers[0].self_attn.q_proj, LoRALinear)
    assert isinstance(merged.layers[0].self_attn.q_proj, nn.Linear)
```

- [ ] **Step 2: Run merged-export tests and observe RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_lora.py -v
```

Expected: FAIL because `merged_lora_weights` and `merged_weight` are missing.

- [ ] **Step 3: Implement non-mutating merged export**

In `lora.py`, import `tree_flatten`, refactor `merge`, and add the exporter:

```python
def merged_weight(self) -> mx.array:
    delta = self.scaling * (self.lora_B @ self.lora_A)
    if delta.dtype != self.linear.weight.dtype:
        delta = delta.astype(self.linear.weight.dtype)
    return self.linear.weight + delta


def merge(self):
    self.linear.weight = self.merged_weight()
    return self.linear


def merged_lora_weights(model) -> dict[str, mx.array]:
    flat_parameters = dict(tree_flatten(model.parameters()))
    lora_modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LoRALinear)
    }
    merged: dict[str, mx.array] = {}
    for parameter_name, parameter in flat_parameters.items():
        if parameter_name.endswith((f".{LORA_A_SUFFIX}", f".{LORA_B_SUFFIX}")):
            continue
        if ".linear." not in parameter_name:
            merged[parameter_name] = parameter
            continue
        module_name, child_parameter = parameter_name.rsplit(".linear.", 1)
        lora_module = lora_modules.get(module_name)
        if lora_module is None:
            merged[parameter_name] = parameter
            continue
        inference_name = f"{module_name}.{child_parameter}"
        merged[inference_name] = (
            lora_module.merged_weight()
            if child_parameter == "weight"
            else parameter
        )
    return merged
```

- [ ] **Step 4: Run LoRA tests and observe GREEN**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_lora.py -v
```

Expected: all LoRA tests PASS and the adapted source model remains wrapped.

- [ ] **Step 5: Commit Task 6**

```bash
git add v2/src/lora.py v2/tests/test_lora.py
git commit -m "perf(v2): export merged LoRA weights directly"
```

---

### Task 7: Dynamic SWAG Candidate Padding

**Files:**
- Modify: `v2/src/ft_swag.py:306-454`
- Modify: `v2/tests/test_ft_swag.py:266-420`

**Interfaces:**
- Consumes: the existing `SwagExampleDataset` and `iter_swag_batches` pipeline.
- Produces: variable-length examples collated to `(batch, candidates, batch_max_sequence)`.

- [ ] **Step 1: Replace fixed-padding expectations with failing dynamic tests**

Change the existing short-example test so the two rows have different candidate lengths and assert only batch-local padding:

```python
def test_build_swag_batches_pads_only_to_longest_candidate(monkeypatch):
    require_mlx_runtime()
    import ft_swag
    from ft_swag import SwagFineTuneConfig

    monkeypatch.setattr(
        ft_swag,
        "iter_swag_examples",
        Spy(
            return_value=iter(
                [
                    ("", ("4", "5 6", "7", "8"), 0),
                    ("", ("9", "10", "11 12 13", "14"), 2),
                ]
            )
        ),
    )
    batch = next(
        ft_swag.build_swag_batches(
            SwagFineTuneConfig(sequence_length=32, batch_size=2),
            FakeTokenizer(),
            epoch=0,
        )
    )

    assert batch["input_ids"].shape == (2, 4, 4)
    assert batch["labels"].shape == (2, 4, 4)
    assert batch["input_ids"][0, 0].tolist() == [1, 4, 3, 3]
    assert batch["labels"][0, 0].tolist() == [4, 2, 3, 3]
    assert batch["input_ids"][1, 2].tolist() == [1, 11, 12, 13]
    assert batch["labels"][1, 2].tolist() == [11, 12, 13, 2]
```

Add a boundary test proving an example with
`len(tokens) - 1 == sequence_length` fits and one with a larger shifted input is
skipped:

```python
@pytest.mark.parametrize(
    ("longest_ending", "expected_batch_count"),
    [("4 5 6", 1), ("4 5 6 7", 0)],
)
def test_swag_sequence_length_is_shifted_input_limit(
    monkeypatch, longest_ending, expected_batch_count
):
    require_mlx_runtime()
    import ft_swag
    from ft_swag import SwagFineTuneConfig

    monkeypatch.setattr(
        ft_swag,
        "iter_swag_examples",
        Spy(
            return_value=iter(
                [("", (longest_ending, "8", "9", "10"), 0)]
            )
        ),
    )

    batches = list(
        ft_swag.build_swag_batches(
            SwagFineTuneConfig(sequence_length=4, batch_size=1),
            FakeTokenizer(),
            epoch=0,
        )
    )

    assert len(batches) == expected_batch_count
```

- [ ] **Step 2: Run SWAG batching tests and observe RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_ft_swag.py -k "batch or sequence_length" -v
```

Expected: fixed padding produces the configured length instead of batch-local length.

- [ ] **Step 3: Emit unpadded examples and pad in the collator**

Replace the fixed `tokens_per_candidate` construction in `SwagExampleDataset.__iter__`:

```python
ending_start = len(bos_tokens) + len(context_ids)
example_tokens = [*bos_tokens, *context_ids, *ending_ids]
input_ids = example_tokens[:-1]
labels = list(example_tokens[1:])
if len(input_ids) > self.sequence_length:
    should_skip = True
    break
ending_label_start = max(ending_start - 1, 0)
labels[:ending_label_start] = [self.pad_token_id] * ending_label_start
candidate_input_ids.append([int(token_id) for token_id in input_ids])
candidate_labels.append([int(token_id) for token_id in labels])
```

Build `bos_tokens` once as either `[]` or `[bos_token_id]`. Then replace `collate_swag_batch` with:

```python
def collate_swag_batch(
    batch: list[dict[str, object]],
    pad_token_id: int,
) -> dict[str, object]:
    max_length = max(
        len(candidate)
        for example in batch
        for candidate in example["input_ids"]
    )

    def pad(values: list[int]) -> list[int]:
        return values + [pad_token_id] * (max_length - len(values))

    return {
        "input_ids": mx.array(
            [[pad(candidate) for candidate in example["input_ids"]] for example in batch],
            dtype=mx.int32,
        ),
        "labels": mx.array(
            [[pad(candidate) for candidate in example["labels"]] for example in batch],
            dtype=mx.int32,
        ),
        "candidate_labels": mx.array(
            [example["candidate_labels"] for example in batch], dtype=mx.int32
        ),
    }
```

Pass `pad_token_id` through `iter_swag_batches` and from `build_swag_batches`.

- [ ] **Step 4: Run SWAG batching tests and observe GREEN**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_ft_swag.py -k "batch or sequence_length" -v
```

Expected: batching tests PASS with batch-local sequence dimensions.

- [ ] **Step 5: Commit Task 7**

```bash
git add v2/src/ft_swag.py v2/tests/test_ft_swag.py
git commit -m "perf(v2): dynamically pad SWAG candidates"
```

---

### Task 8: FP32 Length-Normalized SWAG Ranking

**Files:**
- Modify: `v2/src/ft_swag.py:457-504`
- Modify: `v2/tests/test_ft_swag.py:422-457`

**Interfaces:**
- Consumes: dynamically padded labels from Task 7.
- Produces: `score_swag_candidates(...) -> mx.array` with FP32 mean continuation log-likelihood scores.

- [ ] **Step 1: Add failing dtype, normalization, and empty-score tests**

Replace the summed-score test with a model that gives every valid target the same log-probability and candidates of different lengths:

```python
def test_swag_ranking_scores_are_fp32_length_normalized():
    mx = require_mlx_runtime()
    import ft_swag

    class UniformModel:
        def __call__(self, input_ids):
            logits = mx.zeros((*input_ids.shape, 16), dtype=mx.bfloat16)
            return type("Output", (), {"logits": logits})()

    input_ids = mx.array([[[1, 4, 3], [1, 5, 6]]], dtype=mx.int32)
    labels = mx.array([[[4, 3, 3], [5, 6, 3]]], dtype=mx.int32)

    scores = ft_swag.score_swag_candidates(UniformModel(), input_ids, labels, 3)
    mx.eval(scores)

    assert scores.dtype == mx.float32
    assert float(scores[0, 0].item()) == pytest.approx(float(scores[0, 1].item()))


def test_swag_ranking_rejects_candidate_without_scored_tokens():
    mx = require_mlx_runtime()
    import ft_swag

    class Model:
        def __call__(self, input_ids):
            logits = mx.zeros((*input_ids.shape, 16))
            return type("Output", (), {"logits": logits})()

    input_ids = mx.array([[[1, 3]]], dtype=mx.int32)
    labels = mx.array([[[3, 3]]], dtype=mx.int32)

    with pytest.raises(ValueError, match="at least one scored token"):
        ft_swag.score_swag_candidates(Model(), input_ids, labels, 3)
```

- [ ] **Step 2: Run ranking tests and observe RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_ft_swag.py -k "ranking" -v
```

Expected: scores remain bfloat16/raw sums and the empty candidate is accepted.

- [ ] **Step 3: Implement FP32 mean scoring**

Change `score_swag_candidates` after the model call:

```python
logits = output.logits.astype(mx.float32)
log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
token_log_probs = mx.squeeze(
    mx.take_along_axis(
        log_probs,
        mx.expand_dims(flat_labels, axis=-1),
        axis=-1,
    ),
    axis=-1,
)
label_mask = flat_labels != pad_token_id
valid_counts = mx.sum(label_mask, axis=-1)
mx.eval(valid_counts)
if bool(mx.any(valid_counts == 0).item()):
    raise ValueError("Each SWAG candidate must contain at least one scored token")
score_sums = mx.sum(mx.where(label_mask, token_log_probs, 0.0), axis=-1)
scores = score_sums / valid_counts.astype(mx.float32)
return scores.reshape((batch_size, candidate_count))
```

- [ ] **Step 4: Run ranking tests and observe GREEN**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_ft_swag.py -k "ranking" -v
```

Expected: all ranking tests PASS with FP32 mean scores.

- [ ] **Step 5: Commit Task 8**

```bash
git add v2/src/ft_swag.py v2/tests/test_ft_swag.py
git commit -m "fix(v2): normalize SWAG ranking in fp32"
```

---

### Task 9: Versioned SWAG Checkpoints and Strict Resume

**Files:**
- Modify: `v2/src/ft_swag.py:11-24,40-48,507-710,745-837`
- Modify: `v2/tests/test_ft_swag.py:592-736`

**Interfaces:**
- Consumes: shared checkpoint APIs from Tasks 1–2; `cast_lora_parameters` and `merged_lora_weights` from Tasks 5–6.
- Produces:
  - `save_lora_checkpoint(..., tokenizer_identity, base_checkpoint_identity, ...) -> None`
  - `load_lora_checkpoint(..., expected_model_config, expected_lora_config, tokenizer_identity, base_checkpoint_identity) -> TrainingResumeState`
  - strict new-format SWAG resume with no legacy fallback.

- [ ] **Step 1: Rewrite the SWAG checkpoint round-trip test for the new contract**

In the existing round-trip test:

1. Write real temporary base weights and tokenizer bytes.
2. Build identities using `checkpoint_io`.
3. Cast source/target LoRA parameters to FP32 before optimizer initialization.
4. Pass both identities to save/load.
5. Assert metadata format/kind and that the merged weights strictly load into a plain `SMLLanguageModel`.

Use these key assertions:

```python
assert metadata["format"] == "sml-checkpoint"
assert metadata["format_version"] == 1
assert metadata["checkpoint_kind"] == "swag_lora"
assert metadata["lora_config"] == json_ready(asdict(fine_tune_config.lora))
assert metadata["base_checkpoint"] == base_checkpoint_identity

merged = SMLLanguageModel(model_config)
merged.load_weights(str(checkpoint_path / ft_swag.MODEL_WEIGHTS_NAME), strict=True)
```

After the successful round trip, corrupt the serialized optimizer state and
prove that resume rejects it before assigning a different adapter state:

```python
adapter_arrays = checkpoint_io.load_array_state(
    checkpoint_path / ft_swag.LORA_STATE_NAME,
    "LoRA adapter",
)
first_adapter_name = sorted(adapter_arrays)[0]
adapter_arrays[first_adapter_name] = mx.full(
    adapter_arrays[first_adapter_name].shape,
    0.75,
    dtype=adapter_arrays[first_adapter_name].dtype,
)
checkpoint_io.save_array_state(
    checkpoint_path / ft_swag.LORA_STATE_NAME,
    adapter_arrays,
)
optimizer_arrays = checkpoint_io.load_array_state(
    checkpoint_path / ft_swag.OPTIMIZER_STATE_NAME,
    "optimizer",
)
optimizer_arrays["unexpected"] = mx.zeros((1,), dtype=mx.float32)
checkpoint_io.save_array_state(
    checkpoint_path / ft_swag.OPTIMIZER_STATE_NAME,
    optimizer_arrays,
)
adapter_before_failed_resume = (
    target.layers[0].self_attn.q_proj.lora_A.tolist()
)

with pytest.raises(ValueError, match="optimizer state keys mismatch"):
    ft_swag.load_lora_checkpoint(
        checkpoint_path,
        target,
        target_optimizer,
        expected_model_config=model_config,
        expected_lora_config=fine_tune_config.lora,
        tokenizer_identity=tokenizer_identity,
        base_checkpoint_identity=base_checkpoint_identity,
    )

assert target.layers[0].self_attn.q_proj.lora_A.tolist() == (
    adapter_before_failed_resume
)
```

Add a production boundary named `validate_swag_resume_metadata` and test it by
constructing parsed metadata directly. This keeps compatibility policy
independent of filesystem and optimizer mechanics:

```python
@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("tokenizer", "tokenizer fingerprint mismatch"),
        ("base", "base checkpoint fingerprint mismatch"),
        ("lora", "LoRA configuration mismatch"),
        ("model", "model configuration mismatch"),
    ],
)
def test_lora_resume_rejects_incompatible_checkpoint(
    mismatch, message
):
    from dataclasses import asdict, replace

    import checkpoint_io
    import ft_swag
    from sml import SMLConfig

    model_config = SMLConfig(
        vocab_size=16,
        hidden_size=8,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        original_max_position_embeddings=16,
        rope_scaling_factor=1.0,
        hidden_dropout=0.0,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
    )
    lora_config = ft_swag.LoRAConfig(
        rank=2,
        alpha=4.0,
        dropout=0.0,
        target_modules=("q_proj",),
    )
    tokenizer_identity = {
        "path": "/tmp/tokenizer.model",
        "sha256": "1" * 64,
        "size": 10,
        "vocab_size": 16,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 3,
    }
    base_checkpoint_identity = {
        "path": "/tmp/model.safetensors",
        "sha256": "2" * 64,
        "size": 20,
        "model_config": ft_swag.json_ready(asdict(model_config)),
    }
    raw = {
        "model_config": ft_swag.json_ready(asdict(model_config)),
        "lora_config": ft_swag.json_ready(asdict(lora_config)),
        "base_checkpoint": dict(base_checkpoint_identity),
    }
    parsed = checkpoint_io.ParsedCheckpointMetadata(
        raw=raw,
        checkpoint_kind=checkpoint_io.SWAG_LORA_CHECKPOINT_KIND,
        format_version=1,
        legacy=False,
        step=3,
        model_config=raw["model_config"],
        training_config={},
        data_state=None,
        tokenizer=dict(tokenizer_identity),
    )
    expected_model_config = model_config
    expected_lora_config = lora_config
    if mismatch == "tokenizer":
        tokenizer_identity = {**tokenizer_identity, "sha256": "0" * 64}
    elif mismatch == "base":
        base_checkpoint_identity = {
            **base_checkpoint_identity, "sha256": "0" * 64
        }
    elif mismatch == "lora":
        expected_lora_config = replace(lora_config, alpha=99.0)
    else:
        expected_model_config = replace(model_config, rope_theta=20_000.0)

    with pytest.raises(ValueError, match=message):
        ft_swag.validate_swag_resume_metadata(
            parsed,
            expected_model_config=expected_model_config,
            expected_lora_config=expected_lora_config,
            tokenizer_identity=tokenizer_identity,
            base_checkpoint_identity=base_checkpoint_identity,
        )
```

- [ ] **Step 2: Run SWAG checkpoint tests and observe RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_ft_swag.py -k "checkpoint or resume" -v
```

Expected: save/load signatures and metadata do not satisfy the new contract.

- [ ] **Step 3: Replace copied-model checkpointing with shared serialization**

In `ft_swag.py`:

1. Remove `copy` and `build_merged_model`.
2. Import shared checkpoint APIs and new LoRA helpers.
3. Make `save_lora_checkpoint` require `tokenizer_identity` and `base_checkpoint_identity` keyword arguments.
4. Serialize:

```python
save_safetensors(path / MODEL_WEIGHTS_NAME, merged_lora_weights(model))
save_array_state(path / LORA_STATE_NAME, lora_state_dict(model))
save_array_state(
    path / OPTIMIZER_STATE_NAME,
    tree_flatten(optimizer.state, destination={}),
)
metadata = build_checkpoint_metadata(
    checkpoint_kind=SWAG_LORA_CHECKPOINT_KIND,
    step=step,
    model_config=json_ready(asdict(model_config)),
    training_config=json_ready(asdict(fine_tune_config)),
    data_state=None if data_state is None else json_ready(asdict(data_state)),
    tokenizer=tokenizer_identity,
    extra={
        "lora_config": json_ready(asdict(fine_tune_config.lora)),
        "base_checkpoint": dict(base_checkpoint_identity),
        "stochastic_resume": "not_guaranteed",
        "resume_note": STOCHASTIC_RESUME_NOTE,
    },
)
write_checkpoint_metadata(path, metadata)
```

5. Make `load_lora_checkpoint` read only `SWAG_LORA_CHECKPOINT_KIND`, compare exact JSON-ready model/LoRA configurations, validate tokenizer and base identities, validate adapter and optimizer states, then load them.

```python
def validate_swag_resume_metadata(
    parsed: ParsedCheckpointMetadata,
    *,
    expected_model_config: SMLConfig,
    expected_lora_config: LoRAConfig,
    tokenizer_identity: Mapping[str, object],
    base_checkpoint_identity: Mapping[str, object],
) -> None:
    if parsed.model_config != json_ready(asdict(expected_model_config)):
        raise ValueError("SWAG model configuration mismatch")
    if parsed.raw["lora_config"] != json_ready(asdict(expected_lora_config)):
        raise ValueError("LoRA configuration mismatch")
    if parsed.tokenizer is None:
        raise ValueError("SWAG checkpoint is missing tokenizer identity")
    validate_tokenizer_identity(parsed.tokenizer, tokenizer_identity)
    validate_file_identity(
        parsed.raw["base_checkpoint"],
        base_checkpoint_identity,
        "base checkpoint",
    )
    if parsed.raw["base_checkpoint"].get("model_config") != (
        base_checkpoint_identity.get("model_config")
    ):
        raise ValueError("base checkpoint model configuration mismatch")
```

Call `validate_swag_resume_metadata` from `load_lora_checkpoint` immediately
after parsing and before loading either NPZ file; the helper is the single
compatibility-policy owner.

6. Load both NPZ dictionaries without assigning them. Validate the adapter
   arrays against `lora_state_dict(model)` and the optimizer arrays against the
   freshly initialized flattened optimizer state. Only after both validations
   pass, call strict `load_lora_state_dict`, assign
   `optimizer.state = tree_unflatten(optimizer_arrays)`, and evaluate. This
   ordering ensures a bad optimizer file cannot partially mutate the model.

- [ ] **Step 4: Integrate identities and FP32 adapters into `fine_tune_swag`**

After loading the tokenizer and base configuration:

```python
tokenizer_identity = build_tokenizer_identity(
    fine_tune_config.tokenizer_model_path,
    tokenizer,
    fallback_bos_token_id=checkpoint_model_config.bos_token_id,
    fallback_eos_token_id=checkpoint_model_config.eos_token_id,
    fallback_pad_token_id=checkpoint_model_config.pad_token_id,
)
base_checkpoint_identity = build_model_checkpoint_identity(
    pretrained_path,
    json_ready(asdict(base_model_config)),
)
```

Change `load_pretrained_model_config` to call
`read_checkpoint_metadata(checkpoint_path,
expected_kind=PRETRAINING_CHECKPOINT_KIND,
allow_legacy_pretraining=True)` and construct `SMLConfig` from
`parsed.model_config`. Always obtain `base_model_config` from the selected
pretraining checkpoint, not from SWAG output metadata. After `apply_model_dtype`, call
`cast_lora_parameters(model, mx.float32)` before `optimizer.init`. Delete the
current post-resume `apply_model_dtype(...)` call: the frozen base parameters
were already cast before resume, and strict adapter loading restores FP32 arrays.
No model-wide cast may run after adapter loading.

Pass the identities and expected configurations into `load_lora_checkpoint`, and pass the same identities to every `save_lora_checkpoint` call.

- [ ] **Step 5: Run SWAG checkpoint and LoRA tests and observe GREEN**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_checkpoint_io.py v2/tests/test_lora.py v2/tests/test_ft_swag.py -v
```

Expected: all tests PASS, including incompatible-resume rejection and strict merged-weight loading.

- [ ] **Step 6: Commit Task 9**

```bash
git add v2/src/ft_swag.py v2/tests/test_ft_swag.py
git commit -m "feat(v2): enforce strict SWAG checkpoint resume"
```

---

### Task 10: Stop Before Updating an Already-Complete Resume

**Files:**
- Modify: `v2/src/ft_swag.py:687-710`
- Modify: `v2/tests/test_ft_swag.py`

**Interfaces:**
- Consumes: existing `is_step_limit_reached(global_step, max_steps)`.
- Produces: immediate no-op return from `fine_tune_swag` when restored progress already satisfies `max_steps`.

- [ ] **Step 1: Add a failing integration regression**

Build a complete tiny pretraining checkpoint and a versioned SWAG checkpoint at
step 1 with `max_steps=1`. Patch `build_swag_batches` and `nn.value_and_grad` to
raise if reached, then resume:

```python
def test_fine_tune_swag_does_not_update_resume_already_at_step_limit(
    tmp_path, monkeypatch
):
    mx = require_mlx_runtime()
    import json
    from dataclasses import asdict, replace

    import mlx.optimizers as optim

    import checkpoint_io
    import ft_swag
    from lora import cast_lora_parameters
    from sml import SMLConfig, SMLLanguageModel
    from test_train import tiny_config

    class RunnableTokenizer:
        bos_id = 1
        eos_id = 2
        pad_id = 3

        def get_piece_size(self):
            return 16

        def encode(self, text, out_type=int):
            del out_type
            return [int(part) for part in text.split()]

    tokenizer = RunnableTokenizer()
    tokenizer_path = tmp_path / "tokenizer.model"
    tokenizer_path.write_bytes(b"tiny tokenizer")
    tokenizer_identity = checkpoint_io.build_tokenizer_identity(
        tokenizer_path,
        tokenizer,
        fallback_bos_token_id=1,
        fallback_eos_token_id=2,
        fallback_pad_token_id=3,
    )

    base_checkpoint = tmp_path / "base"
    base_checkpoint.mkdir()
    base_model_config = tiny_config()
    base_model = SMLLanguageModel(base_model_config)
    base_model.save_weights(str(base_checkpoint / ft_swag.MODEL_WEIGHTS_NAME))
    checkpoint_io.write_checkpoint_metadata(
        base_checkpoint,
        checkpoint_io.build_checkpoint_metadata(
            checkpoint_kind=checkpoint_io.PRETRAINING_CHECKPOINT_KIND,
            step=0,
            model_config=ft_swag.json_ready(asdict(base_model_config)),
            training_config={},
            data_state=None,
            tokenizer=tokenizer_identity,
            extra={"input_files": []},
        ),
    )

    fine_tune_config = ft_swag.SwagFineTuneConfig(
        pretrained_checkpoint_path=base_checkpoint,
        output_dir=tmp_path,
        checkpoint_name="sml-swag",
        tokenizer_model_path=tokenizer_path,
        sequence_length=4,
        batch_size=1,
        max_steps=1,
        lr_total_steps=1,
        epochs=1,
        gradient_accumulation_steps=1,
        log_every=1,
        save_every=0,
        lora=ft_swag.LoRAConfig(
            rank=2,
            alpha=4.0,
            dropout=0.0,
            target_modules=("q_proj",),
        ),
    )
    checkpoint_model_config = replace(
        base_model_config,
        vocab_size=tokenizer.get_piece_size(),
        rope_scaling_factor=SMLConfig().rope_scaling_factor,
        original_max_position_embeddings=max(
            base_model_config.original_max_position_embeddings,
            fine_tune_config.sequence_length,
        ),
    )
    adapted = SMLLanguageModel(checkpoint_model_config)
    adapted.load_weights(str(base_checkpoint / ft_swag.MODEL_WEIGHTS_NAME))
    ft_swag.prepare_lora_model(adapted, fine_tune_config)
    ft_swag.apply_model_dtype(adapted, fine_tune_config.autocast_dtype)
    cast_lora_parameters(adapted, mx.float32)
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.0)
    optimizer.init(adapted.trainable_parameters())
    base_checkpoint_identity = checkpoint_io.build_model_checkpoint_identity(
        base_checkpoint,
        ft_swag.json_ready(asdict(base_model_config)),
    )
    checkpoint_path = tmp_path / fine_tune_config.checkpoint_name
    ft_swag.save_lora_checkpoint(
        checkpoint_path,
        adapted,
        optimizer,
        checkpoint_model_config,
        fine_tune_config,
        step=1,
        tokenizer_identity=tokenizer_identity,
        base_checkpoint_identity=base_checkpoint_identity,
    )

    monkeypatch.setattr(ft_swag, "load_tokenizer", Spy(return_value=tokenizer))
    monkeypatch.setattr(
        ft_swag,
        "build_swag_batches",
        Spy(side_effect=AssertionError("dataset must not be read")),
    )
    monkeypatch.setattr(
        ft_swag.nn,
        "value_and_grad",
        Spy(side_effect=AssertionError("gradients must not be constructed")),
    )

    result = ft_swag.fine_tune_swag(
        fine_tune_config,
        resume_from_checkpoint=True,
    )

    assert result == checkpoint_path
    metadata = json.loads((checkpoint_path / ft_swag.METADATA_NAME).read_text())
    assert metadata["step"] == 1
```

- [ ] **Step 2: Run the regression and observe RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_ft_swag.py::TestFtSwag::test_fine_tune_swag_does_not_update_resume_already_at_step_limit -v
```

Expected: FAIL because `nn.value_and_grad` or the data path is reached.

- [ ] **Step 3: Add the pre-loop step-limit guard**

Immediately after `global_step = resume_state.step` and before `model.train()`, dataset construction, or `nn.value_and_grad`:

```python
if is_step_limit_reached(global_step, fine_tune_config.max_steps):
    return checkpoint_path
```

Do not rewrite the checkpoint during this no-op resume; the validated existing checkpoint already represents the requested terminal state.

- [ ] **Step 4: Run the regression and full SWAG tests and observe GREEN**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_ft_swag.py -v
```

Expected: all SWAG tests PASS and the stored step remains unchanged.

- [ ] **Step 5: Commit Task 10**

```bash
git add v2/src/ft_swag.py v2/tests/test_ft_swag.py
git commit -m "fix(v2): stop completed SWAG resumes before update"
```

---

### Task 11: Documentation and Full Verification

**Files:**
- Modify: `v2/README.md`

**Interfaces:**
- Consumes: all production APIs delivered by Tasks 1–10.
- Produces: documented compatibility behavior and a fully verified v2 tree.

- [ ] **Step 1: Document the checkpoint contract and resume behavior**

Add a concise `Checkpoint compatibility` subsection to `v2/README.md` stating:

```markdown
### Checkpoint compatibility

- Current pretraining checkpoints and SentencePiece tokenizer files remain supported.
- New checkpoints record a versioned format and tokenizer content fingerprint.
- SWAG resume requires the exact base checkpoint, tokenizer, model configuration,
  and LoRA configuration used to create the checkpoint.
- SWAG checkpoints created before the versioned format are intentionally not resumable.
```

Also document that SWAG candidate batches use dynamic padding and mean continuation log-likelihood.

- [ ] **Step 2: Run targeted source/layout checks**

Run outside the sandbox:

```bash
uv run pytest v2/tests/test_module_layout.py v2/tests/test_mlx_only_source.py -v
```

Expected: all tests PASS; no forbidden tensor framework appears in v2 source, tests, or README.

- [ ] **Step 3: Run Ruff on all v2 files**

```bash
uv run ruff check v2
uv run ruff format --check v2
```

Expected: both commands exit 0 with no lint or formatting changes required.

- [ ] **Step 4: Run the complete v2 test suite with Metal**

Run outside the sandbox:

```bash
uv run pytest v2/tests
```

Expected: all tests PASS with zero failures. Existing third-party SentencePiece deprecation warnings may remain; no new project warning is acceptable.

- [ ] **Step 5: Review the final diff against all seven findings**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Confirm explicitly:

1. SWAG resume validates base, tokenizer, model, LoRA, adapter, and optimizer identity.
2. A resume at `max_steps` performs no update.
3. SWAG batches pad only to the current batch maximum.
4. LoRA parameters, Adam moments, and ranking reductions are FP32.
5. Candidate scores are mean continuation log-likelihoods.
6. SWAG saves no longer call `copy.deepcopy` or construct a second model.
7. Inference validates tokenizer structure and, for new checkpoints, content identity.

- [ ] **Step 6: Commit Task 11**

```bash
git add v2/README.md
git commit -m "docs(v2): document checkpoint compatibility"
```

---

## Execution Notes

- Follow tasks in order; later tests and signatures depend on earlier shared interfaces.
- Keep each task's red test failure visible in the execution log before editing production code.
- MLX promotes the mixed bfloat16-activation/FP32-adapter matrix products to
  FP32; retain the existing cast of the adapter result back to the base output
  dtype before addition. Do not cast adapter parameters back to bfloat16.
- If `mx.save_safetensors` exposes a strict key restriction not covered by the current MLX documentation, reproduce it with the smallest flat state in `test_checkpoint_io.py` before adapting serialization.
- Do not combine task commits until every task's focused tests are green.
