from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import pytest
from sml.artifacts import checkpoint as checkpoint_module
from sml.artifacts.checkpoint import (
    VerifiedCheckpointContents,
    open_checkpoint_reader,
    resolve_exact_step,
)
from sml.artifacts.manifest import (
    ArrayPayloadRef,
    ArraySpec,
    ArtifactRoot,
    PayloadRef,
    PretrainingCheckpointManifest,
    PretrainingRunManifest,
    VerificationLevel,
    canonical_json_bytes,
    file_identity,
)
from sml.artifacts.verify import verify_artifact
from sml.errors import SMLArtifactError

_PLACEHOLDER_IDENTITY = "sha256:" + "0" * 64


def _payload_ref(path: Path, logical_path: str) -> PayloadRef:
    with path.open("rb") as payload:
        identity = file_identity(payload)
    return PayloadRef(logical_path, identity, path.stat().st_size)


def _array_ref(
    path: Path,
    logical_path: str,
    arrays: dict[str, mx.array],
    *,
    declared_shape: tuple[int, ...] | None = None,
) -> ArrayPayloadRef:
    return ArrayPayloadRef(
        _payload_ref(path, logical_path),
        tuple(
            ArraySpec(
                name,
                declared_shape if declared_shape is not None else tuple(array.shape),
                {
                    mx.bfloat16: "bfloat16",
                    mx.float32: "float32",
                    mx.int32: "int32",
                    mx.uint32: "uint32",
                }[array.dtype],
            )
            for name, array in sorted(arrays.items())
        ),
    )


def _write_valid_checkpoint_run(tmp_path: Path) -> Path:
    run = tmp_path / "valid-run"
    step_directory = run / "checkpoints" / "step-000000000"
    step_directory.mkdir(parents=True)
    run_manifest = PretrainingRunManifest(
        kind="pretraining-run",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model={"rope_scaling_factor": 1.0},
        precision={"working_parameter_dtype": "bfloat16"},
        optimizer={"kind": "adamw"},
        loader={"microbatch_size": 1},
        checkpoint={"interval": 1},
        tokenizer_identity="sha256:" + "1" * 64,
        data_identity="sha256:" + "2" * 64,
        diagnostic_data_locator=None,
    )
    run_manifest = replace(run_manifest, identity=run_manifest.recompute_identity())
    (run / "run.json").write_bytes(canonical_json_bytes(run_manifest))

    groups = {
        "model.safetensors": {"weight": mx.array([1.0], dtype=mx.bfloat16)},
        "master.safetensors": {"weight": mx.array([1.0], dtype=mx.float32)},
        "optimizer.safetensors": {
            "step": mx.array(0, dtype=mx.int32),
            "first_moments.weight": mx.array([0.0], dtype=mx.float32),
            "second_moments.weight": mx.array([0.0], dtype=mx.float32),
        },
        "trainer.safetensors": {
            "accumulation_count": mx.array(0, dtype=mx.int32),
            "next_key": mx.random.key(7),
            "loss_numerator": mx.array(0.0, dtype=mx.float32),
            "accumulators.weight": mx.array([0.0], dtype=mx.float32),
        },
    }
    references = {}
    for logical_path, arrays in groups.items():
        path = step_directory / logical_path
        mx.save_safetensors(path, arrays)
        references[logical_path] = _array_ref(path, logical_path, arrays)

    state_path = step_directory / "state.json"
    state_path.write_bytes(
        canonical_json_bytes(
            {
                "kind": "pretraining-state",
                "version": 1,
                "owning_run_identity": run_manifest.identity,
                "step": 0,
                "rows": 0,
                "microsteps": 0,
                "cursor": {
                    "epoch": 0,
                    "shard_order_position": 0,
                    "row_offset": 0,
                },
            }
        )
    )
    manifest = PretrainingCheckpointManifest(
        kind="pretraining-checkpoint",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        owning_run_identity=run_manifest.identity,
        step=0,
        scalar_state=_payload_ref(state_path, "state.json"),
        model=references["model.safetensors"],
        master=references["master.safetensors"],
        optimizer=references["optimizer.safetensors"],
        trainer=references["trainer.safetensors"],
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (step_directory / "checkpoint.json").write_bytes(canonical_json_bytes(manifest))
    return run


def test_checkpoint_array_validation_uses_one_open_payload_through_posthash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Descriptor reuse must not hide metadata access after payload close."""
    payload_path = tmp_path / "arrays.safetensors"
    payload_path.write_bytes(b"descriptor-owned checkpoint payload")
    reference = ArrayPayloadRef(
        payload=_payload_ref(payload_path, "arrays.safetensors"),
        arrays=(
            ArraySpec("z", (1,), "float32"),
            ArraySpec("a", (1,), "float32"),
        ),
    )
    real_file_identity = checkpoint_module.file_identity
    payload_stream = None
    events: list[tuple[str, int, tuple[int, int] | None, bool, tuple[str, ...]]] = []

    def record(phase: str, stream, names: tuple[str, ...] = ()) -> None:
        is_open = not stream.closed
        inode = None
        if is_open:
            opened = os.fstat(stream.fileno())
            inode = (opened.st_dev, opened.st_ino)
        events.append((phase, id(stream), inode, is_open, names))

    def recording_file_identity(stream):
        record("hash", stream)
        return real_file_identity(stream)

    class MetadataSpy:
        def __init__(self, name: str) -> None:
            self.name = name

        @property
        def shape(self):
            assert payload_stream is not None
            record("shape", payload_stream, (self.name,))
            return (1,)

        @property
        def dtype(self):
            assert payload_stream is not None
            record("dtype", payload_stream, (self.name,))
            return mx.float32

    class RecordingMlx:
        def __getattr__(self, name: str):
            return getattr(mx, name)

        def load(self, stream, *, format):
            nonlocal payload_stream
            assert format == "safetensors"
            payload_stream = stream
            record("load", stream)
            return {"z": MetadataSpy("z"), "a": MetadataSpy("a")}

        def eval(self, *values):
            assert payload_stream is not None
            record("eval", payload_stream, tuple(value.name for value in values))

    monkeypatch.setattr(checkpoint_module, "file_identity", recording_file_identity)
    monkeypatch.setattr(checkpoint_module, "_mlx_core", lambda: RecordingMlx())

    with ArtifactRoot.open(tmp_path, writable=False) as root:
        loaded = checkpoint_module._load_checkpoint_array_payload(
            root,
            reference,
            full=True,
        )

    assert list(loaded) == ["a", "z"]
    assert [event[0] for event in events] == [
        "hash",
        "load",
        "shape",
        "dtype",
        "shape",
        "dtype",
        "eval",
        "hash",
    ]
    assert [event[4] for event in events if event[0] in {"shape", "dtype"}] == [
        ("a",),
        ("a",),
        ("z",),
        ("z",),
    ]
    assert events[-2][4] == ("a", "z")
    assert len({event[1] for event in events}) == 1
    assert len({event[2] for event in events}) == 1
    assert events[0][2] is not None
    assert all(event[3] for event in events)
    assert payload_stream is not None and payload_stream.closed


def test_checkpoint_initial_step_name_swap_fails_opened_entry_revalidation(
    tmp_path: Path,
) -> None:
    """The first name-to-opened-inode check must reject a just-opened step swap."""
    run = _write_valid_checkpoint_run(tmp_path)
    checkpoints = run / "checkpoints"
    step_name = "step-000000000"
    step = checkpoints / step_name
    moved_step = checkpoints / "retained-before-revalidation"

    class SwapBeforeNamedStat(checkpoint_module._OSFilesystemOps):
        def __init__(self) -> None:
            self.opened_descriptor = -1
            self.swapped = False

        def open(self, path, flags, mode=0o777, *, dir_fd=None):
            descriptor = super().open(path, flags, mode, dir_fd=dir_fd)
            if path == step_name:
                self.opened_descriptor = descriptor
            return descriptor

        def stat(self, path, *, dir_fd=None, follow_symlinks=False):
            if path == step_name and self.opened_descriptor >= 0 and not self.swapped:
                step.rename(moved_step)
                shutil.copytree(moved_step, step)
                self.swapped = True
            return super().stat(
                path,
                dir_fd=dir_fd,
                follow_symlinks=follow_symlinks,
            )

    faulting_fs = SwapBeforeNamedStat()
    with (
        pytest.raises(SMLArtifactError, match="entry swap"),
        open_checkpoint_reader(run, step=0, fs=faulting_fs),
    ):
        pytest.fail("initial checkpoint step swap was accepted")

    assert faulting_fs.swapped is True
    with pytest.raises(OSError):
        os.fstat(faulting_fs.opened_descriptor)


def test_checkpoint_step_swap_after_payload_hash_fails_final_named_inode_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A retained step may be consumed, but its replaced published name must fail."""
    run = _write_valid_checkpoint_run(tmp_path)
    step = run / "checkpoints" / "step-000000000"
    moved_step = run / "checkpoints" / "retained-step"
    real_load = mx.load
    swapped = False
    loaded_original = False

    class SwappingMlx:
        def __getattr__(self, name: str):
            return getattr(mx, name)

        def load(self, stream, *, format):
            nonlocal loaded_original, swapped
            if not swapped:
                step.rename(moved_step)
                shutil.copytree(moved_step, step)
                swapped = True
            arrays = real_load(stream, format=format)
            if "weight" in arrays:
                loaded_original = bool(
                    mx.array_equal(
                        arrays["weight"].astype(mx.float32),
                        mx.array([1.0], dtype=mx.float32),
                    )
                )
            return arrays

    monkeypatch.setattr(checkpoint_module, "_mlx_core", lambda: SwappingMlx())

    with (
        pytest.raises(SMLArtifactError, match="checkpoint step 0.*swapped"),
        open_checkpoint_reader(run, step=0),
    ):
        pytest.fail("swapped named step was accepted")

    assert swapped is True
    assert loaded_original is True


def test_full_resolution_rejects_false_safetensors_metadata(tmp_path: Path) -> None:
    """Byte-valid payloads must not earn FULL when their array declaration is false."""
    run = tmp_path / "run"
    step_directory = run / "checkpoints" / "step-000000000"
    step_directory.mkdir(parents=True)
    run_manifest = PretrainingRunManifest(
        kind="pretraining-run",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        model={"rope_scaling_factor": 1.0},
        precision={"working_parameter_dtype": "bfloat16"},
        optimizer={"kind": "adamw"},
        loader={"microbatch_size": 1},
        checkpoint={"interval": 1},
        tokenizer_identity="sha256:" + "1" * 64,
        data_identity="sha256:" + "2" * 64,
        diagnostic_data_locator=None,
    )
    run_manifest = replace(run_manifest, identity=run_manifest.recompute_identity())
    (run / "run.json").write_bytes(canonical_json_bytes(run_manifest))

    groups = {
        "model.safetensors": {"weight": mx.array([1.0], dtype=mx.bfloat16)},
        "master.safetensors": {"weight": mx.array([1.0], dtype=mx.float32)},
        "optimizer.safetensors": {
            "step": mx.array(0, dtype=mx.int32),
            "first_moments.weight": mx.array([0.0], dtype=mx.float32),
            "second_moments.weight": mx.array([0.0], dtype=mx.float32),
        },
        "trainer.safetensors": {
            "accumulation_count": mx.array(0, dtype=mx.int32),
            "next_key": mx.random.key(7),
            "loss_numerator": mx.array(0.0, dtype=mx.float32),
            "accumulators.weight": mx.array([0.0], dtype=mx.float32),
        },
    }
    references = {}
    for logical_path, arrays in groups.items():
        path = step_directory / logical_path
        mx.save_safetensors(path, arrays)
        references[logical_path] = _array_ref(
            path,
            logical_path,
            arrays,
            declared_shape=(2,) if logical_path == "model.safetensors" else None,
        )
    state_path = step_directory / "state.json"
    state_path.write_bytes(
        canonical_json_bytes(
            {
                "kind": "pretraining-state",
                "version": 1,
                "owning_run_identity": run_manifest.identity,
                "step": 0,
                "rows": 0,
                "microsteps": 0,
                "cursor": {
                    "epoch": 0,
                    "shard_order_position": 0,
                    "row_offset": 0,
                },
            }
        )
    )
    manifest = PretrainingCheckpointManifest(
        kind="pretraining-checkpoint",
        version=1,
        identity=_PLACEHOLDER_IDENTITY,
        owning_run_identity=run_manifest.identity,
        step=0,
        scalar_state=_payload_ref(state_path, "state.json"),
        model=references["model.safetensors"],
        master=references["master.safetensors"],
        optimizer=references["optimizer.safetensors"],
        trainer=references["trainer.safetensors"],
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (step_directory / "checkpoint.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(SMLArtifactError, match="array metadata.*model.safetensors"):
        resolve_exact_step(run, step=0, verification=VerificationLevel.FULL)

    corrected_model = _array_ref(
        step_directory / "model.safetensors",
        "model.safetensors",
        groups["model.safetensors"],
    )
    mismatched_master = {"weight": mx.array([2.0], dtype=mx.float32)}
    master_path = step_directory / "master.safetensors"
    mx.save_safetensors(master_path, mismatched_master)
    manifest = replace(
        manifest,
        identity=_PLACEHOLDER_IDENTITY,
        model=corrected_model,
        master=_array_ref(master_path, "master.safetensors", mismatched_master),
    )
    manifest = replace(manifest, identity=manifest.recompute_identity())
    (step_directory / "checkpoint.json").write_bytes(canonical_json_bytes(manifest))

    with pytest.raises(SMLArtifactError, match="exact BF16 cast"):
        verify_artifact(run, full=True)


def test_verified_checkpoint_contents_mappings_are_deeply_immutable() -> None:
    scalar = {
        "kind": "pretraining-state",
        "cursor": {"epoch": 0, "shard_order_position": 0, "row_offset": 0},
    }
    inner = {"weight": mx.array([1.0], dtype=mx.float32)}
    groups = {"model.safetensors": inner}
    contents = VerifiedCheckpointContents(scalar, groups)

    with pytest.raises(TypeError):
        contents.scalar_state["kind"] = "mutated"
    with pytest.raises(TypeError):
        contents.scalar_state["cursor"]["epoch"] = 1
    with pytest.raises(TypeError):
        contents.array_groups["trainer.safetensors"] = {}
    with pytest.raises(TypeError):
        contents.array_groups["model.safetensors"]["weight"] = mx.array(
            [2.0], dtype=mx.float32
        )

    scalar["injected"] = True
    scalar["cursor"]["epoch"] = 9
    inner["other"] = mx.array([0.0], dtype=mx.float32)
    groups["injected"] = {}

    assert "injected" not in contents.scalar_state
    assert contents.scalar_state["cursor"]["epoch"] == 0
    assert "other" not in contents.array_groups["model.safetensors"]
    assert "injected" not in contents.array_groups
    copied_scalar = dict(contents.scalar_state)
    copied_groups = {path: dict(group) for path, group in contents.array_groups.items()}
    assert copied_scalar["kind"] == "pretraining-state"
    assert copied_groups["model.safetensors"]["weight"].shape == (1,)
