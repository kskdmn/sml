from __future__ import annotations

import hashlib
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_map
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.lora import LoRAConfig, apply_lora, split_adapter_parameters

from v2.benchmarks import swag_quality
from v2.benchmarks.swag_quality import (
    CANONICAL_STEPS,
    MODEL_SEED,
    SwagQualityRecord,
    SwagQualityReport,
    SwagQualityWorkload,
    build_swag_quality_workload,
    decide_swag_quality,
    harness_content_identity,
    validate_swag_quality_records,
)
from v2.benchmarks.workload import structured_identity

ROOT = Path(__file__).parents[3]


@pytest.fixture
def tiny_quality_runtime():
    config = ModelConfig(
        vocab_size=300,
        hidden_size=8,
        num_layers=1,
        num_q_heads=2,
        num_kv_heads=1,
        intermediate_size=16,
        original_context_length=64,
        hidden_dropout=0.5,
    )
    model = SMLLanguageModel(config, key=mx.random.key(3))
    apply_lora(model, LoRAConfig(rank=2, dropout=0.5), key=mx.random.key(4))
    adapters, frozen_base = split_adapter_parameters(model.parameters())
    adapters = tree_map(lambda array: mx.ones_like(array) * 0.1, adapters)
    encoded = {
        name: value[:2]
        for name, value in swag_quality._load_encoded_arrays(
            ROOT / swag_quality.VALIDATION_FIXTURE
        ).items()
    }
    return config, model, adapters, frozen_base, encoded


def test_validation_disables_dropout_and_does_not_differentiate(
    monkeypatch, tiny_quality_runtime
):
    config, model, adapters, frozen_base, encoded = tiny_quality_runtime
    real_forward = SMLLanguageModel.forward_arrays
    calls = []

    def record_forward(self, *args, **kwargs):
        calls.append((kwargs["training"], kwargs["key"]))
        return real_forward(self, *args, **kwargs)

    def forbid_gradients(*args, **kwargs):
        raise AssertionError("quality validation must not compute gradients")

    monkeypatch.setattr(SMLLanguageModel, "forward_arrays", record_forward)
    monkeypatch.setattr(mx, "value_and_grad", forbid_gradients)
    mx.random.seed(11)
    first = swag_quality._evaluate_validation(
        model, adapters, frozen_base, encoded, model_config=config
    )
    mx.random.seed(29)
    second = swag_quality._evaluate_validation(
        model, adapters, frozen_base, encoded, model_config=config
    )

    assert first == second
    assert first[2] == 2
    assert np.isfinite(first[0])
    assert calls == [(False, None)] * 4


def test_swag_quality_training_uses_production_counter_keys(
    monkeypatch, tiny_quality_runtime, canonical_workload
):
    config, model, adapters, frozen_base, encoded = tiny_quality_runtime
    real_build = swag_quality.build_swag_kernels
    received_keys = []

    def recording_kernels(*args, **kwargs):
        kernels = real_build(*args, **kwargs)
        real_microstep = kernels.compiled_ranking_microstep_core

        def record_microstep(adapters, base, trainer_tree, *arrays):
            received_keys.append(trainer_tree[2])
            return real_microstep(adapters, base, trainer_tree, *arrays)

        return replace(kernels, compiled_ranking_microstep_core=record_microstep)

    monkeypatch.setattr(swag_quality, "build_swag_kernels", recording_kernels)
    swag_quality._run_runtime(
        runtime="candidate",
        compile=False,
        workload=replace(canonical_workload, ordered_batches=((0,), (1,))),
        model=model,
        frozen_base=frozen_base,
        adapters=adapters,
        trainer_key=mx.random.key(999),
        training=encoded,
        validation=encoded,
        model_config=config,
    )

    assert len(received_keys) == 2
    for index, key in enumerate(received_keys):
        assert bool(
            mx.array_equal(
                key,
                swag_quality.counter_random_key(canonical_workload.model_seed, index),
            )
        )


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
    assert (
        decide_swag_quality(
            replace(passing, candidate_validation_loss=1.011),
        )
        == "fail"
    )
    assert (
        decide_swag_quality(
            replace(passing, candidate_accuracy=0.789),
        )
        == "fail"
    )
    assert (
        decide_swag_quality(
            replace(passing, candidate_examples=511),
        )
        == "fail"
    )


@pytest.mark.parametrize(
    ("change", "value"),
    [
        ("candidate_finite", False),
        ("oracle_finite", False),
        ("candidate_validation_loss", float("nan")),
        ("oracle_validation_loss", float("inf")),
        ("candidate_accuracy", float("nan")),
        ("oracle_examples", 511),
    ],
)
def test_swag_quality_gate_fails_closed_before_thresholds(change, value):
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

    assert decide_swag_quality(replace(passing, **{change: value})) == "fail"


@pytest.fixture(scope="module")
def canonical_workload() -> SwagQualityWorkload:
    return build_swag_quality_workload(ROOT)


def test_harness_identity_includes_publication_dependencies_in_order():
    expected = hashlib.sha256()
    for relative in (
        Path("v2/benchmarks/swag_quality.py"),
        Path("v2/tests/unit/test_swag_quality.py"),
        Path("v2/benchmarks/journal.py"),
        Path("v2/benchmarks/evidence.py"),
        Path("v2/benchmarks/recovery.py"),
    ):
        expected.update((ROOT / relative).read_bytes())

    assert harness_content_identity(ROOT) == f"sha256:{expected.hexdigest()}"


def test_record_creates_evidence_parents_before_loading_data(tmp_path, monkeypatch):
    destinations = swag_quality._canonical_evidence_destinations(
        tmp_path,
        tmp_path / swag_quality.RECORD_MANIFEST_PATH,
        tmp_path / swag_quality.RECORD_RAW_PATH,
        tmp_path / swag_quality.RECORD_REPORT_PATH,
    )
    monkeypatch.setattr(swag_quality, "_root", lambda: tmp_path)
    monkeypatch.setattr(
        swag_quality, "_require_clean_recording_checkout", lambda *_args: None
    )
    monkeypatch.setattr(swag_quality, "_git", lambda *_args: "a" * 40)

    def load_fixture(_path):
        assert all(path.parent.is_dir() for _name, path in destinations.ordered())
        raise RuntimeError("stop before training")

    monkeypatch.setattr(swag_quality, "_load_encoded_arrays", load_fixture)
    assert not any(path.parent.exists() for _name, path in destinations.ordered())
    with pytest.raises(RuntimeError, match="stop before training"):
        swag_quality._record(
            SimpleNamespace(
                manifest=destinations.manifest,
                raw_output=destinations.raw_output,
                output=destinations.report,
            )
        )


def test_record_refuses_concurrent_evidence_writers(tmp_path, monkeypatch):
    args = SimpleNamespace(
        manifest=tmp_path / swag_quality.RECORD_MANIFEST_PATH,
        raw_output=tmp_path / swag_quality.RECORD_RAW_PATH,
        output=tmp_path / swag_quality.RECORD_REPORT_PATH,
    )
    monkeypatch.setattr(swag_quality, "_root", lambda: tmp_path)

    def unexpected_record(*_args):
        raise AssertionError("a second recording reached its runtime")

    monkeypatch.setattr(swag_quality, "_record_locked", unexpected_record)
    with (
        swag_quality.baseline_output_lock(args.manifest, args.raw_output),
        ThreadPoolExecutor(max_workers=1) as executor,
        pytest.raises(RuntimeError, match="already locked"),
    ):
        executor.submit(swag_quality._record, args).result(timeout=10)


@pytest.mark.parametrize("existing_index", [0, 1, 2])
def test_publication_preserves_existing_evidence_on_conflict(tmp_path, existing_index):
    destinations = swag_quality._EvidenceDestinations(
        manifest=tmp_path / "manifest.json",
        raw_output=tmp_path / "raw.jsonl",
        report=tmp_path / "report.json",
    )
    ordered = destinations.ordered()
    for _name, path in ordered[existing_index:]:
        path.write_bytes(b"other recorder's evidence")
    payloads = {name: b'{"new":true}\n' for name, _path in ordered}

    with pytest.raises(FileExistsError):
        swag_quality._publish_evidence(destinations, payloads)

    for _name, path in ordered[:existing_index]:
        assert not path.exists()
    for _name, path in ordered[existing_index:]:
        assert path.read_bytes() == b"other recorder's evidence"


def test_publication_cleanup_preserves_replaced_output(tmp_path, monkeypatch):
    destinations = swag_quality._EvidenceDestinations(
        manifest=tmp_path / "manifest.json",
        raw_output=tmp_path / "raw.jsonl",
        report=tmp_path / "report.json",
    )
    create = swag_quality._durable_create

    def replace_previous_output(path, payload):
        if path == destinations.manifest:
            destinations.raw_output.rename(tmp_path / "original.jsonl")
            destinations.raw_output.write_bytes(b"replacement evidence")
            raise OSError("publication interrupted")
        create(path, payload)

    monkeypatch.setattr(swag_quality, "_durable_create", replace_previous_output)
    with pytest.raises(OSError, match="publication interrupted"):
        swag_quality._publish_evidence(
            destinations,
            {name: b"{}\n" for name, _path in destinations.ordered()},
        )
    assert destinations.raw_output.read_bytes() == b"replacement evidence"


def test_publication_creates_new_evidence_files(tmp_path):
    destinations = swag_quality._EvidenceDestinations(
        manifest=tmp_path / "manifests" / "manifest.json",
        raw_output=tmp_path / "results" / "raw.jsonl",
        report=tmp_path / "results" / "report.json",
    )
    payloads = {name: b'{"complete":true}\n' for name, _path in destinations.ordered()}
    swag_quality._publish_evidence(destinations, payloads)
    for name, path in destinations.ordered():
        assert path.read_bytes() == payloads[name]


def test_workload_pins_256_steps_disjoint_encoded_examples_and_identities(
    canonical_workload,
):
    workload = canonical_workload

    assert CANONICAL_STEPS == 256
    assert workload.optimizer_steps == 256
    assert len(workload.ordered_batches) == 256
    assert workload.score_policy == "fp32-mean-continuation-including-eos-v1"
    assert workload.training_fixture.source_identity != (
        workload.validation_fixture.source_identity
    )
    assert workload.training_fixture.semantic_identity != (
        workload.validation_fixture.semantic_identity
    )
    assert workload.frozen_bf16_base_identity.startswith("sha256:")
    assert workload.fp32_master_identity.startswith("sha256:")
    assert workload.frozen_bf16_base_identity != workload.fp32_master_identity
    assert workload.initial_fp32_adapter_identity.startswith("sha256:")
    assert workload.identity == workload.recompute_identity()
    assert (
        SwagQualityWorkload.from_dict(workload.to_dict()).to_dict()
        == workload.to_dict()
    )

    for fixture in (workload.training_fixture, workload.validation_fixture):
        path = ROOT / fixture.logical_path
        assert path.stat().st_size == fixture.byte_size
        arrays = np.load(path, allow_pickle=False)
        input_ids = arrays["input_ids"]
        assert input_ids.dtype == np.dtype("<i4")
        assert input_ids.ndim == 3
        assert input_ids.shape[1] == 4
        assert int(input_ids.min()) >= 0


def test_workload_rejects_a_validation_example_copied_from_training(
    canonical_workload, tmp_path
):
    for relative in (
        Path("v2/benchmarks/swag_quality.py"),
        Path("v2/tests/unit/test_swag_quality.py"),
        Path(canonical_workload.training_fixture.logical_path),
        Path(canonical_workload.validation_fixture.logical_path),
        *(Path(path) for path in canonical_workload.production_dependency_components),
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / relative
        if source.is_file():
            destination.write_bytes(source.read_bytes())

    training = np.load(
        tmp_path / canonical_workload.training_fixture.logical_path,
        allow_pickle=False,
    )
    validation_path = tmp_path / canonical_workload.validation_fixture.logical_path
    validation = dict(np.load(validation_path, allow_pickle=False))
    validation["input_ids"][0] = np.array(training["input_ids"][0])
    validation["valid_token_mask"][0] = np.array(training["valid_token_mask"][0])
    validation["score_mask"][0] = np.array(training["score_mask"][0])
    validation["labels"][0] = np.array(training["labels"][0])
    np.savez(validation_path, **validation)

    with pytest.raises(ValueError, match="source-disjoint"):
        build_swag_quality_workload(tmp_path)


def _record(
    workload: SwagQualityWorkload,
    runtime: str,
    *,
    validation_loss: float,
    accuracy: float,
    examples: int,
    finite: bool = True,
    frozen_base_identity: str | None = None,
    train_loss: float = 1.0,
) -> SwagQualityRecord:
    record = SwagQualityRecord(
        kind="swag-quality-record",
        version=1,
        identity="sha256:" + "0" * 64,
        runtime=runtime,
        step=CANONICAL_STEPS,
        workload_identity=workload.identity,
        train_loss=train_loss,
        validation_loss=validation_loss,
        validation_accuracy=accuracy,
        real_example_count=examples,
        finite=finite,
        frozen_base_identity=(
            workload.frozen_bf16_base_identity
            if frozen_base_identity is None
            else frozen_base_identity
        ),
        adapter_identity="sha256:" + ("a" if runtime == "candidate" else "b") * 64,
    )
    return replace(record, identity=record.recompute_identity())


def test_raw_validation_rejects_missing_extra_changed_base_and_counts(
    canonical_workload,
):
    records = (
        _record(
            canonical_workload,
            "candidate",
            validation_loss=1.005,
            accuracy=0.80,
            examples=8,
        ),
        _record(
            canonical_workload,
            "oracle",
            validation_loss=1.0,
            accuracy=0.80,
            examples=8,
        ),
    )

    with pytest.raises(ValueError, match="real-example"):
        validate_swag_quality_records(canonical_workload, records)

    passing = (
        _record(
            canonical_workload,
            "candidate",
            validation_loss=1.005,
            accuracy=0.80,
            examples=canonical_workload.validation_fixture.example_count,
        ),
        _record(
            canonical_workload,
            "oracle",
            validation_loss=1.0,
            accuracy=0.80,
            examples=canonical_workload.validation_fixture.example_count,
        ),
    )
    report = validate_swag_quality_records(canonical_workload, passing)
    assert (
        report.candidate_examples
        == report.oracle_examples
        == canonical_workload.validation_fixture.example_count
    )
    assert decide_swag_quality(report) == "pass"

    with pytest.raises(ValueError, match="exactly two"):
        validate_swag_quality_records(canonical_workload, passing[:1])
    with pytest.raises(ValueError, match="exactly two"):
        validate_swag_quality_records(canonical_workload, (*passing, passing[0]))

    changed_base = _record(
        canonical_workload,
        "candidate",
        validation_loss=1.005,
        accuracy=0.80,
        examples=canonical_workload.validation_fixture.example_count,
        frozen_base_identity="sha256:" + "f" * 64,
    )
    with pytest.raises(ValueError, match="base"):
        validate_swag_quality_records(canonical_workload, (changed_base, passing[1]))

    mismatched = _record(
        canonical_workload,
        "oracle",
        validation_loss=1.0,
        accuracy=0.80,
        examples=canonical_workload.validation_fixture.example_count - 1,
    )
    with pytest.raises(ValueError, match="real-example"):
        validate_swag_quality_records(canonical_workload, (passing[0], mismatched))

    nonfinite = _record(
        canonical_workload,
        "candidate",
        validation_loss=1.005,
        accuracy=0.80,
        examples=canonical_workload.validation_fixture.example_count,
        finite=False,
    )
    failed = validate_swag_quality_records(canonical_workload, (nonfinite, passing[1]))
    assert decide_swag_quality(failed) == "fail"


def test_public_record_accepts_only_exactly_256_steps():
    parser = swag_quality._build_parser()
    common = [
        "--manifest",
        "manifest.json",
        "--raw-output",
        "raw.jsonl",
        "--output",
        "report.json",
    ]

    assert parser.parse_args(["record", "--steps", "256", *common]).steps == 256
    for invalid in ("1", "255", "257", "1000"):
        with pytest.raises(SystemExit):
            parser.parse_args(["record", "--steps", invalid, *common])


def test_manifest_fields_and_output_paths_fail_closed(canonical_workload):
    destinations = swag_quality._canonical_evidence_destinations(
        ROOT,
        ROOT / swag_quality.RECORD_MANIFEST_PATH,
        ROOT / swag_quality.RECORD_RAW_PATH,
        ROOT / swag_quality.RECORD_REPORT_PATH,
    )
    command = swag_quality._recording_command_document(ROOT, destinations)
    session_identity = swag_quality._recording_session_identity(
        "a" * 40, canonical_workload.identity, command
    )
    manifest = swag_quality._manifest_document(
        workload=canonical_workload,
        source_commit="a" * 40,
        recording_command=command,
        phase_times={
            "setup": 1.0,
            "candidate": 3.0,
            "oracle": 4.0,
            "validation_serialization": 2.0,
        },
        peak_memory=123,
        raw_identity="sha256:" + "b" * 64,
        raw_file_identity="sha256:" + "c" * 64,
        raw_bytes=1_000,
        report_identity="sha256:" + "d" * 64,
        report_file_identity="sha256:" + "e" * 64,
        report_bytes=500,
        recording_session_identity=session_identity,
    )

    assert swag_quality._validate_manifest_fields(
        manifest, canonical_workload, command
    ) == dict(manifest)
    assert manifest["source_commit"] == manifest["harness_commit"] == "a" * 40
    assert manifest["optimizer_steps"] == 256
    assert manifest["record_count"] == 2

    with pytest.raises(ValueError, match="canonical evidence destinations"):
        swag_quality._canonical_evidence_destinations(
            ROOT,
            ROOT / swag_quality.RECORD_MANIFEST_PATH,
            ROOT / swag_quality.RECORD_RAW_PATH,
            ROOT / "v2/benchmarks/results/forged.json",
        )


def test_swag_recording_accepts_only_current_complete_destination_triple():
    destinations = swag_quality._canonical_evidence_destinations(
        ROOT,
        ROOT / swag_quality.RECORD_MANIFEST_PATH,
        ROOT / swag_quality.RECORD_RAW_PATH,
        ROOT / swag_quality.RECORD_REPORT_PATH,
    )
    command = swag_quality._recording_command_document(ROOT, destinations)
    assert command["destinations"] == {
        "manifest": swag_quality.RECORD_MANIFEST_PATH.as_posix(),
        "raw_output": swag_quality.RECORD_RAW_PATH.as_posix(),
        "report": swag_quality.RECORD_REPORT_PATH.as_posix(),
    }


def _recompute_manifest_identity(manifest: dict[str, object]) -> dict[str, object]:
    body = {key: value for key, value in manifest.items() if key != "identity"}
    return {
        **body,
        "identity": structured_identity("sml-swag-quality-manifest-v1", body),
    }


def test_standalone_validation_rejects_over_budget_and_nonpositive_phases(
    canonical_workload,
):
    destinations = swag_quality._canonical_evidence_destinations(
        ROOT,
        ROOT / swag_quality.RECORD_MANIFEST_PATH,
        ROOT / swag_quality.RECORD_RAW_PATH,
        ROOT / swag_quality.RECORD_REPORT_PATH,
    )
    command = swag_quality._recording_command_document(ROOT, destinations)
    session_identity = swag_quality._recording_session_identity(
        "a" * 40, canonical_workload.identity, command
    )
    manifest = swag_quality._manifest_document(
        workload=canonical_workload,
        source_commit="a" * 40,
        recording_command=command,
        phase_times={
            "setup": 1.0,
            "candidate": 3.0,
            "oracle": 4.0,
            "validation_serialization": 2.0,
        },
        peak_memory=123,
        raw_identity="sha256:" + "b" * 64,
        raw_file_identity="sha256:" + "c" * 64,
        raw_bytes=1_000,
        report_identity="sha256:" + "d" * 64,
        report_file_identity="sha256:" + "e" * 64,
        report_bytes=500,
        recording_session_identity=session_identity,
    )

    over_budget = _recompute_manifest_identity(
        {
            **manifest,
            "phase_wall_time_seconds": {
                "setup": 1.0,
                "candidate": 10_000.0,
                "oracle": 5_000.0,
                "validation_serialization": 1.0,
            },
            "measured_wall_time_seconds": 15_002.0,
        }
    )
    with pytest.raises(ValueError, match="budget"):
        swag_quality._validate_manifest_fields(over_budget, canonical_workload, command)

    nonpositive = _recompute_manifest_identity(
        {
            **manifest,
            "phase_wall_time_seconds": {
                "setup": 0.0,
                "candidate": 3.0,
                "oracle": 4.0,
                "validation_serialization": 2.0,
            },
            "measured_wall_time_seconds": 9.0,
        }
    )
    with pytest.raises(ValueError, match="phase"):
        swag_quality._validate_manifest_fields(nonpositive, canonical_workload, command)


def test_verified_source_snapshot_fully_verifies_pretraining_checkpoint(
    monkeypatch, tmp_path
):
    published = tmp_path / "pretraining-run"
    published.mkdir()
    called: list[tuple[Path, bool]] = []

    monkeypatch.setattr(
        swag_quality,
        "_publish_source_pretraining_run",
        lambda *_args, **_kwargs: published,
        raising=False,
    )

    def fake_resolve(path, *, full_verify):
        called.append((path, full_verify))
        raise RuntimeError("verified")

    monkeypatch.setattr(
        swag_quality, "resolve_model_artifact", fake_resolve, raising=False
    )
    with pytest.raises(RuntimeError, match="verified"):
        swag_quality._verified_source_snapshot(ModelConfig(), LoRAConfig(), MODEL_SEED)
    assert called == [(published, True)]


@pytest.fixture(scope="module")
def current_recorded_evidence(tmp_path_factory, canonical_workload):
    """Build validator inputs from current source and bounded synthetic records."""
    root = tmp_path_factory.mktemp("swag-quality-current-evidence")
    workload = canonical_workload
    copied = {
        *swag_quality.HARNESS_COMPONENTS,
        *(Path(path) for path in workload.production_dependency_components),
        swag_quality.TRAINING_FIXTURE,
        swag_quality.VALIDATION_FIXTURE,
    }
    for relative in copied:
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)

    def git(*arguments):
        return subprocess.run(
            ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "--quiet")
    git("config", "user.name", "Quality Evidence Test")
    git("config", "user.email", "quality-evidence@example.invalid")
    git("add", ".")
    git("commit", "--quiet", "-m", "current quality source")
    commit = git("rev-parse", "HEAD")
    destinations = swag_quality._canonical_evidence_destinations(
        root,
        root / swag_quality.RECORD_MANIFEST_PATH,
        root / swag_quality.RECORD_RAW_PATH,
        root / swag_quality.RECORD_REPORT_PATH,
    )
    command = swag_quality._recording_command_document(root, destinations)
    records = tuple(
        _record(
            workload,
            runtime,
            validation_loss=1.005 if runtime == "candidate" else 1.0,
            accuracy=0.75,
            examples=workload.validation_fixture.example_count,
        )
        for runtime in ("candidate", "oracle")
    )
    documents = [record.to_dict() for record in records]
    raw = b"".join(
        swag_quality.canonical_json_bytes(document) + b"\n" for document in documents
    )
    raw_identity = swag_quality.structured_identity(
        "sml-swag-quality-raw-v1", documents
    )
    report = swag_quality._report_document(
        workload.identity,
        raw_identity,
        validate_swag_quality_records(workload, records),
    )
    report_bytes = swag_quality._canonical_json_file_bytes(report)
    manifest = swag_quality._manifest_document(
        workload=workload,
        source_commit=commit,
        recording_command=command,
        phase_times={
            "setup": 1.0,
            "candidate": 1.0,
            "oracle": 1.0,
            "validation_serialization": 1.0,
        },
        peak_memory=1,
        raw_identity=raw_identity,
        raw_file_identity=swag_quality._payload_identity(raw),
        raw_bytes=len(raw),
        report_identity=report["identity"],
        report_file_identity=swag_quality._payload_identity(report_bytes),
        report_bytes=len(report_bytes),
        recording_session_identity=swag_quality._recording_session_identity(
            commit, workload.identity, command
        ),
    )
    for path in (destinations.manifest, destinations.raw_output, destinations.report):
        path.parent.mkdir(parents=True, exist_ok=True)
    destinations.manifest.write_bytes(swag_quality._canonical_json_file_bytes(manifest))
    destinations.raw_output.write_bytes(raw)
    destinations.report.write_bytes(report_bytes)
    return root, destinations, manifest, workload


def test_current_standalone_evidence_validation(current_recorded_evidence, monkeypatch):
    root, destinations, manifest, _workload = current_recorded_evidence
    assert manifest["recording_command"] == swag_quality._recording_command_document(
        root, destinations
    )
    monkeypatch.setattr(swag_quality, "_root", lambda: root)
    assert (
        swag_quality._validate(
            SimpleNamespace(
                manifest=destinations.manifest,
                raw_input=destinations.raw_output,
                report=destinations.report,
            )
        )
        == 0
    )


@pytest.mark.parametrize("changed_input", [None, "production", "harness", "fixture"])
def test_record_reuses_only_current_workload_evidence(
    current_recorded_evidence, tmp_path, monkeypatch, changed_input
):
    recorded_root, _destinations, _manifest, _workload = current_recorded_evidence
    root = tmp_path / "recording"
    shutil.copytree(recorded_root, root)
    destinations = swag_quality._canonical_evidence_destinations(
        root,
        root / swag_quality.RECORD_MANIFEST_PATH,
        root / swag_quality.RECORD_RAW_PATH,
        root / swag_quality.RECORD_REPORT_PATH,
    )
    original_payloads = {
        path: path.read_bytes() for _name, path in destinations.ordered()
    }
    if changed_input is not None:
        relative = {
            "production": Path("v2/src/sml/training/swag.py"),
            "harness": Path("v2/benchmarks/swag_quality.py"),
            "fixture": swag_quality.TRAINING_FIXTURE,
        }[changed_input]
        source = root / relative
        if changed_input == "fixture":
            arrays = swag_quality._load_encoded_arrays(source)
            arrays["labels"][0] = (arrays["labels"][0] + 1) % 4
            swag_quality._write_npz(source, arrays)
        else:
            source.write_bytes(source.read_bytes() + b"\n# changed recording input\n")
        for arguments in (
            ("add", str(source)),
            ("commit", "--quiet", "-m", "change recording input"),
        ):
            subprocess.run(
                ["git", *arguments], cwd=root, check=True, capture_output=True
            )

    def forbid_quality_execution(*_args, **_kwargs):
        raise AssertionError(
            "existing evidence must be checked before quality execution"
        )

    monkeypatch.setattr(swag_quality, "_root", lambda: root)
    monkeypatch.setattr(swag_quality, "_run_runtime", forbid_quality_execution)
    monkeypatch.setattr(
        swag_quality, "_verified_source_snapshot", forbid_quality_execution
    )
    args = SimpleNamespace(
        manifest=destinations.manifest,
        raw_output=destinations.raw_output,
        output=destinations.report,
    )
    if changed_input is not None:
        with pytest.raises(ValueError, match="recording .* changed"):
            swag_quality._record(args)
    else:
        assert swag_quality._record(args) == 0

    assert all(
        path.read_bytes() == payload for path, payload in original_payloads.items()
    )
    assert (
        swag_quality._validate(
            SimpleNamespace(
                manifest=destinations.manifest,
                raw_input=destinations.raw_output,
                report=destinations.report,
            )
        )
        == 0
    )


def test_recorded_validator_rejects_required_component_omission(
    current_recorded_evidence,
):
    root, _destinations, manifest, workload = current_recorded_evidence
    retained = tuple(
        path
        for path in workload.production_dependency_components
        if path != "v2/src/sml/model/layers.py"
    )
    assert len(retained) + 1 == len(workload.production_dependency_components)
    identity = swag_quality._production_dependency_identity(
        tuple(Path(path) for path in retained),
        lambda path: swag_quality._git_bytes(
            root, "show", f"{manifest['source_commit']}:{path.as_posix()}"
        ),
    )
    tampered = replace(
        workload,
        production_dependency_components=retained,
        production_dependency_identity=identity,
    )
    tampered = replace(tampered, identity=tampered.recompute_identity())
    with pytest.raises(ValueError, match="component set changed"):
        swag_quality._validate_harness_commit(root, manifest["source_commit"], tampered)


@pytest.mark.parametrize("unsupported_version", [0, 1, 3])
def test_workload_accepts_only_current_version(canonical_workload, unsupported_version):
    raw = canonical_workload.to_dict()
    raw["version"] = unsupported_version
    with pytest.raises(ValueError, match="unsupported"):
        type(canonical_workload).from_dict(raw)
