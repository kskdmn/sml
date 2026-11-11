from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from v2.benchmarks import swag_quality


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
