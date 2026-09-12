from __future__ import annotations

import itertools
import json
import random
from pathlib import Path

import pytest
import zstandard as zstd
from sml.data.corpus import (
    CorpusConfig,
    CorpusSamplingConfig,
    discover_corpus_files,
    iter_filtered_texts,
    iter_sampled_texts,
    sample_texts,
)
from sml.errors import SMLDataError


def _write_zstd_jsonl(path: Path, lines: list[bytes]) -> None:
    path.write_bytes(zstd.ZstdCompressor().compress(b"\n".join(lines) + b"\n"))


def test_discovery_is_seeded_without_mutating_global_random_state(tmp_path):
    for name in ("c.jsonl.zst", "a.jsonl.zst", "b.jsonl.zst"):
        (tmp_path / name).write_bytes(b"")
    (tmp_path / ".hidden.jsonl.zst").write_bytes(b"")
    (tmp_path / "ignored.txt").write_bytes(b"")
    (tmp_path / "nested.jsonl.zst").mkdir()
    config = CorpusConfig(
        input_root=tmp_path,
        filename_pattern=r".*\.jsonl\.zst",
        file_order_seed=17,
    )
    control = random.Random(91)
    random.seed(91)

    discovered = discover_corpus_files(config)

    expected = [
        tmp_path / name for name in ("a.jsonl.zst", "b.jsonl.zst", "c.jsonl.zst")
    ]
    random.Random(17).shuffle(expected)
    assert discovered == tuple(expected)
    assert random.random() == control.random()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"filename_pattern": "["}, "filename_pattern"),
        ({"min_text_bytes": -1}, "min_text_bytes"),
        ({"min_text_bytes": 5, "max_text_bytes": 4}, "max_text_bytes"),
        ({"max_rows_per_file": 0}, "max_rows_per_file"),
        ({"max_files": 0}, "max_files"),
        ({"max_files": True}, "max_files"),
        ({"text_field": ""}, "text_field"),
    ],
)
def test_corpus_config_rejects_invalid_public_values(tmp_path, overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        CorpusConfig(input_root=tmp_path, **overrides)


def test_discovery_requires_an_existing_directory_and_expands_home(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path))
    root = tmp_path / "corpus"
    root.mkdir()

    assert CorpusConfig(input_root=Path("~/corpus")).input_root == root
    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_corpus_files(CorpusConfig(input_root=tmp_path / "missing"))
    (tmp_path / "file").write_bytes(b"")
    with pytest.raises(NotADirectoryError, match="directory"):
        discover_corpus_files(CorpusConfig(input_root=tmp_path / "file"))


def test_filtered_texts_stream_zstd_apply_physical_row_cap_and_byte_boundaries(
    tmp_path,
):
    shard = tmp_path / "a.jsonl.zst"
    exactly_min = "é" * 3
    exactly_max = "x" * 8
    _write_zstd_jsonl(
        shard,
        [
            b"",
            b"[]",
            ('{"text":"  ' + exactly_min + '  "}').encode(),
            ('{"text":"a\\u0000  b\\t' + "x" * 4 + '"}').encode(),
            ('{"text":"' + exactly_max + '"}').encode(),
            b'{"text":"too late"}',
        ],
    )
    config = CorpusConfig(
        input_root=tmp_path,
        filename_pattern=r".*\.jsonl\.zst",
        min_text_bytes=6,
        max_text_bytes=8,
        max_rows_per_file=5,
    )

    texts = iter_filtered_texts(config, (shard,))

    assert list(texts) == [exactly_min, "a b xxxx", exactly_max]
    assert texts.physical_lines_read == 5
    assert texts.object_rows_read == 3
    assert texts.texts_used == 3


def test_filtered_texts_is_lazy_and_reports_one_based_malformed_json_line(tmp_path):
    shard = tmp_path / "broken.jsonl.zst"
    _write_zstd_jsonl(shard, [b"", b"[]", b"not-json"])
    texts = iter_filtered_texts(
        CorpusConfig(input_root=tmp_path, min_text_bytes=1),
        (shard,),
    )

    with pytest.raises(SMLDataError, match=r"broken\.jsonl\.zst at line 3"):
        list(texts)


def test_filtered_texts_replace_invalid_utf8_and_ignore_non_string_text(tmp_path):
    shard = tmp_path / "rows.jsonl.zst"
    _write_zstd_jsonl(
        shard,
        [b'{"text":7}', b'{"text":"abc\xffdef"}', b'{"other":"ignored"}'],
    )

    texts = iter_filtered_texts(
        CorpusConfig(
            input_root=tmp_path,
            filename_pattern=r".*\.jsonl\.zst",
            min_text_bytes=1,
        ),
        (shard,),
    )

    assert list(texts) == ["abc�def"]
    assert texts.object_rows_read == 3
    assert texts.texts_used == 1


@pytest.mark.parametrize("missing_bytes", [1, 4, 12])
def test_full_corpus_read_rejects_truncated_frame(tmp_path, missing_bytes):
    shard = tmp_path / "truncated.jsonl.zst"
    compressed = zstd.ZstdCompressor(write_checksum=True).compress(
        b'{"text":"first"}\n{"text":"second"}\n'
    )
    shard.write_bytes(compressed[:-missing_bytes])
    config = CorpusConfig(input_root=tmp_path, min_text_bytes=1, max_rows_per_file=None)

    with pytest.raises(SMLDataError, match="truncated.*incomplete zstd frame"):
        list(iter_filtered_texts(config, (shard,)))


def test_corpus_reads_concatenated_frames_and_checks_final_frame(tmp_path):
    shard = tmp_path / "concatenated.jsonl.zst"
    compressor = zstd.ZstdCompressor(write_checksum=True)
    first = compressor.compress(b'{"text":"first"}\n')
    second = compressor.compress(b'{"text":"second"}\n')
    config = CorpusConfig(input_root=tmp_path, min_text_bytes=1, max_rows_per_file=None)
    shard.write_bytes(first + second)
    assert list(iter_filtered_texts(config, (shard,))) == ["first", "second"]

    shard.write_bytes(first + second[:-1])
    with pytest.raises(SMLDataError, match="incomplete zstd frame"):
        list(iter_filtered_texts(config, (shard,)))


def test_corpus_row_limit_does_not_require_consuming_later_frames(tmp_path):
    shard = tmp_path / "limited.jsonl.zst"
    compressor = zstd.ZstdCompressor(write_checksum=True)
    first = compressor.compress(b'{"text":"first"}\n')
    second = compressor.compress(b'{"text":"second"}\n')
    shard.write_bytes(first + second[:-1])
    config = CorpusConfig(input_root=tmp_path, min_text_bytes=1, max_rows_per_file=1)

    assert list(iter_filtered_texts(config, (shard,))) == ["first"]


def test_default_discovery_samples_all_shard_names_with_a_file_budget(tmp_path):
    for index in range(120):
        (tmp_path / f"data-{index:04d}.jsonl.zst").write_bytes(b"")
    config = CorpusConfig(input_root=tmp_path, shuffle_files=False)

    selected = discover_corpus_files(config)
    shuffled = discover_corpus_files(CorpusConfig(input_root=tmp_path))

    assert len(selected) == 100
    assert list(selected) == sorted(selected)
    assert set(selected) == set(shuffled)
    assert any(path.name.startswith("data-01") for path in selected)
    assert selected == discover_corpus_files(config)
    assert set(selected) != set(
        discover_corpus_files(CorpusConfig(input_root=tmp_path, file_order_seed=43))
    )
    assert (
        len(discover_corpus_files(CorpusConfig(input_root=tmp_path, max_files=None)))
        == 120
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_documents": 0}, "max_documents"),
        ({"max_documents": True}, "max_documents"),
        ({"max_bytes": -1}, "max_bytes"),
        ({"max_bytes": 1.5}, "max_bytes"),
        ({"seed": False}, "seed"),
        ({"max_documents": None, "max_bytes": None}, "at least one"),
    ],
)
def test_sampling_config_rejects_invalid_values(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        CorpusSamplingConfig(**overrides)


def test_sampling_is_seeded_and_does_not_change_global_random_state():
    texts = [f"document {index}" for index in range(200)]
    sampling = CorpusSamplingConfig(max_documents=20, max_bytes=None, seed=17)
    random_state = random.getstate()

    selected = list(sample_texts(texts, sampling))

    assert random.getstate() == random_state
    assert len(selected) == 20
    assert selected == list(sample_texts(texts, sampling))
    assert selected != list(
        sample_texts(
            texts,
            CorpusSamplingConfig(max_documents=20, max_bytes=None, seed=18),
        )
    )
    assert any(text in texts[100:] for text in selected)
    assert selected == [text for text in texts if text in selected]


def test_sampling_obeys_utf8_bytes_and_normalizes_before_deduplication(tmp_path):
    shard = tmp_path / "any-shard.jsonl.zst"
    _write_zstd_jsonl(
        shard,
        [
            json.dumps({"text": text}, ensure_ascii=False).encode("utf-8")
            for text in ("  é\n é  ", "é é", "é\t\x00é", "ab", "é" * 10)
        ],
    )
    config = CorpusConfig(input_root=tmp_path, min_text_bytes=1)

    selected = list(
        iter_sampled_texts(
            config,
            CorpusSamplingConfig(max_documents=None, max_bytes=7),
        )
    )

    assert selected == ["é é", "ab"]
    assert sum(len(text.encode("utf-8")) for text in selected) == 7
    assert list(sample_texts(["é"], CorpusSamplingConfig(max_bytes=1))) == []


def test_sampling_set_is_independent_of_traversal_and_duplicate_placement():
    texts = ("a" * 5, "b" * 7, "c" * 11, "d" * 13, "e" * 17)
    sampling = CorpusSamplingConfig(max_documents=3, max_bytes=21)
    expected = set(sample_texts(texts, sampling))
    assert expected

    for ordering in itertools.permutations(texts):
        selected = list(sample_texts((*ordering, *texts, *reversed(texts)), sampling))
        assert set(selected) == expected
        assert len(selected) == len(set(selected)) <= 3
        assert sum(len(text.encode("utf-8")) for text in selected) <= 21
        assert selected == [text for text in ordering if text in expected]


def test_sampling_represents_multiple_shards_and_later_scanned_rows(tmp_path):
    for name in ("first", "second"):
        _write_zstd_jsonl(
            tmp_path / f"{name}.jsonl.zst",
            [
                json.dumps({"text": f"{name} document {index}"}).encode()
                for index in range(100)
            ],
        )
    selected = list(
        iter_sampled_texts(
            CorpusConfig(input_root=tmp_path, min_text_bytes=1),
            CorpusSamplingConfig(max_documents=20, max_bytes=None),
        )
    )

    assert len(selected) == 20
    assert {text.split()[0] for text in selected} == {"first", "second"}
    assert any(int(text.split()[-1]) >= 50 for text in selected)


def test_sampling_keeps_physical_scan_budget_and_lazy_input_errors(tmp_path):
    shard = tmp_path / "limited.jsonl.zst"
    _write_zstd_jsonl(
        shard,
        [b'{"text":"first"}', b'{"text":"second"}', b"invalid-json"],
    )
    sampling = CorpusSamplingConfig(max_documents=1)
    bounded = CorpusConfig(input_root=tmp_path, min_text_bytes=1, max_rows_per_file=2)
    assert len(list(iter_sampled_texts(bounded, sampling))) == 1

    unbounded = iter_sampled_texts(
        CorpusConfig(input_root=tmp_path, min_text_bytes=1, max_rows_per_file=None),
        sampling,
    )
    with pytest.raises(SMLDataError, match=r"limited\.jsonl\.zst at line 3"):
        next(unbounded)
