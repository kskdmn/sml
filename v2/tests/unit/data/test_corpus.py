from __future__ import annotations

import random
from pathlib import Path

import pytest
import zstandard as zstd
from sml.data.corpus import CorpusConfig, discover_corpus_files, iter_filtered_texts


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

    with pytest.raises(ValueError, match=r"broken\.jsonl\.zst at line 3"):
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
