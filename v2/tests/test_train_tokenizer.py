import importlib.util
import json
import sys
import tempfile
import warnings
from pathlib import Path

from helpers import Spy
import zstandard as zstd

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
SCRIPT_PATH = PROJECT_DIR / "src" / "train_tokenizer.py"
TOKENIZER_PATH = PROJECT_DIR / "src" / "tokenizer.py"
MAX_ROWS_IN_TEST = 3
EXPECTED_NORMALIZED_TEXT_COUNT = 1
REPEATED_WORD_COUNT = 60


def write_zst_text(path: Path, text: str) -> None:
    compressed = zstd.ZstdCompressor().compress(text.encode("utf-8"))
    path.write_bytes(compressed)


def write_zst_rows(path: Path, rows: list[dict[str, object]]) -> None:
    write_zst_text(path, "\n".join(json.dumps(row) for row in rows))


def collect_filtered_texts(tokenizer, input_files, max_rows_per_file=None):
    if max_rows_per_file is None:
        return list(tokenizer.FilteredTextIterable(input_files))

    return list(tokenizer.FilteredTextIterable(input_files, max_rows_per_file))


def load_module(module_name: str, path: Path):
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"builtin type SwigPy.* has no __module__ attribute",
            category=DeprecationWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r"builtin type swigvarlink has no __module__ attribute",
            category=DeprecationWarning,
        )
        spec.loader.exec_module(module)
    return module


def load_script():
    return load_module("train_tokenizer", SCRIPT_PATH)


def load_tokenizer_module():
    return load_module("tokenizer", TOKENIZER_PATH)


class TestTrainTokenizer:
    def test_output_dir_defaults_to_current_numbered_project(self):
        tokenizer = load_script()

        assert PROJECT_DIR / "output" == tokenizer.OUTPUT_DIR

    def test_parse_args_defaults_to_default_tokenizer_model_path(self):
        train_tokenizer = load_script()

        args = train_tokenizer.parse_args([])

        assert train_tokenizer.DEFAULT_TOKENIZER_MODEL_PATH == args.tokenizer_model

    def test_parse_args_accepts_tokenizer_model_path(self):
        train_tokenizer = load_script()

        args = train_tokenizer.parse_args(
            ["--tokenizer-model", "/tmp/custom-tokenizer.model"]
        )

        assert Path("/tmp/custom-tokenizer.model") == args.tokenizer_model

    def test_module_does_not_export_test_only_filtered_text_wrapper(self):
        tokenizer = load_script()

        assert not hasattr(tokenizer, "iter_filtered_texts")

    def test_module_does_not_export_jsonl_lines_wrapper(self):
        tokenizer = load_script()

        assert not hasattr(tokenizer, "iter_jsonl_lines")

    def test_discover_input_files_sorts_matching_zst_shards_and_skips_others(self):
        train_tokenizer = load_script()
        tokenizer = train_tokenizer.tokenizer

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            (root / "dataset-0009.jsonl.zst").write_text("", encoding="utf-8")
            (root / "dataset-0000.jsonl.zst").write_text("", encoding="utf-8")
            (root / "dataset-0010.jsonl.zst").write_text("", encoding="utf-8")
            (root / "dataset-0000.jsonl").write_text("", encoding="utf-8")
            (root / "dataset.jsonl.zst").write_text("", encoding="utf-8")
            (root / ".dataset-0001.jsonl.zst").write_text("", encoding="utf-8")
            (root / "notes.txt").write_text("", encoding="utf-8")

            files = tokenizer.discover_input_files(
                root,
                train_tokenizer.TOKENIZER_SHARD_NAME_REGEX,
            )

        assert [
            "dataset-0000.jsonl.zst",
            "dataset-0009.jsonl.zst",
            "dataset-0010.jsonl.zst",
        ] == [path.name for path in files]

    def test_shuffle_input_files_uses_seeded_deterministic_order(self):
        utils = load_module("utils", SRC_DIR / "utils.py")

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            files = tuple(
                root / name
                for name in (
                    "dataset-0000.jsonl.zst",
                    "dataset-0001.jsonl.zst",
                    "dataset-0002.jsonl.zst",
                )
            )

            first_shuffle = utils.shuffle_input_files(files, seed=42)
            second_shuffle = utils.shuffle_input_files(files, seed=42)

        assert first_shuffle == second_shuffle
        assert sorted(files, key=lambda path: path.name) != list(first_shuffle)

    def test_train_tokenizer_shuffles_discovered_input_files(self, monkeypatch):
        train_tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "sample-0000.jsonl.zst"
            write_zst_rows(
                input_path,
                [{"text": "a" * train_tokenizer.tokenizer.MIN_TEXT_LENGTH}],
            )
            discovered = (input_path,)
            shuffle_spy = Spy(return_value=discovered)
            monkeypatch.setattr(train_tokenizer, "INPUT_DIR", root)
            monkeypatch.setattr(
                train_tokenizer.tokenizer,
                "discover_input_files",
                Spy(return_value=discovered),
            )
            monkeypatch.setattr(
                train_tokenizer,
                "shuffle_input_files",
                shuffle_spy,
            )
            monkeypatch.setattr(
                train_tokenizer.spm.SentencePieceTrainer,
                "train",
                Spy(),
            )
            train_tokenizer.train_tokenizer()

        shuffle_spy.assert_called_once_with(
            discovered,
            seed=train_tokenizer.RANDOM_SEED,
        )

    def test_filtered_text_iterable_reads_only_first_rows_and_filters_min_text_length(
        self,
    ):
        tokenizer = load_tokenizer_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample-0000.jsonl.zst"
            byte_min_text = "\u3042" * ((tokenizer.MIN_TEXT_LENGTH // 3) + 1)
            rows = [
                {"text": "a" * tokenizer.MIN_TEXT_LENGTH},
                {"text": "too short"},
                {"text": byte_min_text},
                {"text": "c" * tokenizer.MIN_TEXT_LENGTH},
            ]
            write_zst_rows(path, rows)

            texts = collect_filtered_texts(
                tokenizer,
                [path],
                max_rows_per_file=MAX_ROWS_IN_TEST,
            )

        assert len(byte_min_text) < tokenizer.MIN_TEXT_LENGTH
        assert (
            len(byte_min_text.encode(tokenizer.TEXT_ENCODING))
            >= tokenizer.MIN_TEXT_LENGTH
        )
        assert ["a" * tokenizer.MIN_TEXT_LENGTH, byte_min_text] == texts

    def test_filtered_text_iterable_keeps_long_utf8_text_when_max_is_unset(self):
        tokenizer = load_tokenizer_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample-0000.jsonl.zst"
            byte_long_text = "\u3042" * 1_000
            valid_text = "a" * tokenizer.MIN_TEXT_LENGTH
            write_zst_rows(path, [{"text": byte_long_text}, {"text": valid_text}])

            texts = collect_filtered_texts(tokenizer, [path])

        assert tokenizer.MAX_TEXT_LENGTH is None
        assert len(byte_long_text.encode(tokenizer.TEXT_ENCODING)) > 2000
        assert [byte_long_text, valid_text] == texts

    def test_filtered_text_iterable_normalizes_multiline_text_to_one_line(self):
        tokenizer = load_tokenizer_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample-0000.jsonl.zst"
            text = "word\n" * REPEATED_WORD_COUNT
            write_zst_rows(path, [{"text": text}])

            texts = collect_filtered_texts(tokenizer, [path])

        assert EXPECTED_NORMALIZED_TEXT_COUNT == len(texts)
        assert "\n" not in texts[0]
        assert len(texts[0]) >= tokenizer.MIN_TEXT_LENGTH

    def test_filtered_text_iterable_removes_null_characters(self):
        tokenizer = load_tokenizer_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample-0000.jsonl.zst"
            text = ("a" * 50) + "\x00" + ("b" * 50)
            write_zst_rows(path, [{"text": text}])

            texts = collect_filtered_texts(tokenizer, [path])

        assert EXPECTED_NORMALIZED_TEXT_COUNT == len(texts)
        assert "\x00" not in texts[0]

    def test_train_tokenizer_supplies_its_shard_regex_to_tokenizer_module(
        self, monkeypatch
    ):
        train_tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "sample-0000.jsonl.zst"
            expected_text = "a" * train_tokenizer.tokenizer.MIN_TEXT_LENGTH
            write_zst_rows(input_path, [{"text": expected_text}])
            discover_input_files = Spy(return_value=(input_path,))

            monkeypatch.setattr(train_tokenizer, "INPUT_DIR", root)
            monkeypatch.setattr(
                train_tokenizer.tokenizer,
                "discover_input_files",
                discover_input_files,
            )
            monkeypatch.setattr(
                train_tokenizer.spm.SentencePieceTrainer,
                "train",
                Spy(),
            )
            train_tokenizer.train_tokenizer()

        discover_input_files.assert_called_once_with(
            root,
            train_tokenizer.TOKENIZER_SHARD_NAME_REGEX,
        )

    def test_iter_zstd_jsonl_lines_reads_zst_files_without_subprocess(
        self, monkeypatch
    ):
        tokenizer = load_tokenizer_module()

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "sample-0000.jsonl.zst"
            text = "\n".join(
                json.dumps({"text": value}) for value in ("first", "second")
            )
            write_zst_text(path, text)

            monkeypatch.setattr(
                "subprocess.Popen",
                Spy(side_effect=AssertionError("zstd files must not shell out")),
            )
            lines = list(tokenizer.iter_zstd_jsonl_lines(path, max_rows_per_file=1))

        assert [(1, '{"text": "first"}\n')] == lines

    def test_train_tokenizer_enables_byte_fallback(self, monkeypatch):
        train_tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "sample-0000.jsonl.zst"
            write_zst_rows(
                input_path,
                [{"text": "a" * train_tokenizer.tokenizer.MIN_TEXT_LENGTH}],
            )

            train = Spy()
            monkeypatch.setattr(train_tokenizer, "INPUT_DIR", root)
            monkeypatch.setattr(train_tokenizer, "OUTPUT_DIR", root / "output")
            monkeypatch.setattr(
                train_tokenizer.spm.SentencePieceTrainer, "train", train
            )
            train_tokenizer.train_tokenizer()

        assert True is train.call_args.kwargs.get("byte_fallback")

    def test_train_tokenizer_uses_tokenizer_model_path(self, monkeypatch):
        train_tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "sample-0000.jsonl.zst"
            tokenizer_model_path = root / "nested" / "custom-tokenizer.model"
            write_zst_rows(
                input_path,
                [{"text": "a" * train_tokenizer.tokenizer.MIN_TEXT_LENGTH}],
            )

            train = Spy()
            monkeypatch.setattr(train_tokenizer, "INPUT_DIR", root)
            monkeypatch.setattr(
                train_tokenizer.spm.SentencePieceTrainer, "train", train
            )
            result = train_tokenizer.train_tokenizer(
                tokenizer_model_path=tokenizer_model_path
            )

        assert (
            str(tokenizer_model_path.with_suffix(""))
            == train.call_args.kwargs["model_prefix"]
        )
        assert tokenizer_model_path == result.model_path
        assert tokenizer_model_path.with_suffix(".vocab") == result.vocab_path

    def test_main_passes_tokenizer_model_path_to_train_tokenizer(self, monkeypatch):
        train_tokenizer = load_script()
        tokenizer_model_path = Path("/tmp/custom-tokenizer.model")
        train_tokenizer_spy = Spy(
            return_value=train_tokenizer.TrainingResult(
                model_path=tokenizer_model_path,
                vocab_path=tokenizer_model_path.with_suffix(".vocab"),
                input_file_count=1,
                rows_read=2,
                texts_used=1,
            )
        )
        monkeypatch.setattr(train_tokenizer, "train_tokenizer", train_tokenizer_spy)
        monkeypatch.setattr("builtins.print", Spy())

        return_code = train_tokenizer.main(
            ["--tokenizer-model", str(tokenizer_model_path)]
        )

        assert train_tokenizer.SUCCESS_RETURN_CODE == return_code
        train_tokenizer_spy.assert_called_once_with(
            tokenizer_model_path=tokenizer_model_path
        )

    def test_train_tokenizer_reserves_conversation_special_tokens(self, monkeypatch):
        train_tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "sample-0000.jsonl.zst"
            write_zst_rows(
                input_path,
                [{"text": "a" * train_tokenizer.tokenizer.MIN_TEXT_LENGTH}],
            )

            train = Spy()
            monkeypatch.setattr(train_tokenizer, "INPUT_DIR", root)
            monkeypatch.setattr(train_tokenizer, "OUTPUT_DIR", root / "output")
            monkeypatch.setattr(
                train_tokenizer.spm.SentencePieceTrainer, "train", train
            )
            train_tokenizer.train_tokenizer()

        assert list(train_tokenizer.tokenizer.CONVERSATION_SPECIAL_TOKENS) == (
            train.call_args.kwargs.get("user_defined_symbols")
        )

    def test_train_tokenizer_omits_max_sentence_length_when_unset(self, monkeypatch):
        train_tokenizer = load_script()

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            input_path = root / "sample-0000.jsonl.zst"
            write_zst_rows(
                input_path,
                [{"text": "a" * train_tokenizer.tokenizer.MIN_TEXT_LENGTH}],
            )

            train = Spy()
            monkeypatch.setattr(train_tokenizer, "INPUT_DIR", root)
            monkeypatch.setattr(train_tokenizer, "OUTPUT_DIR", root / "output")
            monkeypatch.setattr(
                train_tokenizer.spm.SentencePieceTrainer, "train", train
            )
            train_tokenizer.train_tokenizer()

        assert train_tokenizer.tokenizer.MAX_SENTENCE_LENGTH is None
        assert "max_sentence_length" not in train.call_args.kwargs
