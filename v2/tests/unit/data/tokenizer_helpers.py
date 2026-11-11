from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from sml.data.corpus import CorpusConfig
from sml.data.tokenizer import (
    TokenizerTrainingConfig,
)


class _Processor:
    def __init__(
        self,
        *,
        pieces: tuple[str, ...],
        scores: tuple[float, ...] | None = None,
        ids: tuple[int, int, int, int] = (1, 2, 3, 0),
    ) -> None:
        self._pieces = pieces
        self._scores = scores or tuple(float(-index) for index in range(len(pieces)))
        self._ids = ids

    def get_piece_size(self):
        return len(self._pieces)

    def id_to_piece(self, index):
        return self._pieces[index]

    def get_score(self, index):
        return self._scores[index]

    def bos_id(self):
        return self._ids[0]

    def eos_id(self):
        return self._ids[1]

    def pad_id(self):
        return self._ids[2]

    def unk_id(self):
        return self._ids[3]


def _install_fake_sentencepiece(
    monkeypatch, *, pieces=None, scores=None, ids=(1, 2, 3, 0)
):
    pieces = pieces or ("<unk>", "<s>", "</s>", "<pad>", "piece")
    calls = []

    def train(**kwargs):
        calls.append(kwargs)
        list(kwargs["sentence_iterator"])
        kwargs["model_writer"].write(b"model")

    module = SimpleNamespace(
        SentencePieceTrainer=SimpleNamespace(train=train),
        SentencePieceProcessor=lambda **_kwargs: _Processor(
            pieces=pieces,
            scores=scores,
            ids=ids,
        ),
    )
    monkeypatch.setitem(sys.modules, "sentencepiece", module)
    return calls


def _config(tmp_path: Path, **overrides) -> TokenizerTrainingConfig:
    values = {
        "corpus": CorpusConfig(input_root=tmp_path, min_text_bytes=1),
        "vocab_size": 5,
        "hard_vocab_limit": False,
    }
    values.update(overrides)
    return TokenizerTrainingConfig(**values)
