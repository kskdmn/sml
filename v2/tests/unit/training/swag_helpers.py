from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map
from sml.artifacts.manifest import PayloadRef, TokenizerManifest, VerificationLevel
from sml.data.swag import (
    SwagBatch,
    SwagBatchStream,
    SwagCursor,
    prepare_swag_bundle,
)
from sml.inference import ResolvedModel
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training import swag as swag_module
from sml.training.common import (
    LoaderConfig,
    WeightDecayPolicy,
    build_weight_decay_tree,
    initialize_adam_state,
)
from sml.training.lora import LoRAConfig, LoRAInitializerConfig, apply_lora
from sml.training.swag import (
    SwagTrainingConfig,
    build_swag_kernels,
    default_swag_optimizer_config,
)

IDENTITY_A = "sha256:" + "a" * 64


IDENTITY_B = "sha256:" + "b" * 64


IDENTITY_C = "sha256:" + "c" * 64


VALID_ROW: dict[str, object] = {
    "context": "the cat sat",
    "endings": ("on the mat", "in the car", "by the door", "near a tree"),
    "label": 1,
}


class RecordingProcessor:
    def encode(self, text: str | Sequence[str]) -> list[int] | list[list[int]]:
        if isinstance(text, str):
            return self._encode_one(text)
        return [self._encode_one(item) for item in text]

    @staticmethod
    def _encode_one(text: str) -> list[int]:
        words = text.split()
        if not words:
            return []
        return [10 + (index % 20) for index, _word in enumerate(words)]


class RecordingTokenizer:
    def __init__(self, manifest: TokenizerManifest, processor: object) -> None:
        self.path = Path("recording-tokenizer")
        self.manifest = manifest
        self.verification = VerificationLevel.FULL
        self.processor = processor


class FakeSwagProvider:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self.rows = rows

    def resolve(self, source):
        from sml.data.swag import ResolvedSwagSource

        return ResolvedSwagSource(
            backend=source.backend,
            namespace=source.namespace,
            name=source.name,
            dataset_config=source.dataset_config,
            revision=source.revision,
            split=source.split,
            commit="abc123def456",
            provider_fingerprint="fingerprint-v1",
            provider_package="datasets",
            provider_version="2.0.0",
        )

    def iter_rows(self, resolved) -> Iterator[Mapping[str, object]]:
        yield from self.rows


def tokenizer_manifest() -> TokenizerManifest:
    return TokenizerManifest(
        kind="tokenizer",
        version=1,
        identity=IDENTITY_A,
        algorithm="sentencepiece-bpe-v1",
        training={"normalization": "nmt_nfkc"},
        vocab_size=64,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=3,
        unk_token_id=0,
        model=PayloadRef("tokenizer.model", IDENTITY_A, 10),
        vocab=PayloadRef("tokenizer.vocab", IDENTITY_B, 20),
        diagnostic_source_locator="/source/tokenizer",
    )


def tiny_model_config(*, hidden_dropout: float = 0.0) -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        intermediate_size=32,
        original_context_length=32,
        rope_scaling_factor=1.0,
        hidden_dropout=hidden_dropout,
    )


def tiny_base_model() -> ResolvedModel:
    return ResolvedModel(
        artifact_kind="pretraining-checkpoint",
        artifact_identity=IDENTITY_B,
        run_identity=IDENTITY_B,
        step=1,
        checkpoint_identity=IDENTITY_C,
        run_step_identity=IDENTITY_A,
        verification=VerificationLevel.FULL,
        model_config=tiny_model_config(),
        tokenizer=RecordingTokenizer(tokenizer_manifest(), RecordingProcessor()),
        model_arrays={},
    )


def tiny_swag_config(provider):
    from sml.data.swag import SwagPreparationConfig, SwagSourceConfig

    return SwagPreparationConfig(
        provider=provider,
        source=SwagSourceConfig(revision="deadbeef" * 5),
        maximum_length=32,
        bucket_boundaries=(16, 32),
    )


def assert_close(actual: mx.array, expected: mx.array, *, atol: float, rtol: float):
    mx.eval(actual, expected)
    assert bool(mx.allclose(actual, expected, atol=atol, rtol=rtol).item())


def assert_tree_close(actual: object, expected: object, *, atol: float, rtol: float):
    actual_leaves = dict(tree_flatten(actual))
    expected_leaves = dict(tree_flatten(expected))
    assert set(actual_leaves) == set(expected_leaves)
    for name, actual_leaf in actual_leaves.items():
        assert_close(
            actual_leaf,
            expected_leaves[name],
            atol=atol,
            rtol=rtol,
        )


def clone_tree(tree: dict) -> dict:
    return tree_map(lambda value: mx.array(value), tree)


def _first_padded_batch(runtime: TinySwagRuntime) -> SwagBatch:
    loader = replace(
        runtime.config.loader,
        microbatch_size=4,
        prefetch_depth=1,
    )
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        runtime.bundle, loader, cursor=start
    ) as stream:
        for envelope in stream:
            batch = SwagBatch.from_envelope(envelope)
            if not bool(np.array(batch.example_mask).all()):
                return batch
    raise AssertionError("expected a padded synthetic tail batch")


def split_adapter_parameters(parameters: object) -> tuple[object, object]:
    if isinstance(parameters, dict):
        adapters: dict = {}
        frozen: dict = {}
        for key, value in parameters.items():
            if key in {"lora_a", "lora_b"}:
                adapters[key] = value
            elif isinstance(value, (dict, list, tuple)):
                nested_adapters, nested_frozen = split_adapter_parameters(value)
                if nested_adapters:
                    adapters[key] = nested_adapters
                if nested_frozen:
                    frozen[key] = nested_frozen
            else:
                frozen[key] = value
        return adapters, frozen
    if isinstance(parameters, list):
        adapter_items = []
        frozen_items = []
        has_adapters = False
        has_frozen = False
        for value in parameters:
            nested_adapters, nested_frozen = split_adapter_parameters(value)
            adapter_items.append(nested_adapters)
            frozen_items.append(nested_frozen)
            has_adapters = has_adapters or bool(nested_adapters)
            has_frozen = has_frozen or bool(nested_frozen)
        return (
            adapter_items if has_adapters else {},
            frozen_items if has_frozen else {},
        )
    return {}, parameters


def five_swag_rows() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "context": f"the cat sat {index}",
            "endings": VALID_ROW["endings"],
            "label": index % 4,
        }
        for index in range(5)
    )


def tiny_language_model(*, hidden_dropout: float = 0.0) -> SMLLanguageModel:
    model = SMLLanguageModel(
        tiny_model_config(hidden_dropout=hidden_dropout),
        key=mx.random.key(3),
    )
    mx.eval(model.parameters())
    return apply_lora(
        model,
        LoRAConfig(
            rank=2,
            alpha=4.0,
            scaling_mode="lora",
            dropout=0.0,
            target_modules=("q_proj", "v_proj"),
            initializer=LoRAInitializerConfig(lora_a=0.01, lora_b=0.0),
        ),
        key=mx.random.key(6),
    )


@dataclass
class EpochResult:
    real_examples: int
    adapters: dict
    cursor: SwagCursor


@dataclass
class TinySwagRuntime:
    bundle: object
    model: SMLLanguageModel
    frozen_base: dict
    initial_adapters: dict
    weight_decay_tree: dict
    config: SwagTrainingConfig

    def kernels(self, *, compiled: bool):
        return build_swag_kernels(
            self.model,
            replace(self.config, compile=compiled),
            self.weight_decay_tree,
        )

    def core_inputs(self) -> tuple:
        loader = replace(self.config.loader, microbatch_size=2, prefetch_depth=1)
        cursor = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
        with SwagBatchStream._borrowing_bundle(
            self.bundle, loader, cursor=cursor
        ) as stream:
            envelope = next(iter(stream))
            batch = SwagBatch.from_envelope(envelope)
        adapters = clone_tree(self.initial_adapters)
        trainer = swag_module.initial_swag_trainer_state(
            adapters, key=mx.random.key(11)
        )
        return (
            adapters,
            self.frozen_base,
            trainer.to_tree(),
            batch.input_ids,
            batch.score_mask,
            batch.labels,
            batch.example_mask,
            batch.valid_token_mask,
        )

    def train_one_epoch(self, *, fixed_batch_size: int) -> EpochResult:
        adapters = clone_tree(self.initial_adapters)
        optimizer = initialize_adam_state(adapters)
        trainer = swag_module.initial_swag_trainer_state(
            adapters, key=mx.random.key(11)
        )
        loader = replace(
            self.config.loader,
            microbatch_size=fixed_batch_size,
            prefetch_depth=2,
        )
        config = replace(self.config, loader=loader, compile=False)
        kernels = build_swag_kernels(self.model, config, self.weight_decay_tree)
        start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
        real_examples = 0
        window_microsteps = 0
        cursor = start
        with SwagBatchStream._borrowing_bundle(
            self.bundle, loader, cursor=start
        ) as stream:
            for envelope in stream:
                batch = SwagBatch.from_envelope(envelope)
                real_examples += int(
                    np.array(batch.example_mask).astype(np.int32).sum()
                )
                trainer = kernels.ranking_microstep(
                    adapters,
                    self.frozen_base,
                    trainer,
                    batch,
                )
                stream.commit(batch.cursor_after)
                cursor = batch.cursor_after
                window_microsteps += 1
                window_full = window_microsteps >= (
                    kernels.kernel_config.accumulation_steps
                )
                epoch_ended = cursor.epoch > start.epoch
                if window_full or epoch_ended:
                    adapters, optimizer, trainer = kernels.optimizer_step(
                        adapters,
                        optimizer,
                        trainer,
                    )
                    window_microsteps = 0
                if epoch_ended:
                    break
        mx.eval(adapters)
        return EpochResult(
            real_examples=real_examples, adapters=adapters, cursor=cursor
        )


@pytest.fixture
def tiny_swag_runtime(tmp_path: Path) -> Iterator[TinySwagRuntime]:
    bundle = prepare_swag_bundle(
        tiny_swag_config(FakeSwagProvider(five_swag_rows())),
        tiny_base_model(),
        tmp_path / "swag",
    )
    model = tiny_language_model()
    adapters, frozen_base = split_adapter_parameters(model.parameters())
    mx.eval(adapters, frozen_base)
    config = SwagTrainingConfig(
        base_checkpoint=tmp_path / "unused-base",
        data=tmp_path / "unused-data",
        output_run=tmp_path / "unused-run",
        loader=LoaderConfig(
            microbatch_size=4,
            gradient_accumulation_steps=16,
            prefetch_depth=2,
            epoch_seed=42,
        ),
        optimizer=default_swag_optimizer_config(),
        compile=False,
        seed=7,
    )
    runtime = TinySwagRuntime(
        bundle=bundle,
        model=model,
        frozen_base=frozen_base,
        initial_adapters=clone_tree(adapters),
        weight_decay_tree=build_weight_decay_tree(adapters, WeightDecayPolicy()),
        config=config,
    )
    try:
        yield runtime
    finally:
        bundle.close()
