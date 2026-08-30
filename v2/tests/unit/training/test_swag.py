from __future__ import annotations

import inspect
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from mlx.utils import tree_flatten, tree_map
from sml.artifacts.manifest import PayloadRef, TokenizerManifest, VerificationLevel
from sml.data import swag as swag_data
from sml.data.swag import (
    SwagBatch,
    SwagBatchEnvelope,
    SwagBatchStream,
    SwagCursor,
    prepare_swag_bundle,
)
from sml.errors import SMLConfigurationError
from sml.inference import ResolvedModel
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training import swag as swag_module
from sml.training.common import (
    AdamState,
    LoaderConfig,
    WeightDecayPolicy,
    build_weight_decay_tree,
    initialize_adam_state,
)
from sml.training.lora import LoRAConfig, LoRAInitializerConfig, apply_lora
from sml.training.swag import (
    SwagKernelConfig,
    SwagTrainingConfig,
    build_swag_kernels,
    default_swag_optimizer_config,
    score_candidates,
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


def source_contains(module: object, snippet: str) -> bool:
    return snippet in inspect.getsource(module)


def source_has_none_of(module: object, forbidden: list[str]) -> bool:
    source = inspect.getsource(module)
    return all(token not in source for token in forbidden)


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


def assert_tree_equal(actual: dict, expected: dict):
    actual_leaves = dict(tree_flatten(actual))
    expected_leaves = dict(tree_flatten(expected))
    assert set(actual_leaves) == set(expected_leaves)
    mx.eval(*actual_leaves.values(), *expected_leaves.values())
    for name, actual_leaf in actual_leaves.items():
        assert bool(mx.array_equal(actual_leaf, expected_leaves[name]).item()), name


def assert_tree_dtypes(tree: object, dtype: mx.Dtype):
    for name, leaf in tree_flatten(tree):
        assert leaf.dtype == dtype, name


def assert_builtin_array_tree(tree: object) -> None:
    def check(node: object) -> bool:
        if isinstance(node, mx.array):
            return True
        if isinstance(node, dict):
            return all(check(value) for value in node.values())
        if isinstance(node, (list, tuple)):
            return all(check(value) for value in node)
        return False

    assert check(tree)


def clone_tree(tree: dict) -> dict:
    return tree_map(lambda value: mx.array(value), tree)


def _example_mask_host(batch: SwagBatch) -> np.ndarray:
    return np.array(batch.example_mask, dtype=bool)


def _select_examples(batch: SwagBatch, mask: np.ndarray) -> SwagBatch:
    return SwagBatch(
        mx.array(np.array(batch.input_ids)[mask]),
        mx.array(np.array(batch.score_mask)[mask]),
        mx.array(np.array(batch.labels)[mask]),
        mx.array(np.array(batch.example_mask)[mask]),
        mx.array(np.array(batch.valid_token_mask)[mask]),
        batch.cursor_after,
    )


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


def _ranking_trainer(runtime: TinySwagRuntime, batch: SwagBatch):
    kernels = runtime.kernels(compiled=False)
    adapters = clone_tree(runtime.initial_adapters)
    trainer = swag_module.initial_swag_trainer_state(adapters, key=mx.random.key(11))
    trainer = kernels.ranking_microstep(
        adapters,
        runtime.frozen_base,
        trainer,
        batch,
    )
    mx.eval(trainer.to_tree())
    return trainer


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


def logsumexp_np(values: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(peak, axis=axis) + np.log(
        np.sum(np.exp(values - peak), axis=axis)
    )


def fixed_logits(*, dtype) -> mx.array:
    vocab = 8
    length = 6
    logits = mx.arange(4 * length * vocab, dtype=mx.float32).reshape(4, length, vocab)
    logits = (logits - mx.mean(logits)) / 10.0
    return logits.astype(dtype)


def score_fixture_with_unequal_lengths_and_eos() -> tuple[mx.array, mx.array]:
    pad = 3
    eos = 2
    input_ids = mx.array(
        [
            [1, 4, 5, eos, pad, pad],
            [1, 4, 6, 7, eos, pad],
            [1, 4, 0, eos, pad, pad],
            [1, 4, 5, 6, 7, eos],
        ],
        dtype=mx.int32,
    )
    score_mask = mx.array(
        [
            [False, False, True, True, False, False],
            [False, False, True, True, True, False],
            [False, False, True, True, False, False],
            [False, False, True, True, True, True],
        ]
    )
    return input_ids, score_mask


def direct_fp32_target_logit_minus_logsumexp_mean(
    logits: mx.array,
    input_ids: mx.array,
    score_mask: mx.array,
) -> mx.array:
    shift_logits = np.array(logits[..., :-1, :].astype(mx.float32), dtype=np.float64)
    targets = np.array(input_ids[..., 1:], dtype=np.int64)
    mask = np.array(score_mask[..., 1:], dtype=np.float64)
    target_logit = np.take_along_axis(shift_logits, targets[..., None], axis=-1)[..., 0]
    token_ll = target_logit - logsumexp_np(shift_logits, axis=-1)
    counted = np.maximum(mask.sum(axis=-1), 1.0)
    means = (token_ll * mask).sum(axis=-1) / counted
    return mx.array(means.astype(np.float32), dtype=mx.float32)


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
class TwoUpdateResult:
    frozen_base: dict
    adapters: dict
    optimizer: AdamState


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

    def _run(
        self,
        *,
        batch_size: int,
        compiled: bool,
        accumulation_steps: int | None = None,
        max_optimizer_steps: int | None = None,
    ) -> EpochResult | TwoUpdateResult:
        adapters = clone_tree(self.initial_adapters)
        optimizer = initialize_adam_state(adapters)
        trainer = swag_module.initial_swag_trainer_state(
            adapters, key=mx.random.key(11)
        )
        loader = replace(
            self.config.loader,
            microbatch_size=batch_size,
            prefetch_depth=2,
            gradient_accumulation_steps=(
                self.config.loader.gradient_accumulation_steps
                if accumulation_steps is None
                else accumulation_steps
            ),
        )
        config = replace(self.config, loader=loader, compile=compiled)
        kernels = build_swag_kernels(self.model, config, self.weight_decay_tree)
        start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
        real_examples = 0
        optimizer_steps = 0
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
                window_full = int(np.array(trainer.valid_count)) >= (
                    kernels.kernel_config.accumulation_steps
                )
                epoch_ended = cursor.epoch > start.epoch
                if window_full or epoch_ended:
                    adapters, optimizer, trainer = kernels.optimizer_step(
                        adapters,
                        optimizer,
                        trainer,
                    )
                    optimizer_steps += 1
                    if (
                        max_optimizer_steps is not None
                        and optimizer_steps >= max_optimizer_steps
                    ):
                        mx.eval(adapters, optimizer.to_tree(), self.frozen_base)
                        return TwoUpdateResult(
                            frozen_base=self.frozen_base,
                            adapters=adapters,
                            optimizer=optimizer,
                        )
                if epoch_ended:
                    break
        mx.eval(adapters)
        return EpochResult(
            real_examples=real_examples, adapters=adapters, cursor=cursor
        )

    def train_one_epoch(self, *, fixed_batch_size: int) -> EpochResult:
        result = self._run(batch_size=fixed_batch_size, compiled=False)
        assert isinstance(result, EpochResult)
        return result

    def train_one_epoch_unpadded_reference(self) -> EpochResult:
        return self.train_one_epoch(fixed_batch_size=1)

    def run_two_updates(self, *, compiled: bool) -> TwoUpdateResult:
        result = self._run(
            batch_size=1,
            compiled=compiled,
            accumulation_steps=1,
            max_optimizer_steps=2,
        )
        assert isinstance(result, TwoUpdateResult)
        return result


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


def test_candidate_score_is_fp32_mean_including_eos():
    logits = fixed_logits(dtype=mx.bfloat16)
    input_ids, score_mask = score_fixture_with_unequal_lengths_and_eos()
    scores = score_candidates(logits, input_ids, score_mask)
    expected = direct_fp32_target_logit_minus_logsumexp_mean(
        logits, input_ids, score_mask
    )
    mx.eval(scores, expected)
    assert scores.dtype == mx.float32
    assert_close(scores, expected, atol=1e-6, rtol=1e-6)


def test_padded_tail_matches_unpadded_example_weighted_update(tiny_swag_runtime):
    padded = tiny_swag_runtime.train_one_epoch(fixed_batch_size=4)
    eager = tiny_swag_runtime.train_one_epoch_unpadded_reference()
    assert padded.real_examples == eager.real_examples == 5
    assert_tree_close(padded.adapters, eager.adapters, atol=1e-5, rtol=1e-5)
    assert (
        padded.cursor
        == eager.cursor
        == SwagCursor(epoch=1, bucket_order_position=0, row_offset=0)
    )


def test_default_swag_optimizer_preserves_legacy_learning_rate_and_zero_adapter_decay():
    config = default_swag_optimizer_config()
    assert config.learning_rate == 1e-4
    assert config.schedule_steps == 8_192
    assert config.weight_decay.lora_a == 0.0
    assert config.weight_decay.lora_b == 0.0


def test_two_compiled_swag_updates_keep_base_bf16_and_all_optimizer_state_fp32(
    tiny_swag_runtime,
):
    before_base = tree_map(lambda value: mx.array(value), tiny_swag_runtime.frozen_base)
    eager = tiny_swag_runtime.run_two_updates(compiled=False)
    compiled = tiny_swag_runtime.run_two_updates(compiled=True)
    mx.eval(
        eager.adapters,
        eager.optimizer.to_tree(),
        compiled.frozen_base,
        compiled.adapters,
        compiled.optimizer.to_tree(),
        before_base,
    )
    assert_tree_dtypes(compiled.frozen_base, mx.bfloat16)
    assert_tree_equal(compiled.frozen_base, before_base)
    assert_tree_dtypes(compiled.adapters, mx.float32)
    assert_tree_dtypes(compiled.optimizer.first_moments, mx.float32)
    assert_tree_dtypes(compiled.optimizer.second_moments, mx.float32)
    assert_tree_close(compiled.adapters, eager.adapters, atol=1e-5, rtol=1e-5)


def test_swag_kernels_require_a_nonempty_static_lora_policy(tiny_swag_runtime):
    unadapted = SMLLanguageModel(tiny_model_config(), key=mx.random.key(17))

    with pytest.raises(SMLConfigurationError, match="nonempty LoRA forward policy"):
        build_swag_kernels(unadapted, tiny_swag_runtime.config, {})


@pytest.mark.parametrize("compiled", (False, True))
def test_swag_kernels_reject_lora_policy_reassignment_before_execution(
    tiny_swag_runtime,
    compiled: bool,
):
    kernels = tiny_swag_runtime.kernels(compiled=compiled)
    policy = tiny_swag_runtime.model.lora_forward_policy
    assert policy is not None
    tiny_swag_runtime.model.lora_forward_policy = type(policy)(policy.adapters)

    with pytest.raises(
        SMLConfigurationError, match="policy changed after kernel build"
    ):
        kernels.compiled_ranking_microstep_core(*tiny_swag_runtime.core_inputs())


def test_compiled_swag_cores_accept_only_builtin_array_trees(tiny_swag_runtime):
    kernels = tiny_swag_runtime.kernels(compiled=True)
    core_inputs = tiny_swag_runtime.core_inputs()
    assert_builtin_array_tree(core_inputs)
    adapter_paths = [path for path, _leaf in tree_flatten(core_inputs[0])]
    frozen_paths = [path for path, _leaf in tree_flatten(core_inputs[1])]
    assert all(
        path.rsplit(".", 1)[-1] in {"lora_a", "lora_b"} for path in adapter_paths
    )
    assert all(path.rsplit(".", 1)[-1] != "scale" for path in frozen_paths)
    core_outputs = kernels.compiled_ranking_microstep_core(*core_inputs)
    mx.eval(core_outputs)
    assert_builtin_array_tree(core_outputs)
    assert source_has_none_of(
        swag_module,
        [
            "mx.compile(lambda state",
            "mx.compile(lambda batch",
            "mx.compile(lambda config",
        ],
    )
    builder_source = inspect.getsource(swag_module.build_swag_kernels)
    assert "mx.random" not in builder_source
    assert "random." not in builder_source


def test_swag_value_and_grad_targets_adapters_only():
    assert source_contains(swag_module, "mx.value_and_grad(adapter_loss, argnums=0)")
    assert source_has_none_of(
        swag_module,
        ["mx.value_and_grad(combined_parameters", "nn.value_and_grad(model"],
    )


def test_swag_kernel_wrappers_do_not_eval_before_rebuilding_host_state():
    assert "mx.eval(" not in inspect.getsource(
        swag_module.SwagKernels.ranking_microstep
    )
    assert "mx.eval(" not in inspect.getsource(swag_module.SwagKernels.optimizer_step)


def test_swag_optimizer_splits_compiled_fp32_tree_from_host_reconstruction():
    host_source = inspect.getsource(swag_module.SwagKernels.optimizer_step)
    assert "adamw_fp32_update(" not in host_source
    assert "_adamw_fp32_update_tree" not in host_source
    assert "AdamState.from_tree(" not in host_source
    assert "AdamState.from_compiled_tree(" in host_source
    assert "SwagTrainerState.from_compiled_tree(" in host_source
    assert "optimizer.to_tree()" in host_source
    assert "_require_dtype" in host_source
    assert "mx.eval(" not in host_source
    assert "adamw_mixed_precision_update(" not in host_source
    builder_source = inspect.getsource(swag_module.build_swag_kernels)
    assert "_adamw_fp32_update_tree" in builder_source
    assert "normalize_and_clip(" in builder_source
    assert "AdamState" not in builder_source
    module_source = inspect.getsource(swag_module)
    assert "adamw_mixed_precision_update(" not in module_source
    assert ".astype(mx.bfloat16)" not in module_source


def test_compiled_swag_optimizer_core_consumes_builtin_adapter_and_adam_trees(
    tiny_swag_runtime,
):
    kernels = tiny_swag_runtime.kernels(compiled=True)
    ranking_inputs = tiny_swag_runtime.core_inputs()
    adapters = ranking_inputs[0]
    trainer_tree = kernels.compiled_ranking_microstep_core(*ranking_inputs)
    optimizer = initialize_adam_state(adapters)
    outputs = kernels.compiled_optimizer_step_core(
        adapters,
        optimizer.to_tree(),
        trainer_tree,
    )
    mx.eval(outputs)
    assert_builtin_array_tree(outputs)
    next_adapters, next_adam_tree, next_trainer_tree = outputs
    assert_tree_dtypes(next_adapters, mx.float32)
    assert_tree_dtypes(next_adam_tree[1], mx.float32)
    assert_tree_dtypes(next_adam_tree[2], mx.float32)
    changed = False
    for (_, before), (_, after) in zip(
        tree_flatten(adapters), tree_flatten(next_adapters), strict=True
    ):
        if not bool(mx.array_equal(before, after).item()):
            changed = True
            break
    assert changed
    assert int(next_trainer_tree[1].item()) == 0


def test_direct_swag_envelope_validates_numpy_arrays_and_stores_readonly_views():
    batch_size, candidate_count, length = 2, 4, 8
    input_ids = np.arange(
        batch_size * candidate_count * length, dtype=np.int32
    ).reshape(batch_size, candidate_count, length)
    score_mask = np.ones((batch_size, candidate_count, length), dtype=bool)
    labels = np.arange(batch_size, dtype=np.int32)
    example_mask = np.array([True, False])
    valid_token_mask = np.ones((batch_size, candidate_count, length), dtype=bool)
    cursor = SwagCursor(epoch=0, bucket_order_position=0, row_offset=1)
    envelope = SwagBatchEnvelope(
        input_ids,
        score_mask,
        labels,
        example_mask,
        valid_token_mask,
        cursor,
        source_epoch=0,
    )
    for array in (
        envelope.input_ids,
        envelope.score_mask,
        envelope.labels,
        envelope.example_mask,
        envelope.valid_token_mask,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            array[(0,) * array.ndim] = 1
    with pytest.raises(TypeError, match="input_ids"):
        SwagBatchEnvelope(
            input_ids.tolist(),
            score_mask,
            labels,
            example_mask,
            valid_token_mask,
            cursor,
            source_epoch=0,
        )
    with pytest.raises(ValueError, match="score_mask"):
        SwagBatchEnvelope(
            input_ids,
            score_mask.astype(np.int32),
            labels,
            example_mask,
            valid_token_mask,
            cursor,
            source_epoch=0,
        )


def test_swag_envelope_owns_storage_independent_of_caller_arrays():
    batch_size, candidate_count, length = 2, 4, 8
    input_ids = np.arange(
        batch_size * candidate_count * length, dtype=np.int32
    ).reshape(batch_size, candidate_count, length)
    score_mask = np.ones((batch_size, candidate_count, length), dtype=bool)
    labels = np.arange(batch_size, dtype=np.int32)
    example_mask = np.array([True, False])
    valid_token_mask = np.ones((batch_size, candidate_count, length), dtype=bool)
    cursor = SwagCursor(epoch=0, bucket_order_position=0, row_offset=1)
    envelope = SwagBatchEnvelope(
        input_ids,
        score_mask,
        labels,
        example_mask,
        valid_token_mask,
        cursor,
        source_epoch=0,
    )
    original_input = int(envelope.input_ids[0, 0, 0])
    original_label = int(envelope.labels[0])
    original_example = bool(envelope.example_mask[0])
    original_score = bool(envelope.score_mask[0, 0, 0])
    original_valid = bool(envelope.valid_token_mask[0, 0, 0])
    input_ids[0, 0, 0] = original_input + 7
    labels[0] = original_label + 3
    example_mask[0] = not original_example
    score_mask[0, 0, 0] = not original_score
    valid_token_mask[0, 0, 0] = not original_valid
    assert int(envelope.input_ids[0, 0, 0]) == original_input
    assert int(envelope.labels[0]) == original_label
    assert bool(envelope.example_mask[0]) is original_example
    assert bool(envelope.score_mask[0, 0, 0]) is original_score
    assert bool(envelope.valid_token_mask[0, 0, 0]) is original_valid


def test_swag_producer_transfers_owned_envelope_arrays_without_recopying(
    tiny_swag_runtime, monkeypatch
):
    captured: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = []
    original_assemble = swag_data._assemble_batch_arrays

    def capture_assemble(*args, **kwargs):
        arrays = original_assemble(*args, **kwargs)
        captured.append(arrays)
        return arrays

    monkeypatch.setattr(swag_data, "_assemble_batch_arrays", capture_assemble)
    loader = replace(
        tiny_swag_runtime.config.loader,
        microbatch_size=2,
        prefetch_depth=1,
    )
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=start
    ) as stream:
        envelope = next(iter(stream))
    assert captured
    assembled = captured[0]
    pairs = (
        (assembled[0], envelope.input_ids),
        (assembled[1], envelope.valid_token_mask),
        (assembled[2], envelope.score_mask),
        (assembled[3], envelope.labels),
        (assembled[4], envelope.example_mask),
    )
    for owned, stored in pairs:
        assert stored.ctypes.data == owned.ctypes.data
        assert np.shares_memory(stored, owned)
        assert not stored.flags.writeable
        assert not owned.flags.writeable
        assert owned.flags.owndata


def test_assemble_batch_arrays_fills_staging_without_extra_copies(monkeypatch):
    length = 4
    bucket = swag_data.SwagBucket(
        length=length,
        input_ids=np.arange(2 * 4 * length, dtype=np.int32).reshape(2, 4, length),
        valid_token_mask=np.ones((2, 4, length), dtype=bool),
        score_mask=np.ones((2, 4, length), dtype=bool),
        labels=np.array([1, 3], dtype=np.int32),
    )
    manifest = SimpleNamespace(pad_token_id=3, bos_token_id=1, eos_token_id=2)
    staging: list[np.ndarray] = []
    original_empty = swag_data.np.empty
    original_zeros = swag_data.np.zeros
    original_array = swag_data.np.array
    original_owned = swag_data._owned_readonly
    copy_true_calls: list[object] = []
    owned_calls: list[int] = []

    def tracking_empty(*args, **kwargs):
        array = original_empty(*args, **kwargs)
        staging.append(array)
        return array

    def tracking_zeros(*args, **kwargs):
        array = original_zeros(*args, **kwargs)
        staging.append(array)
        return array

    def tracking_array(*args, **kwargs):
        if kwargs.get("copy") is True:
            copy_true_calls.append(args[0] if args else None)
        return original_array(*args, **kwargs)

    def tracking_owned(array):
        owned_calls.append(id(array))
        return original_owned(array)

    monkeypatch.setattr(swag_data.np, "empty", tracking_empty)
    monkeypatch.setattr(swag_data.np, "zeros", tracking_zeros)
    monkeypatch.setattr(swag_data.np, "array", tracking_array)
    monkeypatch.setattr(swag_data, "_owned_readonly", tracking_owned)

    assembled = swag_data._assemble_batch_arrays(
        bucket,
        (0,),
        batch_size=2,
        manifest=manifest,
    )
    assert copy_true_calls == []
    assert owned_calls == []
    for array in assembled:
        assert any(array.ctypes.data == staged.ctypes.data for staged in staging)
        assert not array.flags.writeable
        assert array.flags.owndata
    np.testing.assert_array_equal(assembled[0][0], bucket.input_ids[0])
    assert int(assembled[3][0]) == 1
    assert bool(assembled[4][0]) is True
    assert bool(assembled[4][1]) is False

    input_ids, valid_token_mask, score_mask, labels, example_mask = assembled
    cursor = SwagCursor(epoch=0, bucket_order_position=0, row_offset=1)
    envelope = SwagBatchEnvelope._owned(
        input_ids,
        score_mask,
        labels,
        example_mask,
        valid_token_mask,
        cursor,
        source_epoch=0,
    )
    pairs = (
        (input_ids, envelope.input_ids),
        (valid_token_mask, envelope.valid_token_mask),
        (score_mask, envelope.score_mask),
        (labels, envelope.labels),
        (example_mask, envelope.example_mask),
    )
    for owned, stored in pairs:
        assert stored.ctypes.data == owned.ctypes.data
        assert not stored.flags.writeable

    caller_ids = np.arange(2 * 4 * length, dtype=np.int32).reshape(2, 4, length)
    caller_score = np.ones((2, 4, length), dtype=bool)
    caller_labels = np.arange(2, dtype=np.int32)
    caller_example = np.array([True, False])
    caller_valid = np.ones((2, 4, length), dtype=bool)
    public = SwagBatchEnvelope(
        caller_ids,
        caller_score,
        caller_labels,
        caller_example,
        caller_valid,
        cursor,
        source_epoch=0,
    )
    original_input = int(public.input_ids[0, 0, 0])
    caller_ids[0, 0, 0] = original_input + 7
    assert int(public.input_ids[0, 0, 0]) == original_input
    assert public.input_ids.ctypes.data != caller_ids.ctypes.data


def test_per_slot_ranking_losses_are_finite_before_masking(tiny_swag_runtime):
    padded = _first_padded_batch(tiny_swag_runtime)
    real_mask = _example_mask_host(padded)
    assert real_mask.any()
    assert (~real_mask).any()
    unpadded = _select_examples(padded, real_mask)
    padded_trainer = _ranking_trainer(tiny_swag_runtime, padded)
    reference_trainer = _ranking_trainer(tiny_swag_runtime, unpadded)
    assert bool(mx.isfinite(padded_trainer.loss_numerator).item())
    assert int(padded_trainer.valid_count) == int(real_mask.sum())
    assert int(padded_trainer.valid_count) == int(reference_trainer.valid_count)
    assert int(padded_trainer.correct_count) == int(reference_trainer.correct_count)
    assert_close(
        padded_trainer.loss_numerator,
        reference_trainer.loss_numerator,
        atol=1e-5,
        rtol=1e-5,
    )
    assert_tree_close(
        padded_trainer.accumulators,
        reference_trainer.accumulators,
        atol=1e-5,
        rtol=1e-5,
    )


def test_synthetic_slots_have_no_loss_accuracy_gradient_progress_or_cursor_contribution(
    tiny_swag_runtime,
):
    loader = replace(
        tiny_swag_runtime.config.loader,
        microbatch_size=4,
        prefetch_depth=1,
    )
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=start
    ) as stream:
        batches = [SwagBatch.from_envelope(envelope) for envelope in stream]
    assert any(not bool(np.array(batch.example_mask).all()) for batch in batches)
    tail = next(
        batch for batch in batches if not bool(np.array(batch.example_mask).all())
    )
    pad_slots = ~_example_mask_host(tail)
    real_slots = _example_mask_host(tail)
    assert pad_slots.any()
    assert bool(np.array(tail.valid_token_mask)[pad_slots].any())
    assert bool(np.array(tail.score_mask)[pad_slots].any())
    real_seen = sum(int(np.array(batch.example_mask).sum()) for batch in batches)
    assert real_seen == 5
    assert batches[-1].cursor_after == SwagCursor(
        epoch=1, bucket_order_position=0, row_offset=0
    )

    unpadded = _select_examples(tail, real_slots)
    synthetic = _select_examples(tail, pad_slots)
    padded_trainer = _ranking_trainer(tiny_swag_runtime, tail)
    reference_trainer = _ranking_trainer(tiny_swag_runtime, unpadded)
    synthetic_trainer = _ranking_trainer(tiny_swag_runtime, synthetic)
    assert int(synthetic_trainer.valid_count) == 0
    assert int(synthetic_trainer.correct_count) == 0
    assert float(np.array(synthetic_trainer.loss_numerator)) == 0.0
    for _name, leaf in tree_flatten(synthetic_trainer.accumulators):
        assert bool(mx.allclose(leaf, mx.zeros_like(leaf)).item()), _name
    assert int(padded_trainer.valid_count) == int(reference_trainer.valid_count)
    assert int(padded_trainer.correct_count) == int(reference_trainer.correct_count)
    assert_close(
        padded_trainer.loss_numerator,
        reference_trainer.loss_numerator,
        atol=1e-5,
        rtol=1e-5,
    )
    assert_tree_close(
        padded_trainer.accumulators,
        reference_trainer.accumulators,
        atol=1e-5,
        rtol=1e-5,
    )


def test_full_prefetch_queue_owns_distinct_readonly_storage_and_cannot_advance_committed_cursor(
    tiny_swag_runtime,
):
    loader = replace(
        tiny_swag_runtime.config.loader,
        microbatch_size=1,
        prefetch_depth=3,
    )
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=start
    ) as stream:
        iterator = iter(stream)
        envelopes: list[SwagBatchEnvelope] = [next(iterator)]
        while len(envelopes) < loader.prefetch_depth:
            envelopes.append(next(iterator))
        assert stream.committed_cursor == start
        pointers = []
        for envelope in envelopes:
            for array in (
                envelope.input_ids,
                envelope.score_mask,
                envelope.labels,
                envelope.example_mask,
                envelope.valid_token_mask,
            ):
                assert not array.flags.writeable
                pointers.append(array.__array_interface__["data"][0])
        assert len(set(pointers)) == len(pointers)
        for envelope in envelopes:
            envelope.release()


def test_bucket_tail_compile_shape_is_fixed(tiny_swag_runtime):
    loader = replace(
        tiny_swag_runtime.config.loader,
        microbatch_size=4,
        prefetch_depth=1,
    )
    shapes = []
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=start
    ) as stream:
        for envelope in stream:
            batch = SwagBatch.from_envelope(envelope)
            shapes.append(tuple(int(dimension) for dimension in batch.input_ids.shape))
            stream.commit(batch.cursor_after)
    assert shapes
    assert all(shape[0] == 4 for shape in shapes)
    assert len({shape[-1] for shape in shapes}) == 1


def test_epoch_boundary_commits_with_update(tiny_swag_runtime):
    result = tiny_swag_runtime.train_one_epoch(fixed_batch_size=4)
    assert result.cursor == SwagCursor(epoch=1, bucket_order_position=0, row_offset=0)
    assert result.real_examples == 5
    initial = dict(tree_flatten(tiny_swag_runtime.initial_adapters))
    updated = dict(tree_flatten(result.adapters))
    changed = False
    for name, leaf in updated.items():
        mx.eval(leaf, initial[name])
        changed = changed or (not bool(mx.array_equal(leaf, initial[name]).item()))
    assert changed


def test_second_compiled_step_sees_returned_adapter_optimizer_and_key_state(
    tiny_swag_runtime,
):
    model = SMLLanguageModel(
        tiny_model_config(hidden_dropout=0.1), key=mx.random.key(3)
    )
    apply_lora(
        model,
        LoRAConfig(
            rank=2,
            alpha=1.0,
            scaling_mode="lora",
            dropout=0.5,
            target_modules=("q_proj", "v_proj", "down_proj"),
            initializer=LoRAInitializerConfig(lora_a=0.05, lora_b=0.05),
        ),
        key=mx.random.key(5),
    )
    adapters, frozen_base = split_adapter_parameters(model.parameters())
    mx.eval(adapters, frozen_base)
    before_base = clone_tree(frozen_base)
    runtime = TinySwagRuntime(
        bundle=tiny_swag_runtime.bundle,
        model=model,
        frozen_base=frozen_base,
        initial_adapters=clone_tree(adapters),
        weight_decay_tree=build_weight_decay_tree(adapters, WeightDecayPolicy()),
        config=tiny_swag_runtime.config,
    )

    def run_two_steps(*, compiled: bool):
        kernels = runtime.kernels(compiled=compiled)
        adapters = clone_tree(runtime.initial_adapters)
        optimizer = initialize_adam_state(adapters)
        key0 = mx.random.key(11)
        trainer = swag_module.initial_swag_trainer_state(adapters, key=key0)
        loader = replace(
            runtime.config.loader,
            microbatch_size=1,
            prefetch_depth=2,
            gradient_accumulation_steps=1,
        )
        keys = [key0]
        start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
        with SwagBatchStream._borrowing_bundle(
            runtime.bundle, loader, cursor=start
        ) as stream:
            iterator = iter(stream)
            for _step in range(2):
                batch = SwagBatch.from_envelope(next(iterator))
                trainer = kernels.ranking_microstep(
                    adapters,
                    runtime.frozen_base,
                    trainer,
                    batch,
                )
                keys.append(trainer.next_key)
                adapters, optimizer, trainer = kernels.optimizer_step(
                    adapters,
                    optimizer,
                    trainer,
                )
                stream.commit(batch.cursor_after)
        mx.eval(adapters, optimizer.to_tree(), trainer.to_tree(), *keys)
        return adapters, optimizer, trainer, keys

    compiled_adapters, compiled_optimizer, compiled_trainer, compiled_keys = (
        run_two_steps(compiled=True)
    )
    eager_adapters, eager_optimizer, eager_trainer, eager_keys = run_two_steps(
        compiled=False
    )
    expected_keys = [eager_keys[0]]
    for _microstep in range(2):
        terminal = expected_keys[-1]
        for _site in range(8):
            terminal, _unused = mx.random.split(terminal)
        expected_keys.append(terminal)
    mx.eval(
        compiled_adapters,
        compiled_optimizer.to_tree(),
        compiled_trainer.to_tree(),
        eager_adapters,
        eager_optimizer.to_tree(),
        eager_trainer.to_tree(),
        frozen_base,
        before_base,
        *compiled_keys,
        *eager_keys,
        *expected_keys,
    )
    assert int(np.array(compiled_optimizer.step)) == 2
    assert int(np.array(eager_optimizer.step)) == 2
    assert bool(mx.array_equal(compiled_keys[0], eager_keys[0]).item())
    assert bool(mx.array_equal(eager_keys[1], expected_keys[1]).item())
    assert bool(mx.array_equal(eager_keys[2], expected_keys[2]).item())
    assert bool(mx.array_equal(compiled_keys[1], expected_keys[1]).item())
    assert bool(mx.array_equal(compiled_keys[2], expected_keys[2]).item())
    assert bool(mx.array_equal(compiled_keys[1], eager_keys[1]).item())
    assert bool(mx.array_equal(compiled_keys[2], eager_keys[2]).item())
    assert_tree_dtypes(frozen_base, mx.bfloat16)
    assert_tree_equal(frozen_base, before_base)
    assert_tree_close(compiled_adapters, eager_adapters, atol=1e-5, rtol=1e-5)
    assert_tree_close(
        compiled_optimizer.to_tree(), eager_optimizer.to_tree(), atol=1e-5, rtol=1e-5
    )
    assert_tree_close(
        compiled_trainer.to_tree(), eager_trainer.to_tree(), atol=1e-5, rtol=1e-5
    )
    compiled_trainer_tree = compiled_trainer.to_tree()
    eager_trainer_tree = eager_trainer.to_tree()
    for index in (1, 2, 4):
        assert bool(
            mx.array_equal(
                compiled_trainer_tree[index], eager_trainer_tree[index]
            ).item()
        )


def test_kernel_config_is_frozen_from_loader_optimizer_and_compile(tiny_swag_runtime):
    kernels = tiny_swag_runtime.kernels(compiled=True)
    config = kernels.kernel_config
    assert isinstance(config, SwagKernelConfig)
    assert (
        config.accumulation_steps
        == tiny_swag_runtime.config.loader.gradient_accumulation_steps
    )
    assert (
        config.gradient_clip_norm
        == tiny_swag_runtime.config.optimizer.gradient_clip_norm
    )
    assert config.compile is True


def test_permutation_uses_loader_epoch_seed_not_training_seed():
    stream_source = inspect.getsource(inspect.getmodule(SwagBatchStream))
    assert "PCG64" in stream_source
    assert "SeedSequence" in stream_source
    assert "epoch_seed" in stream_source
    assert "sml.training" not in stream_source


def test_swag_batch_stream_accepts_structural_loader(tiny_swag_runtime):
    loader = SimpleNamespace(prefetch_depth=1, microbatch_size=2, epoch_seed=42)
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle, loader, cursor=start
    ) as stream:
        envelope = next(iter(stream))
        assert envelope.input_ids.shape[0] == 2


def test_swag_batch_stream_rejects_loader_missing_required_attributes(
    tiny_swag_runtime,
):
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    with pytest.raises(TypeError, match="prefetch_depth"):
        SwagBatchStream._borrowing_bundle(
            tiny_swag_runtime.bundle, object(), cursor=start
        )


def test_swag_batch_stream_rejects_invalid_loader_policy_types_and_ranges(
    tiny_swag_runtime,
):
    start = SwagCursor(epoch=0, bucket_order_position=0, row_offset=0)
    base = {"prefetch_depth": 1, "microbatch_size": 2, "epoch_seed": 42}
    cases = (
        ({"prefetch_depth": 0}, ValueError, "prefetch_depth"),
        ({"prefetch_depth": 1.5}, TypeError, "prefetch_depth"),
        ({"microbatch_size": 0}, ValueError, "microbatch_size"),
        ({"microbatch_size": True}, TypeError, "microbatch_size"),
        ({"epoch_seed": -1}, ValueError, "epoch_seed"),
        ({"epoch_seed": 2**32}, ValueError, "epoch_seed"),
        ({"epoch_seed": True}, TypeError, "epoch_seed"),
    )
    for override, error, match in cases:
        loader = SimpleNamespace(**{**base, **override})
        with pytest.raises(error, match=match):
            SwagBatchStream._borrowing_bundle(
                tiny_swag_runtime.bundle, loader, cursor=start
            )
    with SwagBatchStream._borrowing_bundle(
        tiny_swag_runtime.bundle,
        SimpleNamespace(prefetch_depth=1, microbatch_size=2, epoch_seed=0),
        cursor=start,
    ) as stream:
        envelope = next(iter(stream))
        assert envelope.input_ids.shape[0] == 2
