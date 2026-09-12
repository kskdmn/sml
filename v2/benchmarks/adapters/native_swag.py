"""Canonical SWAG trials over production batch assembly and adapter kernels."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from sml.artifacts.manifest import SwagDataManifest, canonical_json_bytes
from sml.data.swag import (
    SwagBatch,
    SwagBatchEnvelope,
    SwagBucket,
    SwagCursor,
    _assemble_batch_arrays,
    _validate_bucket_arrays,
    _write_npy,
)
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.common import (
    LoaderConfig,
    OptimizerConfig,
    WeightDecayPolicy,
    build_weight_decay_tree,
    initialize_adam_state,
)
from sml.training.lora import LoRAConfig, apply_lora, split_adapter_parameters
from sml.training.random import counter_random_key
from sml.training.swag import (
    SwagTrainerState,
    SwagTrainingConfig,
    _merge_adapter_parameters,
    build_swag_kernels,
    initial_swag_trainer_state,
)

from v2.benchmarks.adapters.execution_order import ObservedBatch, verify_input_batches
from v2.benchmarks.parameters import initialize_parameters
from v2.benchmarks.schema import CanonicalWorkload
from v2.benchmarks.workload import fixed_swag_examples, semantic_array_identity


class SwagRuntime:
    def __init__(self, workload: CanonicalWorkload, directory: Path) -> None:
        model_config = ModelConfig(**workload.model)
        optimizer = workload.optimizer["swag"]
        loader = workload.loader["swag"]
        expected_loader = {
            "dataset_name": "allenai/swag",
            "dataset_config": "regular",
            "dataset_split": "train",
            "dataset_revision": "benchmark-fixed-examples-v1",
            "cache_format": "npz-int32-v1",
            "shuffle_examples": False,
        }
        if set(loader) != {
            *expected_loader,
            "sequence_length",
            "example_count",
            "batch_size",
        }:
            raise ValueError("unsupported canonical SWAG loader fields")
        for name in ("sequence_length", "example_count", "batch_size"):
            if type(loader[name]) is not int or loader[name] <= 0:
                raise ValueError(f"SWAG loader {name} must be a positive integer")
        for name, expected in expected_loader.items():
            if type(loader[name]) is not type(expected) or loader[name] != expected:
                raise ValueError(f"unsupported canonical SWAG loader field: {name}")
        if model_config.rope_scaling_factor != 1.0:
            raise ValueError("SWAG benchmark requires unscaled RoPE")
        if optimizer["optimizer_weight_decay"] != 0.0:
            raise ValueError("SWAG benchmark requires the canonical zero AdamW decay")
        if set(optimizer) != {
            "betas",
            "epsilon",
            "bias_correction",
            "optimizer_weight_decay",
            "learning_rate",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "warmup_steps",
            "total_steps",
            "minimum_learning_rate_ratio",
            "seed",
            "lora",
            "parameter_weight_decay",
        }:
            raise ValueError("unsupported canonical SWAG optimizer fields")
        if (
            not isinstance(optimizer["betas"], (list, tuple))
            or len(optimizer["betas"]) != 2
        ):
            raise ValueError("SWAG optimizer betas must contain exactly two values")
        self.seed = optimizer["seed"]
        self.batch_size = loader["batch_size"]
        self.accumulation_steps = optimizer["gradient_accumulation_steps"]
        self.example_index = 0
        self.microstep_index = 0
        self._observed_batches: list[ObservedBatch] = []
        self._completed_steps: list[mx.array] = []
        self._starting_step = 0
        lora = LoRAConfig(**optimizer["lora"])
        config = SwagTrainingConfig(
            base_checkpoint=directory / "base",
            data=directory / "encoded",
            output_run=directory / "run",
            lora=lora,
            optimizer=OptimizerConfig(
                learning_rate=optimizer["learning_rate"],
                beta1=optimizer["betas"][0],
                beta2=optimizer["betas"][1],
                epsilon=optimizer["epsilon"],
                bias_correction=optimizer["bias_correction"],
                schedule_steps=optimizer["total_steps"],
                warmup_steps=optimizer["warmup_steps"],
                minimum_learning_rate_ratio=optimizer["minimum_learning_rate_ratio"],
                gradient_clip_norm=optimizer["max_grad_norm"],
                weight_decay=WeightDecayPolicy(**optimizer["parameter_weight_decay"]),
            ),
            loader=LoaderConfig(
                microbatch_size=self.batch_size,
                gradient_accumulation_steps=self.accumulation_steps,
                prefetch_depth=1,
                epoch_seed=self.seed,
            ),
            seed=self.seed,
        )
        self.model = SMLLanguageModel(model_config, key=mx.random.key(self.seed))
        apply_lora(self.model, lora, key=mx.random.key(self.seed))
        self.initial_parameter_identity = initialize_parameters(self.model, workload)
        self.adapters, self.frozen_base = split_adapter_parameters(
            self.model.parameters()
        )
        self.optimizer = initialize_adam_state(self.adapters)
        self.trainer = initial_swag_trainer_state(
            self.adapters, key=counter_random_key(self.seed, 0)
        )
        self.dropout_enabled = model_config.hidden_dropout > 0.0 or lora.dropout > 0.0
        self.kernels = build_swag_kernels(
            self.model,
            config,
            build_weight_decay_tree(self.adapters, config.optimizer.weight_decay),
        )
        mx.eval(
            self.adapters,
            self.frozen_base,
            self.optimizer.to_tree(),
            self.trainer.to_tree(),
        )
        self._open_encoded(workload, directory / "encoded", model_config)

    def _open_encoded(self, workload, directory, model_config):
        examples = fixed_swag_examples(workload, np.empty((0, 0), dtype=np.int32))
        expected_identity = workload.semantic_identities["canonical_swag_examples"]
        if examples.identity != expected_identity:
            raise ValueError(
                "canonical SWAG examples do not match their workload identity"
            )
        length = int(examples.input_ids.shape[-1])
        if length > model_config.effective_context_length:
            raise ValueError("SWAG benchmark sequence exceeds model context")
        arrays = {
            "input_ids": examples.input_ids,
            "valid_token_mask": examples.input_ids != model_config.pad_token_id,
            "score_mask": examples.labels != model_config.pad_token_id,
            "labels": examples.candidate_labels,
        }
        self._canonical_arrays = {
            **arrays,
            "example_mask": np.ones((len(examples.example_ids),), dtype=np.bool_),
        }
        references = []
        mapped = {}
        try:
            for name, array in arrays.items():
                logical = f"buckets/length-{length:04d}/{name}.npy"
                path = directory / logical
                path.parent.mkdir(parents=True, exist_ok=True)
                references.append(_write_npy(path, array, logical))
                mapped[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            reconstructed = {
                "example_ids": np.asarray(examples.example_ids, dtype=np.int32),
                "input_ids": mapped["input_ids"],
                "labels": np.where(
                    mapped["score_mask"], mapped["input_ids"], model_config.pad_token_id
                ),
                "candidate_labels": mapped["labels"],
            }
            self.canonical_input_identity = semantic_array_identity(
                "sml-benchmark-swag-examples-v1", reconstructed
            )
            if self.canonical_input_identity != expected_identity:
                raise ValueError("native SWAG storage changed the canonical examples")
            _validate_bucket_arrays(
                **mapped,
                vocab_size=model_config.vocab_size,
                pad_token_id=model_config.pad_token_id,
                eos_token_id=model_config.eos_token_id,
                bos_token_id=model_config.bos_token_id,
                maximum_length=length,
                bucket_length=length,
                bucket_boundaries=(length,),
            )
            manifest = SwagDataManifest(
                kind="swag-data",
                version=1,
                identity="sha256:" + "0" * 64,
                source={
                    "backend": "canonical-benchmark-v1",
                    "identity": examples.identity,
                },
                preprocessing={"maximum_length": length, "bucket_boundaries": [length]},
                base_identity=self.initial_parameter_identity,
                tokenizer_identity=workload.semantic_identities["benchmark_tokenizer"],
                vocab_size=model_config.vocab_size,
                bos_token_id=model_config.bos_token_id,
                eos_token_id=model_config.eos_token_id,
                pad_token_id=model_config.pad_token_id,
                unk_token_id=model_config.unk_token_id,
                example_count=len(examples.example_ids),
                dropped_overlength_rows=0,
                buckets=tuple(references),
            )
            self.manifest = replace(manifest, identity=manifest.recompute_identity())
            (directory / "manifest.json").write_bytes(
                canonical_json_bytes(self.manifest)
            )
            self.native_representation_identity = self.manifest.identity
            self.bucket = SwagBucket(length, **mapped)
            self.example_ids = examples.example_ids
            self._mappings = tuple(mapped.values())
        except BaseException:
            for array in mapped.values():
                array._mmap.close()
            raise

    def _next_batch(self) -> SwagBatch:
        count = len(self.example_ids)
        indices = tuple(
            (self.example_index + offset) % count for offset in range(self.batch_size)
        )
        source_epoch = self.example_index // count
        self.example_index += self.batch_size
        inputs, valid, score, labels, real = _assemble_batch_arrays(
            self.bucket, indices, batch_size=self.batch_size, manifest=self.manifest
        )
        envelope = SwagBatchEnvelope._owned(
            inputs,
            score,
            labels,
            real,
            valid,
            SwagCursor(self.example_index // count, 0, self.example_index % count),
            source_epoch=source_epoch,
        )
        return SwagBatch.from_envelope(envelope)

    def run(self, units: int) -> float:
        if type(units) is not int or units < 0:
            raise ValueError("SWAG benchmark units must be nonnegative integers")
        for _ in range(units):
            for _ in range(self.accumulation_steps):
                batch = self._next_batch()
                if self.dropout_enabled:
                    tree = self.trainer.to_tree()
                    self.trainer = SwagTrainerState.from_compiled_tree(
                        (
                            tree[0],
                            tree[1],
                            counter_random_key(self.seed, self.microstep_index),
                            tree[3],
                            tree[4],
                        )
                    )
                    del tree
                self.trainer = self.kernels.ranking_microstep(
                    self.adapters, self.frozen_base, self.trainer, batch
                )
                self._observed_batches.append(
                    ObservedBatch(
                        {name: getattr(batch, name) for name in self._canonical_arrays},
                        cursor=batch.cursor_after,
                    )
                )
                self.microstep_index += 1
            self.adapters, self.optimizer, self.trainer = self.kernels.optimizer_step(
                self.adapters, self.optimizer, self.trainer
            )
            mx.eval(self.adapters, self.optimizer.to_tree(), self.trainer.to_tree())
            self._completed_steps.append(self.optimizer.step)
            self.model.update(
                _merge_adapter_parameters(self.adapters, self.frozen_base)
            )
        return float(units * self.accumulation_steps * self.batch_size)

    def reset_measured_order(self) -> None:
        self.example_index = 0
        self._observed_batches.clear()
        self._completed_steps.clear()
        self._starting_step = int(self.optimizer.step.item())

    def reset_after_warmup(self) -> None:
        self.reset_measured_order()

    def observed_execution_order(self) -> tuple[int, ...]:
        examples = verify_input_batches(
            self._observed_batches, self._canonical_arrays, batch_size=self.batch_size
        )
        steps = tuple(
            int(step.item()) - self._starting_step - 1 for step in self._completed_steps
        )
        if (
            steps != tuple(range(len(steps)))
            or len(examples) != len(steps) * self.accumulation_steps * self.batch_size
        ):
            raise RuntimeError(
                "benchmark observed optimizer updates differ from canonical work"
            )
        return examples

    @property
    def measured_work_ids(self) -> list[int]:
        return list(self.observed_execution_order())

    def close(self) -> None:
        self.bucket = None
        mappings, self._mappings = self._mappings, ()
        for array in mappings:
            array._mmap.close()


def make_runtime(metric, workload, directory):
    if metric != "swag-end-to-end":
        raise ValueError("native SWAG runtime requires swag-end-to-end")
    return SwagRuntime(workload, directory)
