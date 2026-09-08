"""Canonical measurements through the training and publication kernels."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map
from sml.artifacts.checkpoint import publish_checkpoint, publish_run
from sml.artifacts.manifest import canonical_json_bytes
from sml.data.pretraining import PretrainingBatchStream, PretrainingCursor
from sml.model.config import ModelConfig
from sml.model.language_model import SMLLanguageModel
from sml.training.common import (
    LoaderConfig,
    OptimizerConfig,
    PretrainingConfig,
    TrainerState,
    WeightDecayPolicy,
    build_weight_decay_tree,
    initialize_adam_state,
    initialize_base_parameter_state,
)
from sml.training.pretrain import (
    ScalarTrainingState,
    _checkpoint_builder,
    _copy_run_tokenizer,
    _publish_training_state,
    _RestoredTrainingState,
    _run_manifest,
    build_pretraining_kernels,
)
from sml.training.random import counter_random_key

from v2.benchmarks.adapters.prepared_data import _materialize_bundle
from v2.benchmarks.parameters import initialize_parameters
from v2.benchmarks.workload import fixed_canonical_rows


def optimizer_configuration(values) -> OptimizerConfig:
    expected = {
        "name",
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
        "parameter_weight_decay",
        "swag",
    }
    if set(values) != expected or values["name"] != "adamw":
        raise ValueError("native training optimizer has unsupported fields")
    if not isinstance(values["betas"], (list, tuple)) or len(values["betas"]) != 2:
        raise ValueError("native training optimizer requires exactly two betas")
    if values["optimizer_weight_decay"] != 0.0:
        raise ValueError("canonical optimizer decay must use the parameter policy")
    return OptimizerConfig(
        learning_rate=values["learning_rate"],
        beta1=values["betas"][0],
        beta2=values["betas"][1],
        epsilon=values["epsilon"],
        bias_correction=values["bias_correction"],
        schedule_steps=values["total_steps"],
        warmup_steps=values["warmup_steps"],
        minimum_learning_rate_ratio=values["minimum_learning_rate_ratio"],
        gradient_clip_norm=values["max_grad_norm"],
        weight_decay=WeightDecayPolicy(**values["parameter_weight_decay"]),
    )


class PretrainingRuntime:
    def __init__(self, metric, workload, directory: Path):
        expected_loader = {
            "sequence_length",
            "microbatch_size",
            "row_count",
            "canonical_dtype",
            "storage_dtype",
            "row_order",
            "swag",
        }
        if set(workload.loader) != expected_loader:
            raise ValueError("native training loader has unsupported fields")
        self.workload = workload
        self.bundle = _materialize_bundle(workload, directory / "data")
        self.native_representation_identity = self.bundle.manifest.identity
        self.canonical_input_identity = workload.semantic_identities[
            "canonical_training_rows"
        ]
        self.batch_size = workload.loader["microbatch_size"]
        self.accumulation_steps = workload.optimizer["gradient_accumulation_steps"]
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError(
                "native training microbatch_size must be a positive integer"
            )
        if int(workload.loader["row_count"]) % self.batch_size:
            raise ValueError("canonical training rows must form complete batches")
        self.sequence_length = int(workload.loader["sequence_length"])
        self.include_loader = metric in {
            "pretraining-end-to-end",
            "peak-metal-memory",
            "checkpoint-pause",
        }
        self.config = PretrainingConfig(
            data=self.bundle.path,
            output_run=directory / "run",
            model=ModelConfig(**workload.model),
            optimizer=optimizer_configuration(workload.optimizer),
            loader=LoaderConfig(
                microbatch_size=self.batch_size,
                gradient_accumulation_steps=self.accumulation_steps,
                epoch_seed=0,
            ),
            maximum_steps=workload.optimizer["total_steps"],
            maximum_epochs=None,
            seed=workload.optimizer["seed"],
        )
        self.model = SMLLanguageModel(
            self.config.model, key=mx.random.key(self.config.seed)
        )
        self.initial_parameter_identity = initialize_parameters(self.model, workload)
        parameters = initialize_base_parameter_state(self.model.parameters())
        self.model.update(parameters.working_parameters)
        optimizer = initialize_adam_state(parameters.master_parameters)
        trainer = TrainerState(
            accumulators=tree_map(mx.zeros_like, parameters.master_parameters),
            accumulation_count=mx.array(0, dtype=mx.int32),
            next_key=counter_random_key(self.config.seed, 0),
            loss_numerator=mx.array(0.0, dtype=mx.float32),
        )
        self.state = _RestoredTrainingState(
            parameters,
            optimizer,
            trainer,
            ScalarTrainingState(0, 0, 0, PretrainingCursor.initial()),
        )
        mx.eval(parameters.to_tree(), optimizer.to_tree(), trainer.to_tree())
        self.kernels = build_pretraining_kernels(
            self.model,
            self.config,
            build_weight_decay_tree(
                parameters.working_parameters, self.config.optimizer.weight_decay
            ),
        )
        self.stream = None
        self.epoch = 0
        self.row_index = 0
        # Compute-only measurements start with inputs already resident on device.
        self.device_batches = []
        if not self.include_loader:
            rows = fixed_canonical_rows(
                row_count=int(workload.loader["row_count"]),
                row_width=self.sequence_length + 1,
                vocab_size=int(workload.model["vocab_size"]),
            )
            if len(rows) % self.batch_size:
                raise ValueError("canonical compute rows must form complete batches")
            self.device_batches = [
                mx.array(np.ascontiguousarray(rows[index : index + self.batch_size]))
                for index in range(0, len(rows), self.batch_size)
            ]
            mx.eval(self.device_batches)
        else:
            self._open_stream(PretrainingCursor.initial())

    def _open_stream(self, cursor):
        self.stream = PretrainingBatchStream(
            self.bundle,
            batch_size=self.batch_size,
            seed=0,
            prefetch_depth=self.config.loader.prefetch_depth,
            cursor=cursor,
        )
        self.epoch = cursor.epoch

    def _next_rows(self):
        if not self.include_loader:
            rows = self.device_batches[self.row_index % len(self.device_batches)]
            self.row_index += 1
            return rows, self.state.scalar.cursor
        envelope = next(self.stream)
        with envelope:
            rows = mx.array(envelope.rows)
            cursor = envelope.cursor_after
        self.row_index += 1
        return rows, cursor

    def run(self, units: int) -> float:
        for _ in range(units):
            for microstep in range(self.accumulation_steps):
                rows, cursor = self._next_rows()
                if self.config.model.hidden_dropout > 0:
                    self.state.trainer = replace(
                        self.state.trainer,
                        next_key=counter_random_key(
                            self.config.seed, self.state.scalar.microsteps + microstep
                        ),
                    )
                result = self.kernels.microstep(
                    self.state.parameters, self.state.trainer, rows
                )
                self.state.parameters = result.parameters
                self.state.trainer = result.trainer
                del result
            updated = self.kernels.optimizer_step(
                self.state.parameters, self.state.optimizer, self.state.trainer
            )
            self.state.parameters = updated.parameters
            self.state.optimizer = updated.optimizer
            self.state.trainer = updated.trainer
            self.model.update(updated.parameters.working_parameters)
            self.last_metrics = updated.metrics
            del updated
            self.state.scalar = ScalarTrainingState(
                self.state.scalar.step + 1,
                self.state.scalar.rows + self.batch_size * self.accumulation_steps,
                self.state.scalar.microsteps + self.accumulation_steps,
                cursor,
            )
            if self.stream is not None:
                self.stream.commit(cursor)
        return float(
            units * self.batch_size * self.accumulation_steps * self.sequence_length
        )

    def reset_measured_order(self):
        consumed = self.row_index > 0
        self.row_index = 0
        if self.stream is not None and consumed:
            self.stream.close()
            self._open_stream(PretrainingCursor.initial())

    def reset_after_warmup(self):
        self.reset_measured_order()

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class CheckpointRuntime:
    """Time durable publication; perform real preceding updates outside the pause."""

    def __init__(self, workload, directory):
        self.training = PretrainingRuntime("checkpoint-pause", workload, directory)
        try:
            self._initialize_publication()
        except BaseException:
            self.training.close()
            raise

    def _initialize_publication(self):
        self.initial_parameter_identity = self.training.initial_parameter_identity
        self.native_representation_identity = (
            self.training.native_representation_identity
        )
        self.canonical_input_identity = self.training.canonical_input_identity
        self.path = self.training.config.output_run
        self.manifest = _run_manifest(self.training.config, self.training.bundle)

        def build(private_run):
            _copy_run_tokenizer(self.training.bundle.path, private_run)
            (private_run / "checkpoints").mkdir()
            (private_run / "run.json").write_bytes(canonical_json_bytes(self.manifest))
            publish_checkpoint(
                private_run, _checkpoint_builder(self.manifest, self.training.state)
            )
            return self.manifest

        publish_run(self.path, build)
        self.prepared = False

    def prepare_measured_unit(self):
        if self.prepared:
            raise RuntimeError("checkpoint unit already prepared")
        self.training.run(1)
        self.prepared = True

    def run(self, units):
        for _ in range(units):
            if not self.prepared:
                self.prepare_measured_unit()
            _publish_training_state(self.path, self.manifest, self.training.state)
            self.prepared = False
        return float(units)

    def close(self):
        self.training.close()


def make_runtime(metric, workload, directory):
    if metric == "checkpoint-pause":
        return CheckpointRuntime(workload, directory)
    return PretrainingRuntime(metric, workload, directory)
