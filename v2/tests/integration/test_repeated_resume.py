from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest
from sml.artifacts.checkpoint import resolve_latest_step
from sml.artifacts.manifest import VerificationLevel
from sml.data.swag import SwagPreparationConfig, SwagSourceConfig, prepare_swag_bundle
from sml.errors import SMLConfigurationError
from sml.inference import resolve_model_artifact
from sml.training import pretrain
from sml.training import swag as swag_module
from sml.training.common import CheckpointPolicy, LoaderConfig, ResumeOverrides
from sml.training.lora import LoRAConfig, LoRAInitializerConfig
from sml.training.swag import SwagTrainingConfig
from test_pretraining_workflow import (
    _assert_run_states_equal,
    _config,
    _prepared_bundle,
)
from test_swag_workflow import (
    FakeSwagProvider,
    _assert_array_maps_equal,
    _latest_checkpoint_state,
    _prepare_tiny_data,
    _swag_rows,
    _tiny_pretraining_config,
    tiny_swag_training_config,
)


class CheckpointInterruption(RuntimeError):
    pass


@contextmanager
def _interrupt_after_checkpoint(module: ModuleType, step: int):
    publish = module._publish_training_state

    def interrupt(run, manifest, state):
        result = publish(run, manifest, state)
        if state.scalar.step == step:
            raise CheckpointInterruption(f"checkpoint {step}")
        return result

    with patch.object(module, "_publish_training_state", interrupt):
        yield


def _assert_checkpoint_step(run: Path, expected: int) -> None:
    resolved = resolve_latest_step(
        run, writable=False, verification=VerificationLevel.FULL
    )
    assert resolved.step == expected


def _pretraining_config(data: Path, run: Path, *, compiled: bool, dropout: float):
    base = _config(data, run, maximum_steps=12)
    return replace(
        base,
        model=replace(
            base.model,
            hidden_size=32,
            num_layers=2,
            num_q_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            hidden_dropout=dropout,
        ),
        optimizer=replace(base.optimizer, schedule_steps=None),
        loader=LoaderConfig(
            microbatch_size=2,
            gradient_accumulation_steps=2,
            prefetch_depth=3,
            epoch_seed=13,
        ),
        maximum_epochs=None,
        checkpoint=CheckpointPolicy(interval=3),
        log_interval=100,
        compile=compiled,
    )


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "compiled"))
@pytest.mark.parametrize("dropout", (0.0, 0.2), ids=("no-dropout", "dropout"))
def test_pretraining_multiple_checkpoint_resumes_match_uninterrupted_training(
    tmp_path: Path, compiled: bool, dropout: float
) -> None:
    # Three microbatches per epoch produce one full accumulation window and
    # one partial window, dropping one row. Checkpoints span both boundaries.
    data = _prepared_bundle(tmp_path / "data", partitions=((0, 1, 2), (3, 4, 5, 6)))
    config = _pretraining_config(
        data, tmp_path / "full", compiled=compiled, dropout=dropout
    )
    uninterrupted = pretrain.train(config)
    interrupted_run = tmp_path / "interrupted"

    with (
        _interrupt_after_checkpoint(pretrain, 3),
        pytest.raises(CheckpointInterruption, match="checkpoint 3"),
    ):
        pretrain.train(replace(config, output_run=interrupted_run))
    _assert_checkpoint_step(interrupted_run, 3)
    for step in (6, 9):
        with (
            _interrupt_after_checkpoint(pretrain, step),
            pytest.raises(CheckpointInterruption, match=f"checkpoint {step}"),
        ):
            pretrain.resume(interrupted_run, data=data, overrides=ResumeOverrides())
        _assert_checkpoint_step(interrupted_run, step)

    resumed = pretrain.resume(interrupted_run, data=data, overrides=ResumeOverrides())

    assert resumed.step == uninterrupted.step == 12
    assert resumed.epoch == uninterrupted.epoch == 6
    assert resumed.rows == uninterrupted.rows == 36
    assert (resumed.run / "run.json").read_bytes() == (
        uninterrupted.run / "run.json"
    ).read_bytes()
    _assert_run_states_equal(resumed.run, uninterrupted.run)


def test_pretraining_multiple_fresh_process_resumes_match_uninterrupted_training(
    tmp_path: Path,
) -> None:
    data = _prepared_bundle(tmp_path / "data", partitions=((0, 1, 2), (3, 4, 5, 6)))
    config = _pretraining_config(data, tmp_path / "full", compiled=True, dropout=0.2)
    uninterrupted = pretrain.train(config)
    interrupted_run = tmp_path / "interrupted"
    with (
        _interrupt_after_checkpoint(pretrain, 3),
        pytest.raises(CheckpointInterruption, match="checkpoint 3"),
    ):
        pretrain.train(replace(config, output_run=interrupted_run))

    script = textwrap.dedent(
        """
        import sys
        from pathlib import Path
        from unittest.mock import patch
        from sml.training import pretrain
        from sml.training.common import ResumeOverrides

        run, data = Path(sys.argv[1]), Path(sys.argv[2])
        stop_at = int(sys.argv[3])
        publish = pretrain._publish_training_state

        class CheckpointInterruption(RuntimeError):
            pass

        def interrupt(run, manifest, state):
            result = publish(run, manifest, state)
            if state.scalar.step == stop_at:
                raise CheckpointInterruption(state.scalar.step)
            return result

        with patch.object(pretrain, "_publish_training_state", interrupt):
            try:
                result = pretrain.resume(run, data=data, overrides=ResumeOverrides())
            except CheckpointInterruption as error:
                assert error.args == (stop_at,)
            else:
                assert stop_at == 0
                assert result.step == 12
        """
    )
    for stop_at, expected_step in ((6, 6), (9, 9), (0, 12)):
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(interrupted_run),
                str(data),
                str(stop_at),
            ],
            cwd=Path(__file__).resolve().parents[3],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        _assert_checkpoint_step(interrupted_run, expected_step)

    _assert_run_states_equal(interrupted_run, uninterrupted.run)


def _swag_config(tmp_path: Path, *, compiled: bool):
    tokenizer, data = _prepare_tiny_data(tmp_path)
    base_config = _tiny_pretraining_config(
        data.path, tmp_path / "base", tokenizer.manifest.vocab_size
    )
    base_config = replace(
        base_config,
        model=replace(
            base_config.model,
            hidden_size=32,
            num_layers=2,
            num_q_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            hidden_dropout=0.2,
        ),
    )
    base = pretrain.train(base_config).run
    bundle = prepare_swag_bundle(
        SwagPreparationConfig(
            provider=FakeSwagProvider(_swag_rows(5)),
            source=SwagSourceConfig(revision="deadbeef" * 5),
            maximum_length=32,
            bucket_boundaries=(16, 32),
        ),
        resolve_model_artifact(base, full_verify=True),
        tmp_path / "swag-data",
    )
    config = tiny_swag_training_config(
        base,
        bundle,
        tmp_path / "full",
        lora=LoRAConfig(
            rank=3,
            alpha=1.0,
            scaling_mode="lora",
            dropout=0.3,
            target_modules=(
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
            initializer=LoRAInitializerConfig(lora_a=0.05, lora_b=0.05),
        ),
        loader=LoaderConfig(
            microbatch_size=2,
            gradient_accumulation_steps=2,
            prefetch_depth=3,
            epoch_seed=13,
        ),
        maximum_steps=12,
        maximum_epochs=None,
        checkpoint=CheckpointPolicy(interval=3),
        log_interval=100,
        compile=compiled,
    )
    config = replace(
        config,
        optimizer=replace(
            config.optimizer,
            schedule_steps=12,
            warmup_steps=0,
            gradient_clip_norm=0.00001,
        ),
    )
    return config, bundle


def _assert_swag_states_equal(left: Path, right: Path) -> None:
    left_state = _latest_checkpoint_state(left)
    right_state = _latest_checkpoint_state(right)
    assert left_state["scalar"] == right_state["scalar"]
    for group in ("adapters", "optimizer", "trainer"):
        _assert_array_maps_equal(left_state[group], right_state[group])


@pytest.mark.parametrize("compiled", (False, True), ids=("eager", "compiled"))
def test_swag_multiple_checkpoint_resumes_match_uninterrupted_training(
    tmp_path: Path, compiled: bool
) -> None:
    config, bundle = _swag_config(tmp_path, compiled=compiled)
    try:
        uninterrupted = swag_module.finetune(config)
        interrupted_run = tmp_path / "interrupted"
        with (
            _interrupt_after_checkpoint(swag_module, 3),
            pytest.raises(CheckpointInterruption, match="checkpoint 3"),
        ):
            swag_module.finetune(replace(config, output_run=interrupted_run))
        _assert_checkpoint_step(interrupted_run, 3)
        for step in (6, 9):
            with (
                _interrupt_after_checkpoint(swag_module, step),
                pytest.raises(CheckpointInterruption, match=f"checkpoint {step}"),
            ):
                swag_module.resume_finetune(
                    interrupted_run, data=bundle.path, overrides=ResumeOverrides()
                )
            _assert_checkpoint_step(interrupted_run, step)
        resumed = swag_module.resume_finetune(
            interrupted_run, data=bundle.path, overrides=ResumeOverrides()
        )
    finally:
        bundle.close()

    assert resumed.step == uninterrupted.step == 12
    assert resumed.epoch == uninterrupted.epoch == 6
    assert resumed.examples == uninterrupted.examples == 30
    _assert_swag_states_equal(uninterrupted.run, resumed.run)


def test_pretraining_session_limits_preserve_planned_schedule_and_exact_state(
    tmp_path: Path,
) -> None:
    data = _prepared_bundle(tmp_path / "data", partitions=((0, 1, 2), (3, 4, 5, 6)))
    config = replace(
        _pretraining_config(data, tmp_path / "full", compiled=True, dropout=0.2),
        checkpoint=CheckpointPolicy(interval=100),
    )
    uninterrupted = pretrain.train(config)
    segmented = pretrain.train(
        replace(config, output_run=tmp_path / "sessions"), session_steps=3
    )
    assert segmented.step == 3
    _assert_checkpoint_step(segmented.run, 3)
    saved = json.loads((segmented.run / "run.json").read_text())
    assert saved["optimizer"]["schedule_steps"] == 12
    assert saved["checkpoint"]["maximum_steps"] == 12

    # A session limit applies only to that invocation. An uncapped restart from
    # the same step-3 checkpoint must complete all nine remaining updates.
    uncapped_run = tmp_path / "uncapped"
    shutil.copytree(segmented.run, uncapped_run)
    uncapped = pretrain.resume(uncapped_run, data=data, overrides=ResumeOverrides())
    assert uncapped.step == 12
    _assert_run_states_equal(uncapped.run, uninterrupted.run)

    for expected_step in (6, 9):
        segmented = pretrain.resume(
            segmented.run,
            data=data,
            overrides=ResumeOverrides(),
            session_steps=3,
        )
        assert segmented.step == expected_step
        _assert_checkpoint_step(segmented.run, expected_step)
    segmented = pretrain.resume(segmented.run, data=data, overrides=ResumeOverrides())

    assert segmented.step == uninterrupted.step == 12
    _assert_run_states_equal(segmented.run, uninterrupted.run)


def test_swag_session_limits_preserve_planned_schedule_and_exact_state(
    tmp_path: Path,
) -> None:
    config, bundle = _swag_config(tmp_path, compiled=True)
    config = replace(config, checkpoint=CheckpointPolicy(interval=100))
    try:
        uninterrupted = swag_module.finetune(config)
        segmented = swag_module.finetune(
            replace(config, output_run=tmp_path / "sessions"), session_steps=3
        )
        assert segmented.step == 3
        _assert_checkpoint_step(segmented.run, 3)
        saved = json.loads((segmented.run / "run.json").read_text())
        assert saved["optimizer"]["schedule_steps"] == 12
        assert saved["checkpoint"]["maximum_steps"] == 12

        uncapped_run = tmp_path / "uncapped"
        shutil.copytree(segmented.run, uncapped_run)
        uncapped = swag_module.resume_finetune(
            uncapped_run, data=bundle.path, overrides=ResumeOverrides()
        )
        assert uncapped.step == 12
        _assert_swag_states_equal(uncapped.run, uninterrupted.run)

        for expected_step in (6, 9):
            segmented = swag_module.resume_finetune(
                segmented.run,
                data=bundle.path,
                overrides=ResumeOverrides(),
                session_steps=3,
            )
            assert segmented.step == expected_step
            _assert_checkpoint_step(segmented.run, expected_step)
        segmented = swag_module.resume_finetune(
            segmented.run, data=bundle.path, overrides=ResumeOverrides()
        )
    finally:
        bundle.close()

    assert segmented.step == uninterrupted.step == 12
    _assert_swag_states_equal(segmented.run, uninterrupted.run)


@pytest.mark.parametrize("session_steps", (True, False, 0, -1, 1.5, "3", 2**31))
@pytest.mark.parametrize("task", ("train", "finetune", "resume", "resume-finetune"))
def test_invalid_session_limits_fail_before_artifact_access(
    tmp_path: Path, task: str, session_steps: object
) -> None:
    run = tmp_path / "run"
    data = tmp_path / "missing-data"
    with pytest.raises(SMLConfigurationError, match="session_steps"):
        if task == "train":
            pretrain.train(
                _pretraining_config(data, run, compiled=False, dropout=0.0),
                session_steps=session_steps,
            )
        elif task == "finetune":
            swag_module.finetune(
                SwagTrainingConfig(
                    base_checkpoint=tmp_path / "missing-base",
                    data=data,
                    output_run=run,
                ),
                session_steps=session_steps,
            )
        elif task == "resume":
            pretrain.resume(
                run,
                data=data,
                overrides=ResumeOverrides(),
                session_steps=session_steps,
            )
        else:
            swag_module.resume_finetune(
                run,
                data=data,
                overrides=ResumeOverrides(),
                session_steps=session_steps,
            )
    assert not run.exists()
