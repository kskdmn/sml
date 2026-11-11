from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest
from mlx.utils import tree_flatten
from sml import inference
from sml.artifacts import checkpoint
from sml.errors import SMLArtifactError, SMLRuntimeError
from sml.inference import (
    GenerationRequest,
    InferenceRuntimeConfig,
    InferenceSession,
    resolve_model_artifact,
)
from sml.model.config import GenerationConfig
from sml_test_support.inference import publish_new_valid_step


@pytest.mark.parametrize(
    "buckets,message",
    [
        pytest.param((1, 2, 2, 8), "strictly increasing", id="duplicate-bucket"),
        pytest.param((1, 0, 4), "positive", id="nonpositive-bucket"),
    ],
)
def test_session_runtime_config_rejects_invalid_batch_buckets(buckets, message):
    with pytest.raises(ValueError, match=message):
        InferenceRuntimeConfig(batch_size_buckets=buckets)


@pytest.mark.parametrize("factor", (0.0, 0.5, -1.0, float("inf"), float("nan")))
def test_session_runtime_config_rejects_invalid_rope_scales(factor: float) -> None:
    with pytest.raises(ValueError, match="rope_scaling_factor"):
        InferenceRuntimeConfig(rope_scaling_factor=factor)


@pytest.mark.parametrize("factor", (True, False, "8"))
def test_session_runtime_config_rejects_non_numeric_rope_scales(factor: object) -> None:
    with pytest.raises(TypeError, match="rope_scaling_factor"):
        InferenceRuntimeConfig(rope_scaling_factor=factor)


def test_session_extends_pretraining_context_without_changing_source_config(
    tiny_pretraining_run: Path,
) -> None:
    resolved = resolve_model_artifact(tiny_pretraining_run, full_verify=False)
    resolved = replace(
        resolved,
        model_config=replace(resolved.model_config, original_context_length=1_024),
    )
    session = InferenceSession(resolved, InferenceRuntimeConfig())

    assert resolved.model_config.rope_scaling_factor == 1.0
    assert resolved.model_config.effective_context_length == 1_024
    assert session.resolved_model.model_config.rope_scaling_factor == 8.0
    assert session.resolved_model.model_config.effective_context_length == 8_192
    assert session.length_buckets[-1] == 8_192
    assert session.model_identity == resolved.identity()
    assert all(
        session.resolved_model.model_arrays[name] is weight
        for name, weight in resolved.model_arrays.items()
    )


@pytest.mark.parametrize("artifact_kind", ("pretraining-run", "lora-run", "export"))
def test_session_explicit_unscaled_rope_override(
    tiny_pretraining_run: Path,
    artifact_kind: str,
) -> None:
    resolved = resolve_model_artifact(tiny_pretraining_run, full_verify=False)
    resolved = replace(
        resolved,
        artifact_kind=artifact_kind,
        model_config=replace(resolved.model_config, rope_scaling_factor=8.0),
    )
    session = InferenceSession(
        resolved, InferenceRuntimeConfig(rope_scaling_factor=1.0)
    )

    assert resolved.model_config.rope_scaling_factor == 8.0
    assert session.resolved_model.model_config.rope_scaling_factor == 1.0
    assert session.resolved_model.model_config.effective_context_length == 32


@pytest.mark.parametrize("artifact_kind", ("lora-run", "export"))
@pytest.mark.parametrize("factor", (1.0, 4.0, 8.0))
def test_session_preserves_saved_finetuned_rope_scale(
    tiny_pretraining_run: Path,
    artifact_kind: str,
    factor: float,
) -> None:
    resolved = resolve_model_artifact(tiny_pretraining_run, full_verify=False)
    resolved = replace(
        resolved,
        artifact_kind=artifact_kind,
        model_config=replace(resolved.model_config, rope_scaling_factor=factor),
    )
    session = InferenceSession(resolved, InferenceRuntimeConfig())

    assert session.resolved_model.model_config == resolved.model_config
    assert session.length_buckets[-1] == int(32 * factor)


def test_session_model_reuses_resolved_weights(tiny_session: InferenceSession) -> None:
    parameters = dict(tree_flatten(tiny_session._model.parameters()))
    resolved = tiny_session.resolved_model.model_arrays

    assert parameters.keys() == resolved.keys()
    assert all(parameters[name] is weight for name, weight in resolved.items())


def test_resolve_rejects_checkpoint_manifest_as_model_artifact(
    tiny_pretraining_run: Path,
) -> None:
    step_path = next((tiny_pretraining_run / "checkpoints").glob("step-*"))
    with pytest.raises(SMLArtifactError):
        resolve_model_artifact(step_path, full_verify=False)


def test_session_loads_latest_once_and_pins_identity(
    tiny_pretraining_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0
    real_open_latest = inference.open_latest_checkpoint_reader

    @contextmanager
    def counted_open_latest(*args, **kwargs):
        nonlocal calls
        calls += 1
        with real_open_latest(*args, **kwargs) as reader:
            yield reader

    monkeypatch.setattr(inference, "open_latest_checkpoint_reader", counted_open_latest)
    session = InferenceSession.from_checkpoint(tiny_pretraining_run)
    first_identity = session.model_identity
    assert session.resolved_model.model_config.rope_scaling_factor == 8.0
    publish_new_valid_step(tiny_pretraining_run, step=4)
    assert session.model_identity == first_identity
    assert calls == 1


def test_failed_generation_releases_call_guard(
    tiny_session: InferenceSession, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = GenerationRequest(max_new_tokens=2, config=GenerationConfig(seed=17))
    expected = tiny_session.generate("alpha", request)

    def fail_decode(*_args, **_kwargs):
        raise SMLRuntimeError("decode failed")

    with monkeypatch.context() as patch:
        patch.setattr(tiny_session, "_generate_batch", fail_decode)
        with pytest.raises(SMLRuntimeError, match="decode"):
            tiny_session.generate("alpha", request)

    assert tiny_session.generate("alpha", request) == expected
    assert tiny_session.buffer_pool.active_leases == 0


def test_overlapping_session_call_fails_before_state_mutation(
    tiny_session: InferenceSession,
) -> None:
    with (
        tiny_session._call_guard.acquire(),
        pytest.raises(SMLRuntimeError, match="non-reentrant"),
    ):
        tiny_session.generate("blocked", GenerationRequest(max_new_tokens=1))
    assert tiny_session.buffer_pool.active_leases == 0


@pytest.mark.parametrize("full_verify", (False, True))
def test_resolve_model_artifact_does_not_nest_run_access_lock(
    tiny_pretraining_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    full_verify: bool,
) -> None:
    depth = 0
    max_depth = 0
    real_lock = checkpoint._protected_lock

    @contextmanager
    def tracking(protected, *, category, exclusive, wait=False):
        nonlocal depth, max_depth
        if category == "run-access":
            depth += 1
            max_depth = max(max_depth, depth)
        try:
            with real_lock(
                protected,
                category=category,
                exclusive=exclusive,
                wait=wait,
            ):
                yield
        finally:
            if category == "run-access":
                depth -= 1

    monkeypatch.setattr(checkpoint, "_protected_lock", tracking)
    resolve_model_artifact(tiny_pretraining_run, full_verify=full_verify)
    assert max_depth == 1
    assert depth == 0


def test_empty_batch_returns_before_taking_the_call_guard(
    tiny_session: InferenceSession,
) -> None:
    with tiny_session._call_guard.acquire():
        assert tiny_session.generate_batch([]) == ()
    assert tiny_session.buffer_pool.active_leases == 0


def test_infer_constructs_one_session_and_delegates_once_to_generate(
    tiny_pretraining_run: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sml.inference import InferenceConfig, infer

    calls: list[tuple[str, GenerationRequest]] = []
    real_generate = InferenceSession.generate

    def tracking(self, text, request):
        calls.append((text, request))
        return real_generate(self, text, request)

    monkeypatch.setattr(InferenceSession, "generate", tracking)
    request = GenerationRequest(max_new_tokens=1)
    result = infer(
        InferenceConfig(
            checkpoint=tiny_pretraining_run,
            prompt="alpha",
            request=request,
        )
    )
    assert len(calls) == 1
    assert calls[0] == ("alpha", request)
    assert result.token_ids
    assert result.model.run_identity is not None
