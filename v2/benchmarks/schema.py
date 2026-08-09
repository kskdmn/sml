from __future__ import annotations

# Schema validation reports malformed serialized content uniformly as ValueError.
# ruff: noqa: TRY004
import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Literal

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
MetricName = Literal[
    "prepared-data",
    "pretraining-compute",
    "pretraining-end-to-end",
    "swag-end-to-end",
    "inference-prefill",
    "inference-decode",
    "checkpoint-pause",
    "compile-cold-start",
    "peak-metal-memory",
]
MetricDirection = Literal["higher-is-better", "lower-is-better"]
Side = Literal["reference", "candidate"]

METRIC_NAMES: tuple[MetricName, ...] = (
    "prepared-data",
    "pretraining-compute",
    "pretraining-end-to-end",
    "swag-end-to-end",
    "inference-prefill",
    "inference-decode",
    "checkpoint-pause",
    "compile-cold-start",
    "peak-metal-memory",
)

TRIAL_PAYLOAD_FIELDS = frozenset(
    {
        "metric",
        "side",
        "attempt_index",
        "pair_index",
        "process_order",
        "source_commit",
        "source_clean",
        "harness_commit",
        "harness_clean",
        "harness_identity",
        "canonical_workload_identity",
        "native_configuration",
        "native_representation_identity",
        "canonical_row_identity",
        "canonical_input_identity",
        "canonical_projection",
        "execution_order_identity",
        "initial_parameter_identity",
        "comparison_target",
        "warmup_units",
        "measured_units",
        "elapsed_seconds",
        "value",
        "startup_verification_seconds",
        "compilation_seconds",
        "peak_memory_bytes",
        "synchronization_boundaries",
    }
)


def validate_trial_payload(raw: Mapping[str, object]) -> dict[str, JsonValue]:
    if set(raw) != TRIAL_PAYLOAD_FIELDS:
        raise ValueError("trial payload has an invalid field set")
    metric = raw["metric"]
    side = raw["side"]
    if metric not in METRIC_NAMES:
        raise ValueError(f"unsupported benchmark metric: {metric!r}")
    if side not in ("reference", "candidate"):
        raise ValueError(f"unsupported benchmark side: {side!r}")
    integer_fields = (
        "pair_index",
        "attempt_index",
        "process_order",
        "warmup_units",
        "measured_units",
    )
    for name in integer_fields:
        if type(raw[name]) is not int:
            raise ValueError(f"raw trial {name} must be an integer")
    if raw["pair_index"] < 0 or raw["process_order"] < 0:
        raise ValueError("raw trial indices must be non-negative")
    if raw["attempt_index"] not in (0, 1):
        raise ValueError("raw trial attempt_index must be zero or one")
    if raw["warmup_units"] < 0 or raw["measured_units"] <= 0:
        raise ValueError("raw trial unit counts are invalid")
    if raw["source_clean"] is not True or raw["harness_clean"] is not True:
        raise ValueError("raw trials require clean source and harness checkouts")
    for name in ("source_commit", "harness_commit"):
        value = raw[name]
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None:
            raise ValueError(f"{name} must be a full lowercase Git commit")
    for name in (
        "harness_identity",
        "canonical_workload_identity",
        "native_representation_identity",
        "canonical_row_identity",
        "canonical_input_identity",
        "initial_parameter_identity",
        "execution_order_identity",
    ):
        value = raw[name]
        if (
            not isinstance(value, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
        ):
            raise ValueError(f"{name} must be a SHA-256 identity")
    comparison_target = raw["comparison_target"]
    if not isinstance(comparison_target, str) or not comparison_target:
        raise ValueError("comparison_target must be a non-empty string")
    elapsed = raw["elapsed_seconds"]
    value = raw["value"]
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("elapsed_seconds must be finite and positive")
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError("value must be finite and positive")
    optional_float_fields = (
        "startup_verification_seconds",
        "compilation_seconds",
    )
    for name in optional_float_fields:
        item = raw[name]
        if item is not None and (
            type(item) not in (int, float) or not math.isfinite(item) or item < 0
        ):
            raise ValueError(f"{name} must be finite and non-negative")
    peak = raw["peak_memory_bytes"]
    if peak is not None and (type(peak) is not int or peak < 0):
        raise ValueError("peak_memory_bytes must be non-negative")
    mappings = {}
    for name in ("native_configuration", "canonical_projection"):
        item = raw[name]
        if not isinstance(item, dict):
            raise ValueError(f"{name} must be an object")
        mappings[name] = dict(item)
    boundaries = raw["synchronization_boundaries"]
    if not isinstance(boundaries, list) or not all(
        isinstance(item, str) and item for item in boundaries
    ):
        raise ValueError("synchronization_boundaries must be a string list")
    return {
        "metric": metric,
        "side": side,
        "attempt_index": raw["attempt_index"],
        "pair_index": raw["pair_index"],
        "process_order": raw["process_order"],
        "source_commit": raw["source_commit"],
        "source_clean": True,
        "harness_commit": raw["harness_commit"],
        "harness_clean": True,
        "harness_identity": raw["harness_identity"],
        "canonical_workload_identity": raw["canonical_workload_identity"],
        "native_configuration": mappings["native_configuration"],
        "native_representation_identity": raw["native_representation_identity"],
        "canonical_row_identity": raw["canonical_row_identity"],
        "canonical_input_identity": raw["canonical_input_identity"],
        "canonical_projection": mappings["canonical_projection"],
        "execution_order_identity": raw["execution_order_identity"],
        "initial_parameter_identity": raw["initial_parameter_identity"],
        "comparison_target": comparison_target,
        "warmup_units": raw["warmup_units"],
        "measured_units": raw["measured_units"],
        "elapsed_seconds": float(elapsed),
        "value": float(value),
        "startup_verification_seconds": (
            None
            if raw["startup_verification_seconds"] is None
            else float(raw["startup_verification_seconds"])
        ),
        "compilation_seconds": (
            None
            if raw["compilation_seconds"] is None
            else float(raw["compilation_seconds"])
        ),
        "peak_memory_bytes": peak,
        "synchronization_boundaries": list(boundaries),
    }


@dataclass(frozen=True, slots=True)
class WorkUnitDefinition:
    metric: MetricName
    direction: MetricDirection
    numerator: str
    work_unit: str
    start_boundary: str
    end_boundary: str
    measured_units: int

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> WorkUnitDefinition:
        expected = {
            "metric",
            "direction",
            "numerator",
            "work_unit",
            "start_boundary",
            "end_boundary",
            "measured_units",
        }
        if set(raw) != expected:
            raise ValueError("work-unit definition has an invalid field set")
        metric = raw["metric"]
        direction = raw["direction"]
        measured_units = raw["measured_units"]
        if metric not in METRIC_NAMES:
            raise ValueError(f"unsupported benchmark metric: {metric!r}")
        if direction not in ("higher-is-better", "lower-is-better"):
            raise ValueError(f"unsupported metric direction: {direction!r}")
        if type(measured_units) is not int or measured_units <= 0:
            raise ValueError("measured_units must be a positive integer")
        string_fields = {
            name: raw[name]
            for name in ("numerator", "work_unit", "start_boundary", "end_boundary")
        }
        if any(
            not isinstance(value, str) or not value for value in string_fields.values()
        ):
            raise ValueError("work-unit text fields must be non-empty strings")
        return cls(
            metric=metric,
            direction=direction,
            numerator=string_fields["numerator"],
            work_unit=string_fields["work_unit"],
            start_boundary=string_fields["start_boundary"],
            end_boundary=string_fields["end_boundary"],
            measured_units=measured_units,
        )


@dataclass(frozen=True, slots=True)
class CanonicalWorkload:
    schema_version: int
    model: dict[str, JsonValue]
    optimizer: dict[str, JsonValue]
    precision: dict[str, JsonValue]
    loader: dict[str, JsonValue]
    compilation: dict[str, JsonValue]
    generation: dict[str, JsonValue]
    semantic_identities: dict[str, str]
    native_representation_identities: dict[str, str]
    work_units: tuple[WorkUnitDefinition, ...]
    synchronization_boundaries: tuple[str, ...]
    required_environment: dict[str, JsonValue]
    software_requirements: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        raw = asdict(self)
        raw["work_units"] = [asdict(unit) for unit in self.work_units]
        raw["synchronization_boundaries"] = list(self.synchronization_boundaries)
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> CanonicalWorkload:
        expected = {
            "schema_version",
            "model",
            "optimizer",
            "precision",
            "loader",
            "compilation",
            "generation",
            "semantic_identities",
            "native_representation_identities",
            "work_units",
            "synchronization_boundaries",
            "required_environment",
            "software_requirements",
        }
        if set(raw) != expected:
            raise ValueError("canonical workload has an invalid field set")
        if raw["schema_version"] != 1:
            raise ValueError("unsupported canonical workload schema version")
        mappings: dict[str, dict[str, JsonValue]] = {}
        for field_name in (
            "model",
            "optimizer",
            "precision",
            "loader",
            "compilation",
            "generation",
            "semantic_identities",
            "native_representation_identities",
            "required_environment",
            "software_requirements",
        ):
            value = raw[field_name]
            if not isinstance(value, dict):
                raise ValueError(f"{field_name} must be an object")
            mappings[field_name] = dict(value)
        raw_units = raw["work_units"]
        if not isinstance(raw_units, list):
            raise ValueError("work_units must be a list")
        work_units = tuple(
            WorkUnitDefinition.from_dict(unit)
            for unit in raw_units
            if isinstance(unit, dict)
        )
        if len(work_units) != len(raw_units):
            raise ValueError("every work unit must be an object")
        if tuple(unit.metric for unit in work_units) != METRIC_NAMES:
            raise ValueError("work units must contain every metric in canonical order")
        boundaries = raw["synchronization_boundaries"]
        if not isinstance(boundaries, list) or not all(
            isinstance(value, str) and value for value in boundaries
        ):
            raise ValueError("synchronization_boundaries must be non-empty strings")
        return cls(
            schema_version=1,
            model=mappings["model"],
            optimizer=mappings["optimizer"],
            precision=mappings["precision"],
            loader=mappings["loader"],
            compilation=mappings["compilation"],
            generation=mappings["generation"],
            semantic_identities=mappings["semantic_identities"],
            native_representation_identities=mappings[
                "native_representation_identities"
            ],
            work_units=work_units,
            synchronization_boundaries=tuple(boundaries),
            required_environment=mappings["required_environment"],
            software_requirements=mappings["software_requirements"],
        )


@dataclass(frozen=True, slots=True)
class RawTrial:
    schema_version: int
    metric: MetricName
    side: Side
    attempt_index: int
    pair_index: int
    process_order: int
    source_commit: str
    source_clean: bool
    harness_commit: str
    harness_clean: bool
    harness_identity: str
    canonical_workload_identity: str
    native_configuration: dict[str, JsonValue]
    native_representation_identity: str
    canonical_row_identity: str
    canonical_input_identity: str
    canonical_projection: dict[str, JsonValue]
    execution_order_identity: str
    initial_parameter_identity: str
    comparison_target: str
    warmup_units: int
    measured_units: int
    elapsed_seconds: float
    value: float
    startup_verification_seconds: float | None
    compilation_seconds: float | None
    peak_memory_bytes: int | None
    synchronization_boundaries: tuple[str, ...]
    software_versions: dict[str, str]
    hardware: dict[str, JsonValue]
    environment_status: dict[str, JsonValue]
    evidence_session_identity: str
    journal_attempt_index: int
    child_measurement: dict[str, JsonValue]
    post_exit_observation: dict[str, JsonValue]
    post_exit_recovery_samples: tuple[dict[str, JsonValue], ...]
    post_exit_recovery: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        raw = asdict(self)
        raw["synchronization_boundaries"] = list(self.synchronization_boundaries)
        raw["post_exit_recovery_samples"] = list(self.post_exit_recovery_samples)
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> RawTrial:
        expected = {field.name for field in fields(cls)}
        if set(raw) != expected:
            raise ValueError("raw trial has an invalid field set")
        if type(raw["schema_version"]) is not int:
            raise ValueError("raw trial schema_version must be an integer")
        if raw["schema_version"] != 3:
            raise ValueError("unsupported raw trial schema version")
        payload = validate_trial_payload(
            {name: raw[name] for name in TRIAL_PAYLOAD_FIELDS}
        )
        session_identity = raw["evidence_session_identity"]
        if (
            not isinstance(session_identity, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", session_identity) is None
        ):
            raise ValueError("evidence_session_identity must be a SHA-256 identity")
        journal_attempt_index = raw["journal_attempt_index"]
        if type(journal_attempt_index) is not int or journal_attempt_index < 0:
            raise ValueError("journal_attempt_index must be a non-negative integer")
        mappings = {}
        for name in ("software_versions", "hardware", "environment_status"):
            item = raw[name]
            if not isinstance(item, dict):
                raise ValueError(f"{name} must be an object")
            mappings[name] = dict(item)
        evidence = {}
        for name in (
            "child_measurement",
            "post_exit_observation",
            "post_exit_recovery",
        ):
            item = raw[name]
            if not isinstance(item, dict):
                raise ValueError(f"{name} must be an object")
            evidence[name] = dict(item)
        raw_samples = raw["post_exit_recovery_samples"]
        if not isinstance(raw_samples, list) or not all(
            isinstance(item, dict) for item in raw_samples
        ):
            raise ValueError("post_exit_recovery_samples must be a list of objects")
        return cls(
            schema_version=3,
            metric=payload["metric"],
            side=payload["side"],
            attempt_index=payload["attempt_index"],
            pair_index=payload["pair_index"],
            process_order=payload["process_order"],
            source_commit=payload["source_commit"],
            source_clean=True,
            harness_commit=payload["harness_commit"],
            harness_clean=True,
            harness_identity=payload["harness_identity"],
            canonical_workload_identity=payload["canonical_workload_identity"],
            native_configuration=payload["native_configuration"],
            native_representation_identity=payload["native_representation_identity"],
            canonical_row_identity=payload["canonical_row_identity"],
            canonical_input_identity=payload["canonical_input_identity"],
            canonical_projection=payload["canonical_projection"],
            execution_order_identity=payload["execution_order_identity"],
            initial_parameter_identity=payload["initial_parameter_identity"],
            comparison_target=payload["comparison_target"],
            warmup_units=payload["warmup_units"],
            measured_units=payload["measured_units"],
            elapsed_seconds=payload["elapsed_seconds"],
            value=payload["value"],
            startup_verification_seconds=payload["startup_verification_seconds"],
            compilation_seconds=payload["compilation_seconds"],
            peak_memory_bytes=payload["peak_memory_bytes"],
            synchronization_boundaries=tuple(payload["synchronization_boundaries"]),
            software_versions=mappings["software_versions"],
            hardware=mappings["hardware"],
            environment_status=mappings["environment_status"],
            evidence_session_identity=session_identity,
            journal_attempt_index=journal_attempt_index,
            child_measurement=evidence["child_measurement"],
            post_exit_observation=evidence["post_exit_observation"],
            post_exit_recovery_samples=tuple(dict(item) for item in raw_samples),
            post_exit_recovery=evidence["post_exit_recovery"],
        )


@dataclass(frozen=True, slots=True)
class TrialPair:
    pair_index: int
    metric: MetricName
    reference: RawTrial
    candidate: RawTrial

    def __post_init__(self) -> None:
        if self.reference.side != "reference" or self.candidate.side != "candidate":
            raise ValueError("trial pair sides are reversed")
        if self.reference.pair_index != self.pair_index:
            raise ValueError("reference pair index does not match")
        if self.candidate.pair_index != self.pair_index:
            raise ValueError("candidate pair index does not match")
        if self.reference.metric != self.metric or self.candidate.metric != self.metric:
            raise ValueError("trial pair metrics do not match")
