from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass, fields
from typing import Literal, Mapping, TypeAlias

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
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
        if not isinstance(measured_units, int) or measured_units <= 0:
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

    def to_dict(self) -> dict[str, JsonValue]:
        raw = asdict(self)
        raw["synchronization_boundaries"] = list(self.synchronization_boundaries)
        return raw

    @classmethod
    def from_dict(cls, raw: Mapping[str, object]) -> RawTrial:
        expected = {field.name for field in fields(cls)}
        if set(raw) != expected:
            raise ValueError("raw trial has an invalid field set")
        metric = raw["metric"]
        side = raw["side"]
        if metric not in METRIC_NAMES:
            raise ValueError(f"unsupported benchmark metric: {metric!r}")
        if side not in ("reference", "candidate"):
            raise ValueError(f"unsupported benchmark side: {side!r}")
        integer_fields = (
            "schema_version",
            "pair_index",
            "attempt_index",
            "process_order",
            "warmup_units",
            "measured_units",
        )
        if any(not isinstance(raw[name], int) for name in integer_fields):
            raise ValueError("raw trial integer fields must be integers")
        if raw["schema_version"] != 1:
            raise ValueError("unsupported raw trial schema version")
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
            if (
                not isinstance(value, str)
                or re.fullmatch(r"[0-9a-f]{40}", value) is None
            ):
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
        if (
            not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("elapsed_seconds must be finite and positive")
        if (
            not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError("value must be finite and positive")
        optional_float_fields = (
            "startup_verification_seconds",
            "compilation_seconds",
        )
        for name in optional_float_fields:
            item = raw[name]
            if item is not None and (
                not isinstance(item, (int, float))
                or not math.isfinite(item)
                or item < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        peak = raw["peak_memory_bytes"]
        if peak is not None and (not isinstance(peak, int) or peak < 0):
            raise ValueError("peak_memory_bytes must be non-negative")
        mappings = {}
        for name in (
            "native_configuration",
            "canonical_projection",
            "software_versions",
            "hardware",
            "environment_status",
        ):
            item = raw[name]
            if not isinstance(item, dict):
                raise ValueError(f"{name} must be an object")
            mappings[name] = dict(item)
        boundaries = raw["synchronization_boundaries"]
        if not isinstance(boundaries, list) or not all(
            isinstance(item, str) and item for item in boundaries
        ):
            raise ValueError("synchronization_boundaries must be a string list")
        return cls(
            schema_version=1,
            metric=metric,
            side=side,
            attempt_index=raw["attempt_index"],
            pair_index=raw["pair_index"],
            process_order=raw["process_order"],
            source_commit=raw["source_commit"],
            source_clean=True,
            harness_commit=raw["harness_commit"],
            harness_clean=True,
            harness_identity=raw["harness_identity"],
            canonical_workload_identity=raw["canonical_workload_identity"],
            native_configuration=mappings["native_configuration"],
            native_representation_identity=raw["native_representation_identity"],
            canonical_row_identity=raw["canonical_row_identity"],
            canonical_input_identity=raw["canonical_input_identity"],
            canonical_projection=mappings["canonical_projection"],
            execution_order_identity=raw["execution_order_identity"],
            initial_parameter_identity=raw["initial_parameter_identity"],
            comparison_target=comparison_target,
            warmup_units=raw["warmup_units"],
            measured_units=raw["measured_units"],
            elapsed_seconds=float(elapsed),
            value=float(value),
            startup_verification_seconds=(
                None
                if raw["startup_verification_seconds"] is None
                else float(raw["startup_verification_seconds"])
            ),
            compilation_seconds=(
                None
                if raw["compilation_seconds"] is None
                else float(raw["compilation_seconds"])
            ),
            peak_memory_bytes=peak,
            synchronization_boundaries=tuple(boundaries),
            software_versions=mappings["software_versions"],
            hardware=mappings["hardware"],
            environment_status=mappings["environment_status"],
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
