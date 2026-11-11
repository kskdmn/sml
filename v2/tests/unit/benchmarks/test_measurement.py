from types import SimpleNamespace

from v2.benchmarks.runner import (
    measure_native_process,
    process_order,
)
from v2.benchmarks.workload import (
    build_canonical_workload,
)


def test_runner_alternates_fresh_process_order_by_pair():
    assert process_order(0) == ("reference", "candidate")
    assert process_order(1) == ("candidate", "reference")
    assert process_order(2) == ("reference", "candidate")


def test_measurement_protocol_compiles_warms_and_times_with_explicit_syncs():
    events = []
    adapter = SimpleNamespace(
        run_warmup=lambda metric, native, units: events.append(
            ("warmup", metric, native, units)
        ),
        run_measured=lambda metric, native, units: (
            events.append(("measured", metric, native, units)) or 8_192.0
        ),
    )
    clock_values = iter((1.0, 3.0, 10.0, 12.0))

    measurement = measure_native_process(
        adapter=adapter,
        metric="pretraining-end-to-end",
        native_workload="native",
        warmup_units=2,
        measured_units=1,
        synchronize=lambda: events.append(("synchronize",)),
        clock=lambda: next(clock_values),
        peak_memory=lambda: 123,
        reset_peak_memory=lambda: events.append(("reset-peak-memory",)),
    )

    assert events == [
        ("synchronize",),
        ("warmup", "pretraining-end-to-end", "native", 1),
        ("synchronize",),
        ("warmup", "pretraining-end-to-end", "native", 1),
        ("synchronize",),
        ("warmup", "pretraining-end-to-end", "native", 1),
        ("synchronize",),
        ("reset-peak-memory",),
        ("synchronize",),
        ("measured", "pretraining-end-to-end", "native", 1),
        ("synchronize",),
    ]
    assert measurement.elapsed_seconds == 2.0
    assert measurement.compilation_seconds == 2.0
    assert measurement.value == 4_096.0
    assert measurement.peak_memory_bytes == 123


def test_peak_memory_runs_one_measured_step_after_five_warmups_and_peak_reset():
    workload = build_canonical_workload()
    peak_unit = next(
        unit for unit in workload.work_units if unit.metric == "peak-metal-memory"
    )
    events = []
    adapter = SimpleNamespace(
        run_warmup=lambda _metric, _native, units: events.append(("warmup", units)),
        run_measured=lambda _metric, _native, units: (
            events.append(("measured", units)) or 8_192.0
        ),
    )

    measurement = measure_native_process(
        adapter=adapter,
        metric="peak-metal-memory",
        native_workload="native",
        warmup_units=workload.compilation["warmup_units"],
        measured_units=peak_unit.measured_units,
        synchronize=lambda: events.append(("synchronize",)),
        clock=iter((0.0, 1.0, 2.0, 3.0)).__next__,
        peak_memory=lambda: 7_875_602_848,
        reset_peak_memory=lambda: events.append(("reset-peak-memory",)),
    )

    operations = [event for event in events if event != ("synchronize",)]
    assert operations == [
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("warmup", 1),
        ("reset-peak-memory",),
        ("measured", 1),
    ]
    assert measurement.value == 7_875_602_848.0


def test_checkpoint_latency_is_elapsed_time_per_measured_publication():
    measurement = measure_native_process(
        adapter=SimpleNamespace(
            run_warmup=lambda *_args: None,
            run_measured=lambda *_args: 3.0,
        ),
        metric="checkpoint-pause",
        native_workload=object(),
        warmup_units=0,
        measured_units=3,
        synchronize=lambda: None,
        clock=iter((0.0, 1.0, 2.0, 8.0)).__next__,
        peak_memory=lambda: 700,
        reset_peak_memory=lambda: None,
    )
    assert measurement.value == 2.0
