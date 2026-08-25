# V2 Strict Evaluation Artifact Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace metadata-only evaluation output with a strict, immutable, self-identifying artifact containing complete lm-eval results and reproducible task/request/dataset provenance.

**Architecture:** A new `evaluation_result.py` owns JSON normalization, frozen schemas, identities, strict reading, and atomic publication. `evaluation.py` remains the runtime owner: it creates a recording `TaskManager`, records actual lm-eval request order, resolves task YAML/config/dataset provenance, captures the complete `simple_evaluate` return, and assembles the strict result.

**Tech Stack:** Python 3.12.13, lm-evaluation-harness 0.4.12 `TaskManager`, NumPy scalar normalization, canonical SML JSON/structured identities, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-25-v2-final-acceptance-remediation-design.md`

## Global Constraints

- Continue in the current checkout; do not create a worktree.
- Use `uv run`; run every pytest command outside the sandbox.
- Do not edit top-level project files or `uv.lock`.
- Import lm-eval, PyYAML, and datasets-facing types lazily inside the evaluation command path.
- Support exactly HellaSwag and WinoGrande; unsupported task/request methods fail closed.
- Persist no absolute output or provider-package filesystem path.
- Preserve the complete normalized provider return; do not curate away provider fields.
- Use lm-eval 0.4.12's real `TaskManager.load()`, task `config.to_dict()`, populated `task.instances`, and result mapping. The checked-in provider stub mirrors those interfaces offline.

---

## File Structure

- Create `v2/src/sml/evaluation_result.py`: closed JSON type, frozen evaluation records, identity projection, strict canonical reader, immutable publisher.
- Modify `v2/src/sml/evaluation.py`: lazy provider facade, recording task manager/request recorder, task-source/config/dataset provenance, result assembly.
- Modify `v2/src/sml/__init__.py`: export the four evaluation artifact record types.
- Create `v2/tests/unit/test_evaluation_result.py`: schema, normalization, identity, tamper, and publication tests.
- Modify `v2/tests/unit/test_evaluation.py`: provider recording, provenance, complete metric capture, fail-closed runtime tests.
- Modify `v2/tests/integration/test_evaluation_workflow.py`: pinned model identity and immutable result integration behavior.
- Modify `v2/tests/integration/test_part2_swag_flow.py`: export evaluation assertions against task records/provider metrics.
- Modify `v2/tests/integration/test_cli_workflows.py`: strict persisted JSON assertions for base and export evaluation.
- Modify `v2/tests/unit/test_package.py`: exact public export set.
- Modify `v2/tests/fixtures/provider_stubs/lm_eval/__init__.py`: deterministic 0.4.12-shaped offline `TaskManager`, tasks, instances, datasets, and complete result.
- Create `v2/tests/fixtures/provider_stubs/lm_eval/tasks/common.yaml`: included prompt/metric fixture.
- Create `v2/tests/fixtures/provider_stubs/lm_eval/tasks/hellaswag.yaml`: task source with an include closure.
- Create `v2/tests/fixtures/provider_stubs/lm_eval/tasks/winogrande.yaml`: second task source.
- Modify `v2/README.md`: document strict evaluation content and output-path exclusion.

## Frozen Interfaces

`v2/src/sml/evaluation_result.py` owns these exact public records and functions:

```text
JsonScalar: null, bool, int, finite float, or str
JsonValue: JsonScalar, immutable tuple of JsonValue, or immutable string-keyed mapping of JsonValue
JsonObject: immutable string-keyed mapping of JsonValue

EvaluationSourceIdentity:
  logical_name: str
  content_identity: str

EvaluationProviderVersion:
  name: str
  version: str

EvaluationTaskRecord:
  task_name: str
  task_identity: str
  task_yaml: EvaluationSourceIdentity
  include_template_closure: immutable tuple of EvaluationSourceIdentity
  task_metadata_version: str
  prompt_config: Mapping[str, JsonValue]
  few_shot_config: Mapping[str, JsonValue]
  generation_config: Mapping[str, JsonValue]
  metric_normalization_config: Mapping[str, JsonValue]
  seeds: Mapping[str, JsonValue]
  limit: int | None
  ordered_request_identity: str
  lm_eval_package_version: str
  lm_eval_source_commit: str | None
  dataset_revision: str
  dataset_fingerprint: str
  provider_versions: immutable tuple of EvaluationProviderVersion
  metric_payload: Mapping[str, JsonValue]

EvaluationResult:
  kind: Literal["evaluation-result"]
  version: Literal[1]
  identity: str
  model: ModelIdentity
  tasks: immutable tuple of EvaluationTaskRecord
  provider_result: Mapping[str, JsonValue]

normalize_json_value(value: object, context: str) -> JsonValue
evaluation_task_identity(record: EvaluationTaskRecord) -> str
evaluation_result_identity(result: EvaluationResult) -> str
evaluation_result_bytes(result: EvaluationResult) -> bytes
read_evaluation_result(path: Path) -> EvaluationResult
publish_evaluation_result(path: Path, result: EvaluationResult) -> None
```

All four records are frozen, slotted dataclasses. The implementation steps below
provide the complete behavior and tests for every listed function.

## Task 1: Closed JSON Values and Frozen Result Records

**Files:**
- Create: `v2/src/sml/evaluation_result.py`
- Create: `v2/tests/unit/test_evaluation_result.py`

**Interfaces:**
- Consumes: `ModelIdentity`, `VerificationLevel`, `canonical_json_bytes`, and `structured_identity`.
- Produces: the four frozen records plus `normalize_json_value`, `evaluation_task_identity`, and `evaluation_result_identity` from the frozen interface above.

- [ ] **Step 1: Write failing normalization and record-validation tests**

Add tests with concrete model/task builders:

```python
def model_identity() -> ModelIdentity:
    return ModelIdentity(
        artifact_kind="export",
        run_identity=None,
        step=7,
        checkpoint_identity=None,
        run_step_identity=None,
        tokenizer_identity="sha256:" + "1" * 64,
        verification=VerificationLevel.FULL,
    )


def test_json_normalization_is_closed_finite_and_deeply_immutable() -> None:
    normalized = normalize_json_value(
        {"z": np.int64(3), "a": [True, np.float32(1.25), None]},
        context="provider result",
    )
    assert tuple(normalized) == ("a", "z")
    assert normalized["a"] == (True, 1.25, None)
    with pytest.raises(TypeError):
        normalized["z"] = 4
    for invalid in (float("nan"), float("inf"), b"bytes", Path("local"), {1: "x"}):
        with pytest.raises((TypeError, ValueError)):
            normalize_json_value(invalid, context="provider result")


def test_task_and_result_identities_cover_metrics_requests_and_model() -> None:
    task = make_task_record(metric_payload={"acc,none": 0.5})
    task = replace(task, task_identity=evaluation_task_identity(task))
    result = make_result(model=model_identity(), tasks=(task,))
    result = replace(result, identity=evaluation_result_identity(result))
    changed_metric = replace(task, metric_payload={"acc,none": 0.75})
    changed_request = replace(task, ordered_request_identity="sha256:" + "2" * 64)
    assert evaluation_result_identity(
        replace(result, identity="sha256:" + "0" * 64, tasks=(changed_metric,))
    ) != result.identity
    assert evaluation_task_identity(changed_request) != task.task_identity
    assert evaluation_result_identity(
        replace(result, identity="sha256:" + "0" * 64, model=replace(model_identity(), step=8))
    ) != result.identity
```

`make_task_record()` supplies every exact field with deterministic nonempty
values, sorted provider records, and immutable normalized mappings;
`make_result()` supplies kind `evaluation-result`, version `1`, a fixed zero
identity sentinel, the given model/tasks, and `{"results": {"hellaswag": {"acc,none":
0.5}}}` as provider output.

- [ ] **Step 2: Run the focused test and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_evaluation_result.py -q
```

Expected: collection fails because `sml.evaluation_result` and its records do
not exist.

- [ ] **Step 3: Implement closed normalization and frozen records**

Implement scalar handling in this order so `bool` never enters the integer
branch:

```python
def normalize_json_value(value: object, *, context: str) -> JsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, np.generic):
        return normalize_json_value(value.item(), context=context)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{context} object keys must be strings")
        return MappingProxyType(
            {
                key: normalize_json_value(value[key], context=f"{context}.{key}")
                for key in sorted(value)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            normalize_json_value(item, context=f"{context}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{context} contains unsupported value {type(value).__name__}")
```

Every record `__post_init__` validates exact literals, nonempty strings,
SHA-256 identities, positive-or-null limit, unique sorted provider names,
nonempty ordered tasks with unique task names, and the expected concrete nested record types. It
re-normalizes every JSON mapping and installs the immutable value with
`object.__setattr__`.

Implement task identity with domain `sml-evaluation-task-v1`, excluding only
`task_identity` and `metric_payload`. Implement result identity with domain
`sml-evaluation-result-v1`, excluding only `identity` and including metrics and
the complete provider result. Serialize `ModelIdentity.verification` as its
string value in both projections.

- [ ] **Step 4: Run focused tests and format the new files**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/test_evaluation_result.py -q
uv run ruff check v2/src/sml/evaluation_result.py v2/tests/unit/test_evaluation_result.py
uv run ruff format --check v2/src/sml/evaluation_result.py v2/tests/unit/test_evaluation_result.py
```

Expected: all tests and Ruff checks pass.

- [ ] **Step 5: Commit the schema unit**

```bash
git add v2/src/sml/evaluation_result.py v2/tests/unit/test_evaluation_result.py
git commit -m "feat(v2): define strict evaluation result schema"
```

## Task 2: Strict Reader and Immutable Publication

**Files:**
- Modify: `v2/src/sml/evaluation_result.py`
- Modify: `v2/tests/unit/test_evaluation_result.py`

**Interfaces:**
- Consumes: Task 1 records and identities.
- Produces: `evaluation_result_bytes`, `read_evaluation_result`, and `publish_evaluation_result` with canonical, no-replace behavior.

- [ ] **Step 1: Write failing strict-reader and publication tests**

```python
def test_reader_rejects_extra_duplicate_noncanonical_and_tampered_identity(tmp_path: Path) -> None:
    result = identified_result()
    canonical = evaluation_result_bytes(result)
    path = tmp_path / "evaluation.json"
    path.write_bytes(canonical)
    assert read_evaluation_result(path) == result

    documents = (
        canonical.replace(b'"version":1', b'"version":1,"extra":true'),
        canonical.replace(b'"kind":"evaluation-result"', b'"kind":"evaluation-result","kind":"evaluation-result"'),
        json.dumps(json.loads(canonical), indent=2, sort_keys=True).encode() + b"\n",
        canonical.replace(result.identity.encode(), ("sha256:" + "f" * 64).encode()),
    )
    for index, payload in enumerate(documents):
        bad = tmp_path / f"bad-{index}.json"
        bad.write_bytes(payload)
        with pytest.raises(SMLArtifactError):
            read_evaluation_result(bad)


def test_publication_is_idempotent_but_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "evaluation.json"
    result = identified_result()
    publish_evaluation_result(path, result)
    first = path.read_bytes()
    publish_evaluation_result(path, result)
    assert path.read_bytes() == first
    changed = replace(result, identity="sha256:" + "0" * 64, provider_result={"results": {}})
    changed = replace(changed, identity=evaluation_result_identity(changed))
    with pytest.raises(SMLRuntimeError, match="collision"):
        publish_evaluation_result(path, changed)
    assert path.read_bytes() == first
```

Add a publication spy proving the destination is never passed to `os.replace`
and a persisted-byte assertion proving no test output directory appears in the
JSON.

- [ ] **Step 2: Run the strict artifact tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_evaluation_result.py -q
```

Expected: failures report missing reader/serializer/publisher behavior.

- [ ] **Step 3: Implement exact payload conversion and strict parsing**

Build explicit payload dictionaries for every record. The result payload field
set is exactly `kind`, `version`, `identity`, `model`, `tasks`, and
`provider_result`; task/source/provider/model nested field sets are also exact.
Before serialization, recompute and require every task identity and the result
identity so a caller cannot publish a merely well-shaped but self-inconsistent
record.
Use duplicate-rejecting `object_pairs_hook`, non-finite `parse_constant`
rejection, and `canonical_json_bytes(payload) + b"\n"` as the only accepted
file representation.

The core reader order is:

```python
raw_bytes = path.read_bytes()
raw = json.loads(
    raw_bytes.decode("utf-8"),
    object_pairs_hook=_json_object_no_duplicates,
    parse_constant=_reject_json_constant,
)
result = _result_from_payload(raw)
if evaluation_result_bytes(result) != raw_bytes:
    raise SMLArtifactError("evaluation result is not canonical")
if evaluation_result_identity(result) != result.identity:
    raise SMLArtifactError("evaluation result identity mismatch")
return result
```

Task parsing independently recomputes every `task_identity`. Convert
verification only through `VerificationLevel(value)` and reject booleans as
integers for version, step, and limit fields.

- [ ] **Step 4: Implement durable no-replace publication**

Create missing destination parents, then use `mkstemp` in the destination
parent, write bytes, flush, file-`fsync`, and publish with
`os.link(temporary, destination)`. On `FileExistsError`, strictly
read the destination and accept only an equal `EvaluationResult`; invalid or
different existing bytes raise `SMLRuntimeError("evaluation output collision: " + str(destination))`.
Always unlink the temporary and `fsync` the parent after a successful
new link or validated idempotent reuse.

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/test_evaluation_result.py -q
uv run ruff check v2/src/sml/evaluation_result.py v2/tests/unit/test_evaluation_result.py
uv run ruff format --check v2/src/sml/evaluation_result.py v2/tests/unit/test_evaluation_result.py
```

Expected: strict/tamper/publication tests pass.

- [ ] **Step 5: Commit strict persistence**

```bash
git add v2/src/sml/evaluation_result.py v2/tests/unit/test_evaluation_result.py
git commit -m "feat(v2): publish immutable evaluation results"
```

## Task 3: Record lm-eval Task, Request, and Dataset Provenance

**Files:**
- Modify: `v2/src/sml/evaluation.py`
- Modify: `v2/tests/unit/test_evaluation.py`
- Modify: `v2/tests/fixtures/provider_stubs/lm_eval/__init__.py`
- Create: `v2/tests/fixtures/provider_stubs/lm_eval/tasks/common.yaml`
- Create: `v2/tests/fixtures/provider_stubs/lm_eval/tasks/hellaswag.yaml`
- Create: `v2/tests/fixtures/provider_stubs/lm_eval/tasks/winogrande.yaml`

**Interfaces:**
- Consumes: lm-eval 0.4.12 `TaskManager`, `TaskManager.task_index`, `Entry.yaml_path`, `Task.config.to_dict()`, `Task.dataset`, and request `task_name/doc_id/repeats/args`.
- Produces: `_LMEvalProvider`, `_RecordingTaskManager`, `_EvaluationRequestRecorder`, and `_resolve_task_record` returning `EvaluationTaskRecord`.

- [ ] **Step 1: Write failing provider-recording tests**

```python
def test_request_recorder_hashes_actual_order_per_task() -> None:
    recorder = evaluation._EvaluationRequestRecorder()
    first = SimpleNamespace(task_name="hellaswag", doc_id=2, repeats=1, args=("a", " b"))
    second = SimpleNamespace(task_name="hellaswag", doc_id=1, repeats=1, args=("c", " d"))
    recorder.record("loglikelihood", (first, second))
    forward = recorder.identity_for("hellaswag")
    reversed_recorder = evaluation._EvaluationRequestRecorder()
    reversed_recorder.record("loglikelihood", (second, first))
    assert reversed_recorder.identity_for("hellaswag") != forward


def test_task_provenance_hashes_yaml_include_config_and_dataset(fake_provider) -> None:
    manager = evaluation._RecordingTaskManager(fake_provider.make_task_manager())
    loaded = manager.load(["hellaswag"])
    task = loaded["tasks"]["hellaswag"]
    recorder = evaluation._EvaluationRequestRecorder()
    recorder.record("loglikelihood", tuple(task.instances))
    record = evaluation._resolve_task_record(
        task_name="hellaswag",
        task=task,
        provider=fake_provider,
        manager=manager,
        recorder=recorder,
        provider_result={"results": {"hellaswag": {"acc,none": 0.5}}, "git_hash": "abc123"},
        limit=2,
        padding="right",
        seeds=evaluation._evaluation_seeds(),
        provider_versions=evaluation._provider_versions(),
    )
    assert record.task_yaml.logical_name.endswith("hellaswag.yaml")
    assert [source.logical_name for source in record.include_template_closure] == ["tasks/common.yaml"]
    assert record.dataset_revision == "version:1.0.0"
    assert record.dataset_fingerprint.startswith("sha256:")
    assert record.metric_payload == {"acc,none": 0.5}
```

Add failure cases for missing YAML path, include cycle, missing request task name,
missing dataset split fingerprint, empty dataset version, missing task metric,
and ambiguous duplicate provider metric keys.

- [ ] **Step 2: Run the provider tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_evaluation.py -q
```

Expected: tests fail because the provider facade, recording manager, recorder,
and task resolver are absent.

- [ ] **Step 3: Implement the lazy provider facade and recording manager**

`_import_lm_eval()` lazily imports `lm_eval.simple_evaluate`,
`lm_eval.tasks.TaskManager`, and
`lm_eval.tasks._yaml_loader.load_yaml`, then returns:

```python
@dataclass(frozen=True, slots=True)
class _LMEvalProvider:
    simple_evaluate: Callable
    task_manager_type: type
    load_yaml: Callable
    package_root: Path
    package_version: str

    def make_task_manager(self) -> object:
        return self.task_manager_type()
```

`package_root` is the installed `lm_eval` package directory and never enters a
serialized result. `_RecordingTaskManager` delegates unknown attributes to its
inner manager and stores the exact mapping returned by `load()`; a second load
or missing `tasks` mapping is an error.

The offline stub implements the same surface. Its task objects expose
`task_name`, a `config.to_dict()` result, ordered `instances`, and a dataset
mapping whose used split has `info.version == "1.0.0"` and a deterministic
`_fingerprint`. Its `TaskManager.task_index` points at the three checked-in YAML
files. Stub `simple_evaluate` calls `task_manager.load(tasks)`, sends requests
with task metadata through both LM methods, and returns deterministic
`results/configs/versions/n-shot/higher_is_better/n-samples/config/git_hash/date`
fields.

- [ ] **Step 4: Implement actual request-order recording**

Add a recorder to `SMLEvalLM` and its installed `LM` wrapper. Immediately before
dispatch, record each request in the order received:

```python
self._recorder.record("loglikelihood", tuple(requests))
```

and:

```python
self._recorder.record("generate_until", tuple(requests))
```

Each normalized request document contains exactly `request_type`, `task_name`,
`doc_id`, `repeats`, and normalized `args`. `identity_for(task_name)` hashes the
ordered tuple under `sml-evaluation-requests-v1` and fails for no recorded
requests or metadata that cannot identify the owning task.

- [ ] **Step 5: Implement task source/config/dataset resolution**

Resolve the primary YAML from `manager.task_index[task_name].yaml_path`. Read its
bytes for `EvaluationSourceIdentity`; call `provider.load_yaml(path,
resolve_func=False, recursive=False)` only to obtain the declared `include`
field. Recursively visit includes in declaration order, reject cycles and paths
outside `provider.package_root`, and record logical names relative to that root.

Use `task.config.to_dict()`, the explicit `padding` argument, and the fixed
provider invocation constants to normalize these exact projections:

```text
prompt_config:
  output_type, description, process_docs, doc_to_text, doc_to_target,
  doc_to_choice, target_delimiter, fewshot_delimiter, gen_prefix,
  system_instruction=null, apply_chat_template=false,
  adapter_padding=config.padding
few_shot_config:
  num_fewshot, fewshot_split, fewshot_config, fewshot_as_multiturn=true
generation_config:
  generation_kwargs, provider_gen_kwargs=null
metric_normalization_config:
  metric_list, filter_list, repeats, should_decontaminate,
  doc_to_decontamination_query, bootstrap_iters=100000,
  log_samples=true, predict_only=false
```

Missing optional provider config values are represented explicitly as JSON
`null`; callable fields use lm-eval's serialized `to_dict()` value, never
`repr(callable)` from SML.

Select the evaluation split from `validation_split`, else `test_split`; add the
few-shot split only when the effective `num_fewshot` is positive. Require a
nonempty `_fingerprint` and nonempty `info.version` for every selected dataset
split. Use explicit `dataset_kwargs["revision"]` when present; otherwise require
one common dataset version and publish `version:<value>`. Hash ordered
`(split_name, fingerprint)` pairs under `sml-evaluation-dataset-v1`.

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_evaluation.py -q
uv run ruff check v2/src/sml/evaluation.py v2/tests/unit/test_evaluation.py v2/tests/fixtures/provider_stubs/lm_eval
uv run ruff format --check v2/src/sml/evaluation.py v2/tests/unit/test_evaluation.py v2/tests/fixtures/provider_stubs/lm_eval
```

Expected: provenance, failure, batching, and lazy-import tests pass.

- [ ] **Step 6: Commit provider provenance**

```bash
git add v2/src/sml/evaluation.py v2/tests/unit/test_evaluation.py v2/tests/fixtures/provider_stubs/lm_eval
git commit -m "feat(v2): resolve evaluation provenance"
```

## Task 4: Assemble and Publish Complete Evaluation Results

**Files:**
- Modify: `v2/src/sml/evaluation.py`
- Modify: `v2/src/sml/__init__.py`
- Modify: `v2/tests/unit/test_evaluation.py`
- Modify: `v2/tests/unit/test_package.py`
- Modify: `v2/tests/integration/test_evaluation_workflow.py`
- Modify: `v2/tests/integration/test_part2_swag_flow.py`
- Modify: `v2/tests/integration/test_cli_workflows.py`
- Modify: `v2/README.md`

**Interfaces:**
- Consumes: Tasks 1-3 schemas, publisher, provider facade, loaded tasks, request recorder, and complete provider mapping.
- Produces: `evaluate(config) -> EvaluationResult`; `config.output` remains the destination but is absent from the returned/persisted semantic record.

- [ ] **Step 1: Write failing complete-result and output-exclusion tests**

```python
def test_evaluate_preserves_complete_provider_result_and_task_metrics(
    tiny_pretraining_run: Path, fake_lm_eval, tmp_path: Path
) -> None:
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)
    result = evaluate(config)
    persisted = read_evaluation_result(config.output)
    assert persisted == result
    assert result.provider_result["configs"] == fake_lm_eval.result["configs"]
    assert result.tasks[0].metric_payload == {"acc,none": 0.5, "acc_stderr,none": 0.01}
    assert str(tmp_path).encode() not in config.output.read_bytes()
    assert result.identity == evaluation_result_identity(result)


def test_evaluate_fails_before_publish_when_provider_result_is_incomplete(
    tiny_pretraining_run: Path, fake_lm_eval, tmp_path: Path
) -> None:
    fake_lm_eval.result = {"configs": {}}
    config = tiny_evaluation_config(tiny_pretraining_run, tmp_path)
    with pytest.raises(SMLRuntimeError, match="results"):
        evaluate(config)
    assert not config.output.exists()
```

Update identity-pinning/idempotence tests to use `config.output` and task records
rather than `result.output` or string task tuples. Add an `EvaluationConfig`
test rejecting duplicate task names before provider execution; a single
provider result key can never ambiguously satisfy two requested records.

- [ ] **Step 2: Run evaluation unit/integration tests and verify RED**

Run outside the sandbox:

```bash
uv run pytest v2/tests/unit/test_evaluation.py v2/tests/integration/test_evaluation_workflow.py -q
```

Expected: metadata-only result assertions fail and the `simple_evaluate` return
is still discarded.

- [ ] **Step 3: Assemble the identified result from the provider return**

Create the provider, wrapped task manager, recorder, and lm-eval LM before the
call. Pass the manager plus explicit `system_instruction=None`,
`apply_chat_template=False`, `fewshot_as_multiturn=True`, `gen_kwargs=None`,
`bootstrap_iters=100000`, `log_samples=True`, `predict_only=False`, and seed
arguments `random_seed=0`, `numpy_random_seed=1234`,
`torch_random_seed=1234`, and `fewshot_random_seed=1234`. Capture the return and
require a mapping.

Build one task record per unique `config.tasks` entry in caller order. Require
`provider_result["results"]` to contain each requested task and no
unsupported task key. Set `lm_eval_source_commit` from a nonempty provider
`git_hash`, else `None`. Identify each task record, then construct and identify:

```python
result = EvaluationResult(
    kind="evaluation-result",
    version=1,
    identity=_ZERO_IDENTITY,
    model=session.model_identity,
    tasks=tuple(task_records),
    provider_result=normalize_json_value(raw_result, context="lm-eval result"),
)
result = replace(result, identity=evaluation_result_identity(result))
publish_evaluation_result(config.output, result)
return result
```

Delete the old metadata serializer, permissive reader, and output-bearing
`EvaluationResult`. Re-export the strict reader from `sml.evaluation` for API
continuity.

- [ ] **Step 4: Update package exports, CLI assertions, and README**

Export `EvaluationProviderVersion`, `EvaluationResult`,
`EvaluationSourceIdentity`, and `EvaluationTaskRecord` from `sml.evaluation` and
the package root; add them to the exact `EXPECTED_PUBLIC_TYPES` mapping.

Update CLI JSON assertions to require kind/version/identity, `tasks[0].task_name`,
the task metric payload, complete `provider_result`, resolved model identity,
and verification level for both base and export evaluations. Update the README
to state that output contains complete metrics and task/model/dataset/request
provenance and that the destination path is intentionally excluded.

- [ ] **Step 5: Run the complete evaluation gate**

Run outside the sandbox for pytest:

```bash
uv run pytest v2/tests/unit/test_evaluation_result.py v2/tests/unit/test_evaluation.py v2/tests/unit/test_package.py -q
uv run pytest v2/tests/integration/test_evaluation_workflow.py v2/tests/integration/test_part2_swag_flow.py v2/tests/integration/test_cli_workflows.py -q
uv run ruff check v2
uv run ruff format --check v2
```

Expected: strict schema/tamper/publication tests pass; base/export CLI results
contain complete provenance; the LoRA-to-export workflow still passes offline;
Ruff passes.

- [ ] **Step 6: Commit the complete evaluation artifact**

```bash
git add v2/src/sml/evaluation.py v2/src/sml/evaluation_result.py v2/src/sml/__init__.py v2/tests/unit/test_evaluation.py v2/tests/unit/test_evaluation_result.py v2/tests/unit/test_package.py v2/tests/integration/test_evaluation_workflow.py v2/tests/integration/test_part2_swag_flow.py v2/tests/integration/test_cli_workflows.py v2/tests/fixtures/provider_stubs/lm_eval v2/README.md
git commit -m "feat(v2): persist strict evaluation artifacts"
```

## Plan Completion Gate

- [ ] **Step 1: Verify the component diff and tracked state**

```bash
git diff HEAD^ --check
git status --short
```

Expected: no whitespace errors and no uncommitted tracked changes.

- [ ] **Step 2: Review against the evaluation section of the remediation spec**

Confirm the reviewed diff has no output path in persisted bytes, no dropped
provider result field, no unrecorded supported request, no unresolved dataset
provenance, no permissive JSON branch, and no overwrite publication path.

Expected: no Critical, Important, or Minor findings before starting the LoRA
plan.
