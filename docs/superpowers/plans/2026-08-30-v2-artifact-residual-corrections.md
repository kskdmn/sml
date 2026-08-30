# V2 Artifact Residual Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the five Important and three Minor V2 artifact residuals so a
new whole-component review can return zero findings without weakening the
explicit-randomness, retained-owner, or bounded-memory contracts.

**Architecture:** Keep the existing retained-descriptor architecture and add
only shared boundary primitives: bounded progress semantics, a persisted
counter-addressed forward-terminal RNG schedule, one stable retained-root kind
dispatcher, and stat-only closed-world enumeration before semantic proof/use.
Data loaders reuse fixed 1,024-row scans, and all artifact-controlled reads stay
bounded before allocation.

**Tech Stack:** Python 3.12.13, MLX, NumPy/mmap, pytest, Ruff, `uv run`.

**Spec:**
`docs/superpowers/specs/2026-08-30-v2-artifact-residual-corrections-design.md`

## Global Constraints

- Work in the current checkout; the user explicitly authorized it. Do not
  create a worktree for this correction wave.
- Update only `v2` code and V2 tests. Do not modify `pyproject.toml`, `uv.lock`,
  or another top-level project file.
- Preserve the binding forward contract: each active dropout site consumes one
  split in canonical order, forward returns the terminal next-unused key, and
  the checkpoint persists that exact returned key.
- Persist exactly
  `rng_schedule: "counter-addressed-forward-terminal-v1"` in both run checkpoint
  projections. There is no legacy reader, optional field, fallback, or
  conversion path.
- Keep progress independent of external data geometry and never open
  `diagnostic_data_locator` during portable verification.
- Keep FULL and MANIFEST_TRUSTED distinct; trusted verification performs no
  tensor-value reductions.
- Use strict TDD: run each focused RED before implementation, then the focused
  GREEN and the task's scoped regression set.
- Run every `uv run pytest` command outside the sandbox so MLX/Metal can use
  the Apple GPU.
- Use one fresh implementation worker at a time during subagent-driven
  execution. After each task, run specification review and code-quality review,
  with at most five ordinary fix rounds before escalating.
- Do not stage or commit
  `docs/superpowers/handoffs/2026-08-22-v2-performance-refactor-phase-2-handoff.md`;
  it is the sole inherited worktree modification and is updated separately at
  milestones.

---

## File and Responsibility Map

- `v2/src/sml/artifacts/semantics.py`: exact run projection, progress, and
  terminal-key semantic validation.
- `v2/src/sml/artifacts/manifest.py`: strict run schedule identity, stable JSON
  manifest lifecycle, and descriptor-relative candidate stat support.
- `v2/src/sml/artifacts/dispatch.py` (new): neutral exact-candidate retained-root
  dispatcher shared by recursive verification and inference.
- `v2/src/sml/artifacts/verify.py`: recursive verification routed through the
  neutral dispatcher.
- `v2/src/sml/artifacts/arrays.py`: bounded safetensors header parsing.
- `v2/src/sml/artifacts/checkpoint.py`: stat-only closed-world namespace proof
  and single semantic payload-open pass.
- `v2/src/sml/training/random.py`: O(1) counter key derivation retained as the
  per-microstep address primitive.
- `v2/src/sml/training/pretrain.py`: pre-forward key installation, exact
  forward-terminal persistence, schedule projection, and validated resume.
- `v2/src/sml/training/swag.py`: the equivalent LoRA/SWAG transition and resume
  rules.
- `v2/src/sml/data/pretraining.py`: stable nested-tokenizer consumption and
  fixed-row prepared-data scans.
- `v2/src/sml/data/swag.py`: safe owning-stream transfer and fixed-row SWAG
  reductions.
- `v2/src/sml/inference.py`: retained-root model-kind dispatch with ambiguity
  rejection.
- `v2/tests/unit/artifacts/test_manifest.py`: strict schedule and stable-reader
  lifecycle regressions.
- `v2/tests/unit/artifacts/test_recursive_verify.py`: progress, expected-key,
  recursive dispatch, mutation, and cleanup-precedence regressions.
- `v2/tests/unit/artifacts/test_arrays.py`: safetensors read/allocation cap.
- `v2/tests/unit/artifacts/test_checkpoint.py`: actual selected-payload open
  counts and closed-world rejection.
- `v2/tests/unit/data/test_swag.py`: active-lease transfer and chunk-bound tests.
- `v2/tests/integration/test_pretraining_data_workflow.py`: prepared nested
  manifest mutation and fixed-row reduction coverage.
- `v2/tests/integration/test_pretraining_workflow.py`: pretraining schedule,
  terminal key, resume-before-use, and partial-window coverage.
- `v2/tests/integration/test_swag_workflow.py`: LoRA schedule, mixed dropout,
  terminal key, resume-before-use, and example-based partial progress.
- `v2/tests/integration/test_inference_workflow.py`: exact dual-candidate
  rejection at both verification levels.

### Task 1: Accept Exactly Bounded Partial Progress

**Files:**
- Modify: `v2/src/sml/artifacts/semantics.py:218-248`
- Test: `v2/tests/unit/artifacts/test_recursive_verify.py:1360-1410`
- Test: `v2/tests/integration/test_pretraining_workflow.py:450-510`
- Test: `v2/tests/integration/test_swag_workflow.py:500-560`

**Interfaces:**
- Consumes: existing `_verify_progress(scalar, checkpoint_step, loader, lora)`.
- Produces: the same function and return type, with the exact bound
  `step <= microsteps <= step * gradient_accumulation_steps`; Task 2 consumes
  its returned `microsteps` for expected-key derivation.

- [ ] **Step 1: Turn the obsolete incomplete-window rejection into a valid
  partial-window regression**

  In `test_recursive_verify.py`, rename
  `test_full_pretraining_run_rejects_incomplete_resigned_accumulation_progress`
  and retain its re-signing helpers, but assert success for `step=1`,
  `microsteps=1`, and `rows=1` when the saved accumulation limit is two. Rebind
  the current counter-schedule trainer key as well so this task isolates the
  progress rule rather than relying on Task 2's later terminal-key semantics:

  ```python
  run_manifest = read_manifest(
      run,
      PretrainingRunManifest,
      VerificationLevel.MANIFEST_TRUSTED,
  ).manifest
  trainer_path = step / checkpoint.trainer.payload.logical_path
  trainer = dict(mx.load(trainer_path))
  trainer["next_key"] = counter_random_key(
      int(run_manifest.checkpoint["seed"]),
      1,
  )
  mx.save_safetensors(trainer_path, trainer)
  checkpoint = replace(
      checkpoint,
      trainer=_array_ref(
          trainer_path,
          checkpoint.trainer.payload.logical_path,
          trainer,
      ),
  )
  checkpoint = replace(checkpoint, identity=checkpoint.recompute_identity())
  (step / "checkpoint.json").write_bytes(canonical_json_bytes(checkpoint))
  _rebind_latest_checkpoint(run, checkpoint)

  verified = verify_artifact(run, full=True)
  assert verified.manifest.kind == "pretraining-run"
  assert verified.children[-1].manifest.step == 1
  ```

  Add a parametrized corruption immediately beside it for the two impossible
  bounds:

  ```python
  @pytest.mark.parametrize("microsteps", (0, 3))
  def test_full_pretraining_run_rejects_progress_outside_step_bounds(
      tmp_path: Path,
      pretraining_run_template: Path,
      microsteps: int,
  ) -> None:
      run = tmp_path / f"run-invalid-progress-{microsteps}"
      shutil.copytree(pretraining_run_template, run)
      step, checkpoint = _latest_checkpoint(run)
      state_path = step / checkpoint.scalar_state.logical_path
      state = json.loads(state_path.read_text(encoding="utf-8"))
      state["microsteps"] = microsteps
      state["rows"] = microsteps
      state_path.write_bytes(canonical_json_bytes(state))
      rebound = replace(
          checkpoint,
          scalar_state=_payload_ref(
              state_path,
              checkpoint.scalar_state.logical_path,
          ),
      )
      rebound = replace(rebound, identity=rebound.recompute_identity())
      (step / "checkpoint.json").write_bytes(canonical_json_bytes(rebound))
      _rebind_latest_checkpoint(run, rebound)

      with pytest.raises(SMLArtifactError, match="microsteps disagree"):
          verify_artifact(run, full=True)
  ```

- [ ] **Step 2: Add real pretraining and LoRA boundary coverage**

  Extend the existing integration configurations instead of inventing new
  fixtures. Add one pretraining run whose epoch ends after one microstep of a
  two-microstep accumulation window, then exercise FULL standalone verification,
  FULL inference resolution, and resume. Add one LoRA run with
  `microbatch_size=2` and `gradient_accumulation_steps=2` so one real microbatch
  completes a step:

  ```python
  resolved = resolve_latest_step(
      trained.run,
      writable=False,
      verification=VerificationLevel.FULL,
  )
  scalar = read_scalar_state(resolved)
  assert scalar.step == 1
  assert scalar.microsteps == 1
  verify_artifact(trained.run, full=True)
  InferenceSession.from_checkpoint(trained.run, full_verify=True)
  ```

  For LoRA, also load `state.json` through the retained resolved step and assert:

  ```python
  assert state["step"] == 1
  assert state["microsteps"] == 1
  assert 1 <= state["examples"] <= 2
  ```

- [ ] **Step 3: Run the focused tests and confirm RED**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/artifacts/test_recursive_verify.py \
    v2/tests/integration/test_pretraining_workflow.py \
    v2/tests/integration/test_swag_workflow.py \
    -k "partial_progress or step_bounds or partial_window or microbatch_progress" -q
  ```

  Expected: valid partial checkpoints fail with `checkpoint scalar microsteps
  disagree with step`; the corrupt lower/upper-bound cases already fail.

- [ ] **Step 4: Implement the exact internally derivable bound**

  Replace the equality in `_verify_progress` without weakening row/example
  binding:

  ```python
  maximum_microsteps = step * loader.gradient_accumulation_steps
  if not step <= microsteps <= maximum_microsteps:
      raise SMLArtifactError("checkpoint scalar microsteps disagree with step")
  if lora:
      examples = values.get("examples")
      if (
          isinstance(examples, bool)
          or not isinstance(examples, int)
          or not microsteps <= examples <= microsteps * loader.microbatch_size
      ):
          raise SMLArtifactError("checkpoint scalar examples disagree with loader")
  elif values.get("rows") != microsteps * loader.microbatch_size:
      raise SMLArtifactError("checkpoint scalar rows disagree with loader")
  ```

- [ ] **Step 5: Run the focused GREEN and scoped semantics regressions**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/artifacts/test_recursive_verify.py \
    v2/tests/integration/test_pretraining_workflow.py \
    v2/tests/integration/test_swag_workflow.py \
    v2/tests/integration/test_inference_workflow.py -q
  ```

  Expected: all selected tests pass, including step zero and the corrupted
  lower/upper progress bounds.

- [ ] **Step 6: Commit the bounded-progress change**

  ```bash
  git add \
    v2/src/sml/artifacts/semantics.py \
    v2/tests/unit/artifacts/test_recursive_verify.py \
    v2/tests/integration/test_pretraining_workflow.py \
    v2/tests/integration/test_swag_workflow.py
  git commit -m "fix(v2): accept bounded partial checkpoint progress"
  ```

### Task 2: Persist and Enforce Forward-Terminal RNG State

**Files:**
- Modify: `v2/src/sml/artifacts/manifest.py:1036-1112,1750-1790`
- Modify: `v2/src/sml/artifacts/semantics.py:125-220,300-375`
- Modify: `v2/src/sml/training/pretrain.py:70-90,600-690,750-830,920-980,1049-1130`
- Modify: `v2/src/sml/training/swag.py:135-165,680-755,990-1060,1120-1225,1375-1515`
- Test: `v2/tests/unit/artifacts/test_manifest.py:1060-1150`
- Test: `v2/tests/unit/artifacts/test_checkpoint.py:500-560`
- Test: `v2/tests/unit/artifacts/test_checkpoint_semantics.py:110-160,680-730`
- Test: `v2/tests/unit/artifacts/test_lora_base_snapshot.py:40-80`
- Test: `v2/tests/unit/artifacts/test_recursive_verify.py:70-110,1310-1385`
- Test: `v2/tests/unit/training/test_pretrain.py:340-400`
- Test: `v2/tests/unit/training/test_swag.py:1120-1200`
- Test: `v2/tests/integration/test_artifact_integrity.py:320-370,540-590`
- Test: `v2/tests/integration/test_pretraining_workflow.py:450-510,940-1035`
- Test: `v2/tests/integration/test_swag_workflow.py:500-680,1220-1310`

**Interfaces:**
- Consumes: Task 1's bounded `_verify_progress` result and the existing
  `counter_random_key(seed: int, microstep: int) -> mx.array`.
- Produces:
  `TRAINING_RNG_SCHEDULE = "counter-addressed-forward-terminal-v1"`, strict run
  checkpoint projections containing that value, and
  `expected_next_key(seed, microsteps, model, lora=None) -> mx.array` returning
  the terminal forward key.

- [ ] **Step 1: Add strict schedule-projection RED tests**

  In `test_manifest.py`, first add the exact schedule to the pretraining and LoRA
  entries returned by `manifest_fixtures()`. Select those existing fixtures,
  copy their checkpoint projections, and cover missing, unknown, and exact
  values:

  ```python
  @pytest.mark.parametrize("run_type", (PretrainingRunManifest, LoRARunManifest))
  @pytest.mark.parametrize("schedule", (None, "unknown-schedule"))
  def test_run_manifest_requires_exact_rng_schedule(run_type, schedule):
      manifest = next(
          value for value in manifest_fixtures() if isinstance(value, run_type)
      )
      checkpoint = dict(manifest.checkpoint)
      if schedule is None:
          checkpoint.pop("rng_schedule")
      else:
          checkpoint["rng_schedule"] = schedule
      with pytest.raises(ValueError, match="rng_schedule"):
          replace(manifest, checkpoint=checkpoint)

  @pytest.mark.parametrize("run_type", (PretrainingRunManifest, LoRARunManifest))
  def test_run_manifest_accepts_forward_terminal_rng_schedule(run_type):
      manifest = next(
          value for value in manifest_fixtures() if isinstance(value, run_type)
      )
      assert manifest.checkpoint["rng_schedule"] == (
          "counter-addressed-forward-terminal-v1"
      )
  ```

- [ ] **Step 2: Replace the old counter-key test with terminal-key cases**

  In `test_recursive_verify.py`, keep the huge `microsteps` value and count
  split calls rather than forbidding every split. Test model-only, LoRA-only,
  mixed, and disabled sites:

  ```python
  @pytest.mark.parametrize(
      ("hidden_dropout", "lora_dropout", "targets", "expected_splits"),
      (
          (0.2, 0.0, (), 2),
          (0.0, 0.3, ("q_proj", "v_proj"), 4),
          (0.2, 0.3, ("q_proj", "v_proj"), 6),
          (0.0, 0.0, ("q_proj", "v_proj"), 0),
      ),
  )
  def test_expected_training_key_uses_only_current_microstep_sites(
      monkeypatch,
      hidden_dropout,
      lora_dropout,
      targets,
      expected_splits,
  ):
      split_calls = 0
      real_split = semantics_module.mx.random.split

      def counted_split(key):
          nonlocal split_calls
          split_calls += 1
          return real_split(key)

      monkeypatch.setattr(semantics_module.mx.random, "split", counted_split)
      model = replace(
          ModelConfig(num_layers=2),
          hidden_dropout=hidden_dropout,
      )
      lora = None
      if targets:
          lora = LoRAConfig(dropout=lora_dropout, target_modules=targets)
      actual = semantics_module.expected_next_key(
          seed=17,
          microsteps=9_000_000_000_000,
          model=model,
          lora=lora,
      )
      mx.eval(actual)
      assert split_calls == expected_splits
  ```

  Add explicit `microsteps=0` assertions and compare disabled dropout to
  `counter_random_key(seed, 0)`.

- [ ] **Step 3: Add runtime and resume-before-use RED tests**

  Update the pretraining and SWAG unit tests so enabled dropout expects the
  terminal returned by the actual microstep, while disabled dropout preserves
  the initial key. In integration tests:

  - compare uninterrupted and resumed complete checkpoint groups for
    model-only, LoRA-only, mixed, and disabled dropout;
  - assert each published `trainer.next_key` equals
    the `expected_next_key` result for its saved `microsteps`;
  - re-sign `trainer.safetensors`, `checkpoint.json`, and `latest.json` after
    replacing `next_key` with `mx.random.key(999)`;
  - monkeypatch model construction, stream construction, `prune_to_latest`, and
    publication with rejecting spies, then assert resume raises
    `checkpoint trainer next RNG key is incorrect` before any spy runs.

  The core checkpoint assertion is:

  ```python
  expected = expected_next_key(
      seed=config.seed,
      microsteps=state["microsteps"],
      model=config.model,
      lora=getattr(config, "lora", None),
  )
  mx.eval(expected, trainer_arrays["next_key"])
  assert bool(mx.array_equal(trainer_arrays["next_key"], expected))
  ```

- [ ] **Step 4: Run the schedule/key tests and confirm RED**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/artifacts/test_manifest.py \
    v2/tests/unit/artifacts/test_recursive_verify.py \
    v2/tests/unit/training/test_pretrain.py \
    v2/tests/unit/training/test_swag.py \
    v2/tests/integration/test_pretraining_workflow.py \
    v2/tests/integration/test_swag_workflow.py \
    -k "rng_schedule or expected_training_key or dropout or wrong_key or uninterrupted" -q
  ```

  Expected: missing schedule validation, terminal-key comparisons, and
  resume-before-use assertions fail against the counter-overwrite runtime.

- [ ] **Step 5: Make the persisted schedule exact at every reader boundary**

  Add the schema constant in `manifest.py` and require it from both run
  `__post_init__` methods:

  ```python
  TRAINING_RNG_SCHEDULE = "counter-addressed-forward-terminal-v1"

  def _require_training_rng_schedule(checkpoint: Mapping[str, object]) -> None:
      if checkpoint.get("rng_schedule") != TRAINING_RNG_SCHEDULE:
          raise ValueError(
              "checkpoint rng_schedule must be "
              f"{TRAINING_RNG_SCHEDULE!r}"
          )
  ```

  Call it after freezing each run's `checkpoint` mapping and export the constant
  in `manifest.__all__`. Add `rng_schedule` to
  `semantics.checkpoint_configuration`'s exact field set and verify the exact
  constant before returning the projection. Update every direct run-manifest
  constructor in `test_manifest.py`, `test_checkpoint.py`,
  `test_checkpoint_semantics.py`, `test_lora_base_snapshot.py`, and
  `test_artifact_integrity.py` to carry the exact checkpoint projection field;
  do not add fallback defaults in production constructors.

- [ ] **Step 6: Write the exact terminal-key semantic derivation**

  Replace the existing counter-only implementation in `semantics.py`:

  ```python
  def expected_next_key(
      *,
      seed: int,
      microsteps: int,
      model: ModelConfig,
      lora: LoRAConfig | None = None,
  ) -> mx.array:
      model_sites = model.num_layers if model.hidden_dropout > 0.0 else 0
      lora_sites = (
          model.num_layers * len(lora.target_modules)
          if lora is not None and lora.dropout > 0.0
          else 0
      )
      active_sites = model_sites + lora_sites
      if microsteps == 0 or active_sites == 0:
          return counter_random_key(seed, 0)
      key = counter_random_key(seed, microsteps - 1)
      for _site in range(active_sites):
          key, _unused = mx.random.split(key)
      return key
  ```

  Keep work O(active sites), not O(microsteps), and retain the existing uint64
  counter validation in `counter_random_key`.

- [ ] **Step 7: Persist the schedule and install keys before active forwards**

  Add the constant to both `_SAVED_CHECKPOINT_KEYS`, write it in both
  `_run_manifest` checkpoint mappings, and explicitly validate/pop it in both
  `_config_from_run` functions before constructing runtime configuration.

  In each training loop, replace only the key leaf immediately before the
  microstep when at least one site is active:

  ```python
  microstep_index = scalar.microsteps + window_microsteps
  if config.model.hidden_dropout > 0.0:
      trainer_tree = trainer.to_tree()
      trainer = TrainerState.from_compiled_tree(
          (
              trainer_tree[0],
              trainer_tree[1],
              counter_random_key(config.seed, microstep_index),
              trainer_tree[3],
          )
      )
  microstep = kernels.microstep(parameters, trainer, envelope.rows)
  trainer = microstep.trainer
  ```

  Use the five-leaf `SwagTrainerState` equivalent when either model dropout or
  LoRA dropout is active. Delete both post-forward
  `counter_random_key` overwrites with a `+ 1` index. Disabled-dropout paths
  must pass the current trainer state through unchanged.

- [ ] **Step 8: Validate restored terminal keys while the reader is retained**

  Refactor pretraining `_restore_checkpoint` to consume a live
  `CheckpointReader` rather than reopening a `ResolvedStep`. In `resume`, open
  the selected FULL reader, open its retained tokenizer child, call
  `validate_full_run_semantics(reader, tokenizer.manifest)`, and only then build
  restored state. LoRA resume already has the required FULL reader and tokenizer
  child; add the same validator before `_restore_adapter_checkpoint(reader)`.

  Preserve this order:

  ```python
  with open_checkpoint_reader(
      run,
      step=resolved.step,
      expected_checkpoint_identity=resolved.checkpoint.identity,
      verification=VerificationLevel.FULL,
  ) as reader:
      with reader.open_run_child(
          "tokenizer",
          (TokenizerManifest,),
      ) as tokenizer:
          validate_full_run_semantics(reader, tokenizer.manifest)
      restored = _restore_checkpoint(reader)
      resolved = reader.resolved
  ```

  Wrong-key failure must precede cursor validation, pruning, model/kernel/stream
  construction, or publication.

- [ ] **Step 9: Run the focused GREEN and complete training regressions**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/artifacts/test_manifest.py \
    v2/tests/unit/artifacts/test_checkpoint.py \
    v2/tests/unit/artifacts/test_checkpoint_semantics.py \
    v2/tests/unit/artifacts/test_lora_base_snapshot.py \
    v2/tests/unit/artifacts/test_recursive_verify.py \
    v2/tests/unit/training/test_pretrain.py \
    v2/tests/unit/training/test_swag.py \
    v2/tests/integration/test_pretraining_workflow.py \
    v2/tests/integration/test_swag_workflow.py \
    v2/tests/integration/test_inference_workflow.py \
    v2/tests/integration/test_artifact_integrity.py -q
  ```

  Expected: active, mixed, and disabled dropout paths pass in eager/compiled and
  uninterrupted/resumed forms; a re-signed wrong key fails before use.

- [ ] **Step 10: Commit the RNG contract change**

  ```bash
  git add \
    v2/src/sml/artifacts/manifest.py \
    v2/src/sml/artifacts/semantics.py \
    v2/src/sml/training/pretrain.py \
    v2/src/sml/training/swag.py \
    v2/tests/unit/artifacts/test_manifest.py \
    v2/tests/unit/artifacts/test_checkpoint.py \
    v2/tests/unit/artifacts/test_checkpoint_semantics.py \
    v2/tests/unit/artifacts/test_lora_base_snapshot.py \
    v2/tests/unit/artifacts/test_recursive_verify.py \
    v2/tests/unit/training/test_pretrain.py \
    v2/tests/unit/training/test_swag.py \
    v2/tests/integration/test_artifact_integrity.py \
    v2/tests/integration/test_pretraining_workflow.py \
    v2/tests/integration/test_swag_workflow.py
  git commit -m "fix(v2): preserve counter-addressed forward terminal keys"
  ```

### Task 3: Unify Stable Retained-Root Manifest Dispatch

**Files:**
- Create: `v2/src/sml/artifacts/dispatch.py`
- Modify: `v2/src/sml/artifacts/manifest.py:360-565,650-710,1620-1710`
- Modify: `v2/src/sml/artifacts/verify.py:1-275,490-505`
- Modify: `v2/src/sml/data/pretraining.py:1-45,320-390,530-575`
- Modify: `v2/src/sml/inference.py:620-690`
- Test: `v2/tests/unit/artifacts/test_manifest.py:640-710,890-920`
- Test: `v2/tests/unit/artifacts/test_recursive_verify.py:200-310`
- Test: `v2/tests/integration/test_pretraining_data_workflow.py:1450-1580`
- Test: `v2/tests/integration/test_inference_workflow.py:430-510`

**Interfaces:**
- Consumes: `ArtifactRoot`, `_open_stable_payload`, `_parse_manifest`, and
  `OpenedArtifact` from `manifest.py`.
- Produces:
  `open_dispatched_artifact(path: Path, root: ArtifactRoot,
  verification: VerificationLevel) -> OpenedArtifact[ArtifactManifest]` and one
  shared stable-manifest reader used by ordinary, recursive, and nested readers.

- [ ] **Step 1: Add candidate ambiguity and race RED tests**

  Keep recursive ambiguity coverage and add inference coverage for both levels:

  ```python
  @pytest.mark.parametrize("full_verify", (False, True))
  def test_inference_rejects_dual_manifest_candidates(
      tiny_pretraining_run: Path,
      tmp_path: Path,
      full_verify: bool,
  ) -> None:
      ambiguous = tmp_path / f"ambiguous-{full_verify}"
      shutil.copytree(tiny_pretraining_run, ambiguous)
      (ambiguous / "manifest.json").write_bytes(
          (ambiguous / "run.json").read_bytes()
      )
      with pytest.raises(SMLArtifactError, match="exactly one"):
          resolve_model_artifact(ambiguous, full_verify=full_verify)
  ```

  Add recursive parse-time mutations for adding the second candidate and
  replacing the selected named inode. Inject from the real `_parse_manifest`
  callback, not before dispatch, and assert the retained root descriptor closes
  after failure.

- [ ] **Step 2: Add nested-tokenizer lifecycle and dual-failure RED tests**

  In `test_pretraining_data_workflow.py`, monkeypatch the shared manifest parser
  so the nested tokenizer manifest's mtime changes during schema parsing. Assert
  `load_pretraining_bundle`/preflight raises `changed during use`. Add a second
  test that injects an invalid tokenizer schema and makes the stable postcheck
  fail; preserve the semantic schema exception and chain the postcheck error:

  ```python
  with pytest.raises(SMLArtifactError, match="invalid.*tokenizer") as raised:
      preflight_pretraining_bundle(bundle, batch_size=1)
  assert isinstance(raised.value.__cause__, SMLArtifactError)
  assert "changed during use" in str(raised.value.__cause__)
  ```

- [ ] **Step 3: Run the dispatch/lifecycle tests and confirm RED**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/artifacts/test_manifest.py \
    v2/tests/unit/artifacts/test_recursive_verify.py \
    v2/tests/integration/test_pretraining_data_workflow.py \
    v2/tests/integration/test_inference_workflow.py \
    -k "dispatch or candidate or manifest_mutation or tokenizer_lifecycle or dual_failure" -q
  ```

  Expected: inference accepts the ambiguous run, recursive dispatch misses
  parse-time candidate replacement/addition, and the prepared nested parser
  misses mutation after its raw descriptor closes.

- [ ] **Step 4: Extend the common stable manifest primitive**

  Keep JSON decode, duplicate/non-finite rejection, exact type dispatch,
  identity, canonical bytes, and close/postcheck in `_read_manifest_from_root`.
  Extend it with private acquisition and pre-close callbacks that receive the
  opened stat while `_StablePayload` is live:

  ```python
  def _read_manifest_from_root[M: _Manifest](
      root: ArtifactRoot,
      path: Path,
      manifest_types: tuple[type[M], ...],
      *,
      validate_opened: Callable[[os.stat_result], None] | None = None,
      validate_before_close: Callable[[os.stat_result], None] | None = None,
  ) -> M:
      manifest_types = _validated_manifest_types(manifest_types)
      filename = next(
          iter({kind.MANIFEST_FILENAME for kind in manifest_types})
      )
      with _open_stable_payload(root, filename) as payload:
          if validate_opened is not None:
              validate_opened(payload.opened_stat)
          encoded = payload.read()
          manifest = _parse_and_validate_manifest_bytes(
              encoded,
              path / filename,
              manifest_types,
          )
          if validate_before_close is not None:
              validate_before_close(payload.opened_stat)
          return manifest
  ```

  Extract `_parse_and_validate_manifest_bytes` only as the pure semantic portion;
  its body is the existing UTF-8/`json.loads` block followed by
  `_manifest_type_for_raw`, `_parse_manifest`, `recompute_identity`, and
  `canonical_json_bytes` equality in that order. No caller may combine it with
  an untracked raw `open_payload` lifecycle.

- [ ] **Step 5: Add descriptor-relative direct-candidate stat support**

  Add a private `ArtifactRoot._stat_direct_payload` that accepts only a single
  portable component, uses descriptor-relative `os.stat` with
  `dir_fd=self._fd` and `follow_symlinks=False`, returns `None` only for
  `FileNotFoundError`, and
  rejects non-regular or non-single-link entries. Compare all stable fields:

  ```python
  def _same_stable_entry(
      expected: os.stat_result,
      actual: os.stat_result,
  ) -> bool:
      return _stable_stat_fields(expected) == _stable_stat_fields(actual)
  ```

  Do not open either candidate merely to decide which name exists.

- [ ] **Step 6: Create the neutral exact-candidate dispatcher**

  In new `artifacts/dispatch.py`, move the `ArtifactManifest` union and legal
  filename/type table out of `verify.py`. Implement the exact public-internal
  interface:

  ```python
  type _Candidate = tuple[
      str,
      os.stat_result,
      tuple[type[ArtifactManifest], ...],
  ]


  def _candidate_stats(root: ArtifactRoot) -> list[_Candidate]:
      result: list[_Candidate] = []
      for filename, manifest_types in _MANIFEST_CANDIDATES:
          candidate = root._stat_direct_payload(filename)
          if candidate is not None:
              result.append((filename, candidate, manifest_types))
      return result


  def open_dispatched_artifact(
      path: Path,
      root: ArtifactRoot,
      verification: VerificationLevel,
  ) -> OpenedArtifact[ArtifactManifest]:
      try:
          initial = _candidate_stats(root)
          if len(initial) != 1:
              raise SMLArtifactError(
                  "artifact root must contain exactly one of "
                  "run.json or manifest.json"
              )
          filename, opened_expected, manifest_types = initial[0]

          def validate_opened(opened: os.stat_result) -> None:
              if not _same_stable_entry(opened_expected, opened):
                  raise SMLArtifactError("artifact manifest candidate changed")

          def validate_before_close(opened: os.stat_result) -> None:
              final = _candidate_stats(root)
              if len(final) != 1 or final[0][0] != filename:
                  raise SMLArtifactError("artifact manifest candidates changed")
              if not _same_stable_entry(opened, final[0][1]):
                  raise SMLArtifactError("artifact manifest candidate changed")

          manifest = _read_manifest_from_root(
              root,
              path,
              manifest_types,
              validate_opened=validate_opened,
              validate_before_close=validate_before_close,
          )
          return OpenedArtifact(
              path=path,
              root=root,
              manifest=manifest,
              verification=verification,
          )
      except BaseException as error:
          try:
              root.close()
          except BaseException as cleanup_error:
              raise error from cleanup_error
          raise
  ```

  `_candidate_stats` probes in deterministic `run.json`, `manifest.json` order.
  Success transfers the supplied root into `OpenedArtifact`; every failure
  closes it and preserves the semantic error over cleanup failure.

- [ ] **Step 7: Route recursive verification and inference through one owner**

  Delete `_read_candidate_bytes` and the duplicate JSON/parser implementation
  from `verify.py`. `_open_artifact_once`, child artifact opens, and reader-child
  opens call `open_dispatched_artifact`; child helpers then enforce their
  `allowed_types` and close the unexpected owner with semantic-primary
  precedence.

  In inference, remove the `run.json` existence probe and duplicate-root branch.
  Also remove the `(path / "checkpoint.json").exists()` pathname probe from
  `_require_run_path`; canonical `step-*` names still reject there, while a
  renamed direct-step directory has zero legal dispatch candidates and rejects
  through the retained root. Use this owner-only branch:

  ```python
  root = ArtifactRoot.open(path, writable=False)
  with open_dispatched_artifact(path, root, verification) as artifact:
      if isinstance(artifact.manifest, PretrainingRunManifest):
          return _resolve_pretraining_run(
              path,
              full_verify=full_verify,
              expected_run=artifact.manifest,
              run_descriptor=artifact.root.fileno(),
          )
      if isinstance(artifact.manifest, LoRARunManifest):
          return _resolve_lora_run(
              path,
              full_verify=full_verify,
              expected_run=artifact.manifest,
              run_descriptor=artifact.root.fileno(),
          )
      if isinstance(artifact.manifest, ExportManifest):
          return _resolve_opened_export(artifact, full_verify=full_verify)
      raise SMLArtifactError("artifact kind cannot be used for model resolution")
  ```

  Do not reopen `path` after `ArtifactRoot.open`.

- [ ] **Step 8: Replace the prepared-data duplicate tokenizer parser**

  Delete `_parsed_payload_ref` and `_parse_canonical_tokenizer_manifest`. Remove
  the redundant raw re-read of the already-stably-parsed outer manifest. Open a
  retained tokenizer child root, call `_read_manifest_from_root` with
  `(TokenizerManifest,)`, and close the child root with semantic-primary
  precedence before validating outer/inner binding.

- [ ] **Step 9: Run the focused GREEN and artifact-consumer regressions**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/artifacts/test_manifest.py \
    v2/tests/unit/artifacts/test_recursive_verify.py \
    v2/tests/integration/test_pretraining_data_workflow.py \
    v2/tests/integration/test_inference_workflow.py \
    v2/tests/unit/test_inference.py -q
  ```

  Expected: zero/two candidates reject at both levels; parse-time mutations fail;
  semantic errors remain primary; all retained descriptors close exactly once.

- [ ] **Step 10: Commit the stable-dispatch change**

  ```bash
  git add \
    v2/src/sml/artifacts/dispatch.py \
    v2/src/sml/artifacts/manifest.py \
    v2/src/sml/artifacts/verify.py \
    v2/src/sml/data/pretraining.py \
    v2/src/sml/inference.py \
    v2/tests/unit/artifacts/test_manifest.py \
    v2/tests/unit/artifacts/test_recursive_verify.py \
    v2/tests/integration/test_pretraining_data_workflow.py \
    v2/tests/integration/test_inference_workflow.py
  git commit -m "fix(v2): unify retained artifact kind dispatch"
  ```

### Task 4: Cap Safetensors Headers Before Allocation

**Files:**
- Modify: `v2/src/sml/artifacts/arrays.py:15-75`
- Test: `v2/tests/unit/artifacts/test_arrays.py:90-175`

**Interfaces:**
- Consumes: existing
  `read_safetensors_layout(stream: BinaryIO, reference: ArrayPayloadRef)`.
- Produces: the same interface with a hard `100_000_000`-byte header ceiling
  enforced immediately after the fixed eight-byte prefix.

- [ ] **Step 1: Add a bounded-read/allocation RED test**

  Add `import io` to `test_arrays.py`. Add a tracking stream that contains only
  the eight-byte prefix but declares a payload size large enough that the
  current payload-size check does not reject first:

  ```python
  class _TrackingHeaderStream(io.BytesIO):
      def __init__(self, payload: bytes) -> None:
          super().__init__(payload)
          self.read_requests: list[int] = []

      def read(self, size: int = -1) -> bytes:
          self.read_requests.append(size)
          if size > 1024:
              raise AssertionError("oversized header read was attempted")
          return super().read(size)


  def test_safetensors_rejects_oversized_header_before_read_or_allocation():
      claimed = 100_000_001
      stream = _TrackingHeaderStream(claimed.to_bytes(8, "little"))
      reference = ArrayPayloadRef(
          PayloadRef(
              "model.safetensors",
              "sha256:" + "0" * 64,
              claimed + 8,
          ),
          (),
      )
      tracemalloc.start()
      try:
          with pytest.raises(SMLArtifactError, match="header.*limit"):
              arrays_module.read_safetensors_layout(stream, reference)
          _current, peak = tracemalloc.get_traced_memory()
      finally:
          tracemalloc.stop()
      assert stream.read_requests == [8]
      assert peak < 1024 * 1024
  ```

- [ ] **Step 2: Run the focused test and confirm RED**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/artifacts/test_arrays.py::test_safetensors_rejects_oversized_header_before_read_or_allocation \
    -q
  ```

  Expected: `AssertionError: oversized header read was attempted`.

- [ ] **Step 3: Enforce the upstream-compatible ceiling at the trust boundary**

  Add a private constant and check it before payload-size arithmetic or the
  header-sized read:

  ```python
  _MAX_SAFETENSORS_HEADER_BYTES = 100_000_000

  header_length = int.from_bytes(encoded_length, byteorder="little")
  if header_length > _MAX_SAFETENSORS_HEADER_BYTES:
      raise SMLArtifactError(
          f"safetensors header exceeds parser limit: {logical_path}"
      )
  if header_length > reference.payload.byte_size - 8:
      raise SMLArtifactError(
          f"safetensors header exceeds payload bytes: {logical_path}"
      )
  ```

  Do not change JSON whitespace acceptance or trailing padding behavior.

- [ ] **Step 4: Run the focused GREEN and complete array tests**

  Run outside the sandbox:

  ```bash
  uv run pytest v2/tests/unit/artifacts/test_arrays.py -q
  ```

  Expected: all array tests pass, the largest requested read is eight bytes in
  the oversized-header regression, and existing sparse metadata remains bounded.

- [ ] **Step 5: Commit the cap**

  ```bash
  git add \
    v2/src/sml/artifacts/arrays.py \
    v2/tests/unit/artifacts/test_arrays.py
  git commit -m "fix(v2): cap safetensors header reads"
  ```

### Task 5: Make Data Ownership Transfer and Mapped Reductions Bounded

**Files:**
- Modify: `v2/src/sml/data/pretraining.py:530-760`
- Modify: `v2/src/sml/data/swag.py:1190-1340,2148-2210`
- Test: `v2/tests/unit/data/test_swag.py:1040-1160,2130-2375`
- Test: `v2/tests/integration/test_pretraining_data_workflow.py:1450-1580`

**Interfaces:**
- Consumes: existing `SwagDataBundle._bucket_leases`, the `SwagBatchStream`
  constructor with its `cursor` keyword, `_validate_bucket_arrays`, and
  `row_content_identity`.
- Produces: owning stream construction that is failure-atomic with public leases
  and validation reductions whose first dimension never exceeds 1,024 rows.

- [ ] **Step 1: Add active-public-lease transfer RED coverage**

  In `test_swag.py`, borrow a public lease before owning stream construction and
  prove rejection leaves both objects usable and retryable:

  ```python
  lease = bundle.borrow_buckets()
  first = lease.buckets[0].input_ids[0, 0, 0]
  with pytest.raises(SMLDataError, match="active bucket leases"):
      SwagBatchStream(
          bundle,
          _one_example_loader(),
          cursor=SwagCursor.initial(),
      )
  assert lease.buckets[0].input_ids[0, 0, 0] == first
  assert bundle._closed is False
  lease.close()
  stream = SwagBatchStream(
      bundle,
      _one_example_loader(),
      cursor=SwagCursor.initial(),
  )
  stream.close()
  assert bundle._closed is True
  ```

- [ ] **Step 2: Add reduction-operand instrumentation RED tests**

  Build sparse prepared shards and SWAG buckets with more than 2,048 rows. Wrap
  module-local NumPy reductions (`np.min`, `np.max`, `np.any`, `np.all`) to record
  operand shapes while forwarding to the real function. Exercise FULL prepared
  preflight/recursive verification and FULL SWAG loading, then assert:

  ```python
  row_operands = [shape for shape in recorded if shape and shape[0] > 1]
  assert row_operands
  assert max(shape[0] for shape in row_operands) <= 1_024
  ```

  Include the SWAG position-zero mask and labels in the recorded calls so the
  current whole-bucket reductions fail the assertion.

- [ ] **Step 3: Run the ownership/chunk tests and confirm RED**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/data/test_swag.py \
    v2/tests/integration/test_pretraining_data_workflow.py \
    -k "active_public_lease or reduction_operand or multi_chunk" -q
  ```

  Expected: owning-stream rejection changes bundle/stream state incorrectly, and
  at least one prepared or SWAG reduction records a first dimension above 1,024.

- [ ] **Step 4: Reject unsafe transfer before stream state is installed**

  In `SwagBatchStream._initialize`, validate the owning transfer before assigning
  `_bundle`, `_owns_bundle`, `_closed`, or starting the thread:

  ```python
  if bundle._closed:
      raise SMLDataError("SWAG data bundle is closed")
  if owns_bundle and bundle._bucket_leases:
      raise SMLDataError(
          "cannot transfer SWAG bundle ownership with active bucket leases"
      )
  self._bundle = bundle
  self._owns_bundle = owns_bundle
  ```

  Leave `_borrowing_bundle` behavior with `owns_bundle=False` unchanged.

- [ ] **Step 5: Share one fixed prepared-row scan**

  Add `_PREPARED_ROW_SCAN_SIZE = 1_024` and a generator that performs token
  bounds on each chunk and yields its rows to `row_content_identity`:

  ```python
  def _validated_prepared_rows(
      shards: Sequence[np.ndarray],
      *,
      vocab_size: int,
  ) -> Iterator[np.ndarray]:
      for shard in shards:
          for start in range(0, shard.shape[0], _PREPARED_ROW_SCAN_SIZE):
              chunk = shard[start : start + _PREPARED_ROW_SCAN_SIZE]
              if chunk.shape[0] and (
                  int(np.min(chunk)) < 0
                  or int(np.max(chunk)) >= vocab_size
              ):
                  raise SMLArtifactError(
                      "prepared bundle token IDs are outside the tokenizer vocabulary"
                  )
              yield from chunk
  ```

  Use this helper in runtime preflight for bounds and in FULL semantic
  verification as the exact input to `row_content_identity`. Remove both
  whole-shard `min`/`max` comprehensions.

- [ ] **Step 6: Fold all remaining SWAG checks into the 1,024-row scan**

  Add `labels` to `_validate_bucket_arrays` and
  `_validate_bucket_array_chunk`. Slice it with the other arrays and move both
  whole-bucket checks into the chunk function:

  ```python
  if bool(np.any(score_mask[:, :, 0])):
      raise SMLArtifactError("SWAG score mask must be false at position zero")
  if labels.shape[0] and (
      int(np.min(labels)) < 0 or int(np.max(labels)) >= 4
  ):
      raise SMLArtifactError("SWAG labels must be in 0..3")
  ```

  Keep token, mask, boundary, BOS/EOS, continuation, and identity checks in the
  same chunk path. Remove the position-zero and label reductions from
  `_open_buckets`.

- [ ] **Step 7: Run the focused GREEN and complete data regressions**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/data/test_pretraining.py \
    v2/tests/unit/data/test_swag.py \
    v2/tests/integration/test_pretraining_data_workflow.py \
    v2/tests/integration/test_swag_data_workflow.py -q
  ```

  Expected: failure-atomic transfer passes, every recorded multi-row reduction is
  capped at 1,024, and invalid token/label/mask fixtures still reject.

- [ ] **Step 8: Commit the data lifetime/bounds change**

  ```bash
  git add \
    v2/src/sml/data/pretraining.py \
    v2/src/sml/data/swag.py \
    v2/tests/unit/data/test_swag.py \
    v2/tests/integration/test_pretraining_data_workflow.py
  git commit -m "fix(v2): bound remaining mapped data lifetimes"
  ```

### Task 6: Remove Duplicate Selected-Checkpoint Payload Opens

**Files:**
- Modify: `v2/src/sml/artifacts/checkpoint.py:860-1075,2485-2730`
- Test: `v2/tests/unit/artifacts/test_checkpoint.py:1080-1220`

**Interfaces:**
- Consumes: `_scan_closed_world`, `_verify_closed_world`,
  `_verify_checkpoint_semantics`, and
  `_open_verified_step_from_descriptor`.
- Produces: stat-only namespace enumeration plus exactly one descriptor-bound
  semantic open of `state.json` and every checkpoint array group whenever the
  selected proof/use pass runs.

- [ ] **Step 1: Add actual descriptor-open count RED tests**

  Add `Counter` and `ArtifactRoot` to the test imports. Instrument the
  descriptor-opening boundary, not only semantic helper calls. Wrap
  `ArtifactRoot._open_payload_with_stat` and `_opened_entry`, recording direct
  checkpoint filenames from both paths. Parametrize FULL with no requested
  groups and MANIFEST_TRUSTED with one requested group:

  ```python
  @pytest.mark.parametrize(
      ("verification", "groups"),
      (
          (VerificationLevel.FULL, frozenset()),
          (
              VerificationLevel.MANIFEST_TRUSTED,
              frozenset({"model.safetensors"}),
          ),
      ),
  )
  def test_selected_checkpoint_payloads_are_opened_once_for_proof_and_use(
      valid_run: Path,
      monkeypatch: pytest.MonkeyPatch,
      verification: VerificationLevel,
      groups: frozenset[str],
  ) -> None:
      opens: Counter[str] = Counter()
      payload_names = {
          "state.json",
          "model.safetensors",
          "master.safetensors",
          "optimizer.safetensors",
          "trainer.safetensors",
      }
      real_payload_open = ArtifactRoot._open_payload_with_stat
      real_opened_entry = checkpoint._opened_entry

      def counted_payload_open(owner, logical_path):
          if logical_path in payload_names:
              opens[logical_path] += 1
          return real_payload_open(owner, logical_path)

      def counted_opened_entry(
          fs,
          name,
          *,
          parent_descriptor,
          flags,
      ):
          if name in payload_names:
              opens[name] += 1
          return real_opened_entry(
              fs,
              name,
              parent_descriptor=parent_descriptor,
              flags=flags,
          )

      monkeypatch.setattr(
          ArtifactRoot,
          "_open_payload_with_stat",
          counted_payload_open,
      )
      monkeypatch.setattr(checkpoint, "_opened_entry", counted_opened_entry)
      with checkpoint.open_latest_checkpoint_reader(
          valid_run,
          verification=verification,
          load_array_groups=groups,
      ) as reader:
          reader.read_contents()
      assert opens["state.json"] == 1
      for name in (
          "model.safetensors",
          "master.safetensors",
          "optimizer.safetensors",
          "trainer.safetensors",
      ):
          assert opens[name] == 1
  ```

  The counter must include opens issued by `_scan_closed_world` and
  `ArtifactRoot`, so the current structural pass makes this RED.

- [ ] **Step 2: Add stat-only closed-world rejection tests**

  Preserve strict rejection without content opens by parametrizing a symlink,
  hard link, wrong byte size, normalized/case-fold collision, unexpected file,
  and missing file. Assert each fails before `reader.read_contents()` returns.
  The size case re-signs neither manifest nor payload:

  ```python
  step = next((valid_run / "checkpoints").glob("step-*"))
  with (step / "model.safetensors").open("ab") as payload:
      payload.write(b"x")
  with (
      pytest.raises(SMLArtifactError, match="byte size|closed-world"),
      checkpoint.open_latest_checkpoint_reader(
          valid_run,
          verification=VerificationLevel.FULL,
      ),
  ):
      pass
  ```

- [ ] **Step 3: Run the open-count tests and confirm RED**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/artifacts/test_checkpoint.py \
    -k "opened_once or stat_only_closed_world" -q
  ```

  Expected: selected payload counts exceed one because namespace scanning and
  `_verify_closed_world` with `full=False` opens them before semantic proof/use.

- [ ] **Step 4: Make namespace enumeration stat-only**

  Change `_scan_closed_world` to return a mapping from logical file path to its
  no-follow `stat_result` plus the directory set. For regular files, validate
  `st_nlink == 1`, record the stat, and do not call `_opened_entry`:

  ```python
  if stat.S_ISREG(entry_stat.st_mode):
      if entry_stat.st_nlink != 1:
          raise SMLArtifactError(
              f"closed-world payload is hard-linked: {'/'.join(logical_path)}"
          )
      files[logical_path] = entry_stat
  ```

  Keep descriptor opens for directories because recursive traversal must retain
  their inode. Preserve safe names, case-fold collision detection, and special
  file/symlink rejection.

- [ ] **Step 5: Validate declared sizes without opening payload content**

  In `_verify_closed_world`, derive expected reference sizes and compare them to
  the stat-only map. For a publisher-owned manifest, compare against
  `len(canonical_json_bytes(manifest))`. Use the existing stable nested-tokenizer
  reader for nested manifests; never replace semantic parsing with stat alone.

  ```python
  references_by_path = {
      parse_logical_path(reference.logical_path): reference
      for reference in references
  }
  for logical_path, reference in references_by_path.items():
      if actual_files[logical_path].st_size != reference.byte_size:
          raise SMLArtifactError(
              "closed-world payload byte size mismatch: "
              f"{'/'.join(logical_path)}"
          )
  ```

- [ ] **Step 6: Skip structural content proof when semantics opens every payload**

  Add an exact `verify_contents: bool = True` keyword to
  `_verify_closed_world`.
  When `verify_contents` is true, retain existing payload identity/size and
  publisher-manifest canonical checks. In
  `_open_verified_step_from_descriptor`, compute `materialize` before the
  closed-world call and pass `verify_contents=not materialize`:

  ```python
  effective_load_groups = load_array_groups
  if effective_load_groups is None and materialize_byte_groups:
      effective_load_groups = materialize_byte_groups
  materialize = (
      verification is VerificationLevel.FULL
      or effective_load_groups is not None
  )
  _verify_closed_world(
      fs,
      descriptor,
      manifest,
      manifest_present=True,
      full=False,
      verify_contents=not materialize,
  )
  ```

  `_verify_checkpoint_semantics` remains responsible for opening `state.json`
  and every array reference once, holding all verified payload descriptors
  through layout validation, selected materialization, boundary checks, and
  close.

- [ ] **Step 7: Run the focused GREEN and checkpoint suite**

  Run outside the sandbox:

  ```bash
  uv run pytest \
    v2/tests/unit/artifacts/test_checkpoint.py \
    v2/tests/unit/artifacts/test_checkpoint_semantics.py \
    v2/tests/integration/test_artifact_integrity.py \
    v2/tests/integration/test_pretraining_workflow.py \
    v2/tests/integration/test_swag_workflow.py -q
  ```

  Expected: each selected state/array payload opens exactly once in FULL and
  requested-group MANIFEST_TRUSTED paths; every closed-world corruption still
  rejects.

- [ ] **Step 8: Commit the single-proof/open change**

  ```bash
  git add \
    v2/src/sml/artifacts/checkpoint.py \
    v2/tests/unit/artifacts/test_checkpoint.py
  git commit -m "perf(v2): remove duplicate checkpoint payload opens"
  ```

## Per-Task Review Gate

After each task commit, before starting the next task:

1. Dispatch a fresh specification reviewer with the design, this plan's exact
   task section, and the task commit range. Require explicit requirement-by-
   requirement evidence.
2. If specification review passes, dispatch a fresh code-quality reviewer for
   correctness, lifetime/error precedence, performance bounds, and test quality.
3. Save reports and fix-round ledgers under
   `.superpowers/sdd/2026-08-30-v2-artifact-residual-corrections/`.
4. Apply valid findings with focused RED/GREEN tests and a dedicated fix commit,
   then re-review. Stop and escalate only after five ordinary fix rounds.
5. Confirm the handoff remains unstaged before moving to the next task.

## Whole-Component Acceptance Gate

After all six task reviews are clean:

- [ ] Run focused forbidden-pattern checks:

  ```bash
  rg -n "counter_random_key\([^\n]*microsteps[^\n]*\+ 1|root\.open_payload\(.*manifest|path / \"run.json\"|diagnostic_data_locator.*open" v2/src/sml
  ```

  Inspect every match; expected result is no forbidden post-forward counter
  overwrite, raw manifest semantic parser, path-based kind selector, or
  diagnostic-locator open.

- [ ] Run formatting and lint gates:

  ```bash
  uv run ruff check v2
  uv run ruff format --check v2
  ```

- [ ] Run the complete V2 suite outside the sandbox:

  ```bash
  uv run pytest v2/tests
  ```

- [ ] Run repository hygiene checks:

  ```bash
  git diff --check
  git diff --name-only HEAD -- pyproject.toml uv.lock
  git status --short
  ```

  Expected: no whitespace errors, no top-level dependency changes, and the
  tracked handoff is the only inherited modification.

- [ ] Dispatch one fresh whole-component reviewer over the full correction-wave
  commit range. Require a saved report with Critical/Important/Minor findings,
  exact file/line evidence, and an explicit clean/not-clean verdict.

- [ ] If the whole review finds issues, use the user-authorized single combined
  final correction wave followed by one scoped re-review. Do not declare the
  artifact phase clean unless that re-review has zero findings.

- [ ] Update the tracked handoff with commit hashes, exact gate outputs, review
  report paths, residual findings or zero-finding verdict, and the next safe
  action. Keep it unstaged unless the user separately requests a handoff commit.
