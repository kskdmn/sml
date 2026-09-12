# Environment

- Python 3.12.13
- Use `uv run` to run Python scripts.
- Always run `uv run pytest` outside the sandbox so MLX/Metal can access the Apple GPU; request escalation/approval when the environment requires it.

# Rules

- Directories in the project root indicate model versions. For example, `v1` is the first version. Unless the user specifies a version, update the highest-numbered `vN` directory.
- Do not edit top-level project files, such as `pyproject.toml` or `uv.lock`, unless the user explicitly asks for that change. If the task truly requires a top-level edit, ask for approval first.
- Before finishing, run `ruff` and `pytest` on the version you updated. For example, after `v2` changes run `uv run ruff check v2`, `uv run ruff format --check v2`, and `uv run pytest v2/tests`; after `v1` changes, use `v1` instead.
- When removing code, you may add temporary tests to prove the removed path is no longer used. Delete those tests before finishing unless they verify lasting behavior that should remain part of the suite.
- Avoid permanent test-only scaffolding in production code, datasets, scripts, or configuration. Lasting tests should verify real behavior, and temporary verification files must be deleted before finishing. Keep training, fine-tuning, and inference performance the priority.

# Model development preferences

- Train for the chosen step or epoch budget and treat the final checkpoint as the selected model. For example, a 100,000-step run selects the model at step 100,000. Keep intermediate checkpoints for recovery only; do not add evaluation-based checkpoint selection, best-checkpoint retention, early stopping, or periodic validation unless explicitly requested.
- Keep evaluation work limited to what the user requests; prioritize corpus quality and efficient training. Required software correctness tests still apply.
- Make corpus sampling reproducible and expose explicit size and scan limits so tokenizer generation stays manageable. Tokenizer sampling must not impose the same small corpus budget on pretraining.
- Preserve normalized corpus text, including whitespace normalization.
- Match a fresh pretraining run's default learning-rate schedule to its planned update budget. Preserve explicit schedules and the saved schedule when resuming.
- Keep microbatch and gradient-accumulation defaults unchanged unless explicitly requested; do not prioritize larger-microbatch experiments.
- Limit fine-tuning work to SWAG unless the user explicitly expands its scope.
