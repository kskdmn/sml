# Environment

- Python 3.12.13
- Use `uv run` to run Python scripts.
- Always run `uv run pytest` outside the sandbox so MLX/Metal can access the Apple GPU.

# Rules

- Directories in the project root indicate model versions. For example, `v1` is the first version. Unless the user specifies a version, update the latest version.
- Do not edit top-level project files, such as `pyproject.toml` or `uv.lock`, unless the user explicitly asks for that change. If the task truly requires a top-level edit, ask for approval first.
- Before finishing, run `ruff` and `pytest` on the version you updated. For example, after `v2` changes run `uv run ruff check v2`, `uv run ruff format --check v2`, and `uv run pytest v2/tests`; after `v1` changes, use `v1` instead.
- When removing code, you may add temporary tests to prove the removed path is no longer used. Delete those tests before finishing unless they verify lasting behavior that should remain part of the suite.
