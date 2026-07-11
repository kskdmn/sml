# Environment

- Python 3.12.13
- Use `uv run` to run Python scripts.
- Always run `uv run pytest` outside the sandbox so MLX/Metal can access the Apple GPU.

# Rules

- Directories in the project root indicate model versions. For example, `v1` is the first version. Unless the user specifies a version, update the latest version.
- Do not edit top-level project files, such as `pyproject.toml` or `uv.lock`, unless the user explicitly asks for that change. If the task truly requires a top-level edit, ask for approval first.
- Ensure all unit tests pass before finishing.
- When removing code, you may add temporary tests to prove the removed path is no longer used. Delete those tests before finishing unless they verify lasting behavior that should remain part of the suite.
