# Environment

- Python 3.12.13
- Always use `uv run` to run a Python script.

# Rules

- Directories in the project root indicates model version. For example, `v1` is the first version. Unless the user specifies a version, update the latest version. 
- Don't edit files in the top-level (e.g. `pyproject.toml` and `uv.lock`) or clearly ask for the user's approval unless the user asks you to do it.