# Tests

- `unit/`: tests grouped by subsystem and behavior; shared setup lives in local `conftest.py` files and helper modules.
- `integration/`: workflow and CLI subprocess tests.
- `equivalence/`: numerical checks against captured legacy fixtures.
- `sml_test_support/`: inference and offline evaluation fixtures shared across suites.

Run `uv run pytest v2/tests` outside the sandbox so MLX can access the Apple GPU. Use a subsystem path such as `v2/tests/unit/training` for focused runs.

Check style with `uv run ruff check v2` and `uv run ruff format --check v2`.
