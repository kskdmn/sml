import subprocess
import sys


def test_module_entrypoint_is_available():
    completed = subprocess.run(
        [sys.executable, "-m", "sml", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert "SML workflows" in completed.stdout


def test_empty_module_entrypoint_renders_configuration_error_without_traceback():
    completed = subprocess.run(
        [sys.executable, "-m", "sml"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "SMLConfigurationError: a command is required" in completed.stderr
    assert "Traceback" not in completed.stderr
