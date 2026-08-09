from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from sml.errors import SMLConfigurationError


def _build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="SML workflows")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    if "--help" in arguments or "-h" in arguments:
        parser.print_help()
        return 0
    if not arguments:
        raise SMLConfigurationError("a command is required")
    raise SMLConfigurationError(f"unknown command: {arguments[0]}")
