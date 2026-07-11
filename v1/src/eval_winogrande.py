"""
Winogrande evaluation for the local SML checkpoint and tokenizer.

Results default to ``v1/output/winogrande.json``. Override paths and runtime
with ``--checkpoint``, ``--tokenizer``, ``--device``, ``--limit``, and
``--output``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from lm_eval.api.model import LM
from lm_eval.utils import make_table

from config import OUTPUT_DIR, SUCCESS_RETURN_CODE
from eval_utils import (
    SMLEvalLM,
    build_eval_parser,
    evaluate_lm,
    require_results,
    write_results,
)
from infer_sml import DEFAULT_CHECKPOINT_PATH


TASK_NAME = "winogrande"
DEFAULT_RESULTS_PATH = OUTPUT_DIR / "winogrande.json"


def evaluate_winogrande(
    lm: LM,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    limit: int | None = None,
) -> dict[str, Any] | None:
    """
    Run the lm-eval Winogrande task with zero-shot scoring.
    """
    return evaluate_lm(
        lm=lm,
        checkpoint_path=checkpoint_path,
        tasks=[TASK_NAME],
        limit=limit,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Accept an explicit argv for tests while keeping CLI defaults in one place.
    """
    parser = build_eval_parser(
        description="Evaluate the local SML checkpoint on Winogrande.",
        default_results_path=DEFAULT_RESULTS_PATH,
        limit_help="evaluate only the first N Winogrande examples",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Load the checkpoint, run Winogrande, save JSON results, and print a table.
    """
    args = parse_args(argv)
    lm = SMLEvalLM.from_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_model_path=args.tokenizer,
        device_name=args.device,
    )
    results = require_results(
        evaluate_winogrande(
            lm=lm,
            checkpoint_path=args.checkpoint,
            limit=args.limit,
        )
    )

    write_results(args.output, results)
    print(make_table(results))
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
