from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.utils import handle_non_serializable, make_table

from config import OUTPUT_DIR, SUCCESS_RETURN_CODE, TOKENIZER_MODEL_PATH
from infer_sml import (
    DEFAULT_CHECKPOINT_PATH,
    InferenceTokenizer,
    decode_token_ids,
    encode_prompt,
    load_model,
)
from sml import SMLLanguageModel
from train_sml import load_tokenizer, resolve_device


DEFAULT_RESULTS_PATH = OUTPUT_DIR / "humaneval.json"


class SMLHumanEvalLM(LM):
    def __init__(
        self,
        model: SMLLanguageModel,
        tokenizer: InferenceTokenizer,
        device: torch.device,
    ) -> None:
        """
        The lm-eval base class supplies cache hooks; this adapter keeps the local model,
        tokenizer, and torch device together for generation.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._device = device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        tokenizer_model_path: Path,
        device_name: str = "auto",
    ) -> SMLHumanEvalLM:
        """
        Resolve the device once so checkpoint weights and tokenized prompts land on the
        same runtime target.
        """
        device = resolve_device(device_name)
        return cls(
            model=load_model(checkpoint_path, device),
            tokenizer=load_tokenizer(tokenizer_model_path),
            device=device,
        )

    def loglikelihood(
        self,
        requests: list[Instance],
    ) -> list[tuple[float, bool]]:
        """
        HumanEval is a generation task, so scoring continuations by likelihood would
        indicate the adapter is being used for the wrong benchmark.
        """
        del requests
        raise NotImplementedError("SMLHumanEvalLM only supports generation tasks")

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """
        Rolling likelihood is intentionally unsupported because this adapter only
        implements the generation surface lm-eval needs for HumanEval.
        """
        del requests
        raise NotImplementedError("SMLHumanEvalLM only supports generation tasks")

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """
        lm-eval stores each prompt and generation kwargs in Instance.args; generation is
        clamped to the remaining context window before decoding.
        """
        completions: list[str] = []
        for request in requests:
            context, generation_kwargs = request.args
            if generation_kwargs.get("do_sample", False):
                raise ValueError("SMLHumanEvalLM only supports greedy generation")

            input_ids = encode_prompt(
                self.tokenizer,
                context,
                bos_token_id=self.model.config.bos_token_id,
                device=self.device,
            )
            prompt_length = input_ids.shape[1]
            max_length = self.model.config.effective_max_position_embeddings
            if prompt_length > max_length:
                raise ValueError(
                    "HumanEval prompt exceeds the checkpoint context window: "
                    f"{prompt_length} > {max_length}"
                )

            requested_tokens = int(generation_kwargs.get("max_gen_toks", 256))
            max_new_tokens = min(requested_tokens, max_length - prompt_length)
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.model.config.eos_token_id,
            )
            generated_ids = generated[0, prompt_length:].detach().cpu().tolist()
            completion = decode_token_ids(
                self.tokenizer,
                generated_ids,
                bos_token_id=self.model.config.bos_token_id,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
            )
            completion = _truncate_at_stop(
                completion,
                generation_kwargs.get("until", []),
            )
            completions.append(completion)
            self.cache_hook.add_partial("generate_until", request.args, completion)

        return completions


def _truncate_at_stop(text: str, until: str | list[str] | None) -> str:
    """
    The `until` value may be absent, a single string, or several strings; the earliest
    match wins so overlapping stops behave deterministically.
    """
    stop_sequences = [until] if isinstance(until, str) else until or []
    stop_positions = [
        position
        for stop in stop_sequences
        if stop and (position := text.find(stop)) >= 0
    ]
    return text[: min(stop_positions)] if stop_positions else text


def evaluate_humaneval(
    lm: LM,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    limit: int | None = None,
) -> dict[str, Any] | None:
    """
    HumanEval executes generated Python, so lm-eval requires both the environment flag
    and the explicit unsafe-code confirmation.
    """
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"
    return simple_evaluate(
        model=lm,
        model_args={"path": str(checkpoint_path)},
        tasks=["humaneval"],
        num_fewshot=0,
        batch_size=1,
        limit=limit,
        log_samples=False,
        confirm_run_unsafe_code=True,
    )


def write_results(output_path: Path, results: dict[str, Any]) -> None:
    """
    lm-eval results can contain Path and numpy-like values, so JSON serialization goes
    through lm-eval's non-serializable handler.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False,
            default=handle_non_serializable,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Accept an explicit argv for tests while keeping the unsafe-code warning visible in
    CLI help.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the local SML checkpoint on HumanEval. "
            "HumanEval executes model-generated Python code."
        )
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help=f"checkpoint path (default: {DEFAULT_CHECKPOINT_PATH})",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=TOKENIZER_MODEL_PATH,
        help=f"SentencePiece model path (default: {TOKENIZER_MODEL_PATH})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device such as auto, cpu, cuda, cuda:0, or mps",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N HumanEval problems",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
        help=f"JSON results path (default: {DEFAULT_RESULTS_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Wire checkpoint loading, evaluation, JSON output, and table printing for the
    HumanEval CLI.
    """
    args = parse_args(argv)
    lm = SMLHumanEvalLM.from_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_model_path=args.tokenizer,
        device_name=args.device,
    )
    results = evaluate_humaneval(
        lm=lm,
        checkpoint_path=args.checkpoint,
        limit=args.limit,
    )
    if results is None:
        raise RuntimeError("lm_eval did not return results on this process")

    write_results(args.output, results)
    print(make_table(results))
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
