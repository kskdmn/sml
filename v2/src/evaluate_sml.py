"""
Benchmark evaluation for the local SML checkpoint and tokenizer.

Select the lm-eval task with ``--benchmark hellaswag`` or ``--benchmark
winogrande``. Results default to ``v2/output/<benchmark>.json``. Override
paths with ``--model``, ``--tokenizer-model``, ``--limit``, and ``--output``.

The adapter supports likelihood-based multiple-choice tasks and greedy
generation tasks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.utils import handle_non_serializable, make_table

from config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_TOKENIZER_MODEL_PATH,
    OUTPUT_DIR,
    SUCCESS_RETURN_CODE,
)
from infer_sml import InferenceTokenizer, decode_token_ids, encode_prompt, load_model
from utils import load_tokenizer

BENCHMARKS = ("hellaswag", "winogrande")


class SMLEvalLM(LM):
    def __init__(
        self,
        model: Any,
        tokenizer: InferenceTokenizer,
    ) -> None:
        """
        Keep the local model and tokenizer together for lm-eval.
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        tokenizer_model_path: Path,
    ) -> SMLEvalLM:
        """
        Load the checkpoint and tokenizer used by benchmark tasks.
        """
        return cls(
            model=load_model(checkpoint_path),
            tokenizer=load_tokenizer(tokenizer_model_path),
        )

    def loglikelihood(
        self,
        requests: list[Instance],
    ) -> list[tuple[float, bool]]:
        """
        Score each continuation by summing next-token log-probabilities.
        """
        results: list[tuple[float, bool]] = []
        for request in requests:
            context, continuation = request.args
            token_ids, continuation_start = self._encode_context_continuation(
                context,
                continuation,
            )
            max_length = self.model.config.effective_max_position_embeddings
            if len(token_ids) > max_length:
                raise ValueError(
                    "prompt plus continuation exceeds the checkpoint context window: "
                    f"{len(token_ids)} > {max_length}"
                )

            input_ids = mx.array(
                [token_ids],
                dtype=mx.int32,
            )
            output = self.model(input_ids)

            logits = output.logits
            log_probs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            continuation_positions = range(continuation_start, len(token_ids))
            logprob = 0.0
            is_greedy = True
            for target_position in continuation_positions:
                predictor_position = target_position - 1
                target_token_id = token_ids[target_position]
                logprob += float(
                    log_probs[0, predictor_position, target_token_id].item()
                )
                greedy_token_id = int(mx.argmax(logits[0, predictor_position]).item())
                is_greedy = is_greedy and greedy_token_id == target_token_id

            result = (logprob, is_greedy)
            results.append(result)
            self.cache_hook.add_partial("loglikelihood", request.args, result)

        return results

    def _encode_context_continuation(
        self,
        context: str,
        continuation: str,
    ) -> tuple[list[int], int]:
        """
        Tokenize the joined text, then mark the continuation boundary.
        """
        context_ids = self._encode_text(context)
        full_ids = self._encode_text(context + continuation)
        if full_ids[: len(context_ids)] != context_ids:
            continuation_ids = self._encode_text(continuation)
            continuation_start = len(full_ids) - len(continuation_ids)
        else:
            continuation_start = len(context_ids)

        bos_token_id = self.model.config.bos_token_id
        if bos_token_id is not None:
            full_ids = [bos_token_id, *full_ids]
            continuation_start += 1
        elif continuation_start == 0:
            raise ValueError("empty context requires a checkpoint BOS token")

        if continuation_start >= len(full_ids) and continuation:
            raise ValueError("continuation produced no tokens")
        return full_ids, continuation_start

    def _encode_text(self, text: str) -> list[int]:
        return [int(token_id) for token_id in self.tokenizer.encode(text, out_type=int)]

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """
        Perplexity-style rolling likelihood is not needed by these benchmark tasks.
        """
        del requests
        raise NotImplementedError("SMLEvalLM does not support rolling likelihood")

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """
        Greedily generate completions, clamping each request to the context window.
        """
        completions: list[str] = []
        for request in requests:
            context, generation_kwargs = request.args
            if generation_kwargs.get("do_sample", False):
                raise ValueError("SMLEvalLM only supports greedy generation")

            input_ids = encode_prompt(
                self.tokenizer,
                context,
                bos_token_id=self.model.config.bos_token_id,
            )
            prompt_length = input_ids.shape[1]
            max_length = self.model.config.effective_max_position_embeddings
            if prompt_length > max_length:
                raise ValueError(
                    "prompt exceeds the checkpoint context window: "
                    f"{prompt_length} > {max_length}"
                )

            requested_tokens = int(generation_kwargs.get("max_gen_toks", 256))
            max_new_tokens = min(requested_tokens, max_length - prompt_length)
            generated = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                eos_token_id=self.model.config.eos_token_id,
            )
            mx.eval(generated)
            generated_ids = generated[0, prompt_length:].tolist()
            completion = decode_token_ids(
                self.tokenizer,
                generated_ids,
                bos_token_id=self.model.config.bos_token_id,
                eos_token_id=self.model.config.eos_token_id,
                pad_token_id=self.model.config.pad_token_id,
            )
            completion = truncate_at_stop(
                completion,
                generation_kwargs.get("until", []),
            )
            completions.append(completion)
            self.cache_hook.add_partial("generate_until", request.args, completion)

        return completions


def truncate_at_stop(text: str, until: str | list[str] | None) -> str:
    """
    Return text up to the earliest stop sequence, if any.
    """
    stop_sequences = [until] if isinstance(until, str) else until or []
    stop_positions = [
        position
        for stop in stop_sequences
        if stop and (position := text.find(stop)) >= 0
    ]
    return text[: min(stop_positions)] if stop_positions else text


def evaluate_lm(
    lm: LM,
    checkpoint_path: Path,
    tasks: list[str],
    limit: int | None = None,
    **extra_options: Any,
) -> dict[str, Any] | None:
    """
    Call lm-eval with the common local-checkpoint arguments.
    """
    return simple_evaluate(
        model=lm,
        model_args={"path": str(checkpoint_path)},
        tasks=tasks,
        num_fewshot=0,
        batch_size=1,
        limit=limit,
        log_samples=False,
        **extra_options,
    )


def write_results(output_path: Path, results: dict[str, Any]) -> None:
    """
    Serialize lm-eval results, including Path and numpy-like values.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            results,
            output_file,
            indent=2,
            ensure_ascii=False,
            default=handle_non_serializable,
        )
        output_file.write("\n")


def require_results(results: dict[str, Any] | None) -> dict[str, Any]:
    """
    lm-eval returns results only on the primary process.
    """
    if results is None:
        raise RuntimeError("lm_eval did not return results on this process")
    return results


def resolve_results_path(benchmark: str, output: Path | None) -> Path:
    """
    Default the results file to ``output/<benchmark>.json`` unless overridden.
    """
    return OUTPUT_DIR / f"{benchmark}.json" if output is None else output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the local SML checkpoint on an lm-eval benchmark."
    )
    parser.add_argument(
        "--benchmark",
        choices=BENCHMARKS,
        required=True,
        help="benchmark task to run",
    )
    parser.add_argument(
        "--model",
        dest="checkpoint",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"model checkpoint directory (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--tokenizer-model",
        dest="tokenizer_model",
        type=Path,
        default=DEFAULT_TOKENIZER_MODEL_PATH,
        help=f"SentencePiece model path (default: {DEFAULT_TOKENIZER_MODEL_PATH})",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="evaluate only the first N examples",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=f"JSON results path (default: {OUTPUT_DIR}/<benchmark>.json)",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """
    Accept an explicit argv for tests while keeping CLI defaults in one place.
    """
    return build_parser().parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """
    Load the checkpoint, run the selected benchmark with zero-shot scoring, save
    JSON results, and print a table.
    """
    args = parse_args(argv)
    lm = SMLEvalLM.from_checkpoint(
        checkpoint_path=args.checkpoint,
        tokenizer_model_path=args.tokenizer_model,
    )
    results = require_results(
        evaluate_lm(
            lm=lm,
            checkpoint_path=args.checkpoint,
            tasks=[args.benchmark],
            limit=args.limit,
        )
    )

    write_results(resolve_results_path(args.benchmark, args.output), results)
    print(make_table(results))
    return SUCCESS_RETURN_CODE


if __name__ == "__main__":
    raise SystemExit(main())
