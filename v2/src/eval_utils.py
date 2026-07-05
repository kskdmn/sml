"""
Shared lm-eval helpers for SML benchmark scripts.

The adapter supports likelihood-based multiple-choice tasks and greedy
generation tasks while keeping checkpoint loading, result writing, and common
CLI arguments in one place.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from lm_eval import simple_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.utils import handle_non_serializable

from config import TOKENIZER_MODEL_PATH
from infer_sml import (
    DEFAULT_CHECKPOINT_PATH,
    InferenceTokenizer,
    decode_token_ids,
    encode_prompt,
    load_model,
)
from sml import SMLLanguageModel
from train_sml import load_tokenizer, resolve_device


class SMLEvalLM(LM):
    def __init__(
        self,
        model: SMLLanguageModel,
        tokenizer: InferenceTokenizer,
        device: torch.device,
    ) -> None:
        """
        Keep the local model, tokenizer, and runtime device together for lm-eval.
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
    ) -> SMLEvalLM:
        """
        Resolve the device once so checkpoint weights and tokenized prompts match.
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

            input_ids = torch.tensor(
                [token_ids],
                dtype=torch.long,
                device=self.device,
            )
            with torch.no_grad():
                output = self.model(input_ids)

            logits = output.logits
            log_probs = F.log_softmax(logits, dim=-1)
            continuation_positions = range(continuation_start, len(token_ids))
            logprob = 0.0
            is_greedy = True
            for target_position in continuation_positions:
                predictor_position = target_position - 1
                target_token_id = token_ids[target_position]
                logprob += float(
                    log_probs[0, predictor_position, target_token_id].item()
                )
                greedy_token_id = int(logits[0, predictor_position].argmax().item())
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
        Perplexity-style rolling likelihood is not needed by these benchmark scripts.
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
                device=self.device,
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
            generated_ids = generated[0, prompt_length:].detach().cpu().tolist()
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


def build_eval_parser(
    description: str,
    default_results_path: Path,
    limit_help: str,
) -> argparse.ArgumentParser:
    """
    Build the shared checkpoint/tokenizer/device/output parser.
    """
    parser = argparse.ArgumentParser(description=description)
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
        help=limit_help,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_results_path,
        help=f"JSON results path (default: {default_results_path})",
    )
    return parser


def require_results(results: dict[str, Any] | None) -> dict[str, Any]:
    """
    lm-eval returns results only on the primary process.
    """
    if results is None:
        raise RuntimeError("lm_eval did not return results on this process")
    return results
