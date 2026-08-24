from __future__ import annotations

import sys
from types import ModuleType


class LM:
    def __init__(self) -> None:
        pass


api = ModuleType("lm_eval.api")
model = ModuleType("lm_eval.api.model")
model.LM = LM
api.model = model
sys.modules["lm_eval.api"] = api
sys.modules["lm_eval.api.model"] = model


class _Request:
    def __init__(self, args: tuple[object, ...]) -> None:
        self.args = args


def simple_evaluate(
    *,
    model: LM,
    tasks: list[str],
    num_fewshot: int,
    limit: int | None,
    log_samples: bool,
) -> dict[str, object]:
    if not isinstance(model, LM):
        raise TypeError("offline lm-eval requires an LM adapter")
    if num_fewshot != 0 or log_samples is not False:
        raise RuntimeError("unexpected offline lm-eval policy")
    if limit is not None and limit <= 0:
        raise RuntimeError("offline lm-eval limit must be positive")

    results: dict[str, object] = {}
    for task in tasks:
        scored = model.loglikelihood(
            [
                _Request(("alpha", " beta")),
                _Request(("gamma", " delta")),
            ]
        )
        generated = model.generate_until(
            [_Request(("alpha", {"max_gen_toks": 1, "until": ["omega"]}))]
        )
        results[task] = {
            "acc,none": sum(bool(item[1]) for item in scored) / len(scored),
            "generated": generated,
        }
    return {"results": results}
