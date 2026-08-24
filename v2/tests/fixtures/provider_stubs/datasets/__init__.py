from __future__ import annotations

import os


class _Dataset(list[dict[str, object]]):
    _fingerprint = "sml-cli-offline-swag-v1"


_ROWS = _Dataset(
    [
        {
            "startphrase": "a",
            "ending0": " b",
            "ending1": " c",
            "ending2": " d",
            "ending3": " e",
            "label": 0,
        },
        {
            "startphrase": "b",
            "ending0": " c",
            "ending1": " d",
            "ending2": " e",
            "ending3": " a",
            "label": 1,
        },
        {
            "startphrase": "c",
            "ending0": " d",
            "ending1": " e",
            "ending2": " a",
            "ending3": " b",
            "label": 2,
        },
        {
            "startphrase": "d",
            "ending0": " e",
            "ending1": " a",
            "ending2": " b",
            "ending3": " c",
            "label": 3,
        },
    ]
)


def load_dataset(
    path: str,
    name: str,
    *,
    split: str,
    revision: str,
) -> _Dataset:
    if os.environ.get("SML_TEST_DATASETS_FAIL") == "1":
        raise RuntimeError("offline dataset fixture failure")
    if (path, name, split) != ("allenai/swag", "regular", "train"):
        raise RuntimeError("unexpected offline SWAG request")
    if not revision:
        raise RuntimeError("offline SWAG revision is required")
    return _Dataset(_ROWS)
