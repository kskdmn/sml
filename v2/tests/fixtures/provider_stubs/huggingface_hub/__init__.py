from __future__ import annotations

from types import SimpleNamespace

_RESOLVED_COMMIT = "0123456789abcdef0123456789abcdef01234567"


class HfApi:
    def dataset_info(
        self,
        dataset_id: str,
        *,
        revision: str,
        timeout: float,
    ) -> SimpleNamespace:
        if dataset_id != "allenai/swag":
            raise RuntimeError("unexpected offline dataset identity")
        if not revision or timeout <= 0:
            raise RuntimeError("invalid offline dataset resolution request")
        return SimpleNamespace(sha=_RESOLVED_COMMIT)
