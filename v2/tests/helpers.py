from dataclasses import dataclass
from typing import Any

_UNSET = object()


@dataclass(frozen=True)
class RecordedCall:
    args: tuple[Any, ...]
    kwargs: dict[str, Any]


class Spy:
    def __init__(
        self,
        return_value: Any = None,
        *,
        side_effect: Any = None,
        wraps: Any = _UNSET,
    ) -> None:
        self.return_value = return_value
        self.side_effect = side_effect
        self.wraps = wraps
        self.call_args_list: list[RecordedCall] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.call_args_list.append(RecordedCall(args=args, kwargs=kwargs))
        if self.side_effect is not None:
            if isinstance(self.side_effect, BaseException):
                raise self.side_effect
            return self.side_effect(*args, **kwargs)
        if self.wraps is not _UNSET:
            return self.wraps(*args, **kwargs)
        return self.return_value

    @property
    def call_args(self) -> RecordedCall:
        assert self.call_args_list, "spy was not called"
        return self.call_args_list[-1]

    @property
    def call_count(self) -> int:
        return len(self.call_args_list)

    def assert_called_once(self) -> None:
        assert self.call_count == 1

    def assert_called_once_with(self, *args: Any, **kwargs: Any) -> None:
        self.assert_called_once()
        call = self.call_args
        assert call.args == args
        assert call.kwargs == kwargs
