"""Record the internals of one ``test()`` call so that :func:`~pprof_py.inference.funnel_limits` can reuse them.

Inert unless a recording is active: :func:`record` appends to the active sink and otherwise does nothing, so the
statistics of ``test()`` are unchanged. Context variables keep concurrent recordings apart (ADR-003).
"""
from __future__ import annotations

import contextlib
import contextvars
from typing import Any, Dict, Iterator, List, Optional

_SINK: contextvars.ContextVar[Optional[List[Dict[str, Any]]]] = contextvars.ContextVar("pprof_py_test_record",
                                                                                      default=None)


def record(kind: str, **payload: Any) -> None:
    """Append ``{"kind": kind, **payload}`` to the active recording, if any."""
    sink = _SINK.get()
    if sink is not None:
        sink.append({"kind": kind, **payload})


@contextlib.contextmanager
def recording() -> Iterator[List[Dict[str, Any]]]:
    """Collect every :func:`record` call made inside the ``with`` block."""
    token = _SINK.set([])
    try:
        yield _SINK.get()
    finally:
        _SINK.reset(token)
