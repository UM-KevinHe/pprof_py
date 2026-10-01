"""Model plot methods as thin delegates to :mod:`pprof_py.presentation` (one rendering path; D12, spec §13).

The delegates keep the old signatures. Test settings pass through; ``theme``, ``size``, ``title`` and ``highlight``
pass to the new renderers; ``save_path`` saves the result; styling keywords that no longer apply raise a
``DeprecationWarning`` (they will raise a ``TypeError`` in 0.7.0). Displays that cannot agree with their model's
flags keep their old drawing for now, with a ``DeprecationWarning``, and are removed in 0.7.0.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

REMOVAL = "0.7.0"
UNSET = object()                    # tells an explicit ``target=`` from the old default
_RENDER = ("theme", "size", "title", "highlight")


def _split(method: str, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    render = {k: kwargs.pop(k) for k in _RENDER if k in kwargs}
    save_path = kwargs.pop("save_path", None)
    if kwargs:
        warnings.warn(f"{method}(): {', '.join(sorted(kwargs))} no longer have an effect; figures take their style "
                      f"from theme= (pprof_py.presentation.Theme). Passing them will raise a TypeError in {REMOVAL}.",
                      DeprecationWarning, stacklevel=4)
    return render, save_path


def _finish(result: Any, save_path: Optional[str]) -> Any:
    if save_path:
        result.save(save_path)
    return result


def funnel_delegate(model: Any, method: str, *, test_kwargs: Dict[str, Any], alpha: Any, target: Any,
                    kwargs: Dict[str, Any]) -> Any:
    """``model.plot_funnel(...)`` as :func:`pprof_py.presentation.funnel`."""
    from ..presentation import funnel

    render, save_path = _split(method, kwargs)
    if target is not UNSET:
        warnings.warn(f"{method}(): target= has no effect; the reference line is the test's null value. It will "
                      f"raise a TypeError in {REMOVAL}.", DeprecationWarning, stacklevel=3)
    levels = tuple(sorted({round(1.0 - float(a), 10) for a in np.atleast_1d(alpha)}))
    return _finish(funnel(model, levels=levels, **test_kwargs, **render), save_path)


def caterpillar_delegate(model: Any, method: str, *, test_kwargs: Optional[Dict[str, Any]] = None,
                         source: Any = None, use_flags: bool = True, ignored: Iterable[str] = (),
                         kwargs: Dict[str, Any]) -> Any:
    """``model.plot_provider_effects(...)`` and friends as :func:`pprof_py.presentation.caterpillar`."""
    from ..presentation import caterpillar

    render, save_path = _split(method, kwargs)
    notes = list(ignored) + ([] if use_flags else ["use_flags=False (flags are always shown with their test)"])
    if notes:
        warnings.warn(f"{method}(): {'; '.join(notes)} has no effect and will raise a TypeError in {REMOVAL}.",
                      DeprecationWarning, stacklevel=3)
    result = caterpillar(source, **render) if source is not None else caterpillar(model, **(test_kwargs or {}),
                                                                                  **render)
    return _finish(result, save_path)


def legacy(method: str, reason: str) -> None:
    """Warn that a display without a consistent replacement keeps its old drawing until 0.7.0."""
    warnings.warn(f"{method}() is deprecated for this model and will be removed in {REMOVAL}: {reason}",
                  DeprecationWarning, stacklevel=3)
