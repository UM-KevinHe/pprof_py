"""Model plot methods as thin delegates to :mod:`pprof_py.presentation` (one rendering path; D12, spec §13).

The delegates keep the old signatures. Test settings pass through; ``theme``, ``size``, ``title`` and ``highlight``
pass to the new renderers; ``save_path`` saves the result. Keywords the figures do not take raise a ``TypeError``:
the styling keywords, ``target=`` and ``use_flags=False`` were deprecated in 0.6.0 and removed in 0.7.0 (D91).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np

_RENDER = ("theme", "size", "title", "highlight")


def _split(method: str, kwargs: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    render = {k: kwargs.pop(k) for k in _RENDER if k in kwargs}
    save_path = kwargs.pop("save_path", None)
    if kwargs:
        raise TypeError(f"{method}() got unexpected keyword arguments {sorted(kwargs)}; figures take their style "
                        "from theme= (pprof_py.presentation.Theme), and the earlier styling keywords were removed in "
                        "0.7.0")
    return render, save_path


def _finish(result: Any, save_path: Optional[str]) -> Any:
    if save_path:
        result.save(save_path)
    return result


def funnel_delegate(model: Any, method: str, *, test_kwargs: Dict[str, Any], alpha: Any,
                    kwargs: Dict[str, Any]) -> Any:
    """``model.plot_funnel(...)`` as :func:`pprof_py.presentation.funnel`."""
    from ..presentation import funnel

    render, save_path = _split(method, kwargs)
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
        raise TypeError(f"{method}(): {'; '.join(notes)} was removed in 0.7.0")
    result = caterpillar(source, **render) if source is not None else caterpillar(model, **(test_kwargs or {}),
                                                                                  **render)
    return _finish(result, save_path)
