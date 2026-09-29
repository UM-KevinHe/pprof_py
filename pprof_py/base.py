"""Common base class for pprof_py models."""
from __future__ import annotations

import inspect

from .exceptions import NotFittedError

__all__ = ["ProviderModel"]


class ProviderModel:
    """Base class of every pprof_py model.

    Subclasses store each constructor argument as an attribute of the same name
    and set fitted attributes (names ending in ``_``) in ``fit()``. The base
    provides parameter access, a readable ``repr``, and the fitted check.
    """

    @classmethod
    def _param_names(cls) -> list:
        if cls.__init__ is object.__init__:
            return []
        params = inspect.signature(cls.__init__).parameters.values()
        return sorted(p.name for p in params
                      if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD))

    def get_params(self) -> dict:
        """Constructor parameters and their current values."""
        return {name: getattr(self, name, None) for name in self._param_names()}

    def set_params(self, **params) -> "ProviderModel":
        """Set constructor parameters; unknown names raise ``ValueError``."""
        valid = set(self._param_names())
        for name, value in params.items():
            if name not in valid:
                raise ValueError(f"Invalid parameter {name!r} for {type(self).__name__}.")
            setattr(self, name, value)
        return self

    def __repr__(self) -> str:
        defaults = {p.name: p.default for p in inspect.signature(type(self).__init__).parameters.values()
                    if p.default is not inspect.Parameter.empty} if type(self).__init__ is not object.__init__ else {}
        changed = []
        for name, value in self.get_params().items():
            default = defaults.get(name, inspect.Parameter.empty)
            try:
                same = value is default or bool(value == default)
            except Exception:
                same = False
            if not same:
                changed.append(f"{name}={value!r}")
        return f"{type(self).__name__}({', '.join(changed)})"

    def _require_fitted(self, *attributes: str) -> None:
        """Raise ``NotFittedError`` unless every named attribute exists."""
        missing = [a for a in attributes if not hasattr(self, a)]
        if missing:
            raise NotFittedError(f"This {type(self).__name__} instance is not fitted yet. Call `fit` first.")
