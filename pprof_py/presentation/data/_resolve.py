"""Sources to profiles: a fitted model (tested once), a ``test()`` result, or a profile as given."""
from __future__ import annotations

from typing import Any, Optional, Sequence

import pandas as pd

from ._profile import CapabilityError, ProviderProfile

__all__ = ["resolve_profile"]


def resolve_profile(source: Any, args: tuple, kwargs: dict, *, display: str, limits: bool,
                    levels: Optional[Sequence[float]] = None) -> ProviderProfile:
    """A profile from a fitted model (tested once here), a ``test()`` result, or a profile as given."""
    if isinstance(source, ProviderProfile):
        if args or kwargs:
            raise TypeError(f"{display}(): the test settings come from the profile; build a new profile to change "
                            f"them (got {sorted(kwargs) or 'positional arguments'})")
        return source
    if isinstance(source, pd.DataFrame):
        if limits:
            raise CapabilityError(f"{display}() needs control limits from the same test as the flags, and a test() "
                                  "result has none; pass the fitted model, or ProviderProfile.from_model(model, "
                                  "limits=True).")
        return ProviderProfile.from_test(source)
    return ProviderProfile.from_model(source, *args, limits=limits, levels=levels if limits else None, **kwargs)
