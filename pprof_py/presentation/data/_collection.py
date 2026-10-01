"""Several measures of the same providers: an ordered, immutable collection of profiles (spec §6.1; D28)."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Iterator, Mapping

import pandas as pd

from ._profile import CapabilityError, ProviderProfile

__all__ = ["ProfileCollection"]


class ProfileCollection:
    """Named measures of the same providers, in a fixed order.

    Parameters
    ----------
    measures : mapping of label to source
        Each source is a :class:`ProviderProfile`, a ``test()`` result (``ProviderProfile.from_test``) or a fitted
        model (``ProviderProfile.from_model`` with its defaults); labels name the measures in figures and tables.

    Notes
    -----
    Providers need not be the same in every measure: :meth:`providers` is the union (first-seen order) and
    :meth:`common` the intersection. Displays show a provider missing from a measure as not applicable, never drop it.
    """

    __slots__ = ("_profiles",)

    def __init__(self, measures: Mapping[str, Any]) -> None:
        if not measures:
            raise ValueError("a ProfileCollection needs at least one measure")
        out = {}
        for label, source in measures.items():
            label = str(label)
            if label in out:
                raise ValueError(f"duplicate measure label {label!r}")
            if isinstance(source, ProviderProfile):
                out[label] = source
            elif isinstance(source, pd.DataFrame):
                out[label] = ProviderProfile.from_test(source)
            elif hasattr(source, "test"):
                out[label] = ProviderProfile.from_model(source)
            else:
                raise TypeError(f"measure {label!r}: expected a ProviderProfile, a test() result or a fitted model")
        object.__setattr__(self, "_profiles", MappingProxyType(out))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("ProfileCollection is immutable; build a new collection instead")

    @property
    def labels(self) -> list:
        return list(self._profiles)

    @property
    def profiles(self) -> Mapping[str, ProviderProfile]:
        return self._profiles

    def __getitem__(self, label: str) -> ProviderProfile:
        return self._profiles[label]

    def __iter__(self) -> Iterator[str]:
        return iter(self._profiles)

    def __len__(self) -> int:
        return len(self._profiles)

    def __repr__(self) -> str:
        return f"ProfileCollection({len(self)} measures: {', '.join(self.labels)}; {len(self.providers())} providers)"

    def providers(self) -> pd.Index:
        """All providers, in the order they first appear across the measures."""
        seen: dict = {}
        for p in self._profiles.values():
            for pid in p.data.index:
                seen.setdefault(pid, None)
        return pd.Index(list(seen), name="provider_id")

    def common(self) -> pd.Index:
        """Providers present in every measure, in first-seen order."""
        idx = self.providers()
        for p in self._profiles.values():
            idx = idx[idx.isin(p.data.index)]
        return idx

    def column(self, name: str) -> pd.DataFrame:
        """One profile column for every measure: providers (union) by measure labels; missing providers are NA."""
        idx = self.providers()
        return pd.DataFrame({label: p.data[name].reindex(idx) for label, p in self._profiles.items()}, index=idx)

    def require(self, display: str, n: int = 2) -> None:
        if len(self) < n:
            raise CapabilityError(f"{display} needs at least {n} measures; the collection has {len(self)}")
