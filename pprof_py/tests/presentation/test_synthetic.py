"""The synthetic generator shared by docs, the gallery and tests: deterministic, with the documented structure."""
import numpy as np
import pandas as pd

from pprof_py.presentation._synthetic import provider_data


def test_deterministic_and_seeded():
    a, b = provider_data(80), provider_data(80)
    pd.testing.assert_frame_equal(a, b)
    assert a.attrs == b.attrs and not provider_data(80, seed=1).equals(a)


def test_planted_structure():
    d = provider_data(120)
    planted = d.attrs["planted"]
    size = d.groupby("provider_id").size()
    events = d.groupby("provider_id")["y"].sum()
    assert planted["zero_events"] and all(events[p] == 0 and 11 <= size[p] <= 40 for p in planted["zero_events"])
    assert len(planted["high"]) == len(planted["low"]) == 3
    assert not (set(planted["high"]) & set(planted["low"])) and not (set(planted["zero_events"]) & set(planted["high"]))
    assert size.min() >= 8 and size.max() <= 2000 and set(d.columns) == {"provider_id", "x1", "x2", "y", "y2", "y_cont"}
    assert np.isclose(d["x2"].mean(), 0.4, atol=0.05)
