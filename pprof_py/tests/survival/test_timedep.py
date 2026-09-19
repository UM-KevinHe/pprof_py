"""Tests for `data/timedep.py` (`build_skeleton`, `tmerge`, `survsplit`).

Every `tmerge` semantic rule this checks (default `init` for `tdc` vs
`cumtdc`, whether a pre-`tstart` or post-`tstop` update counts, how
`event`/`cumevent` differ from `tdc`/`cumtdc`, chaining calls on top of
an already-multi-row skeleton) is transcribed directly from R survival's
own `tests/tmerge.R` / `tmerge.Rout.save`, run there specifically to
validate `tmerge()` itself -- not just plausible-looking numbers made up
for this package. Two rules were NOT obvious from the man page and were
only pinned down by working through these exact examples: `cumtdc`'s
default (unspecified) `init` reads as NaN until the first qualifying
update, not 0 from the start; and `cumevent` is sparse like `event`
(zero except exactly at a matching row) rather than a running total
carried across every row the way `cumtdc` is.

The `survsplit` checks are reconstructed from R's `tests/survSplit.R`:
the exact `tstart`/`tstop` pattern that script's own `veteran`-dataset
example asserts (reconstructed here from three subjects whose original
follow-up times are recoverable from that assertion, since `veteran`
itself isn't available outside R) and its tie-safety check (splitting at
a cut point that coincides with an existing row boundary must not
produce a duplicate/degenerate breakpoint).
"""
import numpy as np
import pandas as pd
import pytest

from pprof_py.data.timedep import build_skeleton, tmerge, survsplit, UpdateStream


@pytest.fixture
def r_test1():
    """R: data1 <- data.frame(idd=c(1,5,4,3,2,6), x1=1:6, age=50:55);
    test1 <- tmerge(data1, data1, id=idd, death=event(age))
    """
    idd = np.array([1, 5, 4, 3, 2, 6])
    age = np.array([50, 51, 52, 53, 54, 55], dtype=float)
    skeleton = build_skeleton(id=idd, tstop=age)
    test1 = tmerge(
        skeleton.id.to_numpy(), skeleton.tstart.to_numpy(), skeleton.tstop.to_numpy(),
        event={"death": UpdateStream(id=idd, time=age)},
    )
    return idd, age, test1


def test_build_skeleton_plus_event_reproduces_r_test1(r_test1):
    idd, age, test1 = r_test1
    assert test1.id.tolist() == [1, 5, 4, 3, 2, 6]
    assert test1.tstart.tolist() == [0, 0, 0, 0, 0, 0]
    assert test1.tstop.tolist() == [50, 51, 52, 53, 54, 55]
    assert test1.death.tolist() == [1, 1, 1, 1, 1, 1]


def test_tdc_reproduces_r_test2(r_test1):
    idd, age, test1 = r_test1
    idd2 = np.array([2, 5, 1, 2, 1])
    x2 = np.array([5, 4, 3, 2, 1], dtype=float)
    age2 = np.array([48, 47, 46, 45, 44], dtype=float)
    test2 = tmerge(
        test1.id.to_numpy(), test1.tstart.to_numpy(), test1.tstop.to_numpy(),
        tdc={"zed": UpdateStream(id=idd2, time=age2, value=x2)},
        event={"death": UpdateStream(id=idd, time=age)},
    )
    assert test2.id.tolist() == [1, 1, 1, 5, 5, 4, 3, 2, 2, 2, 6]
    assert test2.tstop.tolist() == [44, 46, 50, 47, 51, 52, 53, 45, 48, 54, 55]
    assert test2.death.tolist() == [0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1]
    np.testing.assert_allclose(
        test2.zed.to_numpy(),
        [np.nan, 1, 3, np.nan, 4, np.nan, np.nan, np.nan, 2, 5, np.nan],
        equal_nan=True,
    )


def test_cumtdc_with_explicit_init_and_pre_tstart_updates_reproduces_r_test3a(r_test1):
    """The "surfeit of rows" example: updates before a subject's tstart
    (negative ages) still compete to set the value carried into the
    first row; updates after tstop (and for an id not in the skeleton
    at all) are dropped entirely.
    """
    idd, age, test1 = r_test1
    idd3 = np.array([2, 5, 1, 2, 1, 2, 2, 1, 1, 7, 3, 3])
    x2_3 = np.array([5, 4, 3, 2, 1, 10, 9, 8, 7, 6, 5, 4], dtype=float)
    age3 = np.array([48, 47, 46, 45, 44, -4, -3, -1, -2, 35, 62, 61], dtype=float)

    test3 = tmerge(
        test1.id.to_numpy(), test1.tstart.to_numpy(), test1.tstop.to_numpy(),
        tdc={"xx": UpdateStream(id=idd3, time=age3, value=x2_3)},
        cumtdc={"cx": UpdateStream(id=idd3, time=age3, value=x2_3)},
        cumtdc_init={"cx": 2.0},
        event={"death": UpdateStream(id=idd, time=age)},
    )
    assert test3.id.tolist() == [1, 1, 1, 5, 5, 4, 3, 2, 2, 2, 6]
    np.testing.assert_allclose(
        test3.xx.to_numpy(), [8, 1, 3, np.nan, 4, np.nan, np.nan, 9, 2, 5, np.nan], equal_nan=True,
    )
    np.testing.assert_allclose(test3.cx.to_numpy(), [17, 18, 21, 2, 6, 2, 2, 21, 23, 28, 2])


def test_cumtdc_default_init_and_cumevent_reproduce_r_test3b(r_test1):
    """Chained on top of test2's already-multi-row skeleton, inserting
    yet more breakpoints into existing rows -- and specifically
    exercises `cumtdc`'s DEFAULT (unspecified) init (NaN until the
    first qualifying update, not 0) and `cumevent` (sparse, unlike
    `cumtdc`).
    """
    idd, age, test1 = r_test1
    idd2 = np.array([2, 5, 1, 2, 1])
    x2 = np.array([5, 4, 3, 2, 1], dtype=float)
    age2 = np.array([48, 47, 46, 45, 44], dtype=float)
    test2 = tmerge(
        test1.id.to_numpy(), test1.tstart.to_numpy(), test1.tstop.to_numpy(),
        tdc={"zed": UpdateStream(id=idd2, time=age2, value=x2)},
        event={"death": UpdateStream(id=idd, time=age)},
    )

    idd3 = np.array([5, 5, 1, 1, 6, 4, 3, 2])
    age3 = np.array([45, 50, 44, 48, 53, -5, 0, 20], dtype=float)
    x3 = np.array([1, 5, 2, 3, 7, 4, 6, 8], dtype=float)

    test3b = tmerge(
        test2.id.to_numpy(), test2.tstart.to_numpy(), test2.tstop.to_numpy(),
        cumtdc={"x": UpdateStream(id=idd3, time=age3, value=x3)},
        cumevent={"esum": UpdateStream(id=idd3, time=age3)},
        event={"death": UpdateStream(id=idd, time=age)},
        tdc={"zed": UpdateStream(id=idd2, time=age2, value=x2)},
    )
    assert len(test3b) == 16
    np.testing.assert_allclose(
        test3b.x.to_numpy(),
        [np.nan, 2, 2, 5, np.nan, 1, 1, 6, 4, 6, np.nan, 8, 8, 8, np.nan, 7],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        test3b.esum.to_numpy(), [1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 1, 0, 0, 0, 1, 0],
    )


def test_tmerge_preserves_first_appearance_order():
    # id order in the input is 1,5,4,3,2,6 -- not sorted -- and R's own
    # output preserves that order rather than sorting by id.
    idd = np.array([1, 5, 4, 3, 2, 6])
    age = np.array([50, 51, 52, 53, 54, 55], dtype=float)
    skeleton = build_skeleton(id=idd, tstop=age)
    out = tmerge(skeleton.id.to_numpy(), skeleton.tstart.to_numpy(), skeleton.tstop.to_numpy())
    assert out.id.tolist() == [1, 5, 4, 3, 2, 6]


def test_build_skeleton_rejects_tstart_not_less_than_tstop():
    with pytest.raises(ValueError):
        build_skeleton(id=[1, 2], tstop=[5.0, 3.0], tstart=[5.0, 0.0])


def test_survsplit_matches_r_veteran_reconstruction():
    """Reconstructed from `tests/survSplit.R`'s own assertions on the
    `veteran` dataset (not available outside R) -- three subjects whose
    original follow-up times are recoverable from the exact
    tstart/tstop pattern that script checks for `cut=c(90,180)`.
    """
    ids = np.array([1, 2, 3])
    tstart = np.zeros(3)
    tstop = np.array([72.0, 411.0, 288.0])
    event = np.array([1.0, 1.0, 1.0])
    out = survsplit(ids, tstart, tstop, event, cut=[90, 180])
    assert out.id.tolist() == [1, 2, 2, 2, 3, 3, 3]
    assert out.tstart.tolist() == [0, 0, 90, 180, 0, 90, 180]
    assert out.tstop.tolist() == [72, 90, 180, 411, 90, 180, 288]
    # only the LAST piece of each subject keeps their real event/censoring status
    assert out.event.tolist() == [1, 0, 0, 1, 0, 0, 1]


def test_survsplit_cut_coinciding_with_existing_boundary_has_no_duplicate_breakpoint():
    """A roundoff/tie issue R's own test suite specifically guards
    against: cutting at a point that already coincides with (or is very
    close to) an existing row boundary must not produce a
    zero-width/duplicate row.
    """
    ids = np.array([1, 1, 1, 1])
    t1 = np.array([0, 1.2, 2.8, 4.0])
    t2 = np.array([1.2, 2.8, 4.0, 5.9])
    status = np.zeros(4)
    out = survsplit(ids, t1, t2, status, cut=[1, 4])
    rounded = np.round(out.tstart.to_numpy(), 5)
    assert len(rounded) == len(set(rounded.tolist()))
    assert np.all(out.tstop.to_numpy() > out.tstart.to_numpy())


def test_survsplit_cut_outside_range_is_a_no_op_for_that_subject():
    ids = np.array([1, 2])
    tstart = np.zeros(2)
    tstop = np.array([5.0, 5.0])
    event = np.array([1.0, 0.0])
    out = survsplit(ids, tstart, tstop, event, cut=[100])  # far beyond any follow-up
    assert len(out) == 2
    assert out.tstop.tolist() == [5.0, 5.0]


def test_tmerge_output_recovers_known_time_varying_effect_in_coxph():
    """End-to-end: a piecewise-exponential simulation with a KNOWN
    treatment effect that only takes hold after a per-subject switch
    time, built into (start, stop] form via `build_skeleton` + `tmerge`,
    must let `CoxPH` recover that true coefficient.
    """
    from pprof_py.models.survival.coxph import CoxPH

    rng = np.random.default_rng(5)
    n = 2000
    switch_time = rng.exponential(3.0, n)
    beta_true = -0.6
    rate0 = 0.4

    # Piecewise-exponential simulation via the cumulative-hazard inverse:
    # H(t) = rate0*t for t < switch_time, else rate0*switch_time +
    # rate0*exp(beta_true)*(t - switch_time); solve H(T) = E for E ~ Exp(1).
    E = rng.exponential(1.0, n)
    budget_at_switch = rate0 * switch_time
    event_time = np.where(
        E <= budget_at_switch,
        E / rate0,
        switch_time + (E - budget_at_switch) / (rate0 * np.exp(beta_true)),
    )
    censor = rng.exponential(4.0, n)
    stop = np.minimum(event_time, censor)
    event = (event_time <= censor).astype(float)
    ids = np.arange(n)

    skeleton = build_skeleton(id=ids, tstop=stop)
    switched = switch_time < stop
    merged = tmerge(
        skeleton.id.to_numpy(), skeleton.tstart.to_numpy(), skeleton.tstop.to_numpy(),
        tdc={"treated": UpdateStream(id=ids[switched], time=switch_time[switched])},
        tdc_init={"treated": 0.0},
        event={"death": UpdateStream(id=ids[event.astype(bool)], time=stop[event.astype(bool)])},
    )
    X = merged[["treated"]].to_numpy()
    model = CoxPH().fit(
        X, start=merged.tstart.to_numpy(), stop=merged.tstop.to_numpy(), event=merged.death.to_numpy(),
    )
    z = (model.coef_[0] - beta_true) / model.standard_errors_[0]
    assert abs(z) < 3.5  # well within sampling noise of the true value
