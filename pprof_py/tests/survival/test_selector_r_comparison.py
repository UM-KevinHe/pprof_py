"""Validate CoxPHSelector against R -- not just "same final variables",
but the exact criterion value at every step, which is a much stronger
check (two different search paths could coincidentally land on the same
final set from a slightly-wrong per-step criterion).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from pprof_py.selection import CoxPHSelector

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "..", "r_reference", "data")
RESULTS = os.path.join(HERE, "..", "..", "r_reference", "results")

RTOL = 1e-6


def _compare_history(py_history: pd.DataFrame, r_csv_name: str, criterion_col: str = "aic"):
    """CoxPHSelector logs an explicit terminal "stop" row (showing the
    criterion value the search settled on); R's step() has no equivalent
    row in the returned object (it only ever *prints* a trace) so the
    reference script here doesn't produce one either. Compare the
    overlapping, action-bearing rows exactly, then check the trailing
    stop row (if present) just confirms no further move was silently
    taken.
    """
    r_hist = pd.read_csv(os.path.join(RESULTS, f"{r_csv_name}.csv"))
    trailing_stop = len(py_history) > 0 and py_history.iloc[-1]["action"].startswith("stop")
    comparable = py_history.iloc[:-1] if trailing_stop else py_history

    assert len(comparable) == len(r_hist), (
        f"step count differs: python={len(comparable)} r={len(r_hist)}\n"
        f"python:\n{py_history}\nr:\n{r_hist}"
    )
    for i in range(len(r_hist)):
        py_row = comparable.iloc[i]
        r_row = r_hist.iloc[i]
        assert py_row["n_variables"] == r_row["n_variables"], f"step {i}: variable count differs"
        py_val = py_row[criterion_col]
        r_val = r_row["aic"]
        assert py_val == pytest.approx(r_val, rel=RTOL), f"step {i}: {criterion_col} python={py_val} r={r_val}"
        if pd.notna(r_row["variable"]):
            assert py_row["variable"] == r_row["variable"], (
                f"step {i}: python chose {py_row['variable']!r}, R chose {r_row['variable']!r}"
            )

    if trailing_stop:
        # The stop row's own criterion value must equal the last real
        # move's -- i.e. it's purely a marker, not a move that changed
        # the model.
        assert py_history.iloc[-1][criterion_col] == pytest.approx(
            comparable.iloc[-1][criterion_col], rel=1e-12
        )


def test_forward_aic_matches_r():
    d = pd.read_csv(os.path.join(DATA, "selector_test_data.csv"))
    sel = CoxPHSelector(direction="forward", criterion="aic")
    sel.fit(d[["x1", "x2", "x3", "x4", "x5"]], duration=d["stop"], event=d["event"])
    _compare_history(sel.selection_history_, "forward_aic")
    assert sel.selected_variables_ == ["x1", "x2", "x5"]


def test_backward_aic_matches_r():
    d = pd.read_csv(os.path.join(DATA, "selector_test_data.csv"))
    sel = CoxPHSelector(direction="backward", criterion="aic")
    sel.fit(d[["x1", "x2", "x3", "x4", "x5"]], duration=d["stop"], event=d["event"])
    _compare_history(sel.selection_history_, "backward_aic")


def test_both_aic_matches_r():
    d = pd.read_csv(os.path.join(DATA, "selector_test_data.csv"))
    sel = CoxPHSelector(direction="both", criterion="aic")
    sel.fit(d[["x1", "x2", "x3", "x4", "x5"]], duration=d["stop"], event=d["event"])
    _compare_history(sel.selection_history_, "both_aic")


def test_backward_bic_matches_r_using_nevent_not_nobs():
    """The specific point this test exists for: BIC's log(n) penalty
    must use n_EVENTS (confirmed via R's nobs.coxph), not n_obs -- using
    n_obs would silently produce a different, wrong trajectory."""
    d = pd.read_csv(os.path.join(DATA, "selector_test_data.csv"))
    k_bic = pd.read_csv(os.path.join(RESULTS, "bic_k.csv"))
    assert k_bic["nevent"].iloc[0] < len(d), "sanity: nevent should be less than n_obs given censoring"

    sel = CoxPHSelector(direction="backward", criterion="bic")
    sel.fit(d[["x1", "x2", "x3", "x4", "x5"]], duration=d["stop"], event=d["event"])
    _compare_history(sel.selection_history_, "backward_bic", criterion_col="bic")


def test_forward_aic_with_forced_variable_matches_r():
    d = pd.read_csv(os.path.join(DATA, "selector_test_data.csv"))
    sel = CoxPHSelector(direction="forward", criterion="aic")
    sel.fit(
        d[["x1", "x2", "x3", "x4", "x5"]], duration=d["stop"], event=d["event"],
        forced=["x4"],
    )
    _compare_history(sel.selection_history_, "forward_aic_forced")
    assert "x4" in sel.selected_variables_


def test_forward_aic_with_strata_offset_weights_matches_r():
    """The core requirement: selection must work identically whether or
    not strata/offset/weights are present, and never treat them as
    selectable candidates."""
    d = pd.read_csv(os.path.join(DATA, "selector_strata_data.csv"))
    sel = CoxPHSelector(direction="forward", criterion="aic")
    sel.fit(
        d[["x1", "x2", "x3"]], duration=d["stop"], event=d["event"],
        strata=d["provider"], offset=d["off1"], sample_weight=d["wt"],
    )
    _compare_history(sel.selection_history_, "forward_aic_strata_offset_weights")

    # And confirm strata/offset/weight actually took effect on the final model
    # (not silently dropped somewhere in the selection loop).
    assert sel.final_model_.baseline_hazard_["stratum"].nunique() > 1


def test_candidates_parameter_restricts_pool():
    """`candidates=` must exclude any column not listed, even if it's
    present in X -- distinct from `forced=`, which always includes."""
    d = pd.read_csv(os.path.join(DATA, "selector_test_data.csv"))
    sel = CoxPHSelector(direction="forward", criterion="aic")
    sel.fit(
        d[["x1", "x2", "x3", "x4", "x5"]], duration=d["stop"], event=d["event"],
        candidates=["x3", "x4"],  # deliberately excludes x1, x2, x5 -- the actually-informative ones
    )
    assert set(sel.selected_variables_) <= {"x3", "x4"}
    assert "x1" not in sel.selected_variables_


def test_efron_ties_pass_through():
    d = pd.read_csv(os.path.join(DATA, "selector_test_data.csv"))
    sel = CoxPHSelector(direction="forward", criterion="aic", ties="efron")
    sel.fit(d[["x1", "x2", "x3", "x4", "x5"]], duration=d["stop"], event=d["event"])
    assert sel.final_model_.ties == "efron"
    assert sel.selected_variables_  # should still select something sensible


def test_left_truncated_start_stop_pass_through():
    d = pd.read_csv(os.path.join(DATA, "left_truncation.csv"))
    sel = CoxPHSelector(direction="forward", criterion="aic")
    sel.fit(d[["x1", "x2"]], start=d["start"], stop=d["stop"], event=d["event"])
    assert sel.selected_variables_
    # Confirm start/stop genuinely took effect: fitting the same variables
    # ignoring start (i.e. as if right-censored from 0) should generally
    # give a different log-likelihood than respecting left truncation.
    from pprof_py import CoxPH
    with_truncation = CoxPH().fit(d[sel.selected_variables_], start=d["start"], stop=d["stop"], event=d["event"])
    without_truncation = CoxPH().fit(d[sel.selected_variables_], duration=d["stop"], event=d["event"])
    assert with_truncation.log_likelihood_ != pytest.approx(without_truncation.log_likelihood_)
    assert sel.final_model_.log_likelihood_ == pytest.approx(with_truncation.log_likelihood_)


def test_final_model_is_a_real_fitted_coxph():
    d = pd.read_csv(os.path.join(DATA, "selector_test_data.csv"))
    sel = CoxPHSelector(direction="forward", criterion="aic")
    sel.fit(d[["x1", "x2", "x3", "x4", "x5"]], duration=d["stop"], event=d["event"])

    m = sel.final_model_
    assert list(m.feature_names_in_) == sel.selected_variables_
    assert m.converged_
    assert m.baseline_hazard_ is not None
    assert m.standard_errors_.shape == (len(sel.selected_variables_),)


def test_pvalue_forward_and_backward_are_consistent_with_reported_pvalues():
    """No R package is available to validate the p-value procedure
    itself against (see selector.py's module docstring) -- so this test
    instead checks the property that must hold regardless of which
    textbook convention is used: every variable the forward search adds
    has a p-value below p_enter in the model it was added to, and
    nothing left in a backward search's final model has a p-value above
    p_remove."""
    d = pd.read_csv(os.path.join(DATA, "selector_test_data.csv"))

    sel_fwd = CoxPHSelector(direction="forward", criterion="pvalue", p_enter=0.05)
    sel_fwd.fit(d[["x1", "x2", "x3", "x4", "x5"]], duration=d["stop"], event=d["event"])
    for _, row in sel_fwd.selection_history_.iterrows():
        if row["action"] == "add":
            assert row["p_value"] < 0.05

    sel_bwd = CoxPHSelector(direction="backward", criterion="pvalue", p_remove=0.10)
    sel_bwd.fit(d[["x1", "x2", "x3", "x4", "x5"]], duration=d["stop"], event=d["event"])
    final_p = sel_bwd.final_model_.p_values_
    assert np.all(final_p <= 0.10) or len(sel_bwd.selected_variables_) == 0
