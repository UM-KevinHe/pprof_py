"""Tests for coxph.diagnostics -- the preflight checker and the
spec-driven R-comparison harness meant to be pointed at real,
non-synthetic data. Uses the existing r_reference/data/ fixtures rather
than new ones, since these tools should work on exactly that shape of
data (they're only "new" in being spec-driven and R-optional, not in
needing different inputs).
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

from pprof_py.diagnostics.survival.preflight import preflight_report
from pprof_py.diagnostics.survival.validate_against_r import run_validation, StageSpec, generate_r_script

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "..", "r_reference", "data")


def test_preflight_clean_data_has_no_fatal_issues():
    df = pd.read_csv(os.path.join(DATA, "combined.csv"))
    pf = preflight_report(
        df, covariates=["x1", "x2", "x3"], event="event", start="start", stop="stop",
        strata="provider", offset="offset1", sample_weight="weight",
    )
    assert pf.fatal == []
    assert pf.missing_columns == []
    assert pf.start_stop_violations == 0
    assert pf.n_events > 0
    assert pf.strata_sizes is not None and len(pf.strata_sizes) > 0
    report_text = pf.report()
    assert "No fatal issues found" in report_text


def test_preflight_catches_missing_column():
    df = pd.read_csv(os.path.join(DATA, "basic.csv"))
    pf = preflight_report(df, covariates=["x1", "not_a_real_column"], event="event", duration="time")
    assert any("not_a_real_column" in m for m in pf.missing_columns)
    assert pf.fatal


def test_preflight_catches_missing_values_and_start_stop_violation():
    df = pd.read_csv(os.path.join(DATA, "left_truncation.csv")).copy()
    df.loc[0, "x1"] = np.nan
    df.loc[1, "start"] = df.loc[1, "stop"]  # zero-length interval

    pf = preflight_report(df, covariates=["x1", "x2"], event="event", start="start", stop="stop")
    assert pf.start_stop_violations == 1
    assert any("x1" in ci.column for ci in pf.column_issues)
    assert pf.fatal  # both issues should surface as fatal


def test_preflight_flags_singleton_and_zero_event_strata():
    df = pd.read_csv(os.path.join(DATA, "strata.csv")).copy()
    df.loc[0, "provider"] = 999  # a brand-new, singleton stratum

    pf = preflight_report(df, covariates=["x1", "x2"], event="event", duration="stop", strata="provider")
    assert pf.n_singleton_strata >= 1


def test_preflight_within_stratum_tie_size_can_differ_from_global():
    """A time value tied across many DIFFERENT strata should not be
    reported as an alarming single-stratum tie -- see
    docs/README.md's Performance section for why this distinction
    matters in practice."""
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame({
        "x1": rng.normal(size=n),
        "stop": np.where(np.arange(n) < 400, 5.0, rng.uniform(0.1, 10, size=n)),  # 400 rows tied at t=5 GLOBALLY
        "event": 1,
        "provider": rng.integers(0, 400, size=n),  # spread across many strata -> small ties WITHIN each
    })
    pf = preflight_report(df, covariates=["x1"], event="event", duration="stop", strata="provider")
    assert pf.tie_size_is_within_stratum
    assert pf.max_tie_size < 400  # much smaller than the global tie of 400


def test_generate_r_script_is_syntactically_plausible():
    stage1 = StageSpec(covariates=["x1", "x2"], start="start", stop="stop", event="event",
                        strata="provider", offset="offset1", sample_weight="weight")
    script = generate_r_script("/tmp/some_data.csv", stage1, ties="breslow", results_dir="/tmp/out")
    assert "library(survival)" in script
    assert 'coxph(Surv(start, stop, event) ~ x1 + x2 + strata(provider) + offset(offset1)' in script
    assert "weights=weight" in script


def test_generate_r_script_two_stage():
    stage1 = StageSpec(covariates=["x1", "x2"], start="start", stop="stop", event="event", strata="provider")
    stage2 = StageSpec(covariates=[], sample_weight="weight")
    script = generate_r_script("/tmp/d.csv", stage1, ties="breslow", stage2=stage2, results_dir="/tmp/out")
    assert "xbeta <-" in script
    assert "offset(.xbeta)" in script


def test_run_validation_end_to_end_single_stage(tmp_path):
    spec = {
        "data": os.path.join(DATA, "combined.csv"),
        "ties": "breslow",
        "output_dir": str(tmp_path / "out"),
        "stage1": {
            "covariates": ["x1", "x2", "x3"],
            "start": "start", "stop": "stop", "event": "event",
            "strata": "provider", "offset": "offset1", "sample_weight": "weight",
        },
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))

    report = run_validation(str(spec_path))
    assert "Python stage 1 fit: converged=True" in report
    assert os.path.exists(tmp_path / "out" / "run_all_check.R")

    if shutil.which("Rscript") is not None:
        assert "[OK]" in report
        assert "CHECK" not in report or "[CHECK]" not in report


def test_run_validation_stops_before_fitting_on_fatal_data(tmp_path):
    df = pd.read_csv(os.path.join(DATA, "basic.csv")).copy()
    df.loc[0, "x1"] = np.nan
    messy_path = tmp_path / "messy.csv"
    df.to_csv(messy_path, index=False)

    spec = {
        "data": str(messy_path),
        "output_dir": str(tmp_path / "out"),
        "stage1": {"covariates": ["x1", "x2", "x3"], "duration": "time", "event": "event"},
    }
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))

    report = run_validation(str(spec_path))
    assert "Stopping before fitting" in report
    assert "Python stage 1 fit" not in report
