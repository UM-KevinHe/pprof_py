"""Pre-flight diagnostics for survival data, run BEFORE attempting to fit.

Everything here is read-only inspection of whatever DataFrame you already
have in memory -- nothing in this module makes a network call or writes
data anywhere, so it's safe to point at sensitive data (e.g. a CMS claims
extract) without that being a concern.

The synthetic datasets this package was validated against
(r_reference/data/) were constructed to be clean: no missing values, no
non-numeric junk, no pathologically tiny strata. Real production data
usually isn't, and CoxPH.fit()'s validation (data/survival_validation.py) is
deliberately strict -- it raises immediately on the first NaN it finds,
rather than guessing at an imputation. That's the right behavior for
fitting, but a bare `SurvivalDataError: X contains NaN` gives you no
sense of SCALE (is this 3 rows or 300,000?) or WHICH column, across
however many columns your real extract has. This module answers those
questions up front, in one pass, before you spend a fit's worth of time
finding out the hard way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd


@dataclass
class ColumnIssue:
    role: str
    column: str
    n_missing: int
    n_total: int
    detail: str = ""

    @property
    def pct_missing(self) -> float:
        return self.n_missing / self.n_total if self.n_total else 0.0


@dataclass
class PreflightResult:
    n_rows: int
    column_issues: List[ColumnIssue] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    start_stop_violations: int = 0
    n_events: int = 0
    n_distinct_event_times: int = 0
    max_tie_size: int = 0
    tie_time_of_max: Optional[float] = None
    tie_size_is_within_stratum: bool = False
    strata_sizes: Optional[pd.Series] = None
    n_singleton_strata: int = 0
    n_zero_event_strata: int = 0
    covariate_summary: Optional[pd.DataFrame] = None
    weight_summary: Optional[Dict[str, Any]] = None
    offset_summary: Optional[Dict[str, Any]] = None
    fatal: List[str] = field(default_factory=list)  # would make fit() raise
    warnings: List[str] = field(default_factory=list)  # would fit fine, but worth knowing

    def report(self, label: str = "") -> str:
        lines = []
        title = f"Pre-flight data diagnostics{f' — {label}' if label else ''}"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append(f"Rows: {self.n_rows:,}")

        if self.missing_columns:
            lines.append("\n[FATAL] Columns referenced in the spec but not found in the data:")
            for c in self.missing_columns:
                lines.append(f"  - {c}")

        if self.column_issues:
            lines.append("\nMissing / non-finite values by column:")
            for ci in self.column_issues:
                lines.append(
                    f"  {ci.role:>12s}  '{ci.column}': {ci.n_missing:,} missing "
                    f"({ci.pct_missing:.2%}){' -- ' + ci.detail if ci.detail else ''}"
                )
        else:
            lines.append("\nMissing / non-finite values: none found in mapped columns.")

        if self.start_stop_violations:
            lines.append(
                f"\n[FATAL] {self.start_stop_violations:,} row(s) have start >= stop "
                "(zero-length or negative-length interval) -- fit() will reject these."
            )

        lines.append(f"\nEvents: {self.n_events:,}")
        if self.n_distinct_event_times:
            ratio = self.n_events / self.n_distinct_event_times
            scope = "within any single stratum" if self.tie_size_is_within_stratum else "in the data overall (no strata given)"
            lines.append(
                f"Distinct event times: {self.n_distinct_event_times:,} (global; avg tie size "
                f"{ratio:.1f}x); largest tied group {scope}: {self.max_tie_size:,} events"
                + (f" at t={self.tie_time_of_max}" if self.tie_time_of_max is not None else "")
            )
            if self.max_tie_size >= 500:
                lines.append(
                    f"  [WARNING] A tied event time with {self.max_tie_size:,} simultaneous "
                    f"events {scope} will noticeably slow an Efron fit for that stratum (its "
                    "inner correction loop is O(tie size) for that one time point) -- consider "
                    "ties='breslow' if this is expected and exact tie-breaking accuracy isn't "
                    "the priority for this group."
                )

        if self.strata_sizes is not None:
            lines.append(
                f"\nStrata: {len(self.strata_sizes):,} distinct, sizes range "
                f"{int(self.strata_sizes.min())}-{int(self.strata_sizes.max())} "
                f"(median {int(self.strata_sizes.median())})"
            )
            if self.n_singleton_strata:
                lines.append(
                    f"  [WARNING] {self.n_singleton_strata:,} stratum/strata have exactly "
                    "1 observation -- these contribute nothing to the fit (no within-"
                    "stratum comparison is possible) and are silently harmless, but a "
                    "large count often indicates a join produced more distinct strata "
                    "than intended."
                )
            if self.n_zero_event_strata:
                lines.append(
                    f"  [INFO] {self.n_zero_event_strata:,} stratum/strata have zero "
                    "events -- harmless (they contribute to nobody's risk set at any "
                    "death, so drop out of the likelihood), just noting the count."
                )

        if self.covariate_summary is not None and not self.covariate_summary.empty:
            lines.append("\nCovariate summary:")
            lines.append(self.covariate_summary.to_string())

        if self.weight_summary:
            w = self.weight_summary
            lines.append(
                f"\nWeights: range [{w['min']:.4g}, {w['max']:.4g}], "
                f"{w['n_zero']:,} exactly zero, {w['n_unique']:,} distinct values"
            )
            if w["n_unique"] <= 5:
                lines.append(
                    "  [INFO] Few distinct weight values -- if these are meant as exact "
                    "replication counts, results should match an unweighted fit on the "
                    "literally-expanded data; worth spot-checking if this is the intent."
                )

        if self.offset_summary:
            o = self.offset_summary
            lines.append(f"\nOffset: range [{o['min']:.4g}, {o['max']:.4g}], mean {o['mean']:.4g}")
            if max(abs(o["min"]), abs(o["max"])) > 20:
                lines.append(
                    "  [WARNING] Offset magnitude is large enough that exp(offset) alone "
                    "approaches float64 overflow territory for the most extreme rows -- "
                    "double check units (e.g. accidentally-unlogged exposure time)."
                )

        if self.fatal:
            lines.append("\n[FATAL ISSUES -- fit() will raise on these]")
            for f in self.fatal:
                lines.append(f"  - {f}")
        if self.warnings:
            lines.append("\n[WARNINGS -- fit() will run, but worth a look]")
            for w in self.warnings:
                lines.append(f"  - {w}")
        if not self.fatal:
            lines.append("\nNo fatal issues found -- fit() should accept this data as-is.")

        return "\n".join(lines)


def preflight_report(
    df: pd.DataFrame,
    covariates: List[str],
    event: str,
    duration: Optional[str] = None,
    start: Optional[str] = None,
    stop: Optional[str] = None,
    strata: Optional[str] = None,
    offset: Optional[str] = None,
    sample_weight: Optional[str] = None,
    label: str = "",
) -> PreflightResult:
    """Inspect `df` for the same issues `CoxPH.fit()` would reject or that
    would otherwise be worth knowing about before fitting, using the same
    column-role vocabulary as `fit()` itself (duration OR start+stop,
    event, covariates, strata, offset, sample_weight).
    """
    result = PreflightResult(n_rows=len(df))

    stop_col = stop if stop is not None else duration
    role_cols = [("event", event), ("stop/duration", stop_col)]
    if start is not None:
        role_cols.append(("start", start))
    if strata is not None:
        role_cols.append(("strata", strata))
    if offset is not None:
        role_cols.append(("offset", offset))
    if sample_weight is not None:
        role_cols.append(("weight", sample_weight))
    for c in covariates:
        role_cols.append((f"covariate", c))

    for role, col in role_cols:
        if col not in df.columns:
            result.missing_columns.append(f"{role}: '{col}'")
    if result.missing_columns:
        result.fatal.append("one or more mapped columns are missing from the data (see above)")
        return result  # nothing further can be checked safely

    for role, col in role_cols:
        series = df[col]
        n_na = int(series.isna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        n_bad_numeric = int((numeric.isna() & ~series.isna()).sum())
        n_inf = int(np.isinf(numeric.to_numpy(dtype=float, na_value=0.0)).sum())
        total_missing = n_na + n_bad_numeric
        if total_missing > 0 or n_inf > 0:
            detail = ""
            if n_bad_numeric:
                detail = f"{n_bad_numeric} value(s) are non-numeric"
            if n_inf:
                detail = (detail + "; " if detail else "") + f"{n_inf} value(s) are +/-inf"
            result.column_issues.append(ColumnIssue(role, col, total_missing, len(df), detail))
            result.fatal.append(f"{role} column '{col}' has {total_missing} missing/invalid value(s)")

    if start is not None and stop_col in df.columns and start in df.columns:
        s = pd.to_numeric(df[start], errors="coerce")
        e = pd.to_numeric(df[stop_col], errors="coerce")
        result.start_stop_violations = int(((s >= e) & s.notna() & e.notna()).sum())
        if result.start_stop_violations:
            result.fatal.append(f"{result.start_stop_violations} row(s) have start >= stop")

    event_num = pd.to_numeric(df[event], errors="coerce")
    unique_event_vals = set(event_num.dropna().unique().tolist())
    if not unique_event_vals <= {0.0, 1.0}:
        result.fatal.append(f"event column has non-binary values: {sorted(unique_event_vals)}")
    event_mask = event_num == 1
    result.n_events = int(event_mask.sum())
    if stop_col in df.columns:
        event_times = pd.to_numeric(df.loc[event_mask, stop_col], errors="coerce").dropna()
        if len(event_times):
            counts = event_times.value_counts()
            result.n_distinct_event_times = len(counts)
            result.max_tie_size = int(counts.max())
            result.tie_time_of_max = float(counts.idxmax())

        # What actually drives Efron's per-stratum cost is the largest
        # tie WITHIN a single stratum, not the largest tie in the data
        # overall -- a time value with thousands of events nationally can
        # still be a small, cheap tie within any one facility's own risk
        # set. Report the stratum-aware figure whenever strata is given;
        # it supersedes (and is usually much smaller than) the global one.
        if strata is not None and strata in df.columns and len(event_times):
            within_stratum = (
                df.loc[event_mask, [strata, stop_col]]
                .assign(**{stop_col: pd.to_numeric(df.loc[event_mask, stop_col], errors="coerce")})
                .groupby([strata, stop_col])
                .size()
            )
            if len(within_stratum):
                result.max_tie_size = int(within_stratum.max())
                worst_key = within_stratum.idxmax()
                result.tie_time_of_max = float(worst_key[1])
                result.tie_size_is_within_stratum = True

    if strata is not None and strata in df.columns:
        sizes = df.groupby(strata).size()
        result.strata_sizes = sizes
        result.n_singleton_strata = int((sizes == 1).sum())
        events_by_stratum = df.assign(_ev=event_mask.astype(int)).groupby(strata)["_ev"].sum()
        result.n_zero_event_strata = int((events_by_stratum == 0).sum())

    if covariates:
        cov_df = df[covariates].apply(pd.to_numeric, errors="coerce")
        summary = cov_df.describe().T[["min", "mean", "max", "std"]]
        result.covariate_summary = summary
        near_constant = summary.index[(summary["std"].fillna(0) < 1e-10)].tolist()
        if near_constant:
            result.warnings.append(f"near-constant covariate(s) (std ~ 0): {near_constant}")

    if sample_weight is not None and sample_weight in df.columns:
        w = pd.to_numeric(df[sample_weight], errors="coerce").dropna()
        result.weight_summary = dict(
            min=float(w.min()), max=float(w.max()),
            n_zero=int((w == 0).sum()), n_unique=int(w.nunique()),
        )
        if (w < 0).any():
            result.fatal.append("sample_weight has negative value(s)")

    if offset is not None and offset in df.columns:
        o = pd.to_numeric(df[offset], errors="coerce").dropna()
        result.offset_summary = dict(min=float(o.min()), max=float(o.max()), mean=float(o.mean()))

    return result
