"""Formatting rules shared by tables, labels and alt text (brief §6.4, S12)."""
import numpy as np
import pandas as pd
import pytest

from pprof_py.presentation import formatting as f

M = f.MINUS


def test_minus_sign_negative_zero_and_grouping():
    assert f.fmt_number(-1.5, 1) == f"{M}1.5"
    assert f.fmt_number(-0.004, 2) == "0.00"
    assert f.fmt_number(-0.0) == "0.00"
    assert f.fmt_number(1234.567, 1) == "1,234.6"
    assert f.fmt_number(1234.567, 1, grouping=False) == "1234.6"


def test_non_finite_and_missing_values_are_never_numbers():
    assert f.fmt_number(np.inf) == "\u221e" and f.fmt_number(-np.inf) == f"{M}\u221e"
    for v in (np.nan, None, pd.NA):
        assert f.fmt_number(v) == "\u2014"
    assert f.fmt_number(np.nan, missing="NE") == "NE"


def test_significant_and_scientific():
    assert f.fmt_number(0.012345, 3, kind="sig") == "0.0123"
    assert f.fmt_number(1234.5, 3, kind="sig") == "1,230"
    assert f.fmt_number(2.5e-5, 1, kind="sci") == "2.5 \u00d7 10\u207b\u2075"
    with pytest.raises(ValueError):
        f.fmt_number(1.0, kind="pct")


def test_counts_are_whole_numbers():
    assert f.fmt_count(1245) == "1,245" and f.fmt_count(3.0) == "3" and f.fmt_count(np.nan) == "\u2014"
    with pytest.raises(ValueError):
        f.fmt_count(2.5)


def test_percent_and_ratio():
    assert f.fmt_percent(0.123) == "12.3%" and f.fmt_ratio(1.256) == "1.26"


@pytest.mark.parametrize("args, expected", [
    ((0.90, 0.71, 1.11), "0.90 (0.71\u20131.11)"),
    ((-0.40, -0.62, -0.18), f"{M}0.40 ({M}0.62 to {M}0.18)"),
    ((0.20, -0.10, 0.50), f"0.20 ({M}0.10 to 0.50)"),
    ((1.20, np.nan, np.nan), "1.20 (NI)"),
    ((np.nan, 0.1, 0.2), "\u2014"),
    ((0.0, -np.inf, 1.0), f"0.00 ({M}\u221e to 1.00)"),
])
def test_intervals(args, expected):
    assert f.fmt_interval(*args) == expected


def test_invalid_intervals_raise():
    with pytest.raises(ValueError):
        f.fmt_interval(1.0, 0.5, np.nan)
    with pytest.raises(ValueError):
        f.fmt_interval(1.0, 2.0, 1.5)


def test_p_values_are_never_zero():
    assert f.fmt_p([0.0449, 0.0004, 0.0, 1.0, np.nan]).tolist() == ["0.045", "<0.001", "<0.001", "1.000", "\u2014"]
    assert f.fmt_p(2e-5, sci=True) == "2.0 \u00d7 10\u207b\u2075" and f.fmt_p(0.0, sci=True) == "<0.001"
    assert f.fmt_p(0.00004, digits=4, threshold=0.0001) == "<0.0001"
    with pytest.raises(ValueError):
        f.fmt_p(1.2)


def test_flags():
    s = pd.Series([1, -1, 0, pd.NA], dtype="Int64", index=list("abcd"))
    out = f.fmt_flag(s)
    assert out.tolist() == ["\u25b2", "\u25bc", "\u25cf", "NT"] and out.index.tolist() == list("abcd")
    with pytest.raises(ValueError):
        f.fmt_flag(2)


def test_vectorisation_shapes_and_no_mutation():
    s = pd.Series([1.0, -2.5, np.nan], index=["p1", "p2", "p3"], name="est")
    before = s.copy()
    out = f.fmt_number(s, 1)
    assert isinstance(out, pd.Series) and out.index.equals(s.index) and out.name == "est"
    pd.testing.assert_series_equal(s, before)
    arr = f.fmt_number([1.0, 2.0])
    assert isinstance(arr, np.ndarray) and arr.dtype == object
    assert isinstance(f.fmt_number(1.0), str)
    assert list(arr) == [f.fmt_number(v) for v in (1.0, 2.0)]
    with pytest.raises(ValueError):
        f.fmt_interval([1.0, 2.0], [0.0], [3.0, 4.0])


def test_per_row_digits():
    assert f.fmt_number([1.23456, 1.23456], [2, 4]).tolist() == ["1.23", "1.2346"]


def test_rounding_collision_rule():
    digits, unresolved = f.resolve_digits([1.004, 0.71, 1.0000001], [1.30, 1.11, 1.2], 1.0, [1, 0, 1])
    assert digits.tolist() == [3, 2, 6] and unresolved.tolist() == [False, False, True]
    assert f.fmt_interval(1.15, 1.004, 1.30, digits[0]) == "1.150 (1.004\u20131.300)"
    digits, _ = f.resolve_digits([0.80], [0.996], 1.0, [-1])
    assert digits.tolist() == [3]
    digits, unresolved = f.resolve_digits([0.9, 0.9], [1.1, 1.1], 1.0, pd.array([pd.NA, 0], dtype="Int64"))
    assert digits.tolist() == [2, 2] and not unresolved.any()


def test_wrap_text_fits_width():
    text = "Logistic FE \u00b7 unshrunken fixed effects \u00b7 test: score \u00b7 null: theoretical \u00b7 two-sided 95% "
    lines = f.wrap_text(text * 2, 85, 7).splitlines()
    width = int(85 / (0.55 * 7 * 25.4 / 72))
    assert len(lines) > 1 and all(len(line) <= width for line in lines)
