"""Fine-Gray transform tests.

`finegray_transform`'s two headline checks (`test_r_test1_*` and
`test_r_test3_left_truncation`) are transcribed from R survival's own
`tests/finegray.R` / `finegray.Rout.save` -- the package's bundled,
already-solved worked examples for exactly this function, used here the
same way this project already uses live R output elsewhere: as an exact
numeric target, not a tolerance-loosened approximation.

The split-invariance tests exist because R's own bundled examples are all
one-row-per-subject and so cannot exercise a multi-row (time-varying-
covariate) subject combined with left truncation -- a real gap, found by
constructing exactly that combination and checking a property that must
hold regardless: since `finegray_transform` never looks at covariates,
splitting one subject's row into two contiguous pieces at an arbitrary
point (status 0 on the earlier piece) cannot change the transform's
output for ANY subject, including other subjects entirely. An earlier
version of this module failed that check -- it counted every row's own
`start`/`stop` as a genuine entry/censoring event, rather than only a
subject's true first/last row -- and is kept here as a permanent
regression test.
"""
import numpy as np
import pytest

from coxph.algorithms.competing_risks import finegray_transform

# The 14-subject example used throughout R survival's tests/finegray.R.
TIME = np.array([1, 2, 3, 4, 4, 4, 5, 5, 6, 8, 8, 9, 10, 12], dtype=float)
STATUS = np.array([1, 2, 0, 1, 0, 0, 2, 1, 0, 0, 2, 0, 1, 0], dtype=float)  # 0=cen, 1=type1, 2=type2
IDS = np.arange(1, 15)

# The left-truncated variant of the same 14 subjects ("Test data set 3" in
# finegray.R).
TIME1_TRUNC = np.array([0, 0, 0, 3, 2, 0, 0, 1, 0, 7, 5, 0, 0, 0], dtype=float)
TIME2_TRUNC = TIME


def _csurv_p():
    return np.cumprod([1, 11 / 12, 8 / 10, 5 / 6, 3 / 4, 2 / 3])


def test_r_test1_row_expansion_and_weights():
    """R: `finegray(Surv(time,status)~., fdata)` (failcode = type1)."""
    fg = finegray_transform(np.zeros(14), TIME, STATUS, failcode=1.0, id=IDS)

    expected_id = [1, 2, 2, 2, 2, 3, 4, 5, 6, 7, 7, 8, 9, 10, 11, 11, 12, 13, 14]
    p = _csurv_p()
    expected_wt = [1, p[0], p[1], p[2], p[5], 1, 1, 1, 1, 1, 5 / 12, 1, 1, 1, 1, 1 / 2, 1, 1, 1]

    assert (fg.row + 1).tolist() == expected_id
    np.testing.assert_allclose(fg.weight, expected_wt, atol=1e-9)


def test_r_test1_alternate_failcode():
    """Same data, failcode = type2 -- exercises a *different* set of rows
    needing expansion (now the type1 events are the competing ones).
    """
    fg = finegray_transform(np.zeros(14), TIME, STATUS, failcode=2.0, id=IDS)
    # id=1 (time=1, status=type1) is now competing and is the very first
    # subject, so it must be expanded starting from stop=1.
    id1_weights = fg.weight[fg.row == 0]
    np.testing.assert_allclose(sorted(id1_weights), sorted([1.0, 0.733333, 0.611111]), atol=1e-5)


def test_r_test3_left_truncation():
    """R: "Test data set 3" -- same 14 subjects, now with delayed entry.

    finegray.R validates its own `finegray()` output against an
    independently-computed pair of censoring (G) and truncation (H)
    curves, using their textbook at-risk/event-count definitions rather
    than finegray()'s internal implementation. Reproduced here as the
    ground truth for this package's implementation too.
    """
    status = STATUS
    fg = finegray_transform(TIME1_TRUNC, TIME2_TRUNC, status, failcode=1.0, id=IDS)

    # Ordinary (non-competing-event) rows must be untouched.
    stat2_ids = set(IDS[status == 2].tolist())
    is_expanded = np.isin(fg.row + 1, list(stat2_ids))
    assert np.all(fg.weight[~is_expanded] == 1.0)
    assert np.all(fg.start[~is_expanded] == TIME1_TRUNC[fg.row[~is_expanded]])
    assert np.all(fg.stop[~is_expanded] == TIME2_TRUNC[fg.row[~is_expanded]])

    tt = np.sort(np.unique(np.concatenate([TIME1_TRUNC, TIME2_TRUNC])))
    ntime = tt.size
    Grisk = np.zeros(ntime)
    Gevent = np.zeros(ntime)
    Hrisk = np.zeros(ntime)
    Hevent = np.zeros(ntime)
    for i, t in enumerate(tt):
        Grisk[i] = np.sum(
            ((TIME2_TRUNC > t) & (status > 0) & (TIME1_TRUNC < t))
            | ((TIME2_TRUNC >= t) & (status == 0) & (TIME1_TRUNC < t))
        )
        Gevent[i] = np.sum((TIME2_TRUNC == t) & (status == 0))
        Hrisk[i] = np.sum((TIME2_TRUNC > t) & (TIME1_TRUNC <= t))
        Hevent[i] = np.sum(TIME1_TRUNC == t)
    G = np.cumprod(1 - Gevent / np.maximum(1, Grisk))
    H = np.array([np.prod((1 - Hevent / np.maximum(1, Hrisk))[i + 1 :]) for i in range(ntime)])

    tdata_stop = fg.stop[is_expanded]
    tdata_row = fg.row[is_expanded]
    first_pos = {}
    index = np.empty(tdata_row.size, dtype=int)
    for k, r in enumerate(tdata_row):
        first_pos.setdefault(r, k)
        index[k] = first_pos[r]

    m = np.searchsorted(tt, tdata_stop)
    Gwt = np.concatenate([[1.0], G])[m]
    Hwt = np.concatenate([[0.0], H])[m]
    expected = (Gwt * Hwt) / (Gwt * Hwt)[index]
    np.testing.assert_allclose(fg.weight[is_expanded], expected, atol=1e-9)


@pytest.mark.parametrize("split_row_1based,split_at", [(11, 6.5), (3, 1.5), (2, 1.0)])
def test_split_invariance_under_left_truncation(split_row_1based, split_at):
    """Splitting one subject's row into two contiguous pieces at an
    arbitrary covariate-boundary point must not change ANY subject's
    output -- see module docstring. Covers a competing-event subject
    (row 11), a plain-censored subject (row 3), and a type1 subject
    (row 2), each under left truncation.
    """
    status = STATUS
    fg_base = finegray_transform(TIME1_TRUNC, TIME2_TRUNC, status, failcode=1.0, id=IDS)

    split_idx = split_row_1based - 1
    keep = np.ones(14, dtype=bool)
    keep[split_idx] = False
    new_time1 = np.concatenate([TIME1_TRUNC[keep], [TIME1_TRUNC[split_idx], split_at]])
    new_time2 = np.concatenate([TIME2_TRUNC[keep], [split_at, TIME2_TRUNC[split_idx]]])
    new_status = np.concatenate([status[keep], [0.0, status[split_idx]]])
    new_ids = np.concatenate([IDS[keep], [split_row_1based, split_row_1based]])

    fg_split = finegray_transform(new_time1, new_time2, new_status, failcode=1.0, id=new_ids)

    def by_id(fg, id_map):
        out = {}
        for r, a, b, st, wt in zip(fg.row, fg.start, fg.stop, fg.status, fg.weight):
            out.setdefault(int(id_map[r]), []).append(
                (round(float(a), 6), round(float(b), 6), float(st), round(float(wt), 6))
            )
        return {k: sorted(v) for k, v in out.items()}

    base_by_id = by_id(fg_base, IDS)
    split_by_id = by_id(fg_split, new_ids)

    for subj_id in base_by_id:
        if subj_id == split_row_1based:
            continue  # this subject's own row count legitimately changes
        assert base_by_id[subj_id] == split_by_id.get(subj_id), f"subject {subj_id} changed"

    # And the split subject's own rows, once its two weight-1 base pieces
    # are merged back into one, must exactly reproduce the unsplit base
    # piece plus every later added piece, unchanged.
    base_rows = base_by_id[split_row_1based]
    split_rows = split_by_id[split_row_1based]
    weight_one_pieces = sorted(r for r in split_rows if r[3] == 1.0)
    assert weight_one_pieces[0][0] == TIME1_TRUNC[split_idx]  # merged piece starts where the original did
    merged_stop = max(r[1] for r in weight_one_pieces)
    merged_base = (weight_one_pieces[0][0], merged_stop, 0.0, 1.0)
    added_pieces = [r for r in split_rows if r[3] != 1.0]
    assert sorted([merged_base] + added_pieces) == base_rows


def test_finegray_rejects_transition_before_last_row():
    start = np.array([0.0, 0.0])
    stop = np.array([5.0, 10.0])
    event = np.array([1.0, 0.0])  # first row (not last, by stop within id) has an event
    ids = np.array([1, 1])
    with pytest.raises(ValueError):
        finegray_transform(start, stop, event, failcode=1.0, id=ids)


def test_finegray_rejects_gap_between_rows():
    start = np.array([0.0, 6.0])  # gap: first row ends at 5, second starts at 6
    stop = np.array([5.0, 10.0])
    event = np.array([0.0, 1.0])
    ids = np.array([1, 1])
    with pytest.raises(ValueError):
        finegray_transform(start, stop, event, failcode=1.0, id=ids)


def test_finegray_rejects_strata_change_mid_subject():
    start = np.array([0.0, 5.0])
    stop = np.array([5.0, 10.0])
    event = np.array([0.0, 1.0])
    ids = np.array([1, 1])
    strata = np.array(["a", "b"])
    with pytest.raises(ValueError):
        finegray_transform(start, stop, event, failcode=1.0, id=ids, strata=strata)


def test_no_events_of_failcode_in_a_stratum_contributes_nothing():
    # Two strata; failcode only occurs in stratum 0.
    start = np.zeros(4)
    stop = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.array([1.0, 2.0, 0.0, 2.0])
    strata = np.array([0, 0, 1, 1])
    ids = np.arange(1, 5)
    fg = finegray_transform(start, stop, event, failcode=1.0, id=ids, strata=strata)
    # nothing from stratum 1 should appear at all
    assert np.all(np.isin(fg.row, [0, 1]))


def test_sample_weight_multiplies_final_weight_only():
    fg_unweighted = finegray_transform(np.zeros(14), TIME, STATUS, failcode=1.0, id=IDS)
    sw = np.full(14, 2.0)
    fg_weighted = finegray_transform(np.zeros(14), TIME, STATUS, failcode=1.0, id=IDS, sample_weight=sw)
    np.testing.assert_allclose(fg_weighted.weight, fg_unweighted.weight * 2.0)
    np.testing.assert_array_equal(fg_weighted.start, fg_unweighted.start)
    np.testing.assert_array_equal(fg_weighted.stop, fg_unweighted.stop)
