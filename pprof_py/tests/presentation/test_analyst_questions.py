"""Brief §3.6: each analyst question can be answered in a few lines of public API (the line counts are tested)."""
import inspect

import pytest

from pprof_py import LogisticFixedEffectModel, LogisticRandomEffectModel
from pprof_py.presentation import (FigureResult, ProfileCollection, ProviderProfile, caterpillar, flag_stability,
                                   flag_stability_table, funnel, measure_agreement, multi_measure_table, provider_table,
                                   provider_variation, provider_variation_table)
from pprof_py.presentation._synthetic import provider_data


def unusually_high_or_low(model):
    """Which providers look unusually high or low?"""
    profile = ProviderProfile.from_model(model, limits=True)
    return funnel(profile), profile.data.query("status in ['above', 'below']")


def how_uncertain(model):
    """How uncertain are the estimates?"""
    profile = ProviderProfile.from_model(model, test_method="poibin_exact")
    return caterpillar(profile), profile.data[["estimate", "ci_lower", "ci_upper"]]


def against_the_benchmark(model):
    """How do providers compare with the benchmark?"""
    profile = ProviderProfile.from_model(model)
    return profile.provenance["reference_value"], profile.status_counts()


def extremes_small_or_large(model):
    """Are the extreme providers small or large?"""
    f = ProviderProfile.from_model(model, limits=True).data
    extreme = (f["funnel_estimate"] - 1.0).abs().nlargest(10).index
    return f.loc[extreme, "expected"].median(), f["expected"].median()


def variation_across_providers(random_model):
    """How much variation exists across providers?"""
    return provider_variation(random_model), provider_variation_table(random_model).to_frame()


def how_stable(model):
    """How stable are the rankings? (No ranking is offered; how fragile are the flags?)"""
    return flag_stability(model), flag_stability_table(model).to_frame()["changed"]


def across_measures(model, second_model):
    """How do providers compare across measures?"""
    measures = ProfileCollection({"Measure A": model, "Measure B": second_model})
    return measure_agreement(measures), multi_measure_table(measures).to_frame()


def one_estimate(model, provider):
    """How should one estimate be read, given its uncertainty?"""
    return caterpillar(model, highlight=[provider]), provider_table(model).to_frame().loc[provider]


QUESTIONS = (unusually_high_or_low, how_uncertain, against_the_benchmark, extremes_small_or_large,
             variation_across_providers, how_stable, across_measures, one_estimate)


@pytest.fixture(scope="module")
def fits():
    d = provider_data(120)
    fe, re, fe2 = LogisticFixedEffectModel(), LogisticRandomEffectModel(verbose=False), LogisticFixedEffectModel()
    fe.fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    re.fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    fe2.fit(d, y_var="y2", x_vars=["x1"], provider_var="provider_id")
    return fe, re, fe2


def test_each_question_takes_a_few_lines():
    for q in QUESTIONS:
        body = [line for line in inspect.getsource(q).splitlines()[1:] if line.strip() and '"""' not in line]
        assert len(body) <= 3, (q.__name__, len(body))


def test_each_question_is_answered(fits):
    fe, re, fe2 = fits
    fig, flagged = unusually_high_or_low(fe)
    assert isinstance(fig, FigureResult) and set(flagged["status"]) <= {"above", "below"} and len(flagged) > 0
    fig, intervals = how_uncertain(fe)
    assert isinstance(fig, FigureResult) and (intervals["ci_lower"] <= intervals["ci_upper"]).all()
    reference, counts = against_the_benchmark(fe)
    assert reference is not None and counts["above"] + counts["below"] + counts["not_different"] > 0
    extreme, typical = extremes_small_or_large(fe)
    assert extreme > 0 and typical > 0
    fig, table = variation_across_providers(re)
    assert isinstance(fig, FigureResult) and table.loc["Random-effect SD (\u03c3)", "value"] > 0
    fig, changed = how_stable(fe)
    assert isinstance(fig, FigureResult) and changed.iloc[1:].ge(0).all()
    fig, table = across_measures(fe, fe2)
    union = fe.test().index.union(fe2.test().index)               # providers kept by data preparation in either fit
    assert isinstance(fig, FigureResult) and len(table) == len(union)
    fig, row = one_estimate(fe, "F023")
    assert isinstance(fig, FigureResult) and row["ci_lower"] <= row["estimate"] <= row["ci_upper"]
