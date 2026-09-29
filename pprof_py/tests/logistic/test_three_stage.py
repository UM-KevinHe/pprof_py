"""LogisticThreeStageModel: the pipeline, and parity with R's glmer and glmm.fac.hosp for Stages 2-3."""
import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

from pprof_py import LogisticThreeStageModel

GOLDEN = pathlib.Path(__file__).resolve().parents[1] / "data" / "three_stage"
Z = ["age", "diabetes", "chf", "comorb", "female"]


@pytest.fixture(scope="module")
def fitted():
    raw = pd.read_csv(GOLDEN / "raw.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticThreeStageModel(bound_mode="absolute", convergence_criterion="relative", estimator="he2013").fit(
            raw, "Y", Z, "fac", "hosp")                                # R's Stage 3 settings
    return model, json.loads((GOLDEN / "r_stage23.json").read_text())


def _scalar(v):
    return v[0] if isinstance(v, list) else v


def test_stage1_gives_the_beta_r_was_given(fitted):
    model, r = fitted
    assert np.allclose(np.ravel(model.stage1_.coefficients_["beta"]), r["beta"], rtol=0, atol=1e-12)


def test_stage2_matches_glmer(fitted):
    model, r = fitted
    s2, s3 = model.stage2_, model.stage3_
    assert abs(s3.sigma_ - _scalar(r["sigma"])) < 1e-4
    start = s2.get_random_effects("fac").reindex(s3.provider_ids_).to_numpy() + float(s2.coefficients_["beta"]["(Intercept)"])
    assert np.max(np.abs(start - np.asarray(r["gamma_init"]))) < 1e-4


def test_stage3_matches_glmm_fac_hosp(fitted):
    model, r = fitted
    assert np.max(np.abs(model.stage3_.gamma_ - np.asarray(r["r_gamma"]))) < 2e-4
    ind = model.calculate_standardized_measures(stdz="indirect")["indirect"]
    assert np.max(np.abs(ind["indirect_ratio"].to_numpy(float) - np.asarray(r["r_SRR"]))) < 2e-4


def test_pipeline_methods_are_stage3s(fitted):
    model, _ = fitted
    pd.testing.assert_frame_equal(model.test(test_method="poibin_exact"), model.stage3_.test(test_method="poibin_exact"))
    pd.testing.assert_frame_equal(model.summary(), model.stage1_.summary(test_method="wald"))
    assert model.stage3_.stage1_ is model.stage1_
    assert (model.data_["included"] == 1).sum() == len(model.stage1_.outcome_.ravel())    # Stage 1 fits the included cells


def test_rejects_data_with_the_offset_column():
    raw = pd.read_csv(GOLDEN / "raw.csv").assign(stage1_offset=0.0)
    with pytest.raises(ValueError, match="stage1_offset"):
        LogisticThreeStageModel().fit(raw, "Y", Z, "fac", "hosp")


def test_marginal_estimator_through_the_pipeline(fitted):
    model, _ = fitted
    raw = pd.read_csv(GOLDEN / "raw.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        marginal = LogisticThreeStageModel(bound_mode="absolute", convergence_criterion="relative",
                                           estimator="marginal").fit(raw, "Y", Z, "fac", "hosp")
    assert marginal.stage3_.converged_ and marginal.stage3_.estimator == "marginal"
    assert marginal.stage3_.loglik_ >= model.stage3_.loglik_ - 1e-9
    assert np.array_equal(marginal.stage3_.beta_, model.stage3_.beta_) and marginal.stage3_.sigma_ == model.stage3_.sigma_
