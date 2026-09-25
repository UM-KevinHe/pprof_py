import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except ImportError:
    sm = None

from pprof_py.models.linear import LinearRandomEffectModel

_SKIP_MSG = "statsmodels not installed; cross-validation tests skipped"


def one_factor_validation():
    rng = np.random.default_rng(12345)
    ng, m = 35, 18
    g = np.repeat(np.arange(ng), m)
    x = rng.normal(size=len(g))
    b = rng.normal(0.0, 1.4, ng)
    y = 2.2 + 0.8 * x + b[g] + rng.normal(0.0, 0.9, len(g))
    df = pd.DataFrame({"y": y, "x": x, "g": g})

    py_reml = LinearRandomEffectModel(
        verbose=False, reml=True, max_iter_outer=500, tol_outer=1e-10
    ).fit(df, "y", x_vars=["x"], provider_var="g")

    sm_reml = sm.MixedLM.from_formula(
        "y ~ x", groups="g", data=df
    ).fit(
        reml=True, method=["powell"], maxiter=2000, disp=False
    )

    beta_err = np.max(
        np.abs(py_reml.coefficients_["beta"].to_numpy()
               - sm_reml.fe_params.to_numpy())
    )
    re_sd_err = abs(
        py_reml.random_effect_sd_["g"]
        - float(np.sqrt(sm_reml.cov_re.iloc[0, 0]))
    )
    sigma_err = abs(py_reml.sigma_ - float(np.sqrt(sm_reml.scale)))
    criterion_err = abs(py_reml.objective_ + 2.0 * sm_reml.llf)

    assert beta_err < 1e-6
    assert re_sd_err < 1e-4
    assert sigma_err < 1e-5
    assert criterion_err < 1e-5


def crossed_validation():
    rng = np.random.default_rng(20260915)
    n = 1400
    g1 = rng.integers(0, 45, n)
    g2 = rng.integers(0, 30, n)
    x = rng.normal(size=n)
    b1 = rng.normal(0.0, 1.1, 45)
    b2 = rng.normal(0.0, 0.7, 30)
    y = 1.5 + 1.25 * x + b1[g1] + b2[g2] + rng.normal(0.0, 1.0, n)

    df = pd.DataFrame({
        "y": y,
        "x": x,
        "g1": g1.astype(str),
        "g2": g2.astype(str),
    })

    py = LinearRandomEffectModel(
        verbose=False, reml=True, max_iter_outer=400, tol_outer=1e-9
    ).fit(df, "y", x_vars=["x"], provider_var="g1", cluster_vars=["g2"])

    sm_fit = sm.MixedLM.from_formula(
        "y ~ x",
        groups=np.ones(n),
        vc_formula={"g1": "0 + C(g1)", "g2": "0 + C(g2)"},
        re_formula="0",
        data=df,
    ).fit(reml=True, method=["lbfgs"], maxiter=1000, disp=False)

    beta_err = np.max(
        np.abs(py.coefficients_["beta"].to_numpy()
               - sm_fit.fe_params.to_numpy())
    )
    sd_err = [
        abs(py.random_effect_sd_[name] - np.sqrt(sm_fit.vcomp[i]))
        for i, name in enumerate(["g1", "g2"])
    ]
    sigma_err = abs(py.sigma_ - np.sqrt(sm_fit.scale))
    criterion_err = abs(py.objective_ + 2.0 * sm_fit.llf)

    assert beta_err < 1e-5
    assert max(sd_err) < 1e-4
    assert sigma_err < 1e-5
    assert criterion_err < 1e-5


def ml_override_validation():
    rng = np.random.default_rng(123)
    ng = 24
    g = np.repeat(np.arange(ng), 18)
    x = rng.normal(size=len(g))
    b = rng.normal(0.0, 1.0, ng)
    y = 0.7 + 0.9 * x + b[g] + rng.normal(0, 1.2, len(g))
    df = pd.DataFrame({"y": y, "x": x, "g": g})

    model = LinearRandomEffectModel(
        verbose=False, reml=True, max_iter_outer=300
    )
    fit = model.fit(df, "y", x_vars=["x"], provider_var="g", reml=False)

    ref = sm.MixedLM.from_formula(
        "y ~ x", groups="g", data=df
    ).fit(
        reml=False, method=["powell"], maxiter=2000, disp=False
    )

    assert abs(fit.sigma_ - np.sqrt(ref.scale)) < 1e-5
    assert abs(fit.loglike_ - ref.llf) < 1e-5


def offset_validation():
    rng = np.random.default_rng(77)
    g = np.repeat(np.arange(15), 12)
    x = rng.normal(size=len(g))
    offset = rng.normal(size=len(g))
    y = 1.7 + 0.3 * x + offset + rng.normal(0, 1.0, len(g))
    df = pd.DataFrame({"y": y, "x": x, "g": g, "off": offset})

    fit = LinearRandomEffectModel(
        verbose=False, reml=False, max_iter_outer=200
    ).fit(
        df, "y", x_vars=["x"], provider_var="g", offset_var="off"
    )

    ols = sm.OLS(y - offset, sm.add_constant(x)).fit()
    assert np.max(
        np.abs(fit.coefficients_["beta"].to_numpy() - ols.params)
    ) < 1e-8


if __name__ == "__main__":
    if sm is None:
        print(_SKIP_MSG)
    else:
        one_factor_validation()
        crossed_validation()
        ml_override_validation()
        offset_validation()
        print("All validation tests passed.")
