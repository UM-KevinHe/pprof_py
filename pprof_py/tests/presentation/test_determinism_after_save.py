"""A figure built after a save is byte-identical to the first figure of a fresh interpreter (ADR-006).

Drawing and saving leave FreeType state behind, so text measured later can differ in its last bit; that moved axes by
~1e-16 (changing Matplotlib's content-hashed SVG ids) or left a sub-figure at -0.0 (written "-0" in PDF). Frozen
layouts (axes and sub-figure boxes) are now quantised with negative zero normalised, and the hashed SVG ids
canonicalised. Each figure type runs cold in its own interpreter: build and save (SVG, PDF, PNG), build again and
save again, compare.

The effect was intermittent across interpreters (before the ids were canonicalised, the reliability figure differed
in 5 of 12 runs, and the observed-versus-expected PDF in 2 of 3 before sub-figure boxes were quantised; 0 of 10 or
more afterwards), so one run per type guards the fixed behaviour but detects a regression only with some
probability.
"""
import subprocess
import sys
import textwrap

import pytest

SCRIPT = textwrap.dedent('''
    import sys, warnings, logging
    warnings.simplefilter("ignore"); logging.disable(logging.CRITICAL)
    import numpy as np, pandas as pd
    import pprof_py.presentation as P
    rng = np.random.default_rng(4)
    pid = np.repeat(np.arange(30), rng.integers(20, 80, 30))
    x = rng.normal(size=pid.size)
    y = rng.binomial(1, 1 / (1 + np.exp(-(rng.normal(-1.0, 0.5, 30)[pid] + 0.5 * x))))
    d = pd.DataFrame({"y": y, "x1": x, "provider_id": pid})
    name = sys.argv[1]

    def fit(cls, **kw):
        m = cls(**kw)
        m.fit(d, y_var="y", x_vars=["x1"], provider_var="provider_id")
        return m
    if name == "provider_variation":
        from pprof_py import LogisticRandomEffectModel
        re = fit(LogisticRandomEffectModel, verbose=False)
        make = lambda: P.provider_variation(re)
    elif name in ("multi_measure", "measure_agreement"):
        from pprof_py import LogisticFixedEffectModel
        fe1, fe2 = fit(LogisticFixedEffectModel), LogisticFixedEffectModel()
        fe2.fit(d.assign(y=1 - d["y"]), y_var="y", x_vars=["x1"], provider_var="provider_id")
        col = P.ProfileCollection({"A": P.ProviderProfile.from_model(fe1, test_method="wald"),
                                   "B": P.ProviderProfile.from_model(fe2, test_method="wald")})
        make = (lambda: P.multi_measure(col)) if name == "multi_measure" else (lambda: P.measure_agreement(col))
    elif name == "reliability":
        from pprof_py.measures.iur import BootstrapIUR
        iur = BootstrapIUR(n_boot=20).fit(y.astype(float), np.full(y.size, y.mean()), pid)
        make = lambda: P.reliability(iur)
    else:
        from pprof_py import LogisticFixedEffectModel
        from pprof_py.inference import EmpiricalNull
        fe = fit(LogisticFixedEffectModel)
        prof = P.ProviderProfile.from_model(fe, test_method="poibin_exact", limits=True)
        make = {"funnel": lambda: P.funnel(prof), "caterpillar": lambda: P.caterpillar(prof),
                "forest": lambda: P.forest(fe), "data_quality": lambda: P.data_quality(prof),
                "observed_expected": lambda: P.observed_expected(prof),
                "null_calibration": lambda: P.null_calibration(fe, test_method="score",
                                                               null_model=EmpiricalNull.fitter())}[name]
    a = make()                                  # the first figure of this interpreter
    first = [a.to_bytes(fmt) for fmt in ("svg", "pdf", "png")]
    b = make()                                  # built after saves
    again = [b.to_bytes(fmt) for fmt in ("svg", "pdf", "png")]
    print("SAME" if first == again else "DIFFERENT")
''')
NAMES = ["funnel", "caterpillar", "forest", "data_quality", "observed_expected", "null_calibration", "reliability",
         "provider_variation", "multi_measure", "measure_agreement"]


@pytest.mark.parametrize("name", NAMES)
def test_figures_do_not_depend_on_earlier_saves(name):
    out = subprocess.run([sys.executable, "-c", SCRIPT, name], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "SAME", f"{name}: {out}"
