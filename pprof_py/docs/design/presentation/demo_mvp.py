"""MVP demo: funnel, interval plot and provider table at 20, 1,000 and 10,000 providers (deterministic).

Usage: python demo_mvp.py [OUTPUT_DIR]

Simulated logistic data (log-normal provider sizes, a few planted outliers in both directions, natural zero-event
providers). Funnels use the score test (the logistic fixed-effect default); interval plots and tables use the exact
Poisson-binomial test up to 1,000 providers and the Wald test at 10,000, whose exact intervals would take about a
minute to compute.
"""
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

from pprof_py import LogisticFixedEffectModel
from pprof_py.presentation import ProviderProfile, caterpillar, funnel, provider_table

warnings.simplefilter("ignore", RuntimeWarning)


def simulate(n_providers, seed=2026):
    rng = np.random.default_rng(seed)
    size = np.maximum(12, rng.lognormal(np.log(60), 0.7, n_providers).astype(int))
    gamma = rng.normal(-1.5, 0.35, n_providers)
    k = max(1, n_providers // 30)
    gamma[:k] += 0.8
    gamma[k:2 * k] -= 0.8
    pid = np.repeat(np.arange(n_providers), size)
    x = rng.normal(size=(pid.size, 2))
    y = rng.binomial(1, 1 / (1 + np.exp(-(gamma[pid] + x @ [0.5, -0.3]))))
    return pd.DataFrame({"y": y, "x1": x[:, 0], "x2": x[:, 1], "provider": [f"P{j:05d}" for j in pid]})


def main(out):
    os.makedirs(out, exist_ok=True)
    for n in (20, 1000, 10000):
        t0 = time.perf_counter()
        model = LogisticFixedEffectModel()
        model.fit(simulate(n), y_var="y", x_vars=["x1", "x2"], provider_var="provider")
        fit = time.perf_counter() - t0
        t0 = time.perf_counter()
        fig = funnel(model)
        fig.save(f"{out}/funnel_N{n}.png")
        fig.save(f"{out}/funnel_N{n}.svg")
        test_method = "poibin_exact" if n <= 1000 else "wald"
        profile = ProviderProfile.from_model(model, test_method=test_method)
        caterpillar(profile).save(f"{out}/intervals_N{n}.png")
        provider_table(profile).save_html(f"{out}/provider_table_N{n}.html")
        print(f"{n:6d} providers: fit {fit:5.1f} s, figures and table {time.perf_counter() - t0:5.1f} s "
              f"({test_method} intervals); {fig.alt_text}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "demo_output")
