"""Generate shared synthetic datasets for Phase 4 (competing risks,
robust/clustered variance, time-dependent covariates), on top of the
Phase 1-3 datasets already in r_reference/data/. Run once; the
run_phase4_*.R scripts and tests/test_phase4_r_comparison.py both read
the resulting CSVs.

Same time-rounding hygiene as generate_data.py (see that file's own
comment) applies here -- rounded once, at the end, after all arithmetic.
"""
import os

import numpy as np
import pandas as pd

OUTDIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTDIR, exist_ok=True)
ROUND_DP = 2


def clean_time(x):
    return np.round(x, ROUND_DP)


# --------------------------------------------------- competing risks (1)
# Simple (start=0) competing-risks data: cause 1 and cause 2 each with
# their own covariate effect, enough censoring to be realistic.
rng = np.random.default_rng(201)
n = 400
X = rng.normal(size=(n, 2))
beta1 = np.array([0.5, -0.3])
beta2 = np.array([-0.2, 0.4])
risk1 = np.exp(X @ beta1)
risk2 = np.exp(X @ beta2)
t1 = rng.exponential(1 / (0.35 * risk1))
t2 = rng.exponential(1 / (0.25 * risk2))
censor = rng.exponential(2.2, n)
stop = np.minimum(np.minimum(t1, t2), censor)
event = np.where(stop == t1, 1, np.where(stop == t2, 2, 0))
stop = clean_time(np.maximum(stop, 0.01))
df = pd.DataFrame(X, columns=["x1", "x2"])
df["id"] = np.arange(1, n + 1)
df["stop"] = stop
df["event"] = event
df.to_csv(f"{OUTDIR}/competing_risks_simple.csv", index=False)

# ---------------------------------------- competing risks (2), truncated
rng = np.random.default_rng(202)
n = 400
X = rng.normal(size=(n, 2))
beta1 = np.array([0.4, -0.35])
beta2 = np.array([-0.3, 0.2])
risk1 = np.exp(X @ beta1)
risk2 = np.exp(X @ beta2)
start = clean_time(rng.uniform(0, 0.4, n))
t1 = start + rng.exponential(1 / (0.35 * risk1))
t2 = start + rng.exponential(1 / (0.25 * risk2))
censor = start + rng.exponential(2.2, n)
stop = np.minimum(np.minimum(t1, t2), censor)
event = np.where(stop == t1, 1, np.where(stop == t2, 2, 0))
stop = clean_time(np.maximum(stop, start + 0.01))
df = pd.DataFrame(X, columns=["x1", "x2"])
df["id"] = np.arange(1, n + 1)
df["start"] = start
df["stop"] = stop
df["event"] = event
df.to_csv(f"{OUTDIR}/competing_risks_truncated.csv", index=False)

# ------------------------------------- robust/clustered, strata+truncation
# Several rows per cluster (a repeated-measures-style setup: `cluster` is
# the true subject, `id`/start/stop rows are visits within it -- enough
# clusters, and few enough rows per cluster, for the sandwich estimate to
# be well-behaved) to specifically exercise strata + left truncation +
# clustering together, which statsmodels.PHReg could not (returned NaN)
# -- see docs/R_COMPATIBILITY.md's robust-variance section.
rng = np.random.default_rng(203)
n_clusters = 120
rows_per_cluster = rng.integers(1, 4, n_clusters)
cluster_id = np.repeat(np.arange(n_clusters), rows_per_cluster)
n = cluster_id.shape[0]
X = rng.normal(size=(n, 2))
strata = rng.integers(0, 3, n)
beta_true = np.array([0.45, -0.3])
strata_effect = rng.normal(scale=0.4, size=3)[strata]
risk = np.exp(X @ beta_true + strata_effect)
start = clean_time(rng.uniform(0, 0.3, n))
dur = rng.exponential(1 / (0.4 * risk))
censor = rng.exponential(2.0, n)
stop = clean_time(np.maximum(start + np.minimum(dur, censor), start + 0.01))
event = (dur <= censor).astype(int)
df = pd.DataFrame(X, columns=["x1", "x2"])
df["id"] = np.arange(1, n + 1)
df["cluster"] = cluster_id
df["strata"] = strata
df["start"] = start
df["stop"] = stop
df["event"] = event
df.to_csv(f"{OUTDIR}/robust_strata_truncation.csv", index=False)

# ------------------------------------------------- time-dependent covariates
# A base skeleton (one row per subject, `stop` their overall follow-up
# time and `death` their event indicator) plus a SEPARATE stream of
# covariate-change records (a subset of subjects switch a covariate on
# at a random time before their own stop) -- exactly the tmerge() use
# case, and small enough to inspect by hand if a mismatch ever needs
# debugging.
rng = np.random.default_rng(204)
n = 250
base_risk = rng.normal(size=n) * 0  # unused placeholder to keep RNG draw count stable across edits
switch_time = clean_time(rng.exponential(1.5, n))
beta_true = -0.5
rate0 = 0.3
E = rng.exponential(1.0, n)
budget_at_switch = rate0 * switch_time
event_time = np.where(
    E <= budget_at_switch,
    E / rate0,
    switch_time + (E - budget_at_switch) / (rate0 * np.exp(beta_true)),
)
censor = rng.exponential(3.0, n)
stop = clean_time(np.maximum(np.minimum(event_time, censor), 0.01))
death = (event_time <= censor).astype(int)
switched = switch_time < stop

skeleton = pd.DataFrame({"id": np.arange(1, n + 1), "stop": stop, "death": death})
skeleton.to_csv(f"{OUTDIR}/timedep_skeleton.csv", index=False)

updates = pd.DataFrame({"id": np.arange(1, n + 1)[switched], "time": switch_time[switched]})
updates.to_csv(f"{OUTDIR}/timedep_updates.csv", index=False)

print("Wrote Phase 4 datasets to", OUTDIR)
for f in sorted(os.listdir(OUTDIR)):
    if f.startswith(("competing_risks", "robust_strata", "timedep")):
        print(" ", f)
