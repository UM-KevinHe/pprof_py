"""Generate shared synthetic datasets used by both the R reference scripts
and the Python validation tests, so both languages fit on byte-identical
inputs. Run once; R and pytest both read the resulting CSVs.
"""
import numpy as np
import pandas as pd
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTDIR, exist_ok=True)

# IMPORTANT: any time value that is *computed* (e.g. start + duration)
# must be rounded again AFTER the arithmetic, not just before it -- adding
# two already-rounded decimals in binary floating point (e.g. 0.1 + 0.5)
# does not generally reproduce the clean decimal 0.6 bit-for-bit, so two
# observations "meant" to tie at the same instant can silently end up as
# 0.6 and 0.6000000000000001. Since R and Python/pandas can each collapse
# or not collapse such near-duplicates slightly differently when finding
# unique event times, this is a real source of spurious mismatches in
# validation that has nothing to do with the CoxPH implementation itself
# -- it is purely a test-data hygiene issue, fixed once here by always
# rounding the *final* time column.
ROUND_DP = 2


def clean_time(x):
    return np.round(x, ROUND_DP)


def censored_survival(n, beta_true, offset, rng, tie_round=1):
    p = len(beta_true)
    X = rng.normal(size=(n, p))
    risk = np.exp(X @ beta_true + offset)
    U = rng.uniform(size=n)
    T = -np.log(U) / risk
    C = rng.exponential(scale=np.median(T) * 1.6, size=n)
    obs = np.round(np.minimum(T, C), tie_round)
    obs = np.maximum(obs, 10 ** (-tie_round))
    event = (T <= C).astype(int)
    return X, obs, event


# ---------------------------------------------------------------- basic
rng = np.random.default_rng(101)
X, stop, event = censored_survival(500, np.array([0.7, -0.4, 0.25]), 0.0, rng)
df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
df["time"] = stop
df["event"] = event
df.to_csv(f"{OUTDIR}/basic.csv", index=False)

# ---------------------------------------------------------- left truncation
rng = np.random.default_rng(102)
p = 2
X = rng.normal(size=(400, p))
beta_true = np.array([0.6, -0.3])
risk = np.exp(X @ beta_true)
U = rng.uniform(size=400)
dur = -np.log(U) / risk
start = np.round(rng.uniform(0, dur.mean() * 0.6, size=400), 1)
C = rng.exponential(scale=np.median(dur) * 1.6, size=400)
dur_obs = np.round(np.minimum(dur, C), 1)
dur_obs = np.maximum(dur_obs, 0.1)
stop = clean_time(start + dur_obs)
event = (dur <= C).astype(int)
df = pd.DataFrame(X, columns=["x1", "x2"])
df["start"] = start
df["stop"] = stop
df["event"] = event
df.to_csv(f"{OUTDIR}/left_truncation.csv", index=False)

# ------------------------------------------------------------------ strata
rng = np.random.default_rng(103)
n = 600
p = 2
X = rng.normal(size=(n, p))
beta_true = np.array([0.5, 0.3])
strata = rng.integers(0, 5, size=n)  # 5 unbalanced "providers"
strata_effect = np.array([0.0, 0.4, -0.3, 0.8, -0.5])[strata]  # baseline hazard differs a lot by stratum
risk = np.exp(X @ beta_true + strata_effect)
U = rng.uniform(size=n)
dur = -np.log(U) / risk
C = rng.exponential(scale=np.median(dur) * 1.6, size=n)
stop = np.round(np.minimum(dur, C), 1)
stop = np.maximum(stop, 0.1)
event = (dur <= C).astype(int)
df = pd.DataFrame(X, columns=["x1", "x2"])
df["stop"] = stop
df["event"] = event
df["provider"] = strata
df.to_csv(f"{OUTDIR}/strata.csv", index=False)

# ------------------------------------------------------------------ offset
rng = np.random.default_rng(104)
n = 400
p = 2
X = rng.normal(size=(n, p))
beta_true = np.array([0.5, -0.6])
offset = rng.normal(scale=0.5, size=n)
risk = np.exp(X @ beta_true + offset)
U = rng.uniform(size=n)
dur = -np.log(U) / risk
C = rng.exponential(scale=np.median(dur) * 1.6, size=n)
stop = np.round(np.minimum(dur, C), 1)
stop = np.maximum(stop, 0.1)
event = (dur <= C).astype(int)
df = pd.DataFrame(X, columns=["x1", "x2"])
df["stop"] = stop
df["event"] = event
df["log_exposure"] = offset
df.to_csv(f"{OUTDIR}/offset.csv", index=False)

# ------------------------------------------------------------------ weights
rng = np.random.default_rng(105)
n = 400
p = 2
X = rng.normal(size=(n, p))
beta_true = np.array([0.4, 0.5])
risk = np.exp(X @ beta_true)
U = rng.uniform(size=n)
dur = -np.log(U) / risk
C = rng.exponential(scale=np.median(dur) * 1.6, size=n)
stop = np.round(np.minimum(dur, C), 1)
stop = np.maximum(stop, 0.1)
event = (dur <= C).astype(int)
weight = np.round(rng.uniform(0.3, 3.0, size=n), 2)
df = pd.DataFrame(X, columns=["x1", "x2"])
df["stop"] = stop
df["event"] = event
df["weight"] = weight
df.to_csv(f"{OUTDIR}/weights.csv", index=False)

# --------------------------------------------------------- combined (SHR-like)
# Mirrors the real workflow: (start, stop] + strata(provider) + offset + weights, together.
rng = np.random.default_rng(106)
n = 900
p = 3
X = rng.normal(size=(n, p))
beta_true = np.array([0.35, -0.25, 0.5])
strata = rng.integers(0, 6, size=n)
strata_effect = rng.normal(scale=0.6, size=6)[strata]
offset = rng.normal(scale=0.3, size=n)
weight = np.round(rng.uniform(0.5, 2.5, size=n), 2)
risk = np.exp(X @ beta_true + strata_effect + offset)
U = rng.uniform(size=n)
dur = -np.log(U) / risk
start = np.round(rng.uniform(0, dur.mean() * 0.5, size=n), 1)
C = rng.exponential(scale=np.median(dur) * 1.6, size=n)
dur_obs = np.round(np.minimum(dur, C), 1)
dur_obs = np.maximum(dur_obs, 0.1)
stop = clean_time(start + dur_obs)
event = (dur <= C).astype(int)
df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
df["start"] = start
df["stop"] = stop
df["event"] = event
df["provider"] = strata
df["offset1"] = offset
df["weight"] = weight
df.to_csv(f"{OUTDIR}/combined.csv", index=False)

print("Wrote datasets to", OUTDIR)
for f in sorted(os.listdir(OUTDIR)):
    print(" ", f)

# ------------------------------------------------------- SMR two-stage
# Mirrors Section 34's SMR pattern specifically (distinct from the SHR
# pattern in `combined` above): stage 1 has strata but NO offset/weights;
# stage 2 has its OWN covariates (covars2) in addition to the offset, and
# no weights anywhere. covars1 and covars2 are deliberately different
# columns, since that's the realistic case (stage 1 adjusts for one set
# of covariates, stage 2 tests a different one against the stage-1 risk
# score as an offset).
rng = np.random.default_rng(107)
n = 700
X1 = rng.normal(size=(n, 2))          # covars1: x1, x2
Z2 = rng.normal(size=(n, 1))          # covars2: z1
beta1_true = np.array([0.4, -0.3])
beta2_true = np.array([0.55])
strata = rng.integers(0, 5, size=n)
strata_effect = rng.normal(scale=0.5, size=5)[strata]
risk = np.exp(X1 @ beta1_true + Z2 @ beta2_true + strata_effect)
U = rng.uniform(size=n)
dur = -np.log(U) / risk
start = np.round(rng.uniform(0, dur.mean() * 0.4, size=n), 1)
C = rng.exponential(scale=np.median(dur) * 1.6, size=n)
dur_obs = np.maximum(np.round(np.minimum(dur, C), 1), 0.1)
stop = clean_time(start + dur_obs)
event = (dur <= C).astype(int)
df = pd.DataFrame(X1, columns=["x1", "x2"])
df["z1"] = Z2[:, 0]
df["start"] = start
df["stop"] = stop
df["event"] = event
df["provider"] = strata
df.to_csv(f"{OUTDIR}/smr_two_stage.csv", index=False)

print("Wrote smr_two_stage.csv")
