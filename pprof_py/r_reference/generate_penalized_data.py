"""Generate the additional synthetic dataset Phase 3 needs: enough
covariates (several truly zero) for LASSO/ridge/elastic-net path and
cross-validation to be meaningful. Phase 1/2's existing datasets
(strata.csv, offset.csv, weights.csv, left_truncation.csv, basic.csv
[which already has heavy ties -- 27 unique times over 500 rows],
combined.csv [strata+offset+weights+start/stop together]) are reused
as-is for the "does penalization preserve strata/offset/weights/
start-stop/ties" checks -- no need to regenerate those, and reusing
them means Phase 3's capability-preservation tests run on the exact
same data Phase 1/2 already validated.

Run once; R and pytest both read the resulting CSVs.
"""
import numpy as np
import pandas as pd
import os

OUTDIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(OUTDIR, exist_ok=True)
ROUND_DP = 2


def clean_time(x):
    return np.round(x, ROUND_DP)


# ------------------------------------------------------------- penalized_wide
# 20 covariates, only 8 truly nonzero, mildly correlated in blocks (so
# LASSO has real selection work to do and ridge/elastic-net have
# something to shrink jointly), continuous-ish times (light rounding,
# so ties exist but are not the dominant feature -- basic.csv already
# covers the heavy-ties case).
rng = np.random.default_rng(201)
n, p = 400, 20
block = rng.normal(size=(n, 4))
noise = rng.normal(size=(n, p))
X = noise.copy()
# correlate a few columns with a shared latent block so the penalized
# fit has to trade off correlated predictors, not just pick among
# independent ones
X[:, 0] += 0.8 * block[:, 0]
X[:, 1] += 0.8 * block[:, 0]
X[:, 2] += 0.8 * block[:, 1]
X[:, 3] += 0.8 * block[:, 1]
beta_true = np.zeros(p)
beta_true[[0, 2, 4, 6, 8, 10, 12, 14]] = [0.9, -0.7, 0.5, -0.5, 0.35, -0.3, 0.25, -0.2]
risk = np.exp(X @ beta_true)
U = rng.uniform(size=n)
dur = -np.log(U) / risk
C = rng.exponential(scale=np.median(dur) * 1.7, size=n)
stop = np.round(np.minimum(dur, C), 1)
stop = np.maximum(stop, 0.1)
event = (dur <= C).astype(int)
df = pd.DataFrame(X, columns=[f"x{i+1}" for i in range(p)])
df["stop"] = stop
df["event"] = event
df.to_csv(f"{OUTDIR}/penalized_wide.csv", index=False)

# explicit fold assignment so R's cv.glmnet(foldid=...) and Python's
# PenalizedCoxPHCV(fold_id=...) can be compared on identical folds
rng_fold = np.random.default_rng(202)
n_folds = 8
fold_id = rng_fold.permutation(np.tile(np.arange(1, n_folds + 1), int(np.ceil(n / n_folds)))[:n])
pd.DataFrame({"fold_id": fold_id}).to_csv(f"{OUTDIR}/penalized_wide_foldid.csv", index=False)

print("Wrote penalized_wide.csv and penalized_wide_foldid.csv to", OUTDIR)
