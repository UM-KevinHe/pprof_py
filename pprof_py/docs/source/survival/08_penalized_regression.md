# Chapter 8 — Penalized Regression: Ridge, LASSO, and Elastic Net

## 8.1 The problem more covariates create

Every model so far has used a handful of carefully chosen covariates.
Real data often tempts you with dozens: every comorbidity flag, every
lab value, every interaction term the source methodology in Chapter 4
lists (age×race, ethnicity×race, diabetes×vintage, and more). Throwing
all of them into an ordinary Cox model creates a specific, predictable
failure mode as the number of covariates grows relative to the number
of *events* (not rows — events, since those are what actually drive
the partial likelihood in Section 2.3): coefficients become large,
unstable, and wildly sensitive to which patients happened to die first
in your particular sample. A model like this can fit your training
data almost perfectly and still generalize terribly to new patients,
because much of what it "learned" was sampling noise dressed up as
signal.

**Penalization** fixes this by adding a cost for coefficient size
directly into what the model is trying to maximize. Instead of finding
the $\beta$ that purely maximizes the partial likelihood from Section
2.3, penalized regression finds the $\beta$ that maximizes

$$
\ell(\beta) - \lambda \left[ (1-\alpha)\tfrac{1}{2}\|\beta\|_2^2 + \alpha\|\beta\|_1 \right]
$$

— the ordinary log partial likelihood $\ell(\beta)$, minus a penalty
that grows with the size of the coefficients. $\lambda \ge 0$ controls
*how much* penalty to apply overall (0 recovers the ordinary,
unpenalized Cox model exactly); $\alpha \in [0,1]$ controls the *shape*
of the penalty, and is the whole distinction between ridge, LASSO, and
elastic net.

## 8.2 Ridge, LASSO, and elastic net: what the shape buys you

- **Ridge** ($\alpha=0$, the $\|\beta\|_2^2$ term alone) shrinks every
  coefficient toward zero, proportionally, but essentially never sets
  one to *exactly* zero. It's the right tool when you believe most of
  your covariates carry at least a little real signal, and you mainly
  want to tame instability from having too many of them relative to
  your event count.
- **LASSO** ($\alpha=1$, the $\|\beta\|_1$ term alone) can shrink a
  coefficient all the way to exactly zero, which makes it a form of
  **automatic variable selection**: the covariates that survive with a
  nonzero coefficient are, in effect, the ones the model chose to keep.
  It's the right tool when you suspect many of your candidate
  covariates are genuinely irrelevant and you want the model to say so.
- **Elastic net** (anything strictly between) blends the two — LASSO's
  selection behavior, softened by a bit of ridge's stability, which
  matters most when several candidate covariates are highly correlated
  with each other (LASSO alone tends to arbitrarily pick just one of a
  correlated group and zero out the rest, which can be a less
  reproducible, less interpretable outcome than letting them share the
  credit).

`coxph` calls this mixing parameter `alpha` — matching R's `glmnet`
convention exactly, which is worth flagging since "alpha" means the
mixing weight here, not the regularization strength (that's `lambda`,
next).

## 8.3 The regularization path

For a fixed `alpha`, `PenalizedCoxPH` doesn't fit just one model — it
fits an entire sequence of them, one for each of a decreasing ladder of
`lambda` values, from a $\lambda$ large enough to zero out *every*
coefficient down to one small enough to be nearly unpenalized:

```python
from pprof_py import PenalizedCoxPH

lasso_path = PenalizedCoxPH(alpha=1.0).fit(X, duration=time, event=death)

lasso_path.lambda_path_        # the 100 lambda values tried, largest first
lasso_path.coef_path_          # shape (100, n_features) -- one row of coefficients per lambda
lasso_path.n_nonzero_path_     # how many coefficients are nonzero at each lambda
```

Plotting `coef_path_` against `lambda_path_` (on a log scale) produces
the classic "coefficient path" plot: every coefficient starts at zero
on the left (maximum penalty) and fans out toward its unpenalized value
on the right, with LASSO's individual lines visibly *entering* the plot
at different lambda values as they become nonzero — a direct visual
read of "which covariates does the model consider most important,"
since more important covariates survive penalization down to a larger
$\lambda$.

## 8.4 Picking a lambda: cross-validation

The path alone doesn't tell you *which* lambda to actually use for
prediction or reporting — that choice needs external evidence about
which one generalizes best, which is exactly what k-fold
cross-validation provides: hold out a fold, fit on the rest across the
whole lambda path, measure how well each lambda predicts the held-out
fold, and repeat across all folds.

```python
from pprof_py import PenalizedCoxPHCV

cv = PenalizedCoxPHCV(alpha=1.0, n_folds=10, random_state=0).fit(
    X, duration=time, event=death,
)

cv.lambda_min_    # the lambda with the best average cross-validated performance
cv.lambda_1se_    # the largest lambda within one standard error of that best performance
cv.coef_          # coefficients at lambda_min_ -- ready to use directly
```

`lambda_min_` and `lambda_1se_` embody a real judgment call, not just
two numbers to pick between arbitrarily: `lambda_min_` gives the best
*point estimate* of predictive performance, but that estimate is itself
noisy (it came from a finite number of folds); `lambda_1se_` — a
slightly larger, more aggressively penalized lambda, still within one
standard error of the best observed performance — is the standard,
deliberately more conservative choice when you want a **simpler**
model that's statistically indistinguishable from the best one found,
favoring interpretability and stability over chasing the single best
cross-validated point estimate. Use `select="lambda_1se"` in the
constructor to make that the default `.coef_` instead:

```python
cv_parsimonious = PenalizedCoxPHCV(alpha=1.0, select="lambda_1se", random_state=0).fit(
    X, duration=time, event=death,
)
```

`cv.final_estimator_` gives you the fully-fitted `PenalizedCoxPH`-style
object at the selected lambda, if you need more than just `coef_` —
the fitted deviance, the number of nonzero coefficients, and so on.

## 8.5 A worked comparison on the running cohort

Suppose you're not sure which of ten candidate covariates — the five
you've used throughout this guide, plus five weaker, more speculative
ones — actually belong in a facility profiling model:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(4)
cohort["lab_a"] = rng.normal(0, 1, len(cohort))       # pure noise, no true effect
cohort["lab_b"] = rng.normal(0, 1, len(cohort))       # pure noise, no true effect
cohort["nursing_home"] = rng.binomial(1, 0.1, len(cohort))
cohort["bmi_low"] = rng.binomial(1, 0.15, len(cohort))
cohort["albumin_low"] = rng.binomial(1, 0.2, len(cohort))

candidates = ["age", "sex", "diabetes", "comorbidity_count", "vintage_years",
              "lab_a", "lab_b", "nursing_home", "bmi_low", "albumin_low"]
X_wide = cohort[candidates]

cv = PenalizedCoxPHCV(alpha=1.0, n_folds=10, random_state=0).fit(
    X_wide, duration=cohort["time"], event=cohort["death"],
)
pd.Series(cv.coef_, index=candidates).round(4)
```

```
age                  0.0452
sex                 -0.0169
diabetes             0.1565
comorbidity_count    0.1646
vintage_years       -0.0292
lab_a                0.0000
lab_b                0.0826
nursing_home        -0.1864
bmi_low             -0.1691
albumin_low          0.1374
```

`lab_a` — one of the two variables built with genuinely zero effect —
was correctly zeroed out. `lab_b`, the other pure-noise variable, was
not: with only 259 deaths in a cohort of 4,000 (Chapter 4's own point
about reliability applies here too — small event counts mean noisy
cross-validated estimates), the penalty chosen by `lambda_min_` wasn't
quite aggressive enough to suppress every irrelevant covariate. This is
a genuinely useful, honest thing to see in a tutorial: LASSO's variable
selection is a statistical procedure with its own sampling variability,
not a deterministic oracle — it will occasionally keep a coefficient
that isn't real, and occasionally shrink one that is (`vintage_years`
here ends up smaller than the unpenalized fit from Chapter 4 would
suggest). Switching to the more conservative `select="lambda_1se"` on
this same data pushes the penalty hard enough to zero out *every*
covariate except age — illustrating the real tension Section 8.4
described: `lambda_1se_` buys simplicity and stability at a real risk
of discarding genuine, if modest, signal. Neither choice is "wrong";
they trade off differently, and knowing which one you got is exactly
why `coxph` exposes both `lambda_min_` and `lambda_1se_` rather than
picking silently on your behalf.

## 8.6 A caution worth stating plainly

Penalized coefficients are biased *by design* — shrinking toward zero
is the entire mechanism that stabilizes the model, and it comes at the
cost of point estimates that are no longer centered on the truth in
the way an ordinary, unpenalized Cox coefficient is. Because of this,
`coxph` does not report standard errors, p-values, or confidence
intervals for penalized coefficients, and neither does `glmnet` — the
sampling distribution of a penalized estimate is not the same one
`summary()`'s ordinary Wald machinery from Chapter 3 assumes, and
reporting a p-value as though it were would be actively misleading. Use
penalization for prediction and for principled variable selection; use
an ordinary, unpenalized refit of just the selected variables (Chapter
9's territory) when you need a hazard ratio with an honest confidence
interval to report.

## 8.7 What's next

Chapter 9 covers a complementary, older approach to the same "which
covariates belong" question — stepwise selection by AIC, BIC, or
p-value — including exactly the refit-after-selection pattern the
previous paragraph pointed to.
