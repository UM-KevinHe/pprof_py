# Chapter 3 — Fitting Your First Model

This chapter is entirely hands-on. By the end of it you will have
fit a Cox model on the running cohort, read every number in its
output, and generated predictions from it.

## 3.1 Preparing the data

`CoxPH` wants three things at minimum: a covariate matrix, a time (or
start/stop pair), and an event indicator. A pandas DataFrame works
directly as the covariate matrix — column names are preserved and used
throughout the model's output, which is worth doing even though a
plain NumPy array works too.

```python
import numpy as np
import pandas as pd
from pprof_py import CoxPH

# cohort is the DataFrame built in Chapter 0

X = cohort[["age", "sex", "diabetes", "comorbidity_count"]]
```

That's it — no need to one-hot encode `sex` (already 0/1) or manually
add an intercept (Section 3.6 explains why a Cox model doesn't have
one at all).

## 3.2 Fitting the model

```python
model = CoxPH(ties="efron").fit(
    X,
    duration=cohort["time"],
    event=cohort["death"],
)
```

`duration=`/`event=` is the right pair of arguments for ordinary
right-censored data with no left truncation — everyone's `start` is
implicitly 0. The moment you need left truncation (Chapter 4 onward),
you'll switch to `start=`/`stop=` instead; `coxph` will tell you clearly
if you try to pass both.

That single `.fit()` call ran the entire partial-likelihood
maximization from Chapter 2: it swept through every observed death,
built each one's risk set, and used Newton-Raphson iterations to find
the $\hat\beta$ that makes the observed deaths look as likely as
possible relative to everyone else who could have died instead. Every
piece of that process is now available as an attribute on `model`.

## 3.3 Reading `summary()`

```python
print(model.summary().round(4).to_string())
```

```
                     coef  exp(coef)  se(coef)        z       p  lower_95%  upper_95%
age                0.0471     1.0482    0.0047  10.0650  0.0000     0.0379     0.0562
sex               -0.0577     0.9439    0.1253  -0.4608  0.6449    -0.3032     0.1878
diabetes           0.1989     1.2200    0.1244   1.5985  0.1099    -0.0450     0.4428
comorbidity_count  0.1823     1.1999    0.0505   3.6129  0.0003     0.0834     0.2812
```

Every column, in plain language:

- **`coef`** — the estimated $\hat\beta$ from Section 2.3: the change
  in *log*-hazard per one-unit increase in the covariate.
- **`exp(coef)`** — the hazard ratio, Section 2.2's whole point:
  each additional comorbidity multiplies the hazard of death by about
  1.20, i.e. a 20% higher instantaneous risk of death, holding age,
  sex, and diabetes fixed.
- **`se(coef)`** — the standard error of $\hat\beta$: how much this
  estimate would plausibly wobble if you repeated the study on a new
  sample from the same population. Smaller is more precise. (Chapter 5
  covers exactly when this "plain" standard error can be misleading and
  what to do about it.)
- **`z`** and **`p`** — the Wald test $z = \hat\beta / \text{se}(\hat\beta)$
  and its two-sided p-value: how many standard errors is this estimate
  away from zero, and how surprising would that be if the true effect
  were actually zero? A p-value near 0 (like age and comorbidity count
  here) means "this effect is very unlikely to be pure noise"; sex's
  p-value of 0.64 means "we cannot distinguish this estimate from zero
  with the data we have" — which, since our synthetic cohort was built
  with no true sex effect, is the right conclusion. Diabetes is the
  instructive case: the cohort was built with a real diabetes effect
  (0.35 on the log-hazard scale), yet with 259 deaths the estimate,
  0.20, has p = 0.11 and an interval that includes zero. Failing to
  reject is not evidence of no effect.
- **`lower_95%`/`upper_95%`** — the 95% confidence interval on `coef`
  (not `exp(coef)` — exponentiate the endpoints yourself if you want
  the hazard-ratio-scale interval). Widen or narrow it with
  `CoxPH(confidence_level=0.90)` or any level you need.

## 3.4 The rest of the fitted attributes

```python
print(round(model.log_likelihood_, 2), round(model.log_likelihood_null_, 2))
```
```
-1985.45 -2045.19
```

`log_likelihood_null_` is the partial log-likelihood of a model with
*no* covariates at all (every patient equally likely to be the one who
died at each risk set); `log_likelihood_` is the log-likelihood at the
fitted $\hat\beta$. Their difference, doubled, is a **likelihood ratio
test** of "do these covariates, together, explain anything at all" —
a single overall significance test for the whole model, complementary
to (and often more reliable than, in small samples) the individual
Wald tests in `summary()`:

```python
from scipy import stats

lr_statistic = 2 * (model.log_likelihood_ - model.log_likelihood_null_)
p_value = stats.chi2.sf(lr_statistic, df=len(model.coef_))
print(f"LR = {lr_statistic:.1f} on {len(model.coef_)} df, p = {p_value:.1e}")
```
```
LR = 119.5 on 4 df, p = 6.9e-25
```

A few more attributes worth knowing:

```python
print(model.n_obs_, model.n_events_)   # rows fit on; how many were deaths, not censored
print(model.converged_, model.n_iter_)  # Newton-Raphson reached a stationary point; iterations taken
print(model.feature_names_in_)          # the column names, in order, matching coef_
```
```
4000 259
True 4
['age' 'sex' 'diabetes' 'comorbidity_count']
```

Always glance at `converged_` before trusting anything else — an
unconverged fit (rare, but possible with severe collinearity or a
tiny number of events) means every other number on this list should be
treated with real suspicion.

## 3.5 The baseline hazard

Section 2.1 promised that $\lambda_0(t)$ is estimated with no
assumption on its shape. `coxph` estimates it using the **Breslow
estimator** for the cumulative baseline hazard:

```python
print(model.baseline_hazard_.head())
```

```
   stratum   time    hazard  survival
0        0  0.009  0.000008  0.999992
1        0  0.016  0.000016  0.999984
2        0  0.017  0.000023  0.999977
3        0  0.018  0.000031  0.999969
4        0  0.019  0.000039  0.999961
```

`hazard` here is the *cumulative* baseline hazard $\hat\Lambda_0(t)$ up
to and including that time (Section 1.4's relationship
$S(t)=\exp(-\Lambda(t))$ is exactly how the adjacent `survival` column
is computed). One subtlety worth knowing, because it trips people up
comparing against raw formulas: this baseline is reported at the
*mean* value of any offset in the model (matching R's own
`basehaz(centered=FALSE)` convention) — it is not automatically the
hazard at a literal "everything is zero" patient if the model has
offsets. `predict_cumulative_hazard`, next, sidesteps this entirely by
computing predictions directly rather than reading this convenience
attribute.

## 3.6 Why there's no intercept

You may have noticed `CoxPH` never estimates an intercept term
(`fit_intercept=False` is the default, and normally should stay that
way). This isn't an oversight — it's a mathematical fact about the
model. Look back at the partial likelihood in Section 2.3: an intercept
would add a constant $\beta_0$ to every single patient's linear
predictor, in every risk set, with no exceptions. Since the partial
likelihood only ever compares patients *within the same risk set* via
a ratio, a constant added to everyone's numerator and every term of the
denominator cancels out identically, exactly the way $\lambda_0(t)$
itself cancelled out in Section 2.3. An intercept term literally cannot
be estimated by this method — it is absorbed entirely into the shape of
$\lambda_0(t)$, which is left deliberately free to be anything, and
that's exactly where a constant risk shift belongs.

## 3.7 Predictions

Four related prediction methods, each a small step past the last:

```python
new_patients = pd.DataFrame({
    "age": [55, 75], "sex": [0, 1], "diabetes": [0, 1], "comorbidity_count": [0, 3],
})

print(model.predict_linear(new_patients).round(3))           # X @ coef_ -- the raw linear predictor
print(model.predict_partial_hazard(new_patients).round(3))   # exp(X @ coef_) -- relative risk vs. baseline
cumulative = model.predict_cumulative_hazard(new_patients)   # a full curve, one column per patient
survival = model.predict_survival_function(new_patients)     # exp(-cumulative hazard) -- the survival curve
print(cumulative.head(2))
```
```
[2.589 4.218]
[13.31  67.885]
              0         1
time
0.009  0.000104  0.000529
0.016  0.000208  0.001058
```

`predict_cumulative_hazard`/`predict_survival_function` return a
DataFrame indexed by time, one column per row of the input. Reading it:
patient 0 (age 55, no diabetes, no comorbidities) has an estimated
cumulative hazard of 0.0001 (a 0.01% chance of death) by time 0.009,
the first death time in the data, versus 0.0005 for patient 1 (age 75,
diabetic, 3 comorbidities) — about five times the risk, the ratio of
their partial hazards, at every time. Plot it directly:

```python
import matplotlib.pyplot as plt

survival_curves = model.predict_survival_function(new_patients)
survival_curves.plot()
plt.xlabel("Years")
plt.ylabel("Estimated survival probability")
```

## 3.8 A first diagnostic: martingale residuals

Every fitted model comes with **martingale residuals** already
computed:

```python
print(model.martingale_residuals_[:5].round(4))
```
```
[-0.085  -0.0131 -0.0409 -0.0562 -0.0396]
```

Conceptually, a martingale residual answers: *for this specific
patient, how many events did we actually observe (0 or 1) compared to
how many the model expected, given their covariates and how long they
were followed?*

$$
M_i = \delta_i - \hat\Lambda_i(T_i)
$$

A patient who died ($\delta_i=1$) but had a very *low* expected
cumulative hazard $\hat\Lambda_i(T_i)$ gets a residual close to $+1$ —
the model was surprised by their death. A patient who survived a very
long time despite a high expected hazard gets a strongly *negative*
residual — the model expected them to have died already. Residuals
clustered strongly in one direction for a subgroup the model doesn't
already account for (say, one particular facility) is a signal that
something about that subgroup isn't captured by the current covariates
— exactly the kind of check worth running before trusting any
facility-level comparison in the next chapter.

## 3.9 What's next

You can now fit a plain Cox model, interpret every number it produces,
and generate predictions from it. Chapter 4 takes this exact machinery
— stratification from Section 2.5, offsets from Section 2.6 — and
assembles it into the two-stage Standardized Mortality Ratio method:
the reason this package exists in the first place.
