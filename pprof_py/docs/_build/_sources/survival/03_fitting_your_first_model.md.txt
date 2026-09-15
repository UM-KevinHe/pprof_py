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
model.summary()
```

```
                        coef  exp(coef)  se(coef)         z             p  lower_95%  upper_95%
age                   0.0447     1.0457    0.0021   21.286  1.780000e-100     0.0406     0.0489
sex                  -0.0523     0.9491    0.0631   -0.829  4.070000e-01    -0.1760     0.0714
diabetes              0.3691     1.4465    0.0625    5.906  3.510000e-09     0.2467     0.4916
comorbidity_count     0.2103     1.2341    0.0271    7.760  8.470000e-15     0.1572     0.2634
```

Every column, in plain language:

- **`coef`** — the estimated $\hat\beta$ from Section 2.3: the change
  in *log*-hazard per one-unit increase in the covariate.
- **`exp(coef)`** — the hazard ratio, Section 2.2's whole point:
  diabetes multiplies the hazard of death by about 1.45, i.e. a 45%
  higher instantaneous risk of death, holding age, sex, and
  comorbidity count fixed.
- **`se(coef)`** — the standard error of $\hat\beta$: how much this
  estimate would plausibly wobble if you repeated the study on a new
  sample from the same population. Smaller is more precise. (Chapter 5
  covers exactly when this "plain" standard error can be misleading and
  what to do about it.)
- **`z`** and **`p`** — the Wald test $z = \hat\beta / \text{se}(\hat\beta)$
  and its two-sided p-value: how many standard errors is this estimate
  away from zero, and how surprising would that be if the true effect
  were actually zero? A p-value near 0 (like diabetes and
  comorbidity count here) means "this effect is very unlikely to be
  pure noise"; sex's p-value of 0.41 means "we cannot distinguish this
  estimate from zero with the data we have" — which, since our
  synthetic cohort was built with no true sex effect, is exactly the
  right conclusion for the model to reach.
- **`lower_95%`/`upper_95%`** — the 95% confidence interval on `coef`
  (not `exp(coef)` — exponentiate the endpoints yourself if you want
  the hazard-ratio-scale interval). Widen or narrow it with
  `CoxPH(confidence_level=0.90)` or any level you need.

## 3.4 The rest of the fitted attributes

```python
model.log_likelihood_, model.log_likelihood_null_
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
```

A few more attributes worth knowing:

```python
model.n_obs_          # 4000 -- rows fit on
model.n_events_       # how many were actual deaths, not censored
model.converged_      # True if Newton-Raphson found a stationary point
model.n_iter_         # how many iterations it took (usually well under 10)
model.feature_names_in_   # the column names, in order, matching coef_
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
model.baseline_hazard_.head()
```

```
   stratum      time    hazard  survival
0        0  0.031000  0.000210  0.999790
1        0  0.052000  0.000431  0.999569
2        0  0.077000  0.000657  0.999343
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

model.predict_linear(new_patients)           # X @ coef_  -- the raw linear predictor
model.predict_partial_hazard(new_patients)   # exp(X @ coef_)  -- relative risk vs. baseline
model.predict_cumulative_hazard(new_patients)   # a full curve, one column per patient
model.predict_survival_function(new_patients)   # exp(-cumulative hazard) -- the survival curve
```

`predict_cumulative_hazard`/`predict_survival_function` return a
DataFrame indexed by time, one column per row of the input:

```
        0         1
time
0.031  0.00007  0.00089
0.052  0.00014  0.00183
...
```

Reading it: patient 0 (age 55, no diabetes, no comorbidities) has an
estimated 0.007% cumulative chance of death by time 0.031, versus 0.09%
for patient 1 (age 75, diabetic, 3 comorbidities) — already, at the
very first observed event time in the data, several times the risk,
compounding further as time goes on. Plot it directly:

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
model.martingale_residuals_[:5]
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
