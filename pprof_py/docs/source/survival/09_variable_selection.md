# Chapter 9 — Automated Variable Selection

## 9.1 An older, more transparent alternative to penalization

Chapter 8 approached "which covariates belong in the model" through
penalization — every coefficient shrinks simultaneously, and some
happen to land on exactly zero. **Stepwise selection** approaches the
same question completely differently: it fits a whole series of
*ordinary, unpenalized* Cox models, one covariate addition or removal
at a time, and uses a simple scoring rule to decide whether each change
makes the model meaningfully better.

The appeal of this approach is transparency: at every step, you can see
exactly which variable was added or dropped and exactly why, and the
final model is an ordinary Cox fit with honest, standard confidence
intervals (Chapter 8's Section 8.6 caveat about penalized coefficients
never having proper standard errors simply doesn't apply here — a
stepwise-selected model's final coefficients come from an entirely
unpenalized refit). The trade-off: stepwise selection makes a sequence
of discrete, somewhat greedy decisions, and — a real, well-documented
statistical issue — treating the final model's p-values as if the
covariate set had been chosen in advance overstates how confident you
should really be, since the selection process itself already used the
data once to decide what to even test.

## 9.2 The three scoring rules

`CoxPHSelector` decides whether to add or remove a covariate using one
of three criteria:

- **AIC** (Akaike Information Criterion) rewards a better log-likelihood
  fit but penalizes each additional parameter by a fixed amount (2 per
  parameter) — a variable is worth adding if it improves the fit by
  more than that fixed cost.
- **BIC** (Bayesian Information Criterion) penalizes each additional
  parameter more heavily as the sample size grows
  ($\ln(n)$ per parameter instead of a flat 2), which makes it
  systematically pickier than AIC about adding variables, and typically
  lands on a smaller, more conservative final model.
- **P-value** thresholds (`p_enter`, `p_remove`) mimic the classic,
  purely hypothesis-testing-driven stepwise procedure: add a variable
  if its own p-value would be below `p_enter`; remove one already in
  the model if its p-value has since risen above `p_remove`.

`direction` controls the search strategy: `"forward"` starts from
nothing and adds variables one at a time; `"backward"` starts from
everything and removes them one at a time; `"both"` allows a variable
added earlier to be dropped later if a subsequent addition made it
redundant (genuinely useful, since forward selection alone can't undo
an earlier choice that later turned out to be a mistake).

## 9.3 Running it

> **Continues from Chapter 8:** the snippets below reuse `X_wide`, the wide covariate matrix built there, together with `cohort` from Chapter 0.

```python
from pprof_py import CoxPHSelector

selector = CoxPHSelector(direction="forward", criterion="aic").fit(
    X_wide, duration=cohort["time"], event=cohort["death"],
)

selector.selected_variables_
selector.selection_history_
selector.final_model_.summary()   # an ordinary, unpenalized CoxPH -- full summary() available
```

On the same ten-candidate cohort from Chapter 8:

```
selected: ['age', 'comorbidity_count', 'lab_b', 'diabetes']

   step action                                  variables          aic
0     0  start                                         []  4090.389913
1     1    add                                      [age]  3988.176934
2     2    add                   [age, comorbidity_count]  3977.696345
3     3    add            [age, comorbidity_count, lab_b]  3976.663960
4     4    add  [age, comorbidity_count, lab_b, diabetes]  3976.077701
5     5   stop  [age, comorbidity_count, lab_b, diabetes]  3976.077701
```

Notice `lab_b` — one of the two covariates built with genuinely zero
true effect — got selected here too, exactly the same variable that
tripped up LASSO's `lambda_min_` selection in Chapter 8. That's not a
coincidence, and it's worth sitting with rather than treating as a
tutorial embarrassment: with only 259 real events in this cohort, both
a fundamentally different selection strategy (penalization) and this
one (stepwise AIC) found the same piece of noise attractive enough to
keep. That agreement is itself informative — it's a sign that this
particular dataset's event count is genuinely too limited to reliably
separate ten candidate covariates, no matter which selection method you
reach for. More data (or fewer candidates, chosen on subject-matter
grounds before ever looking at their p-values) is the real fix, not
switching selection methods until one happens to give a cleaner-looking
answer.

Requiring a stronger threshold, and forcing a variable you already know
belongs in the model regardless of what the data says (a common,
reasonable choice for age or another well-established risk factor,
just as the source ESRD methodology treats certain adjustments as
mandatory rather than up for selection):

```python
selector_bic = CoxPHSelector(direction="backward", criterion="bic").fit(
    X_wide, duration=cohort["time"], event=cohort["death"], forced=["age"],
)
selector_bic.selected_variables_
```

```
['age', 'comorbidity_count']
```

BIC's heavier penalty for model complexity, combined with backward
elimination's different (and here, more conservative) path through the
same search space, lands on a noticeably sparser, cleaner model —
correctly excluding `lab_b` this time. There's a real lesson in seeing
both results side by side: **the criterion and search direction you
choose are not neutral technical details — they can materially change
which variables end up in your final model**, and that choice deserves
the same deliberate justification you'd give any other modeling
decision, not a default left on autopilot.

## 9.4 Selection versus penalization: which to reach for

|  | Stepwise selection | Penalization |
|---|---|---|
| Final coefficients | Ordinary, unpenalized — honest SEs and CIs | Biased toward zero by design — no valid SEs |
| Transparency | See every step's reasoning | A single tuned `lambda`, less step-by-step visibility |
| Handles many correlated candidates | Can behave erratically, keeping one and dropping near-duplicates somewhat arbitrarily | Elastic net specifically handles this better |
| Typical use | A moderate number of candidates, and you want a clean, reportable final model with proper inference | Many candidates, or prediction is the primary goal rather than individually interpretable, testable coefficients |

Many practitioners use both together, in the sequence this guide has
now covered: penalization (Chapter 8) to narrow a large candidate set
down to a manageable shortlist, followed by an ordinary or
stepwise-refined fit on just that shortlist to get final, honestly
reportable inference — exactly the "unpenalized refit of just the
selected variables" Chapter 8 pointed toward.

## 9.5 What's next

Every technique in this guide has now been introduced on its own.
Chapter 10 puts them together into a single, start-to-finish analysis
of the running cohort — from raw patient-year records to a final,
defensible facility profiling report — the way you'd actually structure
a real project rather than one isolated technique at a time.
