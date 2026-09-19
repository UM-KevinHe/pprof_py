(standardization_methods)=
# Direct vs. Indirect Standardization Methods

Standardization methods are crucial in statistical modeling, especially
in fields like healthcare and social sciences. They enable fair
comparisons between groups (e.g., providers, regions) by adjusting for
differences in their underlying populations or case-mix. This document
explores two common approaches: indirect and direct standardization,
detailing their application in linear, logistic, and survival models.

```{contents} Table of Contents
:local:
:depth: 2
```

## 1. Introduction to Standardization

When comparing outcomes across different groups, raw (unadjusted)
figures can be misleading. Groups often differ systematically in
characteristics that influence outcomes (e.g., patient severity,
demographics). Standardization aims to provide a more equitable
comparison by accounting for these differences.

- **Linear Models**: Standardization often involves comparing adjusted
  mean differences in the outcome variable (e.g., difference in length
  of stay, cost).
- **Logistic Models**: For binary outcomes, standardization typically
  focuses on comparing adjusted rates or ratios (e.g., standardized
  mortality ratio (SMR), standardized infection ratio (SIR), or
  standardized rates).
- **Survival Models**: For time-to-event data, indirect
  standardization compares observed events to expected events derived
  from a fitted Cox model, yielding the **Standardized Mortality
  Ratio** (SMR) or **Standardized Hospitalization Ratio** (SHR).

This document focuses on model-based standardization, particularly
within the context of mixed models where outcomes are influenced by
both fixed (common) and random (group-specific) effects.

## 2. Indirect Standardization

### 2.1. Core Concept and Purpose

Indirect standardization evaluates how a group's outcome (often based
on model predictions) deviates from an **expected outcome**. This
expected outcome is calculated using a **baseline model** (typically
representing common, observable characteristics, like fixed effects)
applied to the specific case-mix of the group in question.

- **Focus**: To assess how much a group's performance differs from
  what would be expected if it performed according to a common
  baseline, given its specific population characteristics.
- **Purpose**:
  - To isolate and quantify the impact of group-specific factors
    (e.g., random effects or unmodeled characteristics) beyond the
    baseline predictions.
  - To identify groups performing significantly differently than
    expected under a common, fixed structure (akin to calculating SMRs
    or SIRs).

### 2.2. Process

The process generally involves comparing "observed" (full model) values
to "expected" (baseline model) values for the group.

**Step 1: Define "Observed" Values (Model-Based)**

For model-based standardization, the "observed" values for a group are
typically derived from the **full model predictions** (including both
fixed and random effects) for individuals within that group. Using
model-fitted values (e.g., $\hat{y}_i$ or $\hat{p}_i$ from the full
model) rather than raw actual outcomes helps to:

- Focus on modeled effects, keeping the analysis within the model's
  framework.
- Reduce the influence of random noise in the raw data not captured by
  the model.
- Ensure consistency in comparisons relative to the model's structure.

**Step 2: Define "Expected" Values**

The "expected" values are predictions from a **baseline model** applied
to the individuals within the group. This baseline often consists of
only the fixed effects component of a mixed model.

- **Linear Models**: Expected sum for group $k$ is
  $\sum_{i \in k} (X_i\hat{\boldsymbol\beta})$.
- **Logistic Models**: Expected sum of probabilities for group $k$ is
  $\sum_{i \in k} \hat{p}_i(\text{fixed only})$.
- **Survival Models**: Expected events for facility $k$ is
  $E_k = \sum_{i \in k} \hat{\Lambda}_0(t_i) \exp(X_i\hat{\boldsymbol\beta})$,
  where $\hat{\Lambda}_0$ is the national baseline cumulative hazard
  evaluated at each patient's follow-up time (see
  [Chapter 4](survival/04_indirect_standardization_smr_shr)).

**Step 3: Calculate the Standardized Measure**

- **Linear Models (Indirect Standardized Difference — ISD)**:

  The ISD quantifies the average difference per unit in the group
  attributable to factors beyond the baseline.

  $$
  \text{ISD}_k = \frac{\sum_{i \in k} \hat{y}_i(\text{full model})
  - \sum_{i \in k} (X_i\hat{\boldsymbol\beta})}{n_k}
  $$

- **Logistic Models (Indirect Standardized Ratio — ISR)**:

  The ISR compares the sum of model-predicted probabilities (full
  model) to the sum of expected probabilities (baseline model).

  $$
  \text{ISR}_k = \frac{\sum_{i \in k} \hat{p}_i(\text{full model})}
  {\sum_{i \in k} \hat{p}_i(\text{fixed only})}
  $$

- **Survival Models (Standardized Mortality/Hospitalization Ratio)**:

  $$
  \text{SMR}_k = \frac{O_k}{E_k}
  $$

  where $O_k$ is the observed event count at facility $k$ and $E_k$
  is the expected count under the national model. An SMR > 1 indicates
  more events than expected; SMR < 1 indicates fewer.

### 2.3. When to Use Indirect Standardization

- When the primary interest is in the incremental impact of
  group-specific effects (e.g., random effects) over and above
  predictions from a common, fixed-effects structure.
- To assess how a group's outcomes deviate from an expectation based on
  general population rates or a baseline model, applied to that group's
  specific demographic or case-mix.
- **Key Question Answered**: "How different is this group's outcome
  from what we'd expect based on common factors, given its specific
  case-mix?"

## 3. Direct Standardization

### 3.1. Core Concept and Purpose

Direct standardization evaluates a group's specific modeled effect
(e.g., its fixed effects plus its unique random effect) by applying
this effect to a **standard population**. The outcome is then compared
to the outcome derived from applying a **reference effect** (e.g., an
average or null random effect) to the same standard population.

- **Focus**: To assess the impact of a group's specific performance
  characteristic if it were applied universally to a standard
  population, allowing for comparison against a common benchmark.
- **Purpose**:
  - To compare the magnitude of different groups' specific effects on
    a common footing.
  - To understand the overall impact of a group's estimated random
    effect in a broader, standardized context.

### 3.2. Process

The process involves applying different effects to a common standard
population and comparing the results.

**Step 1: Define the Standard Population**

This is a common reference population to which different effects will be
applied. It could be the entire dataset, a subset, or an external
population. Let $N_{\text{std pop}}$ be its size.

**Step 2: Calculate Expected Outcomes under the Group's Specific Effect**

Apply the full modeled effect of group $k$ (fixed effects +
$\hat{\gamma}_k$, its specific random effect) to each individual $j$ in
the *standard population*.

- **Linear Models**: Sum of predictions is
  $\sum_{j \in \text{std pop}} (X_j\hat{\boldsymbol\beta} + \hat{\gamma}_k)$.
- **Logistic Models**: Sum of probabilities is
  $\sum_{j \in \text{std pop}} \hat{p}_j(\text{fixed} + \text{random effect}_k)$.

**Step 3: Calculate Expected Outcomes under a Reference Effect**

Apply a reference effect (e.g., a null random effect,
$\text{RE}_{\text{ref}}=0$, or the mean/median of estimated random
effects) along with fixed effects to each individual $j$ in the
*standard population*.

- **Linear Models**: Sum of predictions is
  $\sum_{j \in \text{std pop}} (X_j\hat{\boldsymbol\beta} + \text{RE}_{\text{ref}})$.
- **Logistic Models**: Sum of probabilities is
  $\sum_{j \in \text{std pop}} \hat{p}_j(\text{fixed} + \text{RE}_{\text{ref}})$.

**Step 4: Calculate the Standardized Measure**

- **Linear Models (Direct Standardized Difference — DSD)**:

  $$
  \text{DSD}_k = \frac{
    \sum_{j \in \text{std pop}} (X_j\hat{\boldsymbol\beta} + \hat{\gamma}_k)
    - \sum_{j \in \text{std pop}} (X_j\hat{\boldsymbol\beta} + \text{RE}_{\text{ref}})
  }{N_{\text{std pop}}}
  $$

- **Logistic Models (Direct Standardized Ratio — DSR)**:

  $$
  \text{DSR}_k = \frac{
    \sum_{j \in \text{std pop}} \hat{p}_j(\text{fixed} + \text{random effect}_k)
  }{
    \sum_{j \in \text{std pop}} \hat{p}_j(\text{fixed} + \text{RE}_{\text{ref}})
  }
  $$

### 3.3. When to Use Direct Standardization

- When you need to compare the magnitude of different groups' specific
  effects as if they were applied to a common, standard population.
- To understand how much each group's specific modeled effect deviates
  from an overall average or baseline effect when projected onto a
  common scale.
- **Key Question Answered**: "How different is this group's specific
  estimated effect compared to an average or null effect, if applied
  consistently to a standard population?"

## 4. Comparing Indirect and Direct Standardization

### 4.1. Key Distinctions

- **Reference Point**:
  - **Indirect**: Compares the group's outcome (within its *own
    case-mix*) to an expected outcome based on a standard *model or
    rate structure*.
  - **Direct**: Compares the outcome from applying the group's
    *specific effect* to a *standard population* against the outcome
    from applying a reference effect to that same standard population.

- **Primary Question**:
  - **Indirect**: "How does this group's actual (or model-fitted)
    performance compare to what's expected for a group with its
    specific characteristics, under a baseline scenario?"
  - **Direct**: "What would be the impact if this group's specific way
    of performing (its estimated effect) was generalized to a standard
    population, compared to a reference way of performing?"

### 4.2. Complementary Perspectives

Neither method is inherently superior; they offer different,
complementary insights:

- **Indirect Standardization** is useful for understanding deviations
  *from a baseline model* for a specific group, considering its unique
  composition. It highlights how much a group's random effect causes
  its predictions to differ from fixed-effect-only predictions.
- **Direct Standardization** is useful for understanding the impact of
  a group's *specific estimated effect* if it were generalized. It
  helps compare the magnitude of different groups' unique
  characteristics (e.g., their random effects) on a common demographic
  footing.

### 4.3. A Note on Model-Based Standardization

In the context of mixed models or other statistical models, using
model-fitted values (e.g., $\hat{y}_i$ or $\hat{p}_i$) for
calculations — especially for the "observed" component in indirect
standardization or as the basis for group-specific effects in direct
standardization — is common. This approach:

- Keeps the analysis within the model's predictive framework.
- Focuses on systematic variation captured by the model rather than
  raw, noisy data.
- Allows for the quantification of effects attributed to specific
  model components (like random effects).

## 5. Standardization in Survival Models

Survival analysis adds a time-to-event dimension that changes
standardization from comparing rates or means to comparing **event
counts against expected hazard exposure**. The general concept —
"compare what was observed to what was expected under a baseline model"
— is the same; the mechanics differ because the baseline is now a
**cumulative hazard function** rather than a probability or a mean.

### 5.1. The SMR / SHR via Indirect Standardization

In `pprof_py`, the Standardized Mortality Ratio (SMR) and Standardized
Hospitalization Ratio (SHR) are computed using a fitted `CoxPH` model.
The two-step process is:

1. **Fit a Cox model** on the full national population, including
   patient-level risk adjusters (age, comorbidities, etc.) — but
   **not** including the facility as a covariate. This yields
   $\hat{\boldsymbol\beta}$ and the baseline cumulative hazard
   $\hat{\Lambda}_0(t)$.

2. **Compute expected events per facility** by accumulating each
   patient's individual predicted hazard up to their observed follow-up
   time:

   $$
   E_k = \sum_{i \in k} \hat{\Lambda}_0(t_i)
   \exp(X_i \hat{\boldsymbol\beta})
   $$

   Then:

   $$
   \text{SMR}_k = \frac{O_k}{E_k}
   $$

An SMR of 1.30 for facility $k$ means 30 % more deaths than the
national model would predict for that specific patient mix. This is
the method used by CMS for the Standardized Mortality Ratio (SMR) and
Standardized Hospitalization Ratio (SHR) in dialysis facility reports.

For a complete worked example with real output, see
[Chapter 4 — Indirect Standardization: The SMR](survival/04_indirect_standardization_smr_shr).

### 5.2. Relationship to Linear and Logistic Standardization

The key difference is in what the "expected" value represents:

| Model family | "Expected" for group $k$ | Measure | Scale |
|---|---|---|---|
| Linear | $\sum_{i \in k} X_i\hat{\boldsymbol\beta}$ | ISD = $(O_k - E_k)/n_k$ | Outcome units (e.g., days) |
| Logistic | $\sum_{i \in k} \hat{p}_i(\text{fixed only})$ | ISR = $O_k / E_k$ | Ratio (unitless) |
| Survival | $\sum_{i \in k} \hat{\Lambda}_0(t_i)\exp(X_i\hat{\boldsymbol\beta})$ | SMR = $O_k / E_k$ | Ratio (unitless) |

All three use the same conceptual framework: compare observed outcomes
to expected outcomes under a national/baseline model, adjusted for each
provider's specific patient mix.

## 6. Choosing the Right Method

The choice between indirect and direct standardization depends
fundamentally on the research question:

- Choose **Indirect Standardization** if you are asking: "How
  different is this group's outcome from what we'd expect based on
  common factors and its specific case-mix?" This is often about
  assessing performance relative to an individualized baseline.

- Choose **Direct Standardization** if you are asking: "How different
  is this group's specific estimated effect (e.g., its quality or
  efficiency factor) compared to an average or null effect, if this
  factor were applied consistently across a standard population?" This
  is often about comparing the inherent "effect" of different groups on
  a level playing field.

Careful consideration of the analytical goals and the interpretation
desired will guide the selection of the most appropriate
standardization method.
