.. _standardization_methods:

===========================================
Direct vs. Indirect Standardization Methods
===========================================

Standardization methods are crucial in statistical modeling, especially in fields like healthcare and social sciences. They enable fair comparisons between groups (e.g., providers, regions) by adjusting for differences in their underlying populations or case-mix. This document explores two common approaches: indirect and direct standardization, detailing their application in both linear and logistic models.

.. contents:: Table of Contents
   :local:
   :depth: 2

1. Introduction to Standardization
==================================

When comparing outcomes across different groups, raw (unadjusted) figures can be misleading. Groups often differ systematically in characteristics that influence outcomes (e.g., patient severity, demographics). Standardization aims to provide a more equitable comparison by accounting for these differences.

*   **Linear Models**: Standardization often involves comparing adjusted mean differences in the outcome variable (e.g., difference in length of stay, cost).
*   **Logistic Models**: For binary outcomes, standardization typically focuses on comparing adjusted rates or ratios (e.g., standardized mortality ratio (SMR), standardized infection ratio (SIR), or standardized rates).

This document focuses on model-based standardization, particularly within the context of mixed models where outcomes are influenced by both fixed (common) and random (group-specific) effects.

2. Indirect Standardization
=========================

2.1. Core Concept and Purpose
-----------------------------

Indirect standardization evaluates how a group's outcome (often based on model predictions) deviates from an **expected outcome**. This expected outcome is calculated using a **baseline model** (typically representing common, observable characteristics, like fixed effects) applied to the specific case-mix of the group in question.

*   **Focus**: To assess how much a group's performance differs from what would be expected if it performed according to a common baseline, given its specific population characteristics.
*   **Purpose**:
    *   To isolate and quantify the impact of group-specific factors (e.g., random effects or unmodeled characteristics) beyond the baseline predictions.
    *   To identify groups performing significantly differently than expected under a common, fixed structure (akin to calculating SMRs or SIRs).

2.2. Process
------------

The process generally involves comparing "observed" (full model) values to "expected" (baseline model) values for the group.

**Step 1: Define "Observed" Values (Model-Based)**

For model-based standardization, the "observed" values for a group are typically derived from the **full model predictions** (including both fixed and random effects) for individuals within that group.
Using model-fitted values (e.g., :math:`\hat{y}_i` or :math:`\hat{p}_i` from the full model) rather than raw actual outcomes helps to:

    *   Focus on modeled effects, keeping the analysis within the model's framework.
    *   Reduce the influence of random noise in the raw data not captured by the model.
    *   Ensure consistency in comparisons relative to the model's structure.

**Step 2: Define "Expected" Values**

The "expected" values are predictions from a **baseline model** applied to the individuals within the group. This baseline often consists of only the fixed effects component of a mixed model.
   
    *   **Linear Models**: Expected sum for group :math:`k` is :math:`\sum_{i \in k} (X_i\boldsymbol{\hat{\beta}})`.
    *   **Logistic Models**: Expected sum of probabilities for group :math:`k` is :math:`\sum_{i \in k} \hat{p}_i(\text{fixed only})`.

**Step 3: Calculate the Standardized Measure**

*   **Linear Models (Indirect Standardized Difference - ISD)**:

    The ISD quantifies the average difference per unit in the group attributable to factors beyond the baseline.

    .. math::
       \text{ISD}_k = \frac{\sum_{i \in k} \hat{y}_i(\text{full model}) - \sum_{i \in k} (X_i\boldsymbol{\hat{\beta}})}{n_k}

    Where :math:`\hat{y}_i(\text{full model})` is the prediction from the full model for individual :math:`i` in group :math:`k`, :math:`X_i\boldsymbol{\hat{\beta}}` is the prediction from the fixed-effects-only baseline model, and :math:`n_k` is the size of group :math:`k`.

*   **Logistic Models (Indirect Standardized Ratio - ISR / Indirect Standardized Rate)**:

    The ISR compares the sum of model-predicted probabilities (full model) to the sum of expected probabilities (baseline model) for the group.

    .. math::
       \text{ISR}_k = \frac{\sum_{i \in k} \hat{p}_i(\text{full model})}{\sum_{i \in k} \hat{p}_i(\text{fixed only})}

    The **Indirect Standardized Rate** can then be obtained by:

    .. math::
       \text{Indirect Rate}_k = \text{ISR}_k \times \text{Overall Population Rate}

2.3. When to Use Indirect Standardization
-----------------------------------------

*   When the primary interest is in the incremental impact of group-specific effects (e.g., random effects) over and above predictions from a common, fixed-effects structure.
*   To assess how a group's outcomes deviate from an expectation based on general population rates or a baseline model, applied to that group's specific demographic or case-mix.
*   **Key Question Answered**: "How different is this group's outcome from what we'd expect based on common factors, given its specific case-mix?"

3. Direct Standardization
=======================

3.1. Core Concept and Purpose
-----------------------------

Direct standardization evaluates a group's specific modeled effect (e.g., its fixed effects plus its unique random effect) by applying this effect to a **standard population**. The outcome is then compared to the outcome derived from applying a **reference effect** (e.g., an average or null random effect) to the same standard population.

*   **Focus**: To assess the impact of a group's specific performance characteristic if it were applied universally to a standard population, allowing for comparison against a common benchmark.
*   **Purpose**:

    *   To compare the magnitude of different groups' specific effects on a common footing.
    *   To understand the overall impact of a group's estimated random effect in a broader, standardized context.

3.2. Process
------------

The process involves applying different effects to a common standard population and comparing the results.

**Step 1: Define the Standard Population**

This is a common reference population to which different effects will be applied. It could be the entire dataset, a subset, or an external population. Let :math:`N_{\text{std pop}}` be its size.

**Step 2: Calculate Expected Outcomes under the Group's Specific Effect**

Apply the full modeled effect of group :math:`k` (fixed effects + :math:`\hat{\gamma}_k`, its specific random effect) to each individual :math:`j` in the *standard population*.

    *   **Linear Models**: Sum of predictions is :math:`\sum_{j \in \text{std pop}} (X_j\boldsymbol{\hat{\beta}} + \hat{\gamma}_k)`.
    *   **Logistic Models**: Sum of probabilities is :math:`\sum_{j \in \text{std pop}} \hat{p}_j(\text{fixed} + \text{random effect}_k)`.

**Step 3: Calculate Expected Outcomes under a Reference Effect**

Apply a reference effect (e.g., a null random effect, :math:`\text{RE}_{\text{ref}}=0`, or the mean/median of estimated random effects) along with fixed effects to each individual :math:`j` in the *standard population*.

    *   **Linear Models**: Sum of predictions is :math:`\sum_{j \in \text{std pop}} (X_j\boldsymbol{\hat{\beta}} + \text{RE}_{\text{ref}})`.
    *   **Logistic Models**: Sum of probabilities is :math:`\sum_{j \in \text{std pop}} \hat{p}_j(\text{fixed} + \text{RE}_{\text{ref}})`.

**Step 4: Calculate the Standardized Measure**

*   **Linear Models (Direct Standardized Difference - DSD)**:

    The DSD quantifies the average difference in outcome if group :math:`k`'s effect versus a reference effect were applied to the standard population.

    .. math::
       \text{DSD}_k = \frac{\sum_{j \in \text{std pop}} (X_j\boldsymbol{\hat{\beta}} + \hat{\gamma}_k) - \sum_{j \in \text{std pop}} (X_j\boldsymbol{\hat{\beta}} + \text{RE}_{\text{ref}})}{N_{\text{std pop}}}

*   **Logistic Models (Direct Standardized Ratio - DSR / Direct Standardized Rate)**:

    The DSR compares the total expected events if group :math:`k`'s specific effect were applied to the standard population, versus if a reference effect were applied.

    .. math::
       \text{DSR}_k = \frac{\sum_{j \in \text{std pop}} \hat{p}_j(\text{fixed} + \text{random effect}_k)}{\sum_{j \in \text{std pop}} \hat{p}_j(\text{fixed} + \text{RE}_{\text{ref}})}

    The **Direct Standardized Rate** can then be obtained by:

    .. math::
       \text{Direct Rate}_k = \text{DSR}_k \times \text{Overall Population Rate (of the standard population)}

3.3. When to Use Direct Standardization
---------------------------------------

*   When you need to compare the magnitude of different groups' specific effects as if they were applied to a common, standard population.
*   To understand how much each group's specific modeled effect deviates from an overall average or baseline effect when projected onto a common scale.
*   **Key Question Answered**: "How different is this group's specific estimated effect compared to an average or null effect, if applied consistently to a standard population?"

4. Comparing Indirect and Direct Standardization
===============================================

4.1. Key Distinctions
---------------------

*   **Reference Point**:

    *   **Indirect**: Compares the group's outcome (within its *own case-mix*) to an expected outcome based on a standard *model or rate structure*.
    *   **Direct**: Compares the outcome from applying the group's *specific effect* to a *standard population* against the outcome from applying a reference effect to that same standard population.
*   **Primary Question**:

    *   **Indirect**: "How does this group's actual (or model-fitted) performance compare to what's expected for a group with its specific characteristics, under a baseline scenario?"
    *   **Direct**: "What would be the impact if this group's specific way of performing (its estimated effect) was generalized to a standard population, compared to a reference way of performing?"

4.2. Complementary Perspectives
--------------------------------
Neither method is inherently superior; they offer different, complementary insights:

*   **Indirect Standardization** is useful for understanding deviations *from a baseline model* for a specific group, considering its unique composition. It highlights how much a group's random effect, for instance, causes its predictions to differ from fixed-effect-only predictions.
*   **Direct Standardization** is useful for understanding the impact of a group's *specific estimated effect* if it were generalized. It helps compare the magnitude of different groups' unique characteristics (e.g., their random effects) on a common demographic footing.

4.3. A Note on Model-Based Standardization
------------------------------------------

In the context of mixed models or other statistical models, using model-fitted values (e.g., :math:`\hat{y}_i` or :math:`\hat{p}_i`) for calculations, especially for the "observed" component in indirect standardization or as the basis for group-specific effects in direct standardization, is common. This approach:
   
    *   Keeps the analysis within the model's predictive framework.
    *   Focuses on systematic variation captured by the model rather than raw, noisy data.
    *   Allows for the quantification of effects attributed to specific model components (like random effects).

5. Conclusion: Choosing the Right Method
========================================

The choice between indirect and direct standardization depends fundamentally on the research question:

*   Choose **Indirect Standardization** if you are asking:
    "How different is this group's outcome from what we'd expect based on common factors and its specific case-mix?"
    This is often about assessing performance relative to an individualized baseline.

*   Choose **Direct Standardization** if you are asking:
    "How different is this group's specific estimated effect (e.g., its quality or efficiency factor) compared to an average or null effect, if this factor were applied consistently across a standard population?"
    This is often about comparing the inherent "effect" of different groups on a level playing field.

Careful consideration of the analytical goals and the interpretation desired will guide the selection of the most appropriate standardization method.



.. .. _standardization_methods:

.. ===========================================
.. Direct vs. Indirect Standardization Methods
.. ===========================================

.. Standardization methods are crucial in statistical modeling, especially in fields like healthcare and social sciences, for making fair comparisons between groups (e.g., providers, regions) by adjusting for differences in their underlying populations or case-mix. This document explores two common approaches: indirect and direct standardization, and how they apply to both linear and logistic models.

.. .. contents:: Table of Contents
..    :local:
..    :depth: 2

.. Overview of Standardization
.. ===========================

.. When comparing outcomes across different groups, raw (unadjusted) outcomes can be misleading because the groups might differ systematically in characteristics that influence the outcome (e.g., patient severity, demographics). Standardization aims to provide a more equitable comparison by accounting for these differences.

.. *   **Linear Models**: In linear models, standardized differences often refer to the adjusted mean difference in the outcome variable (e.g., difference in length of stay, cost).
.. *   **Logistic Models**: In logistic models (for binary outcomes), standardization typically focuses on comparing adjusted rates or ratios (e.g., standardized mortality ratio (SMR), standardized infection ratio (SIR), or standardized rates).

.. Indirect Standardization
.. ========================

.. Purpose
.. -------
.. Indirect standardization evaluates how much the observed outcome (or a model-based representation of it) for each group deviates from an expected outcome. This expected outcome is typically derived from a baseline model that accounts for common, observable characteristics (often represented by fixed effects in a mixed model). The "indirect" aspect comes from comparing the observed to an expected value calculated using a common standard (e.g., overall population rates or fixed-effects-only predictions).

.. Process
.. -------

.. 1.  **Expected Values**:

..     *   Calculate the expected outcome for each observation or group based on a reference model.

..         *   **Linear Models**: This is often the sum of predictions from the fixed effects component (:math:`X\boldsymbol{\hat{\beta}}`) for all individuals within a group.
..         *   **Logistic Models**: This is often the sum of expected probabilities derived from the fixed effects component (e.g., :math:`\sum_i p_i(\text{fixed only})`) for all individuals in a group.
    
..     *   These values represent what would be observed if only the common, adjustable factors (fixed effects) were at play, providing a baseline.

.. 2.  **Observed Values (Model-Based)**:

..     *   Determine the "observed" outcome for each group. In the context of model-based standardization, particularly with mixed models, this often refers to the sum of predictions from the *full model* (including both fixed and random effects, i.e., :math:`\text{fitted values } \hat{y}_i`) for all individuals within that group.
..     *   Using model-fitted values rather than raw actual outcomes (`y_i`) helps to:

..         *   **Focus on Modeled Effects**: It assesses deviations based on what the *model* predicts, isolating the analysis within the model's framework.
..         *   **Reduce Noise**: It minimizes the influence of random variation in the raw data not captured by the systematic components of the model.
..         *   **Maintain Consistency**: It ensures that comparisons are made relative to the model's predictive structure.

.. 3.  **Calculating the Standardized Measure**:

..     *   **Linear Models (Standardized Difference)**:

..         *   The difference is calculated as: :math:`\text{Observed Sum} - \text{Expected Sum}`.
..         *   This deviation indicates the additional effect attributable to factors captured by group-specific components (e.g., random effects).
..         *   This difference is often normalized by the group size to get an average per-unit difference.

..         .. math::
..            \text{Indirect Standardized Difference}_k = \frac{\sum_{i \in k} \hat{y}_i - \sum_{i \in k} (X_i\boldsymbol{\hat{\beta}})}{n_k}

..     *   **Logistic Models (Standardized Ratio/Rate)**:

..         *   **Standardized Ratio (e.g., SMR/SIR-like)**:

..            .. math::
..               \text{Indirect Standardized Ratio}_k = \frac{\sum_{i \in k} \hat{p}_i(\text{full model})}{\sum_{i \in k} \hat{p}_i(\text{fixed only})}
          
..            This compares the sum of model-predicted probabilities (full model) for group :math:`k` to the sum of expected probabilities if only fixed effects applied to group :math:`k`'s individuals.
       
..         *   **Indirect Standardized Rate**: The ratio can then be multiplied by an overall population rate (e.g., overall observed event rate in the dataset) to get a standardized rate for the group.
           
..            .. math::
..               \text{Indirect Standardized Rate}_k = \text{Indirect Standardized Ratio}_k \times \text{Overall Population Rate}

.. Importance
.. ----------
.. Indirect standardization helps pinpoint which groups perform significantly differently from what a baseline (fixed effects) model predicts. It highlights the scale and direction of group deviations, illuminating the impact of factors beyond simple fixed effects, often captured by random effects or unmodeled group-specific characteristics.

.. Direct Standardization
.. ======================

.. Purpose
.. -------
.. Direct standardization evaluates how each group's modeled performance (accounting for its specific characteristics, often including random effects) would compare if applied to a standard population structure, or alternatively, how it deviates from an overall model prediction that already incorporates average group effects.

.. Process
.. -------

.. 1.  **Group-Specific Expected Values (Full Model)**:

..     *   Calculate the expected outcome for each group using the *full model* predictions for individuals within that group. This incorporates both fixed effects and the group's specific random effect.
        
..         *   **Linear Models**: :math:`\sum_{i \in k} (X_i\boldsymbol{\hat{\beta}} + \hat{\gamma}_k)` where :math:`\hat{\gamma}_k` is the random effect for group :math:`k`.
..         *   **Logistic Models**: :math:`\sum_{i \in k} \hat{p}_i(\text{fixed} + \text{random effect}_k)`.

.. 2.  **Reference/Baseline Value**:

..     *   **Linear Models**: This could be the overall mean predicted outcome across all observations from the full model, or the prediction from a "null" random effect (e.g., :math:`\gamma_k = 0` or mean/median of :math:`\hat{\gamma}_k`).
..     *   **Logistic Models**: This could be the sum of predicted probabilities if all individuals were subjected to an "average" or "null" random effect (e.g., :math:`\text{RE}=0`, or median/mean of BLUPs), or the overall observed event rate. For ratios, the denominator is often the sum of probabilities predicted for the *entire population* under the influence of that specific group's random effect.

.. 3.  **Calculating the Standardized Measure**:

..     *   **Linear Models (Standardized Difference)**:

..         *   The difference is: :math:`(\text{Group-Specific Expected Sum}) - (\text{Reference Sum})`.
..         *   This is often normalized by the total number of observations in the dataset or the group size, depending on the specific comparison goal.
        
..         .. math::
..            \text{Direct Standardized Difference}_k = \frac{\sum_{i \in \text{dataset}} (X_i\boldsymbol{\hat{\beta}} + \hat{\gamma}_k) - \sum_{i \in \text{dataset}} (X_i\boldsymbol{\hat{\beta}} + \text{RE}_{\text{null}})}{N_{\text{total}}}
        
..         This formula shows applying group k's effect to the whole dataset and comparing it to applying a null effect to the whole dataset.

..     *   **Logistic Models (Standardized Ratio/Rate)**:

..         *   **Direct Standardized Ratio**:

..            .. math::
..               \text{Direct Standardized Ratio}_k = \frac{\sum_{i \in \text{dataset}} \hat{p}_i(\text{fixed} + \text{random effect}_k)}{\sum_{i \in \text{dataset}} \hat{p}_i(\text{fixed} + \text{RE}_{\text{null}})}
          
..            This compares the total expected events if group :math:`k`'s specific random effect were applied to the entire population, versus if a null/average random effect were applied to the entire population.
        
..         *   **Direct Standardized Rate**: The ratio is then multiplied by an overall population rate.

..            .. math::
..               \text{Direct Standardized Rate}_k = \text{Direct Standardized Ratio}_k \times \text{Overall Population Rate}

.. Importance
.. ----------
.. Direct standardized measures reveal how group performances deviate from an overall or average expectation when fully accounting for their specific modeled effects (including random effects). It helps identify which groups stand out when their unique, modeled contributions are projected onto a common reference.

.. Indirect vs. Direct Standardization: A Comparison
.. =================================================

.. **Indirect Standardization**
.. ---------------------------

.. *   **Focus**: Compares a group's (model-fitted) outcome against predictions based on a baseline model (e.g., fixed effects only).
.. *   **Calculation (Conceptual)**:

..     *   **"Observed"**: Full model prediction for the group (Fixed + Random Effects).
..     *   **"Expected"**: Baseline model prediction for the group (Fixed Effects Only).
..     *   **Measure**: Difference (linear) or Ratio (logistic) of "Observed" to "Expected".

.. *   **Purpose**: Assesses how much group outcomes differ from what would be expected if only considering common adjustable factors. Isolates the impact of group-specific (random) effects relative to this baseline.
.. *   **When to Use**:

..     *   To isolate and quantify the contribution of random effects or unmodeled group-specific factors.
..     *   To identify groups performing significantly differently than expected under a common, fixed structure.

.. **Direct Standardization**
.. --------------------------

.. *   **Focus**: Evaluates a group's specific modeled effect (Fixed + its own Random Effect) by applying it to a standard population or comparing it against an overall/average population effect.
.. *   **Calculation (Conceptual)**:

..     *   **"Group-Specific Expected (applied to standard pop.)"**: Prediction if the group's specific random effect was applied to everyone.
..     *   **"Reference Expected (applied to standard pop.)"**: Prediction if a null/average random effect was applied to everyone.
..     *   **Measure**: Difference (linear) or Ratio (logistic) of these two.

.. *   **Purpose**: Evaluates group performance considering its total modeled effect, projected onto a common scale for comparison against an average or null effect.
.. *   **When to Use**:

..     *   When the focus is on understanding the total modeled variability among groups, accounting for their unique estimated effects.
..     *   To analyze how much each group's specific effect deviates from an overall average or baseline effect when applied consistently.

.. Which is Better?
.. ----------------
.. Neither method is inherently superior; they offer complementary perspectives.

.. *   **Indirect Standardization** is useful for understanding deviations *from a baseline model* (e.g., how much does the random effect add/subtract from fixed-effect predictions for this group?). It's often used when the "expected" is based on broader population characteristics applied to the group's specific case-mix.
.. *   **Direct Standardization** is useful for understanding the impact of a group's *specific estimated effect* if it were generalized or compared to an average effect applied to a standard population. It helps compare the magnitude of different groups' estimated random effects on a common footing.

.. **Choice Guidelines**:

.. *   **Use Indirect Standardization** when:

..     *   You're interested in the incremental impact of group-specific (random) effects beyond fixed-effect predictions.
..     *   The focus is on assessing deviations from a common, fixed-structure expectation.
..     *   Often aligns with calculating SMRs/SIRs where observed events in a group are compared to expected events based on general population rates applied to that group's demographic structure.

.. *   **Use Direct Standardization** when:

..     *   You need to compare the magnitude of different groups' specific effects (e.g., how would outcomes change if Group A's effect applied to everyone vs. Group B's effect?).
..     *   It’s crucial to understand the overall impact of a group's estimated random effect in a broader context.

.. In summary, the choice depends on the analytical question:

.. *   Are you asking "How different is this group's outcome from what we'd expect based on common factors?" (Indirect)
.. *   Or are you asking "How different is this group's specific estimated effect compared to an average or null effect, if applied consistently?" (Direct)

.. The interpretation of "observed" and "expected" can vary slightly based on the model (fixed vs. random effects) and the specific goals of the standardization. In mixed models, using model-fitted values for the "observed" component in indirect standardization helps to keep the analysis within the model's predictive framework.