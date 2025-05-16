.. _fixed_vs_random_effects:

====================================
Fixed Effects vs. Random Effects Models
====================================

When analyzing data with a grouped or hierarchical structure (e.g., patients within hospitals, students within schools, repeated measurements on individuals), a key modeling decision is how to account for group-level heterogeneity. Two common approaches are fixed effects (FE) models and random effects (RE) models. This document outlines their differences, assumptions, and appropriate use cases.

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
============

Both fixed and random effects models aim to control for unobserved group-specific characteristics that might influence the outcome variable. However, they differ fundamentally in how they treat these group-specific effects and the assumptions they make.

Fixed Effects (FE) Models
=========================
In a fixed effects model, the group-specific effects are treated as fixed, unknown parameters to be estimated. Essentially, each group is allowed to have its own intercept (or its own set of coefficients if interactions with group dummies are included).

**Simple Linear Representation** (Group Intercepts)
For an observation :math:`i` in group :math:`k`:

.. math::
   y_{ik} = X_{ik}\boldsymbol{\beta} + \gamma_k + \epsilon_{ik}

Where:

*   :math:`y_{ik}` is the outcome for observation :math:`i` in group :math:`k`.
*   :math:`X_{ik}` is a vector of observed covariates.
*   :math:`\boldsymbol{\beta}` is a vector of fixed coefficients for the covariates, common across all groups.
*   :math:`\gamma_k` is the fixed effect (intercept) for group :math:`k`. This captures all time-invariant, unobserved heterogeneity specific to group :math:`k`.
*   :math:`\epsilon_{ik}` is the idiosyncratic error term.

Key Assumptions & Properties
----------------------------

1.  **Correlation with Covariates**: The primary strength of FE models is that they allow the unobserved group-specific effects (:math:`\gamma_k`) to be correlated with the observed covariates (:math:`X_{ik}`). This is crucial if unobserved group characteristics that affect the outcome are also related to the predictors.
2.  **Strict Exogeneity**: For unbiased estimation of :math:`\boldsymbol{\beta}`, the covariates :math:`X_{ik}` must be strictly exogenous conditional on :math:`\gamma_k`. This means :math:`E(\epsilon_{ik} | X_{ik}, \gamma_k) = 0`.
3.  **Estimation**: FE models are often estimated using:
    *   **Least Squares Dummy Variable (LSDV)**: Including a dummy variable for each group (except one reference group). This is computationally intensive for many groups.
    *   **Within-Group Transformation (Demeaning)**: Subtracting the group-mean from each variable. This eliminates :math:`\gamma_k` and allows :math:`\boldsymbol{\beta}` to be estimated by OLS on the transformed data.
4.  **Incidental Parameters Problem**: If the number of groups (:math:`m`) is large and the number of observations per group (:math:`n_k`) is small, the estimates of :math:`\gamma_k` can be imprecise. However, the estimates of :math:`\boldsymbol{\beta}` can still be consistent as :math:`N \rightarrow \infty` (if :math:`n_k` is fixed) or as :math:`n_k \rightarrow \infty`. For non-linear models (like logistic FE), consistency of :math:`\boldsymbol{\beta}` often requires :math:`n_k \rightarrow \infty`.
5.  **No Time-Invariant Group-Level Predictors**: FE models cannot estimate the effect of variables that are constant within groups (e.g., hospital type, gender of an individual in a panel) because the within-group transformation removes them.

When to Use Fixed Effects
-------------------------

*   **Primary Goal**: To control for unobserved heterogeneity that is potentially correlated with the predictors. This is the main reason to choose FE over RE.
*   **Data Structure**: Panel data (individuals/groups observed over time) or clustered data where group-specific unobservables are a concern.
*   **Interest in Within-Group Variation**: FE models primarily leverage within-group variation to estimate coefficients.
*   **Robustness to Endogeneity**: If you suspect that unobserved group characteristics are correlated with your :math:`X` variables, FE is generally preferred for estimating :math:`\boldsymbol{\beta}`.

Random Effects (RE) Models
==========================


In a random effects model, the group-specific effects are treated as random variables drawn from a common distribution, typically assumed to be normal. These effects are part of the error term.

**Simple Linear Representation** (Random Intercepts):
For an observation :math:`i` in group :math:`k`:

.. math::
   y_{ik} = X_{ik}\boldsymbol{\beta} + u_k + \epsilon_{ik}

Where:

*   :math:`y_{ik}`, :math:`X_{ik}`, :math:`\boldsymbol{\beta}`, :math:`\epsilon_{ik}` are as defined before.
*   :math:`u_k` is the random effect for group :math:`k`. It is assumed to be a random variable, typically :math:`u_k \sim N(0, \sigma_u^2)`.
*   The composite error term is :math:`v_{ik} = u_k + \epsilon_{ik}`.

Key Assumptions & Properties
----------------------------
1.  **No Correlation with Covariates (Crucial Assumption)**: The random effects :math:`u_k` are assumed to be uncorrelated with the observed covariates :math:`X_{ik}`. That is, :math:`Cov(X_{ik}, u_k) = 0`. If this assumption is violated, the estimates of :math:`\boldsymbol{\beta}` will be biased and inconsistent.
2.  **Distributional Assumption**: :math:`u_k` are typically assumed to be i.i.d. from a specific distribution (e.g., Normal).
3.  **Estimation**: RE models are often estimated using Generalized Least Squares (GLS) or Maximum Likelihood Estimation (MLE). These methods account for the correlation structure in the composite error term :math:`v_{ik}` induced by the shared :math:`u_k` within groups.
4.  **Efficiency**: If the no-correlation assumption holds, RE estimators are more efficient (have smaller variances) than FE estimators because they use both within-group and between-group variation.
5.  **Time-Invariant Group-Level Predictors**: RE models *can* estimate the effects of variables that are constant within groups, as these variables are not removed by the estimation procedure.
6.  **Prediction of Random Effects**: The specific values of :math:`u_k` are not directly estimated as parameters but can be predicted after model fitting, often referred to as Best Linear Unbiased Predictors (BLUPs).

When to Use Random Effects
--------------------------

*   **Primary Goal**: To account for within-group correlation and improve efficiency, *assuming* the unobserved group effects are uncorrelated with predictors.
*   **Data Structure**: Panel data or clustered data.
*   **Interest in Group-Level Predictors**: If you need to estimate the effect of time-invariant group characteristics.
*   **Generalizability**: If the groups in your sample are considered a random sample from a larger population of groups, and you want to make inferences about that population.
*   **Efficiency is Key**: If you are confident that :math:`Cov(X_{ik}, u_k) = 0`, RE is more efficient.

Key Differences Summarized
==========================


.. list-table::
   :header-rows: 1
   :widths: 25 35 35

   * - Feature
     - Fixed Effects (FE) Model
     - Random Effects (RE) Model
   * - **Group Effects**
     - Treated as fixed parameters to estimate
     - Treated as random variables from a distribution
   * - **Correlation with X**
     - Allowed (:math:`Cov(X_{ik}, \gamma_k) \neq 0`)
     - Not allowed (:math:`Cov(X_{ik}, u_k) = 0`)
   * - **Estimation of** :math:`\boldsymbol{\beta}`
     - Uses within-group variation
     - Uses both within & between-group variation
   * - **Efficiency**
     - Less efficient if RE assumptions hold
     - More efficient if assumptions hold
   * - **Time-Invariant Predictors**
     - Cannot estimate their effects
     - Can estimate their effects
   * - **Consistency of** :math:`\boldsymbol{\beta}`
     - Consistent even if :math:`Cov(X_{ik}, \gamma_k) \neq 0`
     - Consistent only if :math:`Cov(X_{ik}, u_k) = 0`
   * - **Inference Focus**
     - Conditional on groups in sample
     - Unconditional, generalizable to population of groups


Choosing Between Fixed and Random Effects
=========================================

1.  **Hausman Test**:

    *   The Hausman test is commonly used to help decide between FE and RE models.
    *   **Null Hypothesis (H0)**: The random effects model is appropriate (i.e., :math:`Cov(X_{ik}, u_k) = 0`).
    *   **Alternative Hypothesis (H1)**: The fixed effects model is appropriate (i.e., :math:`Cov(X_{ik}, u_k) \neq 0`).
    *   If the test rejects H0, it suggests that the RE model's key assumption is violated, and the FE model is preferred for consistent estimation of :math:`\boldsymbol{\beta}`.
    *   If the test fails to reject H0, it suggests that the RE model might be appropriate and more efficient.
    *   *Caution*: The Hausman test has its own assumptions and limitations.

2.  **Theoretical Considerations**:

    *   Consider the nature of your data and research question. Are the unobserved group characteristics likely to be correlated with your predictors? If so, FE is safer.
    *   Are you interested in estimating the effects of time-invariant group-level variables? If so, RE (or alternative models like Hausman-Taylor) might be necessary.

3.  **Research Goal**:

    *   If the primary goal is to obtain unbiased and consistent estimates of the effects of :math:`X` variables, and there's a strong suspicion of correlation between unobserved group effects and :math:`X`, FE is generally the preferred approach despite its limitations.
    *   If the groups are a random sample and you want to generalize, and you believe the orthogonality assumption holds, RE is more efficient.

Relationship to Standardization
===============================

Both FE and RE models provide a basis for standardization:

*   **Fixed Effects Models**: The estimated fixed effects (:math:`\hat{\gamma}_k`) directly represent the adjusted mean outcome for each group after controlling for :math:`X`. Standardized differences can be calculated directly from these :math:`\hat{\gamma}_k` values (e.g., :math:`\hat{\gamma}_k - \text{reference}`).
*   **Random Effects Models**: The predicted random effects (BLUPs, :math:`\hat{u}_k`) represent the deviation of each group from the average, after controlling for :math:`X`. These :math:`\hat{u}_k` values, combined with fixed effect predictions, are used to calculate standardized differences or ratios.

The choice of FE or RE impacts the interpretation of these standardized measures. FE-based standardization is conditional on the specific groups in the sample, while RE-based standardization aims for broader population inferences.

Conclusion
==========

The choice between fixed and random effects models is a critical one in panel and clustered data analysis. It hinges on the assumptions one is willing to make about the correlation between unobserved group-specific effects and the covariates of interest. Fixed effects models offer robustness against such correlations but cannot estimate effects of time-invariant predictors. Random effects models are more efficient and can handle time-invariant predictors but require the strong assumption that unobserved group effects are uncorrelated with covariates. Careful consideration of the research question, data properties, and diagnostic tests like the Hausman test should guide this decision.