.. pprof_py documentation master file, created by
   sphinx-quickstart on YYYY-MM-DD.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pprof_py's Documentation
====================================

**pprof_py** is a Python package that provides a variety of risk-adjusted models for provider profiling, efficiently handling large-scale provider data. Implemented with Python and NumPy, it offers tools for calculating standardized measures, hypothesis testing, and visualization to evaluate healthcare provider performance and identify significant deviations from expected standards.

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: Table of Contents:

   introduction
   installation
   usage
   modules
   references

Introduction
------------

Provider profiling involves assessing and comparing the performance of healthcare providers by evaluating specific metrics that reflect quality of care, efficiency, and patient outcomes. To achieve this, it is essential to fit robust statistical models and design appropriate measures.

**pprof_py** is the Python implementation of the original R package, offering a comprehensive toolkit for fitting a variety of risk-adjusted models. Each model includes features for calculating standardized measures, conducting statistical inference, and visualizing results. The package is designed to address key limitations in existing R functions for provider profiling, particularly their computational inefficiency when applied to large-scale provider data.

Features:
- Logistic fixed effect models using the SerBIN algorithm.
- Linear fixed effect models with profile-based methods.
- Parallel computing for enhanced performance.
- Support for diverse outcomes (binary, continuous).
- Direct and indirect standardization methods.

Installation
------------

**Note:** _This package is in early development. Please report any issues._

To install `pprof_py`, clone the repository and use `pip`:

.. code-block:: bash

   git clone https://github.com/UM-KevinHe/pprof_py.git
   cd pprof_py
   pip install .

Getting Help
------------

If you encounter any problems or bugs, contact us:
- xhliuu@umich.edu
- lfluo@umich.edu
- taoxu@umich.edu
- kevinhe@umich.edu

References
----------

1. Bates, D., Mächler, M., Bolker, B., & Walker, S. (2015). *Fitting linear mixed-effects models using lme4.* Journal of Statistical Software, 67(1), 1-48. https://doi.org/10.18637/jss.v067.i01
2. He, K., Kalbfleisch, J. D., Li, Y., & Li, Y. (2013). *Evaluating hospital readmission rates in dialysis facilities; adjusting for hospital effects.* Lifetime Data Analysis, 19, 490-512. https://link.springer.com/article/10.1007/s10985-013-9264-6
3. He, K. (2019). *Indirect and direct standardization for evaluating transplant centers.* Journal of Hospital Administration, 8(1), 9-14. https://www.sciedupress.com/journal/index.php/jha/article/view/14304
4. Hsiao, C. (2022). *Analysis of panel data.* Cambridge University Press.
5. Wu, W., Kuriakose, J. P., Weng, W., Burney, R. E., & He, K. (2023). *Test-specific funnel plots for healthcare provider profiling leveraging individual- and summary-level information.* Health Services and Outcomes Research Methodology, 23(1), 45-58. https://pubmed.ncbi.nlm.nih.gov/37621728/
6. Wu, W., Yang, Y., Kang, J., & He, K. (2022). *Improving large‐scale estimation and inference for profiling health care providers.* Statistics in Medicine, 41(15), 2840-2853. https://onlinelibrary.wiley.com/doi/full/10.1002/sim.938
