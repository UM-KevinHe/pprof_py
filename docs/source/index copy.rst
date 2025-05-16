.. pprof_py documentation master file, created by
   sphinx-quickstart on YYYY-MM-DD.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to pprof_py's Documentation
====================================

**pprof_py** is a Python package that provides a variety of risk-adjusted models for provider profiling, efficiently handling large-scale provider data. Implemented with Python and NumPy, it offers tools for calculating standardized measures, hypothesis testing, and visualization to evaluate healthcare provider performance and identify significant deviations from expected standards.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   base_model
   linear_fixed_effect_model
   linear_random_effect_model
   logistic_fixed_effect_model
   logistic_random_effect_model
   direct_vs_indirect_standardization
   fixed_vs_random_effects
   api


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
.. bibliography:: references.bib
   :list: enumerate
   :all: