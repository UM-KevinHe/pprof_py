# pprof_py

**pprof_py** is a Python package that provides a variety of risk-adjusted models for provider profiling, efficiently handling large-scale provider data. Implemented with Python and NumPy, it offers tools for calculating standardized measures, hypothesis testing, and visualization to evaluate healthcare provider performance and identify significant deviations from expected standards.

## Introduction

Provider profiling involves assessing and comparing the performance of healthcare providers by evaluating specific metrics that reflect quality of care, efficiency, and patient outcomes. To achieve this, it is essential to fit robust statistical models and design appropriate measures.

`pprof_py` is the Python implementation of the original R package, offering a comprehensive toolkit for fitting a variety of risk-adjusted models. Each model includes features for calculating standardized measures, conducting statistical inference, and visualizing results. The package is designed to address key limitations in existing R functions for provider profiling, particularly their computational inefficiency when applied to large-scale provider data.

For instance, the logistic fixed effect model employs a serial blockwise inversion Newton (SerBIN) algorithm to leverage the block structure of the information matrix, while linear fixed effect models utilize a profile-based method. With added parallel computing capabilities, `pprof_py` significantly improves computational speed. The package supports diverse outcomes (e.g., binary and continuous) and offers both direct and indirect standardization methods.

## Installation

**Note:** _This package is still in early development, so please report any issues or bugs you encounter._

You can install `pprof_py` from GitHub:

```bash
git clone https://github.com/yourusername/pprof_py.git
cd pprof_py
pip install .
```
