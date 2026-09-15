.. _api:

API Reference
=============

This section provides the API reference for the ``pprof_py`` package.
All primary model classes are importable directly from ``pprof_py``.

.. contents::
   :local:
   :depth: 2

Package Root
------------

.. automodule:: pprof_py
   :members:
   :undoc-members:
   :show-inheritance:

Survival Models
---------------

.. autoclass:: pprof_py.models.survival.CoxPH
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pprof_py.models.survival.PenalizedCoxPH
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pprof_py.models.survival.PenalizedCoxPHCV
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pprof_py.models.survival.CauseSpecificCoxPH
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pprof_py.models.survival.FineGrayPH
   :members:
   :undoc-members:
   :show-inheritance:

Logistic Models
---------------

.. autoclass:: pprof_py.models.logistic.LogisticFixedEffectModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pprof_py.models.logistic.LogisticRandomEffectModel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pprof_py.models.logistic.LogisticMixedEffectModel
   :members:
   :undoc-members:
   :show-inheritance:

Linear Models
-------------

.. autoclass:: pprof_py.models.linear.LinearFixedEffectModel
   :members:
   :undoc-members:
   :show-inheritance:

Variable Selection
------------------

.. autoclass:: pprof_py.selection.CoxPHSelector
   :members:
   :undoc-members:
   :show-inheritance:

Inference
---------

.. autofunction:: pprof_py.inference.huber_location_scale

.. autofunction:: pprof_py.inference.estimate_empirical_null

Plotting
--------

.. autofunction:: pprof_py.plotting.plot_caterpillar

Utilities
---------

.. autofunction:: pprof_py.utils.setup_logger

.. autofunction:: pprof_py.utils.proc_freq

.. autofunction:: pprof_py.utils.sigmoid

Exceptions
----------

.. autoclass:: pprof_py.exceptions.NotFittedError
   :members:
   :show-inheritance:

