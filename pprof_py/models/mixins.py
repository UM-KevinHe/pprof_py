"""Structural-typing contracts (Protocols) for model classes.

These were originally ``ABC`` marker mixins that contributed no shared
implementation — only ``@abstractmethod`` signatures used as documentation.
Tier 3 of the refactoring plan replaced them with ``typing.Protocol``
equivalents: same names, same method signatures, zero behavior change,
but no more forced MRO position and no more ``abstractmethod`` machinery
for something that was always just an interface contract.

Model classes that previously listed these in their bases can continue to
do so (a class that inherits a Protocol explicitly is considered to
implement it), or they can drop the inheritance entirely and satisfy the
Protocol structurally — both are valid under Python's typing system.
"""
from __future__ import annotations

from typing import Any, runtime_checkable, Protocol


@runtime_checkable
class SummaryMixin(Protocol):
    """Structural contract for models that provide ``summary()``.

    Any class that implements ``summary(level, null, alternative)`` satisfies
    this Protocol, whether or not it inherits from ``SummaryMixin`` explicitly.
    """

    def summary(self, level: float = 0.95, null: float = 0, alternative: str = "two_sided") -> Any:
        """Generate summary statistics of the model fit.

        Parameters
        ----------
        level : float, optional
            Confidence level for the intervals (default is 0.95).
        null : float, optional
            The null hypothesis value for the parameter estimates (default is 0).
        alternative : str, optional
            Specifies the alternative hypothesis, must be one of
            ``"two_sided"``, ``"greater"``, or ``"less"`` (default is
            ``"two_sided"``).

        Returns
        -------
        Any
            A model-specific summary object containing all relevant statistics.
        """
        ...


@runtime_checkable
class TestMixin(Protocol):
    """Structural contract for models that provide ``test()``.

    Any class that implements ``test(*args, **kwargs)`` satisfies this
    Protocol, whether or not it inherits from ``TestMixin`` explicitly.
    """

    def test(self, *args: Any, **kwargs: Any) -> Any:
        """Conduct hypothesis tests on model parameters.

        Parameters
        ----------
        *args : Any
            Positional arguments specific to the testing procedure.
        **kwargs : Any
            Keyword arguments specific to the testing procedure.

        Returns
        -------
        Any
            An object (or dictionary) containing the results of the
            hypothesis tests, such as p-values, test statistics, and
            degrees of freedom.
        """
        ...


@runtime_checkable
class PlotMixin(Protocol):
    """Structural contract for models that provide standard plotting methods.

    Any class that implements the methods below satisfies this Protocol,
    whether or not it inherits from ``PlotMixin`` explicitly.
    """

    def plot_funnel(self, *args: Any, **kwargs: Any) -> None:
        """Generate a funnel plot comparing provider performance."""
        ...

    def plot_residuals(self, *args: Any, **kwargs: Any) -> None:
        """Plot residuals versus fitted values."""
        ...

    def plot_qq(self, *args: Any, **kwargs: Any) -> None:
        """Generate a Q-Q plot for model residuals."""
        ...

    def plot_provider_effects(self, *args: Any, **kwargs: Any) -> None:
        """Plot provider effects with confidence intervals."""
        ...

    def plot_standardized_measures(self, *args: Any, **kwargs: Any) -> None:
        """Plot standardized measures with confidence intervals."""
        ...

    def plot_coefficient_forest(self, *args: Any, **kwargs: Any) -> None:
        """Create a forest plot of the covariate coefficients."""
        ...