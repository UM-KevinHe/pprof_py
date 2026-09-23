"""Structural-typing contracts (Protocols) for model classes.

These are ``typing.Protocol`` interfaces (same names and method signatures
as the original ABC marker mixins, zero behavior change) that document
which methods a model class is expected to provide.  Using Protocol instead
of ABC avoids forced MRO positioning and unnecessary ``abstractmethod``
machinery for what is purely an interface contract.

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

    def test(self, providers: Any = None, *, reference: Any = ..., null_model: Any = None,
             alternative: str = "two_sided", level: float = 0.95, critical: Any = None, **kwargs: Any) -> Any:
        """Test each provider's effect against a reference effect.

        The shared contract: family-specific statistics (selected with
        ``test_method`` where a family offers several), a reference effect
        ``reference`` (``"median"``, ``"mean"``, or a number), an optional null
        model, and a ``pandas.DataFrame`` indexed by provider with columns
        :data:`pprof_py.inference.PROVIDER_TEST_COLUMNS`, where ``flag`` is +1
        above the reference, -1 below, 0 not significant, and NA not tested.
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