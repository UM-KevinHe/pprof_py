"""Sphinx extension: render the presentation gallery at build time (brief §10: images come from a deterministic
script during the build, never saved by hand).

On ``builder-inited`` it fits models to the package's synthetic data (``pprof_py.presentation._synthetic``), renders
every figure type with the deterministic renderers into ``_static/gallery/``, and writes
``presentation/_gallery_items.md`` (titles, questions, generated alt text, the call). Files are rewritten only when
their bytes change, so repeated builds do not trigger rebuilds.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Iterator, Tuple

ITEMS = {
    "funnel": ("Funnel plot", "Which providers depart from the reference by more than volume-driven noise explains?",
               "funnel(model)"),
    "caterpillar": ("Interval plot with volume panel", "How large and how uncertain are the provider estimates?",
                    "caterpillar(profile)"),
    "observed_expected": ("Observed versus expected", "Where do observed counts depart from expected, and at what "
                          "volume?", "observed_expected(profile)"),
    "forest": ("Coefficient forest", "How large are the covariate effects?", "forest(model)"),
    "data_quality": ("Data quality", "Who is missing or unreliable, and why?", "data_quality(profile)"),
    "null_calibration": ("Null calibration", "Is the theoretical null credible, and how much does calibration change "
                         "the flags?", "null_calibration(model, test_method=\"score\", null_model=EmpiricalNull.fitter())"),
    "reliability": ("Reliability", "How reliably does the measure separate providers of each size?", "reliability(iur)"),
    "provider_variation": ("Between-provider variation", "How much real variation exists across providers?",
                           "provider_variation(random_effect_model)"),
    "shrinkage": ("Shrinkage", "How far does pooling move each provider?", "shrinkage(model, random_effect_model)"),
    "flag_stability": ("Flag stability", "How fragile are the flags?", "flag_stability(model, test_method=\"score\")"),
    "multi_measure": ("Several measures", "How do providers compare across measures?", "multi_measure(measures)"),
    "measure_agreement": ("Agreement of two measures", "Do the measures agree, provider by provider?",
                          "measure_agreement(measures)"),
}


def figures() -> Iterator[Tuple[str, object]]:
    """``(name, FigureResult)`` for every gallery item, from 200 synthetic providers."""
    import numpy as np

    from pprof_py import LogisticFixedEffectModel, LogisticRandomEffectModel
    from pprof_py.inference import EmpiricalNull
    from pprof_py.measures.iur import BootstrapIUR
    from pprof_py import presentation as P
    from pprof_py.presentation._synthetic import provider_data

    d = provider_data(200)
    fe, re, fe2 = LogisticFixedEffectModel(), LogisticRandomEffectModel(verbose=False), LogisticFixedEffectModel()
    fe.fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    re.fit(d, y_var="y", x_vars=["x1", "x2"], provider_var="provider_id")
    fe2.fit(d, y_var="y2", x_vars=["x1"], provider_var="provider_id")
    exact = P.ProviderProfile.from_model(fe, test_method="poibin_exact", limits=True)
    iur = BootstrapIUR(n_boot=50).fit(d["y"].to_numpy(float), np.full(len(d), d["y"].mean()), d["provider_id"].to_numpy())
    measures = P.ProfileCollection({"Measure A": P.ProviderProfile.from_model(fe, test_method="wald"),
                                    "Measure B": P.ProviderProfile.from_model(fe2, test_method="wald")})
    makers = {
        "funnel": lambda: P.funnel(fe), "caterpillar": lambda: P.caterpillar(exact),
        "observed_expected": lambda: P.observed_expected(exact), "forest": lambda: P.forest(fe),
        "data_quality": lambda: P.data_quality(exact),
        "null_calibration": lambda: P.null_calibration(fe, test_method="score", null_model=EmpiricalNull.fitter(
            size=fe.provider_sizes_, n_groups=3)),
        "reliability": lambda: P.reliability(iur), "provider_variation": lambda: P.provider_variation(re),
        "shrinkage": lambda: P.shrinkage(fe, re), "flag_stability": lambda: P.flag_stability(fe, test_method="score"),
        "multi_measure": lambda: P.multi_measure(measures), "measure_agreement": lambda: P.measure_agreement(measures),
    }
    for name in ITEMS:
        yield name, makers[name]()


def _write(path: Path, data: bytes) -> None:
    if not path.exists() or path.read_bytes() != data:
        path.write_bytes(data)


def _alt(text: str) -> str:
    return text.replace("[", "(").replace("]", ")")


def build(app) -> None:
    src = Path(app.srcdir)
    images = src / "_static" / "gallery"
    images.mkdir(parents=True, exist_ok=True)
    lines = []
    logging.disable(logging.WARNING)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, fig in figures():
                _write(images / f"{name}.svg", fig.to_bytes("svg"))
                title, question, call = ITEMS[name]
                lines += [f"(gallery-{name.replace('_', '-')})=", f"## {title}", "", f"*Answers:* {question}", "",
                          f"![{_alt(fig.alt_text)}](/_static/gallery/{name}.svg)", "", "```{code-block} python",
                          call, "```", ""]
    finally:
        logging.disable(logging.NOTSET)
    _write(src / "presentation" / "_gallery_items.md", "\n".join(lines).encode("utf-8"))


def setup(app):
    app.connect("builder-inited", build)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
