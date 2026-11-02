"""Regenerate the static figures of the validation pages from the JSON files in ``validation/data/``.

    python pprof_py/docs/source/validation/make_figures.py

Writes SVG files into ``_static/validation/``, plus the self-contained interactive pages (``ladder.html``,
``explorer.html``, ``benchmark.html``, ``refs_repro.html``, ``clustered_tests.html``: inline CSS/JS from ``interactive_assets/``, the JSON embedded) that the
validation pages show above the static figures. The JSON files are the run records of the live comparison and the
scaling benchmark, both on synthetic data; nothing here reads live data.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parent / "_static" / "validation"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE, AQUA, AMBER, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#16202a", "#6b7680", "#e6e9ec"
plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 9.5, "axes.edgecolor": "#c9cfd5", "axes.linewidth": .8, "axes.titleweight": "semibold",
    "axes.titlesize": 10.5, "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "grid.color": GRID, "grid.linewidth": .7, "legend.frameon": False, "svg.fonttype": "none",
})
plt.rcParams["svg.hashsalt"] = "pprof_py-validation"   # fixed ids, with no date: reruns reproduce the committed SVGs


def _save(fig, name):
    """Write an SVG that reruns reproduce byte for byte: no date, fixed ids, no trailing spaces in the path data."""
    path = OUT / name
    fig.savefig(path, metadata={"Date": None})
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def ladder():
    """One dot per (comparison, component): the largest |R - Py|, on a log axis with precision bands."""
    rep = json.loads((DATA / "logistic_fe_report.json").read_text())
    bench = json.loads((DATA / "benchmark.json").read_text())
    rows, fam = [], {}
    live = f"Example data\n{rep['data']['rows']:,} rows · {rep['data']['providers']} providers"
    fam[live] = BLUE
    for s in rep["summary"]:
        if s["value"] and s["tolerance"] not in (None, "=1") and "degenerate" not in s["quantity"]:
            lab = s["quantity"].split(" (")[0].replace("test ", "").replace(": p-value max|diff|", " p")
            lab = {"beta": "β", "gamma": "γ", "indirect ratio": "ind. ratio", "indirect rate": "ind. rate", "direct ratio": "dir. ratio",
                   "direct rate": "dir. rate", "wald p": "Wald p", "score p": "score p", "poibin_exact p": "exact p"}.get(lab, lab)
            rows.append((live, lab, s["value"]))
    for b in bench:
        g = f"Benchmark\n{b['rows']:,} rows · {b['providers']:,} providers"
        fam[g] = AQUA
        for key, lab in [("beta_maxdiff", "β"), ("gamma_maxdiff", "γ")]:
            if b["agreement"].get(key):
                rows.append((g, lab, b["agreement"][key]))
        for key, lab in [("beta_maxdiff", "β (1e−8)"), ("gamma_maxdiff", "γ (1e−8)")]:   # optional rerun
            if b.get("agreement_r_tol_1e-8", {}).get(key):
                rows.append((g, lab, b["agreement_r_tol_1e-8"][key]))
    groups = list(dict.fromkeys(r[0] for r in rows))
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(groups) + 1.7))
    bands = [(-17, -10, "#e3f3e8", "floating-point limit"), (-10, -5, "#edf5ee", "library / operation order"), (-5, -3, "#fff7df", "tol. / rounding"), (-3, 0, "#fde8e6", "explain")]
    for lo, hi, col, lab in bands:
        ax.axvspan(lo, hi, color=col, lw=0, zorder=0)
        ax.text((lo + hi) / 2, len(groups) - .35, lab, ha="center", va="bottom", fontsize=7, color=MUTED)
    for gi, g in enumerate(groups):
        y = len(groups) - 1 - gi
        ax.axhline(y, color="#e9ecef", lw=.7, zorder=1)
        items = sorted((r for r in rows if r[0] == g), key=lambda r: r[2])
        last, flip = -99, False
        for _, lab, v in items:
            x = np.log10(v)
            ax.scatter([x], [y], s=34, color=(ORANGE if "1e−8" in lab else fam[g]), edgecolor="white", lw=1.2, zorder=3)
            if x - last > 1.6:          # room for a label on the same side as the previous one
                flip = False
            elif x - last > .45:        # close: alternate above / below
                flip = not flip
            else:                       # too close: dot only (the interactive version has the tooltip)
                continue
            ax.text(x, y + (-.42 if flip else .22), lab, ha="center", va="center", fontsize=7.5, color=MUTED, zorder=4)
            last = x
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups[::-1], fontsize=8.5, color=INK, linespacing=1.4)
    ax.set_xlim(-17, 0)
    ax.set_ylim(-.7, len(groups) - .2)
    ax.set_xticks(range(-16, 1, 2))
    ax.set_xticklabels([f"1e{t}" for t in range(-16, 1, 2)])
    ax.set_xlabel("max |R − Py| (log scale; smaller is better)")
    for c, lab in [(BLUE, "live comparison, pprof::ExampleDataBinary"), (AQUA, "scaling benchmark, R at its default tol = 1e−5"), (ORANGE, "benchmark, R rerun at tol = 1e−8")]:
        ax.scatter([], [], color=c, s=34, label=lab)
    ax.legend(loc="lower left", ncol=2, fontsize=7.5, bbox_to_anchor=(0, -.32))
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "precision_ladder.svg")
    plt.close(fig)


def live_report():
    """Per-provider γ: R against Python, and the size of the difference against the tolerance."""
    rep = json.loads((DATA / "logistic_fe_report.json").read_text())
    P = rep["providers"]
    reg = [p for p in P if not p["degenerate"]]
    deg = [p for p in P if p["degenerate"]]
    fig, (a, b) = plt.subplots(1, 2, figsize=(8.2, 3.6), gridspec_kw={"width_ratios": [1.1, 1]})
    xs = [p["gamma_r"] for p in reg]; ys = [p["gamma_py"] for p in reg]
    lo, hi = min(xs + ys + [p["gamma_r"] for p in deg] + [p["gamma_py"] for p in deg]), max(xs + ys)
    a.plot([lo, hi], [lo, hi], ls="--", color="#9aa3ab", lw=1, zorder=1)
    a.scatter(xs, ys, s=22, color=BLUE, edgecolor="white", lw=.8, zorder=3, label="provider with events")
    a.scatter([p["gamma_r"] for p in deg], [p["gamma_py"] for p in deg], s=26, color=AMBER, edgecolor="white", lw=.8, zorder=3, label="zero-event provider")
    a.set_xlabel("γ̂ · R (pprof 1.0.3)"); a.set_ylabel("γ̂ · Python (pprof_py 0.7.0)")
    a.set_title("Provider effects, R against Python")
    a.legend(loc="upper left", fontsize=8)
    a.grid(True)
    quantities = [("gamma", "γ", 1e-5), ("indirect_ratio", "indirect ratio", 1e-5), ("direct_ratio", "direct ratio", 1e-5), ("wald_p", "Wald p", 1e-3), ("score_p", "score p", 1e-3), ("poibin_exact_p", "exact p", 1e-3)]
    for i, (k, lab, tol) in enumerate(quantities):
        d = [abs(p[k + "_r"] - p[k + "_py"]) for p in reg]
        d = [np.log10(v) if v > 0 else -16 for v in d]
        y = len(quantities) - 1 - i
        b.scatter(d, y + np.random.default_rng(i).uniform(-.18, .18, len(d)), s=12, color=BLUE, alpha=.7, lw=0, zorder=3)
        b.plot([np.log10(tol)] * 2, [y - .4, y + .4], color=ORANGE, lw=1.4, zorder=4)
    b.set_yticks(range(len(quantities))); b.set_yticklabels([q[1] for q in quantities][::-1], color=INK)
    b.set_xlim(-16.5, -1.5); b.set_xticks([-16, -12, -8, -4, -2]); b.set_xticklabels(["0", "1e−12", "1e−8", "1e−4", "1e−2"])
    b.set_xlabel("|R − Py| per provider (log scale); orange = tolerance")
    b.set_title("Differences against the tolerance")
    b.grid(True, axis="x")
    fig.tight_layout()
    _save(fig, "logistic_fe_live.svg")
    plt.close(fig)


def benchmark():
    """Runtime against data size, both axes logarithmic."""
    B = json.loads((DATA / "benchmark.json").read_text())
    rows = [r["rows"] for r in B]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    series = [("Fit · R (C++)", [r["r"]["fit"] for r in B], ORANGE, "-"), ("Fit · Python (NumPy)", [r["py"]["fit"] for r in B], BLUE, "-"),
              ("Exact test · R", [r["r"]["test_exact.poisbinom"] for r in B], ORANGE, "--"), ("Exact test · Python", [r["py"]["test_poibin_exact"] for r in B], BLUE, "--")]
    for lab, ys, c, ls in series:
        ax.plot(rows, ys, ls=ls, color=c, lw=2, marker="o", ms=5, mec="white", mew=1, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("rows (200 to 7,000 providers)"); ax.set_ylabel("seconds")
    ax.set_title("Logistic fixed effect: runtime against data size")
    ax.grid(True, which="major")
    ax.legend(fontsize=8, loc="upper left")
    for r in B:
        ax.annotate(f"{r['providers']:,} prov.", (r["rows"], r["py"]["fit"]), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    _save(fig, "benchmark.svg")
    plt.close(fig)


def refs_repro():
    """One dot per regenerated reference file: the largest |committed - regenerated|, grouped by test family."""
    P = json.loads((DATA / "frozen_refs_repro.json").read_text())
    groups = ["CoxPH (survival)", "phase 4", "selection", "penalized (glmnet)", "pprofR clustered tests"]
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(groups) + 1.7))
    bands = [(-17, -10, "#e3f3e8", "floating-point limit"), (-10, -5, "#edf5ee", "library / operation order"), (-5, -3, "#fff7df", "tol. / rounding"), (-3, 1, "#fde8e6", "explain")]
    for lo, hi, col, lab in bands:
        ax.axvspan(lo, hi, color=col, lw=0, zorder=0)
        ax.text((lo + hi) / 2, len(groups) - .35, lab, ha="center", va="bottom", fontsize=7, color=MUTED)
    for gi, g in enumerate(groups):
        y = len(groups) - 1 - gi
        ax.axhline(y, color="#e9ecef", lw=.7, zorder=1)
        for i, f in enumerate(x for x in P["files"] if x["group"] == g):
            x = -17 if f["identical"] or f["max_abs"] <= 0 else max(-17, np.log10(f["max_abs"]))
            material = (not f["identical"]) and f["max_rel"] > 1e-6 and f["max_abs"] > 1e-6
            ax.scatter([x], [y + ((i % 5) - 2) * .06], s=22, color=AQUA if f["identical"] else ORANGE if material else BLUE, edgecolor="white", lw=.8, zorder=3)
            if material:
                ax.text(x, y + .28, f["file"].replace("penalized_wide_", ""), ha="center", fontsize=6.5, color=ORANGE)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([f"{g}\n{sum(1 for f in P['files'] if f['group'] == g)} files" for g in groups[::-1]], fontsize=8.5, color=INK, linespacing=1.4)
    ax.set_xlim(-17.3, 1); ax.set_ylim(-.7, len(groups) - .2)
    ax.set_xticks([-17] + list(range(-16, 1, 2))); ax.set_xticklabels(["0"] + [f"1e{t}" for t in range(-16, 1, 2)])
    ax.set_xlabel("largest |committed − regenerated| per file (log scale; 0 = identical)")
    for c, lab in [(AQUA, "identical"), (BLUE, "last digits only"), (ORANGE, "differs materially (glmnet 4.1.8 vs 4.1.10 cross-validation)")]:
        ax.scatter([], [], color=c, s=22, label=lab)
    ax.legend(loc="lower left", ncol=3, fontsize=7.5, bbox_to_anchor=(0, -.32))
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "refs_repro.svg")
    plt.close(fig)


def clustered():
    """Patient-clustered tests against pprofR: one dot per (cohort, quantity), the largest |R - Py| over providers."""
    C = json.loads((DATA / "clustered_tests.json").read_text())
    quantities = [("gamma", "γ"), ("robust_wald_stat", "robust Wald z"), ("robust_wald_se", "robust Wald SE"), ("robust_wald_ci_lower", "robust Wald CI"),
                  ("stabilized_wald_stat", "stab. Wald z"), ("stabilized_wald_se", "stab. Wald SE"), ("robust_score_stat", "robust score z"),
                  ("stabilized_score_stat", "stab. score z"), ("score_stat", "score z"), ("robust_wald_p", "robust Wald p"), ("stabilized_wald_p", "stab. Wald p"),
                  ("robust_score_p", "robust score p"), ("stabilized_score_p", "stab. score p"), ("beta_robust_wald_beta", "β"), ("beta_robust_wald_se", "β robust SE"),
                  ("beta_robust_wald_stat", "β robust z")]
    rows, groups = [], []
    for c in C["cohorts"]:
        g = f"{c['rows']:,} rows\n{c['providers']:,} providers · {c['patients']:,} patients"
        groups.append(g)
        for run, col in (("tol_1e-5", AQUA), ("tol_1e-8", ORANGE), ("tol_1e-12_nobacktrack", BLUE)):
            a = c["r"].get(run, {}).get("agreement")
            if a:
                rows += [(g, lab, a[k], col) for k, lab in quantities if a.get(k, 0) > 0]
    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(groups) + 1.7))
    bands = [(-17, -10, "#e3f3e8", "floating-point limit"), (-10, -5, "#edf5ee", "library / operation order"), (-5, -3, "#fff7df", "tol. / rounding"), (-3, 0, "#fde8e6", "explain")]
    for lo, hi, col, lab in bands:
        ax.axvspan(lo, hi, color=col, lw=0, zorder=0)
        ax.text((lo + hi) / 2, len(groups) - .35, lab, ha="center", va="bottom", fontsize=7, color=MUTED)
    for gi, g in enumerate(groups):
        y = len(groups) - 1 - gi
        ax.axhline(y, color="#e9ecef", lw=.7, zorder=1)
        items = sorted((r for r in rows if r[0] == g), key=lambda r: r[2])
        worst = {col: max((r for r in items if r[3] == col), key=lambda r: r[2]) for col in {r[3] for r in items}}
        for _, lab, v, col in items:
            x = max(-17, np.log10(v))
            ax.scatter([x], [y], s=34, color=col, edgecolor="white", lw=1.2, zorder=3)
        # one label per R run: its largest difference, which is what the row is about
        for k, (_, lab, v, col) in enumerate(sorted(worst.values(), key=lambda r: r[2])):
            ax.text(max(-17, np.log10(v)), y + (.22 if k % 2 == 0 else -.42), lab, ha="center", va="center", fontsize=7, color=col, zorder=4)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups[::-1], fontsize=8.5, color=INK, linespacing=1.4)
    ax.set_xlim(-17, 0); ax.set_ylim(-.7, len(groups) - .2)
    ax.set_xticks(range(-16, 1, 2)); ax.set_xticklabels([f"1e{t}" for t in range(-16, 1, 2)])
    ax.set_xlabel("max |R − Py| over providers (log scale; smaller is better)")
    for col, lab in [(AQUA, "pprofR at its default tol = 1e−5"), (ORANGE, "rerun at tol = 1e−8"), (BLUE, "rerun at tol = 1e−12, no step-halving")]:
        ax.scatter([], [], color=col, s=34, label=lab)
    ax.legend(loc="lower left", ncol=3, fontsize=7.5, bbox_to_anchor=(-.28, -.32))
    ax.grid(False)
    fig.tight_layout()
    _save(fig, "clustered_tests.svg")
    plt.close(fig)


def interactive():
    """Self-contained interactive pages: the precision ladder and the per-provider explorer."""
    assets = HERE / "interactive_assets"
    css = (assets / "validation.css").read_text(encoding="utf-8")
    js = (assets / "validation.js").read_text(encoding="utf-8")
    rep = json.loads((DATA / "logistic_fe_report.json").read_text())
    bench = json.loads((DATA / "benchmark.json").read_text())
    repro = json.loads((DATA / "frozen_refs_repro.json").read_text())
    clus = json.loads((DATA / "clustered_tests.json").read_text())
    pages = {
        "refs_repro": {"kind": "repro", "title": "Regenerating the frozen R references on a second machine", "question": "Do the 117 committed R reference files come back when their scripts are rerun on another platform with newer packages, and where do they differ?",
                       "caption": "Committed: " + repro["committed"]["os"] + ", R " + repro["committed"]["R"] + ". Regenerated: " + repro["regenerated"]["os"] + ", R " + repro["regenerated"]["R"] + ", " + repro["regenerated"]["date"] + ". Shaded bands as in the precision ladder; a dot at 0 is a byte-identical file.",
                       "repro": repro},
        "ladder": {"kind": "ladder", "title": "Precision ladder", "question": "How large is the largest R–Python difference for each comparison and component?",
                   "caption": "Hover a dot for its value and source. Shaded bands: below 1e−10 is the floating-point limit; 1e−10 to 1e−5 reflects library or operation-order effects; 1e−5 to 1e−3 reflects convergence tolerance or rounding of the R output; above 1e−2 calls for an explanation. The one value there is γ at 7,000 providers under R's default tolerance (1e−5); rerun at tol = 1e−8 (orange) β and γ agree to 4e−16 and 5e−15. The live comparison keeps R at its default tolerance, so its points stop at R's stopping rule rather than at the floating-point limit.",
                   "report": {"summary": rep["summary"], "data": rep["data"]}, "benchmark": bench},
        "explorer": {"kind": "explorer", "title": "Per-provider explorer · logistic fixed-effect live comparison", "question": "Choose a quantity: R against Python per provider, the size of each difference against the tolerance, and a Bland–Altman view; the table is linked to the charts.",
                     "caption": f"pprof_py {rep['provenance']['pprof_py_version']} (commit {rep['provenance']['pprof_py_commit']}) against pprof {rep['provenance']['pprof_r_version']} on {rep['data']['rows']:,} rows and {rep['data']['providers']} providers; run of {rep['provenance']['generated_at'][:10]}.",
                     "report": {"providers": rep["providers"], "provenance": rep["provenance"], "data": rep["data"]}},
        "benchmark": {"kind": "benchmark", "title": "Scaling benchmark · runtime of R and Python", "question": "How do fit and provider-test times grow from 20,000 to 5,000,000 rows, and where is Python faster or slower?",
                      "caption": "Five synthetic datasets, 20,000 to 5,000,000 rows, single machine (Apple Silicon, 64 GB RAM), R single-threaded at its default tolerance.",
                      "benchmark": bench},
        "clustered_tests": {"kind": "clustered", "title": "Patient-clustered provider tests · live comparison against pprofR", "question": "For the four patient-clustered tests (robust and stabilized Wald, robust and stabilized score), how far apart are pprofR and pprof_py on the same cohort, from 20,000 to 5,000,000 rows?",
                            "caption": f"pprof_py {clus['provenance']['pprof_py_version']} (commit {clus['provenance']['pprof_py_commit']}) against pprofR {clus['provenance']['pprofR_version']} (commit {clus['provenance']['pprofR_commit'][:7]}) on five synthetic patient-clustered cohorts; run of {clus['provenance']['generated_at'][:10]}. Shaded bands as in the precision ladder. Hover a dot for its value; the tables below hold the flags, the times and every value.",
                            "clustered": clus},
    }
    for name, payload in pages.items():
        data = json.dumps(payload, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{payload['title']} · pprof_py validation</title>
<style>
{css}
</style>
</head>
<body>
<main id="root"></main>
<script id="pprof-data" type="application/json">{data}</script>
<script>
{js}
</script>
</body>
</html>
"""
        (OUT / f"{name}.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    ladder(); live_report(); benchmark(); refs_repro(); clustered(); interactive()
    print("figures written to", OUT)
