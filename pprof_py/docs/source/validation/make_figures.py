"""Regenerate the static figures of the validation pages from the JSON files in ``validation/data/``.

    python pprof_py/docs/source/validation/make_figures.py

Writes SVG files into ``_static/validation/``, plus two self-contained interactive pages (``ladder.html``,
``explorer.html``: inline CSS/JS from ``interactive_assets/``, the JSON embedded) that the validation pages embed in a
collapsed ``<details>`` under the static figures. The JSON files are copies of the run records kept in the
pprof-validation repository (https://github.com/kevintan701/pprof-validation); nothing here reads live data.
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


def ladder():
    """One dot per (measure, component): the largest |R - Py|, on a log axis with precision bands."""
    cdr = json.loads((DATA / "cdr_parity.json").read_text())
    rep = json.loads((DATA / "logistic_fe_report.json").read_text())
    rows, fam = [], {}
    for m in cdr["measures"]:
        if m["status"] != "pass":
            continue
        f = m["family"].lower()
        fam[m["measure"]] = ORANGE if "cox" in f else AQUA if "three" in f else BLUE
        for key, lab in [("beta_maxdiff", "β"), ("gamma_maxdiff", "γ"), ("rate_maxdiff", "rate"), ("z_maxdiff", "z"), ("srr_maxdiff", "SRR"), ("ratio_maxdiff", "ratio")]:
            if m.get(key):
                rows.append((m["measure"], lab, m[key]))
    fam["Live report"] = BLUE
    for s in rep["summary"]:
        if s["value"] and s["tolerance"] not in (None, "=1") and "degenerate" not in s["quantity"]:
            lab = s["quantity"].split(" (")[0].replace("test ", "").replace(": p-value max|diff|", " p")
            lab = {"beta": "β", "gamma": "γ", "indirect ratio": "ind. ratio", "indirect rate": "ind. rate", "direct ratio": "dir. ratio",
                   "direct rate": "dir. rate", "wald p": "Wald p", "score p": "score p", "poibin_exact p": "exact p"}.get(lab, lab)
            rows.append(("Live report", lab, s["value"]))
    groups = list(dict.fromkeys(r[0] for r in rows))
    fig, ax = plt.subplots(figsize=(8.2, 0.42 * len(groups) + 1.3))
    bands = [(-17, -10, "#e3f3e8", "floating-point limit"), (-10, -5, "#edf5ee", "library / order of operations"), (-5, -3, "#fff7df", "tolerance / rounding"), (-3, 0, "#fde8e6", "investigate")]
    for lo, hi, col, lab in bands:
        ax.axvspan(lo, hi, color=col, lw=0, zorder=0)
        ax.text((lo + hi) / 2, len(groups) - .35, lab, ha="center", va="bottom", fontsize=8, color=MUTED)
    for gi, g in enumerate(groups):
        y = len(groups) - 1 - gi
        ax.axhline(y, color="#e9ecef", lw=.7, zorder=1)
        items = sorted((r for r in rows if r[0] == g), key=lambda r: r[2])
        last, flip = -99, False
        for _, lab, v in items:
            x = np.log10(v)
            ax.scatter([x], [y], s=34, color=fam[g], edgecolor="white", lw=1.2, zorder=3)
            if x - last > .95:          # room for a label on the same side as the previous one
                flip = False
            elif x - last > .45:        # close: alternate above / below
                flip = not flip
            else:                       # too close: dot only (the interactive version has the tooltip)
                continue
            ax.text(x, y + (-.42 if flip else .22), lab, ha="center", va="center", fontsize=7.5, color=MUTED, zorder=4)
            last = x
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups[::-1], fontsize=9, color=INK)
    ax.set_xlim(-17, 0)
    ax.set_ylim(-.7, len(groups) - .2)
    ax.set_xticks(range(-16, 1, 2))
    ax.set_xticklabels([f"1e{t}" for t in range(-16, 1, 2)])
    ax.set_xlabel("max |R − Py| (log scale; smaller is better)")
    for c, lab in [(BLUE, "logistic family"), (AQUA, "three-stage"), (ORANGE, "Cox family")]:
        ax.scatter([], [], color=c, s=34, label=lab)
    ax.legend(loc="lower left", ncol=3, fontsize=8, bbox_to_anchor=(0, -.26))
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "precision_ladder.svg")
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
    fig.savefig(OUT / "logistic_fe_live.svg")
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
    fig.savefig(OUT / "benchmark.svg")
    plt.close(fig)


def flags():
    """Providers whose flag differs between R and Python, per logistic measure (real data)."""
    cdr = json.loads((DATA / "cdr_parity.json").read_text())
    ms = [m for m in cdr["measures"] if m.get("flags_total")]
    fig, ax = plt.subplots(figsize=(6.4, 1.1 + .5 * len(ms)))
    names = [m["measure"] for m in ms]; mis = [m["flags_total"] - m["flags_agree"] for m in ms]
    ax.barh(names[::-1], mis[::-1], color=[("#1b7f4b" if v == 0 else BLUE) for v in mis[::-1]], height=.55)
    for i, m in enumerate(ms[::-1]):
        v = m["flags_total"] - m["flags_agree"]
        ax.text(v + .08, i, f"{v} of {m['flags_total']:,} · {100 * m['flags_agree'] / m['flags_total']:.2f}% agree".replace(".00%", "%"), va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, max(5, max(mis)) + 1.5)
    ax.set_xlabel("providers with a different flag (FY2027 production data)")
    ax.set_title("Flag agreement, logistic measures")
    ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(OUT / "flags.svg")
    plt.close(fig)


def interactive():
    """Self-contained interactive pages: the precision ladder and the per-provider explorer."""
    assets = HERE / "interactive_assets"
    css = (assets / "validation.css").read_text(encoding="utf-8")
    js = (assets / "validation.js").read_text(encoding="utf-8")
    cdr = json.loads((DATA / "cdr_parity.json").read_text())
    rep = json.loads((DATA / "logistic_fe_report.json").read_text())
    pages = {
        "ladder": {"kind": "ladder", "title": "Precision ladder", "question": "How large is the largest R–Python difference for each measure and component?",
                   "caption": "Hover a dot for its value and source. Shaded bands: below 1e−10 is the floating-point limit; 1e−10 to 1e−5 reflects library or operation-order effects; 1e−5 to 1e−3 reflects convergence tolerance or rounding of the R output; above 1e−2 would call for investigation.",
                   "cdr": cdr, "report": {"summary": rep["summary"]}},
        "explorer": {"kind": "explorer", "title": "Per-provider explorer · logistic fixed-effect live comparison", "question": "Choose a quantity: R against Python per provider, the size of each difference against the tolerance, and a Bland–Altman view; the table is linked to the charts.",
                     "caption": f"pprof_py {rep['provenance']['pprof_py_version']} (commit {rep['provenance']['pprof_py_commit']}) against pprof {rep['provenance']['pprof_r_version']} on {rep['data']['rows']:,} rows and {rep['data']['providers']} providers; run of {rep['provenance']['generated_at'][:10]}.",
                     "report": {"providers": rep["providers"], "provenance": rep["provenance"], "data": rep["data"]}},
        "benchmark": {"kind": "benchmark", "title": "Scaling benchmark · runtime of R and Python", "question": "How do fit and provider-test times grow from 20,000 to 1,000,000 rows, and where is Python faster or slower?",
                      "caption": "Four synthetic datasets, single machine (Apple Silicon, 64 GB RAM), R single-threaded at its default tolerance.",
                      "benchmark": json.loads((DATA / "benchmark.json").read_text())},
        "flags": {"kind": "flags", "title": "Flag agreement · logistic measures on production data", "question": "How many facilities does each measure classify differently in R and Python?",
                  "caption": "Hover a bar for the counts and the note recorded with the run. The remaining differences are borderline cases with p close to 0.05.", "cdr": cdr},
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
    ladder(); live_report(); benchmark(); flags(); interactive()
    print("figures written to", OUT)
