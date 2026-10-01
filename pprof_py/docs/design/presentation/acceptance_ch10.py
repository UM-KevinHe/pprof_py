"""Usage: python acceptance_ch10.py [REPO_ROOT] [OUTPUT_DIR]

Acceptance (spec §16): survival Chapter 10's report rebuilt with pprof_py.presentation, compared with the chapter.

Runs the chapters' own code (00-05, then 10) in one namespace, as a reader does, then builds the report from one
CoxPH.test() call: a profile with funnel limits, a provider table, a funnel and an interval plot.
"""
import contextlib
import io
import logging
import re
import sys
import warnings

warnings.simplefilter("ignore")
logging.disable(logging.CRITICAL)
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pathlib import Path  # noqa: E402

TREE = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).resolve().parents[4])
OUT = sys.argv[2] if len(sys.argv) > 2 else "acceptance_output"
sys.path.insert(0, TREE)
import pprof_py  # noqa: E402

assert pprof_py.__file__.startswith(TREE), pprof_py.__file__
SRC = f"{TREE}/pprof_py/docs/source/survival/"
PAGES = ["00_start_here.md", "01_survival_data_foundations.md", "02_the_cox_model.md", "03_fitting_your_first_model.md",
         "04_indirect_standardization_smr_shr.md", "05_robust_and_clustered_variance.md", "10_complete_case_study.md"]
ns = {"__name__": "__chapters__"}
for page in PAGES:
    for code in re.findall(r"```python\n(.*?)```", open(SRC + page).read(), re.S):
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, page, "exec"), ns)
records, stage2, xbeta, report = ns["records"], ns["stage2"], ns["xbeta"], ns["report"]

from pprof_py.presentation import ProviderProfile, caterpillar, funnel, provider_table  # noqa: E402

# --- the new layer: one test, every output -------------------------------------------------------------------
profile = ProviderProfile.from_model(stage2, pd.DataFrame(index=records.index), limits=True,
                                     start=records["start"], stop=records["stop"], event=records["death"],
                                     provider_id=records["facility_id"], offset=xbeta)
table = provider_table(profile, p_values=True)
fig_funnel, fig_intervals = funnel(profile), caterpillar(profile, size="double")
# ---------------------------------------------------------------------------------------------------------------

import os  # noqa: E402

os.makedirs(OUT, exist_ok=True)
table.save_html(f"{OUT}/ch10_provider_table.html")
table.save_markdown(f"{OUT}/ch10_provider_table.md")
fig_funnel.save(f"{OUT}/ch10_funnel.png")
fig_funnel.save(f"{OUT}/ch10_funnel.svg")
fig_intervals.save(f"{OUT}/ch10_intervals.png")

f = profile.data.loc[report.index]
print(f"facilities: chapter {len(report)}, profile {len(profile)}; test_method {profile.provenance['test_method']}")
print("observed identical:", np.array_equal(f["observed"].to_numpy(), report["observed"].to_numpy()))
print(f"expected: max |profile - chapter| = {np.max(np.abs(f['expected'] - report['expected'])):.2e}; "
      f"totals {f['expected'].sum():.4f} vs {report['expected'].sum():.4f}")
print(f"O/E: max |profile - chapter| = {np.max(np.abs(f['estimate'] - report['SMR'])):.2e}")
print(f"mid-p p-values: max |profile - chapter| = {np.max(np.abs(f['p_value'] - report['p_value'])):.2e}")
chapter_flag = np.where(report["flag"] == "higher than expected", 1, np.where(report["flag"] == "lower than expected", -1, 0))
print("flags that differ from the chapter's interval-based flags:",
      list(report.index[chapter_flag != f["flag"].astype(int).to_numpy()]))
inconsistent = report.index[(report["p_value"] < 0.05) != (report["flag"] != "")]
print("chapter rows whose p-value and interval disagree:", list(inconsistent))
for fac in inconsistent:
    r, p = report.loc[fac], f.loc[fac]
    print(f"  facility {fac:g}: chapter p {r['p_value']:.3f}, interval {r['ci_lower']:.3f}-{r['ci_upper']:.3f}, "
          f"flag '{r['flag']}' | new p {p['p_value']:.3f}, interval {p['ci_lower']:.3f}-{p['ci_upper']:.3f}, "
          f"flag {int(p['flag'])}")
print("new layer: interval/flag disagreements", len(profile.provenance["s3_violations"]),
      "| funnel: outside limits == flagged:",
      bool((((f["funnel_estimate"] > f["funnel_upper"]) | (f["funnel_estimate"] < f["funnel_lower"]))
            == (f["flag"] != 0)).all()))
print("alt text:", fig_funnel.alt_text)
