# ADR-007 — Test CI (decision D3)

**Status:** accepted (maintainer, D3) · 2026-09-30 · Delivered as `round_ci_tests.diff` (56 lines, one new file)

## Decision
`.github/workflows/tests.yml`:
* **Triggers:** push to `main`, pull requests, manual runs.
* **Runner and matrix:** ubuntu-latest; Python `"3.10"` (the declared floor after D15) and `"3.14"` (newest); `fail-fast: false`.
* **Install and run:** `pip install -e ".[dev]"`, then `python -m pytest pprof_py/tests -p no:cacheprovider -q -rfEs` with `MPLBACKEND=Agg`.
* **Caching and concurrency:** pip cache keyed on `pyproject.toml`; superseded pull-request runs are cancelled, pushes to `main` are not; 30-minute timeout.
* **Actions:** `actions/checkout@v7` and `actions/setup-python@v7`, both Node 24 per their `action.yml`.

Round R1 also raises `requires-python` to 3.10 (D15) and the numba floor to 0.57 (D16), and moves `docs_deploy.yml` to Python 3.10 with `checkout@v7`/`setup-python@v7` (Node 24); its Pages actions are unchanged.

## Evidence
* actionlint 1.7.12: no findings. YAML parses. No whitespace errors.
* `git apply --check` and `git apply` on a fresh clone of `test/mixed-effect` (394b911): exit 0, empty stderr, resulting file identical.
* Local rehearsal, full results in spec §12.4:
  * 3.10: 689 passed.
  * 3.14: passes apart from the environment-only tzdata test.
  * 3.9: one exact-equality failure (`test_penalty_factor_is_inert_under_the_pure_group_lasso[GroupLassoLinear]`, −1.06e-16 vs 0.0).
  * Declared minimums: not installable together.
  * Lowest-direct: fails `test_sigma_sensitivity`.

## Consequences
* With the 3.10 floor both jobs are expected green: the 3.9-only group-lasso failure no longer applies, and 3.10 passed the whole suite locally.
* Proposed later job, once floors are fixed (spec §17 item 8):
```yaml
      - name: Install lowest direct dependencies
        run: |
          python -m pip install "uv==0.12.*"
          uv pip install --system --resolution lowest-direct -e ".[dev]"
```
* Separately recommended: move `docs_deploy.yml` off Node 20 actions (`setup-python@v5`) and off Python 3.9.
