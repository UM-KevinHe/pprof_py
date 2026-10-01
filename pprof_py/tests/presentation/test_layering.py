"""Import direction and import cost (ADR-001).

The statistical layer never imports presentation code or Matplotlib at module level; the model plotting mixins are
the only plotting modules it may import; presentation code never uses pyplot, print or show; and importing
``pprof_py`` loads no pyplot.
"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2]
STATISTICAL = [d for d in ("models", "algorithms", "inference", "measures", "data", "selection", "diagnostics",
                           "utils") if (PKG / d).is_dir()]
MIXINS = {"pprof_py.plotting.logistic", "pprof_py.plotting.linear"}


class _Imports(ast.NodeVisitor):
    def __init__(self):
        self.found, self.depth = [], 0

    def visit_FunctionDef(self, node):
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_If(self, node):
        t = node.test
        if (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (isinstance(t, ast.Attribute)
                                                                     and t.attr == "TYPE_CHECKING"):
            for n in node.orelse:
                self.visit(n)
            return
        self.generic_visit(node)

    def visit_Import(self, node):
        self.found.append((node, self.depth))

    visit_ImportFrom = visit_Import


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    v = _Imports()
    v.visit(tree)
    package = list(path.relative_to(PKG.parent).with_suffix("").parts)[:-1]
    out = []
    for node, depth in v.found:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        else:
            base = package[: len(package) - node.level + 1] if node.level else []
            mod = ".".join(base + ([node.module] if node.module else []))
            names = [mod] if node.module else [f"{mod}.{a.name}" for a in node.names]
        out += [(name, depth, node.lineno) for name in names]
    return tree, out


def _files(*subdirs):
    for sub in subdirs:
        yield from sorted((PKG / sub).rglob("*.py"))


def test_statistical_layer_imports_no_presentation_code_at_module_level():
    assert {"models", "inference"} <= set(STATISTICAL)
    bad = []
    for path in _files(*STATISTICAL):
        for name, depth, line in _imports(path)[1]:
            if depth:
                continue
            if name.startswith("pprof_py.presentation") or name.startswith("matplotlib"):
                bad.append(f"{path.relative_to(PKG)}:{line} {name}")
            elif name.startswith("pprof_py.plotting") and name not in MIXINS:
                bad.append(f"{path.relative_to(PKG)}:{line} {name}")
    assert not bad, bad


def test_pyplot_is_never_imported_at_module_level_by_plotting_or_presentation():
    bad = [f"{p.relative_to(PKG)}:{line}" for p in _files("plotting", "presentation")
           for name, depth, line in _imports(p)[1] if name.startswith("matplotlib.pyplot") and depth == 0]
    assert not bad, bad


def test_presentation_code_never_uses_pyplot_print_or_show():
    bad = []
    for path in _files("presentation"):
        tree, imports = _imports(path)
        bad += [f"{path.relative_to(PKG)}:{line} {name}" for name, _, line in imports if "pyplot" in name]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and ((isinstance(node.func, ast.Name) and node.func.id == "print")
                                               or (isinstance(node.func, ast.Attribute) and node.func.attr == "show")):
                bad.append(f"{path.relative_to(PKG)}:{node.lineno} call")
    assert not bad, bad


@pytest.mark.parametrize("module", ["pprof_py", "pprof_py.presentation"])
def test_importing_does_not_load_pyplot(module):
    code = f"import sys, {module}; print('matplotlib.pyplot' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         cwd=str(PKG.parent))
    assert out.stdout.strip() == "False"
