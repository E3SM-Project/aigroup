"""Architecture tests.

These guard the one property that makes the package survive a change of
framework. They are cheap and unglamorous, and without them the boundary erodes
in a month.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "xaig"
CORE_ALLOWED = set(sys.stdlib_module_names) | {"yaml", "xaig"}
SUBPACKAGES = ("caig", "daig", "taig", "adapters")


def _modules(path: Path) -> list[Path]:
    return sorted(p for p in path.rglob("*.py"))


def _imported_roots(path: Path) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _imported_full(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("module", _modules(SRC / "core"), ids=lambda p: p.name)
def test_core_imports_only_stdlib_and_yaml(module: Path) -> None:
    """core/ is the durable half; a heavy import here would make the base tier heavy."""
    forbidden = _imported_roots(module) - CORE_ALLOWED
    assert not forbidden, f"{module.name} imports {sorted(forbidden)} outside stdlib+pyyaml"


@pytest.mark.parametrize("module", _modules(SRC / "core"), ids=lambda p: p.name)
def test_core_does_not_import_its_consumers(module: Path) -> None:
    """core must not know that caig, daig, taig or any adapter exists."""
    bad = {n for n in _imported_full(module) if any(n.startswith(f"xaig.{s}") for s in SUBPACKAGES)}
    assert not bad, f"{module.name} imports consumer module(s) {sorted(bad)}"


API_MODULES = [p for p in _modules(SRC) if p.name == "api.py"]
CLI_MODULES = [p for p in _modules(SRC) if p.name == "cli.py"]


@pytest.mark.parametrize("module", API_MODULES, ids=lambda p: str(p))
def test_api_modules_are_click_free(module: Path) -> None:
    """The API is a peer of the CLI, not a thing built on top of it."""
    assert "click" not in _imported_roots(module), f"{module} imports click"


@pytest.mark.parametrize("module", CLI_MODULES, ids=lambda p: str(p))
def test_cli_modules_do_not_reach_past_their_api(module: Path) -> None:
    """A CLI formats and parses; it must not talk to adapters directly."""
    bad = {n for n in _imported_full(module) if n.startswith("xaig.adapters")}
    assert not bad, f"{module} imports {sorted(bad)}; go through the api instead"


def test_taig_is_isolated() -> None:
    """Toys and diagnostics are siblings; neither may depend on the other."""
    for name in ("taig", "daig"):
        for module in _modules(SRC / name):
            others = {"caig", "daig", "taig"} - {name}
            bad = {
                n
                for n in _imported_full(module)
                if any(n.startswith(f"xaig.{o}") for o in others)
            }
            assert not bad, f"{module} imports sibling subpackage(s) {sorted(bad)}"
