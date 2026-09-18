"""Architecture tests.

These guard the properties that let the package survive a change of framework
and keep growing without tangling. They are cheap and unglamorous, and without
them the boundaries erode in a month.

The rules live in the two tables below, which double as the documentation of who
may depend on whom. Adding a subpackage without declaring its edges fails
``test_every_unit_declares_its_dependencies``, so the decision is always made on
purpose.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "xaig"

# unit -> the other xaig units it may import. A unit is a subpackage or a
# top-level module; every unit may import itself and the bare ``xaig`` package.
# Domains never import adapters: they reach them through ``core.registry``.
ALLOWED: dict[str, set[str]] = {
    "__init__": {"core"},
    "_cli": {"core"},  # subcommands are named by string and loaded lazily
    "_render": set(),
    "core": set(),
    "adapters": {"core"},
    "caig": {"core", "_render"},
    "daig": {"core", "_render"},
    "taig": {"core"},
}

# unit -> third-party roots it may import. Units absent from this table are not
# limited (an adapter's whole job is to import a framework).
THIRD_PARTY: dict[str, set[str]] = {
    "__init__": set(),
    "_cli": {"click"},
    "_render": set(),
    "core": set(),
    "caig": {"click", "yaml"},
}

# The entry-point group shares the adapters package's name; it is an identifier,
# not an import path.
CORE_STRING_EXCEPTIONS = {"xaig.adapters"}

HEAVY = ("numpy", "xarray", "netCDF4", "torch", "matplotlib", "pandas", "scipy", "streamlit")


def _units() -> dict[str, list[Path]]:
    units: dict[str, list[Path]] = {}
    for entry in sorted(SRC.iterdir()):
        if entry.is_dir() and (entry / "__init__.py").exists():
            units[entry.name] = sorted(entry.rglob("*.py"))
        elif entry.suffix == ".py":
            units[entry.stem] = [entry]
    return units


UNITS = _units()
MODULES = [(unit, path) for unit, paths in UNITS.items() for path in paths]
_ids = [str(path.relative_to(SRC)) for _, path in MODULES]


def _imports(path: Path) -> set[str]:
    """Every module a file imports, as an absolute dotted name. Relative imports
    are resolved rather than skipped, and ``from xaig import _cli`` counts as
    importing ``xaig._cli``."""
    package = list(path.relative_to(SRC.parent).parts[:-1])
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def _xaig_units(names: set[str]) -> set[str]:
    out = set()
    for name in names:
        parts = name.split(".")
        if parts[0] == "xaig" and len(parts) > 1 and parts[1] in UNITS:
            out.add(parts[1])
    return out


def _third_party(names: set[str]) -> set[str]:
    roots = {n.split(".")[0] for n in names if n}
    return roots - set(sys.stdlib_module_names) - {"xaig"}


def test_every_unit_declares_its_dependencies() -> None:
    assert set(UNITS) == set(ALLOWED), (
        "src/xaig and the ALLOWED table disagree; declare what a new unit may import: "
        f"{sorted(set(UNITS) ^ set(ALLOWED))}"
    )


@pytest.mark.parametrize(("unit", "module"), MODULES, ids=_ids)
def test_units_import_only_what_they_declared(unit: str, module: Path) -> None:
    bad = _xaig_units(_imports(module)) - ALLOWED[unit] - {unit}
    assert not bad, f"{module.relative_to(SRC)} imports xaig.{sorted(bad)}; {unit} may not"


_CEILINGED = [m for m in MODULES if m[0] in THIRD_PARTY]


@pytest.mark.parametrize(
    ("unit", "module"), _CEILINGED, ids=[str(p.relative_to(SRC)) for _, p in _CEILINGED]
)
def test_third_party_ceilings(unit: str, module: Path) -> None:
    """core is stdlib-only and the base tier is pyyaml + click; nothing heavier."""
    bad = _third_party(_imports(module)) - THIRD_PARTY[unit]
    assert not bad, f"{module.relative_to(SRC)} imports {sorted(bad)}, above {unit}'s ceiling"


@pytest.mark.parametrize("module", UNITS["core"], ids=lambda p: p.name)
def test_core_does_not_name_its_consumers_in_strings(module: Path) -> None:
    """An import hidden in a string (``resources.files("xaig.caig...")``) is still a
    dependency, and one the import checks above cannot see."""
    consumers = tuple(f"xaig.{u}" for u in UNITS if u != "core")
    bad = {
        node.value
        for node in ast.walk(ast.parse(module.read_text()))
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in CORE_STRING_EXCEPTIONS
        and any(node.value == c or node.value.startswith((c + ".", c + ":")) for c in consumers)
    }
    assert not bad, f"{module.name} names consumer module(s) {sorted(bad)} in a string"


API_MODULES = [p for _, p in MODULES if p.name == "api.py"]


@pytest.mark.parametrize("module", API_MODULES, ids=lambda p: str(p.relative_to(SRC)))
def test_api_modules_are_click_free(module: Path) -> None:
    """The CLI is a client of the API; the API knows no front end."""
    assert "click" not in _third_party(_imports(module)), f"{module} imports click"


def test_the_checks_see_relative_and_from_package_imports(tmp_path: Path, monkeypatch) -> None:
    """The earlier version of this file skipped both, so neither rule bit."""
    fake = tmp_path / "xaig" / "core"
    fake.mkdir(parents=True)
    module = fake / "leak.py"
    module.write_text("from ..caig import api\nfrom . import model\nfrom xaig import _cli\n")
    monkeypatch.setattr(sys.modules[__name__], "SRC", tmp_path / "xaig")
    assert _xaig_units(_imports(module)) == {"caig", "core", "_cli"}


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_core_imports_no_consumer_at_runtime() -> None:
    result = _run(
        "import sys, pkgutil, importlib, xaig.core\n"
        "for m in pkgutil.iter_modules(xaig.core.__path__):\n"
        "    importlib.import_module('xaig.core.' + m.name)\n"
        "leaked = sorted(m for m in sys.modules\n"
        "                if m.startswith('xaig.') and m.split('.')[1] != 'core')\n"
        "assert not leaked, leaked\n"
    )
    assert result.returncode == 0, result.stderr


def test_the_base_tier_is_light() -> None:
    """Importing xaig, tracking a campaign and printing --help must not pull in
    the scientific stack -- whether or not it happens to be installed. Listing
    the commands imports every cli module, so this also holds those to the rule
    that heavy imports happen inside the command that needs them."""
    result = _run(
        "import sys\n"
        "import xaig, xaig.caig, xaig._cli\n"
        "from click.testing import CliRunner\n"
        "out = CliRunner().invoke(xaig._cli.cli, ['--help'])\n"
        "assert out.exit_code == 0, out.output\n"
        f"heavy = [m for m in {HEAVY!r} if m in sys.modules]\n"
        "assert not heavy, heavy\n"
    )
    assert result.returncode == 0, result.stderr
