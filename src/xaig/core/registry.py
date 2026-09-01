"""Adapter lookup.

Adapters are found through the ``xaig.adapters`` entry-point group, so a future
framework can ship its own adapter as a separate distribution and be usable here
without a single edit to xaig. Built-ins are resolved lazily by import path so
that the base install never imports an adapter's heavy dependencies until asked.
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any

from xaig.core.errors import AdapterError

_BUILTIN: dict[str, str] = {"table": "xaig.adapters.table:TableDiscoverer"}
_REGISTERED: dict[str, Callable[..., Any]] = {}

_GROUP = "xaig.adapters"


def register(name: str, factory: Callable[..., Any]) -> None:
    """Register an adapter in-process. Mainly for tests and notebooks."""
    _REGISTERED[name] = factory


def unregister(name: str) -> None:
    _REGISTERED.pop(name, None)


def _load_path(path: str) -> Callable[..., Any]:
    module_name, _, attr = path.partition(":")
    try:
        return getattr(import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise AdapterError(f"could not load adapter {path!r}: {exc}") from exc


def available() -> list[str]:
    names = set(_REGISTERED) | set(_BUILTIN)
    names.update(ep.name for ep in entry_points(group=_GROUP))
    return sorted(names)


def get(name: str) -> Callable[..., Any]:
    """Resolve an adapter factory by name.

    In-process registrations win, so a test or notebook can shadow a shipped
    adapter without uninstalling anything.
    """
    if name in _REGISTERED:
        return _REGISTERED[name]
    for ep in entry_points(group=_GROUP):
        if ep.name == name:
            return ep.load()
    if name in _BUILTIN:
        return _load_path(_BUILTIN[name])
    known = ", ".join(available()) or "none"
    raise AdapterError(f"unknown adapter {name!r}; available: {known}")
