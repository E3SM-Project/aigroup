"""Optional dependencies, asked for by name.

The base install is deliberately tiny, so anything heavier sits behind an extra
named after the subpackage that needs it. A missing one should say which extra
to install, not surface as a bare ``ModuleNotFoundError`` three imports deep.

At module level, keep the import conventional so editors still understand it::

    try:
        import numpy as np
    except ImportError as exc:
        raise missing_extra("numpy", "daig") from exc

Inside a function, ``require`` does both steps.
"""

from __future__ import annotations

from importlib import import_module
from types import ModuleType

from xaig.core.errors import MissingExtraError


def missing_extra(module: str, extra: str) -> MissingExtraError:
    return MissingExtraError(
        f"{module} is not installed; it comes with the {extra!r} extra: "
        f"uv pip install 'xaig[{extra}]'"
    )


def require(module: str, extra: str) -> ModuleType:
    """Import ``module``, or explain which extra provides it."""
    try:
        return import_module(module)
    except ImportError as exc:
        raise missing_extra(module, extra) from exc
