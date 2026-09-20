"""Optional dependencies, asked for by name.

The base install is deliberately tiny, so anything heavier sits behind an extra
named after the subpackage that needs it. A missing one should say which extra
to install *and how, from where this xaig came from* -- not surface as a bare
``ModuleNotFoundError`` three imports deep, and not point at a package index that
a checkout was never installed from.

At module level, keep the import conventional so editors still understand it::

    try:
        import heavy
    except ImportError as exc:
        raise missing_extra("heavy", "the-extra") from exc

Inside a function, ``require`` does both steps.
"""

from __future__ import annotations

import json
from importlib import import_module
from importlib.metadata import PackageNotFoundError, distribution
from types import ModuleType
from urllib.parse import unquote, urlparse

from xaig.core.errors import MissingExtraError

_DISTRIBUTION = "xaig"


def _origin() -> dict:
    """PEP 610's record of a direct install (a directory, a VCS url), or ``{}`` for
    one that came from an index."""
    try:
        return json.loads(distribution(_DISTRIBUTION).read_text("direct_url.json") or "{}")
    except (PackageNotFoundError, ValueError):
        return {}


def install_hint(extra: str) -> str:
    """The command that adds ``extra`` to this installation of xaig."""
    origin = _origin()
    url = origin.get("url", "")
    if url.startswith("file://"):
        path = unquote(urlparse(url).path)
        editable = "-e " if origin.get("dir_info", {}).get("editable") else ""
        command = f"uv pip install {editable}'{path}[{extra}]'"
        return f"{command}  (in that checkout: `uv sync --extra {extra}`)"
    if url and ("vcs_info" in origin or "archive_info" in origin):
        target = url
        if "vcs_info" in origin:
            # The commit that is installed, not the branch it came from: a branch has
            # moved since, and asking for an extra must not quietly change the code.
            info = origin["vcs_info"]
            revision = info.get("commit_id") or info.get("requested_revision")
            target = f"{info.get('vcs', 'git')}+{url}" + (f"@{revision}" if revision else "")
        if origin.get("subdirectory"):
            target += f"#subdirectory={origin['subdirectory']}"
        return f"uv pip install '{_DISTRIBUTION}[{extra}] @ {target}'"
    return f"uv pip install '{_DISTRIBUTION}[{extra}]'"


def missing_extra(module: str, extra: str) -> MissingExtraError:
    return MissingExtraError(
        f"{module} is not installed; it comes with the {extra!r} extra: {install_hint(extra)}"
    )


def require(module: str, extra: str) -> ModuleType:
    """Import ``module``, or explain which extra provides it."""
    try:
        return import_module(module)
    except ImportError as exc:
        raise missing_extra(module, extra) from exc
