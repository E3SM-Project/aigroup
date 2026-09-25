"""How the launcher tells the app what to open. Standard library only: the
launcher has to import on a base install."""

from __future__ import annotations

import os
from pathlib import Path

LATENTS_ENV = "XAIG_WAIG_LATENTS"


def configured_latents() -> list[str]:
    """One archive per line: a URL holds the colon that would separate paths."""
    return [p for p in os.environ.get(LATENTS_ENV, "").splitlines() if p]


def discover_archives(path: str | Path) -> list[str]:
    """``path`` itself if it is a latent archive, else the archives directly inside
    it -- so one directory of models, components or experiments fills the app's
    drop-down. A path that holds none is kept: the app then says what is wrong
    with it, which a silent nothing would not. A URL (``hf://datasets/...``) is not
    a local path, and is kept as given for the adapter to open."""
    if "://" in str(path):
        return [str(path)]
    root = Path(path).resolve()
    if (root / "manifest.json").is_file() or not root.is_dir():
        return [str(root)]
    inside = sorted(str(p.parent) for p in root.glob("*/manifest.json"))
    return inside or [str(root)]
