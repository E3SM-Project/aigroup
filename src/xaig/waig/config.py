"""How the launcher tells the app what to open. Standard library only: the
launcher has to import on a base install."""

from __future__ import annotations

import os

LATENTS_ENV = "XAIG_WAIG_LATENTS"
SPEC_ENV = "XAIG_WAIG_SPEC"
SOURCE_ENV = "XAIG_WAIG_SOURCE"


def configured_latents() -> list[str]:
    return [p for p in os.environ.get(LATENTS_ENV, "").split(os.pathsep) if p]
