"""xaig -- light, framework-agnostic tooling for E3SM AI campaigns.

Domains, and the presentation downstream of them:

- ``xaig.diagnostics``  what emulators hold inside: the latent space, on a grid
- ``xaig.blocks``       neural blocks trained on those latents: a sparse autoencoder
- ``xaig.figures``      figures of what ``diagnostics`` computes, with no web framework
- ``xaig.widgets``      a local web app over all of the above; nothing imports it

Everything framework-specific lives in ``xaig.adapters``. See ``AGENTS.md``.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
