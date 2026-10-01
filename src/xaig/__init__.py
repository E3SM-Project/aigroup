"""xaig -- light, framework-agnostic tooling for E3SM AI campaigns.

Domains, and the presentation downstream of them:

- ``xaig.latents``  what emulators hold inside: the latent space, on a grid
- ``xaig.nn``       torch modules trained on those latents: a sparse autoencoder
- ``xaig.figures``  figures of what ``latents`` computes, with no web framework
- ``xaig.app``      a local web app over all of the above; nothing imports it

Everything framework-specific lives in ``xaig.adapters``. See ``AGENTS.md``.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
