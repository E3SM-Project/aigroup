"""xaig -- light, framework-agnostic tooling for E3SM AI campaigns.

Domains, and the presentation downstream of them:

- ``xaig.daig``  diagnostics of emulators' internals: the latent space, on a grid
- ``xaig.taig``  neural blocks trained on ``daig``'s latents: a sparse autoencoder
- ``xaig.faig``  figures of what ``daig`` computes, with no web framework in them
- ``xaig.waig``  a local web app over all of the above; nothing imports it

Everything framework-specific lives in ``xaig.adapters``. See ``AGENTS.md``.
"""

from __future__ import annotations

__version__ = "0.3.0"

__all__ = ["__version__"]
