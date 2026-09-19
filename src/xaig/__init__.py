"""xaig -- light, framework-agnostic tooling for E3SM AI campaigns.

Three domains, and two layers of presentation downstream of them:

- ``xaig.caig``  campaign tracking (offline; no live server)
- ``xaig.daig``  diagnostics of emulators: their outputs and their internals
- ``xaig.taig``  reusable neural blocks and toy architectures, trained on ``daig``'s latents
- ``xaig.faig``  figures of what ``daig`` computes, with no web framework in them
- ``xaig.waig``  a local web app over all of the above; nothing imports it

Everything framework-specific lives in ``xaig.adapters``. See ``AGENTS.md``.
"""

from __future__ import annotations

from xaig.core.model import Campaign, MetricSeries, Run, RunStatus

__version__ = "0.1.0"

__all__ = ["Campaign", "MetricSeries", "Run", "RunStatus", "__version__"]
