"""xaig -- light, framework-agnostic tooling for E3SM AI campaigns.

Domains, and the presentation downstream of them:

- ``xaig.caig``  campaign tracking (offline; no live server)

Everything framework-specific lives in ``xaig.adapters``. See ``AGENTS.md``.
"""

from __future__ import annotations

from xaig.core.model import Campaign, MetricSeries, Run, RunStatus

__version__ = "0.1.0"

__all__ = ["Campaign", "MetricSeries", "Run", "RunStatus", "__version__"]
