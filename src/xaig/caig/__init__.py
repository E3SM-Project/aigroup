"""caig -- campaign tracking. Offline by design: no server, no live streaming."""

from __future__ import annotations

from xaig.caig.api import check_ids, load_campaign, resolve_spec

__all__ = ["check_ids", "load_campaign", "resolve_spec"]
