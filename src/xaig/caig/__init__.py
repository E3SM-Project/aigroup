"""caig -- campaign tracking. Offline by design: no server, no live streaming."""

from __future__ import annotations

from xaig.caig.api import Finding, check_campaign, load_campaign, resolve_spec
from xaig.caig.spec import CampaignSpec, Factor
from xaig.caig.spec import bundled_names as bundled_specs
from xaig.caig.spec import load as load_spec

__all__ = [
    "CampaignSpec",
    "Factor",
    "Finding",
    "bundled_specs",
    "check_campaign",
    "load_campaign",
    "load_spec",
    "resolve_spec",
]
