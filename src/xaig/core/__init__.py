"""Durable core: data model, protocols, specs, adapter registry.

Depends on the standard library and PyYAML, and on nothing else. Ever.
"""

from __future__ import annotations

from xaig.core.errors import AdapterError, SpecError, XaigError
from xaig.core.model import AttrValue, Campaign, MetricSeries, Run, RunStatus
from xaig.core.protocols import (
    ArtifactStore,
    Discoverer,
    MetricSource,
    StatusProbe,
    resolve_status,
)
from xaig.core.spec import CampaignSpec, Factor

__all__ = [
    "AdapterError",
    "ArtifactStore",
    "AttrValue",
    "Campaign",
    "CampaignSpec",
    "Discoverer",
    "Factor",
    "MetricSeries",
    "MetricSource",
    "Run",
    "RunStatus",
    "SpecError",
    "StatusProbe",
    "XaigError",
    "resolve_status",
]
