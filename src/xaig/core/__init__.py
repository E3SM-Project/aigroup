"""Durable core: the run-shaped data model, shared protocols, adapter registry.

Depends on the standard library and nothing else, and knows nothing about the
subpackages built on it.
"""

from __future__ import annotations

from xaig.core.errors import (
    AdapterError,
    MissingExtraError,
    RequestError,
    SpecError,
    XaigError,
)
from xaig.core.model import (
    RESERVED_ATTRS,
    AttrValue,
    Campaign,
    Issue,
    MetricSeries,
    Run,
    RunStatus,
    coerce_attr,
)
from xaig.core.protocols import (
    ArtifactStore,
    Discoverer,
    IdParser,
    MetricSource,
    StatusProbe,
    resolve_status,
)

__all__ = [
    "RESERVED_ATTRS",
    "AdapterError",
    "ArtifactStore",
    "AttrValue",
    "Campaign",
    "Discoverer",
    "IdParser",
    "Issue",
    "MetricSeries",
    "MetricSource",
    "MissingExtraError",
    "RequestError",
    "Run",
    "RunStatus",
    "SpecError",
    "StatusProbe",
    "XaigError",
    "coerce_attr",
    "resolve_status",
]
