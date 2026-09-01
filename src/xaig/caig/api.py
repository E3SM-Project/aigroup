"""Campaign tracking, the importable half.

This module returns objects. It prints nothing and formats nothing -- the CLI is
a sibling of this API, not a layer above it, and neither may depend on the other.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from xaig.core import registry
from xaig.core import spec as spec_module
from xaig.core.errors import AdapterError
from xaig.core.model import Campaign
from xaig.core.spec import CampaignSpec


def resolve_spec(spec: str | Path | CampaignSpec) -> CampaignSpec:
    return spec if isinstance(spec, CampaignSpec) else spec_module.load(spec)


def _instantiate(factory: Any, options: Mapping[str, Any]) -> Any:
    """Build an adapter, passing only the arguments its signature accepts.

    Adapters are free to have whatever signature suits their source; this keeps
    the spec's ``discovery`` block from becoming a lowest-common-denominator.
    """
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory(**dict(options))
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        accepted = dict(options)
    else:
        accepted = {k: v for k, v in options.items() if k in params}
    missing = [
        name
        for name, p in params.items()
        if p.default is inspect.Parameter.empty
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
        and name not in accepted
    ]
    if missing:
        raise AdapterError(f"adapter is missing required option(s): {', '.join(missing)}")
    return factory(**accepted)


def load_campaign(
    spec: str | Path | CampaignSpec,
    source: str | Path | None = None,
    adapter: str | None = None,
    **overrides: Any,
) -> Campaign:
    """Discover every run described by ``spec``.

    ``source`` is whatever the adapter reads -- a table, a directory, a URL. The
    spec's ``discovery`` block supplies defaults; keyword arguments override it.
    """
    resolved = resolve_spec(spec)
    options: dict[str, Any] = dict(resolved.discovery)
    name = adapter or options.pop("adapter", None)
    options.pop("adapter", None)
    if not name:
        raise AdapterError(f"spec {resolved.name!r} names no adapter and none was given")
    if source is not None:
        options["path"] = str(source)
    options.update(overrides)
    options["spec"] = resolved

    discoverer = _instantiate(registry.get(name), options)
    if not hasattr(discoverer, "discover"):
        raise AdapterError(f"adapter {name!r} does not implement Discoverer")
    return Campaign(name=resolved.name, runs=tuple(discoverer.discover()))


def check_ids(campaign: Campaign, spec: str | Path | CampaignSpec) -> list[tuple[str, str]]:
    """Round-trip every run id through the spec; return ``(id, reason)`` for failures.

    A silent disagreement between a run id and the factors it claims is the worst
    failure a campaign can have, because every table and plot is labelled by the id.
    """
    resolved = resolve_spec(spec)
    if not (resolved.id_pattern and resolved.id_template):
        return []
    failures: list[tuple[str, str]] = []
    for run in campaign:
        try:
            attrs = resolved.parse_id(run.id)
            rebuilt = resolved.format_id(attrs)
        except Exception as exc:
            failures.append((run.id, str(exc)))
            continue
        if rebuilt != run.id:
            failures.append((run.id, f"round-trips to {rebuilt!r}"))
    return failures
