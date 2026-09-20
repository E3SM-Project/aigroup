"""Campaign tracking, the importable half.

This module returns objects. It prints nothing and formats nothing. The CLI, a
notebook and any other front end are all clients of it; it knows none of them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from xaig.caig import spec as spec_module
from xaig.caig.spec import CampaignSpec
from xaig.core import registry
from xaig.core.errors import AdapterError, SpecError
from xaig.core.model import Campaign, Run, RunStatus
from xaig.core.protocols import Discoverer, MetricSource, StatusProbe


def resolve_spec(spec: str | Path | CampaignSpec) -> CampaignSpec:
    return spec if isinstance(spec, CampaignSpec) else spec_module.load(spec)


def load_campaign(
    spec: str | Path | CampaignSpec,
    source: str | Path | None = None,
    adapter: str | None = None,
    metrics: Iterable[str] = (),
    **options: Any,
) -> Campaign:
    """Discover every run described by ``spec``.

    ``source`` is whatever the adapter reads -- a table, a directory, a URL. The
    spec's ``discovery`` block names the adapter and supplies its defaults
    (including a default ``source``); keyword arguments override it. Naming a
    different ``adapter`` drops those defaults, since they were written for the
    spec's own.

    An adapter is one object that may do more than discover: when it is also a
    ``StatusProbe`` it settles the runs whose status discovery left unknown, and
    when it is a ``MetricSource`` the series named in ``metrics`` are attached.
    """
    resolved = resolve_spec(spec)
    discovery = dict(resolved.discovery)
    declared = discovery.pop("adapter", None)
    name = adapter or declared
    if not name:
        raise AdapterError(f"spec {resolved.name!r} names no adapter and none was given")
    if name != declared:
        discovery = {}
    default_source = discovery.pop("source", None)
    chosen = source if source is not None else default_source

    built = registry.create(
        name,
        source=None if chosen is None else str(chosen),
        options={**discovery, **options},
        context={"spec": resolved},
    )
    if not isinstance(built, Discoverer):
        raise AdapterError(f"adapter {name!r} does not implement Discoverer")

    runs: list[Run] = list(built.discover())
    if isinstance(built, StatusProbe):
        runs = [
            replace(r, status=built.probe(r)) if r.status is RunStatus.UNKNOWN else r for r in runs
        ]
    wanted = tuple(metrics)
    if wanted and isinstance(built, MetricSource):
        runs = [replace(r, metrics={**r.metrics, **built.metrics(r, wanted)}) for r in runs]
    return Campaign(name=resolved.name, runs=tuple(runs))


@dataclass(frozen=True, slots=True)
class Finding:
    """One thing ``check_campaign`` found wrong with one run."""

    run_id: str
    kind: str
    message: str


def check_campaign(campaign: Campaign, spec: str | Path | CampaignSpec) -> list[Finding]:
    """Everything about a campaign that would mislabel a later table or plot.

    Two questions, reported apart because they have different fixes:

    - ``id``: does every run id fit the spec's grammar, round-trip through it
      byte for byte, and occur once? A silent disagreement between an id and the
      factors it claims is the worst failure a campaign can have, because every
      table and plot is labelled by the id.
    - ``metadata`` / ``status``: did the source say something about a run that
      contradicts its id, or that could not be interpreted? These are the issues
      an adapter recorded instead of raising.
    """
    resolved = resolve_spec(spec)
    findings: list[Finding] = []
    counts = Counter(run.id for run in campaign)
    seen: set[str] = set()
    for run in campaign:
        if run.id in seen:
            continue
        seen.add(run.id)
        if counts[run.id] > 1:
            findings.append(Finding(run.id, "id", f"appears {counts[run.id]} times"))
        if resolved.id_pattern:
            findings.extend(_check_id(run.id, resolved))
    for run in campaign:
        findings.extend(Finding(run.id, i.kind, i.message) for i in run.issues if i.kind != "id")
    return findings


def _check_id(run_id: str, spec: CampaignSpec) -> list[Finding]:
    try:
        attrs = spec.parse_id(run_id)
    except SpecError as exc:
        return [Finding(run_id, "id", str(exc))]
    if not spec.id_template:
        return []
    try:
        rebuilt = spec.format_id(attrs)
    except SpecError as exc:
        return [Finding(run_id, "id", str(exc))]
    return [] if rebuilt == run_id else [Finding(run_id, "id", f"round-trips to {rebuilt!r}")]
