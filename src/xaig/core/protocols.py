"""The four protocols that separate durable code from disposable code.

Everything framework-specific -- file layouts, log formats, schedulers, metric
storage -- is reached only through these. Core calls them; adapters implement
them. Adding support for a profoundly different system means writing a new
adapter module, never editing core.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Protocol, runtime_checkable

from xaig.core.model import MetricSeries, Run, RunStatus


@runtime_checkable
class Discoverer(Protocol):
    """Finds runs. The source may be a directory, a table, a database, an API."""

    def discover(self) -> Iterator[Run]: ...


@runtime_checkable
class StatusProbe(Protocol):
    """Decides a run's lifecycle state.

    Probes are consulted in order and the first confident answer wins, so a probe
    must return ``UNKNOWN`` rather than guess when it has no evidence.
    """

    def probe(self, run: Run) -> RunStatus: ...


@runtime_checkable
class MetricSource(Protocol):
    """Supplies scalar series for a run.

    Implementations must tolerate partial data: a run being written right now has
    truncated logs and missing epochs, and that is a normal reading, not an error.
    """

    def metrics(self, run: Run, names: Iterable[str] | None = None) -> dict[str, MetricSeries]: ...


@runtime_checkable
class ArtifactStore(Protocol):
    """Names the non-scalar outputs of a run (fields, checkpoints, plots).

    Returns handles, not open files: core never opens an artifact, because
    knowing how to open one is exactly the knowledge that does not survive a
    change of framework.
    """

    def artifacts(self, run: Run) -> dict[str, str]: ...


def resolve_status(run: Run, probes: Iterable[StatusProbe]) -> RunStatus:
    """First confident probe wins; ``UNKNOWN`` if none is."""
    for probe in probes:
        status = probe.probe(run)
        if status is not RunStatus.UNKNOWN:
            return status
    return RunStatus.UNKNOWN
