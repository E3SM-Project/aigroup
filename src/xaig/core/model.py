"""Framework-agnostic data model.

Nothing here knows what a training framework, a scheduler, or an output file is.
A run is an identifier plus a bag of attributes; that is the whole contract.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace

AttrValue = str | int | float | bool | None

# Addressable like attributes in filters, sorts and tables, but owned by the run
# itself -- so no attribute may take these names and quietly stand in for them.
RESERVED_ATTRS = ("id", "status")

# ASCII on purpose: ``str.isdigit`` also accepts characters such as "²" that
# ``int`` then refuses.
_INT = re.compile(r"-?[0-9]+")


def coerce_attr(text: str | None) -> AttrValue:
    """Text to an attribute value: blank is missing, whole numbers become ints so
    ``batch=16`` compares and sorts numerically, and anything else is left alone --
    ``E01`` must stay ``E01``."""
    text = (text or "").strip()
    if not text:
        return None
    return int(text) if _INT.fullmatch(text) else text


def _sort_key(value: AttrValue) -> tuple[int, float, str]:
    """Numbers numerically, then text, then missing: batch 8 sorts before 16."""
    if value is None:
        return (2, 0.0, "")
    if isinstance(value, int | float):
        return (0, float(value), "")
    return (1, 0.0, str(value))


class RunStatus(enum.Enum):
    """Lifecycle of a run.

    ``INTERRUPTED`` is deliberately distinct from ``FAILED``: a scheduler that
    preempts and requeues a job leaves it interrupted, not broken. Collapsing the
    two is the single easiest way to misread a live campaign.
    """

    PENDING = "pending"
    RUNNING = "running"
    INTERRUPTED = "interrupted"
    FINISHED = "finished"
    FAILED = "failed"
    UNKNOWN = "unknown"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class Issue:
    """Something a reader noticed about a run and chose not to raise.

    A partially-readable campaign is more useful than an exception, but only if
    what was tolerated stays visible. ``kind`` is ``"id"`` (the id does not fit
    the campaign's grammar), ``"metadata"`` (two sources disagree about the run)
    or ``"status"`` (a status could not be interpreted).
    """

    kind: str
    message: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.message}"


@dataclass(frozen=True, slots=True)
class MetricSeries:
    """An ordered scalar series. ``steps`` and ``values`` are parallel."""

    name: str
    steps: tuple[float, ...] = ()
    values: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if len(self.steps) != len(self.values):
            raise ValueError(
                f"metric {self.name!r}: {len(self.steps)} steps vs {len(self.values)} values"
            )

    def __len__(self) -> int:
        return len(self.values)

    @property
    def last(self) -> float | None:
        return self.values[-1] if self.values else None

    def at(self, step: float) -> float | None:
        """Value at exactly ``step``, or None. No interpolation: a missing epoch
        must read as missing, never as a neighbour's number."""
        for s, v in zip(self.steps, self.values, strict=True):
            if s == step:
                return v
        return None


@dataclass(frozen=True, slots=True)
class Run:
    """One run. ``location`` is an opaque adapter-owned string, not necessarily a path."""

    id: str
    attrs: dict[str, AttrValue] = field(default_factory=dict)
    status: RunStatus = RunStatus.UNKNOWN
    location: str | None = None
    metrics: dict[str, MetricSeries] = field(default_factory=dict)
    issues: tuple[Issue, ...] = ()

    def __post_init__(self) -> None:
        reserved = [k for k in RESERVED_ATTRS if k in self.attrs]
        if reserved:
            raise ValueError(
                f"run {self.id!r}: attribute name(s) {', '.join(reserved)} are reserved "
                "for the run's own id and status"
            )

    def __hash__(self) -> int:
        # Consistent with the generated __eq__ (equal runs share an id); the
        # generated hash would fail on the dict fields.
        return hash(self.id)

    def get(self, key: str, default: AttrValue = None) -> AttrValue:
        """An attribute, or the run's own ``id`` / ``status`` under those names."""
        if key == "id":
            return self.id
        if key == "status":
            return str(self.status)
        return self.attrs.get(key, default)

    def with_attrs(self, **attrs: AttrValue) -> Run:
        return replace(self, attrs={**self.attrs, **attrs})

    def matches(self, **query: AttrValue) -> bool:
        """True when every queried attribute equals the run's. Compared as strings so
        ``batch=16`` and ``batch="16"`` behave the same from CLI and API alike."""
        return all(str(self.get(k)) == str(v) for k, v in query.items())


@dataclass(frozen=True, slots=True)
class Campaign:
    """A named collection of runs."""

    name: str
    runs: tuple[Run, ...] = ()

    def __iter__(self) -> Iterator[Run]:
        return iter(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    def get(self, run_id: str) -> Run | None:
        return next((r for r in self.runs if r.id == run_id), None)

    def filter(self, **query: AttrValue) -> Campaign:
        return Campaign(self.name, tuple(r for r in self.runs if r.matches(**query)))

    def sorted_by(self, *keys: str) -> Campaign:
        def sort_key(r: Run) -> tuple[tuple[int, float, str], ...]:
            return tuple(_sort_key(r.get(k)) for k in keys)

        return Campaign(self.name, tuple(sorted(self.runs, key=sort_key)))

    def attr_names(self) -> list[str]:
        """Union of attribute names across runs, in first-seen order."""
        seen: dict[str, None] = {}
        for r in self.runs:
            seen.update(dict.fromkeys(r.attrs))
        return list(seen)

    def table(self, columns: Iterable[str] | None = None) -> list[dict[str, AttrValue]]:
        """Rows as plain dicts. Deliberately not a DataFrame: the base tier
        carries no pandas, and callers who want one can build it in a line."""
        cols = list(columns) if columns is not None else self.attr_names()
        cols = [c for c in cols if c not in RESERVED_ATTRS]
        return [
            {"id": r.id, "status": str(r.status), **{c: r.attrs.get(c) for c in cols}}
            for r in self.runs
        ]
