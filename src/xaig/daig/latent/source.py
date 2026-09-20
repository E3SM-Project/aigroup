"""Where latents come from: the ``LatentSource`` contract.

This is the seam between running a model and studying it. Recording activations
needs the model's own environment (torch, the framework, a checkpoint, often a
different Python); analysing them needs numpy. An exporter on one side writes
latents down, an adapter on the other reads them back through this protocol, and
nothing here ever imports a model.

The contract lives next to its consumers, all of which sit on ``daig``. It moves
to ``xaig.core`` when a subpackage that does not needs it, and not before.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from xaig.core import registry
from xaig.core.errors import AdapterError, RequestError
from xaig.core.extras import missing_extra
from xaig.daig.grid import Grid

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "daig") from exc

DEFAULT_ADAPTER = "latent-archive"

# "0425-01-03T18:00:00", with or without a time of day. Parsed by hand because
# the calendars below hold dates (30 February, year 425 without leap days) that
# ``datetime`` refuses.
_LABEL = re.compile(
    r"(-?\d+)-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?)?\s*"
)
_POSITION = re.compile(r"-?[0-9]+")  # ASCII on purpose, as in core.model
_MONTH_STARTS = (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334)
_LEAP_MONTH_STARTS = (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335)


def _day_number(year: int, month: int, day: int, calendar: str) -> int | None:
    """Days since an arbitrary origin, or None for a calendar not known here."""
    if not 1 <= month <= 12:
        return None
    if calendar in ("noleap", "365_day"):
        return year * 365 + _MONTH_STARTS[month - 1] + day
    if calendar in ("all_leap", "366_day"):
        return year * 366 + _LEAP_MONTH_STARTS[month - 1] + day
    if calendar == "360_day":
        return year * 360 + (month - 1) * 30 + day
    if calendar in ("standard", "gregorian", "proleptic_gregorian"):
        try:
            return date(year, month, day).toordinal()
        except ValueError:
            return None
    return None


@dataclass(frozen=True, slots=True)
class LayerInfo:
    """One place in a network where activations were recorded."""

    index: int
    label: str
    n_channels: int


@dataclass(frozen=True, slots=True)
class LatentInfo:
    """What a source holds, without loading any of it.

    Times are the source's own labels, kept as text: emulators run on calendars
    (no-leap, year 425) that the usual datetime types cannot hold.
    ``elapsed_seconds`` does the one piece of arithmetic they allow: seconds since
    the first. ``off_grid_layers`` were recorded but live on a coarser grid than
    ``grid()`` describes -- the inner levels of a U-Net, say -- so they are listed
    and not loadable here.

    ``experiment`` is whatever the exporter recorded about how this run differs
    from a plain one -- a noise seed, a perturbed input, a steered channel -- and
    ``options`` is how the adapter was told to read it (a mask variable). Both are
    free-form, JSON-ready, and carried into the provenance of every result: two
    archives of the same checkpoint are otherwise indistinguishable.
    """

    source: str
    times: tuple[str, ...]
    layers: tuple[LayerInfo, ...]
    n_nodes: int
    model: str | None = None
    component: str | None = None
    checkpoint: str | None = None
    calendar: str | None = None
    timestep_seconds: int | None = None
    off_grid_layers: tuple[LayerInfo, ...] = ()
    experiment: Mapping[str, Any] = field(default_factory=dict)
    options: Mapping[str, Any] = field(default_factory=dict)

    def provenance(self) -> dict[str, Any]:
        """What a result must carry to be traceable to the run that produced it."""
        out: dict[str, Any] = {
            "source": self.source,
            "model": self.model,
            "component": self.component,
            "checkpoint": self.checkpoint,
        }
        if self.options:
            out["options"] = dict(self.options)
        if self.experiment:
            out["experiment"] = dict(self.experiment)
        return out

    def identity(self) -> dict[str, str]:
        """Which network this is, as far as the source says: the model, component
        and checkpoint it declares. Channel 42 of one trained network is not
        channel 42 of another, so anything that lines two things up by channel
        index compares these first. What is undeclared is left out, and is not held
        against it."""
        declared = {"model": self.model, "component": self.component, "checkpoint": self.checkpoint}
        return {key: str(value) for key, value in declared.items() if value}

    @property
    def name(self) -> str:
        """A short human label: the model and component, else the directory."""
        named = " · ".join(str(x) for x in (self.model, self.component) if x)
        return named or Path(self.source).name

    def layer(self, index: int) -> LayerInfo:
        for layer in self.layers:
            if layer.index == index:
                return layer
        known = ", ".join(str(layer.index) for layer in self.layers)
        raise RequestError(f"no layer {index}; layers are {known}")

    @property
    def last_layer(self) -> int:
        return max(layer.index for layer in self.layers)

    def time_index(self, time: str | int) -> int:
        """Position of a time given by its label, or by position already."""
        if isinstance(time, int):
            if not -len(self.times) <= time < len(self.times):
                raise RequestError(
                    f"time index {time} is out of range for {len(self.times)} time(s)"
                )
            return time % len(self.times)
        if time not in self.times:
            raise RequestError(f"no latents at {time!r}; times are {', '.join(self.times)}")
        return self.times.index(time)

    def elapsed_seconds(self) -> tuple[float, ...] | None:
        """Seconds from the first time to each, under the source's calendar.

        Positions are not lead times: an exporter keeps the forward calls it was
        asked to, and the gaps between them are whatever they are. None when the
        calendar is undeclared or the labels are not dates -- nothing is guessed,
        and a caller falls back to positions knowingly.
        """
        if not self.calendar:
            return None
        stamps: list[float] = []
        for label in self.times:
            match = _LABEL.fullmatch(label)
            if match is None:
                return None
            year, month, day = (int(match.group(i)) for i in (1, 2, 3))
            days = _day_number(year, month, day, self.calendar.lower())
            if days is None:
                return None
            hour, minute = int(match.group(4) or 0), int(match.group(5) or 0)
            seconds = float(match.group(6) or 0)
            stamps.append(days * 86400.0 + hour * 3600.0 + minute * 60.0 + seconds)
        return tuple(s - stamps[0] for s in stamps)


@runtime_checkable
class LatentSource(Protocol):
    """Supplies recorded activations, one layer at one time, selectively.

    ``load`` returns float32 ``(n_nodes, n_channels)``, narrowed to ``nodes`` and
    ``channels`` when they are given. Selection is part of the contract because
    the full array rarely fits comfortably: one time of a 9-layer, 384-channel
    1-degree model is 0.9 GB, while a region of one layer is a few hundred KB. An
    implementation should read only what was asked for.

    The array returned is new and the caller's to modify: analyses centre it in
    place rather than hold a second copy.
    """

    def info(self) -> LatentInfo: ...

    def grid(self) -> Grid: ...

    def load(
        self,
        time: str | int,
        layer: int,
        channels: Sequence[int] | None = None,
        nodes: Sequence[int] | None = None,
    ) -> np.ndarray: ...


@runtime_checkable
class ReferenceFields(Protocol):
    """Physical fields recorded alongside the latents, on the same nodes.

    An optional second capability of a latent adapter, asked for with
    ``isinstance``: what the model was looking at (or produced) at each latent
    time, so a channel can be set against sea-surface temperature or a steered run
    against its control. ``field`` returns float64 ``(n_nodes,)`` at a *latent*
    time, NaN where the field is missing.

    This is not a contract for setting an emulator against a reference (levels,
    variables through time, two datasets); it is the few fields an exporter chose
    to keep next to its activations.
    """

    def field_names(self) -> tuple[str, ...]: ...

    def field(self, name: str, time: str | int) -> np.ndarray: ...


def parse_time(text: str) -> str | int:
    """A time as a person types it: a position (``0``, ``-1``) or a label."""
    return int(text) if _POSITION.fullmatch(text) else text


def open_source(source: str | Path, adapter: str = DEFAULT_ADAPTER, **options: Any) -> LatentSource:
    """Open latents through the adapter registry, like every other source in xaig."""
    built = registry.create(adapter, source=str(source), options=options)
    if not isinstance(built, LatentSource):
        raise AdapterError(f"adapter {adapter!r} does not implement LatentSource")
    return built
