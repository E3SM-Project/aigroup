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
        index compares these first. Undeclared fields are left out; callers
        decide whether incomplete identity requires an explicit override."""
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
    time, NaN where the field is missing; with ``lead``, the field that many times
    later on the source's own reference axis, which holds every forward step's state
    even where latents were kept for fewer. A forward pass reads the state at its own
    time and writes the next: ``lead=1`` is what the pass starting there produced.

    This is not a contract for setting an emulator against a reference (levels,
    variables through time, two datasets); it is the few fields an exporter chose
    to keep next to its activations.
    """

    def field_names(self) -> tuple[str, ...]: ...

    def field(self, name: str, time: str | int, lead: int = 0) -> np.ndarray: ...


def parse_time(text: str) -> str | int:
    """A time as a person types it: a position (``0``, ``-1``) or a label."""
    return int(text) if _POSITION.fullmatch(text) else text


def open_source(source: str | Path, adapter: str = DEFAULT_ADAPTER, **options: Any) -> LatentSource:
    """Open latents through the adapter registry, like every other source in xaig."""
    built = registry.create(adapter, source=str(source), options=options)
    if not isinstance(built, LatentSource):
        raise AdapterError(f"adapter {adapter!r} does not implement LatentSource")
    return built


def differing_identity(ours: Mapping[str, Any], theirs: Mapping[str, Any]) -> list[str]:
    """What two declared identities disagree about, as text; undeclared is not
    disagreement."""
    return [
        f"{key} {ours[key]!r} against {theirs[key]!r}"
        for key in ("model", "component", "checkpoint")
        if ours.get(key) and theirs.get(key) and str(ours[key]) != str(theirs[key])
    ]


def check_comparable(
    a: LatentSource, b: LatentSource, *, layer: int, across_models: bool = False
) -> None:
    """Refuse to set two sources against each other unless node ``i`` of one is
    node ``i`` of the other and channel ``c`` of ``layer`` is the same channel.

    A perturbed run is compared with its control node for node and channel for
    channel, so a differing grid or width is not a detail: the difference would
    be computed, and would mean nothing. Nor is a differing network: two
    checkpoints trained apart agree on a channel's index and on nothing about it,
    so sources that declare different identities are refused too. ``across_models``
    lifts that one check, for the cases where the index does carry over (a
    fine-tune of the same weights), and it is the caller's to justify.
    """
    info_a, info_b = a.info(), b.info()
    apart = differing_identity(info_a.identity(), info_b.identity())
    if apart and not across_models:
        raise RequestError(
            f"{info_a.source} and {info_b.source} are different networks ({'; '.join(apart)}), "
            "and a channel's index means nothing between two; pass across_models=True "
            "only if it does here"
        )
    width_a, width_b = info_a.layer(layer).n_channels, info_b.layer(layer).n_channels
    if width_a != width_b:
        raise RequestError(
            f"layer {layer} has {width_a} channel(s) in {info_a.source} "
            f"and {width_b} in {info_b.source}"
        )
    grid_a, grid_b = a.grid(), b.grid()
    if grid_a.n_nodes != grid_b.n_nodes or grid_a.shape != grid_b.shape:
        raise RequestError(
            f"{info_a.source} and {info_b.source} are on different grids "
            f"({grid_a.n_nodes} and {grid_b.n_nodes} nodes)"
        )
    same_lat = np.allclose(grid_a.lat, grid_b.lat)
    if not (same_lat and np.allclose(grid_a.lon % 360.0, grid_b.lon % 360.0)):
        raise RequestError(
            f"{info_a.source} and {info_b.source} have the same number of nodes in different places"
        )


def shared_grid(a: LatentSource, b: LatentSource) -> Grid:
    """The first source's grid, valid only where both are: what a comparison of
    the two may weigh and map. A node one run marks invalid holds whatever it
    holds there -- NaN, or a number that means nothing -- and one such node would
    otherwise decide the whole difference."""
    grid_a, grid_b = a.grid(), b.grid()
    return Grid(
        lat=grid_a.lat,
        lon=grid_a.lon,
        shape=grid_a.shape,
        mask=grid_a.valid & grid_b.valid,
        area=grid_a.area,
    )


def check_basis_fits(
    basis: Any, info: LatentInfo, layer: int, *, allow_unverified: bool = False
) -> None:
    """Refuse a basis that says it was fitted on another network or another layer.

    Its width matching is not enough: every layer of a model is equally wide, and
    so is every seed of a campaign. What a basis's file records under
    ``fitted_on`` is compared with where it is being used. Missing layer,
    model, component or checkpoint identity on either side requires an explicit
    ``allow_unverified=True``. Known mismatches are refused even with the override.
    """
    width = info.layer(layer).n_channels
    if basis.n_channels != width:
        raise RequestError(
            f"the basis reads {basis.n_channels} channel(s), and layer {layer} has {width}"
        )
    fitted = basis.meta.get("fitted_on") or {}
    apart = differing_identity(info.identity(), fitted.get("provenance") or {})
    if fitted.get("layer") is not None and int(fitted["layer"]) != layer:
        apart.append(f"layer {layer} against the layer {fitted['layer']} it was fitted on")
    if apart:
        where = basis.meta.get("path") or "given"
        raise RequestError(f"the basis ({where}) was not fitted here: {'; '.join(apart)}")

    provenance = fitted.get("provenance") or {}
    identity = info.identity()
    missing = [
        key
        for key in ("model", "component", "checkpoint")
        if not provenance.get(key) or not identity.get(key)
    ]
    if fitted.get("layer") is None:
        missing.append("layer")
    if missing and not allow_unverified:
        raise RequestError(
            f"basis compatibility is unverified (missing {', '.join(missing)}); "
            "pass allow_unverified_basis=True (CLI: --allow-unverified-basis) to proceed"
        )
