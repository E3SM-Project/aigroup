"""Through time, and between two runs.

``analyse_region`` is a snapshot. These ask the questions a perturbation or a
steering experiment raises: how does a region's response evolve from one physics
step to the next, which channels did the change reach, and how fast does the
difference from the control grow?

Two runs are set against each other node for node and channel for channel, so
they must be the same model on the same grid at the same times; that is checked,
not assumed. A stochastic model makes that comparison mean something only when
both runs drew the same noise; a third run that differs from the control *only*
in its noise is the baseline a difference has to clear.

Two views through time: a storyline (how closely each layer follows a physical
field, time by time) and a Hovmoller diagram (one quantity along a latitude band,
longitude against time), which is how travelling waves show themselves.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from xaig import __version__
from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra
from xaig.daig.latent.analysis import ChannelRanking, Region
from xaig.daig.latent.basis import Decomposition
from xaig.daig.latent.samples import _time_labels
from xaig.daig.latent.source import (
    LatentSource,
    check_basis_fits,
    check_comparable,
    shared_grid,
)

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "daig") from exc

_BLOCK = 8192


@dataclass(frozen=True, eq=False)
class RegionSeries:
    """A region's mean response at every time. ``values`` is
    ``(n_times, len(columns))``; ``columns`` are channel or feature indices.
    ``elapsed_seconds`` is None when the source's calendar cannot say."""

    settings: dict[str, Any]
    provenance: dict[str, Any]
    times: tuple[str, ...]
    elapsed_seconds: tuple[float, ...] | None
    columns: tuple[int, ...]
    values: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "times": list(self.times),
            "elapsed_seconds": None if self.elapsed_seconds is None else list(self.elapsed_seconds),
            "columns": list(self.columns),
            "values": self.values.tolist(),
        }


def _elapsed(source: LatentSource, labels: Sequence[str]) -> tuple[float, ...] | None:
    info = source.info()
    elapsed = info.elapsed_seconds()
    if elapsed is None:
        return None
    return tuple(elapsed[info.times.index(label)] for label in labels)


def region_series(
    source: LatentSource,
    *,
    layer: int,
    region: Region,
    channels: Sequence[int] | None = None,
    basis: Decomposition | None = None,
    features: Sequence[int] | None = None,
    times: Sequence[str | int] | None = None,
    centred: bool = False,
    allow_unverified_basis: bool = False,
) -> RegionSeries:
    """The area-weighted mean over a region, at every time, of some channels --
    or, given a ``basis``, of some of its ``features``.

    Uncentred, only the region's nodes are read, so a series over every time of
    an archive costs a few MB. ``centred`` removes each channel's global mean at
    each time, which takes a full read per time. A basis sees raw latents, as it
    does everywhere, and is checked as it is everywhere (``check_basis_fits``):
    ``allow_unverified_basis`` lets one through that does not say where it was
    fitted, never one that says somewhere else.
    """
    info, grid = source.info(), source.grid()
    width = info.layer(layer).n_channels
    labels = _time_labels(source, times)
    nodes = grid.within(region.lat, region.lon, region.radius_km)
    if nodes.size == 0:
        raise RequestError(
            f"no valid nodes within {region.radius_km:g} km of "
            f"({region.lat:g}, {region.lon:g}); widen the region"
        )
    if basis is not None:
        check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
        columns = tuple(int(f) for f in (range(basis.n_features) if features is None else features))
    else:
        columns = tuple(int(c) for c in (range(width) if channels is None else channels))
        bad = [c for c in columns if not 0 <= c < width]
        if bad:
            raise RequestError(f"channel(s) {bad} outside 0..{width - 1}")
    weights = grid.weights()[nodes]
    weights = weights / weights.sum()

    rows = []
    for label in labels:
        if basis is not None:
            local = basis.transform(source.load(label, layer, nodes=nodes), features=columns)
        elif centred:
            full = source.load(label, layer, channels=list(columns))
            local = (full - grid.mean(full))[nodes]
        else:
            local = source.load(label, layer, channels=list(columns), nodes=nodes)
        rows.append(weights @ local.astype(np.float64))
    settings = {
        "layer": layer,
        "region": asdict(region),
        "columns": "features" if basis is not None else "channels",
        "centred": bool(centred and basis is None),
        "basis": None if basis is None else basis.meta.get("path"),
        "allow_unverified_basis": allow_unverified_basis,
    }
    return RegionSeries(
        settings=settings,
        provenance={**info.provenance(), "xaig": __version__},
        times=tuple(labels),
        elapsed_seconds=_elapsed(source, labels),
        columns=columns,
        values=np.array(rows),
    )


# -- a run against its control ------------------------------------------------


def _mean_square(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Area-weighted mean of ``values**2`` per channel, without squaring a layer
    in one piece."""
    total = np.zeros(values.shape[1])
    for start in range(0, values.shape[0], _BLOCK):
        keep = np.flatnonzero(weights[start : start + _BLOCK] > 0.0) + start
        if keep.size:
            block = values[keep].astype(np.float64)
            total += weights[keep] @ (block * block)
    return total


def _common_times(control: LatentSource, experiment: LatentSource, times) -> list[str]:
    labels = _time_labels(control, times)
    theirs = set(experiment.info().times)
    if times is None:
        labels = [label for label in labels if label in theirs]
    missing = [label for label in labels if label not in theirs]
    if missing or not labels:
        absent = ", ".join(missing) or "any time the control has"
        raise RequestError(f"{experiment.info().source} has no latents at {absent}")
    return labels


@dataclass(frozen=True, eq=False)
class PairedDifference:
    """One run minus its control, at one layer and time.

    ``ranking`` orders channels by the area-weighted RMS of the difference, which
    is what ``rms`` holds for every channel. ``maps`` is the difference itself
    for the ranked channels, ``(n_nodes, len(ranking.channels))``, NaN where
    either run's grid is invalid: only nodes valid in both are weighed, and
    ``settings["n_nodes_compared"]`` says how many that was.
    """

    settings: dict[str, Any]
    provenance: dict[str, Any]
    ranking: ChannelRanking
    rms: np.ndarray
    maps: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "total_rms": float(np.sqrt(np.mean(self.rms**2))),
            "ranking": [
                {"channel": int(c), "rms": float(s)}
                for c, s in zip(self.ranking.channels, self.ranking.scores, strict=True)
            ],
        }


def _pair_provenance(control: LatentSource, experiment: LatentSource) -> dict[str, Any]:
    return {
        "control": control.info().provenance(),
        "experiment": experiment.info().provenance(),
        "xaig": __version__,
    }


def difference(
    control: LatentSource,
    experiment: LatentSource,
    *,
    time: str | int,
    layer: int,
    top: int = 15,
    across_models: bool = False,
) -> PairedDifference:
    """Which channels a perturbation reached, and where: ``experiment - control``
    at one layer and time, channels ranked by how much they moved.

    The two must be one network on one grid (``check_comparable``); what is
    compared is the nodes valid in both.
    """
    check_comparable(control, experiment, layer=layer, across_models=across_models)
    label = control.info().times[control.info().time_index(time)]
    experiment.info().time_index(label)
    grid = shared_grid(control, experiment)
    weights = grid.weights()
    delta = experiment.load(label, layer)
    delta -= control.load(label, layer)
    rms = np.sqrt(_mean_square(weights, delta))
    chosen = np.argsort(rms, kind="stable")[::-1][: max(top, 0)]
    maps = np.where(grid.valid[:, None], delta[:, chosen].astype(np.float64), np.nan)
    return PairedDifference(
        settings={
            "time": label,
            "layer": layer,
            "top": top,
            "across_models": across_models,
            "n_nodes_compared": int(grid.valid.sum()),
        },
        provenance=_pair_provenance(control, experiment),
        ranking=ChannelRanking(chosen, rms[chosen]),
        rms=rms,
        maps=maps,
    )


@dataclass(frozen=True, eq=False)
class DifferenceGrowth:
    """How far a run is from its control, time by time. ``rms`` is
    ``(n_times, n_layers)``: the area-weighted RMS difference over all channels.
    ``relative`` divides it by the control's own RMS about its global mean, so 1
    means the two runs differ as much as the control varies across the globe.

    Given a ``noise`` run -- the control again, with a different seed -- ``noise_rms``
    is how far that one is from the control, and ``signal_to_noise`` is
    ``rms / noise_rms``: above 1, the experiment moved the layer more than a new
    draw of the model's own noise does."""

    settings: dict[str, Any]
    provenance: dict[str, Any]
    times: tuple[str, ...]
    elapsed_seconds: tuple[float, ...] | None
    layers: tuple[int, ...]
    rms: np.ndarray
    relative: np.ndarray
    noise_rms: np.ndarray | None = None

    @property
    def signal_to_noise(self) -> np.ndarray | None:
        if self.noise_rms is None:
            return None
        out = np.full_like(self.rms, np.nan)
        np.divide(self.rms, self.noise_rms, out=out, where=self.noise_rms > 0.0)
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "times": list(self.times),
            "elapsed_seconds": None if self.elapsed_seconds is None else list(self.elapsed_seconds),
            "layers": list(self.layers),
            "rms": self.rms.tolist(),
            "relative": self.relative.tolist(),
            "noise_rms": None if self.noise_rms is None else self.noise_rms.tolist(),
        }


def difference_growth(
    control: LatentSource,
    experiment: LatentSource,
    *,
    layers: Sequence[int] | None = None,
    times: Sequence[str | int] | None = None,
    across_models: bool = False,
    noise: LatentSource | None = None,
) -> DifferenceGrowth:
    """Follow a perturbation through the network and through time: the size of
    ``experiment - control`` at every layer and every time the two share, over
    the nodes valid in both. ``noise``, a rerun of the control with another seed,
    adds the baseline a stochastic model's differences have to clear."""
    chosen = tuple(x.index for x in control.info().layers) if layers is None else tuple(layers)
    for layer in chosen:
        check_comparable(control, experiment, layer=layer, across_models=across_models)
        if noise is not None:
            check_comparable(control, noise, layer=layer, across_models=across_models)
    labels = _common_times(control, experiment, times)
    if noise is not None:
        labels = _common_times(control, noise, labels)
    grid = shared_grid(control, experiment)
    if noise is not None:
        shared_grid(control, noise)
    weights = grid.weights()
    rms = np.zeros((len(labels), len(chosen)))
    relative = np.zeros_like(rms)
    noise_rms = None if noise is None else np.zeros_like(rms)
    for i, label in enumerate(labels):
        for j, layer in enumerate(chosen):
            reference = control.load(label, layer)
            delta = experiment.load(label, layer)
            delta -= reference
            rms[i, j] = np.sqrt(_mean_square(weights, delta).mean())
            if noise is not None:
                delta = noise.load(label, layer)
                delta -= reference
                noise_rms[i, j] = np.sqrt(_mean_square(weights, delta).mean())
            reference -= grid.mean(reference).astype(reference.dtype)
            spread = np.sqrt(_mean_square(weights, reference).mean())
            relative[i, j] = rms[i, j] / spread if spread > 0.0 else np.nan
    provenance = _pair_provenance(control, experiment)
    if noise is not None:
        provenance["noise"] = noise.info().provenance()
    return DifferenceGrowth(
        settings={
            "layers": list(chosen),
            "across_models": across_models,
            "n_nodes_compared": int(grid.valid.sum()),
            "noise": None if noise is None else noise.info().source,
        },
        provenance=provenance,
        times=tuple(labels),
        elapsed_seconds=_elapsed(control, labels),
        layers=chosen,
        rms=rms,
        relative=relative,
        noise_rms=noise_rms,
    )


# -- a channel against a physical field ---------------------------------------


def correlate_field(
    latents: np.ndarray, values: np.ndarray, weights: np.ndarray | None = None
) -> np.ndarray:
    """Area-weighted Pearson correlation of every channel (or feature) with one
    per-node field: ``(n_nodes, k)`` against ``(n_nodes,)`` gives ``(k,)``.

    Nodes where the field is NaN or the weight is zero are left out, which is
    how land drops out of a correlation with sea-surface temperature. A channel
    that does not vary -- every counted node holds the same number -- reads NaN,
    and that is the only one that does: each channel is centred on its own mean
    before its moments are taken, so an offset of a million takes nothing from a
    variation of one.
    """
    latents = np.asarray(latents)
    values = np.asarray(values, dtype=np.float64)
    if latents.ndim != 2 or values.shape != (latents.shape[0],):
        raise ValueError(f"expected (n, k) and (n,), got {latents.shape} and {values.shape}")
    w = np.ones(values.size) if weights is None else np.asarray(weights, dtype=np.float64)
    keep = np.flatnonzero((w > 0.0) & np.isfinite(values))
    if keep.size < 2:
        raise RequestError("fewer than two nodes have both a weight and a field value")
    w = w[keep] / w[keep].sum()
    y = values[keep] - w @ values[keep]
    n_columns = latents.shape[1]

    def blocks():
        for start in range(0, keep.size, _BLOCK):
            stop = start + _BLOCK
            yield latents[keep[start:stop]].astype(np.float64), w[start:stop], y[start:stop]

    # Two passes. Moments about zero lose a small variation beside a large mean
    # (and a tolerance to catch the constant channel would then catch that too),
    # so the mean comes first and everything after is about it. Whether a channel
    # varies at all is asked of the data, exactly: its extremes differ or they do not.
    mean, low, high = np.zeros(n_columns), np.full(n_columns, np.inf), np.full(n_columns, -np.inf)
    for block, wb, _ in blocks():
        mean += wb @ block
        low, high = np.minimum(low, block.min(axis=0)), np.maximum(high, block.max(axis=0))
    spread, cross = np.zeros(n_columns), np.zeros(n_columns)
    for block, wb, yb in blocks():
        block -= mean
        spread += wb @ (block * block)
        cross += (wb * yb) @ block
    scale = spread * (w @ (y * y))
    out = np.full(n_columns, np.nan)
    np.divide(cross, np.sqrt(scale), out=out, where=(high > low) & (scale > 0.0))
    return np.clip(out, -1.0, 1.0)


@dataclass(frozen=True, eq=False)
class FieldRanking:
    """Channels (or a basis's features) in order of how closely they follow one
    physical field at one time. ``correlation`` holds every column's; ``ranking``
    the strongest, by absolute value, with the sign kept in ``scores``."""

    settings: dict[str, Any]
    provenance: dict[str, Any]
    ranking: ChannelRanking
    correlation: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "ranking": [
                {"column": int(c), "correlation": float(s)}
                for c, s in zip(self.ranking.channels, self.ranking.scores, strict=True)
            ],
        }


_FEATURES_AT_ONCE = 256
# Feature activations held at once, when a basis is set against a field: 2**27 float64s
# is 1 GiB. Below it a layer is encoded once; above it, in slices of features -- which
# costs a full encoding per slice when features compete (TopK), but bounds memory.
_ACTIVATIONS_AT_ONCE = 2**27


def _correlate_features(
    basis: Decomposition, latents: np.ndarray, values: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Every feature of ``basis`` against one field, encoding the layer as few times as
    memory allows."""
    if latents.shape[0] * basis.n_features <= _ACTIVATIONS_AT_ONCE:
        return correlate_field(basis.transform(latents), values, weights)
    return np.concatenate(
        [
            correlate_field(basis.transform(latents, features=range(start, stop)), values, weights)
            for start in range(0, basis.n_features, _FEATURES_AT_ONCE)
            for stop in [min(start + _FEATURES_AT_ONCE, basis.n_features)]
        ]
    )


def rank_by_field(
    source: LatentSource,
    *,
    time: str | int,
    layer: int,
    field: str,
    top: int = 15,
    basis: Decomposition | None = None,
    allow_unverified_basis: bool = False,
    lead: int = 0,
) -> FieldRanking:
    """Which channels -- or which of a ``basis``'s features -- track a physical
    field the source kept beside its latents (``ReferenceFields``): the start of
    a feature-finding expedition. Correlation is over valid nodes, area-weighted,
    and says nothing about cause.

    ``lead`` sets the latents against the field that many reference times later.
    A forward pass reads the state at its own time and writes the next, so an
    output the pass produces -- precipitation, say -- is at ``lead=1``; at 0 it is
    the previous pass's, which this one never saw."""
    from xaig.daig.latent.source import ReferenceFields

    info, grid = source.info(), source.grid()
    if not isinstance(source, ReferenceFields):
        raise RequestError(f"{info.source} keeps no physical fields beside its latents")
    if field not in source.field_names():
        known = ", ".join(source.field_names()) or "none"
        raise RequestError(f"no field {field!r} in {info.source}; fields are {known}")
    label = info.times[info.time_index(time)]
    info.layer(layer)
    if basis is not None:
        check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
    values = _field_with_values(source, field, label, lead)
    latents = source.load(label, layer)
    weights = grid.weights()
    if basis is None:
        correlation = correlate_field(latents, values, weights)
    else:
        correlation = _correlate_features(basis, latents, values, weights)
    order = np.argsort(np.nan_to_num(np.abs(correlation), nan=-1.0), kind="stable")[::-1]
    chosen = order[: max(top, 0)]
    return FieldRanking(
        settings={
            "time": label,
            "layer": layer,
            "field": field,
            "lead": lead,
            "top": top,
            "columns": "features" if basis is not None else "channels",
            "basis": None if basis is None else basis.meta.get("path"),
            "allow_unverified_basis": allow_unverified_basis,
        },
        provenance={**info.provenance(), "xaig": __version__},
        ranking=ChannelRanking(chosen, correlation[chosen]),
        correlation=correlation,
    )


def _field_at(source: LatentSource, field: str, label: str, lead: int = 0) -> np.ndarray:
    """A field at a latent time, or ``lead`` reference times after it. A source that
    predates ``lead`` is asked the old way when none is wanted."""
    return source.field(field, label, lead=lead) if lead else source.field(field, label)


def _field_with_values(source: LatentSource, field: str, label: str, lead: int = 0) -> np.ndarray:
    values = _field_at(source, field, label, lead)
    if not np.isfinite(values[source.grid().valid]).any():
        where = label if not lead else f"{lead:+d} time(s) from {label}"
        raise RequestError(
            f"{field!r} has no values at {where}. A model's diagnostic outputs usually "
            "start one step after its initial state: try a later time, or lead=1"
        )
    return values


def _reference_fields(source: LatentSource, field: str):
    from xaig.daig.latent.source import ReferenceFields

    info = source.info()
    if not isinstance(source, ReferenceFields):
        raise RequestError(f"{info.source} keeps no physical fields beside its latents")
    if field not in source.field_names():
        known = ", ".join(source.field_names()) or "none"
        raise RequestError(f"no field {field!r} in {info.source}; fields are {known}")
    return source


# -- a storyline: where a field lives in the network, time by time -------------


@dataclass(frozen=True, eq=False)
class FieldStoryline:
    """How closely each layer follows one physical field, at every time.

    ``best`` is ``(n_times, n_layers)``: the largest absolute correlation any
    channel (or feature, when a layer has a basis) reaches with the field, and
    ``column`` which one it was, with ``sign`` its correlation's sign. Bright at the
    first layer means the field comes in with the inputs; bright only deep in the
    network means the network builds it. NaN where the field has no values."""

    settings: dict[str, Any]
    provenance: dict[str, Any]
    times: tuple[str, ...]
    elapsed_seconds: tuple[float, ...] | None
    layers: tuple[int, ...]
    best: np.ndarray
    column: np.ndarray
    sign: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "times": list(self.times),
            "layers": list(self.layers),
            "best": self.best.tolist(),
            "column": self.column.tolist(),
            "sign": self.sign.tolist(),
        }


def field_storyline(
    source: LatentSource,
    *,
    field: str,
    layers: Sequence[int] | None = None,
    times: Sequence[str | int] | None = None,
    bases: dict[int, Decomposition] | None = None,
    allow_unverified_basis: bool = False,
    lead: int = 0,
) -> FieldStoryline:
    """Follow one physical field through the network and through time.

    ``bases`` maps a layer to a basis for it; a layer without one is read by its
    channels. Each layer at each time is loaded once, so a storyline over nine
    layers and twenty times reads the archive once. ``lead`` is as for
    ``rank_by_field``: 1 follows an output to the pass that produced it; a time
    whose led field the source does not hold is left empty."""
    _reference_fields(source, field)
    info, grid = source.info(), source.grid()
    chosen = tuple(x.index for x in info.layers) if layers is None else tuple(layers)
    bases = dict(bases or {})
    for layer in chosen:
        info.layer(layer)
        if layer in bases:
            check_basis_fits(bases[layer], info, layer, allow_unverified=allow_unverified_basis)
    unused = sorted(set(bases) - set(chosen))
    if unused:
        raise RequestError(f"bases given for layer(s) {unused}, which the storyline does not read")
    labels = _time_labels(source, times)
    weights = grid.weights()
    best = np.full((len(labels), len(chosen)), np.nan)
    column = np.full(best.shape, -1, dtype=np.int64)
    sign = np.zeros(best.shape, dtype=np.int8)
    for i, label in enumerate(labels):
        try:
            values = _field_at(source, field, label, lead)
        except RequestError:
            if not lead:
                raise
            continue  # past the end of the source's reference times
        if not np.isfinite(values[grid.valid]).any():
            continue
        for j, layer in enumerate(chosen):
            latents = source.load(label, layer)
            basis = bases.get(layer)
            if basis is None:
                r = correlate_field(latents, values, weights)
            else:
                r = _correlate_features(basis, latents, values, weights)
            if np.isnan(r).all():
                continue
            k = int(np.nanargmax(np.abs(r)))
            best[i, j], column[i, j], sign[i, j] = abs(r[k]), k, np.sign(r[k])
    return FieldStoryline(
        settings={
            "field": field,
            "lead": lead,
            "layers": list(chosen),
            "bases": {int(k): v.meta.get("path") for k, v in bases.items()},
            "allow_unverified_basis": allow_unverified_basis,
        },
        provenance={**info.provenance(), "xaig": __version__},
        times=tuple(labels),
        elapsed_seconds=_elapsed(source, labels),
        layers=chosen,
        best=best,
        column=column,
        sign=sign,
    )


# -- a Hovmoller diagram: one quantity along a latitude band -------------------


@dataclass(frozen=True, eq=False)
class Hovmoller:
    """One quantity averaged over a latitude band, per longitude column and time.

    ``values`` is ``(n_times, n_lon)``, area-weighted over the band's valid nodes
    (NaN where a column has none); ``lon`` the columns' longitudes, in the grid's
    own order. Tilted stripes are something travelling: their slope is its speed."""

    settings: dict[str, Any]
    provenance: dict[str, Any]
    times: tuple[str, ...]
    elapsed_seconds: tuple[float, ...] | None
    lon: np.ndarray
    values: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "times": list(self.times),
            "lon": self.lon.tolist(),
            "values": self.values.tolist(),
        }


def hovmoller(
    source: LatentSource,
    *,
    lat_min: float,
    lat_max: float,
    field: str | None = None,
    layer: int | None = None,
    channel: int | None = None,
    basis: Decomposition | None = None,
    feature: int | None = None,
    times: Sequence[str | int] | None = None,
    allow_unverified_basis: bool = False,
) -> Hovmoller:
    """A Hovmoller diagram of a physical ``field``, or of one ``channel`` of a
    ``layer``, or of one ``feature`` of a ``basis`` on that layer, between two
    latitudes. Needs a structured grid: a mesh has no longitude columns."""
    info, grid = source.info(), source.grid()
    if grid.shape is None:
        raise RequestError("a Hovmoller diagram needs a structured grid; this one is a mesh")
    if not lat_min < lat_max:
        raise RequestError(f"lat_min ({lat_min}) must be below lat_max ({lat_max})")
    by_field = field is not None
    by_channel = layer is not None and channel is not None and basis is None
    by_feature = layer is not None and basis is not None and feature is not None
    if by_field + by_channel + by_feature != 1:
        raise RequestError(
            "give exactly one of: a field; a layer and a channel; a layer, a basis and a feature"
        )
    if by_field:
        _reference_fields(source, field)
    else:
        info.layer(layer)
        if by_channel and not 0 <= channel < info.layer(layer).n_channels:
            raise RequestError(f"layer {layer} has no channel {channel}")
        if by_feature:
            check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
            if not 0 <= feature < basis.n_features:
                raise RequestError(f"the basis has no feature {feature}")
    n_lat, n_lon = grid.shape
    lat_rows = grid.lat.reshape(n_lat, n_lon).mean(axis=1)
    rows = np.flatnonzero((lat_rows >= lat_min) & (lat_rows <= lat_max))
    if rows.size == 0:
        raise RequestError(f"no grid rows between {lat_min} and {lat_max} degrees")
    nodes = (rows[:, None] * n_lon + np.arange(n_lon)[None, :]).ravel()
    weights = (grid.weights() * grid.valid)[nodes].reshape(rows.size, n_lon)
    column_weight = weights.sum(axis=0)
    labels = _time_labels(source, times)
    values = np.full((len(labels), n_lon), np.nan)
    for i, label in enumerate(labels):
        if by_field:
            band = source.field(field, label)[nodes]
        elif by_channel:
            band = source.load(label, layer, channels=[channel], nodes=nodes)[:, 0]
        else:
            band = basis.transform(source.load(label, layer, nodes=nodes), features=[feature])[:, 0]
        band = band.reshape(rows.size, n_lon)
        w = np.where(np.isfinite(band), weights, 0.0)
        total = w.sum(axis=0)
        values[i] = np.where(
            total > 0.0,
            (w * np.nan_to_num(band)).sum(axis=0) / np.where(total > 0, total, 1.0),
            np.nan,
        )
    lon = grid.lon.reshape(n_lat, n_lon)[rows[0]]
    return Hovmoller(
        settings={
            "lat_min": lat_min,
            "lat_max": lat_max,
            "rows": int(rows.size),
            "field": field,
            "layer": layer,
            "channel": channel,
            "feature": feature,
            "basis": None if basis is None else basis.meta.get("path"),
            "columns_without_nodes": int((column_weight == 0).sum()),
        },
        provenance={**info.provenance(), "xaig": __version__},
        times=tuple(labels),
        elapsed_seconds=_elapsed(source, labels),
        lon=np.asarray(lon, dtype=np.float64),
        values=values,
    )
