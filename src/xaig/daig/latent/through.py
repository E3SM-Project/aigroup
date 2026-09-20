"""Through time, and between two runs.

``analyse_region`` is a snapshot. These ask the questions a perturbation or a
steering experiment raises: how does a region's response evolve from one physics
step to the next, which channels did the change reach, and how fast does the
difference from the control grow?

Two runs are set against each other node for node and channel for channel, so
they must be the same model on the same grid at the same times; that is checked,
not assumed.
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
    means the two runs differ as much as the control varies across the globe."""

    settings: dict[str, Any]
    provenance: dict[str, Any]
    times: tuple[str, ...]
    elapsed_seconds: tuple[float, ...] | None
    layers: tuple[int, ...]
    rms: np.ndarray
    relative: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "times": list(self.times),
            "elapsed_seconds": None if self.elapsed_seconds is None else list(self.elapsed_seconds),
            "layers": list(self.layers),
            "rms": self.rms.tolist(),
            "relative": self.relative.tolist(),
        }


def difference_growth(
    control: LatentSource,
    experiment: LatentSource,
    *,
    layers: Sequence[int] | None = None,
    times: Sequence[str | int] | None = None,
    across_models: bool = False,
) -> DifferenceGrowth:
    """Follow a perturbation through the network and through time: the size of
    ``experiment - control`` at every layer and every time the two share, over
    the nodes valid in both."""
    chosen = tuple(x.index for x in control.info().layers) if layers is None else tuple(layers)
    for layer in chosen:
        check_comparable(control, experiment, layer=layer, across_models=across_models)
    labels = _common_times(control, experiment, times)
    grid = shared_grid(control, experiment)
    weights = grid.weights()
    rms = np.zeros((len(labels), len(chosen)))
    relative = np.zeros_like(rms)
    for i, label in enumerate(labels):
        for j, layer in enumerate(chosen):
            reference = control.load(label, layer)
            delta = experiment.load(label, layer)
            delta -= reference
            rms[i, j] = np.sqrt(_mean_square(weights, delta).mean())
            reference -= grid.mean(reference).astype(reference.dtype)
            spread = np.sqrt(_mean_square(weights, reference).mean())
            relative[i, j] = rms[i, j] / spread if spread > 0.0 else np.nan
    return DifferenceGrowth(
        settings={
            "layers": list(chosen),
            "across_models": across_models,
            "n_nodes_compared": int(grid.valid.sum()),
        },
        provenance=_pair_provenance(control, experiment),
        times=tuple(labels),
        elapsed_seconds=_elapsed(control, labels),
        layers=chosen,
        rms=rms,
        relative=relative,
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


def rank_by_field(
    source: LatentSource,
    *,
    time: str | int,
    layer: int,
    field: str,
    top: int = 15,
    basis: Decomposition | None = None,
    allow_unverified_basis: bool = False,
) -> FieldRanking:
    """Which channels -- or which of a ``basis``'s features -- track a physical
    field the source kept beside its latents (``ReferenceFields``): the start of
    a feature-finding expedition. Correlation is over valid nodes, area-weighted,
    and says nothing about cause."""
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
    values = source.field(field, label)
    latents = source.load(label, layer)
    weights = grid.weights()
    if basis is None:
        correlation = correlate_field(latents, values, weights)
    else:
        correlation = np.concatenate(
            [
                correlate_field(
                    basis.transform(latents, features=range(start, stop)), values, weights
                )
                for start in range(0, basis.n_features, _FEATURES_AT_ONCE)
                for stop in [min(start + _FEATURES_AT_ONCE, basis.n_features)]
            ]
        )
    order = np.argsort(np.nan_to_num(np.abs(correlation), nan=-1.0), kind="stable")[::-1]
    chosen = order[: max(top, 0)]
    return FieldRanking(
        settings={
            "time": label,
            "layer": layer,
            "field": field,
            "top": top,
            "columns": "features" if basis is not None else "channels",
            "basis": None if basis is None else basis.meta.get("path"),
            "allow_unverified_basis": allow_unverified_basis,
        },
        provenance={**info.provenance(), "xaig": __version__},
        ranking=ChannelRanking(chosen, correlation[chosen]),
        correlation=correlation,
    )
