"""What a feature is, asked without a field in mind.

``rank_by_field`` and ``field_storyline`` start from a physical field and ask which
feature follows it; one correlation then stands for a whole map. These start from
the features. A census says, for every channel or feature of a layer at one time,
how much of the world it is active over, how strongly, and where it peaks: a
catalogue to browse. A profile takes one of them and sets every physical field
where it is active against where it is not, which describes it by all the fields at
once and suits a feature that is on in one place and off everywhere else -- the
kind a correlation undersells.

"Active" means above a threshold, zero by default: for a sparse dictionary, where
most activations are exactly zero, that is "firing"; for channels or PCA scores it
is "positive", and a threshold of the reader's choosing may say more.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from xaig import __version__
from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra
from xaig.daig.latent.analysis import Region
from xaig.daig.latent.basis import Decomposition
from xaig.daig.latent.samples import _time_labels
from xaig.daig.latent.source import LatentSource, ReferenceFields, check_basis_fits

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "daig") from exc

_BLOCK = 8192
_ORDERS = ("coverage", "mean", "strength", "peak")


def _nodes(source: LatentSource, region: Region | None) -> np.ndarray:
    grid = source.grid()
    if region is None:
        return np.flatnonzero(grid.valid)
    nodes = grid.within(region.lat, region.lon, region.radius_km)
    if nodes.size == 0:
        raise RequestError(
            f"no valid nodes within {region.radius_km:g} km of "
            f"({region.lat:g}, {region.lon:g}); widen the region"
        )
    return nodes


def _check_layer(
    source: LatentSource, layer: int, basis: Decomposition | None, allow_unverified_basis: bool
) -> int:
    info = source.info()
    width = info.layer(layer).n_channels
    if basis is None:
        return width
    check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
    return basis.n_features


# -- a census: every feature of a layer at one time ----------------------------


@dataclass(frozen=True, eq=False)
class FeatureCensus:
    """Every channel (or feature) of one layer at one time, area-weighted over the
    valid nodes read. ``coverage`` is the share of the area where it is active;
    ``mean`` its mean over all of that area, ``strength`` its mean where active
    (NaN where it never is); ``peak`` its largest value, found at ``peak_lat``,
    ``peak_lon``."""

    settings: dict[str, Any]
    provenance: dict[str, Any]
    coverage: np.ndarray
    mean: np.ndarray
    strength: np.ndarray
    peak: np.ndarray
    peak_lat: np.ndarray
    peak_lon: np.ndarray

    def ranked(self, by: str = "coverage", top: int | None = None) -> np.ndarray:
        """Column indices, largest first by ``coverage``, ``mean``, ``strength`` or
        ``peak``. Columns never active come last whatever the order."""
        if by not in _ORDERS:
            raise RequestError(f"order by one of {', '.join(_ORDERS)}, not {by!r}")
        key = np.nan_to_num(getattr(self, by), nan=-np.inf)
        key = np.where(self.coverage > 0.0, key, -np.inf)
        order = np.argsort(key, kind="stable")[::-1]
        return order if top is None else order[: max(top, 0)]

    def summary(self, by: str = "coverage", top: int | None = None) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "columns": [
                {
                    "column": int(c),
                    "coverage": float(self.coverage[c]),
                    "mean": float(self.mean[c]),
                    "strength": float(self.strength[c]),
                    "peak": float(self.peak[c]),
                    "peak_lat": float(self.peak_lat[c]),
                    "peak_lon": float(self.peak_lon[c]),
                }
                for c in self.ranked(by, top)
            ],
        }


def feature_census(
    source: LatentSource,
    *,
    time: str | int,
    layer: int,
    basis: Decomposition | None = None,
    region: Region | None = None,
    threshold: float = 0.0,
    allow_unverified_basis: bool = False,
) -> FeatureCensus:
    """A catalogue of one layer at one time: for each channel -- or each of a
    ``basis``'s features -- how much of the area it is active over, how strongly,
    and where it peaks. Over the whole grid, or a ``region``. The layer is encoded
    once, a block of nodes at a time."""
    info, grid = source.info(), source.grid()
    label = info.times[info.time_index(time)]
    n_columns = _check_layer(source, layer, basis, allow_unverified_basis)
    nodes = _nodes(source, region)
    weights = grid.weights()[nodes]
    weights = weights / weights.sum()
    latents = source.load(label, layer, nodes=nodes)

    area, total = np.zeros(n_columns), np.zeros(n_columns)
    active_total = np.zeros(n_columns)
    peak = np.full(n_columns, -np.inf)
    where = np.zeros(n_columns, dtype=np.int64)
    for start in range(0, nodes.size, _BLOCK):
        block = latents[start : start + _BLOCK]
        values = basis.transform(block) if basis is not None else block.astype(np.float64)
        w = weights[start : start + _BLOCK]
        on = values > threshold
        area += w @ on
        total += w @ values
        active_total += w @ np.where(on, values, 0.0)
        higher = values.max(axis=0) > peak
        peak = np.where(higher, values.max(axis=0), peak)
        where = np.where(higher, start + values.argmax(axis=0), where)
    strength = np.full(n_columns, np.nan)
    np.divide(active_total, area, out=strength, where=area > 0.0)
    return FeatureCensus(
        settings={
            "time": label,
            "layer": layer,
            "columns": "features" if basis is not None else "channels",
            "basis": None if basis is None else basis.meta.get("path"),
            "region": None if region is None else asdict(region),
            "threshold": threshold,
            "allow_unverified_basis": allow_unverified_basis,
        },
        provenance={**info.provenance(), "xaig": __version__},
        coverage=np.clip(area, 0.0, 1.0),
        mean=total,
        strength=strength,
        peak=peak,
        peak_lat=grid.lat[nodes[where]].astype(np.float64),
        peak_lon=grid.lon[nodes[where]].astype(np.float64),
    )


# -- a profile: one feature against every field ---------------------------------


@dataclass(frozen=True, eq=False)
class FeatureProfile:
    """One channel or feature described by the physical fields kept beside it.

    For each field, over the nodes and times read: its area-weighted mean where the
    column is active (``active_mean``) and where it is not (``inactive_mean``), and
    their difference in units of the field's own spread over both (``effect``), so
    that fields of any unit can be read on one axis. ``fields`` is ordered by the
    size of the effect, largest first; a field that does not vary, or that has no
    values on one side, reads NaN and comes last. ``coverage`` is the share of the
    area and time the column is active over."""

    settings: dict[str, Any]
    provenance: dict[str, Any]
    times: tuple[str, ...]
    coverage: float
    fields: tuple[str, ...]
    effect: np.ndarray
    active_mean: np.ndarray
    inactive_mean: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "provenance": self.provenance,
            "times": list(self.times),
            "coverage": self.coverage,
            "fields": [
                {
                    "field": name,
                    "effect": float(e),
                    "active_mean": float(a),
                    "inactive_mean": float(i),
                }
                for name, e, a, i in zip(
                    self.fields, self.effect, self.active_mean, self.inactive_mean, strict=True
                )
            ],
        }


def feature_profile(
    source: LatentSource,
    *,
    layer: int,
    column: int,
    basis: Decomposition | None = None,
    times: Sequence[str | int] | None = None,
    fields: Sequence[str] | None = None,
    region: Region | None = None,
    threshold: float = 0.0,
    allow_unverified_basis: bool = False,
) -> FeatureProfile:
    """Every physical field where one channel -- or one of a ``basis``'s features
    -- is active, against where it is not, pooled over ``times`` (all, by default)
    and over the whole grid or a ``region``. Says what the column goes with, not
    what it causes."""
    info, grid = source.info(), source.grid()
    if not isinstance(source, ReferenceFields):
        raise RequestError(f"{info.source} keeps no physical fields beside its latents")
    n_columns = _check_layer(source, layer, basis, allow_unverified_basis)
    if not 0 <= column < n_columns:
        kind = "feature" if basis is not None else "channel"
        raise RequestError(f"layer {layer} has no {kind} {column}; there are {n_columns}")
    known = source.field_names()
    names = tuple(known if fields is None else fields)
    unknown = [n for n in names if n not in known]
    if unknown:
        raise RequestError(f"no field(s) {unknown} in {info.source}; fields are {', '.join(known)}")
    if not names:
        raise RequestError(f"{info.source} keeps no physical fields to profile against")
    labels = _time_labels(source, times)
    nodes = _nodes(source, region)
    weights = grid.weights()[nodes]

    # Per field: weight, weighted sum and weighted sum of squares, on each side --
    # of the field less the first mean seen, so a large offset (surface pressure)
    # costs no precision and a constant field has exactly no spread.
    sums = np.zeros((len(names), 2, 3))
    shift = np.full(len(names), np.nan)
    active_area = total_area = 0.0
    for label in labels:
        latents = source.load(label, layer, nodes=nodes)
        if basis is not None:
            values = basis.transform(latents, features=[column])[:, 0]
        else:
            values = latents[:, column].astype(np.float64)
        on = values > threshold
        active_area += float(weights[on].sum())
        total_area += float(weights.sum())
        for k, name in enumerate(names):
            field = source.field(name, label)[nodes]
            ok = np.isfinite(field)
            if np.isnan(shift[k]) and ok.any():
                shift[k] = float(weights[ok] @ field[ok] / weights[ok].sum())
            field = field - shift[k]
            for side, mask in ((0, on & ok), (1, ~on & ok)):
                w, f = weights[mask], field[mask]
                sums[k, side] += (w.sum(), w @ f, w @ (f * f))

    weight, first, second = sums[..., 0], sums[..., 1], sums[..., 2]
    with np.errstate(invalid="ignore", divide="ignore"):
        side_mean = first / weight
        pooled_mean = first.sum(axis=1) / weight.sum(axis=1)
        pooled_var = second.sum(axis=1) / weight.sum(axis=1) - pooled_mean**2
        spread = np.sqrt(np.where(pooled_var > 0.0, pooled_var, np.nan))
        effect = (side_mean[:, 0] - side_mean[:, 1]) / spread
    effect = np.where((weight > 0).all(axis=1), effect, np.nan)
    side_mean = side_mean + shift[:, None]
    order = np.argsort(np.nan_to_num(np.abs(effect), nan=-1.0), kind="stable")[::-1]
    return FeatureProfile(
        settings={
            "layer": layer,
            "column": column,
            "columns": "features" if basis is not None else "channels",
            "basis": None if basis is None else basis.meta.get("path"),
            "region": None if region is None else asdict(region),
            "threshold": threshold,
            "allow_unverified_basis": allow_unverified_basis,
        },
        provenance={**info.provenance(), "xaig": __version__},
        times=tuple(labels),
        coverage=active_area / total_area if total_area > 0 else float("nan"),
        fields=tuple(names[k] for k in order),
        effect=effect[order],
        active_mean=side_mean[order, 0],
        inactive_mean=side_mean[order, 1],
    )
