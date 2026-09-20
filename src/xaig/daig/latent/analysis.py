"""What a model's internals respond to, region by region.

Pure functions over ``(n_nodes, n_channels)`` arrays, plus one routine that strings
them into the usual question -- *which channels light up here, where else does
the model look like this, and what are the main patterns?* -- and returns a
result carrying the settings and provenance needed to reproduce it.

Nothing here draws, prints, or knows a user interface exists. A notebook, a batch
job and the CLI all call the same functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from xaig import __version__
from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra
from xaig.daig.latent.basis import PCA, Decomposition, _floating, fit_pca, top_loadings
from xaig.daig.latent.source import LatentSource, check_basis_fits

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "daig") from exc

REFERENCES = ("nearest", "mean")


@dataclass(frozen=True, slots=True)
class Region:
    """A spherical cap: everything within ``radius_km`` of a point."""

    lat: float
    lon: float
    radius_km: float


@dataclass(frozen=True, eq=False)
class ChannelRanking:
    """Channels in the order they are to be shown. ``scores`` is parallel to
    ``channels``; ``pinned`` names the ones placed first regardless of score."""

    channels: np.ndarray
    scores: np.ndarray
    pinned: tuple[int, ...] = ()


def rank_channels(latents: np.ndarray, top: int = 15, pinned: Sequence[int] = ()) -> ChannelRanking:
    """The ``top`` channels by peak absolute activation over the given nodes.

    ``pinned`` channels lead the list whatever they score (to follow one channel
    across layers, or compare against a published index), and the ranked ones fill
    what is left. Pass the latents of a region to ask what responds *there*. A
    channel that is NaN at every node has no peak and ranks last.
    """
    latents = _floating(latents)
    if latents.ndim != 2 or latents.shape[0] == 0:
        raise ValueError(f"expected (n_nodes, n_channels) with nodes, got {latents.shape}")
    n_channels = latents.shape[1]
    magnitude = np.abs(latents)
    silent = np.isnan(magnitude).all(axis=0)
    scores = np.where(silent, np.nan, np.nanmax(np.where(silent, 0.0, magnitude), axis=0))
    lead = list(dict.fromkeys(int(c) for c in pinned))
    bad = [c for c in lead if not 0 <= c < n_channels]
    if bad:
        raise RequestError(f"pinned channel(s) {bad} outside 0..{n_channels - 1}")
    order = np.argsort(np.where(silent, -np.inf, scores), kind="stable")[::-1]
    ranked = [int(c) for c in order if int(c) not in lead]
    chosen = np.array((lead + ranked)[: max(top, 0)], dtype=int)
    return ChannelRanking(chosen, scores[chosen], tuple(c for c in lead if c in chosen))


def cosine_similarity(latents: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cosine of the angle between every node's latent vector and ``reference``.

    A node whose vector is all zeros has no direction, and reads NaN rather than
    raising a warning or borrowing a number.
    """
    latents = _floating(latents)
    # Stay in the layer's own precision: only per-node results are widened, so a
    # 100 MB layer is never copied to float64 just to be normalised.
    reference = np.asarray(reference, dtype=latents.dtype)
    dots = (latents @ reference).astype(np.float64)
    squares = np.einsum("ij,ij->i", latents, latents).astype(np.float64)
    norms = np.sqrt(squares) * float(np.linalg.norm(reference.astype(np.float64)))
    out = np.full(latents.shape[0], np.nan)
    np.divide(dots, norms, out=out, where=norms > 0.0)
    return np.clip(out, -1.0, 1.0)


def load_channels(
    source: LatentSource,
    *,
    time: str | int,
    layer: int,
    channels: Sequence[int],
    centred: bool = False,
) -> np.ndarray:
    """A few channels over the whole grid, ``(n_nodes, len(channels))``, ready to
    map: NaN where the grid is invalid, and with each channel's area-weighted
    global mean removed when ``centred``. Only the channels asked for are kept in
    memory; how little is *read* is up to the source (an archive stores a node's
    channels side by side, so its pages are touched all the same)."""
    grid = source.grid()
    values = source.load(time, layer, channels=list(channels)).astype(np.float64)
    if centred:
        values -= grid.mean(values)
    values[~grid.valid] = np.nan
    return values


@dataclass(frozen=True, eq=False)
class RegionAnalysis:
    """Everything ``analyse_region`` found, with what it takes to find it again.

    Per-node arrays span the whole grid and are NaN where the grid is invalid.
    ``similarity_top`` uses only the ranked channels. ``scores`` is
    ``(n_nodes, len(features))``, one column per entry of ``feature_info``, or
    None when no decomposition was asked for; ``pca`` is the basis fitted in the
    region, when that is where the features came from.
    """

    settings: dict[str, Any]
    provenance: dict[str, Any]
    nodes: np.ndarray
    ranking: ChannelRanking
    similarity: np.ndarray
    similarity_top: np.ndarray
    pca: PCA | None = None
    scores: np.ndarray | None = None
    feature_info: tuple[dict[str, Any], ...] = ()

    def summary(self) -> dict[str, Any]:
        """The result minus its large arrays, as plain JSON-ready data: enough to
        rerun the analysis and to check that a rerun agrees."""
        out: dict[str, Any] = {
            "settings": self.settings,
            "provenance": self.provenance,
            "n_region_nodes": int(self.nodes.size),
            "ranking": [
                {"channel": int(c), "peak_abs": float(s), "pinned": int(c) in self.ranking.pinned}
                for c, s in zip(self.ranking.channels, self.ranking.scores, strict=True)
            ],
        }
        if self.scores is not None:
            out["features"] = list(self.feature_info)
        return out


def _feature_info(basis: Decomposition, features: Sequence[int], label: str, **extra: Any):
    loadings = top_loadings(basis, features)
    return tuple(
        {
            "feature": int(f),
            "label": f"{label}{int(f)}",
            **basis.describe(int(f)),
            **{name: float(values[i]) for name, values in extra.items()},
            "top_loadings": [{"channel": c, "loading": v} for c, v in loadings[i]],
        }
        for i, f in enumerate(features)
    )


def analyse_region(
    source: LatentSource,
    *,
    time: str | int,
    layer: int,
    region: Region,
    rank_layer: int | None = None,
    top: int = 15,
    pinned: Sequence[int] = (),
    centred: bool = False,
    reference: str = "nearest",
    n_components: int = 0,
    basis: Decomposition | None = None,
) -> RegionAnalysis:
    """Rank channels in a region, then map similarity and a decomposition.

    Channels are ranked at ``rank_layer`` (the last layer unless told otherwise:
    what the network ends up emphasising) from the region's nodes alone.
    Similarity and the decomposition are computed at ``layer``. The two must be
    equally wide: a channel is followed from one to the other by its index, which
    only means something along a residual stream.

    ``centred`` removes each channel's area-weighted global mean first. Without
    it, channels carrying a large constant offset dominate both the ranking and
    the cosine similarity.

    ``reference`` says what "this region" means as a single vector: ``"nearest"``
    is the valid node closest to the region's centre, ``"mean"`` the area-weighted
    mean over the region. Either is deterministic and independent of node order.

    ``n_components`` features are mapped. Without a ``basis`` they are principal
    components fitted on the region's nodes, area-weighted. With one -- a global
    PCA, a sparse dictionary from ``load_basis`` -- they are its features that
    respond most strongly in the region: by the peak there of what a feature
    *contributes*, its activation times the length of its direction, since a
    dictionary is free to trade one for the other and an activation alone would
    rank by that accident. The basis must have been fitted on this network and
    layer (``check_basis_fits``), and sees the raw latents whatever ``centred``
    says, since it carries the standardisation it was fitted with.

    One layer is in memory at a time, and nothing else of its size: on a 1-degree,
    384-channel model that is about 100 MB however many layers and times exist.
    Everything that can be refused is refused before the first read.
    """
    if reference not in REFERENCES:
        raise RequestError(f"reference must be one of {', '.join(REFERENCES)}; got {reference!r}")
    info, grid = source.info(), source.grid()
    rank_layer = info.last_layer if rank_layer is None else rank_layer
    width, rank_width = info.layer(layer).n_channels, info.layer(rank_layer).n_channels
    if width != rank_width:
        raise RequestError(
            f"layer {layer} has {width} channel(s) and layer {rank_layer}, where channels are "
            f"ranked, has {rank_width}; rank at a layer as wide as the one analysed"
        )
    if basis is not None:
        check_basis_fits(basis, info, layer)
    time_label = info.times[info.time_index(time)]

    nodes = grid.within(region.lat, region.lon, region.radius_km)
    if nodes.size == 0:
        raise RequestError(
            f"no valid nodes within {region.radius_km:g} km of "
            f"({region.lat:g}, {region.lon:g}); widen the region"
        )
    weights = grid.weights()
    if n_components:
        limit = basis.n_features if basis is not None else max(min(nodes.size - 1, width), 0)
        if not 1 <= n_components <= limit:
            what = "feature(s)" if basis is not None else "component(s)"
            hint = "" if basis is not None else " -- widen the region or ask for fewer"
            raise RequestError(
                f"{n_components} {what} asked of {nodes.size} node(s) x {width} channel(s); "
                f"at most {limit} exist{hint}"
            )

    def centre(full: np.ndarray) -> np.ndarray:
        full -= grid.mean(full).astype(full.dtype)  # in place: load() hands over a fresh array
        return full

    ranking = None
    if rank_layer != layer:
        # Centring needs the global mean, so it costs a full read even for a
        # handful of nodes; uncentred, only the region is read. Either way this
        # layer is let go before the next is loaded.
        if centred:
            ranked_at = centre(source.load(time_label, rank_layer))[nodes]
        else:
            ranked_at = source.load(time_label, rank_layer, nodes=nodes)
        ranking = rank_channels(ranked_at, top=top, pinned=pinned)
        del ranked_at

    latents = source.load(time_label, layer)
    scores, pca, feature_info = None, None, ()
    if n_components and basis is not None:
        local = np.abs(basis.transform(latents[nodes])).max(axis=0)
        peak = np.nan_to_num(local * np.linalg.norm(basis.directions(), axis=1), nan=-np.inf)
        features = np.argsort(peak, kind="stable")[::-1][:n_components]
        scores = basis.transform(latents, features=features)
        feature_info = _feature_info(
            basis, features, "F", peak_abs=local[features], peak_contribution=peak[features]
        )
    if centred:
        centre(latents)
    if ranking is None:
        ranking = rank_channels(latents[nodes], top=top, pinned=pinned)

    if reference == "nearest":
        vector = latents[grid.nearest(region.lat, region.lon)]
    else:
        vector = (weights[nodes] / weights[nodes].sum()) @ latents[nodes]
    valid = grid.valid
    similarity = np.where(valid, cosine_similarity(latents, vector), np.nan)
    chosen = ranking.channels
    similarity_top = np.where(valid, cosine_similarity(latents[:, chosen], vector[chosen]), np.nan)

    if n_components and basis is None:
        pca = fit_pca(latents[nodes], n_components, weights=weights[nodes])
        scores = pca.transform(latents)
        feature_info = _feature_info(pca, range(n_components), "PC")
    if scores is not None:
        scores = np.where(valid[:, None], scores, np.nan)

    settings = {
        "time": time_label,
        "layer": layer,
        "rank_layer": rank_layer,
        "region": asdict(region),
        "top": top,
        "pinned": [int(c) for c in pinned],
        "centred": centred,
        "reference": reference,
        "n_components": n_components,
        "basis": None if basis is None else basis.meta.get("path"),
    }
    return RegionAnalysis(
        settings=settings,
        provenance={**info.provenance(), "xaig": __version__},
        nodes=nodes,
        ranking=ranking,
        similarity=similarity,
        similarity_top=similarity_top,
        pca=pca,
        scores=scores,
        feature_info=feature_info,
    )
