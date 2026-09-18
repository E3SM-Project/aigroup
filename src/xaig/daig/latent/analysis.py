"""What a model's internals respond to, region by region.

Pure functions over ``(n_nodes, n_channels)`` arrays, plus one routine that strings
them into the usual question -- *which channels light up here, where else does
the model look like this, and what are the main patterns?* -- and returns a
result carrying the settings and provenance needed to reproduce it.

Nothing here draws, prints, or knows a user interface exists. A notebook, a batch
job, the CLI and a web app all call the same functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from xaig.core.extras import missing_extra
from xaig.daig.latent.source import LatentSource

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
    what is left. Pass the latents of a region to ask what responds *there*.
    """
    latents = np.asarray(latents)
    if latents.ndim != 2 or latents.shape[0] == 0:
        raise ValueError(f"expected (n_nodes, n_channels) with nodes, got {latents.shape}")
    n_channels = latents.shape[1]
    scores = np.nanmax(np.abs(latents), axis=0)
    lead = list(dict.fromkeys(int(c) for c in pinned))
    bad = [c for c in lead if not 0 <= c < n_channels]
    if bad:
        raise ValueError(f"pinned channel(s) {bad} outside 0..{n_channels - 1}")
    ranked = [int(c) for c in np.argsort(scores, kind="stable")[::-1] if int(c) not in lead]
    chosen = np.array((lead + ranked)[: max(top, 0)], dtype=int)
    return ChannelRanking(chosen, scores[chosen], tuple(c for c in lead if c in chosen))


def cosine_similarity(latents: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Cosine of the angle between every node's latent vector and ``reference``.

    A node whose vector is all zeros has no direction, and reads NaN rather than
    raising a warning or borrowing a number -- it is how land looks to an ocean
    model.
    """
    latents = np.asarray(latents)
    if not np.issubdtype(latents.dtype, np.floating):
        latents = latents.astype(np.float64)
    # Stay in the layer's own precision: only per-node results are widened, so a
    # 100 MB layer is never copied to float64 just to be normalised.
    reference = np.asarray(reference, dtype=latents.dtype)
    dots = (latents @ reference).astype(np.float64)
    squares = np.einsum("ij,ij->i", latents, latents).astype(np.float64)
    norms = np.sqrt(squares) * float(np.linalg.norm(reference.astype(np.float64)))
    out = np.full(latents.shape[0], np.nan)
    np.divide(dots, norms, out=out, where=norms > 0.0)
    return np.clip(out, -1.0, 1.0)


@dataclass(frozen=True, eq=False)
class PCA:
    """A fitted principal-component basis. ``components`` is ``(k, n_channels)``."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray

    def transform(self, latents: np.ndarray) -> np.ndarray:
        """Project ``(n_nodes, n_channels)`` onto the components: ``(n_nodes, k)``.

        Computed as ``X @ C.T - mean @ C.T`` in the latents' own precision, so
        projecting a whole layer allocates the ``(n_nodes, k)`` result and nothing
        the size of the layer.
        """
        latents = np.asarray(latents)
        if not np.issubdtype(latents.dtype, np.floating):
            latents = latents.astype(np.float64)
        projected = (latents @ self.components.T.astype(latents.dtype)).astype(np.float64)
        return projected - self.mean @ self.components.T

    def top_loadings(self, n: int = 6) -> list[list[tuple[int, float]]]:
        """Per component, the ``n`` channels that weigh most, as (channel, loading)."""
        order = np.argsort(np.abs(self.components), axis=1)[:, ::-1][:, :n]
        return [
            [(int(c), float(row[c])) for c in idx]
            for row, idx in zip(self.components, order, strict=True)
        ]


def fit_pca(latents: np.ndarray, n_components: int, weights: np.ndarray | None = None) -> PCA:
    """Principal components of ``(n_nodes, n_channels)`` by SVD.

    ``weights`` (per node, any scale) make it an area-weighted PCA, which matters
    wherever a region reaches toward a pole and grid rows bunch up. Signs are
    fixed so each component's largest loading is positive: PCA leaves them
    arbitrary, and a map that flips colour between two runs of the same analysis
    is not reproducible.
    """
    x = np.asarray(latents, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_nodes, n_channels), got {x.shape}")
    limit = min(x.shape)
    if not 1 <= n_components <= limit:
        raise ValueError(
            f"{n_components} component(s) asked of {x.shape[0]} node(s) x {x.shape[1]} "
            f"channel(s); at most {limit} exist -- widen the region or ask for fewer"
        )
    w = np.ones(x.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (x.shape[0],) or not np.all(w >= 0.0) or w.sum() <= 0.0:
        raise ValueError("weights must be non-negative, one per node, and not all zero")
    w = w / w.sum()
    mean = w @ x
    _, singular, vt = np.linalg.svd((x - mean) * np.sqrt(w)[:, None], full_matrices=False)
    variance = singular**2
    total = variance.sum()
    ratio = variance / total if total > 0.0 else np.zeros_like(variance)
    components = vt[:n_components].copy()
    lead = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(n_components), lead])
    components *= np.where(signs == 0.0, 1.0, signs)[:, None]
    return PCA(mean=mean, components=components, explained_variance_ratio=ratio[:n_components])


@dataclass(frozen=True, eq=False)
class RegionAnalysis:
    """Everything ``analyse_region`` found, with what it takes to find it again.

    Per-node arrays span the whole grid and are NaN where the grid is invalid.
    ``similarity_top`` uses only the ranked channels; ``scores`` is
    ``(n_nodes, n_components)``, or None when no PCA was asked for.
    """

    settings: dict[str, Any]
    provenance: dict[str, Any]
    nodes: np.ndarray
    ranking: ChannelRanking
    similarity: np.ndarray
    similarity_top: np.ndarray
    pca: PCA | None = None
    scores: np.ndarray | None = None

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
        if self.pca is not None:
            out["pca"] = [
                {
                    "component": k,
                    "explained_variance_ratio": float(ratio),
                    "top_loadings": [{"channel": c, "loading": v} for c, v in loadings],
                }
                for k, (ratio, loadings) in enumerate(
                    zip(self.pca.explained_variance_ratio, self.pca.top_loadings(), strict=True)
                )
            ]
        return out


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
) -> RegionAnalysis:
    """Rank channels in a region, then map similarity and principal components.

    Channels are ranked at ``rank_layer`` (the last layer unless told otherwise:
    what the network ends up emphasising) from the region's nodes alone.
    Similarity and PCA are computed at ``layer``.

    ``centred`` removes each channel's area-weighted global mean first. Without
    it, channels carrying a large constant offset dominate both the ranking and
    the cosine similarity.

    ``reference`` says what "this region" means as a single vector: ``"nearest"``
    is the valid node closest to the region's centre, ``"mean"`` the area-weighted
    mean over the region. Either is deterministic and independent of node order.

    One layer is in memory at a time, and nothing else of its size: on a 1-degree,
    384-channel model that is about 100 MB however many layers and times exist.
    """
    if reference not in REFERENCES:
        raise ValueError(f"reference must be one of {', '.join(REFERENCES)}; got {reference!r}")
    info, grid = source.info(), source.grid()
    rank_layer = info.last_layer if rank_layer is None else rank_layer
    for index in (layer, rank_layer):
        info.layer(index)
    time_label = info.times[info.time_index(time)]

    nodes = grid.within(region.lat, region.lon, region.radius_km)
    if nodes.size == 0:
        raise ValueError(
            f"no valid nodes within {region.radius_km:g} km of "
            f"({region.lat:g}, {region.lon:g}); widen the region"
        )
    weights = grid.weights()

    def load(index: int, only: np.ndarray | None = None) -> np.ndarray:
        """A layer, or some of its nodes, centred if asked. Centring needs the
        global mean, so it costs a full read even for a handful of nodes."""
        if not centred:
            return source.load(time_label, index, nodes=only)
        full = source.load(time_label, index)
        full -= grid.mean(full).astype(full.dtype)  # in place: load() hands over a fresh array
        return full if only is None else full[only]

    ranking = rank_channels(load(rank_layer, nodes), top=top, pinned=pinned)

    latents = load(layer)
    if reference == "nearest":
        vector = latents[grid.nearest(region.lat, region.lon)]
    else:
        vector = (weights[nodes] / weights[nodes].sum()) @ latents[nodes]
    valid = grid.valid
    similarity = np.where(valid, cosine_similarity(latents, vector), np.nan)
    chosen = ranking.channels
    similarity_top = np.where(valid, cosine_similarity(latents[:, chosen], vector[chosen]), np.nan)

    pca = scores = None
    if n_components:
        pca = fit_pca(latents[nodes], n_components, weights=weights[nodes])
        scores = np.where(valid[:, None], pca.transform(latents), np.nan)

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
    }
    return RegionAnalysis(
        settings=settings,
        provenance=info.provenance(),
        nodes=nodes,
        ranking=ranking,
        similarity=similarity,
        similarity_top=similarity_top,
        pca=pca,
        scores=scores,
    )
