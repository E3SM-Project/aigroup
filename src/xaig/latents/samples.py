"""Many times at once: moments of a layer, and batches to train on.

``analyze_region`` looks at one time. Fitting something *to a model* -- a global
PCA, a sparse dictionary -- wants every node of every time, which is a million
vectors and does not need to be in memory: moments accumulate, and batches are
drawn a time at a time.

The two rules of ``latents.grid`` hold here as well, and are as easy to forget:

- **Area.** A 1-degree grid has as many nodes in its last row as on the equator,
  covering 1/115 of the area. Nodes are therefore weighted by area in the
  moments and *drawn* by area in the batches, so a plain mean over a batch is
  already the area-weighted loss.
- **Mask.** Invalid nodes are never counted and never drawn. A third of an ocean
  model's nodes are land, and the network does not leave them empty: on the
  SamudrACE-E3SMv3 ocean its last layer is *larger* over land than over sea (RMS
  0.60 against 0.50). Nothing in the latents marks those nodes as meaningless, so
  a source opened without its mask trains a dictionary on them and cannot warn.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra
from xaig.latents.basis import PCA, fix_signs
from xaig.latents.source import LatentSource, read_latents

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

_BLOCK = 8192


def _time_labels(source: LatentSource, times: Sequence[str | int] | None) -> list[str]:
    """The labels of the times asked for (all by default); none is refused."""
    info = source.info()
    labels = list(info.times) if times is None else [info.times[info.time_index(t)] for t in times]
    if not labels:
        raise RequestError(f"no times selected from {info.source}")
    return labels


@dataclass(frozen=True, eq=False)
class Moments:
    """Area-weighted first and second moments of a layer's channels, over every
    valid node of the times given."""

    mean: np.ndarray
    covariance: np.ndarray
    times: tuple[str, ...]
    layer: int
    provenance: Mapping[str, Any] = field(default_factory=dict)
    network_layer: int | None = None

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    @property
    def scale(self) -> float:
        """One number to divide a centered layer by so that a node's vector has
        unit mean square per channel: what a dictionary is trained on, keeping the
        channels' relative sizes (a per-channel scale would not)."""
        return float(np.sqrt(np.trace(self.covariance) / self.covariance.shape[0]))


def accumulate_moments(
    source: LatentSource, *, layer: int, times: Sequence[str | int] | None = None
) -> Moments:
    """Moments of ``layer`` over ``times`` (all of them by default).

    Accumulated in float64 a block of nodes at a time: one layer is in memory,
    and the sums are ``(n_channels, n_channels)`` however many times there are.
    The covariance is built from centered sums, each block merged into the running
    mean and scatter as it arrives, rather than as ``E[x x'] - E[x] E[x]'``, which
    loses every digit of a channel whose mean dwarfs its spread.
    """
    info, labels = source.info(), _time_labels(source, times)
    n_channels = info.layer(layer).n_channels
    weights = source.grid().weights() / len(labels)
    total = 0.0
    shift = None  # one node's vector, taken off everything: a constant channel is then exactly 0
    mean = np.zeros(n_channels)
    scatter = np.zeros((n_channels, n_channels))  # weighted sum of centered outer products
    for label in labels:
        latents = read_latents(source, label, layer)
        for start in range(0, latents.shape[0], _BLOCK):
            keep = np.flatnonzero(weights[start : start + _BLOCK] > 0.0) + start
            if keep.size:
                block = latents[keep].astype(np.float64)
                if shift is None:
                    shift = block[0].copy()
                block -= shift
                w = weights[keep]
                block_weight = float(w.sum())
                block_mean = (w @ block) / block_weight
                block -= block_mean
                block_scatter = (block * w[:, None]).T @ block
                merged = total + block_weight
                gap = block_mean - mean
                scatter += block_scatter + np.outer(gap, gap) * (total * block_weight / merged)
                mean += gap * (block_weight / merged)
                total = merged
    return Moments(
        mean=mean + (0.0 if shift is None else shift),
        covariance=scatter / total,
        times=tuple(labels),
        layer=layer,
        provenance=info.provenance(),
        network_layer=info.layer(layer).position,
    )


def pca_from_moments(moments: Moments, n_components: int) -> PCA:
    """Principal components from a covariance: the global counterpart of
    ``fit_pca``, for when the nodes are too many to decompose directly."""
    n_channels = moments.covariance.shape[0]
    if not 1 <= n_components <= n_channels:
        raise RequestError(
            f"{n_components} component(s) asked of {n_channels} channel(s); "
            f"at most {n_channels} exist"
        )
    variance, vectors = np.linalg.eigh(moments.covariance)
    order = np.argsort(variance)[::-1][:n_components]
    total = float(np.clip(variance, 0.0, None).sum())
    ratio = np.clip(variance[order], 0.0, None) / total if total > 0.0 else np.zeros(order.size)
    return PCA(
        mean=moments.mean,
        components=fix_signs(vectors[:, order].T.copy()),
        explained_variance_ratio=ratio,
        meta={
            "fitted_on": {
                "provenance": dict(moments.provenance),
                "layer": moments.layer,
                "network_layer": moments.network_layer,
                "times": list(moments.times),
            }
        },
    )


def iter_batches(
    source: LatentSource,
    *,
    layer: int,
    batch_size: int = 4096,
    times: Sequence[str | int] | None = None,
    target_layer: int | None = None,
    epochs: int = 1,
    seed: int = 0,
) -> Iterator[np.ndarray | tuple[np.ndarray, np.ndarray]]:
    """Batches of node vectors from ``layer``, float32 ``(batch_size, n_channels)``.

    Each epoch visits every time once, in a shuffled order, and draws as many
    nodes from it as it has valid ones -- with replacement, in proportion to area.
    One time's draws are in memory at once, so a batch comes from a single time;
    its nodes are spread over the globe, which is where most of the variety is.

    With ``target_layer`` the batches are ``(inputs, targets)`` pairs of the same
    nodes at two layers, which is what a transcoder trains on.
    """
    labels = _time_labels(source, times)
    if batch_size < 1:
        raise RequestError("batch_size must be at least 1")
    probability = source.grid().weights()
    n_draws = int(np.count_nonzero(probability))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for position in rng.permutation(len(labels)):
            label = labels[position]
            nodes = rng.choice(probability.size, size=n_draws, replace=True, p=probability)
            inputs = read_latents(source, label, layer, nodes=nodes)
            if target_layer is None:
                targets = None
            else:
                targets = read_latents(source, label, target_layer, nodes=nodes)
            for start in range(0, n_draws, batch_size):
                stop = start + batch_size
                if targets is None:
                    yield inputs[start:stop]
                else:
                    yield inputs[start:stop], targets[start:stop]
