"""Many times at once: moments of a layer, and batches to train on.

``analyse_region`` looks at one time. Fitting something *to a model* -- a global
PCA, a sparse dictionary -- wants every node of every time, which is a million
vectors and does not need to be in memory: moments accumulate, and batches are
drawn a time at a time.

The two rules of ``daig.grid`` hold here as well, and are as easy to forget:

- **Area.** A 1-degree grid has as many nodes in its last row as on the equator,
  covering 1/115 of the area. Nodes are therefore weighted by area in the
  moments and *drawn* by area in the batches, so a plain mean over a batch is
  already the area-weighted loss.
- **Mask.** Invalid nodes are never counted and never drawn. A third of an ocean
  model's nodes are land, all zeros, and a dictionary trained on them spends its
  features describing nothing.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra
from xaig.daig.latent.basis import PCA, fix_signs
from xaig.daig.latent.source import LatentSource

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "daig") from exc

_BLOCK = 8192


def _time_labels(source: LatentSource, times: Sequence[str | int] | None) -> list[str]:
    info = source.info()
    if times is None:
        return list(info.times)
    return [info.times[info.time_index(t)] for t in times]


@dataclass(frozen=True, eq=False)
class Moments:
    """Area-weighted first and second moments of a layer's channels, over every
    valid node of the times given."""

    mean: np.ndarray
    covariance: np.ndarray
    times: tuple[str, ...]
    layer: int

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.clip(np.diag(self.covariance), 0.0, None))

    @property
    def scale(self) -> float:
        """One number to divide a centred layer by so that a node's vector has
        unit mean square per channel: what a dictionary is trained on, keeping the
        channels' relative sizes (a per-channel scale would not)."""
        return float(np.sqrt(np.trace(self.covariance) / self.covariance.shape[0])) or 1.0


def accumulate_moments(
    source: LatentSource, *, layer: int, times: Sequence[str | int] | None = None
) -> Moments:
    """Moments of ``layer`` over ``times`` (all of them by default).

    Accumulated in float64 a block of nodes at a time: one layer is in memory,
    and the sums are ``(n_channels, n_channels)`` however many times there are.
    """
    info, labels = source.info(), _time_labels(source, times)
    if not labels:
        raise RequestError("no times to accumulate over")
    n_channels = info.layer(layer).n_channels
    weights = source.grid().weights() / len(labels)
    first = np.zeros(n_channels)
    second = np.zeros((n_channels, n_channels))
    for label in labels:
        latents = source.load(label, layer)
        for start in range(0, latents.shape[0], _BLOCK):
            keep = np.flatnonzero(weights[start : start + _BLOCK] > 0.0) + start
            if keep.size:
                block = latents[keep].astype(np.float64)
                first += weights[keep] @ block
                second += (block * weights[keep, None]).T @ block
    return Moments(
        mean=first, covariance=second - np.outer(first, first), times=tuple(labels), layer=layer
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
        meta={"fitted_on": {"layer": moments.layer, "times": list(moments.times)}},
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
    if not labels:
        raise RequestError("no times to draw batches from")
    if batch_size < 1:
        raise RequestError("batch_size must be at least 1")
    probability = source.grid().weights()
    n_draws = int(np.count_nonzero(probability))
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        for position in rng.permutation(len(labels)):
            label = labels[position]
            nodes = rng.choice(probability.size, size=n_draws, replace=True, p=probability)
            inputs = source.load(label, layer, nodes=nodes)
            if target_layer is None:
                targets = None
            else:
                targets = source.load(label, target_layer, nodes=nodes)
            for start in range(0, n_draws, batch_size):
                stop = start + batch_size
                if targets is None:
                    yield inputs[start:stop]
                else:
                    yield inputs[start:stop], targets[start:stop]
