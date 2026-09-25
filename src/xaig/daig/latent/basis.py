"""Bases: fitted things that turn a node's channels into features.

A principal-component basis, a sparse dictionary learned by an autoencoder, a
transcoder -- they differ in how they were found and agree in what is done with
them afterwards: project a layer onto them, map a feature, ask which channels it
is made of, push a model along it. ``Decomposition`` is that shared part, so an
analysis and the CLI take "the method" as a value.

A basis is fitted once, often somewhere else (a dictionary is trained with torch,
in ``xaig.taig``), and used many times, so it has a file of its own: one ``.npz``
holding plain arrays and a JSON record of how it was made. It is xaig's own
interchange format, written and read here and nowhere else. It is also the
hand-off *back* to the model's environment, which needs nothing but numpy to read
a row of ``directions`` and steer along it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "daig") from exc

ACTIVATIONS = ("relu", "topk", "bspline")
FORMAT = 1
_BLOCK = 4096  # nodes per step when a whole layer is encoded


@runtime_checkable
class Decomposition(Protocol):
    """A fitted map from channels to features.

    ``transform`` takes *raw* latents, ``(n_nodes, n_channels)``: whatever
    centring or scaling a basis was fitted with is its own business and travels
    with it. ``features`` narrows the result to some columns, which matters for a
    dictionary of thousands: a map of six features must not cost all of them.
    ``directions`` is each feature in channel space, ``(n_features, n_channels)``
    -- what it is made of, and what to add to a layer to steer along it.
    """

    kind: str
    meta: Mapping[str, Any]

    @property
    def n_features(self) -> int: ...

    @property
    def n_channels(self) -> int: ...

    def transform(
        self, latents: np.ndarray, features: Sequence[int] | None = None
    ) -> np.ndarray: ...

    def directions(self) -> np.ndarray: ...

    def describe(self, feature: int) -> dict[str, Any]: ...


def _floating(latents: np.ndarray) -> np.ndarray:
    """At least float32. Archives keep float16, whose squares overflow at 256: a
    memory-mapped layer handed straight to an analysis would otherwise come back
    as infinities, with no warning."""
    latents = np.asarray(latents)
    if latents.dtype == np.float16 or not np.issubdtype(latents.dtype, np.floating):
        return latents.astype(np.float32 if latents.dtype == np.float16 else np.float64)
    return latents


def _chosen(features: Sequence[int] | None, n_features: int) -> np.ndarray | None:
    if features is None:
        return None
    chosen = np.asarray(features, dtype=np.intp).ravel()
    if chosen.size and (chosen.min() < 0 or chosen.max() >= n_features):
        raise RequestError(f"feature(s) outside 0..{n_features - 1}: {chosen.tolist()}")
    return chosen


def top_loadings(basis: Decomposition, features: Sequence[int], n: int = 6):
    """Per feature, the ``n`` channels that weigh most, as (channel, loading)."""
    rows = basis.directions()[np.asarray(features, dtype=np.intp)]
    order = np.argsort(np.abs(rows), axis=1)[:, ::-1][:, :n]
    return [[(int(c), float(row[c])) for c in idx] for row, idx in zip(rows, order, strict=True)]


def fix_signs(components: np.ndarray) -> np.ndarray:
    """Make each row's largest loading positive. A decomposition leaves signs
    arbitrary, and a map that flips colour between two runs of the same analysis
    is not reproducible."""
    lead = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), lead])
    return components * np.where(signs == 0.0, 1.0, signs)[:, None]


# -- principal components ---------------------------------------------------


@dataclass(frozen=True, eq=False)
class PCA:
    """A fitted principal-component basis. ``components`` is ``(k, n_channels)``."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    meta: Mapping[str, Any] = field(default_factory=dict)
    kind = "pca"

    @property
    def n_features(self) -> int:
        return int(self.components.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.components.shape[1])

    def transform(self, latents: np.ndarray, features: Sequence[int] | None = None) -> np.ndarray:
        """Project ``(n_nodes, n_channels)`` onto the components: ``(n_nodes, k)``.

        Centre and project in float64, a bounded block of nodes at a time.
        Subtracting the mean before projection preserves small variations around
        large offsets without allocating a float64 copy of the whole layer.
        """
        latents = np.asarray(latents)
        chosen = _chosen(features, self.n_features)
        components = self.components if chosen is None else self.components[chosen]
        if latents.ndim != 2 or latents.shape[1] != self.n_channels:
            raise RequestError(f"this PCA reads {self.n_channels} channel(s); got {latents.shape}")
        components = components.astype(np.float64)
        out = np.empty((latents.shape[0], components.shape[0]), dtype=np.float64)
        for start in range(0, latents.shape[0], _BLOCK):
            block = latents[start : start + _BLOCK].astype(np.float64) - self.mean
            out[start : start + _BLOCK] = block @ components.T
        return out

    def directions(self) -> np.ndarray:
        return self.components

    def describe(self, feature: int) -> dict[str, Any]:
        return {"explained_variance_ratio": float(self.explained_variance_ratio[feature])}

    def top_loadings(self, n: int = 6) -> list[list[tuple[int, float]]]:
        """Per component, the ``n`` channels that weigh most, as (channel, loading)."""
        return top_loadings(self, range(self.n_features), n)


def fit_pca(latents: np.ndarray, n_components: int, weights: np.ndarray | None = None) -> PCA:
    """Principal components of ``(n_nodes, n_channels)`` by SVD.

    ``weights`` (per node, any scale) make it an area-weighted PCA, which matters
    wherever a region reaches toward a pole and grid rows bunch up. Signs are
    fixed so each component's largest loading is positive.

    Removing the mean costs a degree of freedom: ``n`` nodes carry at most
    ``n - 1`` directions of variance. Zero-weight nodes are excluded, even if
    their values are missing; positive-weight nodes must hold finite values.
    """
    x = np.asarray(latents, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"expected (n_nodes, n_channels), got {x.shape}")
    w = np.ones(x.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)
    if (
        w.shape != (x.shape[0],)
        or not np.isfinite(w).all()
        or not np.all(w >= 0.0)
        or not np.any(w > 0.0)
    ):
        raise ValueError("weights must be non-negative, one per node, and not all zero")
    counted = int(np.count_nonzero(w))
    limit = max(min(counted - 1, x.shape[1]), 0)
    if not 1 <= n_components <= limit:
        raise RequestError(
            f"{n_components} component(s) asked of {counted} node(s) x {x.shape[1]} "
            f"channel(s); at most {limit} exist -- widen the region or ask for fewer"
        )
    positive = w > 0.0
    x, w = x[positive], w[positive]
    if not np.isfinite(x).all():
        raise RequestError("PCA needs finite latents at every positive-weight node")
    w = w / w.max()
    w = w / w.sum()
    mean = w @ x
    _, singular, vt = np.linalg.svd((x - mean) * np.sqrt(w)[:, None], full_matrices=False)
    variance = singular**2
    total = variance.sum()
    ratio = variance / total if total > 0.0 else np.zeros_like(variance)
    return PCA(
        mean=mean,
        components=fix_signs(vt[:n_components].copy()),
        explained_variance_ratio=ratio[:n_components],
    )


# -- sparse dictionaries ----------------------------------------------------


def spline_knots(n_intervals: int, upper: float) -> np.ndarray:
    """Greville abscissae of a uniform cubic B-spline on ``[0, upper]``: the
    coefficients at which the spline is exactly the identity, so a learnable
    activation can start life as a ReLU."""
    step = upper / n_intervals
    return (np.arange(n_intervals + 3) - 1.0) * step


def bspline_activation(z: np.ndarray, coefficients: np.ndarray, upper: float) -> np.ndarray:
    """A learnable activation per feature: zero for ``z <= 0``, a uniform cubic
    B-spline on ``(0, upper]``, and a line of slope one beyond it.

    ``z`` is ``(n, k)`` and ``coefficients`` ``(k, n_intervals + 3)``. The hard zero
    keeps the code sparse whatever the spline learns; with coefficients at
    ``spline_knots`` it is a ReLU, and training bends it from there (a threshold,
    a saturation). Only four basis functions are non-zero anywhere, so they are
    evaluated in closed form rather than by recursion. The torch twin in
    ``xaig.taig`` is tested to agree with this to float precision.
    """
    n_intervals = coefficients.shape[1] - 3
    step = upper / n_intervals
    inside = np.clip(z, 0.0, upper)
    cell = np.minimum((inside / step).astype(np.intp), n_intervals - 1)
    u = inside / step - cell
    blend = (
        (1.0 - u) ** 3 / 6.0,
        (3.0 * u**3 - 6.0 * u**2 + 4.0) / 6.0,
        (-3.0 * u**3 + 3.0 * u**2 + 3.0 * u + 1.0) / 6.0,
        u**3 / 6.0,
    )
    feature = np.arange(coefficients.shape[0])[None, :]
    spline = sum(b * coefficients[feature, cell + m] for m, b in enumerate(blend))
    return np.where(z > 0.0, spline + np.maximum(z - upper, 0.0), 0.0)


def topk_mask(z: np.ndarray, k: int) -> np.ndarray:
    """True at each row's ``k`` largest entries -- exactly ``k``, whatever is tied.

    Everything above the ``k``-th largest value is kept, and the places left over
    go to the entries *equal* to it in order of index. "At least the k-th value"
    keeps every tie instead, so four equal activations would all survive a top-1.
    The rule is spelt with comparisons and a running count so that the torch twin
    in ``xaig.taig`` can spell it identically: the two must never disagree about
    which feature of a tie fired.
    """
    cut = np.partition(z, -k, axis=1)[:, -k][:, None]
    above, tied = z > cut, z == cut
    room = k - above.sum(axis=1, keepdims=True)
    return above | (tied & (np.cumsum(tied, axis=1) <= room))


NODE_NORM_EPS = 1e-5  # as ACE's ChannelLayerNorm


def node_normalise(latents: np.ndarray, eps: float = NODE_NORM_EPS) -> np.ndarray:
    """Each node's vector centred over its channels and scaled to unit RMS: a
    layer norm per node, with no learned scale or offset.

    What a pre-norm network does to its residual stream before each block reads
    it, less the affine the network learns (and may condition on noise). It is
    also the per-node normalisation of MacMillan & Ouellette (2025) up to a
    constant: they scale to unit length, which is this divided by the square root
    of the width. Float32 in, float32 out, computed in float64.
    """
    x = np.asarray(latents, dtype=np.float64)
    centred = x - x.mean(axis=-1, keepdims=True)
    scale = np.sqrt((centred**2).mean(axis=-1, keepdims=True) + eps)
    return (centred / scale).astype(np.float32)


def _node_moments(latents: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(latents, dtype=np.float64)
    mean = x.mean(axis=-1, keepdims=True)
    return mean, np.sqrt(((x - mean) ** 2).mean(axis=-1, keepdims=True) + eps)


@dataclass(frozen=True, eq=False)
class Dictionary:
    """A sparse autoencoder, fitted elsewhere, as plain arrays.

    ``encoder`` and ``decoder`` are ``(n_features, n_channels)``. Inputs are
    standardised as ``(x - input_mean) / input_scale`` before encoding, because
    that is how the dictionary was trained and a feature's activation means
    nothing otherwise. With ``output_*`` set apart from ``input_*`` it is a
    transcoder: it reads one layer and writes another.

    With ``node_norm`` each node is first put through ``node_normalise`` -- the
    dictionary was trained on normalised nodes -- and is still handed raw
    latents: ``reconstruct`` puts each node's own mean and size back.
    """

    encoder: np.ndarray
    encoder_bias: np.ndarray
    decoder: np.ndarray
    decoder_bias: np.ndarray
    input_mean: np.ndarray
    input_scale: float = 1.0
    output_mean: np.ndarray | None = None
    output_scale: float | None = None
    activation: str = "relu"
    k: int | None = None
    spline: np.ndarray | None = None
    spline_upper: float = 1.0
    node_norm: bool = False
    meta: Mapping[str, Any] = field(default_factory=dict)
    kind = "dictionary"

    def __post_init__(self) -> None:
        n_features, n_channels = self.encoder.shape
        if self.activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {', '.join(ACTIVATIONS)}")
        if self.decoder.shape[0] != n_features or self.encoder_bias.shape != (n_features,):
            raise ValueError("encoder, encoder_bias and decoder disagree on the number of features")
        if self.input_mean.shape != (n_channels,):
            raise ValueError("input_mean must hold one value per input channel")
        if self.decoder_bias.shape != (self.decoder.shape[1],):
            raise ValueError("decoder_bias must hold one value per output channel")
        if self.activation == "topk" and not (self.k and 1 <= self.k <= n_features):
            raise ValueError(f"a topk dictionary needs 1 <= k <= {n_features}")
        if self.activation == "bspline" and (
            self.spline is None or self.spline.shape[0] != n_features
        ):
            raise ValueError("a bspline dictionary needs spline coefficients, one row per feature")
        if self.node_norm and self.output_mean is not None:
            raise ValueError("node_norm is for an autoencoder; a transcoder writes another layer")

    @property
    def n_features(self) -> int:
        return int(self.encoder.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.encoder.shape[1])

    def _activate(self, z: np.ndarray, rows: np.ndarray | None) -> np.ndarray:
        if self.activation == "bspline":
            coefficients = self.spline if rows is None else self.spline[rows]
            return bspline_activation(z, coefficients.astype(z.dtype), self.spline_upper)
        z = np.maximum(z, 0.0)
        if self.activation == "topk" and self.k < z.shape[1]:
            z = np.where(topk_mask(z, self.k), z, 0.0)
        return z

    def transform(self, latents: np.ndarray, features: Sequence[int] | None = None) -> np.ndarray:
        """Feature activations, ``(n_nodes, n_features)`` or the columns asked for.

        Encoded a block of nodes at a time, and -- except for ``topk``, where a
        node's features compete -- only through the encoder rows asked for.
        """
        latents = _floating(latents)
        if latents.ndim != 2 or latents.shape[1] != self.n_channels:
            raise RequestError(
                f"this dictionary reads {self.n_channels} channel(s); got {latents.shape}"
            )
        chosen = _chosen(features, self.n_features)
        rows = None if self.activation == "topk" else chosen
        encoder = (self.encoder if rows is None else self.encoder[rows]).astype(np.float32)
        bias = (self.encoder_bias if rows is None else self.encoder_bias[rows]).astype(np.float32)
        mean = self.input_mean.astype(np.float32)
        width = self.n_features if chosen is None else chosen.size
        out = np.empty((latents.shape[0], width), dtype=np.float64)
        for start in range(0, latents.shape[0], _BLOCK):
            block = latents[start : start + _BLOCK].astype(np.float32)
            if self.node_norm:
                block = node_normalise(block)
            block = (block - mean) / self.input_scale
            active = self._activate(block @ encoder.T + bias, rows)
            if rows is None and chosen is not None:
                active = active[:, chosen]
            out[start : start + _BLOCK] = active
        return out

    def reconstruct(self, latents: np.ndarray) -> np.ndarray:
        """What the dictionary makes of ``latents``, back in the units of the layer
        it writes: the input layer for an autoencoder, another for a transcoder."""
        mean = self.input_mean if self.output_mean is None else self.output_mean
        scale = self.input_scale if self.output_scale is None else self.output_scale
        out = np.empty((np.shape(latents)[0], self.decoder.shape[1]), dtype=np.float32)
        for start in range(0, out.shape[0], _BLOCK):
            block = latents[start : start + _BLOCK]
            active = self.transform(block).astype(np.float32)
            rebuilt = (active @ self.decoder + self.decoder_bias) * scale + mean.astype(np.float32)
            if self.node_norm:  # back from each node's normalisation to its own units
                node_mean, node_scale = _node_moments(block, NODE_NORM_EPS)
                rebuilt = rebuilt * node_scale + node_mean
            out[start : start + _BLOCK] = rebuilt
        return out

    def directions(self) -> np.ndarray:
        return self.decoder

    def describe(self, feature: int) -> dict[str, Any]:
        return {"decoder_norm": float(np.linalg.norm(self.decoder[feature]))}


# -- the file ---------------------------------------------------------------

_PCA_ARRAYS = ("mean", "components", "explained_variance_ratio")
_DICTIONARY_ARRAYS = (
    "encoder", "encoder_bias", "decoder", "decoder_bias", "input_mean", "output_mean", "spline",
)  # fmt: skip
_DICTIONARY_SCALARS = (
    "input_scale", "output_scale", "activation", "k", "spline_upper", "node_norm",
)  # fmt: skip


def save_basis(path: str | Path, basis: Decomposition, **meta: Any) -> Path:
    """Write a basis as one ``.npz``: its arrays, plus ``kind`` and a JSON ``meta``
    (the basis's own, with ``meta`` over it -- say what it was fitted on).

    Anything with numpy can read it back, including the environment that runs the
    model: ``np.load(path)["components"][0]`` is a direction to steer along.
    """
    path = Path(path)
    if isinstance(basis, PCA):
        arrays = {name: getattr(basis, name) for name in _PCA_ARRAYS}
        scalars: dict[str, Any] = {}
    elif isinstance(basis, Dictionary):
        arrays = {n: getattr(basis, n) for n in _DICTIONARY_ARRAYS if getattr(basis, n) is not None}
        scalars = {n: getattr(basis, n) for n in _DICTIONARY_SCALARS}
    else:
        raise TypeError(f"do not know how to save a {type(basis).__name__}")
    record = {"format": FORMAT, **scalars, "meta": {**basis.meta, **meta}}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:  # a handle, so numpy cannot append a second ".npz"
        np.savez(handle, kind=np.array(basis.kind), record=np.array(json.dumps(record)), **arrays)
    return path


def load_basis(path: str | Path) -> PCA | Dictionary:
    """Read what ``save_basis`` wrote. The file's own path is noted in ``meta`` so
    a result can say which basis it used."""
    path = Path(path)
    if not path.is_file():
        raise RequestError(f"no basis file: {path}")
    try:
        with np.load(path, allow_pickle=False) as stored:
            kind = str(stored["kind"])
            record = json.loads(str(stored["record"]))
            arrays = {name: stored[name] for name in stored.files if name not in ("kind", "record")}
    except (OSError, KeyError, ValueError) as exc:
        raise RequestError(f"{path} is not a basis file written by xaig ({exc!r})") from exc
    if record.get("format") != FORMAT:
        raise RequestError(f"{path}: basis format {record.get('format')!r}, expected {FORMAT}")
    meta = {**record.pop("meta", {}), "path": str(path)}
    record.pop("format")
    try:
        if kind == "pca":
            return PCA(**arrays, meta=meta)
        if kind == "dictionary":
            return Dictionary(**arrays, **record, meta=meta)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"{path}: not a usable {kind} basis ({exc})") from exc
    raise RequestError(f"{path}: unknown kind of basis {kind!r}")
