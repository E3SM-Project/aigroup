"""Evaluating a frozen basis on times it was not fitted on.

A dictionary fitted and scored on the same nodes says how well it memorised them. The
protocol here is the smallest that says more:

1. **Split by time, not by node.** Neighbouring nodes and neighbouring steps are
   nearly the same sample, so a random split leaks. ``split_time_blocks`` cuts the
   times into contiguous blocks, holds some out, and drops a buffer of ``gap`` times
   beside every held-out block from the training side. ``split_groups`` holds out whole
   trajectories, ``split_archives`` a whole other archive. A split is a list of time
   labels, written into every result.
2. **Fit on the training side only.** Standardisation (``accumulate_moments``), a PCA, a
   dictionary (``fit_sae(times=split.train)``): all of it. ``evaluate_basis`` refuses a
   basis that says it saw a held-out time.
3. **Freeze it and measure both sides**: reconstruction, sparsity, dead features,
   redundancy. The training numbers are there to be compared with, not reported.

Beside that, ``seed_stability`` asks whether two trainings found the same features, and
``fidelity_curve`` sets a PCA's explained variance against a dictionary's at the same
sparsity. Everything returns objects, prints nothing, and ``to_dict`` makes plain JSON.
Nothing here trains: seeds and sweeps are ``xaig.nn``'s, and arrive as ``Dictionary``s.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xaig import __version__
from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra
from xaig.latents.basis import (
    PCA,
    Decomposition,
    Dictionary,
    basis_provenance,
    result_provenance,
)
from xaig.latents.samples import accumulate_moments, pca_from_moments
from xaig.latents.source import (
    LatentSource,
    check_basis_fits,
    check_comparable,
    differing_identity,
    read_latents,
)

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

_BLOCK = 4096  # nodes per step when a layer is encoded or compared


def jsonable(value: Any) -> Any:
    """JSON-ready: plain Python, and None where a number is not one (JSON has no NaN)."""
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _summary(values: np.ndarray) -> dict[str, float | None]:
    """The distribution of a set of numbers, in a few of them."""
    if values.size == 0:
        return {}
    quantiles = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
    return {
        "min": float(values.min()), "p10": float(quantiles[0]), "p25": float(quantiles[1]),
        "median": float(quantiles[2]), "p75": float(quantiles[3]), "p90": float(quantiles[4]),
        "max": float(values.max()), "mean": float(values.mean()),
    }  # fmt: skip


def save_result(path: str | Path, result: Any) -> Path:
    """Write what an evaluation returned as JSON: its settings, split, numbers and
    provenance. Anything here with a ``to_dict`` will do."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(result.to_dict()), indent=2) + "\n")
    return path


# -- splits -------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """Which times a basis is fitted on, which it is scored on, and which are left out
    between them. Explicit labels, not a rule: a result that says how it was split
    says it by listing the times.

    ``scheme`` is ``"blocks"``, ``"groups"`` or ``"archives"``; for the last the
    held-out times are another archive's, named by ``test_source``.
    """

    scheme: str
    train: tuple[str, ...]
    test: tuple[str, ...]
    buffer: tuple[str, ...] = ()
    gap: int = 0
    test_source: str | None = None
    detail: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scheme": self.scheme, "train": list(self.train), "test": list(self.test),
            "buffer": list(self.buffer), "gap": self.gap, **self.detail,
        }  # fmt: skip
        if self.test_source is not None:
            out["test_source"] = self.test_source
        return out


def _unique(times: Sequence[str]) -> list[str]:
    labels = [str(t) for t in times]
    if len(set(labels)) != len(labels):
        raise RequestError("times must be distinct labels to be split")
    return labels


def _nonempty(split: Split) -> Split:
    if not split.train:
        raise RequestError(
            f"the {split.scheme} split leaves nothing to fit on ({len(split.test)} held out, "
            f"{len(split.buffer)} in the buffer): hold out less, or shrink the gap"
        )
    if not split.test:
        raise RequestError(f"the {split.scheme} split holds nothing out")
    return split


def split_time_blocks(
    times: Sequence[str],
    *,
    n_blocks: int = 5,
    test_blocks: Sequence[int] = (-1,),
    gap: int = 1,
) -> Split:
    """Cut ``times`` (in time order) into ``n_blocks`` contiguous, near-equal blocks and
    hold ``test_blocks`` out. A training time within ``gap`` positions of a held-out one
    goes to the buffer instead, on both sides of the block: at least one, so that no
    training step is adjacent to a held-out one.

    The buffer is in positions, not seconds: with an archive that keeps steps 1-4 and 9-12
    a gap of 1 already spans the unkept steps between them.
    """
    labels = _unique(times)
    if gap < 1:
        raise RequestError("gap must be at least 1: a training step may not touch a held-out one")
    if not 2 <= n_blocks <= len(labels):
        raise RequestError(f"{len(labels)} time(s) cannot be cut into {n_blocks} block(s)")
    chosen = set()
    for block in test_blocks:
        if not -n_blocks <= block < n_blocks:
            raise RequestError(f"no block {block}; blocks are 0..{n_blocks - 1}")
        chosen.add(block % n_blocks)
    blocks = np.array_split(np.arange(len(labels)), n_blocks)
    held = np.concatenate([blocks[b] for b in sorted(chosen)])
    near = np.zeros(len(labels), dtype=bool)
    for position in held:
        near[max(position - gap, 0) : position + gap + 1] = True
    is_test = np.zeros(len(labels), dtype=bool)
    is_test[held] = True
    return _nonempty(
        Split(
            scheme="blocks",
            train=tuple(labels[i] for i in np.flatnonzero(~near)),
            test=tuple(labels[i] for i in np.flatnonzero(is_test)),
            buffer=tuple(labels[i] for i in np.flatnonzero(near & ~is_test)),
            gap=gap,
            detail={
                "n_blocks": n_blocks,
                "test_blocks": sorted(chosen),
                "blocks": [[labels[i] for i in block] for block in blocks],
            },
        )
    )


def split_groups(
    times: Sequence[str], groups: Mapping[str, str], test_groups: Sequence[str]
) -> Split:
    """Hold out whole trajectories (or any grouping of times: one ensemble member, one
    season). ``groups`` names the group of every time. Times of different groups share
    no buffer, which is right only if the groups really are independent: members that
    branch from one initial state are not."""
    labels = _unique(times)
    missing = [t for t in labels if t not in groups]
    if missing:
        raise RequestError(f"no group given for {len(missing)} time(s), the first {missing[0]!r}")
    named = sorted({groups[t] for t in labels})
    unknown = [g for g in test_groups if g not in named]
    if unknown:
        raise RequestError(f"no group {unknown[0]!r}; groups are {', '.join(named)}")
    if not test_groups:
        raise RequestError("test_groups is empty: nothing would be held out")
    held = set(test_groups)
    return _nonempty(
        Split(
            scheme="groups",
            train=tuple(t for t in labels if groups[t] not in held),
            test=tuple(t for t in labels if groups[t] in held),
            detail={"test_groups": sorted(held), "groups": {t: groups[t] for t in labels}},
        )
    )


def split_archives(train: LatentSource, test: LatentSource, *, layer: int) -> Split:
    """Fit on every time of one archive and score on every time of another. The two must
    be the same network at the same place, on one grid (``check_comparable``): they are
    different runs of it, such as another initial state or another period."""
    check_comparable(train, test, layer=layer)
    return _nonempty(
        Split(
            scheme="archives",
            train=tuple(train.info().times),
            test=tuple(test.info().times),
            test_source=test.info().source,
        )
    )


# -- measuring one basis on some times ----------------------------------------------


@dataclass(frozen=True, eq=False)
class SideMetrics:
    """One basis on one side of a split, area-weighted, every time counting equally.

    ``explained_variance`` is one minus the error over the variance of the target about
    the *training* mean the basis carries, summed over channels; ``mse`` is the mean
    squared error per channel, in the layer's own units. A feature is *active* at a node
    where the magnitude of its activation exceeds ``active_above`` (zero by default: any
    activation at all). ``firing_rate`` is, per feature, the share of the area
    it is active over; ``dead`` are the features that never were.
    """

    times: tuple[str, ...]
    explained_variance: float
    mse: float
    mean_active_features: float
    firing_rate: np.ndarray

    @property
    def l0_fraction(self) -> float:
        return self.mean_active_features / self.firing_rate.size

    @property
    def dead(self) -> tuple[int, ...]:
        return tuple(int(i) for i in np.flatnonzero(self.firing_rate == 0.0))

    @property
    def dead_fraction(self) -> float:
        return len(self.dead) / self.firing_rate.size

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_times": len(self.times),
            "explained_variance": self.explained_variance,
            "mse": self.mse,
            "mean_active_features": self.mean_active_features,
            "l0_fraction": self.l0_fraction,
            "dead_fraction": self.dead_fraction,
            "dead": list(self.dead),
            "firing_rate": _summary(self.firing_rate),
        }


def _reference_mean(basis: Decomposition) -> np.ndarray:
    """What a basis is centred on in the layer it writes."""
    if isinstance(basis, PCA):
        return basis.mean
    out = basis.output_mean if basis.output_mean is not None else basis.input_mean
    return np.asarray(out, dtype=np.float64)


def _target_layer(basis: Decomposition, target_layer: int | None) -> int | None:
    fitted = (basis.meta.get("fitted_on") or {}).get("target_layer")
    target_layer = fitted if target_layer is None else target_layer
    writes_other = isinstance(basis, Dictionary) and basis.output_mean is not None
    if writes_other and target_layer is None:
        raise RequestError("this dictionary writes another layer (a transcoder): give target_layer")
    if not writes_other and target_layer is not None:
        raise RequestError("target_layer is for a transcoder; this basis rebuilds its own layer")
    return target_layer


def measure(
    basis: Decomposition,
    source: LatentSource,
    times: Sequence[str | int],
    *,
    layer: int,
    target_layer: int | None = None,
    active_above: float = 0.0,
) -> SideMetrics:
    """``basis`` frozen, over ``times`` of ``source``.

    A block of nodes at a time, valid nodes only, weighted by area; each time's weights
    sum to one and the times are averaged. Read through ``read_latents``, so a
    non-finite activation is refused, not carried into the mean.
    """
    if not hasattr(basis, "decode"):
        raise RequestError(f"cannot reconstruct with a {type(basis).__name__}: it has no decoder")
    if not times:
        raise RequestError("no times to measure on")
    info, weights = source.info(), source.grid().weights()
    keep = np.flatnonzero(weights > 0.0)
    reference = _reference_mean(basis)
    error = power = active = 0.0
    rate = np.zeros(basis.n_features)
    n_times = len(times)
    for time in times:
        x = read_latents(source, time, layer)[keep]
        target = x if target_layer is None else read_latents(source, time, target_layer)[keep]
        if target.shape[1] != reference.size:
            raise RequestError(
                f"layer {target_layer if target_layer is not None else layer} has "
                f"{target.shape[1]} channel(s); the basis writes {reference.size}"
            )
        w = weights[keep] / n_times
        for start in range(0, keep.size, _BLOCK):
            part = slice(start, start + _BLOCK)
            scores = basis.transform(x[part])
            wanted = target[part].astype(np.float64)
            error += float(w[part] @ ((wanted - basis.decode(scores)) ** 2).sum(axis=1))
            power += float(w[part] @ ((wanted - reference) ** 2).sum(axis=1))
            fired = np.abs(scores) > active_above
            active += float(w[part] @ fired.sum(axis=1))
            rate += w[part] @ fired
    return SideMetrics(
        times=tuple(info.times[info.time_index(t)] for t in times),
        explained_variance=1.0 - error / power if power > 0.0 else float("nan"),
        mse=error / reference.size,
        mean_active_features=active,
        firing_rate=rate,
    )


# -- redundancy ---------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Redundancy:
    """How alike a basis's feature directions are, by signed cosine (a feature that
    writes the opposite of another is not its copy: activations are not negative).

    ``max_cosine`` is, per feature, its closest other feature's cosine and ``nearest``
    that feature; ``n_near_duplicates`` counts features with one at or above
    ``threshold``, and ``n_pairs`` the pairs. A zero direction has cosine zero with
    everything and is counted in ``n_zero``. It is a property of the directions, not of
    the data: dead features are in it.
    """

    max_cosine: np.ndarray
    nearest: np.ndarray
    threshold: float
    n_near_duplicates: int
    n_pairs: int
    n_zero: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_features": int(self.max_cosine.size),
            "n_near_duplicates": self.n_near_duplicates,
            "n_pairs": self.n_pairs,
            "n_zero_directions": self.n_zero,
            "max_cosine": _summary(self.max_cosine),
        }


_ROUNDING = 1e-6  # a copy's cosine is 1 only to within float32 rounding


def _unit_rows(directions: np.ndarray) -> tuple[np.ndarray, int]:
    rows = np.asarray(directions, dtype=np.float32)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    return np.divide(rows, norm, out=np.zeros_like(rows), where=norm > 0.0), int(
        np.count_nonzero(norm == 0.0)
    )


def redundancy(basis: Decomposition, *, threshold: float = 0.95) -> Redundancy:
    """The closest pair structure of ``basis.directions()``. Blocked, so a dictionary of
    thousands of features never holds its whole cosine matrix (one block of rows
    against all of them)."""
    if not -1.0 < threshold <= 1.0:
        raise RequestError("threshold is a cosine: above -1 and at most 1")
    unit, n_zero = _unit_rows(basis.directions())
    n = unit.shape[0]
    if n < 2:  # a lone direction has nothing to resemble
        return Redundancy(np.full(n, np.nan), np.full(n, -1), threshold, 0, 0, n_zero)
    best, nearest, pairs = np.empty(n), np.empty(n, dtype=np.intp), 0
    for start in range(0, n, _BLOCK):
        cosine = unit[start : start + _BLOCK] @ unit.T
        rows = np.arange(cosine.shape[0])
        cosine[rows, rows + start] = -np.inf  # itself
        nearest[start : start + _BLOCK] = cosine.argmax(axis=1)
        best[start : start + _BLOCK] = cosine.max(axis=1)
        pairs += int(np.count_nonzero(cosine >= threshold - _ROUNDING))
    return Redundancy(
        max_cosine=best,
        nearest=nearest,
        threshold=threshold,
        n_near_duplicates=int(np.count_nonzero(best >= threshold - _ROUNDING)),
        n_pairs=pairs // 2,
        n_zero=n_zero,
    )


# -- one frozen basis, both sides ---------------------------------------------------


@dataclass(frozen=True, eq=False)
class Evaluation:
    """A frozen basis, measured on the times it was fitted on and the times it was not.

    ``fitted_on`` says how the basis relates to the split: ``"train"`` (it was fitted on
    exactly the training times, or the training archive), ``"other"`` (on other times or
    another archive, none of them held out) or ``"unknown"`` (it does not say, so the
    split cannot vouch that the test is held out).
    """

    kind: str
    n_features: int
    layer: int
    target_layer: int | None
    split: Split
    train: SideMetrics
    test: SideMetrics
    redundancy: Redundancy
    fitted_on: str
    provenance: Mapping[str, Any]

    @property
    def dormant(self) -> tuple[int, ...]:
        """Features that fired on the training side and never on the held-out side."""
        gone = set(self.test.dead) - set(self.train.dead)
        return tuple(sorted(gone))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "n_features": self.n_features,
            "layer": self.layer,
            "target_layer": self.target_layer,
            "fitted_on": self.fitted_on,
            "split": self.split.to_dict(),
            "train": self.train.to_dict(),
            "test": self.test.to_dict(),
            "dormant": list(self.dormant),
            "redundancy": self.redundancy.to_dict(),
            "provenance": dict(self.provenance),
        }


def _check_leak(basis: Decomposition, split: Split, source: str) -> str:
    """How the basis relates to the split, or a ``RequestError`` if it saw the held-out
    side. Said by the basis's own record, which is all there is to go on."""
    fitted = basis.meta.get("fitted_on") or {}
    seen = fitted.get("times")
    if split.scheme == "archives":
        from_source = (fitted.get("provenance") or {}).get("source")
        if from_source is not None and from_source == split.test_source:
            raise RequestError(
                f"the basis was fitted on {from_source}, which is the held-out archive"
            )
        if from_source is None:
            return "unknown"
        return "train" if from_source == source else "other"
    if seen is None:
        return "unknown"
    leaked = sorted(set(map(str, seen)) & set(split.test))
    if leaked:
        raise RequestError(
            f"the basis was fitted on {len(leaked)} held-out time(s), the first {leaked[0]!r}; "
            "refit it on split.train"
        )
    return "train" if set(map(str, seen)) == set(split.train) else "other"


def evaluate_basis(
    basis: Decomposition,
    source: LatentSource,
    split: Split,
    *,
    layer: int,
    target_layer: int | None = None,
    test_source: LatentSource | None = None,
    active_above: float = 0.0,
    duplicate_threshold: float = 0.95,
    allow_unverified_basis: bool = False,
) -> Evaluation:
    """``basis`` frozen, on both sides of ``split``: reconstruction, sparsity, dead
    features and redundancy. The basis must have been fitted on the training side
    (refused if it records a held-out time) and must fit the layer
    (``check_basis_fits``). With a ``split_archives`` split, ``test_source`` is the
    archive the held-out times belong to.
    """
    if (split.scheme == "archives") != (test_source is not None):
        raise RequestError("test_source goes with, and only with, a split by archives")
    info = source.info()
    check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
    target_layer = _target_layer(basis, target_layer)
    fitted_on = _check_leak(basis, split, info.source)
    provenance = result_provenance(info, basis=basis)
    if test_source is not None:
        provenance["test_source"] = result_provenance(test_source.info())
    common = {"layer": layer, "target_layer": target_layer, "active_above": active_above}
    return Evaluation(
        kind=basis.kind,
        n_features=basis.n_features,
        layer=layer,
        target_layer=target_layer,
        split=split,
        train=measure(basis, source, split.train, **common),
        test=measure(basis, test_source or source, split.test, **common),
        redundancy=redundancy(basis, threshold=duplicate_threshold),
        fitted_on=fitted_on,
        provenance=provenance,
    )


# -- do two trainings find the same features -----------------------------------------


def assign(cost: np.ndarray) -> np.ndarray:
    """The Hungarian method: for each row of ``cost`` (``n <= m``), the column it takes,
    all distinct, at least total cost. Shortest augmenting paths with potentials, the
    inner step in numpy -- scipy is not a dependency of this package."""
    cost = np.asarray(cost, dtype=np.float64)
    n, m = cost.shape
    if n > m:
        raise ValueError("assign wants no more rows than columns")
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    owner, way = np.zeros(m + 1, dtype=np.intp), np.zeros(m + 1, dtype=np.intp)
    for row in range(1, n + 1):
        owner[0], column = row, 0
        slack, used = np.full(m + 1, np.inf), np.zeros(m + 1, dtype=bool)
        while True:
            used[column] = True
            reduced = cost[owner[column] - 1] - u[owner[column]] - v[1:]
            free = ~used[1:]
            closer = free & (reduced < slack[1:])
            slack[1:][closer] = reduced[closer]
            way[1:][closer] = column
            candidates = np.where(free, slack[1:], np.inf)
            chosen = int(candidates.argmin()) + 1
            delta = candidates[chosen - 1]
            u[owner[used]] += delta
            v[used] -= delta
            slack[1:][free] -= delta
            column = chosen
            if owner[column] == 0:
                break
        while column:
            before = way[column]
            owner[column] = owner[before]
            column = before
    taken = np.empty(n, dtype=np.intp)
    for column in range(1, m + 1):
        if owner[column]:
            taken[owner[column] - 1] = column - 1
    return taken


def match_features(a: Decomposition, b: Decomposition) -> tuple[np.ndarray, np.ndarray]:
    """Pair every feature of ``a`` with a different feature of ``b`` so that the total
    cosine between decoder directions is greatest. Returns ``(partner, cosine)``: for each
    feature of ``a``, the index in ``b`` and their cosine. Needs as many features in
    each, and holds both ``n x n`` cosines in memory."""
    return _best_match(_unit_rows(a.directions())[0], _unit_rows(b.directions())[0])


def _best_match(ua: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if ua.shape != ub.shape:
        raise RequestError(
            f"cannot match {ua.shape[0]}x{ua.shape[1]} to {ub.shape[0]}x{ub.shape[1]}"
        )
    cosine = ua @ ub.T
    partner = assign(-cosine)
    return partner, cosine[np.arange(ua.shape[0]), partner]


@dataclass(frozen=True, eq=False)
class Stability:
    """How much of what one training finds another finds too.

    For every pair of dictionaries, features are matched one to one by decoder cosine
    (``match_features``). ``similarity[(i, j)]`` is the cosine of each feature of ``i`` to
    its partner in ``j``. A feature *recurs* in ``j`` if that is at least ``threshold``;
    ``recurrence[i]`` is the share of ``i``'s features that recur in every other one.
    ``chance`` is the same matching against random directions: with few channels a
    forced pairing finds high cosines by luck, and it is the number to beat.
    """

    threshold: float
    similarity: Mapping[tuple[int, int], np.ndarray]
    recurrence: tuple[float, ...]
    chance: np.ndarray
    same_training_times: bool
    provenance: Mapping[str, Any]

    @property
    def matched(self) -> np.ndarray:
        return np.concatenate(list(self.similarity.values()))

    @property
    def fraction_recurring(self) -> float:
        """Of every matched feature, in every pair, the share at or above threshold."""
        return float(np.mean(self.matched >= self.threshold - _ROUNDING))

    def to_dict(self) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "n_dictionaries": len(self.recurrence),
            "fraction_recurring": self.fraction_recurring,
            "recurrence": list(self.recurrence),
            "matched_similarity": _summary(self.matched),
            "pairs": {
                f"{i}-{j}": {
                    "fraction_recurring": float(np.mean(s >= self.threshold - _ROUNDING)),
                    "matched_similarity": _summary(s),
                }
                for (i, j), s in self.similarity.items()
            },
            "chance_similarity": _summary(self.chance),
            "same_training_times": self.same_training_times,
            "provenance": dict(self.provenance),
        }


def seed_stability(
    dictionaries: Sequence[Decomposition],
    *,
    threshold: float = 0.9,
    allow_unverified: bool = False,
) -> Stability:
    """Compare two or more bases fitted to the same layer, by matching their features.

    They must read one network at one place -- what each says it was fitted on is
    compared, and a missing identity needs ``allow_unverified=True`` -- and be the same
    width. Training times may differ; ``same_training_times`` says whether they did.
    """
    if len(dictionaries) < 2:
        raise RequestError("stability compares at least two bases")
    if not -1.0 < threshold <= 1.0:
        raise RequestError("threshold is a cosine: above -1 and at most 1")
    first = dictionaries[0]
    shapes = {(d.n_features, d.n_channels) for d in dictionaries}
    if len(shapes) > 1:
        raise RequestError(f"bases of different shapes cannot be matched: {sorted(shapes)}")
    fitted = [d.meta.get("fitted_on") or {} for d in dictionaries]
    places = {f.get("network_layer", f.get("layer")) for f in fitted}
    apart = [
        text
        for f in fitted[1:]
        for text in differing_identity(
            (fitted[0].get("provenance") or {}), (f.get("provenance") or {})
        )
    ]
    if len(places - {None}) > 1:
        apart.append(f"layers {sorted(places - {None})}")
    if apart:
        raise RequestError(f"these were not fitted to one place in one network: {'; '.join(apart)}")
    unknown = None in places or any(
        not (f.get("provenance") or {}).get(k)
        for f in fitted
        for k in ("model", "component", "checkpoint")
    )
    if unknown and not allow_unverified:
        raise RequestError(
            "which layer of which network these were fitted to is not recorded; "
            "pass allow_unverified=True to compare them anyway"
        )
    n = len(dictionaries)
    similarity = {
        (i, j): match_features(dictionaries[i], dictionaries[j])[1]
        for i in range(n)
        for j in range(n)
        if i != j
    }
    floor = threshold - _ROUNDING
    recurrence = tuple(
        float(np.mean(np.all([similarity[(i, j)] >= floor for j in range(n) if j != i], axis=0)))
        for i in range(n)
    )
    random = np.random.default_rng(0).normal(size=first.directions().shape)
    chance = _best_match(_unit_rows(first.directions())[0], _unit_rows(random)[0])[1]
    times = {tuple(map(str, f["times"])) if f.get("times") is not None else None for f in fitted}
    return Stability(
        threshold=threshold,
        similarity=similarity,
        recurrence=recurrence,
        chance=chance,
        same_training_times=len(times) == 1 and None not in times,
        provenance={
            "xaig": __version__,
            "bases": [basis_provenance(d) for d in dictionaries],
        },
    )


# -- fidelity against sparsity -------------------------------------------------------


@dataclass(frozen=True, eq=False)
class CurvePoint:
    """One basis on the fidelity-sparsity plane. ``setting`` is what made it: ``k``
    components for a PCA, the training settings (``k``, ``l1``, ``activation``, ``seed``)
    for a dictionary. ``pca_at_same_sparsity`` is a PCA's held-out explained variance at
    this point's held-out mean active features, interpolated between the PCA points, or
    None where the PCA curve does not reach."""

    method: str
    label: str
    setting: Mapping[str, Any]
    evaluation: Evaluation
    pca_at_same_sparsity: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "label": self.label,
            "setting": dict(self.setting),
            "n_features": self.evaluation.n_features,
            "mean_active_features": self.evaluation.test.mean_active_features,
            "explained_variance": self.evaluation.test.explained_variance,
            "train_mean_active_features": self.evaluation.train.mean_active_features,
            "train_explained_variance": self.evaluation.train.explained_variance,
            "dead_fraction": self.evaluation.test.dead_fraction,
            "near_duplicates": self.evaluation.redundancy.n_near_duplicates,
            "pca_at_same_sparsity": self.pca_at_same_sparsity,
            "basis": self.evaluation.provenance.get("basis"),
        }


@dataclass(frozen=True, eq=False)
class FidelityCurve:
    """Explained variance against mean active features, on held-out data, for a PCA at
    each ``k`` and for each dictionary supplied."""

    split: Split
    layer: int
    points: tuple[CurvePoint, ...]
    provenance: Mapping[str, Any]

    def method(self, name: str) -> tuple[CurvePoint, ...]:
        return tuple(p for p in self.points if p.method == name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "split": self.split.to_dict(),
            "points": [p.to_dict() for p in self.points],
            "provenance": dict(self.provenance),
        }


def _setting(basis: Decomposition) -> dict[str, Any]:
    trained = basis.meta.get("training") or {}
    keys = ("n_features", "activation", "k", "l1", "seed", "epochs")
    return {key: trained[key] for key in keys if key in trained}


def fidelity_curve(
    source: LatentSource,
    split: Split,
    *,
    layer: int,
    pca_components: Sequence[int],
    dictionaries: Sequence[Dictionary] = (),
    labels: Sequence[str] | None = None,
    test_source: LatentSource | None = None,
    active_above: float = 0.0,
    duplicate_threshold: float = 0.95,
    allow_unverified_basis: bool = False,
) -> FidelityCurve:
    """A PCA at each of ``pca_components`` against the ``dictionaries`` supplied, all
    scored on the held-out side of ``split``.

    The PCA is fitted here, once, on the training times only (``accumulate_moments``) and
    cut to ``k`` components; the dictionaries come from the caller (``xaig.nn`` trains a
    sweep of ``k`` or ``l1``) and are held to ``evaluate_basis``'s rules, the one that
    they were not fitted on a held-out time included. Each carries its own evaluation.
    A PCA keeps exactly ``k`` features active at every node, so its curve runs from one
    to the number of channels; a dictionary beyond that range has nothing to be set
    against.
    """
    if not pca_components:
        raise RequestError("pca_components is empty: there is no curve to set a dictionary against")
    if labels is not None and len(labels) != len(dictionaries):
        raise RequestError("labels, if given, are one per dictionary")
    for dictionary in dictionaries:
        if getattr(dictionary, "output_mean", None) is not None:
            raise RequestError(
                "a fidelity curve compares autoencoders; a transcoder writes another layer"
            )
    moments = accumulate_moments(source, layer=layer, times=list(split.train))
    widest = pca_from_moments(moments, max(pca_components))
    common = {
        "layer": layer,
        "test_source": test_source,
        "active_above": active_above,
        "duplicate_threshold": duplicate_threshold,
        "allow_unverified_basis": allow_unverified_basis,
    }
    points: list[CurvePoint] = []
    for k in sorted(set(pca_components)):
        basis = PCA(
            widest.mean,
            widest.components[:k],
            widest.explained_variance_ratio[:k],
            meta=widest.meta,
        )
        points.append(
            CurvePoint("pca", f"pca-{k}", {"k": k}, evaluate_basis(basis, source, split, **common))
        )
    x = np.array([p.evaluation.test.mean_active_features for p in points])
    y = np.array([p.evaluation.test.explained_variance for p in points])
    for index, dictionary in enumerate(dictionaries):
        evaluation = evaluate_basis(dictionary, source, split, **common)
        sparsity = evaluation.test.mean_active_features
        reached = x[0] <= sparsity <= x[-1]
        setting = _setting(dictionary)
        label = labels[index] if labels is not None else f"sae-{index}"
        points.append(
            CurvePoint(
                "sae", label, setting, evaluation,
                float(np.interp(sparsity, x, y)) if reached else None,
            )
        )  # fmt: skip
    return FidelityCurve(
        split=split,
        layer=layer,
        points=tuple(points),
        provenance=result_provenance(source.info()),
    )
