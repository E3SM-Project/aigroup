"""Steering: change a feature inside a running model and see what it does.

Everything else in this package reads. An intervention is the first time an adapter
*writes*: the model is run, and at one layer and time the latents it produced are
replaced by an edited copy, after which the forward pass continues. ``docs/package/
interventions.md`` is the design note; in short, four arms are run from one initial
state with one noise seed, and compared pairwise with the first:

- ``control``         no hooks;
- ``reconstruction``  the latents are replaced by ``decode(encode(h))``: what passing
                      through the dictionary costs, with no feature touched;
- ``feature``         ``h + delta * direction`` for one feature of the basis;
- ``random``          the same ``delta`` along a random unit direction scaled to the
                      feature direction's length, drawn ``n_random`` times.

The feature's response on each physical field is then set against the random draws as an
effect size and an empirical rank. Seeds repeat the whole thing under other noise for an
uncertainty. What a system must provide is ``Intervenable``; the experiment itself knows
no framework, takes a ``Decomposition`` and returns an object. Needs the ``latents`` extra.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from xaig.core import registry
from xaig.core.errors import AdapterError, RequestError
from xaig.core.extras import missing_extra
from xaig.latents.basis import Decomposition, Dictionary, result_provenance
from xaig.latents.grid import Grid
from xaig.latents.source import LatentInfo, check_basis_fits

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

MODES = ("add", "scale", "clamp")
CONTROL, RECONSTRUCTION, FEATURE, RANDOM = "control", "reconstruction", "feature", "random"


# -- the contract -------------------------------------------------------------


@dataclass(frozen=True)
class Hook:
    """A place inside the forward pass where the latents may be replaced.

    ``edit`` receives the ``(n_nodes, n_channels)`` tensor the model produced at
    ``layer`` during forward step ``time`` (counted from 0 in a run) and returns what
    the pass continues with, of the same shape.
    """

    layer: int
    time: int
    edit: Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class Rollout:
    """What a run produced.

    ``fields[name]`` is ``(steps, n_nodes)``: what forward step ``t`` wrote, NaN where
    the field means nothing (the ``lead=1`` convention of ``ReferenceFields``).
    ``latents[(layer, time)]`` is the tensor at each ``record``ed place, taken *after*
    any hook there: what the pass actually continued with.
    """

    fields: Mapping[str, np.ndarray]
    latents: Mapping[tuple[int, int], np.ndarray] = field(default_factory=dict)


@runtime_checkable
class Intervenable(Protocol):
    """A system that can be run forward with its latents edited on the way.

    An optional capability of an adapter, asked for with ``isinstance`` and separate from
    ``LatentSource``: reading a recording is cheap and repeatable, running a model is
    neither, and most sources cannot. Nothing of the model's framework crosses it:
    hooks take and return numpy arrays, and the initial state is whatever
    ``initial_state`` handed out, opaque to xaig.

    ``run`` must be deterministic given ``noise_seed``, and the noise it draws must not
    depend on what the hooks did: that is what makes arms comparable pairwise. Without a
    seed the run has no noise. A hook at a layer or time that does not exist is a
    ``RequestError``.
    """

    def info(self) -> LatentInfo: ...

    def grid(self) -> Grid: ...

    def initial_state(self, start: str | int = 0) -> Any: ...

    def run(
        self,
        initial_state: Any,
        steps: int,
        hooks: Sequence[Hook] = (),
        *,
        noise_seed: int | None = None,
        record: Sequence[tuple[int, int]] = (),
    ) -> Rollout: ...


def open_intervenable(source: str | None, adapter: str, **options: Any) -> Intervenable:
    """Open a system that can be run, through the adapter registry. ``source`` is None
    for one that reads nothing."""
    where = None if source is None else str(source)
    built = registry.create(adapter, source=where, options=options)
    if not isinstance(built, Intervenable):
        raise AdapterError(f"adapter {adapter!r} does not implement Intervenable")
    return built


# -- what is done -------------------------------------------------------------


def feature_direction(basis: Decomposition, feature: int) -> np.ndarray:
    """The direction of ``feature`` in the units of the layer the basis reads, float64.

    A dictionary's decoder row is in standardised units, so it is multiplied by the scale
    its decode applies. A transcoder writes another layer: its directions are not
    directions of the layer it reads, and cannot steer it.
    """
    if not 0 <= feature < basis.n_features:
        raise RequestError(f"feature {feature} is not in 0..{basis.n_features - 1}")
    directions = np.asarray(basis.directions(), dtype=np.float64)
    if directions.shape[1] != basis.n_channels:
        raise RequestError(
            f"this basis reads {basis.n_channels} channel(s) and writes {directions.shape[1]}: "
            "a transcoder's directions belong to another layer, and cannot steer this one"
        )
    direction = directions[feature]
    if not np.isfinite(direction).all():
        raise RequestError(f"the direction of feature {feature} holds values that are not finite")
    if isinstance(basis, Dictionary):
        direction = direction * float(
            basis.input_scale if basis.output_scale is None else basis.output_scale
        )
    return direction


@dataclass(frozen=True)
class Intervention:
    """One feature, changed at one layer, at some forward steps.

    ``mode`` says what ``amount`` is: ``add`` it to the feature's activation, ``scale``
    the activation by it, or ``clamp`` the activation to it. Each is, node by node, a
    change ``delta`` in the activation, and the layer moves by ``delta * direction``.
    ``nodes`` restricts the edit to those nodes (None: every node).
    """

    layer: int
    feature: int
    mode: str = "add"
    amount: float = 1.0
    times: tuple[int, ...] = (0,)
    nodes: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise RequestError(f"mode must be one of {', '.join(MODES)}, not {self.mode!r}")
        if not np.isfinite(self.amount):
            raise RequestError(f"the amount must be a finite number, not {self.amount!r}")
        if not self.times:
            raise RequestError("an intervention needs at least one time to act at")
        if self.nodes is not None and not self.nodes:
            raise RequestError("an intervention restricted to no nodes changes nothing")

    def deltas(self, basis: Decomposition, latents: np.ndarray) -> np.ndarray:
        """The change in the feature's activation at each node, ``(n_nodes,)`` float64."""
        if self.mode == "add":
            delta = np.full(latents.shape[0], self.amount, dtype=np.float64)
        else:
            active = basis.transform(latents, [self.feature])[:, 0]
            delta = (self.amount - 1.0) * active if self.mode == "scale" else self.amount - active
        if self.nodes is not None:
            kept = np.zeros(delta.shape, dtype=bool)
            kept[list(self.nodes)] = True
            delta = np.where(kept, delta, 0.0)
        return delta

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer, "feature": self.feature, "mode": self.mode,
            "amount": self.amount, "times": list(self.times),
            "nodes": None if self.nodes is None else list(self.nodes),
        }  # fmt: skip


# -- the result ---------------------------------------------------------------


@dataclass(frozen=True)
class Run:
    """One arm under one noise seed. ``outcomes[name]`` is the area-weighted global mean
    of a field at each step; ``latent_rms[(layer, time)]`` the area-weighted RMS
    difference of the latents from the control's, there (zero for the control)."""

    arm: str
    seed: int
    draw: int | None
    outcomes: Mapping[str, np.ndarray]
    latent_rms: Mapping[tuple[int, int], float]


@dataclass(frozen=True)
class Pairing:
    """One run against the control of the same seed: ``differences[name]`` is the series
    of the outcome minus the control's, and ``response[name]`` its mean over the steps
    from the first edit on."""

    arm: str
    seed: int
    draw: int | None
    differences: Mapping[str, np.ndarray]
    response: Mapping[str, float]


@dataclass(frozen=True)
class Effect:
    """The feature's response on one field against what random directions do.

    ``feature_response`` and ``reconstruction_response`` are signed means over seeds of
    the paired response (``feature_se``: the standard error across seeds, None for one).
    ``random_responses`` holds one seed-mean per draw, and everything compared uses
    magnitudes. ``rank`` is the feature's place among itself and the draws (1 is a larger
    response than every draw), ``percentile`` the share of draws it exceeds (ties count
    half), ``p_value`` the share of the ``n + 1`` that match or beat it, and
    ``effect_size`` how many standard deviations of the draws' magnitudes it sits above
    their mean (None for fewer than two draws, or draws that do not vary).
    """

    field: str
    feature_response: float
    feature_se: float | None
    reconstruction_response: float
    random_responses: np.ndarray
    effect_size: float | None
    rank: int
    percentile: float
    p_value: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field, "feature_response": self.feature_response,
            "feature_se": self.feature_se,
            "reconstruction_response": self.reconstruction_response,
            "random_responses": self.random_responses, "effect_size": self.effect_size,
            "rank": self.rank, "n_draws": int(self.random_responses.size),
            "percentile": self.percentile, "p_value": self.p_value,
        }  # fmt: skip


@dataclass(frozen=True)
class SteeringResult:
    """Every run, every pairing against its control, and the summary, with the settings
    and the provenance of the system and the basis that produced them."""

    spec: Mapping[str, Any]
    provenance: Mapping[str, Any]
    runs: tuple[Run, ...]
    pairings: tuple[Pairing, ...]
    effects: Mapping[str, Effect]

    def arm(self, arm: str, seed: int, draw: int | None = None) -> Run:
        for run in self.runs:
            if (run.arm, run.seed, run.draw) == (arm, seed, draw):
                return run
        raise RequestError(f"no run of arm {arm!r} with seed {seed} and draw {draw}")

    def pairing(self, arm: str, seed: int, draw: int | None = None) -> Pairing:
        for pairing in self.pairings:
            if (pairing.arm, pairing.seed, pairing.draw) == (arm, seed, draw):
                return pairing
        raise RequestError(f"no pairing of arm {arm!r} with seed {seed} and draw {draw}")

    def to_dict(self) -> dict[str, Any]:
        def place(key: tuple[int, int]) -> str:
            return f"{key[0]}:{key[1]}"

        return {
            "spec": dict(self.spec),
            "provenance": dict(self.provenance),
            "effects": {name: e.to_dict() for name, e in self.effects.items()},
            "runs": [
                {
                    "arm": r.arm, "seed": r.seed, "draw": r.draw, "outcomes": dict(r.outcomes),
                    "latent_rms": {place(k): v for k, v in r.latent_rms.items()},
                }
                for r in self.runs
            ],
            "pairings": [
                {
                    "arm": p.arm, "seed": p.seed, "draw": p.draw, "response": dict(p.response),
                    "differences": dict(p.differences),
                }
                for p in self.pairings
            ],
        }  # fmt: skip


# -- running ------------------------------------------------------------------


def _guarded(edit: Callable[[np.ndarray], np.ndarray], hook: Hook, valid: np.ndarray) -> Hook:
    """``hook`` with its edit held to the contract: the shape it was given, and numbers
    wherever a node is valid. A NaN let through here would surface, many steps later and
    far from its cause, as a NaN in a field."""

    def checked(latents: np.ndarray) -> np.ndarray:
        out = np.asarray(edit(latents))
        if out.shape != latents.shape:
            raise RequestError(
                f"the edit at layer {hook.layer}, time {hook.time} returned {out.shape} "
                f"for {latents.shape}"
            )
        if not np.isfinite(out[valid]).all():
            raise RequestError(
                f"the edit at layer {hook.layer}, time {hook.time} produced values that are "
                "not finite on valid nodes"
            )
        return out

    return Hook(hook.layer, hook.time, checked)


def _global_means(
    fields: Mapping[str, np.ndarray], names: Sequence[str], grid: Grid, steps: int
) -> dict[str, np.ndarray]:
    """Area-weighted mean of each field over the valid nodes, per step. A valid node that
    holds NaN or infinity is a ``RequestError``, as in ``read_latents``."""
    valid = grid.valid
    weights = grid.weights()[valid]  # raises where the mask leaves no node
    out = {}
    for name in names:
        if name not in fields:
            raise RequestError(
                f"the system wrote no field {name!r}; it wrote {', '.join(sorted(fields))}"
            )
        values = np.asarray(fields[name], dtype=np.float64)
        if values.shape != (steps, grid.n_nodes):
            raise RequestError(
                f"field {name!r} has shape {values.shape}, expected {(steps, grid.n_nodes)}"
            )
        values = values[:, valid]
        bad = np.flatnonzero(~np.isfinite(values).all(axis=1))
        if bad.size:
            raise RequestError(
                f"field {name!r} is not finite on valid nodes at step {int(bad[0])}; a node the "
                "system cannot supply belongs in its mask, not left NaN"
            )
        out[name] = values @ weights
    return out


def _rms(difference: np.ndarray, grid: Grid) -> float:
    valid = grid.valid
    squares = (difference[valid].astype(np.float64) ** 2).mean(axis=1)
    return float(np.sqrt(squares @ grid.weights()[valid]))


def _random_unit(n_channels: int, seed: int, draw: int) -> np.ndarray:
    vector = np.random.default_rng([seed, draw]).normal(size=n_channels)
    return vector / np.linalg.norm(vector)


def _summarise(
    names: Sequence[str], pairings: Sequence[Pairing], seeds: Sequence[int], n_random: int
) -> dict[str, Effect]:
    effects = {}
    for name in names:
        by_seed = {p.seed: p.response[name] for p in pairings if p.arm == FEATURE}
        feature = np.array([by_seed[s] for s in seeds])
        reconstruction = np.mean([p.response[name] for p in pairings if p.arm == RECONSTRUCTION])
        draws = np.array(
            [
                np.mean([p.response[name] for p in pairings if p.arm == RANDOM and p.draw == j])
                for j in range(n_random)
            ]
        )
        size, magnitudes = abs(float(feature.mean())), np.abs(draws)
        spread = float(magnitudes.std(ddof=1)) if n_random > 1 else 0.0
        effects[name] = Effect(
            field=name,
            feature_response=float(feature.mean()),
            feature_se=(
                float(feature.std(ddof=1) / np.sqrt(feature.size)) if feature.size > 1 else None
            ),
            reconstruction_response=float(reconstruction),
            random_responses=draws,
            effect_size=(size - float(magnitudes.mean())) / spread if spread > 0.0 else None,
            rank=1 + int(np.count_nonzero(magnitudes > size)),
            percentile=100.0 * float(
                (np.count_nonzero(magnitudes < size) + 0.5 * np.count_nonzero(magnitudes == size))
                / n_random
            ),
            p_value=(1 + int(np.count_nonzero(magnitudes >= size))) / (n_random + 1),
        )  # fmt: skip
    return effects


def run_steering(
    system: Intervenable,
    basis: Decomposition,
    intervention: Intervention,
    *,
    steps: int,
    seeds: Sequence[int] = (0,),
    n_random: int = 20,
    random_seed: int = 0,
    fields: Sequence[str] | None = None,
    record_layers: Sequence[int] | None = None,
    initial_state: Any = None,
    allow_unverified_basis: bool = False,
) -> SteeringResult:
    """Run the four arms for every seed, and compare them with the control.

    The control runs first and records the latents at each place the intervention
    acts; ``delta`` is worked out from those, so every arm is edited by the same amount
    at the same nodes however the runs drift apart afterwards (it is worked out again for
    each seed, whose control differs). Within a seed all arms share the noise.
    ``record_layers`` are the layers whose latents are compared (default: the
    intervention's own), at every step. ``fields`` default to every field the system
    writes. Nothing is read or run before the request has been checked.
    """
    if not isinstance(system, Intervenable):
        raise RequestError("this system cannot be intervened on: it is not Intervenable")
    info, grid = system.info(), system.grid()
    layer = intervention.layer
    steps, seeds = int(steps), [int(s) for s in seeds]
    if steps < 1:
        raise RequestError("steps must be at least 1")
    if not seeds or len(set(seeds)) != len(seeds):
        raise RequestError("seeds must be a non-empty list of distinct integers")
    if n_random < 1:
        raise RequestError("at least one random-direction draw is needed to say what is unusual")
    if fields is not None and not fields:
        raise RequestError("no fields were asked for")
    if layer not in {entry.index for entry in info.layers}:
        raise RequestError(f"no layer {layer}; the system has {[e.index for e in info.layers]}")
    late = [t for t in intervention.times if not 0 <= t < steps]
    if late:
        raise RequestError(f"cannot act at time(s) {late} in a run of {steps} step(s)")
    if intervention.nodes is not None:
        if any(not 0 <= n < grid.n_nodes for n in intervention.nodes):
            raise RequestError(f"nodes must lie in 0..{grid.n_nodes - 1}")
        if not grid.valid[list(intervention.nodes)].any():
            raise RequestError("every node asked for is masked out")
    check_basis_fits(basis, info, layer, allow_unverified=allow_unverified_basis)
    direction = feature_direction(basis, intervention.feature)
    length = float(np.linalg.norm(direction))
    recorded = sorted({layer, *(record_layers or ())})
    if unknown := [r for r in recorded if r not in {e.index for e in info.layers}]:
        raise RequestError(f"cannot record layer(s) {unknown}: not in the system")
    places = [(r, t) for r in recorded for t in range(steps)]
    times = sorted(set(intervention.times))
    state = system.initial_state() if initial_state is None else initial_state

    def run(hooks: Sequence[Hook], seed: int) -> Rollout:
        return system.run(state, steps, hooks, noise_seed=seed, record=places)

    valid = grid.valid
    runs: list[Run] = []
    pairings: list[Pairing] = []
    controls: dict[int, tuple[Rollout, dict[str, np.ndarray]]] = {}
    names: list[str] = []

    for seed in seeds:
        control = run((), seed)
        if not names:
            names = list(fields) if fields is not None else sorted(control.fields)
            if not names:
                raise RequestError("the system wrote no fields to measure")
        series = _global_means(control.fields, names, grid, steps)
        controls[seed] = (control, series)
        runs.append(Run(CONTROL, seed, None, series, {place: 0.0 for place in places}))
        deltas = {}  # this seed's control: a noisy run has its own activations
        for t in times:
            if (layer, t) not in control.latents:
                raise RequestError(f"the system did not record layer {layer} at time {t}")
            deltas[t] = intervention.deltas(basis, control.latents[(layer, t)])
        arms: list[tuple[str, int | None, np.ndarray | None]] = [
            (RECONSTRUCTION, None, None),
            (FEATURE, None, direction),
        ]
        for draw in range(n_random):
            arms.append((RANDOM, draw, _random_unit(basis.n_channels, random_seed, draw) * length))
        for arm, draw, vector in arms:
            hooks = [Hook(layer, t, _edit(basis, deltas[t], vector)) for t in times]
            rollout = run([_guarded(h.edit, h, valid) for h in hooks], seed)
            outcomes = _global_means(rollout.fields, names, grid, steps)
            first = times[0]
            differences = {n: outcomes[n] - series[n] for n in names}
            runs.append(
                Run(
                    arm,
                    seed,
                    draw,
                    outcomes,
                    {p: _rms(rollout.latents[p] - control.latents[p], grid) for p in places},
                )  # fmt: skip
            )
            pairings.append(
                Pairing(
                    arm, seed, draw, differences,
                    {n: float(differences[n][first:].mean()) for n in names},
                )
            )  # fmt: skip

    spec = {
        "intervention": intervention.to_dict(), "steps": steps, "seeds": seeds,
        "n_random": n_random, "random_seed": random_seed, "fields": names,
        "record_layers": recorded, "feature_length": length,
    }  # fmt: skip
    return SteeringResult(
        spec=spec,
        provenance=result_provenance(info, basis=basis),
        runs=tuple(runs),
        pairings=tuple(pairings),
        effects=_summarise(names, pairings, seeds, n_random),
    )


def _edit(basis: Decomposition, deltas: np.ndarray, vector: np.ndarray | None):
    """The edit of one arm at one time: the basis's reconstruction (``vector`` None), or
    the layer moved by ``deltas`` along ``vector``."""
    if vector is None:
        return lambda latents: basis.reconstruct(latents).astype(latents.dtype)
    return lambda latents: _moved(latents, deltas, vector)


def _moved(latents: np.ndarray, deltas: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """``latents`` moved by ``deltas[node] * direction``, in their own dtype."""
    return latents + (deltas[:, None] * direction[None, :]).astype(latents.dtype)
