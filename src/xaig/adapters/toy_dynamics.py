"""A toy system that can be intervened on, with a planted answer. For tests and docs.

Not a model of anything: a small linear dynamical system over a lat-lon grid, written so
that what an edit at a latent layer does to the physical fields can be worked out by hand.

    state x (n_nodes, 3): temperature, moisture, pressure
    forcing u (n_nodes, 3): drifts on its own, with noise: u' = 0.9 u + noise * eps_u
    layer 0   h0 = u @ E          (4 channels)
    layer 1   h1 = h0 @ M         (M swaps channels 0 and 1)
    next      x' = decay * x + h1 @ D + noise * eps_x

``D`` sends channel 0 to temperature with weight ``gain``, channel 1 to moisture, channel
2 to pressure, channel 3 to pressure at half weight. So adding ``a`` to channel 0 of layer 1
(the planted feature) moves temperature by ``gain * a`` on that step and by ``decay`` times
as much on each step after, and nothing else; the same edit at layer 0 reaches moisture
instead, by way of ``M``; and a random direction mostly does neither. The latents do not
read ``x`` back, so no edit feeds into the next one's input: responses are exactly those.

Noise is a function of the seed alone (drawn up front, not from the state), so two runs
with one seed differ only by what the hooks did. The initial state is ``(n_nodes, 6)``:
``x`` then ``u``.

``planted_dictionary`` is a dictionary that reads three of the four channels, so it has a
reconstruction error with a known cause: the fourth channel, and what its ReLU drops.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra
from xaig.latents.basis import Dictionary
from xaig.latents.grid import Grid
from xaig.latents.intervene import Hook, Rollout
from xaig.latents.source import LatentInfo, LayerInfo

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

FIELDS = ("temperature", "moisture", "pressure")
WIDTH = 4


class ToyDynamics:
    """``Intervenable`` (and nothing else): it reads no source, so its options are
    keyword-only. ``masked`` nodes, the first ones, have fields that are NaN, like land."""

    def __init__(
        self,
        *,
        n_lat: int = 4,
        n_lon: int = 6,
        gain: float = 2.0,
        decay: float = 0.5,
        noise: float = 0.05,
        masked: int = 0,
        seed: int = 0,
    ) -> None:
        if not 0 <= masked < n_lat * n_lon:
            raise RequestError(f"masked must leave a node valid: 0 <= masked < {n_lat * n_lon}")
        self._options = {
            "n_lat": n_lat, "n_lon": n_lon, "gain": gain, "decay": decay, "noise": noise,
            "masked": masked, "seed": seed,
        }  # fmt: skip
        self.decay, self.noise, self.seed = decay, noise, seed
        lat, lon = np.meshgrid(
            np.linspace(-75.0, 75.0, n_lat), np.arange(n_lon) * (360.0 / n_lon), indexing="ij"
        )
        mask = np.arange(n_lat * n_lon) >= masked
        self._grid = Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(n_lat, n_lon), mask=mask)
        self.encoder = np.array([[1, 0, 0, 0.5], [0, 1, 0, 0.5], [0, 0, 1, 0]], dtype=np.float64)
        self.mixer = np.eye(WIDTH)[[1, 0, 2, 3]]
        self.decoder = np.array([[gain, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0.5]], dtype=np.float64)
        self._info = LatentInfo(
            source="toy-dynamics",
            times=("start",),
            layers=(LayerInfo(0, "encoder", WIDTH), LayerInfo(1, "mixed", WIDTH)),
            n_nodes=self._grid.n_nodes,
            model="toy-dynamics",
            component="linear",
            checkpoint=f"planted-{seed}",
            options=self._options,
        )

    def info(self) -> LatentInfo:
        return self._info

    def grid(self) -> Grid:
        return self._grid

    def initial_state(self, start: str | int = 0) -> np.ndarray:
        if isinstance(start, str):
            raise RequestError("this system starts from a number, the seed offset of its state")
        rng = np.random.default_rng([self.seed, int(start)])
        return rng.normal(size=(self._grid.n_nodes, 2 * len(FIELDS)))

    def run(
        self,
        initial_state: Any,
        steps: int,
        hooks: Sequence[Hook] = (),
        *,
        noise_seed: int | None = None,
        record: Sequence[tuple[int, int]] = (),
    ) -> Rollout:
        n = self._grid.n_nodes
        state = np.asarray(initial_state, dtype=np.float64)
        if state.shape != (n, 2 * len(FIELDS)):
            raise RequestError(f"the state is {state.shape}, expected {(n, 2 * len(FIELDS))}")
        x, u = state[:, : len(FIELDS)], state[:, len(FIELDS) :]
        if steps < 1:
            raise RequestError("steps must be at least 1")
        edits = {}
        for hook in hooks:
            if hook.layer not in (0, 1) or not 0 <= hook.time < steps:
                raise RequestError(f"no place at layer {hook.layer}, time {hook.time}")
            edits.setdefault((hook.layer, hook.time), []).append(hook.edit)
        for place in record:
            if place[0] not in (0, 1) or not 0 <= place[1] < steps:
                raise RequestError(f"cannot record layer {place[0]} at time {place[1]}")
        eps = (
            None
            if noise_seed is None
            else np.random.default_rng(noise_seed).normal(size=(steps, n, 2 * len(FIELDS)))
        )
        written = np.empty((steps, n, len(FIELDS)))
        kept: dict[tuple[int, int], np.ndarray] = {}

        def hooked(h: np.ndarray, layer: int, t: int) -> np.ndarray:
            h = h.astype(np.float32)  # latents are float32, as a recorded model's are
            for edit in edits.get((layer, t), ()):
                h = np.asarray(edit(h.copy()), dtype=np.float32)
            if (layer, t) in record:
                kept[(layer, t)] = h.copy()
            return h

        for t in range(steps):
            h0 = hooked(u @ self.encoder, 0, t)
            h1 = hooked(h0.astype(np.float64) @ self.mixer, 1, t)
            x = self.decay * x + h1.astype(np.float64) @ self.decoder
            u = 0.9 * u
            if eps is not None:
                x = x + self.noise * eps[t, :, : len(FIELDS)]
                u = u + self.noise * eps[t, :, len(FIELDS) :]
            written[t] = x
        fields = {name: written[:, :, i].copy() for i, name in enumerate(FIELDS)}
        for values in fields.values():
            values[:, ~self._grid.valid] = np.nan
        return Rollout(fields=fields, latents=kept)

    def planted_dictionary(self, layer: int = 1) -> Dictionary:
        """A dictionary for ``layer`` with three features: channels 0, 1 and 2. Feature 0
        of layer 1 is the planted one. It carries the identity a basis fitted here would."""
        if layer not in (0, 1):
            raise RequestError(f"no layer {layer}")
        frame = np.eye(WIDTH, dtype=np.float32)[:3]
        return Dictionary(
            encoder=frame,
            encoder_bias=np.zeros(3, dtype=np.float32),
            decoder=frame.copy(),
            decoder_bias=np.zeros(WIDTH, dtype=np.float32),
            input_mean=np.zeros(WIDTH, dtype=np.float32),
            meta={"fitted_on": {"provenance": self._info.identity(), "layer": layer}},
        )
