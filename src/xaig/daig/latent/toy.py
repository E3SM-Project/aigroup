"""A toy emulator, so that a latent archive can be made with nothing but numpy.

It is small enough to read in a minute and shaped like the real thing where that
matters to whatever reads its latents:

- a Gaussian lat-lon grid with longitudes in -180..180, and a continent where the
  sea-surface temperature is missing;
- a per-node MLP with a residual stream: an encoder, then blocks that each add to
  what they were given, so channel ``c`` means the same thing at every layer;
- a rollout: temperature is carried from step to step, under a sun that goes round
  and a storm that drifts east, so a run that is pushed diverges from its twin;
- a no-leap calendar, steps that are *kept* or not, and physical fields recorded at
  the kept times and at the time each of those steps started from -- so the fields
  hold more times than the latents, and begin one step earlier.

One thing is planted, so that an analysis has a known right answer: channel
``STORM_CHANNEL`` carries the storm, from the encoder to the last block, and
precipitation is read off it. ``OFFSET_CHANNEL`` sits on a constant, which is
what centring is for. The rest is seeded noise.

    from xaig.daig.latent.toy import write_toy

    write_toy("scratch/toy/control")
    write_toy("scratch/toy/steered", steer=(2, 7, 3.0))   # +3 on channel 7 of layer 2, every step

Nothing here is a model of the atmosphere. It is a model of an *archive*.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from xaig import __version__
from xaig.core import registry
from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra
from xaig.daig.grid import Grid, great_circle_km

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "daig") from exc

MODEL = "xaig-toy"
WIDTH = 16
N_BLOCKS = 3
STORM_CHANNEL = 5
OFFSET_CHANNEL = 2
INPUTS = ("temperature", "insolation", "sst", "storm", "sin_lat", "cos_lat")
TIMESTEP_SECONDS = 21600
# A leap year by the usual rule, so that a calendar without leap days shows: the
# default run crosses from 28 February straight into 1 March.
START = (424, 2, 27, 0)
_MONTH_DAYS = (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)


@dataclass(frozen=True)
class ToyRun:
    """What one rollout kept: everything an archive is made of."""

    grid: Grid
    times: tuple[str, ...]
    layers: tuple[tuple[str, np.ndarray], ...]
    fields: dict[str, np.ndarray]
    field_times: tuple[str, ...]
    kept_steps: tuple[int, ...]
    experiment: dict[str, Any] = field(default_factory=dict)


def noleap_label(hours: int, start: tuple[int, int, int, int] = START) -> str:
    """The label ``hours`` after ``start`` on a calendar with no leap days."""
    year, month, day, hour = start
    total = hour + hours
    day_of_year = sum(_MONTH_DAYS[: month - 1]) + (day - 1) + total // 24
    year, day_of_year = year + day_of_year // 365, day_of_year % 365
    month = 0
    while day_of_year >= _MONTH_DAYS[month]:
        day_of_year -= _MONTH_DAYS[month]
        month += 1
    return f"{year:04d}-{month + 1:02d}-{day_of_year + 1:02d}T{total % 24:02d}:00:00"


def parse_steps(text: str) -> tuple[int, ...]:
    """``"1-4,9-12"`` as steps 1, 2, 3, 4, 9, 10, 11, 12."""
    steps: set[int] = set()
    try:
        for part in text.split(","):
            first, _, last = part.strip().partition("-")
            steps.update(range(int(first), int(last or first) + 1))
    except ValueError as exc:
        raise RequestError(f"steps look like '1-4,9-12', not {text!r}") from exc
    return tuple(sorted(steps))


def toy_grid(n_lat: int = 24, n_lon: int = 48) -> Grid:
    """Gaussian latitudes, south to north; cell-centred longitudes in -180..180."""
    nodes, _ = np.polynomial.legendre.leggauss(n_lat)
    lat_1d = np.degrees(np.arcsin(nodes))
    lon_1d = -180.0 + (np.arange(n_lon) + 0.5) * (360.0 / n_lon)
    lon, lat = np.meshgrid(lon_1d, lat_1d)
    return Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(n_lat, n_lon))


def land(grid: Grid) -> np.ndarray:
    """One continent. The model runs over it all the same, as an ocean model does:
    what it holds there means nothing, and is not zero."""
    return (np.abs(grid.lat - 5.0) < 35.0) & (np.abs(grid.lon - 15.0) < 35.0)


@dataclass(frozen=True)
class ToyModel:
    """The weights. Seeded noise, with the two planted channels written over it."""

    w_in: np.ndarray  # (inputs, WIDTH)
    bias: np.ndarray  # (WIDTH,)
    blocks: tuple[tuple[np.ndarray, np.ndarray, np.ndarray], ...]  # (a, c, b) per block
    w_out: np.ndarray  # (WIDTH, 2): temperature tendency, precipitation

    @classmethod
    def seeded(cls, seed: int = 0) -> ToyModel:
        rng = np.random.default_rng(seed)
        scale = 0.5 / np.sqrt(WIDTH)
        w_in = rng.normal(0.0, scale, (len(INPUTS), WIDTH))
        w_in[:, STORM_CHANNEL] = 0.0
        w_in[INPUTS.index("storm"), STORM_CHANNEL] = 4.0
        w_in[:, OFFSET_CHANNEL] = 0.0
        bias = np.zeros(WIDTH)
        bias[OFFSET_CHANNEL] = 5.0
        blocks = []
        for _ in range(N_BLOCKS):
            a, b = rng.normal(0.0, scale, (WIDTH, WIDTH)), rng.normal(0.0, scale, (WIDTH, WIDTH))
            b[:, [STORM_CHANNEL, OFFSET_CHANNEL]] = 0.0  # nothing else writes to the planted two
            blocks.append((a, rng.normal(0.0, 0.1, WIDTH), b))
        w_out = rng.normal(0.0, scale, (WIDTH, 2))
        w_out[:, 1] = 0.0
        w_out[STORM_CHANNEL, 1] = 1.0
        return cls(w_in=w_in, bias=bias, blocks=tuple(blocks), w_out=w_out)

    def forward(
        self, inputs: np.ndarray, steer: tuple[int, int, float] | None = None
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Every layer's activations ``(n_nodes, WIDTH)``, and the two outputs."""
        hidden = inputs @ self.w_in + self.bias
        layers = []
        for index in range(N_BLOCKS + 1):
            if index:
                a, c, b = self.blocks[index - 1]
                hidden = hidden + np.maximum(hidden @ a + c, 0.0) @ b
            if steer is not None and steer[0] == index:
                hidden = hidden.copy()
                hidden[:, steer[1]] += steer[2]
            layers.append(hidden)
        return layers, hidden @ self.w_out


def layer_labels() -> tuple[str, ...]:
    return ("encoder output", *(f"block {i} output" for i in range(1, N_BLOCKS + 1)))


def toy_run(
    n_steps: int = 12,
    keep: Sequence[int] = (1, 2, 3, 4, 9, 10, 11, 12),
    seed: int = 0,
    steer: tuple[int, int, float] | None = None,
    grid: Grid | None = None,
) -> ToyRun:
    """Roll the toy forward ``n_steps`` and keep the steps asked for.

    Step ``k`` (from 1) takes the state at ``t_(k-1)`` to ``t_k``; what it held is
    labelled ``t_k``. ``steer`` is ``(layer, channel, amount)``, added at every node
    and every step: a twin of the same seed without it is its control.
    """
    kept = tuple(sorted(set(int(k) for k in keep)))
    if not kept or kept[0] < 1 or kept[-1] > n_steps:
        raise RequestError(f"steps to keep must lie in 1..{n_steps}, got {list(keep)}")
    if steer is not None and not (0 <= steer[0] <= N_BLOCKS and 0 <= steer[1] < WIDTH):
        raise RequestError(
            f"steer is (layer 0..{N_BLOCKS}, channel 0..{WIDTH - 1}, amount), got {steer}"
        )
    grid = grid or toy_grid()
    model = ToyModel.seeded(seed)
    rng = np.random.default_rng(seed + 1)
    dry = land(grid)
    sin_lat, cos_lat = np.sin(np.radians(grid.lat)), np.cos(np.radians(grid.lat))
    sst = np.where(dry, np.nan, 28.0 * cos_lat**2)
    temperature = 300.0 - 40.0 * sin_lat**2 + rng.normal(0.0, 0.5, grid.n_nodes)

    wanted = sorted(set(kept) | {k - 1 for k in kept})  # and the state each began from
    fields: dict[str, list[np.ndarray]] = {"temperature": [], "precipitation": [], "sst": []}
    recorded: list[list[np.ndarray]] = [[] for _ in range(N_BLOCKS + 1)]
    precipitation = np.full(grid.n_nodes, np.nan)  # no step has produced any at t_0
    for step in range(n_steps + 1):
        if step:
            sun = -90.0 * step  # a quarter turn westward every six hours
            insolation = np.maximum(cos_lat * np.cos(np.radians(grid.lon - sun)), 0.0)
            away = great_circle_km(grid.lat, grid.lon, 10.0, -150.0 + 12.0 * step)
            storm = np.exp(-((away / 1500.0) ** 2))
            inputs = np.stack(
                [
                    (temperature - 280.0) / 20.0,
                    insolation,
                    np.nan_to_num(sst / 28.0),
                    storm,
                    sin_lat,
                    cos_lat,
                ],
                axis=1,
            )
            layers, outputs = model.forward(inputs, steer)
            temperature = temperature + 2.0 * np.tanh(outputs[:, 0])
            precipitation = np.maximum(outputs[:, 1], 0.0)
            if step in kept:
                for index, activation in enumerate(layers):
                    recorded[index].append(activation)
        if step in wanted:
            fields["temperature"].append(temperature.copy())
            fields["precipitation"].append(precipitation.copy())
            fields["sst"].append(sst)

    experiment: dict[str, Any] = {"seed": seed, "kept_steps": list(kept), "xaig": __version__}
    if steer is not None:
        experiment["steer"] = {"layer": steer[0], "channel": steer[1], "by": steer[2]}
    hours = TIMESTEP_SECONDS // 3600
    return ToyRun(
        grid=grid,
        times=tuple(noleap_label(hours * k) for k in kept),
        layers=tuple(
            (label, np.stack(rows)) for label, rows in zip(layer_labels(), recorded, strict=True)
        ),
        fields={name: np.stack(rows) for name, rows in fields.items()},
        field_times=tuple(noleap_label(hours * k) for k in wanted),
        kept_steps=kept,
        experiment=experiment,
    )


def write_toy(
    path: str | Path, adapter: str = "latent-archive", overwrite: bool = False, **run: Any
) -> Path:
    """Run the toy and write what it kept, through whatever ``adapter`` writes."""
    write = getattr(registry.get(adapter), "write", None)
    if write is None:
        raise RequestError(f"adapter {adapter!r} reads latents but cannot write them")
    result = toy_run(**run)
    return write(
        path,
        grid=result.grid,
        times=result.times,
        layers=result.layers,
        fields=result.fields,
        field_times=result.field_times,
        model=MODEL,
        component="atmosphere",
        checkpoint=f"seed-{result.experiment['seed']}",
        calendar="noleap",
        timestep_seconds=TIMESTEP_SECONDS,
        experiment=result.experiment,
        overwrite=overwrite,
    )


__all__ = [
    "MODEL",
    "OFFSET_CHANNEL",
    "STORM_CHANNEL",
    "ToyModel",
    "ToyRun",
    "noleap_label",
    "parse_steps",
    "toy_grid",
    "toy_run",
    "write_toy",
]
