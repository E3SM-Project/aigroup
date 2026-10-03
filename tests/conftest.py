"""Synthetic fixtures only.

Nothing here touches a real campaign or a real filesystem outside tmp_path: the
suite has to pass on a laptop with no scratch mounted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# -- a synthetic latent archive -------------------------------------------
#
# Small enough to read at a glance, with structure planted so each analysis has
# a known right answer:
#
#   channel 4   a bump centred on BUMP, growing with depth: what "responds here"
#   channel 1   a constant offset of 50 at every node: what centring is for
#   the rest    small seeded noise

BUMP = (7.5, 45.0)
N_LAT, N_LON, N_CHANNELS, N_TIMES = 12, 24, 6, 2
LATENT_TIMES = ["0425-01-01T06:00:00", "0425-01-01T12:00:00"]


def write_latent_archive(
    path: Path, mesh: bool = False, mask=None, reference=None, experiment=None, shift=None
) -> Path:
    """Write an archive in the layout ``adapters/latent_archive.py`` reads.

    The noise is seeded, so two archives are twins node for node. ``shift`` is
    ``(channel, amount)``, added everywhere from the second time on: a perturbed
    run whose difference from its control is known exactly.
    """
    import json

    import numpy as np

    lat_1d = np.linspace(-82.5, 82.5, N_LAT)
    lon_1d = np.arange(N_LON) * 15.0
    lon, lat = np.meshgrid(lon_1d, lat_1d)
    lat, lon = lat.ravel(), lon.ravel()
    rng = np.random.default_rng(0)
    distance = np.hypot(lat - BUMP[0], lon - BUMP[1])
    path.mkdir(parents=True, exist_ok=True)
    steps = []
    for index in range(3):
        data = rng.normal(0.0, 0.05, (N_TIMES, lat.size, N_CHANNELS))
        data[:, :, 1] += 50.0
        data[:, :, 4] += (1 + index) * 3.0 * np.exp(-((distance / 20.0) ** 2))
        if shift is not None:
            data[1:, :, shift[0]] += shift[1]
        if mask is not None:
            data[:, ~mask, :] = 0.0
        np.save(path / f"step_{index:02d}.npy", data.astype(np.float16))
        steps.append(
            {"index": index, "label": f"block {index}", "file": f"step_{index:02d}.npy",
             "n_channels": N_CHANNELS}
        )  # fmt: skip
    grid = {"lat": lat, "lon": lon}
    if not mesh:
        grid["grid_shape"] = np.array([N_LAT, N_LON])
    if mask is not None and reference is None:
        grid["mask"] = mask
    np.savez(path / "grid.npz", **grid)
    manifest = {
        "model": "toy-emulator",
        "component": "atmosphere",
        "checkpoint": "toy.ckpt",
        "calendar": "noleap",
        "timestep_seconds": 21600,
        "n_nodes": int(lat.size),
        "latent_times": LATENT_TIMES,
        "steps": steps,
        "extra_steps": [
            {"index": 100, "label": "coarse", "file": "extra/step_100.npy", "n_channels": 12}
        ],
    }
    if reference is not None:
        manifest["reference_file"] = reference
    if experiment is not None:
        manifest["experiment"] = experiment
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.fixture
def latent_archive(tmp_path: Path) -> Path:
    pytest.importorskip("numpy")
    return write_latent_archive(tmp_path / "latents")


class MemorySource:
    """A ``LatentSource`` held in memory, for what an archive on disk cannot say
    (float32 beside 1e8, a NaN where a mask says there should be none).

    ``data`` is ``{layer: (n_times, n_nodes, n_channels)}``; the grid is a plain
    ``n_lat x n_lon`` one, so nodes have area weights that are far from uniform.
    """

    def __init__(self, data, *, n_lat=6, n_lon=8, mask=None):
        import numpy as np

        from xaig.latents import LatentInfo, LayerInfo
        from xaig.latents.grid import Grid

        lat, lon = np.meshgrid(
            np.linspace(-75.0, 75.0, n_lat), np.arange(n_lon) * 45.0, indexing="ij"
        )
        self._grid = Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(n_lat, n_lon), mask=mask)
        self._data = {layer: np.asarray(values) for layer, values in data.items()}
        n_times = next(iter(self._data.values())).shape[0]
        self._info = LatentInfo(
            source="memory",
            times=tuple(f"t{i}" for i in range(n_times)),
            layers=tuple(
                LayerInfo(layer, f"layer {layer}", values.shape[2])
                for layer, values in self._data.items()
            ),
            n_nodes=n_lat * n_lon,
            model="mem",
            component="mem",
            checkpoint="mem.ckpt",
        )

    def info(self):
        return self._info

    def grid(self):
        return self._grid

    def load(self, time, layer, channels=None, nodes=None):
        position = self._info.time_index(time)
        out = self._data[layer][position]
        if nodes is not None:
            out = out[list(nodes)]
        if channels is not None:
            out = out[:, list(channels)]
        return out.copy()
