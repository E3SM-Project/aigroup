"""Synthetic fixtures only.

Nothing here touches a real campaign or a real filesystem outside tmp_path: the
suite has to pass on a laptop with no scratch mounted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Deliberately unlike aug26 -- different separator, different field order, a
# two-letter factor key, no realm. If core is quietly shaped by one campaign,
# this fixture is what fails.
TOY_SPEC = r"""
name: toy
description: A tiny campaign that looks nothing like aug26.
id_pattern: '^run(?P<num>\d+)-(?P<mode>fast|slow)-(?P<knobs>[a-z0-9-]+)-s(?P<seed>\d+)$'
id_template: 'run{num:03d}-{mode}-{knobs}-s{seed:d}'
factor_field: knobs
factor_separator: "-"
factors:
  - { key: d, name: depth, width: 1 }
  - { key: w, name: width, width: 2 }
parent_key: num
parents:
  2: 1
metric: loss
noise_floor: 0.01
discovery:
  adapter: table
  id_column: name
"""

TOY_TABLE = """name,mode,note
run001-fast-d1-w08-s1,fast,baseline
run002-fast-d2-w08-s1,fast,deeper
run003-slow-d1-w16-s2,slow,wider and slower
"""


@pytest.fixture
def toy_spec_path(tmp_path: Path) -> Path:
    path = tmp_path / "toy.yaml"
    path.write_text(TOY_SPEC)
    return path


@pytest.fixture
def toy_table_path(tmp_path: Path) -> Path:
    path = tmp_path / "runs.csv"
    path.write_text(TOY_TABLE)
    return path


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


def write_latent_archive(path: Path, mesh: bool = False, mask=None, reference=None) -> Path:
    """Write an archive in the layout ``adapters/latent_archive.py`` reads."""
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
    (path / "manifest.json").write_text(json.dumps(manifest))
    return path


@pytest.fixture
def latent_archive(tmp_path: Path) -> Path:
    pytest.importorskip("numpy")
    return write_latent_archive(tmp_path / "latents")
