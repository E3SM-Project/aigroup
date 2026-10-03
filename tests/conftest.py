"""Synthetic fixtures only.

Nothing here touches a real campaign or a real filesystem outside tmp_path: the
suite has to pass on a laptop with no scratch mounted.
"""

from __future__ import annotations

import shutil
import sys
import types
from pathlib import Path

import pytest

# -- a synthetic latent archive -------------------------------------------
#
# Small enough to read at a glance, with structure planted so each analysis has
# a known right answer:
#
#   channel 4   a bump centered on BUMP, growing with depth: what "responds here"
#   channel 1   a constant offset of 50 at every node: what centering is for
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


# -- a fake Hugging Face hub ------------------------------------------------

URL = "hf://datasets/owner/latents/control"


class _EntryNotFound(Exception):
    pass


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake ``huggingface_hub`` serving ``tmp_path/remote`` as the dataset
    ``owner/latents``, into a cache under ``tmp_path/cache``. ``heads`` maps the
    branch and tag names the hub knows to commits: ``main`` and the default are
    ``abc``, the tag ``v1`` is ``def``; move one, and what a name means moves with it.
    A commit names itself. Anything else is a 404."""
    import numpy as np

    remote = tmp_path / "remote"
    write_latent_archive(remote / "control")
    (remote / "control" / "bases" / "old").mkdir(parents=True)
    np.savez(remote / "control" / "bases" / "pca_L02.npz", placeholder=np.zeros(1))
    cache = tmp_path / "cache"
    fetched: list[str] = []
    asked: dict = {"lookups": []}
    heads = {"main": "abc", "v1": "def"}
    commits = {"abc", "def", "ghi"}

    class HfApi:
        def repo_info(self, repo_id, repo_type, revision=None):
            asked["repo"] = (repo_id, repo_type)
            asked["lookups"].append(revision)
            if repo_id != "owner/latents":
                raise RuntimeError("404 Client Error")
            name = "main" if revision is None else revision
            if name in heads:
                return types.SimpleNamespace(sha=heads[name])
            if name in commits:
                return types.SimpleNamespace(sha=name)
            raise RuntimeError(f"404 Client Error: revision {revision!r} not found")

        def list_repo_tree(self, repo_id, path_in_repo, repo_type, revision):
            folder = remote / path_in_repo
            if not folder.is_dir():
                raise _EntryNotFound(path_in_repo)
            for entry in sorted(folder.iterdir()):
                name = f"{path_in_repo}/{entry.name}"
                if entry.is_file():
                    yield types.SimpleNamespace(path=name, size=entry.stat().st_size)
                else:
                    yield types.SimpleNamespace(path=name, tree_id="t")

    def hf_hub_download(repo_id, filename, repo_type, revision):
        assert (repo_id, repo_type) == ("owner/latents", "dataset")
        asked.setdefault("revisions", set()).add(revision)
        source = remote / filename
        if not source.is_file():
            raise _EntryNotFound(filename)
        target = cache / "snapshots" / revision / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy(source, target)
            fetched.append(filename)
        return str(target)

    module = types.ModuleType("huggingface_hub")
    module.HfApi, module.hf_hub_download = HfApi, hf_hub_download
    utils = types.ModuleType("huggingface_hub.utils")
    utils.EntryNotFoundError = _EntryNotFound
    module.utils = utils
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return types.SimpleNamespace(fetched=fetched, asked=asked, heads=heads, remote=remote)


# -- experiment records, written the way a person would: by the commands ------------------


def write_steering_record(tmp_path: Path):
    """Run ``latents steer`` on the planted system with ``--record``: ``(full result, path)``."""
    import json

    from click.testing import CliRunner

    from xaig._cli import cli
    from xaig.adapters.toy_dynamics import ToyDynamics
    from xaig.latents import save_basis

    basis = tmp_path / "planted.npz"
    save_basis(basis, ToyDynamics(masked=3).planted_dictionary())
    record = tmp_path / "records" / "steer-record.json"
    args = [
        "latents", "steer", "--adapter", "toy-dynamics", "--adapter-option", "masked=3",
        "--basis", str(basis), "--layer", "1", "--feature", "0", "--amount", "1.5",
        "--time", "1", "--steps", "5", "--seeds", "0,1", "--random-draws", "12",
        "--record", str(record), "--json",
    ]  # fmt: skip
    done = CliRunner().invoke(cli, args)
    assert done.exit_code == 0, done.output
    return json.loads(done.output), record


def write_evaluation_record(tmp_path: Path):
    """Run ``latents evaluate`` on a toy archive, two PCAs fitted on its training times, with
    ``--record``: ``(full result, path, the two basis files)``."""
    import json

    from click.testing import CliRunner

    from xaig._cli import cli

    run = CliRunner()
    archive = str(tmp_path / "toy")
    assert run.invoke(cli, ["latents", "toy", archive]).exit_code == 0
    fits = []
    for name in ("a.npz", "b.npz"):
        out = str(tmp_path / name)
        fit = ["latents", "pca", archive, "--components", "4", "--out", out]
        done = run.invoke(cli, [*fit, *(x for i in range(5) for x in ("--time", str(i)))])
        assert done.exit_code == 0, done.output
        fits.append(out)
    record = tmp_path / "records" / "eval-record.json"
    args = ["latents", "evaluate", archive, "--blocks", "4", "--pca", "1,4", "--stability"]
    for basis in fits:
        args += ["--basis", basis]
    done = run.invoke(cli, [*args, "--mask-variable", "sst", "--record", str(record), "--json"])
    assert done.exit_code == 0, done.output
    return json.loads(done.output), record, fits
