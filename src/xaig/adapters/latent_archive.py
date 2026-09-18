"""Read a latent archive: a directory of activations recorded from one model.

The layout is the interchange format between the environment that can run a
model and the one that studies it (``docs/package/latents.md`` is the reference)::

    manifest.json   times, layers, provenance
    grid.npz        lat, lon per node; optionally grid_shape, mask, area
    step_XX.npy     (n_times, n_nodes, n_channels), any float dtype, one per layer
    reference.nc    optional physical fields on the same grid

Layer files are memory-mapped, and only the requested time, nodes and channels
are ever read into memory.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xaig.core.errors import AdapterError
from xaig.core.extras import missing_extra, require
from xaig.daig.grid import Grid
from xaig.daig.latent.source import LatentInfo, LayerInfo

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "daig") from exc

MANIFEST = "manifest.json"
GRID = "grid.npz"


def _layers(entries: Any, where: Path) -> list[dict[str, Any]]:
    try:
        return [
            {
                "info": LayerInfo(int(e["index"]), str(e.get("label", "")), int(e["n_channels"])),
                "file": str(e["file"]),
            }
            for e in entries or []
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise AdapterError(
            f"{where}: each step needs 'index', 'n_channels' and 'file' ({exc!r})"
        ) from exc


class LatentArchive:
    """A ``LatentSource`` over one archive directory.

    The grid's mask comes from ``grid.npz`` when the archive carries one. For an
    archive that does not, ``mask_variable`` names a variable of the reference
    file that is missing exactly where nodes are meaningless -- ``sst`` for an
    ocean model, whose activations over land mean nothing. That path needs
    netCDF4 (the ``netcdf`` extra).
    """

    def __init__(self, path: str | Path, mask_variable: str | None = None) -> None:
        self.path = Path(path)
        self.mask_variable = mask_variable
        manifest_path = self.path / MANIFEST
        if not manifest_path.is_file():
            raise AdapterError(f"not a latent archive (no {MANIFEST}): {self.path}")
        try:
            self._manifest: dict[str, Any] = json.loads(manifest_path.read_text())
            times = tuple(str(t) for t in self._manifest["latent_times"])
            n_nodes = int(self._manifest["n_nodes"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise AdapterError(f"{manifest_path}: unreadable manifest ({exc!r})") from exc

        layers = _layers(self._manifest.get("steps"), manifest_path)
        if not layers:
            raise AdapterError(f"{manifest_path}: no 'steps' to read")
        self._files = {entry["info"].index: entry["file"] for entry in layers}
        if len(self._files) != len(layers):
            raise AdapterError(f"{manifest_path}: a step index appears more than once")
        self._arrays: dict[int, np.ndarray] = {}
        self._grid: Grid | None = None
        timestep = self._manifest.get("timestep_seconds")
        self._info = LatentInfo(
            source=str(self.path),
            times=times,
            layers=tuple(entry["info"] for entry in layers),
            n_nodes=n_nodes,
            model=self._manifest.get("model"),
            component=self._manifest.get("component"),
            checkpoint=self._manifest.get("checkpoint"),
            calendar=self._manifest.get("calendar"),
            timestep_seconds=None if timestep is None else int(timestep),
            off_grid_layers=tuple(
                entry["info"] for entry in _layers(self._manifest.get("extra_steps"), manifest_path)
            ),
        )

    def info(self) -> LatentInfo:
        return self._info

    def grid(self) -> Grid:
        if self._grid is None:
            self._grid = self._read_grid()
        return self._grid

    def _read_grid(self) -> Grid:
        grid_path = self.path / GRID
        if not grid_path.is_file():
            raise AdapterError(f"{self.path}: no {GRID}")
        with np.load(grid_path) as stored:
            names = set(stored.files)
            if not {"lat", "lon"} <= names:
                raise AdapterError(f"{grid_path}: needs 'lat' and 'lon', has {sorted(names)}")
            lat, lon = (np.asarray(stored[k], dtype=np.float64).ravel() for k in ("lat", "lon"))
            shape = tuple(int(v) for v in stored["grid_shape"]) if "grid_shape" in names else None
            mask = np.asarray(stored["mask"], dtype=bool).ravel() if "mask" in names else None
            area = np.asarray(stored["area"], dtype=np.float64).ravel() if "area" in names else None
        if lat.size != self._info.n_nodes:
            raise AdapterError(
                f"{grid_path}: {lat.size} nodes, but the manifest says {self._info.n_nodes}"
            )
        if mask is None and self.mask_variable:
            mask = self._mask_from_reference(self.mask_variable)
        try:
            return Grid(lat=lat, lon=lon, shape=shape, mask=mask, area=area)
        except ValueError as exc:
            raise AdapterError(f"{grid_path}: {exc}") from exc

    def _mask_from_reference(self, variable: str) -> np.ndarray:
        name = self._manifest.get("reference_file")
        if not name or not (self.path / name).is_file():
            raise AdapterError(f"{self.path}: mask_variable needs a reference file, and has none")
        netcdf = require("netCDF4", "netcdf")
        with netcdf.Dataset(self.path / name) as ds:
            if variable not in ds.variables:
                raise AdapterError(
                    f"{self.path / name}: no variable {variable!r} to take a mask from"
                )
            first = np.ma.masked_invalid(ds.variables[variable][0])
        mask = ~np.ma.getmaskarray(first).ravel()
        if mask.size != self._info.n_nodes:
            raise AdapterError(
                f"{variable!r} has {mask.size} points, but the archive has "
                f"{self._info.n_nodes} nodes"
            )
        return mask

    def _array(self, layer: int) -> np.ndarray:
        if layer not in self._arrays:
            info = self._info.layer(layer)
            file = self.path / self._files[layer]
            if not file.is_file():
                raise AdapterError(f"layer {layer}: {file} is missing")
            array = np.load(file, mmap_mode="r")
            expected = (len(self._info.times), self._info.n_nodes, info.n_channels)
            if array.shape != expected:
                raise AdapterError(
                    f"{file}: shape {array.shape}, but the manifest implies {expected}"
                )
            self._arrays[layer] = array
        return self._arrays[layer]

    def load(
        self,
        time: str | int,
        layer: int,
        channels: Sequence[int] | None = None,
        nodes: Sequence[int] | None = None,
    ) -> np.ndarray:
        block = self._array(layer)[self._info.time_index(time)]
        # Rows first: on a memory map this touches only the pages those nodes
        # live in, which is what keeps a regional read cheap.
        if nodes is not None:
            block = block[np.asarray(nodes, dtype=np.intp)]
        if channels is not None:
            block = block[:, np.asarray(channels, dtype=np.intp)]
        return np.array(block, dtype=np.float32)  # always a copy: the caller may modify it
