"""Read a latent archive: a directory of activations recorded from one model.

The layout is the interchange format between the environment that can run a
model and the one that studies it::

    manifest.json   times, layers, provenance, and what was done to the run
    grid.npz        lat, lon per node; optionally grid_shape, mask, area
    step_XX.npy     (n_times, n_nodes, n_channels), any float dtype, one per layer
    reference.nc    optional physical fields on the same grid

Layer files are memory-mapped, and only the requested time, nodes and channels
are ever read into memory. The reference file is opened with xarray, and only
when a mask or a field is asked of it.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from xaig.core.errors import AdapterError, RequestError
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
    """A ``LatentSource``, and ``ReferenceFields``, over one archive directory.

    The grid's mask comes from ``grid.npz`` when the archive carries one. For an
    archive that does not, ``mask_variable`` names a variable of the reference
    file that is missing exactly where nodes are meaningless -- ``sst`` for an
    ocean model, whose activations over land mean nothing.
    """

    def __init__(self, path: str | Path, mask_variable: str | None = None) -> None:
        self.path = Path(path)
        self.mask_variable = mask_variable
        manifest_path = self.path / MANIFEST
        if not self.path.is_dir():
            raise AdapterError(f"no such directory: {self.path}")
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
        experiment = self._manifest.get("experiment") or {}
        if not isinstance(experiment, dict):
            raise AdapterError(f"{manifest_path}: 'experiment' must be a mapping")
        self._fields: tuple[str, ...] | None = None
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
            experiment=experiment,
            options={"mask_variable": mask_variable} if mask_variable else {},
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

    def _reference(self) -> Path | None:
        name = self._manifest.get("reference_file")
        return self.path / name if name and (self.path / name).is_file() else None

    def _open_reference(self, wanted_for: str):
        path = self._reference()
        if path is None:
            raise AdapterError(f"{self.path}: {wanted_for} needs a reference file, and has none")
        # Undecoded times: emulators run on calendars (no-leap, year 425) that would
        # otherwise have to be understood just to be thrown away.
        return path, require("xarray", "daig").open_dataset(path, decode_times=False)

    def _mask_from_reference(self, variable: str) -> np.ndarray:
        path, dataset = self._open_reference("mask_variable")
        with dataset as ds:
            if variable not in ds.variables:
                raise AdapterError(f"{path}: no variable {variable!r} to take a mask from")
            field = ds[variable]
            # The first sample along whatever leads the grid: time, then level, ...
            while field.ndim > 1 and field.size != self._info.n_nodes:
                field = field.isel({field.dims[0]: 0})
            mask = field.notnull().values.ravel()
        if mask.size != self._info.n_nodes:
            raise AdapterError(
                f"{variable!r} has {mask.size} points, but the archive has "
                f"{self._info.n_nodes} nodes"
            )
        return mask

    # -- ReferenceFields ------------------------------------------------------

    def _reference_times(self) -> tuple[str, ...]:
        """Labels of the reference file's time axis. It usually holds more times
        than the latents do: the state each forward call started from, too."""
        return tuple(str(t) for t in self._manifest.get("reference_times") or self._info.times)

    def field_names(self) -> tuple[str, ...]:
        """Variables holding one value per node per reference time."""
        if self._fields is None:
            if self._reference() is None:
                self._fields = ()
            else:
                _, dataset = self._open_reference("a field")
                shape = (len(self._reference_times()), self._info.n_nodes)
                with dataset as ds:
                    self._fields = tuple(
                        sorted(
                            str(name)
                            for name, variable in ds.data_vars.items()
                            if variable.ndim >= 2
                            and (variable.shape[0], variable.size // variable.shape[0]) == shape
                        )
                    )
        return self._fields

    def field(self, name: str, time: str | int) -> np.ndarray:
        label = self._info.times[self._info.time_index(time)]
        if name not in self.field_names():
            known = ", ".join(self.field_names()) or "none"
            raise RequestError(f"no field {name!r} in {self.path}; fields are {known}")
        times = self._reference_times()
        if label not in times:
            raise RequestError(f"{self.path}: the reference file has no time {label!r}")
        _, dataset = self._open_reference("a field")
        with dataset as ds:
            variable = ds[name]
            values = variable.isel({variable.dims[0]: times.index(label)}).values
        return np.asarray(values, dtype=np.float64).ravel()

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
        try:
            if nodes is not None:
                block = block[np.asarray(nodes, dtype=np.intp)]
            if channels is not None:
                block = block[:, np.asarray(channels, dtype=np.intp)]
        except IndexError as exc:
            shape = self._array(layer).shape[1:]
            raise RequestError(
                f"layer {layer} holds {shape[0]} nodes x {shape[1]} channels ({exc})"
            ) from exc
        return np.array(block, dtype=np.float32)  # always a copy: the caller may modify it
