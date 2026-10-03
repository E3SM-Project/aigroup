"""Read a bundle: a second latent layout, unlike the archive on purpose.

It exists to show that a system with another file layout, another vocabulary and another
dimension order is a new module here and an entry point, and nothing else. Where an archive
keeps ``(n_times, n_nodes, n_channels)`` in one file per layer, a bundle keeps channels
first, on a grid that is two-dimensional, one file per level *and* time::

    bundle.json              what the system is, its clock, its levels, its grid, notes
    coords.npz               lat (n_lat,), lon (n_lon,); optional ocean (n_lat, n_lon) bool,
                             True where the model says something; optional cell_area
    levels/L00/T0000.npy     (width, n_lat, n_lon), any float dtype
    fields/<name>.npy        optional (n_stamps, n_lat, n_lon): physical fields

Nodes are the grid read in C order, ``node = i_lat * n_lon + i_lon``, which is what every
``Grid`` of xaig means. Files are memory-mapped, and a read takes the channels asked for
first and the nodes second, so a region of one level touches only the pages it needs.

``bundle.json`` says::

    {"format": "xaig-bundle", "version": 1,
     "system": {"name": ..., "part": ..., "weights": ...},
     "clock": {"calendar": "noleap", "step_seconds": 21600, "stamps": [...],
               "field_stamps": [...]},
     "levels": [{"id": 0, "name": ..., "width": 16, "network_position": 4}],
     "grid": [n_lat, n_lon], "fields": [...], "notes": {...}}

``write`` is the other half, with the signature ``write_archive`` has, so the toy emulator
(``write_toy(path, adapter="bundle-dir")``) and anything else that writes archives writes
bundles too, and the reader is tested against it.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from xaig.core.errors import AdapterError, RequestError
from xaig.core.extras import missing_extra
from xaig.latents.grid import Grid
from xaig.latents.source import LatentInfo, LayerInfo, selection

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "latents") from exc

MANIFEST = "bundle.json"
COORDS = "coords.npz"
FORMAT = "xaig-bundle"
VERSION = 1


def _level_file(level: int, stamp: int) -> str:
    return f"levels/L{level:02d}/T{stamp:04d}.npy"


class BundleDir:
    """A ``LatentSource``, and ``ReferenceFields``, over one bundle directory.

    ``unmasked`` ignores the bundle's ocean mask and treats every node as valid, for a
    reader who wants what the model wrote over land too."""

    def __init__(self, path: str | Path, *, unmasked: bool = False) -> None:
        self.path = Path(path)
        self.unmasked = bool(unmasked)
        if not self.path.is_dir():
            raise AdapterError(f"no such directory: {self.path}")
        manifest_path = self.path / MANIFEST
        if not manifest_path.is_file():
            raise AdapterError(f"not a bundle (no {MANIFEST}): {self.path}")
        try:
            manifest = json.loads(manifest_path.read_text())
            if manifest["format"] != FORMAT or manifest["version"] != VERSION:
                raise AdapterError(
                    f"{manifest_path}: format {manifest['format']!r} version "
                    f"{manifest['version']!r}; this reads {FORMAT!r} version {VERSION}"
                )
            self._shape = (int(manifest["grid"][0]), int(manifest["grid"][1]))
            clock, system = manifest["clock"], manifest.get("system") or {}
            stamps = tuple(str(t) for t in clock["stamps"])
            levels = [
                LayerInfo(
                    int(level["id"]),
                    str(level.get("name", "")),
                    int(level["width"]),
                    None
                    if level.get("network_position") is None
                    else int(level["network_position"]),
                )
                for level in manifest["levels"]
            ]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise AdapterError(f"{manifest_path}: unreadable manifest ({exc!r})") from exc
        if not levels or not stamps:
            raise AdapterError(f"{manifest_path}: a bundle needs levels and stamps")
        if len({level.index for level in levels}) != len(levels):
            raise AdapterError(f"{manifest_path}: a level id appears more than once")
        notes = manifest.get("notes") or {}
        if not isinstance(notes, dict):
            raise AdapterError(f"{manifest_path}: 'notes' must be a mapping")
        self._field_names = tuple(str(n) for n in manifest.get("fields") or ())
        self._field_stamps = tuple(str(t) for t in clock.get("field_stamps") or stamps)
        self._grid: Grid | None = None
        step = clock.get("step_seconds")
        self._info = LatentInfo(
            source=str(self.path),
            times=stamps,
            layers=tuple(levels),
            n_nodes=self._shape[0] * self._shape[1],
            model=system.get("name"),
            component=system.get("part"),
            checkpoint=system.get("weights"),
            calendar=clock.get("calendar"),
            timestep_seconds=None if step is None else int(step),
            experiment=notes,
            options={"unmasked": True} if self.unmasked else {},
        )

    @staticmethod
    def write(path: str | Path, **contents: Any) -> Path:
        """``write_bundle``, reachable from the class the registry hands out."""
        return write_bundle(path, **contents)

    def info(self) -> LatentInfo:
        return self._info

    def grid(self) -> Grid:
        if self._grid is None:
            self._grid = self._read_grid()
        return self._grid

    def _read_grid(self) -> Grid:
        file = self.path / COORDS
        if not file.is_file():
            raise AdapterError(f"{self.path}: no {COORDS}")
        with np.load(file) as stored:
            names = set(stored.files)
            if not {"lat", "lon"} <= names:
                raise AdapterError(f"{file}: needs 'lat' and 'lon', has {sorted(names)}")
            lat_1d, lon_1d = (np.asarray(stored[k], dtype=np.float64) for k in ("lat", "lon"))
            ocean = np.asarray(stored["ocean"], dtype=bool) if "ocean" in names else None
            area = (
                np.asarray(stored["cell_area"], dtype=np.float64) if "cell_area" in names else None
            )
        if (lat_1d.size, lon_1d.size) != self._shape:
            raise AdapterError(
                f"{file}: {lat_1d.size} latitudes and {lon_1d.size} longitudes, but the "
                f"manifest says a grid of {self._shape}"
            )
        lat, lon = np.meshgrid(lat_1d, lon_1d, indexing="ij")
        mask = None if self.unmasked or ocean is None else ocean.ravel()
        try:
            return Grid(
                lat=lat.ravel(), lon=lon.ravel(), shape=self._shape, mask=mask,
                area=None if area is None else area.ravel(),
            )  # fmt: skip
        except ValueError as exc:
            raise AdapterError(f"{file}: {exc}") from exc

    # -- ReferenceFields ------------------------------------------------------

    def field_names(self) -> tuple[str, ...]:
        return self._field_names

    def field(self, name: str, time: str | int, lead: int = 0) -> np.ndarray:
        label = self._info.times[self._info.time_index(time)]
        if name not in self._field_names:
            known = ", ".join(self._field_names) or "none"
            raise RequestError(f"no field {name!r} in {self.path}; fields are {known}")
        if label not in self._field_stamps:
            raise RequestError(f"{self.path}: the fields hold no time {label!r}")
        index = self._field_stamps.index(label) + lead
        if not 0 <= index < len(self._field_stamps):
            raise RequestError(f"{self.path}: the fields end before {lead:+d} time(s) from {label}")
        stored = self._open(f"fields/{name}.npy", (len(self._field_stamps), *self._shape))
        return np.asarray(stored[index], dtype=np.float64).ravel()

    # -- LatentSource ---------------------------------------------------------

    def _open(self, name: str, expected: tuple[int, ...]) -> np.ndarray:
        file = self.path / name
        if not file.is_file():
            raise AdapterError(f"{file} is missing")
        array = np.load(file, mmap_mode="r")
        if array.shape != expected:
            raise AdapterError(f"{file}: shape {array.shape}, but the manifest implies {expected}")
        return array

    def load(
        self,
        time: str | int,
        layer: int,
        channels: Sequence[int] | None = None,
        nodes: Sequence[int] | None = None,
    ) -> np.ndarray:
        width = self._info.layer(layer).n_channels
        stamp = self._info.time_index(time)
        block = self._open(_level_file(layer, stamp), (width, *self._shape))
        block = block.reshape(width, self._info.n_nodes)  # a view: the file is C-ordered
        # Channels (the leading axis) first, then nodes: on a memory map the rows asked
        # for are the only pages read.
        held = f"level {layer} holds {width} channels x {self._info.n_nodes} nodes"
        if channels is not None:
            block = block[selection(channels, width, "channel", held)]
        if nodes is not None:
            block = block[:, selection(nodes, block.shape[1], "node", held)]
        return np.array(block.T, dtype=np.float32, order="C")  # a copy, nodes first


def write_bundle(
    path: str | Path,
    *,
    grid: Grid,
    times: Sequence[str],
    layers: Sequence[tuple[str, np.ndarray]],
    network_layers: Sequence[int] | None = None,
    fields: Mapping[str, np.ndarray] | None = None,
    field_times: Sequence[str] | None = None,
    model: str | None = None,
    component: str | None = None,
    checkpoint: str | None = None,
    calendar: str | None = None,
    timestep_seconds: int | None = None,
    experiment: Mapping[str, Any] | None = None,
    overwrite: bool = False,
) -> Path:
    """Write one bundle directory, in float32, from what ``write_archive`` is given:
    ``layers`` as ``(label, array)`` pairs of ``(n_times, n_nodes, n_channels)``,
    ``fields`` as ``(n_field_times, n_nodes)``. Only a grid of a shape (latitudes by
    longitudes) fits the layout. A demonstration, so written in place, not staged."""
    out = Path(path)
    times = [str(t) for t in times]
    if grid.shape is None:
        raise RequestError("a bundle holds a latitude-longitude grid; this one is a mesh")
    n_lat, n_lon = grid.shape
    if not times or len(set(times)) != len(times):
        raise RequestError("a bundle needs at least one time, and each only once")
    if not layers:
        raise RequestError("a bundle needs at least one level")
    if network_layers is not None and len(network_layers) != len(layers):
        raise RequestError(f"{len(network_layers)} network layer(s) for {len(layers)} level(s)")
    for index, (label, array) in enumerate(layers):
        if array.ndim != 3 or array.shape[:2] != (len(times), grid.n_nodes):
            raise RequestError(
                f"level {index} ({label!r}) has shape {array.shape}; expected "
                f"({len(times)} times, {grid.n_nodes} nodes, channels)"
            )
        if array.dtype.kind not in "biuf":
            raise RequestError(f"level {index} must hold real numeric values")
    fields = dict(fields or {})
    stamps = [str(t) for t in (field_times if field_times is not None else times)]
    if fields:
        if len(set(stamps)) != len(stamps) or not set(times) <= set(stamps):
            raise RequestError("field times must be unique and hold every latent time")
        for name, values in fields.items():
            if values.shape != (len(stamps), grid.n_nodes):
                raise RequestError(
                    f"field {name!r} has shape {values.shape}; expected "
                    f"({len(stamps)} times, {grid.n_nodes} nodes)"
                )
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise RequestError(f"{out} must be a directory, not a file or symlink")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise RequestError(f"{out} is not empty; pass overwrite=True to replace it")

    levels = []
    for index, (label, array) in enumerate(layers):
        level = {"id": index, "name": str(label), "width": int(array.shape[2])}
        if network_layers is not None:
            level["network_position"] = int(network_layers[index])
        levels.append(level)
    clock = {"calendar": calendar, "step_seconds": timestep_seconds, "stamps": times}
    if fields:
        clock["field_stamps"] = stamps
    manifest: dict[str, Any] = {
        "format": FORMAT, "version": VERSION,
        "system": {"name": model, "part": component, "weights": checkpoint},
        "clock": {k: v for k, v in clock.items() if v is not None},
        "levels": levels, "grid": [n_lat, n_lon], "fields": sorted(fields),
        "notes": dict(experiment or {}),
    }  # fmt: skip
    try:
        metadata = json.dumps(manifest, indent=2)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"bundle metadata must be JSON serializable: {exc}") from exc

    coords: dict[str, np.ndarray] = {
        "lat": grid.lat.reshape(grid.shape)[:, 0], "lon": grid.lon.reshape(grid.shape)[0, :]
    }  # fmt: skip
    if grid.mask is not None:
        coords["ocean"] = np.asarray(grid.mask, dtype=bool).reshape(grid.shape)
    if grid.area is not None:
        coords["cell_area"] = np.asarray(grid.area).reshape(grid.shape)

    if out.exists():
        shutil.rmtree(out)
    (out / "levels").mkdir(parents=True)
    np.savez(out / COORDS, **coords)
    for index, (_, array) in enumerate(layers):
        (out / "levels" / f"L{index:02d}").mkdir()
        for stamp in range(len(times)):
            block = array[stamp].T.reshape(array.shape[2], n_lat, n_lon)
            np.save(out / _level_file(index, stamp), block.astype(np.float32))
    if fields:
        (out / "fields").mkdir()
        for name, values in fields.items():
            block = values.reshape(len(stamps), n_lat, n_lon).astype(np.float32)
            np.save(out / "fields" / f"{name}.npy", block)
    (out / MANIFEST).write_text(metadata)
    return out
