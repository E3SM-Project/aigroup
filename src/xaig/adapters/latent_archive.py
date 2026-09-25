"""Read a latent archive: a directory of activations recorded from one model.

The layout is the interchange format between the environment that can run a
model and the one that studies it (``docs/package/latents.md`` is the reference)::

    manifest.json   times, layers, provenance, and what was done to the run
    grid.npz        lat, lon per node; optionally grid_shape, mask, area
    step_XX.npy     (n_times, n_nodes, n_channels), any float dtype, one per layer
    reference.nc    optional physical fields on the same grid

Layer files are memory-mapped, and only the requested time, nodes and channels
are ever read into memory. The reference file is opened with xarray, and only
when a mask or a field is asked of it.

An archive can also be read straight from a Hugging Face dataset repository,
``hf://datasets/<owner>/<repo>/<folder>`` (``xaig[hf]``). Each file is downloaded
the first time something needs it and cached, so a notebook that looks at one
layer downloads one layer. Every file comes from the one revision the archive was
opened at.

``write_archive`` is the other half: whatever can hand over arrays -- a toy model,
an exporter hooked into a real one -- writes the layout through it, so the writer
and the reader are tested against each other rather than against a description.
An archive too large to hold in memory is written a time at a time instead:
``start_archive`` lays it out, ``ArchiveFiller`` fills it (from several processes
at once, each on its own times), and ``finish_archive`` makes it readable.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import warnings
from collections.abc import Mapping, Sequence
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

HF_PREFIX = "hf://datasets/"
MANIFEST = "manifest.json"
GRID = "grid.npz"
REFERENCE = "reference.nc"
_WHOLE_READ_FRACTION = 8  # read a whole time once this share of its nodes is asked for
PARTIAL = "manifest.partial.json"  # an archive being filled; the reader refuses it
WRITTEN = "written.npy"  # (n_times, n_layers) flags, one per cell filled


class _HubFolder:
    """One folder of a Hugging Face dataset repository, fetched a file at a time."""

    def __init__(self, url: str, revision: str | None) -> None:
        rest = url[len(HF_PREFIX) :].strip("/")
        parts = rest.split("/")
        if len(parts) < 3 or not all(parts):
            raise AdapterError(f"expected {HF_PREFIX}<owner>/<repo>/<folder>, got {url!r}")
        self.url = url
        self.repo_id = "/".join(parts[:2])
        self.folder = "/".join(parts[2:])
        hub = require("huggingface_hub", "hf")
        self._api = hub.HfApi()
        try:
            self.revision = revision or self._api.repo_info(self.repo_id, repo_type="dataset").sha
        except Exception as exc:  # the hub's own errors: no network, no such repo, no access
            raise AdapterError(f"{url}: cannot reach the dataset repository ({exc})") from exc
        self._download = hub.hf_hub_download

    def fetch(self, name: str) -> Path | None:
        """Local path of one file of the folder, or None when the repository lacks it."""
        from huggingface_hub.utils import EntryNotFoundError

        try:
            local = self._download(
                self.repo_id,
                f"{self.folder}/{name}",
                repo_type="dataset",
                revision=self.revision,
            )
        except EntryNotFoundError:
            return None
        except Exception as exc:
            raise AdapterError(f"{self.url}: could not download {name} ({exc})") from exc
        return Path(local)

    def list(self, folder: str) -> list[str]:
        """Names of the files directly inside ``folder`` of this one, downloading none."""
        from huggingface_hub.utils import EntryNotFoundError

        where = f"{self.folder}/{folder.strip('/')}"
        try:
            entries = list(
                self._api.list_repo_tree(
                    self.repo_id, path_in_repo=where, repo_type="dataset", revision=self.revision
                )
            )
        except EntryNotFoundError:
            return []
        except Exception as exc:
            raise AdapterError(f"{self.url}: could not list {folder} ({exc})") from exc
        return [e.path.rsplit("/", 1)[-1] for e in entries if getattr(e, "size", None) is not None]


def _layers(entries: Any, where: Path) -> list[dict[str, Any]]:
    try:
        return [
            {
                "info": LayerInfo(
                    int(e["index"]),
                    str(e.get("label", "")),
                    int(e["n_channels"]),
                    None if e.get("network_layer") is None else int(e["network_layer"]),
                ),
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

    def __init__(
        self, path: str | Path, mask_variable: str | None = None, revision: str | None = None
    ) -> None:
        self.mask_variable = mask_variable
        self._hub: _HubFolder | None = None
        if str(path).startswith(HF_PREFIX):
            self._hub = _HubFolder(str(path), revision)
            fetched = self._hub.fetch(MANIFEST)
            if fetched is None:
                raise AdapterError(f"not a latent archive (no {MANIFEST}): {path}")
            self.path = fetched.parent
        elif revision is not None:
            raise AdapterError("'revision' applies to hf:// sources only")
        else:
            self.path = Path(path)
        self._name = str(path) if self._hub is not None else str(self.path)
        manifest_path = self.path / MANIFEST
        if not self.path.is_dir():
            raise AdapterError(f"no such directory: {self.path}")
        if not manifest_path.is_file():
            if (self.path / PARTIAL).is_file():
                raise AdapterError(f"{self.path} is still being written (finish_archive)")
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
            source=self._name,
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

    @staticmethod
    def write(path: str | Path, **contents: Any) -> Path:
        """``write_archive``, reachable from the class the registry hands out."""
        return write_archive(path, **contents)

    def info(self) -> LatentInfo:
        return self._info

    def file(self, name: str) -> Path | None:
        """Local path of a file of the archive (``bases/sae_L08.npz``, say), or None
        when there is no such file. From a hub, this is when it is downloaded."""
        local = self.path / name
        if self._hub is not None and not local.is_file():
            fetched = self._hub.fetch(name)
            return fetched if fetched is not None and fetched.is_file() else None
        return local if local.is_file() else None

    def files(self, folder: str) -> tuple[str, ...]:
        """The files directly inside one folder of the archive, as ``file`` names them
        (``bases/sae_L08.npz``), sorted; none when there is no such folder. Nothing is
        downloaded: a hub archive is listed where it is."""
        folder = folder.strip("/")
        if self._hub is not None:
            names = self._hub.list(folder)
        else:
            where = self.path / folder
            names = [p.name for p in where.iterdir() if p.is_file()] if where.is_dir() else []
        return tuple(sorted(f"{folder}/{name}" for name in names))

    def grid(self) -> Grid:
        if self._grid is None:
            self._grid = self._read_grid()
        return self._grid

    def _read_grid(self) -> Grid:
        grid_path = self.file(GRID)
        if grid_path is None:
            raise AdapterError(f"{self._name}: no {GRID}")
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
        return self.file(name) if name else None

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

    def field(self, name: str, time: str | int, lead: int = 0) -> np.ndarray:
        label = self._info.times[self._info.time_index(time)]
        if name not in self.field_names():
            known = ", ".join(self.field_names()) or "none"
            raise RequestError(f"no field {name!r} in {self.path}; fields are {known}")
        times = self._reference_times()
        if label not in times:
            raise RequestError(f"{self.path}: the reference file has no time {label!r}")
        index = times.index(label) + lead
        if not 0 <= index < len(times):
            raise RequestError(
                f"{self.path}: the reference file ends before {lead:+d} time(s) from {label}"
            )
        _, dataset = self._open_reference("a field")
        with dataset as ds:
            variable = ds[name]
            values = variable.isel({variable.dims[0]: index}).values
        return np.asarray(values, dtype=np.float64).ravel()

    def _array(self, layer: int) -> np.ndarray:
        if layer not in self._arrays:
            info = self._info.layer(layer)
            file = self.file(self._files[layer]) or self.path / self._files[layer]
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
        # live in, which is what keeps a regional read cheap. Many nodes in no
        # particular order -- a training batch drawn over the globe -- are the
        # opposite case: a page fault each, in random order, defeats read-ahead,
        # and on Lustre reading the whole time in one pass and picking the rows
        # from memory was 14x faster (0.32 s against 4.6 s for 64,800 of 64,800).
        try:
            if nodes is not None:
                rows = np.asarray(nodes, dtype=np.intp)
                if rows.size * _WHOLE_READ_FRACTION >= block.shape[0]:
                    block = np.array(block)  # a copy: one sequential read, not a view
                block = block[rows]
            if channels is not None:
                block = block[:, np.asarray(channels, dtype=np.intp)]
        except IndexError as exc:
            shape = self._array(layer).shape[1:]
            raise RequestError(
                f"layer {layer} holds {shape[0]} nodes x {shape[1]} channels ({exc})"
            ) from exc
        return np.array(block, dtype=np.float32)  # always a copy: the caller may modify it


def write_archive(
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
    dtype: str = "float16",
    overwrite: bool = False,
) -> Path:
    """Write one archive directory in the layout ``LatentArchive`` reads.

    ``layers`` are ``(label, array)`` pairs, each array ``(n_times, n_nodes,
    n_channels)``, indexed by their position. ``network_layers`` says where each sits
    in the network, for an archive that keeps some layers and not others: a basis is
    matched to a layer by it. ``fields`` are physical fields on
    the same nodes, each ``(n_field_times, n_nodes)``; ``field_times`` labels their
    time axis and must hold every latent time -- it usually holds more, the state
    each step started from among them. The grid's mask and area travel in
    ``grid.npz`` when the grid has them. Field time labels must be unique.

    Metadata and shapes are checked up front; files are written in a sibling
    staging directory before replacing the destination. Failed writes leave an
    existing archive intact. Float conversion may round values, but overflow is
    refused; choose a wider ``dtype`` when necessary. Existing NaNs are preserved.
    Replacement is not crash-atomic and concurrent writers are not supported.
    """
    out = Path(path)
    times = [str(t) for t in times]
    n_nodes = grid.n_nodes
    storage_dtype = _storage_dtype(dtype)
    _check_times(times)
    if not layers:
        raise RequestError("an archive needs at least one layer")
    for index, (label, array) in enumerate(layers):
        if array.ndim != 3 or array.shape[:2] != (len(times), n_nodes):
            raise RequestError(
                f"layer {index} ({label!r}) has shape {array.shape}; expected "
                f"({len(times)} times, {n_nodes} nodes, channels)"
            )
    fields = dict(fields or {})
    labels = [str(t) for t in (field_times if field_times is not None else times)]
    if fields:
        if len(set(labels)) != len(labels):
            raise RequestError("field times must each appear only once")
        absent = [t for t in times if t not in labels]
        if absent:
            raise RequestError(f"the fields hold no time {absent[0]!r}, which the latents do")
        for name, values in fields.items():
            if values.shape != (len(labels), n_nodes):
                raise RequestError(
                    f"field {name!r} has shape {values.shape}; expected "
                    f"({len(labels)} times, {n_nodes} nodes)"
                )
    _check_destination(out, overwrite)

    stored = _grid_arrays(grid)
    steps = _steps([(label, array.shape[2]) for label, array in layers], None, network_layers)
    manifest = _manifest(
        grid, times, steps, model=model, component=component, checkpoint=checkpoint,
        calendar=calendar, timestep_seconds=timestep_seconds, experiment=experiment,
    )  # fmt: skip
    if fields:
        manifest["reference_file"] = REFERENCE
        manifest["reference_times"] = labels
    metadata = _dumps(manifest)

    out.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f".{out.name}-", dir=out.parent))
    staged, previous = work / "archive", work / "previous"
    try:
        staged.mkdir()
        np.savez(staged / GRID, **stored)
        for entry, (_, array) in zip(steps, layers, strict=True):
            if array.dtype.kind not in "biuf":
                raise RequestError(f"layer {entry['index']} must hold real numeric values")
            try:
                with np.errstate(over="raise", invalid="raise"):
                    converted = np.asarray(array, dtype=storage_dtype)
            except FloatingPointError as exc:
                raise RequestError(
                    f"layer {entry['index']} overflows {storage_dtype}; choose a wider dtype"
                ) from exc
            np.save(staged / entry["file"], converted)
            del converted
        if fields:
            _write_reference(staged / REFERENCE, grid, fields)
        (staged / MANIFEST).write_text(metadata)
        if out.exists():
            out.rename(previous)
        try:
            staged.rename(out)
        except BaseException:
            if previous.exists():
                previous.rename(out)
            raise
        if previous.exists():
            shutil.rmtree(previous)
    finally:
        # If rollback itself failed, retain the backup for manual recovery.
        if not previous.exists():
            shutil.rmtree(work)
    return out


def _write_reference(file: Path, grid: Grid, fields: Mapping[str, np.ndarray]) -> None:
    xarray = require("xarray", "daig")
    n_times = next(iter(fields.values())).shape[0]
    if grid.shape is not None:
        dims: tuple[str, ...] = ("time", "lat", "lon")
        shape: tuple[int, ...] = (n_times, *grid.shape)
        coords = {
            "lat": grid.lat.reshape(grid.shape)[:, 0],
            "lon": grid.lon.reshape(grid.shape)[0, :],
        }
    else:
        dims, shape, coords = ("time", "node"), (n_times, grid.n_nodes), {}
    # Times are labelled by the manifest; the file's own axis is positions, so no
    # calendar has to be encoded only to be ignored on the way back in.
    variables = {
        name: (dims, np.asarray(values, dtype=np.float32).reshape(shape))
        for name, values in fields.items()
    }
    dataset = xarray.Dataset(variables, coords={"time": np.arange(n_times), **coords})
    with warnings.catch_warnings():
        # netCDF4 1.7's write path trips a NumPy 2.5 deprecation of its own.
        warnings.simplefilter("ignore", DeprecationWarning)
        dataset.to_netcdf(file)


# -- shared by both writers --------------------------------------------------


def _storage_dtype(dtype: str) -> np.dtype:
    try:
        storage_dtype = np.dtype(dtype)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"invalid archive dtype {dtype!r}") from exc
    if storage_dtype.kind != "f":
        raise RequestError(f"archive dtype must be floating point, got {dtype!r}")
    return storage_dtype


def _check_times(times: Sequence[str]) -> None:
    if not times or len(set(times)) != len(times):
        raise RequestError("an archive needs at least one time, and each only once")


def _check_destination(out: Path, overwrite: bool) -> None:
    if out.is_symlink() or (out.exists() and not out.is_dir()):
        raise RequestError(f"{out} must be a directory, not a file or symlink")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise RequestError(f"{out} is not empty; pass overwrite=True to replace it")


def _grid_arrays(grid: Grid) -> dict[str, np.ndarray]:
    stored: dict[str, np.ndarray] = {"lat": grid.lat, "lon": grid.lon}
    if grid.shape is not None:
        stored["grid_shape"] = np.array(grid.shape)
        stored["lat_1d"] = grid.lat.reshape(grid.shape)[:, 0]
        stored["lon_1d"] = grid.lon.reshape(grid.shape)[0, :]
    if grid.mask is not None:
        stored["mask"] = grid.mask
    if grid.area is not None:
        stored["area"] = grid.area
    return stored


def _steps(
    layers: Sequence[tuple[str, int]],
    directories: Sequence[str] | None,
    network_layers: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """One manifest entry per layer: ``step_XX.npy`` beside the manifest, or
    ``<directory>/latents.npy`` when each layer is given a directory of its own;
    with ``network_layers``, where each sits in the network."""
    if network_layers is not None and len(network_layers) != len(layers):
        raise RequestError(
            f"{len(network_layers)} network layer(s) for {len(layers)} layer(s); give one each"
        )
    if directories is not None:
        if len(directories) != len(layers) or len(set(directories)) != len(directories):
            raise RequestError("give each layer its own directory, one per layer")
        bad = [d for d in directories if not d or Path(d).is_absolute() or ".." in Path(d).parts]
        if bad:
            raise RequestError(f"a layer directory must be a plain relative name, not {bad[0]!r}")
    files = [
        f"step_{index:02d}.npy" if directories is None else f"{directories[index]}/latents.npy"
        for index in range(len(layers))
    ]
    steps = [
        {"index": index, "label": str(label), "file": file, "n_channels": int(n_channels)}
        for index, ((label, n_channels), file) in enumerate(zip(layers, files, strict=True))
    ]
    if network_layers is not None:
        for step, position in zip(steps, network_layers, strict=True):
            step["network_layer"] = int(position)
    return steps


def _manifest(
    grid: Grid,
    times: Sequence[str],
    steps: list[dict[str, Any]],
    *,
    model: str | None,
    component: str | None,
    checkpoint: str | None,
    calendar: str | None,
    timestep_seconds: int | None,
    experiment: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "model": model,
        "component": component,
        "checkpoint": checkpoint,
        "calendar": calendar,
        "timestep_seconds": timestep_seconds,
        "n_nodes": grid.n_nodes,
        "latent_times": list(times),
        "steps": steps,
        "extra_steps": [],
    }
    if grid.shape is not None:
        manifest["grid_shape"] = list(grid.shape)
    if experiment:
        manifest["experiment"] = dict(experiment)
    return {key: value for key, value in manifest.items() if value is not None}


def _dumps(manifest: Mapping[str, Any]) -> str:
    try:
        return json.dumps(manifest, indent=2)
    except (TypeError, ValueError) as exc:
        raise RequestError(f"archive metadata must be JSON serializable: {exc}") from exc


# -- writing an archive a time at a time ---------------------------------------


def start_archive(
    path: str | Path,
    *,
    grid: Grid,
    times: Sequence[str],
    layers: Sequence[tuple[str, int]],
    directories: Sequence[str] | None = None,
    network_layers: Sequence[int] | None = None,
    model: str | None = None,
    component: str | None = None,
    checkpoint: str | None = None,
    calendar: str | None = None,
    timestep_seconds: int | None = None,
    experiment: Mapping[str, Any] | None = None,
    dtype: str = "float32",
    overwrite: bool = False,
) -> Path:
    """Lay out an archive to be filled a time at a time, for when its layers do
    not fit in memory: one layer of a 1-degree, 384-channel model over two
    thousand times is 200 GB in float32.

    ``layers`` are ``(label, n_channels)`` pairs. With ``directories`` each layer
    lives in a directory of its own (``<directory>/latents.npy``), where whatever
    is later made from it can sit beside it; ``network_layers`` is as for
    ``write_archive``. Every layer file is allocated at its
    full size, which a file system that supports holes does not spend until it is
    written. Until ``finish_archive`` the manifest is ``manifest.partial.json``,
    and the reader refuses the directory: an unfilled cell would read as zeros.
    """
    out = Path(path)
    times = [str(t) for t in times]
    storage_dtype = _storage_dtype(dtype)
    _check_times(times)
    if not layers:
        raise RequestError("an archive needs at least one layer")
    if any(int(n) < 1 for _, n in layers):
        raise RequestError("every layer needs at least one channel")
    _check_destination(out, overwrite)
    steps = _steps(layers, directories, network_layers)
    manifest = _manifest(
        grid, times, steps, model=model, component=component, checkpoint=checkpoint,
        calendar=calendar, timestep_seconds=timestep_seconds, experiment=experiment,
    )  # fmt: skip
    manifest["dtype"] = storage_dtype.name
    metadata = _dumps(manifest)

    if out.exists() and any(out.iterdir()):
        shutil.rmtree(out)
    # An empty directory is kept, not remade: it may carry file-system settings
    # (Lustre striping, say) that every layer file made in it inherits.
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / GRID, **_grid_arrays(grid))
    for entry in steps:
        file = out / entry["file"]
        file.parent.mkdir(parents=True, exist_ok=True)
        shape = (len(times), grid.n_nodes, entry["n_channels"])
        allocated = np.lib.format.open_memmap(file, mode="w+", dtype=storage_dtype, shape=shape)
        del allocated  # the header and the length are all that is written
    np.save(out / WRITTEN, np.zeros((len(times), len(steps)), dtype=np.uint8))
    (out / PARTIAL).write_text(metadata)
    return out


class ArchiveFiller:
    """Fills an archive laid out by ``start_archive``, one layer at one time per
    ``put``.

    Several fillers may work on one archive at once -- a process per GPU, say --
    so long as no two write the same time: each cell is a separate region of the
    files, and ``written.npy`` flags each cell once it is complete. ``missing()``
    says which times are yet to be done, so a filler interrupted part way is
    resumed rather than restarted.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        partial = self.path / PARTIAL
        if not partial.is_file():
            if (self.path / MANIFEST).is_file():
                raise RequestError(f"{self.path} is finished; start a new archive to write")
            raise AdapterError(f"not an archive being filled (no {PARTIAL}): {self.path}")
        manifest = json.loads(partial.read_text())
        self.times: tuple[str, ...] = tuple(manifest["latent_times"])
        self.n_nodes = int(manifest["n_nodes"])
        self._steps = manifest["steps"]
        self._dtype = np.dtype(manifest["dtype"])
        self._arrays: dict[int, np.ndarray] = {}
        self._written = np.load(self.path / WRITTEN, mmap_mode="r+")

    def _array(self, layer: int) -> np.ndarray:
        if layer not in self._arrays:
            if not 0 <= layer < len(self._steps):
                raise RequestError(f"no layer {layer}; the archive has {len(self._steps)}")
            self._arrays[layer] = np.load(self.path / self._steps[layer]["file"], mmap_mode="r+")
        return self._arrays[layer]

    def put(self, time: str | int, layer: int, values: np.ndarray) -> None:
        """Write one layer at one time: ``(n_nodes, n_channels)``, any real dtype."""
        index = time if isinstance(time, int) else self._time_index(time)
        if not 0 <= index < len(self.times):
            raise RequestError(f"time index {index} is out of range for {len(self.times)} time(s)")
        array = self._array(layer)
        if values.shape != array.shape[1:]:
            raise RequestError(
                f"layer {layer} takes {array.shape[1:]} (nodes, channels), not {values.shape}"
            )
        if values.dtype.kind not in "biuf":
            raise RequestError(f"layer {layer} must hold real numeric values")
        try:
            with np.errstate(over="raise", invalid="raise"):
                array[index] = np.asarray(values, dtype=self._dtype)
        except FloatingPointError as exc:
            raise RequestError(
                f"layer {layer} overflows {self._dtype}; start the archive with a wider dtype"
            ) from exc
        self._written[index, layer] = 1

    def _time_index(self, label: str) -> int:
        if label not in self.times:
            raise RequestError(f"the archive has no time {label!r}")
        return self.times.index(label)

    def missing(self) -> list[int]:
        """Positions of the times at which some layer is still unwritten."""
        return [int(i) for i in np.flatnonzero(~np.asarray(self._written, dtype=bool).all(axis=1))]

    def flush(self) -> None:
        for array in self._arrays.values():
            array.flush()
        self._written.flush()


def finish_archive(path: str | Path) -> Path:
    """Make a filled archive readable: refused while any cell is unwritten."""
    out = Path(path)
    filler = ArchiveFiller(out)
    left = filler.missing()
    if left:
        shown = ", ".join(filler.times[i] for i in left[:3])
        raise RequestError(f"{len(left)} time(s) are not fully written yet, starting {shown}")
    filler.flush()
    del filler
    (out / WRITTEN).unlink()
    (out / PARTIAL).rename(out / MANIFEST)
    return out
