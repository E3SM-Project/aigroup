"""Record an ACE atmosphere's latents into a latent archive, one step at a time.

Rather than rolling the model forward for decades to reach the times wanted, each
kept time is taken from the reference data and the model is stepped once from it,
as validation does. The latents are then the network's representation of the real
state at that time, never of its own drift, and every kept time costs one forward
step; any number of processes can share the work, each on its own times.

What is recorded is the residual stream of the SFNO: the input to each block,
before its first layer norm, and the output of the last block, which is what the
decoder reads. With eight blocks, nine layers, each in a directory of its own.

Three stages, so that the one that takes the time can be spread over GPUs::

    python -m xaig.adapters.ace_export start --checkpoint CKPT --data DATA.yaml --out DIR
    srun -n 4 python -m xaig.adapters.ace_export fill --out DIR
    python -m xaig.adapters.ace_export finish --out DIR

``start`` records the checkpoint, the data's configuration and the noise seed in
the manifest, and ``fill`` takes them from there, so every process steps the same
network from the same data. ``fill`` does what is left undone, so a job that runs
out of time is resumed by running it again. It needs ACE's environment (``fme``,
torch); this module is the only place in xaig that knows ACE exists, and the
archive it writes needs numpy alone to read.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time as clock
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra

try:
    import numpy as np
    import torch
    from torch import nn
except ImportError as exc:
    raise missing_extra("torch", "taig") from exc

from xaig.adapters.latent_archive import ArchiveFiller, finish_archive, start_archive
from xaig.daig.grid import Grid

LABEL_FORMAT = "%Y-%m-%dT%H:%M:%S"
log = logging.getLogger("xaig.ace_export")


# -- the parts that know nothing of ACE ------------------------------------------


def keep_every(n_times: int, first: int, stop: int, every: int) -> list[int]:
    """Positions ``first, first + every, ...`` below ``stop`` (and ``n_times``).

    An ``every`` that shares no factor with the number of steps in a day moves the
    kept times round the diurnal cycle: 29 six-hourly steps land on each of the
    four hours of the day in turn."""
    if every < 1:
        raise RequestError("every must be at least 1")
    return list(range(first, min(stop, n_times), every))


def find_blocks(module: nn.Module) -> tuple[nn.Module, nn.ModuleList]:
    """The residual blocks of a network, and the module holding them: the first
    ``blocks`` list found, however many wrappers the network sits inside."""
    for sub in module.modules():
        blocks = getattr(sub, "blocks", None)
        if isinstance(blocks, nn.ModuleList) and len(blocks) > 0:
            return sub, blocks
    raise RequestError(f"no list of blocks in {type(module).__name__}")


def layer_names(n_blocks: int) -> list[tuple[str, str]]:
    """``(label, directory)`` of each recorded layer: every block's input, then
    the last block's output."""
    names = [
        (f"block {i} input (residual stream)", f"block_{i:02d}_input") for i in range(n_blocks)
    ]
    last = n_blocks - 1
    return [*names, (f"block {last} output (decoder input)", f"block_{last:02d}_output")]


class BlockRecorder:
    """Keeps, for each forward call of a network, the input of each of its blocks
    and the output of the last, as ``(batch, channels, *grid)`` tensors.

    Used as a context manager, so the hooks never outlive the recording. A second
    forward call before ``take()`` is refused: a stepper that calls its network
    more than once a step would otherwise hand over the wrong one silently."""

    def __init__(self, blocks: nn.ModuleList) -> None:
        self.blocks = blocks
        self._handles: list[Any] = []
        self._seen: dict[int, torch.Tensor] = {}

    def __enter__(self) -> BlockRecorder:
        last = len(self.blocks)
        for i, block in enumerate(self.blocks):
            self._handles.append(block.register_forward_pre_hook(self._keeper(i, "in")))
        self._handles.append(self.blocks[-1].register_forward_hook(self._keeper(last, "out")))
        return self

    def __exit__(self, *exc: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _keeper(self, index: int, which: str):
        def keep(module, args, output=None):
            if index in self._seen:
                raise RuntimeError(
                    f"layer {index} was recorded twice before being taken: the network "
                    "is called more than once per step, and which call to keep is not known"
                )
            tensor = args[0] if which == "in" else output
            self._seen[index] = tensor.detach()

        return keep

    def take(self) -> list[torch.Tensor]:
        """Every layer of the last forward call, in order, and forget them."""
        n_layers = len(self.blocks) + 1
        if len(self._seen) != n_layers:
            raise RuntimeError(f"recorded {len(self._seen)} of {n_layers} layers")
        taken = [self._seen[i] for i in range(n_layers)]
        self._seen.clear()
        return taken


def to_nodes(tensor: torch.Tensor, shape: tuple[int, int]) -> np.ndarray:
    """``(channels, n_lat, n_lon)`` on the GPU -> ``(n_nodes, channels)`` float32,
    nodes in the grid's row-major order. Any padding the network carries beyond
    the grid is cut off."""
    cut = tensor[:, : shape[0], : shape[1]]
    return cut.permute(1, 2, 0).reshape(shape[0] * shape[1], -1).float().cpu().numpy()


# -- the parts that know ACE -----------------------------------------------------


def _load(checkpoint: str, data: dict[str, Any]):
    """The stepper, and the reference data with what it needs for one step."""
    os.environ.setdefault("FME_DISTRIBUTED_BACKEND", "none")  # one process per GPU, alone
    import dacite
    from fme.ace.stepper.single_module import load_stepper
    from fme.core.dataset.xarray import XarrayDataConfig, XarrayDataset

    stepper = load_stepper(checkpoint)
    stepper.set_eval()
    config = dacite.from_dict(XarrayDataConfig, data, config=dacite.Config(strict=True))
    requirements = stepper._config.get_evaluation_window_data_requirements(n_forward_steps=1)
    dataset = XarrayDataset(
        config,
        requirements.names,
        requirements.n_timesteps_schedule,
        allow_missing_variables=requirements.allow_missing_variables,
    )
    return stepper, dataset, requirements


def _grid(dataset) -> Grid:
    coords = dataset.properties.horizontal_coordinates
    lat = np.asarray(coords.lat.cpu(), dtype=np.float64)
    lon = np.asarray(coords.lon.cpu(), dtype=np.float64)
    lat2, lon2 = np.meshgrid(lat, lon, indexing="ij")
    return Grid(lat=lat2.ravel(), lon=lon2.ravel(), shape=(lat.size, lon.size))


def _labels(dataset) -> list[str]:
    return [t.strftime(LABEL_FORMAT) for t in dataset.all_times]


def _positions(dataset, start: str, stop: str, every: int) -> list[int]:
    labels = _labels(dataset)
    if start not in labels:
        raise RequestError(f"the data has no time {start!r}")
    first = labels.index(start)
    end = next((i for i, t in enumerate(labels) if t >= stop), len(labels))
    # the step from the last kept time needs the time after it as forcing
    return keep_every(len(labels) - 1, first, end, every)


def _collate(items):
    return [k for k, _ in items], [sample for _, sample in items]


class _Windows(torch.utils.data.Dataset):
    """Two-time windows (the kept time and the next) at the given positions."""

    def __init__(self, dataset, positions: Sequence[int]) -> None:
        self.dataset, self.positions = dataset, list(positions)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, k: int):
        position = self.positions[k]
        return k, self.dataset.get_sample_by_time_slice(slice(position, position + 2))


def _batches(dataset, positions, batch_size: int, workers: int) -> Iterator[tuple[list[int], Any]]:
    loader = torch.utils.data.DataLoader(
        _Windows(dataset, positions),
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=_collate,
        prefetch_factor=4 if workers else None,
    )
    yield from loader


def start(args: argparse.Namespace) -> None:
    import yaml

    data = yaml.safe_load(Path(args.data).read_text())
    stepper, dataset, _ = _load(args.checkpoint, data)
    positions = _positions(dataset, args.start, args.stop, args.every)
    labels = _labels(dataset)
    network, blocks = find_blocks(stepper.modules[0])
    width = int(getattr(network, "embed_dim", args.channels))
    names = layer_names(len(blocks))
    timestep = dataset.properties.timestep
    start_archive(
        args.out,
        grid=_grid(dataset),
        times=[labels[p] for p in positions],
        layers=[(label, width) for label, _ in names],
        directories=[directory for _, directory in names],
        model=args.model,
        component="atmosphere",
        checkpoint=str(Path(args.checkpoint).resolve()),
        calendar=dataset.all_times.calendar,
        timestep_seconds=None if timestep is None else int(timestep.total_seconds()),
        experiment={
            "method": "one step from the reference state at each kept time",
            "data_config": data,
            "start": args.start,
            "stop": args.stop,
            "every_n_steps": args.every,
            "noise_seed": args.seed,
            "time_label": "the state the step was taken from",
        },
        dtype="float32",
    )
    print(f"started {args.out}: {len(positions)} time(s) x {len(names)} layer(s) of {width}")


def fill(args: argparse.Namespace) -> None:
    rank = int(os.environ.get("SLURM_PROCID", args.rank))
    world = int(os.environ.get("SLURM_NTASKS", args.world))
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("SLURM_LOCALID", 0)) % torch.cuda.device_count())
    from fme.ace.data_loading.batch_data import BatchData
    from fme.core.device import get_device

    device = get_device()
    manifest = json.loads((Path(args.out) / "manifest.partial.json").read_text())
    experiment = manifest["experiment"]
    seed = int(experiment["noise_seed"])
    stepper, dataset, requirements = _load(manifest["checkpoint"], experiment["data_config"])
    filler = ArchiveFiller(args.out)
    labels = _labels(dataset)
    wanted = {label: i for i, label in enumerate(filler.times)}
    todo = filler.missing()[rank::world]
    positions = [labels.index(filler.times[i]) for i in todo]
    shape = tuple(manifest["grid_shape"])
    _, blocks = find_blocks(stepper.modules[0])
    dims = list(dataset.properties.horizontal_coordinates.dims)
    log.warning("rank %d/%d: %d time(s) to do on %s", rank, world, len(todo), device)

    writer = ThreadPoolExecutor(max_workers=2)
    pending: list[Future] = []
    started, done = clock.time(), 0

    def write(index: int, arrays: list[np.ndarray]) -> None:
        for layer, values in enumerate(arrays):
            filler.put(index, layer, values)

    with torch.no_grad(), BlockRecorder(blocks) as recorder:
        for n_batch, (ks, samples) in enumerate(
            _batches(dataset, positions, args.batch_size, args.workers)
        ):
            batch = BatchData.from_sample_tuples(
                samples,
                horizontal_dims=dims,
                allow_missing_variables=requirements.allow_missing_variables,
            ).to_device()
            first = todo[ks[0]]
            torch.manual_seed(seed * 1_000_003 + first)  # the same noise, however split
            initial = batch.get_start(stepper.prognostic_names, stepper.n_ic_timesteps)
            stepper.predict(initial, batch, compute_derived_variables=False)
            layers = recorder.take()
            for j, k in enumerate(ks):
                index = todo[k]
                label = batch.time.values[j, 0].strftime(LABEL_FORMAT)
                if wanted.get(label) != index:
                    raise RuntimeError(f"batch time {label} is not archive time {index}")
                arrays = [to_nodes(tensor[j], shape) for tensor in layers]
                while len(pending) >= 4:  # bound what waits in memory to be written
                    pending.pop(0).result()
                pending.append(writer.submit(write, index, arrays))
            done += len(ks)
            if n_batch % 25 == 0:
                rate = done / (clock.time() - started)
                log.warning(
                    "rank %d: %d/%d time(s), %.2f/s, ~%.0f min left",
                    rank, done, len(todo), rate, (len(todo) - done) / max(rate, 1e-9) / 60,
                )  # fmt: skip
            if n_batch % 100 == 99:
                for future in pending:
                    future.result()
                pending.clear()
                filler.flush()
    for future in pending:
        future.result()
    writer.shutdown()
    filler.flush()
    log.warning("rank %d: done, %d time(s) in %.1f min", rank, done, (clock.time() - started) / 60)


def finish(args: argparse.Namespace) -> None:
    finish_archive(args.out)
    print(f"finished {args.out}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m xaig.adapters.ace_export", description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    for name in ("start", "fill", "finish"):
        stage = stages.add_parser(name)
        stage.add_argument("--out", required=True, help="the archive directory")
    first = stages.choices["start"]
    first.add_argument("--checkpoint", required=True)
    first.add_argument("--data", required=True, help="an XarrayDataConfig, as YAML")
    first.add_argument("--seed", type=int, default=0, help="noise seed")
    first.add_argument("--start", default="1950-01-01T06:00:00")
    first.add_argument("--stop", default="1990-01-01T06:00:00", help="exclusive")
    first.add_argument("--every", type=int, default=29, help="keep one step in this many")
    first.add_argument("--model", default=None, help="a name for the network")
    first.add_argument("--channels", type=int, default=384, help="if it cannot be read")
    work = stages.choices["fill"]
    work.add_argument("--batch-size", type=int, default=4)
    work.add_argument("--workers", type=int, default=6, help="data-loading processes")
    work.add_argument("--rank", type=int, default=0, help="unless SLURM_PROCID says")
    work.add_argument("--world", type=int, default=1, help="unless SLURM_NTASKS says")
    args = parser.parse_args(argv)
    logging.basicConfig(format="%(asctime)s %(message)s", level=logging.WARNING)
    {"start": start, "fill": fill, "finish": finish}[args.stage](args)


if __name__ == "__main__":
    main()
