"""Diagnostics, the command half: a thin client of the ``daig`` APIs.

This module must import on a base install (``xaig --help`` lists every command),
so numpy-backed modules are imported inside the commands that use them. What a
command cannot do it learns as a ``RequestError``, which the top-level command
prints as one line.
"""

from __future__ import annotations

import click

from xaig import _render

_adapter_option = click.option(
    "--adapter", default="latent-archive", show_default=True, help="How SOURCE is read."
)
_mask_option = click.option(
    "--mask-variable",
    help="Reference-file variable that is missing where nodes mean nothing (e.g. sst).",
)


def _open(source: str, adapter: str, mask_variable: str | None):
    from xaig.daig.latent import open_source

    options = {"mask_variable": mask_variable} if mask_variable else {}
    return open_source(source, adapter=adapter, **options)


@click.group(name="daig")
def daig() -> None:
    """Diagnose emulators: what they hold inside."""


@daig.group("latent")
def latent() -> None:
    """Explore activations recorded from inside a model."""


@latent.command("info")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
def info_cmd(source, adapter, mask_variable) -> None:
    """Describe what SOURCE holds, without loading it."""
    from xaig.daig.latent import ReferenceFields

    opened = _open(source, adapter, mask_variable)
    info, grid = opened.info(), opened.grid()
    shape = "x".join(str(n) for n in grid.shape) if grid.shape else "mesh"
    provenance = info.provenance()
    experiment = provenance.pop("experiment", {})
    provenance.pop("options", None)
    click.echo(
        _render.pairs(
            {
                **provenance,
                "calendar": info.calendar,
                "timestep_s": info.timestep_seconds,
                "grid": f"{shape}, {grid.n_nodes} nodes, {int(grid.valid.sum())} valid",
                "times": f"{len(info.times)}: {info.times[0]} .. {info.times[-1]}",
                **{f"experiment.{key}": value for key, value in experiment.items()},
            }
        )
    )
    click.echo("\nlayers")
    rows = [{"layer": x.index, "channels": x.n_channels, "label": x.label} for x in info.layers]
    click.echo(_render.table(rows))
    if info.off_grid_layers:
        click.echo(f"\n{len(info.off_grid_layers)} more layer(s) on coarser grids, not loadable")
    if isinstance(opened, ReferenceFields) and opened.field_names():
        click.echo(f"\n{len(opened.field_names())} reference field(s)")


@latent.command("toy")
@click.argument("out", type=click.Path(file_okay=False))
@click.option("--steps", type=int, default=12, show_default=True, help="Steps to roll forward.")
@click.option("--keep", default="1-4,9-12", show_default=True, help="Steps whose latents are kept.")
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--steer",
    metavar="LAYER:CHANNEL:AMOUNT",
    help="Add AMOUNT to one channel of one layer at every step; a twin without it is its control.",
)
@click.option("--overwrite", is_flag=True, help="Replace OUT if it holds anything.")
@_adapter_option
def toy_cmd(out, steps, keep, seed, steer, overwrite, adapter) -> None:
    """Run a toy emulator and write its latents to OUT, with nothing but numpy.

    An MLP with a residual stream on a small Gaussian grid: no checkpoint, no
    data, a few seconds. What comes out is read like any other archive.
    """
    from xaig.core.errors import RequestError
    from xaig.daig.latent.toy import parse_steps, write_toy

    pushed = None
    if steer:
        try:
            layer, channel, amount = steer.split(":")
            pushed = (int(layer), int(channel), float(amount))
        except ValueError as exc:
            raise RequestError(f"--steer is LAYER:CHANNEL:AMOUNT, not {steer!r}") from exc
    path = write_toy(
        out,
        adapter=adapter,
        overwrite=overwrite,
        n_steps=steps,
        keep=parse_steps(keep),
        seed=seed,
        steer=pushed,
    )
    click.echo(f"wrote {path}; try `xaig daig latent info {path} --mask-variable sst`")


__all__ = ["daig"]
