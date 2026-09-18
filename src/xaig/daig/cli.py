"""Diagnostics, the command half: a thin client of the ``daig`` APIs.

This module must import on a base install (``xaig --help`` lists every command),
so numpy-backed modules are imported inside the commands that use them.
"""

from __future__ import annotations

import json

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
    """Diagnose emulators: their outputs and their internals."""


@daig.group("latent")
def latent() -> None:
    """Explore activations recorded from inside a model."""


@latent.command("info")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
def info_cmd(source, adapter, mask_variable) -> None:
    """Describe what SOURCE holds, without loading it."""
    opened = _open(source, adapter, mask_variable)
    info, grid = opened.info(), opened.grid()
    shape = "x".join(str(n) for n in grid.shape) if grid.shape else "mesh"
    click.echo(
        _render.pairs(
            {
                **info.provenance(),
                "calendar": info.calendar,
                "timestep_s": info.timestep_seconds,
                "grid": f"{shape}, {grid.n_nodes} nodes, {int(grid.valid.sum())} valid",
                "times": f"{len(info.times)}: {info.times[0]} .. {info.times[-1]}",
            }
        )
    )
    click.echo("\nlayers")
    rows = [{"layer": x.index, "channels": x.n_channels, "label": x.label} for x in info.layers]
    click.echo(_render.table(rows))
    if info.off_grid_layers:
        click.echo(f"\n{len(info.off_grid_layers)} more layer(s) on coarser grids, not loadable")


@latent.command("region")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--time", "time", default="0", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to analyse.  [default: the last]")
@click.option("--rank-layer", type=int, help="Layer to rank channels at.  [default: the last]")
@click.option("--lat", type=float, required=True)
@click.option("--lon", type=float, required=True)
@click.option("--radius-km", type=float, default=1000.0, show_default=True)
@click.option("--top", type=int, default=15, show_default=True, help="Channels to rank.")
@click.option("--pin", "pinned", type=int, multiple=True, help="Channel to list first; repeatable.")
@click.option("--centred", is_flag=True, help="Remove each channel's global mean first.")
@click.option("--reference", type=click.Choice(["nearest", "mean"]), default="nearest")
@click.option("--pcs", "n_components", type=int, default=0, help="Principal components to fit.")
@click.option("--json", "as_json", is_flag=True, help="Emit settings, provenance and results.")
def region_cmd(source, adapter, mask_variable, time, layer, as_json, lat, lon, radius_km, **kw):
    """Rank the channels that respond in a region; optionally fit a PCA there."""
    from xaig.daig.latent import Region, analyse_region

    opened = _open(source, adapter, mask_variable)
    try:
        result = analyse_region(
            opened,
            time=int(time) if time.lstrip("-").isdigit() else time,
            layer=opened.info().last_layer if layer is None else layer,
            region=Region(lat, lon, radius_km),
            **kw,
        )
    except (KeyError, ValueError) as exc:
        raise click.ClickException(str(exc.args[0])) from exc
    summary = result.summary()
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    click.echo(f"{summary['n_region_nodes']} node(s) at layer {s['layer']}, time {s['time']}\n")
    rows = [
        {"rank": i, "channel": r["channel"], "peak_abs": f"{r['peak_abs']:.4g}"}
        for i, r in enumerate(summary["ranking"], start=1)
    ]
    click.echo(_render.table(rows))
    for pc in summary.get("pca", []):
        loadings = "  ".join(f"{x['channel']}({x['loading']:+.2f})" for x in pc["top_loadings"])
        click.echo(
            f"\nPC{pc['component']}  {100 * pc['explained_variance_ratio']:5.1f}%  {loadings}"
        )


__all__ = ["daig"]
