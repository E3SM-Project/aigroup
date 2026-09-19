"""Diagnostics, the command half: a thin client of the ``daig`` APIs.

This module must import on a base install (``xaig --help`` lists every command),
so numpy-backed modules are imported inside the commands that use them. What a
command cannot do it learns as a ``RequestError``, which the top-level command
prints as one line.
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
_basis_option = click.option(
    "--basis",
    "basis_path",
    type=click.Path(dir_okay=False),
    help="A basis file (from `latent pca` or `xaig taig sae`) whose features to use.",
)
_json_option = click.option(
    "--json", "as_json", is_flag=True, help="Emit settings, provenance and results."
)


def _open(source: str, adapter: str, mask_variable: str | None):
    from xaig.daig.latent import open_source

    options = {"mask_variable": mask_variable} if mask_variable else {}
    return open_source(source, adapter=adapter, **options)


def _basis(path: str | None):
    if path is None:
        return None
    from xaig.daig.latent import load_basis

    return load_basis(path)


def _time(text: str) -> str | int:
    from xaig.daig.latent import parse_time

    return parse_time(text)


def _region_options(command):
    for option in (
        click.option("--radius-km", type=float, default=1000.0, show_default=True),
        click.option("--lon", type=float, required=True),
        click.option("--lat", type=float, required=True),
    ):
        command = option(command)
    return command


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
        click.echo(f"\n{len(opened.field_names())} reference field(s); see `latent fields`")


@latent.command("region")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--time", "time", default="0", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to analyse.  [default: the last]")
@click.option("--rank-layer", type=int, help="Layer to rank channels at.  [default: the last]")
@_region_options
@click.option("--top", type=int, default=15, show_default=True, help="Channels to rank.")
@click.option("--pin", "pinned", type=int, multiple=True, help="Channel to list first; repeatable.")
@click.option("--centred", is_flag=True, help="Remove each channel's global mean first.")
@click.option("--reference", type=click.Choice(["nearest", "mean"]), default="nearest")
@click.option(
    "--pcs",
    "--features",
    "n_components",
    type=int,
    default=0,
    help="Features to map: principal components fitted in the region, or with --basis "
    "the features of it that respond most there.",
)
@_basis_option
@_json_option
def region_cmd(
    source, adapter, mask_variable, time, layer, as_json, lat, lon, radius_km, basis_path, **kw
):
    """Rank the channels that respond in a region; optionally map a decomposition."""
    from xaig.daig.latent import Region, analyse_region

    opened = _open(source, adapter, mask_variable)
    result = analyse_region(
        opened,
        time=_time(time),
        layer=opened.info().last_layer if layer is None else layer,
        region=Region(lat, lon, radius_km),
        basis=_basis(basis_path),
        **kw,
    )
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
    for feature in summary.get("features", []):
        loadings = "  ".join(
            f"{x['channel']}({x['loading']:+.2f})" for x in feature["top_loadings"]
        )
        if "peak_abs" in feature:  # a given basis: how strongly it responds here
            size = f"peak {feature['peak_abs']:.3g}"
        else:
            size = f"{100 * feature['explained_variance_ratio']:5.1f}%"
        click.echo(f"\n{feature['label']}  {size}  {loadings}")


@latent.command("series")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer to follow.  [default: the last]")
@_region_options
@click.option("--channel", "channels", type=int, multiple=True, help="Repeatable.")
@click.option("--feature", "features", type=int, multiple=True, help="With --basis; repeatable.")
@click.option("--centred", is_flag=True, help="Remove each channel's global mean at each time.")
@_basis_option
@_json_option
def series_cmd(
    source, adapter, mask_variable, layer, lat, lon, radius_km, channels, features, centred,
    basis_path, as_json,
):  # fmt: skip
    """Follow a region's mean response through every time of SOURCE."""
    from xaig.daig.latent import Region, region_series

    if not channels and not (basis_path and features):
        raise click.UsageError("name what to follow: --channel N, or --basis FILE --feature N")
    opened = _open(source, adapter, mask_variable)
    result = region_series(
        opened,
        layer=opened.info().last_layer if layer is None else layer,
        region=Region(lat, lon, radius_km),
        channels=channels or None,
        basis=_basis(basis_path),
        features=features or None,
        centred=centred,
    )
    if as_json:
        click.echo(json.dumps(result.summary(), indent=2))
        return
    hours = result.elapsed_seconds
    rows = [
        {
            "time": label,
            "hours": "" if hours is None else f"{hours[i] / 3600:g}",
            **{str(c): f"{v:.4g}" for c, v in zip(result.columns, result.values[i], strict=True)},
        }
        for i, label in enumerate(result.times)
    ]
    click.echo(_render.table(rows))


@latent.command("pca")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer to decompose.  [default: the last]")
@click.option("--components", type=int, default=16, show_default=True)
@click.option("--time", "times", multiple=True, help="Time label or position; repeatable.")
@click.option("--out", type=click.Path(dir_okay=False), required=True, help="Basis file to write.")
def pca_cmd(source, adapter, mask_variable, layer, components, times, out) -> None:
    """Fit a global, area-weighted PCA over every node and time; write a basis.

    The baseline any learned dictionary has to beat, and usable wherever one is:
    `latent region --basis`, `latent series --basis`, the web app.
    """
    from xaig.daig.latent import accumulate_moments, pca_from_moments, save_basis

    opened = _open(source, adapter, mask_variable)
    layer = opened.info().last_layer if layer is None else layer
    moments = accumulate_moments(opened, layer=layer, times=[_time(t) for t in times] or None)
    basis = pca_from_moments(moments, components)
    save_basis(out, basis, provenance=opened.info().provenance())
    explained = 100 * float(basis.explained_variance_ratio.sum())
    click.echo(
        f"wrote {out}: {components} component(s) of layer {layer} over "
        f"{len(moments.times)} time(s), {explained:.1f}% of the variance"
    )


@latent.command("diff")
@click.argument("control", type=click.Path())
@click.argument("experiment", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--time", "time", default="-1", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to compare.  [default: the last]")
@click.option("--top", type=int, default=15, show_default=True, help="Channels to rank.")
@click.option(
    "--growth", is_flag=True, help="Instead: how large the difference is, by layer and time."
)
@_json_option
def diff_cmd(control, experiment, adapter, mask_variable, time, layer, top, growth, as_json):
    """Set a perturbed or steered run against its CONTROL, node for node."""
    from xaig.daig.latent import difference, difference_growth

    a, b = _open(control, adapter, mask_variable), _open(experiment, adapter, mask_variable)
    if growth:
        grown = difference_growth(a, b, layers=None if layer is None else [layer])
        if as_json:
            click.echo(json.dumps(grown.summary(), indent=2))
            return
        names = [f"layer {x}" for x in grown.layers]
        rows = [
            {"time": label, **{n: f"{v:.3g}" for n, v in zip(names, row, strict=True)}}
            for label, row in zip(grown.times, grown.relative, strict=True)
        ]
        click.echo("RMS difference, relative to the control's own spread\n")
        click.echo(_render.table(rows))
        return
    result = difference(
        a, b, time=_time(time), layer=a.info().last_layer if layer is None else layer, top=top
    )
    summary = result.summary()
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    click.echo(f"layer {s['layer']}, time {s['time']}: total RMS {summary['total_rms']:.4g}\n")
    rows = [
        {"rank": i, "channel": r["channel"], "rms": f"{r['rms']:.4g}"}
        for i, r in enumerate(summary["ranking"], start=1)
    ]
    click.echo(_render.table(rows))


@latent.command("fields")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--field", help="Rank channels by correlation with this field.  [default: list them]")
@click.option("--time", "time", default="0", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to correlate.  [default: the last]")
@click.option("--top", type=int, default=15, show_default=True)
@_basis_option
@_json_option
def fields_cmd(source, adapter, mask_variable, field, time, layer, top, basis_path, as_json):
    """Which channels (or features) track a physical field kept beside the latents."""
    from xaig.daig.latent import ReferenceFields, rank_by_field

    opened = _open(source, adapter, mask_variable)
    if field is None:
        names = opened.field_names() if isinstance(opened, ReferenceFields) else ()
        click.echo("\n".join(names) if names else "no reference fields")
        return
    result = rank_by_field(
        opened,
        time=_time(time),
        layer=opened.info().last_layer if layer is None else layer,
        field=field,
        top=top,
        basis=_basis(basis_path),
    )
    summary = result.summary()
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    click.echo(f"{s['columns']} of layer {s['layer']} against {field} at {s['time']}\n")
    rows = [
        {"rank": i, s["columns"][:-1]: r["column"], "correlation": f"{r['correlation']:+.3f}"}
        for i, r in enumerate(summary["ranking"], start=1)
    ]
    click.echo(_render.table(rows))


__all__ = ["daig"]
