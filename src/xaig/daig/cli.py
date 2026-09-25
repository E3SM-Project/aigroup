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
_unverified_option = click.option(
    "--allow-unverified-basis",
    is_flag=True,
    help="Allow a basis with incomplete model/layer identity; known mismatches still fail.",
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
        click.echo(f"\n{len(opened.field_names())} reference field(s); see `latent fields`")


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
@_unverified_option
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
@_unverified_option
@_json_option
def series_cmd(
    source, adapter, mask_variable, layer, lat, lon, radius_km, channels, features, centred,
    basis_path, allow_unverified_basis, as_json,
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
        allow_unverified_basis=allow_unverified_basis,
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
    `latent region --basis`, `latent series --basis`.
    """
    from xaig.daig.latent import accumulate_moments, pca_from_moments, save_basis

    opened = _open(source, adapter, mask_variable)
    layer = opened.info().last_layer if layer is None else layer
    moments = accumulate_moments(opened, layer=layer, times=[_time(t) for t in times] or None)
    basis = pca_from_moments(moments, components)
    save_basis(out, basis)
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
@click.option(
    "--across-models",
    is_flag=True,
    help="Compare although the two declare different networks. A channel's index means "
    "nothing between checkpoints trained apart; this is for when it does (a fine-tune).",
)
@click.option(
    "--noise",
    type=click.Path(),
    help="With --growth: the control rerun with another seed. Adds the difference that "
    "noise alone makes, and the experiment's difference as a multiple of it.",
)
@_json_option
def diff_cmd(
    control, experiment, adapter, mask_variable, time, layer, top, growth, across_models, noise,
    as_json,
):  # fmt: skip
    """Set a perturbed or steered run against its CONTROL, node for node."""
    from xaig.daig.latent import difference, difference_growth

    if noise is not None and not growth:
        raise click.UsageError("--noise goes with --growth")
    a, b = _open(control, adapter, mask_variable), _open(experiment, adapter, mask_variable)
    if growth:
        grown = difference_growth(
            a,
            b,
            layers=None if layer is None else [layer],
            across_models=across_models,
            noise=None if noise is None else _open(noise, adapter, mask_variable),
        )
        if as_json:
            click.echo(json.dumps(grown.summary(), indent=2))
            return
        names = [f"layer {x}" for x in grown.layers]
        click.echo("RMS difference, relative to the control's own spread\n")
        click.echo(_layer_table(grown.times, names, grown.relative))
        if grown.noise_rms is not None:
            click.echo("\nRMS difference, as a multiple of what the noise run's makes\n")
            click.echo(_layer_table(grown.times, names, grown.signal_to_noise))
        return
    result = difference(
        a,
        b,
        time=_time(time),
        layer=a.info().last_layer if layer is None else layer,
        top=top,
        across_models=across_models,
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


def _layer_table(times, names, values) -> str:
    rows = [
        {"time": label, **{n: f"{v:.3g}" for n, v in zip(names, row, strict=True)}}
        for label, row in zip(times, values, strict=True)
    ]
    return _render.table(rows)


@latent.command("storyline")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--field", required=True, help="The physical field to follow.")
@click.option(
    "--layer", "layers", type=int, multiple=True, help="Layer to read; repeatable.  [default: all]"
)
@click.option(
    "--bases",
    help="Basis file per layer, as a template with {layer}: bases/sae_L{layer:02d}.npz. "
    "Layers without a file are read by their channels.",
)
@click.option("--time", "times", multiple=True, help="Time label or position; repeatable.")
@_unverified_option
@_json_option
def storyline_cmd(
    source, adapter, mask_variable, field, layers, bases, times, allow_unverified_basis, as_json
):
    """Where a physical field lives in the network, time by time: the best |r| per layer."""
    from pathlib import Path

    from xaig.daig.latent import field_storyline

    opened = _open(source, adapter, mask_variable)
    chosen = list(layers) or [x.index for x in opened.info().layers]
    found = {}
    if bases:
        for layer in chosen:
            path = Path(bases.format(layer=layer))
            if path.is_file():
                found[layer] = _basis(str(path))
        if not found:
            raise click.UsageError(f"no basis file matches {bases!r} for layers {chosen}")
    result = field_storyline(
        opened,
        field=field,
        layers=chosen,
        times=[_time(t) for t in times] or None,
        bases=found,
        allow_unverified_basis=allow_unverified_basis,
    )
    if as_json:
        click.echo(json.dumps(result.summary(), indent=2))
        return
    kind = "feature (basis)" if found else "channel"
    click.echo(f"best |r| of any {kind} with {field}, by layer and time\n")
    click.echo(_layer_table(result.times, [f"layer {x}" for x in result.layers], result.best))


@latent.command("hovmoller")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--lat-min", type=float, required=True)
@click.option("--lat-max", type=float, required=True)
@click.option("--field", help="A physical field kept beside the latents.")
@click.option("--layer", type=int, help="With --channel, or with --basis and --feature.")
@click.option("--channel", type=int)
@_basis_option
@click.option("--feature", type=int)
@click.option("--bins", type=int, default=12, show_default=True, help="Longitude bins printed.")
@click.option("--out", type=click.Path(dir_okay=False), help="Also write the full diagram (.npz).")
@_unverified_option
@_json_option
def hovmoller_cmd(
    source, adapter, mask_variable, lat_min, lat_max, field, layer, channel, basis_path, feature,
    bins, out, allow_unverified_basis, as_json,
):  # fmt: skip
    """A quantity along a latitude band, longitude against time: travelling things tilt."""
    import numpy as np

    from xaig.daig.latent import hovmoller

    opened = _open(source, adapter, mask_variable)
    result = hovmoller(
        opened,
        lat_min=lat_min,
        lat_max=lat_max,
        field=field,
        layer=layer,
        channel=channel,
        basis=_basis(basis_path),
        feature=feature,
        allow_unverified_basis=allow_unverified_basis,
    )
    if out:
        np.savez(out, values=result.values, lon=result.lon, times=np.array(result.times))
    if as_json:
        click.echo(json.dumps(result.summary(), indent=2))
        return
    lon = np.mod(result.lon, 360.0)
    edges = np.linspace(0.0, 360.0, max(bins, 1) + 1)
    which = np.clip(np.digitize(lon, edges) - 1, 0, len(edges) - 2)
    names = [f"{edges[k]:.0f}E" for k in range(len(edges) - 1)]
    with np.errstate(invalid="ignore"):
        binned = np.stack(
            [np.nanmean(result.values[:, which == k], axis=1) for k in range(len(edges) - 1)],
            axis=1,
        )
    what = field or (
        f"layer {layer} channel {channel}" if basis_path is None else f"feature {feature}"
    )
    click.echo(
        f"{what}, {lat_min} to {lat_max} degrees, mean per {360 / (len(edges) - 1):.0f} degrees\n"
    )
    click.echo(_layer_table(result.times, names, binned))


@latent.command("fields")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--field", help="Rank channels by correlation with this field.  [default: list them]")
@click.option("--time", "time", default="0", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to correlate.  [default: the last]")
@click.option("--top", type=int, default=15, show_default=True)
@_basis_option
@_unverified_option
@_json_option
def fields_cmd(
    source, adapter, mask_variable, field, time, layer, top, basis_path,
    allow_unverified_basis, as_json,
):  # fmt: skip
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
        allow_unverified_basis=allow_unverified_basis,
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


def _optional_region(lat, lon, radius_km):
    from xaig.daig.latent import Region

    if (lat is None) != (lon is None):
        raise click.UsageError("give both --lat and --lon, or neither")
    return None if lat is None else Region(lat, lon, radius_km)


_optional_region_options = [
    click.option("--lat", type=float, help="With --lon: read only a region around this point."),
    click.option("--lon", type=float),
    click.option("--radius-km", type=float, default=2000.0, show_default=True),
]
_threshold_option = click.option(
    "--threshold",
    type=float,
    default=0.0,
    show_default=True,
    help="Active means above this: firing, for a sparse basis; positive, for channels.",
)


def _with(options):
    def apply(command):
        for option in reversed(options):
            command = option(command)
        return command

    return apply


@latent.command("census")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--time", "time", default="0", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to survey.  [default: the last]")
@_basis_option
@click.option(
    "--by",
    type=click.Choice(["coverage", "mean", "strength", "peak"]),
    default="coverage",
    show_default=True,
    help="Order: share of the area active, mean, mean where active, or largest value.",
)
@click.option("--top", type=int, default=20, show_default=True)
@_with(_optional_region_options)
@_threshold_option
@_unverified_option
@_json_option
def census_cmd(
    source, adapter, mask_variable, time, layer, basis_path, by, top, lat, lon, radius_km,
    threshold, allow_unverified_basis, as_json,
):  # fmt: skip
    """Every channel (or feature) of a layer: how much of the world it is active over."""
    from xaig.daig.latent import feature_census

    opened = _open(source, adapter, mask_variable)
    result = feature_census(
        opened,
        time=_time(time),
        layer=opened.info().last_layer if layer is None else layer,
        basis=_basis(basis_path),
        region=_optional_region(lat, lon, radius_km),
        threshold=threshold,
        allow_unverified_basis=allow_unverified_basis,
    )
    summary = result.summary(by=by, top=top)
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    click.echo(f"{s['columns']} of layer {s['layer']} at {s['time']}, by {by}\n")
    rows = [
        {
            s["columns"][:-1]: c["column"],
            "coverage": f"{c['coverage']:.3f}",
            "mean": f"{c['mean']:.3g}",
            "strength": f"{c['strength']:.3g}",
            "peak": f"{c['peak']:.3g}",
            "at": f"{c['peak_lat']:.0f}, {c['peak_lon']:.0f}",
        }
        for c in summary["columns"]
    ]
    click.echo(_render.table(rows))


@latent.command("profile")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer to read.  [default: the last]")
@click.option("--channel", type=int, help="The channel to profile.")
@_basis_option
@click.option("--feature", type=int, help="With --basis: the feature to profile.")
@click.option("--time", "times", multiple=True, help="Time label or position; repeatable.")
@click.option("--field", "fields", multiple=True, help="Repeatable.  [default: every field]")
@_with(_optional_region_options)
@_threshold_option
@_unverified_option
@_json_option
def profile_cmd(
    source, adapter, mask_variable, layer, channel, basis_path, feature, times, fields, lat, lon,
    radius_km, threshold, allow_unverified_basis, as_json,
):  # fmt: skip
    """Every physical field where one channel (or feature) is active, against where it is not."""
    from xaig.daig.latent import feature_profile

    if (channel is None) == (basis_path is None or feature is None):
        raise click.UsageError("name one: --channel N, or --basis FILE --feature N")
    opened = _open(source, adapter, mask_variable)
    result = feature_profile(
        opened,
        layer=opened.info().last_layer if layer is None else layer,
        column=feature if channel is None else channel,
        basis=_basis(basis_path) if channel is None else None,
        times=[_time(t) for t in times] or None,
        fields=fields or None,
        region=_optional_region(lat, lon, radius_km),
        threshold=threshold,
        allow_unverified_basis=allow_unverified_basis,
    )
    if as_json:
        click.echo(json.dumps(result.summary(), indent=2))
        return
    s = result.settings
    click.echo(
        f"{s['columns'][:-1]} {s['column']} of layer {s['layer']}: active over "
        f"{result.coverage:.1%} of the area and {len(result.times)} time(s)\n"
    )
    rows = [
        {"field": n, "effect": f"{e:+.2f}", "active": f"{a:.4g}", "inactive": f"{i:.4g}"}
        for n, e, a, i in zip(
            result.fields, result.effect, result.active_mean, result.inactive_mean, strict=True
        )
    ]
    click.echo(_render.table(rows))


__all__ = ["daig"]
