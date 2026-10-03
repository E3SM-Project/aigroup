"""``xaig latents``, the command half: a thin client of the ``xaig.latents`` APIs.

This module must import on a base install (``xaig --help`` lists every command),
so numpy-backed modules are imported inside the commands that use them. What a
command cannot do it learns as a ``RequestError``, which the top-level command
prints as one line.
"""

from __future__ import annotations

import functools
import json
import shlex
from pathlib import Path

import click

from xaig import _render

_adapter_option = click.option(
    "--adapter", default="latent-archive", show_default=True, help="How SOURCE is read."
)


def _remember(key: str):
    """An option no command takes as a parameter: it is kept on the click context, where
    ``open_for_cli`` and ``_basis`` find it. Pinning a source or a basis is a way of
    reading one, so it need not widen the signature of every command."""

    def callback(ctx, param, value):
        if value is not None:
            ctx.meta[f"xaig.{key}"] = value
        return value

    return callback


def _remembered(key: str):
    return click.get_current_context().meta.get(f"xaig.{key}")


_revision_click_option = click.option(
    "--revision",
    expose_value=False,
    callback=_remember("revision"),
    help="For hf:// sources: the branch, tag or commit to read. Resolved to a commit when "
    "opened (a full 40-character commit is taken as it is), and recorded; applies to every "
    "hf:// source of the command.",
)


def revision_option(command):
    """``--revision``, refused by a command none of whose sources is on the hub: it would
    be silently ignored, and a reproduction that believed it was pinned would not be."""

    @functools.wraps(command)
    def checked(*args, **kwargs):
        if _remembered("revision") and not any(
            isinstance(value, str) and value.startswith("hf://") for value in kwargs.values()
        ):
            from xaig.core.errors import RequestError

            raise RequestError(
                "--revision applies to hf:// sources, and none of this command's sources is one"
            )
        return command(*args, **kwargs)

    return _revision_click_option(checked)


_mask_click_option = click.option(
    "--mask-variable",
    help="Reference-file variable that is missing where nodes mean nothing (e.g. sst).",
)
_basis_click_option = click.option(
    "--basis",
    "basis_path",
    type=click.Path(dir_okay=False),
    help="A basis file (from `latents pca` or `xaig nn sae`) whose features to use.",
)
_basis_sha_option = click.option(
    "--basis-sha256",
    expose_value=False,
    callback=_remember("basis_sha256"),
    metavar="HASH",
    help="Refuse the --basis file unless its content has this sha256 (reproduction commands "
    "give it).",
)


_record_option = click.option(
    "--record", "record_path", type=click.Path(dir_okay=False),
    help="Also write a compact experiment record (JSON): provenance, settings, results and "
    "the command that reproduces it, which `xaig app` opens.",
)  # fmt: skip


def _mask_option(command):
    """``--mask-variable``, and ``--revision`` beside it: every command that opens a source."""
    return revision_option(_mask_click_option(command))


def _basis_option(command):
    return _basis_sha_option(_basis_click_option(command))


_unverified_option = click.option(
    "--allow-unverified-basis",
    is_flag=True,
    help="Allow a basis with incomplete model/layer identity; known mismatches still fail.",
)
_lead_option = click.option(
    "--lead",
    type=int,
    default=0,
    show_default=True,
    help="Set the latents against the field this many times later: 1 for what that forward "
    "pass produced (an output), 0 for what it read (an input).",
)
_json_option = click.option(
    "--json", "as_json", is_flag=True, help="Emit settings, provenance and results."
)


def open_for_cli(source: str, adapter: str, mask_variable: str | None):
    """Open SOURCE as the command line asks, ``--revision`` included."""
    from xaig.latents import open_source

    options = {"mask_variable": mask_variable} if mask_variable else {}
    revision = _remembered("revision")
    if revision and source.startswith("hf://"):  # a local source has no revision to pin
        options["revision"] = revision
    return open_source(source, adapter=adapter, **options)


def _basis(path: str | None):
    sha256 = _remembered("basis_sha256")
    if path is None:
        if sha256 is not None:
            from xaig.core.errors import RequestError

            raise RequestError("--basis-sha256 pins the --basis file, and no --basis is given")
        return None
    from xaig.latents import load_basis

    return load_basis(path, sha256=sha256)


def _time(text: str) -> str | int:
    from xaig.latents import parse_time

    return parse_time(text)


def _region_options(command):
    for option in (
        click.option("--radius-km", type=float, default=1000.0, show_default=True),
        click.option("--lon", type=float, required=True),
        click.option("--lat", type=float, required=True),
    ):
        command = option(command)
    return command


@click.group(name="latents")
def latents() -> None:
    """Explore activations recorded from inside a model."""


@latents.command("info")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
def info_cmd(source, adapter, mask_variable) -> None:
    """Describe what SOURCE holds, without loading it."""
    from xaig.latents import ReferenceFields

    opened = open_for_cli(source, adapter, mask_variable)
    info, grid = opened.info(), opened.grid()
    shape = "x".join(str(n) for n in grid.shape) if grid.shape else "mesh"
    provenance = info.provenance()
    experiment = provenance.pop("experiment", {})
    provenance.pop("options", None)
    revision = provenance.pop("revision", {})
    if revision:
        provenance["requested_revision"] = revision["requested"] or "(default)"
        provenance["commit"] = revision["commit"]
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
        click.echo(f"\n{len(opened.field_names())} reference field(s); see `latents fields`")


@latents.command("toy")
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
    from xaig.latents.toy import parse_steps, write_toy

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
    click.echo(f"wrote {path}; try `xaig latents info {path} --mask-variable sst`")


@latents.command("region")
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
    from xaig.latents import Region, analyse_region

    opened = open_for_cli(source, adapter, mask_variable)
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


@latents.command("series")
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
    from xaig.latents import Region, region_series

    if not channels and not (basis_path and features):
        raise click.UsageError("name what to follow: --channel N, or --basis FILE --feature N")
    opened = open_for_cli(source, adapter, mask_variable)
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


@latents.command("pca")
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
    `latents region --basis`, `latents series --basis`.
    """
    from xaig.latents import accumulate_moments, basis_hash, pca_from_moments, save_basis

    opened = open_for_cli(source, adapter, mask_variable)
    layer = opened.info().last_layer if layer is None else layer
    moments = accumulate_moments(opened, layer=layer, times=[_time(t) for t in times] or None)
    basis = pca_from_moments(moments, components)
    save_basis(out, basis)
    explained = 100 * float(basis.explained_variance_ratio.sum())
    click.echo(
        f"wrote {out}: {components} component(s) of layer {layer} over "
        f"{len(moments.times)} time(s), {explained:.1f}% of the variance; "
        f"--basis-sha256 {basis_hash(basis)}"
    )


@latents.command("diff")
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
    "--allow-unverified-sources",
    is_flag=True,
    help="Compare although a model, component, checkpoint or network layer is undeclared by "
    "either run; known mismatches still fail.",
)
@click.option(
    "--noise",
    type=click.Path(),
    help="With --growth: the control rerun with another seed. Adds the difference that "
    "noise alone makes, and the experiment's difference as a multiple of it.",
)
@_json_option
def diff_cmd(
    control, experiment, adapter, mask_variable, time, layer, top, growth, across_models,
    allow_unverified_sources, noise, as_json,
):  # fmt: skip
    """Set a perturbed or steered run against its CONTROL, node for node."""
    from xaig.latents import difference, difference_growth

    if noise is not None and not growth:
        raise click.UsageError("--noise goes with --growth")
    a = open_for_cli(control, adapter, mask_variable)
    b = open_for_cli(experiment, adapter, mask_variable)
    if growth:
        grown = difference_growth(
            a,
            b,
            layers=None if layer is None else [layer],
            across_models=across_models,
            allow_unverified=allow_unverified_sources,
            noise=None if noise is None else open_for_cli(noise, adapter, mask_variable),
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
        allow_unverified=allow_unverified_sources,
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


@latents.command("storyline")
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
@_lead_option
@_unverified_option
@_json_option
def storyline_cmd(
    source, adapter, mask_variable, field, layers, bases, times, lead, allow_unverified_basis,
    as_json,
):  # fmt: skip
    """Where a physical field lives in the network, time by time: the best |r| per layer."""
    from pathlib import Path

    from xaig.latents import field_storyline

    opened = open_for_cli(source, adapter, mask_variable)
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
        lead=lead,
    )
    if as_json:
        click.echo(json.dumps(result.summary(), indent=2))
        return
    kind = "feature (basis)" if found else "channel"
    click.echo(f"best |r| of any {kind} with {field}, by layer and time\n")
    click.echo(_layer_table(result.times, [f"layer {x}" for x in result.layers], result.best))


@latents.command("hovmoller")
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

    from xaig.latents import hovmoller

    opened = open_for_cli(source, adapter, mask_variable)
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


@latents.command("fields")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--field", help="Rank channels by correlation with this field.  [default: list them]")
@click.option("--time", "time", default="0", show_default=True, help="Time label, or position.")
@click.option("--layer", type=int, help="Layer to correlate.  [default: the last]")
@click.option("--top", type=int, default=15, show_default=True)
@_basis_option
@_lead_option
@_unverified_option
@_json_option
def fields_cmd(
    source, adapter, mask_variable, field, time, layer, top, basis_path, lead,
    allow_unverified_basis, as_json,
):  # fmt: skip
    """Which channels (or features) track a physical field kept beside the latents."""
    from xaig.latents import ReferenceFields, rank_by_field

    opened = open_for_cli(source, adapter, mask_variable)
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
        lead=lead,
    )
    summary = result.summary()
    if as_json:
        click.echo(json.dumps(summary, indent=2))
        return
    s = summary["settings"]
    later = f", {lead:+d} time(s) later" if lead else ""
    click.echo(f"{s['columns']} of layer {s['layer']} against {field} at {s['time']}{later}\n")
    rows = [
        {"rank": i, s["columns"][:-1]: r["column"], "correlation": f"{r['correlation']:+.3f}"}
        for i, r in enumerate(summary["ranking"], start=1)
    ]
    click.echo(_render.table(rows))


def _optional_region(lat, lon, radius_km):
    from xaig.latents import Region

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


@latents.command("census")
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
    from xaig.latents import feature_census

    opened = open_for_cli(source, adapter, mask_variable)
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


@latents.command("profile")
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
@_lead_option
@_unverified_option
@_json_option
def profile_cmd(
    source, adapter, mask_variable, layer, channel, basis_path, feature, times, fields, lat, lon,
    radius_km, threshold, lead, allow_unverified_basis, as_json,
):  # fmt: skip
    """Every physical field where one channel (or feature) is active, against where it is not."""
    from xaig.latents import feature_profile

    by_channel = channel is not None and basis_path is None and feature is None
    by_feature = channel is None and basis_path is not None and feature is not None
    if not (by_channel or by_feature):
        raise click.UsageError("name one: --channel N, or --basis FILE --feature N")
    opened = open_for_cli(source, adapter, mask_variable)
    result = feature_profile(
        opened,
        layer=opened.info().last_layer if layer is None else layer,
        column=feature if channel is None else channel,
        basis=_basis(basis_path),
        times=[_time(t) for t in times] or None,
        fields=fields or None,
        region=_optional_region(lat, lon, radius_km),
        threshold=threshold,
        allow_unverified_basis=allow_unverified_basis,
        lead=lead,
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


def _ints(text: str) -> list[int]:
    try:
        return [int(part) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise click.BadParameter(f"{text!r} is not a comma-separated list of integers") from exc


def _percent(value: float | None) -> str:
    return "-" if value is None or value != value else f"{100 * value:.1f}%"


@latents.command("evaluate")
@click.argument("source", type=click.Path())
@_adapter_option
@_mask_option
@click.option("--layer", type=int, help="Layer the bases were fitted to.  [default: the last]")
@click.option("--blocks", type=int, default=5, show_default=True, help="Contiguous time blocks.")
@click.option(
    "--test-block", "test_blocks", type=int, multiple=True,
    help="Block to hold out; repeatable.  [default: the last]",
)  # fmt: skip
@click.option(
    "--gap", type=int, default=1, show_default=True,
    help="Training times dropped beside each held-out block.",
)  # fmt: skip
@click.option("--split-only", is_flag=True, help="Print the split (to fit on) and stop.")
@click.option(
    "--basis", "basis_paths", type=click.Path(dir_okay=False), multiple=True,
    help="A basis file fitted on the training times (see --split-only); repeatable.",
)  # fmt: skip
@click.option(
    "--basis-sha256", "basis_hashes", multiple=True, metavar="HASH",
    help="Refuse the --basis file in the same position unless its content has this sha256; "
    "give one per --basis, or none (the printed reproduce line gives them).",
)  # fmt: skip
@click.option(
    "--pca", "pca_text", metavar="K,K,...",
    help="Fit a PCA on the training times at these ranks, and set the bases against it.",
)  # fmt: skip
@click.option("--stability", is_flag=True, help="Match the features of the bases to each other.")
@click.option("--recur-above", type=float, default=0.9, show_default=True,
              help="Cosine at which a matched feature recurs.")  # fmt: skip
@click.option("--active-above", type=float, default=0.0, show_default=True,
              help="An activation above this magnitude is active.")  # fmt: skip
@click.option("--duplicate-above", type=float, default=0.95, show_default=True,
              help="Cosine at which two decoder directions are near-duplicates.")  # fmt: skip
@click.option("--out", type=click.Path(dir_okay=False), help="Also write the results (JSON).")
@_record_option
@_unverified_option
@_json_option
def evaluate_cmd(
    source, adapter, mask_variable, layer, blocks, test_blocks, gap, split_only, basis_paths,
    basis_hashes, pca_text, stability, recur_above, active_above, duplicate_above, out,
    record_path,
    allow_unverified_basis, as_json,
):  # fmt: skip
    """Score frozen bases on times they were not fitted on.

    Times are cut into contiguous blocks, the last held out by default, with a buffer.
    Each --basis (fit it with `--time` on the training times: `--split-only` lists
    them) is scored on both sides: explained variance, active features, dead and
    near-duplicate features. --pca sets them against a PCA fitted here, on the
    training times only; --stability matches the bases to each other.
    """
    from xaig.core.errors import RequestError
    from xaig.latents import (
        Dictionary,
        basis_hash,
        evaluate_basis,
        evaluation_record,
        fidelity_curve,
        load_basis,
        result_provenance,
        save_record,
        seed_stability,
        split_time_blocks,
    )
    from xaig.latents.evaluate import jsonable

    if basis_hashes and len(basis_hashes) != len(basis_paths):
        raise RequestError(
            f"{len(basis_hashes)} --basis-sha256 for {len(basis_paths)} --basis: "
            "give one per --basis, in the same order, or none"
        )
    if record_path and split_only:
        raise click.UsageError("--record keeps results, and --split-only computes none")
    opened = open_for_cli(source, adapter, mask_variable)
    info = opened.info()
    layer = info.last_layer if layer is None else layer
    split = split_time_blocks(
        info.times, n_blocks=blocks, test_blocks=tuple(test_blocks) or (-1,), gap=gap
    )
    payload: dict = {
        "layer": layer,
        "split": split.to_dict(),
        "provenance": result_provenance(info),
    }
    bases = [
        load_basis(path, sha256=basis_hashes[i] if basis_hashes else None)
        for i, path in enumerate(basis_paths)
    ]
    ranks = _ints(pca_text) if pca_text else []
    if not split_only:
        if not bases and not ranks:
            raise click.UsageError(
                "give --basis FILE (repeatable) or --pca K,K,..., or --split-only"
            )
        common = {"layer": layer, "active_above": active_above,
                  "allow_unverified_basis": allow_unverified_basis}  # fmt: skip
        evaluations = [
            evaluate_basis(b, opened, split, duplicate_threshold=duplicate_above, **common)
            for b in bases
        ]
        payload["evaluations"] = [e.to_dict() for e in evaluations]
        curve = matching = None
        if ranks:
            autoencoders = [
                (b, Path(p).stem)
                for b, p in zip(bases, basis_paths, strict=True)
                if isinstance(b, Dictionary)
            ]
            curve = fidelity_curve(
                opened, split, pca_components=ranks, duplicate_threshold=duplicate_above,
                dictionaries=[b for b, _ in autoencoders], labels=[n for _, n in autoencoders],
                **common,
            )  # fmt: skip
            payload["curve"] = curve.to_dict()
        if stability:
            matching = seed_stability(
                bases, threshold=recur_above, allow_unverified=allow_unverified_basis
            )
            payload["stability"] = matching.to_dict()
        settings = {
            "adapter": adapter, "mask_variable": mask_variable, "layer": layer,
            "blocks": blocks, "test_blocks": list(test_blocks) or [-1], "gap": gap,
            "bases": [str(p) for p in basis_paths],
            "basis_sha256": [basis_hash(b) for b in bases], "pca": ranks, "stability": stability,
            "recur_above": recur_above, "active_above": active_above,
            "duplicate_above": duplicate_above,
            "allow_unverified_basis": allow_unverified_basis,
        }  # fmt: skip
        command = _evaluate_command(source, info, settings)
    payload = jsonable(payload)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(payload, indent=2) + "\n")
    if record_path:  # refused with --split-only, so everything below was computed
        save_record(
            record_path,
            evaluation_record(
                info, split, evaluations=evaluations, curve=curve, stability=matching,
                bases=bases, settings=settings, command=command,
            ),
        )  # fmt: skip
    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return
    click.echo(
        f"layer {layer}: fit on {len(split.train)} time(s), hold out {len(split.test)} "
        f"({', '.join(split.test)}), {len(split.buffer)} in the buffer"
    )
    if split_only:
        click.echo("\ntrain  " + "\n       ".join(split.train))
        return
    rows = [
        {
            "basis": Path(p).name, "features": e["n_features"], "fitted": e["fitted_on"],
            "ev_train": _percent(e["train"]["explained_variance"]),
            "ev_test": _percent(e["test"]["explained_variance"]),
            "active_train": f"{e['train']['mean_active_features']:.2f}",
            "active_test": f"{e['test']['mean_active_features']:.2f}",
            "dead_test": _percent(e["test"]["dead_fraction"]),
            "duplicates": e["redundancy"]["n_near_duplicates"],
        }
        for p, e in zip(basis_paths, payload["evaluations"], strict=True)
    ]  # fmt: skip
    if rows:
        click.echo("\n" + _render.table(rows))
    if "curve" in payload:
        rows = [
            {
                "point": p["label"], "active": f"{p['mean_active_features']:.2f}",
                "ev_test": _percent(p["explained_variance"]),
                "pca_there": _percent(p["pca_at_same_sparsity"]),
            }
            for p in payload["curve"]["points"]
        ]  # fmt: skip
        click.echo("\nheld out, against mean active features\n" + _render.table(rows))
    if "stability" in payload:
        s = payload["stability"]
        click.echo(
            f"\nstability of {s['n_dictionaries']} bases: {_percent(s['fraction_recurring'])} of "
            f"matched features at cosine >= {s['threshold']:g} "
            f"(median {s['matched_similarity']['median']:.3f}; random directions "
            f"{s['chance_similarity']['median']:.3f})"
        )
    click.echo(f"\nreproduce: {command}")


def _evaluate_command(source: str, info, settings: dict) -> str:
    """The shell command that redoes an evaluation. It pins the archive's commit, which
    an ``hf://`` source resolves, and each basis by the content hash of the file read."""
    s = settings
    words = ["xaig", "latents", "evaluate", source, "--adapter", s["adapter"]]
    if s["mask_variable"]:
        words += ["--mask-variable", s["mask_variable"]]
    if info.commit:
        words += ["--revision", info.commit]
    words += ["--layer", s["layer"], "--blocks", s["blocks"], "--gap", s["gap"]]
    for block in s["test_blocks"]:
        words += ["--test-block", block]
    for path, sha256 in zip(s["bases"], s["basis_sha256"], strict=True):
        words += ["--basis", path, "--basis-sha256", sha256]
    if s["pca"]:
        words += ["--pca", ",".join(map(str, s["pca"]))]
    if s["stability"]:
        words += ["--stability", "--recur-above", f"{s['recur_above']:g}"]
    words += ["--active-above", f"{s['active_above']:g}"]
    words += ["--duplicate-above", f"{s['duplicate_above']:g}"]
    if s["allow_unverified_basis"]:
        words.append("--allow-unverified-basis")
    return shlex.join(str(w) for w in words)


def _steer_command(source, settings: dict, spec: dict, basis_sha256: str) -> str:
    """The shell command that redoes a steering experiment, the basis pinned by content."""
    s, steer = settings, spec["steer"]
    words = ["xaig", "latents", "steer", *([source] if source else [])]
    words += ["--adapter", s["adapter"]]
    for key, value in s["adapter_options"].items():
        words += ["--adapter-option", f"{key}={json.dumps(value)}"]
    words += ["--basis", s["basis"], "--basis-sha256", basis_sha256]
    words += ["--layer", steer["layer"], "--feature", steer["feature"]]
    words += ["--mode", steer["mode"], "--amount", f"{steer['amount']!r}"]
    for t in steer["times"]:
        words += ["--time", t]
    words += ["--steps", spec["steps"], "--seeds", ",".join(map(str, spec["seeds"]))]
    words += ["--random-draws", spec["n_random"], "--random-seed", spec["random_seed"]]
    for name in s["fields"]:
        words += ["--field", name]
    for layer in s["record_layers"]:
        words += ["--record-layer", layer]
    if s["allow_unverified_basis"]:
        words.append("--allow-unverified-basis")
    return shlex.join(str(w) for w in words)


def _adapter_options(pairs: tuple[str, ...]) -> dict:
    """``KEY=VALUE`` pairs as adapter options; a value is JSON where it can be (``3``,
    ``true``, ``0.5``) and text otherwise."""
    options = {}
    for pair in pairs:
        key, equals, value = pair.partition("=")
        if not equals or not key:
            raise click.BadParameter(f"{pair!r} is not KEY=VALUE", param_hint="--adapter-option")
        try:
            options[key] = json.loads(value)
        except ValueError:
            options[key] = value
    return options


@latents.command("steer")
@click.argument("source", required=False)
@click.option("--adapter", required=True, help="An adapter that can be run (Intervenable).")
@click.option("--adapter-option", "adapter_options", multiple=True, metavar="KEY=VALUE",
              help="An option of the adapter; repeatable.")  # fmt: skip
@_basis_option
@click.option("--layer", type=int, required=True, help="Layer the basis reads; the edit acts here.")
@click.option("--feature", type=int, required=True, help="The feature of the basis to change.")
@click.option("--mode", type=click.Choice(["add", "scale", "clamp"]), default="add",
              show_default=True, help="What --amount does to the activation.")  # fmt: skip
@click.option("--amount", type=float, required=True, help="Added, scaled by, or clamped to.")
@click.option("--time", "times", type=int, multiple=True,
              help="Forward step to act at; repeatable.  [default: 0]")  # fmt: skip
@click.option("--steps", type=int, required=True, help="Forward steps to run.")
@click.option("--seeds", "seeds_text", default="0", show_default=True, metavar="N,N,...",
              help="Noise seeds; every arm of a seed shares its noise.")  # fmt: skip
@click.option("--random-draws", type=int, default=20, show_default=True,
              help="Random directions of the same size, to set the feature against.")  # fmt: skip
@click.option("--random-seed", type=int, default=0, show_default=True)
@click.option("--field", "fields", multiple=True, help="A physical field to measure; repeatable.  "
              "[default: all]")  # fmt: skip
@click.option("--record-layer", "record_layers", type=int, multiple=True,
              help="Also compare the latents of this layer; repeatable.")  # fmt: skip
@click.option("--out", type=click.Path(dir_okay=False), help="Also write the result (JSON).")
@_record_option
@_unverified_option
@_json_option
def steer_cmd(
    source, adapter, adapter_options, basis_path, layer, feature, mode, amount, times, steps,
    seeds_text, random_draws, random_seed, fields, record_layers, out, record_path,
    allow_unverified_basis, as_json,
):  # fmt: skip
    """Change a feature inside a running system and set what happens against chance.

    Four arms share one initial state and one noise seed per seed: a control, one with the
    latents replaced by the dictionary's reconstruction, one with the feature changed, and
    --random-draws with a random direction of the same length changed by the same amount.
    Each field's response to the feature is reported against the random directions'.
    """
    from xaig.latents import (
        Steer,
        open_intervenable,
        run_steering,
        save_record,
        steering_record,
    )
    from xaig.latents.evaluate import jsonable, save_result

    basis = _basis(basis_path)
    if basis is None:
        raise click.UsageError("--basis FILE is required")
    options = _adapter_options(adapter_options)
    system = open_intervenable(source, adapter, **options)
    steer = Steer(
        layer=layer, feature=feature, mode=mode, amount=amount, times=tuple(times) or (0,)
    )
    result = run_steering(
        system, basis, steer, steps=steps, seeds=_ints(seeds_text),
        n_random=random_draws, random_seed=random_seed, fields=list(fields) or None,
        record_layers=list(record_layers) or None,
        allow_unverified_basis=allow_unverified_basis,
    )  # fmt: skip
    spec = result.spec
    settings = {
        "adapter": adapter, "adapter_options": options, "basis": basis_path,
        "fields": list(fields), "record_layers": list(record_layers),
        "allow_unverified_basis": allow_unverified_basis,
    }  # fmt: skip
    command = _steer_command(source, settings, spec, result.provenance["basis"]["sha256"])
    if out:
        save_result(out, result)
    if record_path:
        save_record(record_path, steering_record(result, settings=settings, command=command))
    if as_json:
        click.echo(json.dumps(jsonable(result.to_dict()), indent=2))
        return
    click.echo(
        f"{mode} {amount:g} on feature {feature} of layer {layer} at time(s) "
        f"{', '.join(map(str, spec['steer']['times']))}; {steps} step(s), "
        f"seed(s) {', '.join(map(str, spec['seeds']))}, {spec['n_random']} random direction(s)"
    )
    rows = [
        {
            "field": name, "response": f"{e.feature_response:.4g}",
            "+-": "-" if e.feature_se is None else f"{e.feature_se:.2g}",
            "recon_only": f"{e.reconstruction_response:.4g}",
            "random_|resp|": f"{float(abs(e.random_responses).mean()):.4g}",
            "effect_size": "-" if e.effect_size is None else f"{e.effect_size:.2f}",
            "rank": f"{e.rank}/{e.random_responses.size + 1}", "p": f"{e.p_value:.2g}",
        }
        for name, e in result.effects.items()
    ]  # fmt: skip
    click.echo("\n" + _render.table(rows))
    click.echo(f"\nreproduce: {command}")


__all__ = ["latents"]
