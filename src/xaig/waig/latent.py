"""The latent explorer: pick a model and a region, see what its channels do there,
and set them against a physical field the archive keeps beside them.

Widgets and layout only. The analysis is ``xaig.daig.latent`` and the figures are
``xaig.faig``; the last tab hands back the settings, command and code that
reproduce what is on screen.

Streamlit reruns this function on every interaction, so whatever is slow is
cached -- and bounded, since a cache of per-node arrays grows without being asked
to. The 100 MB layers themselves are never cached: the analysis reads them and
lets them go.
"""

from __future__ import annotations

import json
import os
import shlex

import numpy as np
import streamlit as st

from xaig.core.errors import RequestError, XaigError
from xaig.daig.latent import (
    FeatureProfile,
    FieldRanking,
    ReferenceFields,
    Region,
    RegionAnalysis,
    RegionSeries,
    analyse_region,
    feature_profile,
    load_basis,
    load_channels,
    open_source,
    rank_by_field,
    region_series,
)
from xaig.daig.latent.source import check_basis_fits
from xaig.faig import map_figure, profile_figure, series_figure, to_png, why_no_coastlines
from xaig.waig.config import configured_latents

_CACHED = 16
_COLUMNS = 3
_IN_REGION = "PCA fitted in the region"
_FROM_FILE = "a basis file (global PCA, SAE)…"
_NO_FIELD = "none"
_BASES = "bases"  # where an archive keeps basis files fitted on it
_PROFILED_TIMES = 5


@st.cache_resource(max_entries=16, show_spinner=False)
def _open(path: str, mask_variable: str | None):
    options = {"mask_variable": mask_variable} if mask_variable else {}
    return open_source(path, **options)


def _stamp(path: str | None) -> int | None:
    """When a file last changed. A basis is refitted to the same name many times
    in an afternoon, and a cache keyed on the name alone would go on showing the
    first one: everything that depends on the file is keyed on this as well."""
    try:
        return os.stat(path).st_mtime_ns if path else None
    except OSError:
        return None


@st.cache_resource(max_entries=4, show_spinner="Reading the basis…")
def _basis(path: str, stamp: int | None):
    return load_basis(path)


@st.cache_data(max_entries=_CACHED, show_spinner="Analysing the region…")
def _analyse(
    path: str, mask_variable: str | None, settings: dict, stamp: int | None = None
) -> RegionAnalysis:
    kwargs = dict(settings)
    region = Region(**kwargs.pop("region"))
    basis = kwargs.pop("basis")
    return analyse_region(
        _open(path, mask_variable),
        region=region,
        basis=_basis(basis, stamp) if basis else None,
        **kwargs,
    )


@st.cache_data(max_entries=_CACHED, show_spinner="Reading channels…")
def _channels(path: str, mask_variable: str | None, time: str, layer: int, channels, centred):
    return load_channels(
        _open(path, mask_variable), time=time, layer=layer, channels=channels, centred=centred
    )


@st.cache_data(max_entries=_CACHED, show_spinner="Following the region through time…")
def _series(
    path: str, mask_variable: str | None, layer: int, region, columns, centred, basis, stamp,
    unverified: bool = False,
) -> RegionSeries:  # fmt: skip
    chosen = (
        {"features": columns, "basis": _basis(basis, stamp), "allow_unverified_basis": unverified}
        if basis
        else {"channels": columns}
    )
    return region_series(
        _open(path, mask_variable), layer=layer, region=Region(*region), centred=centred, **chosen
    )


@st.cache_data(max_entries=_CACHED, show_spinner="Looking for basis files…")
def _found_bases(
    archives: tuple[str, ...], path: str, mask_variable: str | None, layer: int
) -> list[tuple[str, str]]:
    """``(label, file)`` for every basis kept in the ``bases/`` folder of any open
    archive that fits this archive's layer. Whether it fits is ``daig``'s check, so a
    basis fitted on the control is offered for every run of the same network, and one
    fitted on another network or layer is not. An archive that cannot be opened, or
    cannot list its files, offers none."""
    info = _open(path, mask_variable).info()
    found = []
    for archive in archives:
        try:
            other = _open(archive, None)
            names = other.files(_BASES) if hasattr(other, "files") else ()
        except XaigError:
            continue
        run = archive.rstrip("/").rsplit("/", 1)[-1]
        for name in names:
            if not name.endswith(".npz"):
                continue
            try:
                local = other.file(name)
                basis = load_basis(local) if local is not None else None
                if basis is None:
                    continue
                check_basis_fits(basis, info, layer)
            except XaigError:
                continue
            kind = type(basis).__name__
            label = f"{name.rsplit('/', 1)[-1]} · {kind}, {basis.n_features} features · {run}"
            found.append((label, str(local)))
    return found


@st.cache_data(max_entries=_CACHED, show_spinner="Setting the layer against the field…")
def _rank(
    path, mask_variable, time, layer, field, basis, stamp, unverified: bool = False, lead: int = 0
) -> FieldRanking:
    return rank_by_field(
        _open(path, mask_variable), time=time, layer=layer, field=field, top=_COLUMNS * 2 - 1,
        basis=_basis(basis, stamp) if basis else None, allow_unverified_basis=unverified,
        lead=lead,
    )  # fmt: skip


@st.cache_data(max_entries=_CACHED, show_spinner=False)
def _field(
    path: str, mask_variable: str | None, field: str, time: str, lead: int = 0
) -> np.ndarray:
    source = _open(path, mask_variable)
    return source.field(field, time, lead=lead) if lead else source.field(field, time)


@st.cache_data(max_entries=_CACHED, show_spinner="Reading them…")
def _columns(path, mask_variable, time, layer, columns, basis, stamp) -> np.ndarray:
    """``(n_nodes, len(columns))``: channels as recorded, or a basis's features."""
    source = _open(path, mask_variable)
    if basis:
        return _basis(basis, stamp).transform(source.load(time, layer), features=list(columns))
    return np.asarray(source.load(time, layer, channels=list(columns)), dtype=np.float64)


@st.cache_data(max_entries=_CACHED, show_spinner="Profiling it against every field…")
def _profile(
    path, mask_variable, layer, column, basis, stamp, times, unverified: bool = False, lead: int = 0
) -> FeatureProfile:
    return feature_profile(
        _open(path, mask_variable), layer=layer, column=column, times=list(times),
        basis=_basis(basis, stamp) if basis else None, allow_unverified_basis=unverified,
        lead=lead,
    )  # fmt: skip


@st.cache_data(max_entries=6 * _CACHED, show_spinner=False)
def _map(
    path: str, mask_variable: str | None, values, title, label, region, limit, dark,
    symmetric: bool = True,
):  # fmt: skip
    fig = map_figure(
        _open(path, mask_variable).grid(),
        values,
        title=title,
        label=label,
        region=Region(*region),
        symmetric=symmetric,
        limit=limit,
        dark=dark,
    )
    return to_png(fig)


def _dark_page() -> bool:
    """Whether the viewer's theme is dark, where Streamlit is new enough to say."""
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", "light") == "dark"


def _parse_channels(text: str, n_channels: int) -> tuple[list[int], list[str]]:
    wanted, refused = [], []
    for token in text.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if token.isascii() and token.isdigit() and int(token) < n_channels:
            wanted.append(int(token))
        else:
            refused.append(token)
    return list(dict.fromkeys(wanted)), refused


def _archive_label(path: str) -> str:
    """The model and component an archive says it holds, then where it is: the
    drop-down is for choosing a model, and a path alone does not say which."""
    try:
        name = _open(path, None).info().name
    except XaigError:
        return path
    return f"{name}  ·  {path}"


def _pick_archive() -> tuple[str | None, str | None]:
    known = st.session_state.setdefault("latent_archives", configured_latents())
    with st.sidebar:
        st.subheader("Model")
        added = st.text_input("Open an archive", placeholder="/path/to/latents/atmosphere")
        if added and added not in known:
            known.append(added)
        if not known:
            return None, None
        path = st.selectbox(
            "Archive", known, index=len(known) - 1 if added else 0, format_func=_archive_label
        )
        mask_variable = st.text_input(
            "Mask variable",
            help="A reference-file variable that is missing where nodes mean nothing, "
            "such as sst for an ocean model. Leave empty when the archive has its own mask.",
        )
    return path, mask_variable.strip() or None


def _controls(info, path: str, mask_variable: str | None, fields: tuple[str, ...]):
    """The analysis's settings, and the physical field to set it against (or None)."""
    layers = {f"{x.index} · {x.label}" if x.label else str(x.index): x.index for x in info.layers}
    names = list(layers)
    with st.sidebar:
        st.subheader("Where and when")
        time = st.selectbox("Time", info.times)
        layer = layers[st.selectbox("Layer", names, index=len(names) - 1)]
        left, right = st.columns(2)
        lat = left.number_input("Latitude", -90.0, 90.0, 5.0, step=1.0)
        lon = right.number_input("Longitude", -180.0, 360.0, -140.0, step=1.0)
        radius_km = st.slider("Radius (km)", 100, 5000, 1500, step=100)

        st.subheader("How")
        centred = st.checkbox(
            "Centre channels",
            value=True,
            help="Remove each channel's area-weighted global mean. Without it, channels "
            "carrying a large constant offset dominate the ranking and the similarity.",
        )
        rank_layer = layers[
            st.selectbox(
                "Rank channels at",
                names,
                index=len(names) - 1,
                help="The last layer shows what the network ends up emphasising; layer 0 "
                "shows what the encoder does.",
            )
        ]
        n_channels = info.layer(rank_layer).n_channels
        top = st.slider("Channels to rank", 3, 18, 6, step=3)
        pinned, refused = _parse_channels(
            st.text_input("Pin channels", placeholder="223, 45", help="Listed first, whatever."),
            n_channels,
        )
        if refused:
            st.warning(f"Not channels 0–{n_channels - 1}: {', '.join(refused)}")
        reference = st.radio(
            "Compare against",
            ["nearest", "mean"],
            format_func={"nearest": "the node at the centre", "mean": "the region's mean"}.get,
            horizontal=True,
        )

        st.subheader("Features")
        archives = tuple(st.session_state.get("latent_archives", ()))
        found = dict(_found_bases(archives, path, mask_variable, layer))
        method = st.selectbox(
            "Method",
            [_IN_REGION, *found, _FROM_FILE],
            help="A basis file comes from `xaig daig latent pca` (a global PCA) or "
            "`xaig taig sae` (a sparse autoencoder); its features that respond most "
            "strongly in the region are the ones mapped. Listed by name are the basis "
            f"files in the `{_BASES}/` folder of any open archive that fit this layer.",
        )
        basis, unverified = found.get(method), False
        if method == _FROM_FILE:
            basis = st.text_input("Basis file", placeholder="/path/to/sae.npz").strip() or None
            unverified = st.checkbox(
                "Allow an unverified basis",
                help="A basis that does not say which model and layer it was fitted on is "
                "refused unless this is ticked. One that says another model or layer is "
                "refused either way.",
            )
        n_components = st.number_input("Features to map", 0, 16, 4)

        field = None
        if fields:
            st.subheader("A physical field")
            field = st.selectbox(
                "Field",
                [_NO_FIELD, *fields],
                help="One of the fields the archive keeps beside its latents. The Field "
                "tab ranks the layer's channels, or the basis's features, by how closely "
                "they follow it.",
            )
            field = None if field == _NO_FIELD else field
            lead = st.radio(
                "Set it against",
                [0, 1],
                format_func={
                    0: "what the pass reads (this time)",
                    1: "what it writes (the next)",
                }.get,
                help="A forward pass reads the state at its own time and writes the next. An "
                "input belongs with the first; an output the model produces, precipitation "
                "say, with the second.",
            )
    return (field, lead if field else 0), {
        "time": time,
        "layer": layer,
        "region": {"lat": lat, "lon": lon, "radius_km": float(radius_km)},
        "rank_layer": rank_layer,
        "top": top,
        "pinned": pinned,
        "centred": centred,
        "reference": reference,
        "n_components": int(n_components) if basis or method == _IN_REGION else 0,
        "basis": basis,
        "allow_unverified_basis": bool(basis and unverified),
    }


def _gallery(
    path, mask_variable, fields, titles, label, region, limit, across=_COLUMNS, symmetric=None
) -> None:
    """``symmetric`` is one flag per map, or None for a scale centred on zero throughout."""
    columns = st.columns(across)
    dark = _dark_page()
    symmetric = [True] * len(titles) if symmetric is None else symmetric
    for i, (values, title) in enumerate(zip(fields, titles, strict=True)):
        png = _map(path, mask_variable, values, title, label, region, limit, dark, symmetric[i])
        columns[i % across].image(png, width="stretch")


def _signed(values: np.ndarray) -> bool:
    """Whether a map has both signs, and so wants a scale centred on zero."""
    finite = values[np.isfinite(values)]
    return bool(finite.size and finite.min() < 0.0 < finite.max())


def _field_tab(path, mask_variable, settings, field, lead, region) -> None:
    if field is None:
        st.info("Pick a physical field in the sidebar to set this layer against it.")
        return
    time, layer, basis = settings["time"], settings["layer"], settings["basis"]
    stamp, unverified = _stamp(basis), settings["allow_unverified_basis"]
    kind = "feature" if basis else "channel"
    try:
        ranking = _rank(path, mask_variable, time, layer, field, basis, stamp, unverified, lead)
    except RequestError as exc:
        st.warning(str(exc))
        return
    ranked = [int(c) for c in ranking.ranking.channels]
    when = f"{time}, against the {field} this pass writes" if lead else time
    st.markdown(
        f"The {kind}s of layer {layer} that follow **{field}** most closely at {when}: "
        "area-weighted correlation over the valid nodes. It says where to look, not what "
        f"causes what. Features come from the *Method* in the sidebar: a basis file, or "
        "else the layer's own channels."
    )
    st.dataframe(
        [
            {"rank": i, kind: c, "correlation": round(float(ranking.correlation[c]), 3)}
            for i, c in enumerate(ranked, start=1)
        ],
        hide_index=True,
        width="stretch",
    )
    values = _field(path, mask_variable, field, time, lead)
    columns = _columns(path, mask_variable, time, layer, tuple(ranked), basis, stamp)
    maps = [values, *columns.T]
    titles = [f"{field} · {time}" + (f" {lead:+d} step" if lead else "")] + [
        f"{kind.capitalize()} {c} · layer {layer} · r = {ranking.correlation[c]:+.2f}"
        for c in ranked
    ]
    _gallery(
        path, mask_variable, maps, titles, None, region, None,
        symmetric=[_signed(m) for m in maps],
    )  # fmt: skip

    st.markdown(
        f"**What one {kind} goes with.** Every field the archive keeps, where the {kind} "
        "is active (above zero) against where it is not, in units of each field's own "
        "spread. It describes the "
        f"{kind} by all the fields at once, which a single correlation cannot."
    )
    left, right = st.columns([1, 2])
    column = left.selectbox(kind.capitalize(), ranked)
    times = _source_times(path, mask_variable)
    # Spread through the run. A fixed stride can land on the same hour every day --
    # every 4th of a 6-hourly run is always noon somewhere -- and pool one hour only.
    spread = np.linspace(0, len(times) - 1, min(len(times), _PROFILED_TIMES))
    chosen = tuple(times[i] for i in np.unique(spread.round().astype(int)))
    if not right.toggle(
        f"Profile it (reads {len(chosen)} times)", value=False, help="Reads the layer at each."
    ):
        profile = None
    else:
        try:
            profile = _profile(
                path, mask_variable, layer, column, basis, stamp, chosen, unverified, lead
            )
        except RequestError as exc:
            st.warning(str(exc))
            profile = None
    if profile is not None:
        st.caption(
            f"{kind.capitalize()} {column} is active over {profile.coverage:.1%} of the area, "
            f"over {len(profile.times)} times from {profile.times[0]} to {profile.times[-1]}."
        )
        fig = profile_figure(
            profile.fields, profile.effect, title=f"{kind} {column} of layer {layer}",
            dark=_dark_page(),
        )  # fmt: skip
        st.image(to_png(fig), width="stretch")

    command = [
        f"xaig daig latent fields {shlex.quote(path)} --field {shlex.quote(field)}",
        f"--time {shlex.quote(time)} --layer {layer} --top {len(ranked)}",
    ]
    profile_command = [
        f"xaig daig latent profile {shlex.quote(path)} --layer {layer}",
        *(f"--time {shlex.quote(t)}" for t in chosen),
    ]
    if basis:
        command.append(f"--basis {shlex.quote(basis)}")
        profile_command += [f"--basis {shlex.quote(basis)}", f"--feature {column}"]
    else:
        profile_command.append(f"--channel {column}")
    if lead:
        command.append(f"--lead {lead}")
        profile_command.append(f"--lead {lead}")
    if unverified:
        command.append("--allow-unverified-basis")
        profile_command.append("--allow-unverified-basis")
    if mask_variable:
        command.append(f"--mask-variable {shlex.quote(mask_variable)}")
        profile_command.append(f"--mask-variable {shlex.quote(mask_variable)}")
    with st.expander("Reproduce"):
        st.code(" \\\n    ".join(command), language="bash")
        st.code(" \\\n    ".join(profile_command), language="bash")


@st.cache_data(max_entries=_CACHED, show_spinner=False)
def _source_times(path: str, mask_variable: str | None) -> tuple[str, ...]:
    return tuple(_open(path, mask_variable).info().times)


def reproduction(path: str, mask_variable: str | None, settings: dict) -> tuple[str, str]:
    """The shell command and the Python that redo an analysis, one option a line.
    Numbers are written in full: a rounded latitude is a different region."""
    s, r = settings, settings["region"]
    options = [
        ("--time", s["time"]), ("--layer", s["layer"]), ("--rank-layer", s["rank_layer"]),
        ("--lat", repr(r["lat"])), ("--lon", repr(r["lon"])),
        ("--radius-km", repr(r["radius_km"])), ("--top", s["top"]),
        ("--reference", s["reference"]), ("--features", s["n_components"]),
        *(("--pin", c) for c in s["pinned"]),
    ]  # fmt: skip
    if s.get("basis"):
        options.append(("--basis", s["basis"]))
    if mask_variable:
        options.append(("--mask-variable", mask_variable))
    lines = [shlex.join(["xaig", "daig", "latent", "region", path])]
    lines += [shlex.join([flag, str(value)]) for flag, value in options]
    lines += ["--centred"] if s["centred"] else []
    lines += ["--allow-unverified-basis"] if s.get("allow_unverified_basis") else []
    command = " \\\n    ".join(lines)

    opened = f"open_source({path!r}" + (
        f", mask_variable={mask_variable!r})" if mask_variable else ")"
    )
    region = ", ".join(f"{k}={v!r}" for k, v in r.items())
    arguments = [f"region=Region({region})"]
    unsaid = ("region", "basis") + (
        () if s.get("allow_unverified_basis") else ("allow_unverified_basis",)
    )
    arguments += [f"{k}={v!r}" for k, v in s.items() if k not in unsaid]
    names = ["Region", "analyse_region", "open_source"]
    if s.get("basis"):
        arguments.append(f"basis=load_basis({s['basis']!r})")
        names.insert(2, "load_basis")
    python = (
        f"from xaig.daig.latent import {', '.join(names)}\n\n"
        f"source = {opened}\n"
        "result = analyse_region(\n    source,\n    " + ",\n    ".join(arguments) + ",\n)"
    )
    return command, python


def _reproduce(path: str, mask_variable: str | None, result: RegionAnalysis) -> None:
    command, python = reproduction(path, mask_variable, result.settings)
    st.markdown("**From a terminal**")
    st.code(command, language="bash")
    st.markdown("**From Python**")
    st.code(python, language="python")
    summary = json.dumps(result.summary(), indent=2)
    st.download_button("Download settings and results (JSON)", summary, "latent_region.json")
    st.json(result.summary(), expanded=1)


def _feature_rows(result: RegionAnalysis) -> list[dict]:
    rows = []
    for info in result.feature_info:
        row = {"feature": info["label"]}
        if "explained_variance_ratio" in info:
            row["explained variance (%)"] = round(100 * info["explained_variance_ratio"], 2)
        if "peak_abs" in info:
            row["peak |activation| in the region"] = info["peak_abs"]
        row["strongest channels (loading)"] = "  ".join(
            f"{x['channel']} ({x['loading']:+.2f})" for x in info["top_loadings"]
        )
        rows.append(row)
    return rows


def _through_time(path, mask_variable, settings, result: RegionAnalysis) -> None:
    follow_features = bool(settings["basis"] and result.feature_info)
    what = "features" if follow_features else "channels"
    st.markdown(
        f"The region's area-weighted mean of the ranked {what}, at every time the archive "
        "holds. Times are placed by the archive's own calendar, so a gap between kept "
        "steps looks like one."
    )
    if not st.toggle("Follow them through time", value=False, help="Reads every time once."):
        return
    columns = (
        tuple(info["feature"] for info in result.feature_info)
        if follow_features
        else tuple(int(c) for c in result.ranking.channels)
    )
    series = _series(
        path,
        mask_variable,
        settings["layer"],
        tuple(settings["region"].values()),
        columns,
        settings["centred"],
        settings["basis"] if follow_features else None,
        _stamp(settings["basis"]) if follow_features else None,
        settings["allow_unverified_basis"],
    )
    hours = None if series.elapsed_seconds is None else [s / 3600 for s in series.elapsed_seconds]
    fig = series_figure(
        series.values,
        x=hours,
        tick_labels=series.times,
        labels=[f"{'F' if follow_features else 'ch'} {c}" for c in series.columns],
        x_label=None if hours is None else f"hours since {series.times[0]}",
        y_label="regional mean",
        title=f"Layer {settings['layer']} through time",
        dark=_dark_page(),
    )
    st.image(to_png(fig), width="stretch")
    st.dataframe(
        [
            {"time": label, **{str(c): float(v) for c, v in zip(series.columns, row, strict=True)}}
            for label, row in zip(series.times, series.values, strict=True)
        ],
        hide_index=True,
        width="stretch",
    )


def page() -> None:
    st.title("Latents")
    path, mask_variable = _pick_archive()
    if path is None:
        st.info(
            "Open a latent archive from the sidebar, or start the app with "
            "`xaig waig --latents PATH`."
        )
        return
    try:
        source = _open(path, mask_variable)
        info, grid = source.info(), source.grid()
    except XaigError as exc:
        st.error(str(exc))
        return

    shape = "×".join(str(n) for n in grid.shape) if grid.shape else "mesh"
    st.caption(
        " · ".join(str(x) for x in (info.model, info.component, info.checkpoint) if x)
        + f" · {shape}, {int(grid.valid.sum()):,} of {grid.n_nodes:,} nodes valid"
    )
    if info.experiment:
        st.caption("experiment: " + json.dumps(dict(info.experiment)))
    fields = source.field_names() if isinstance(source, ReferenceFields) else ()
    (field, lead), settings = _controls(info, path, mask_variable, tuple(fields))
    no_features = None
    try:
        try:
            result = _analyse(path, mask_variable, settings, _stamp(settings["basis"]))
        except RequestError as exc:
            if not settings["n_components"]:
                raise
            # A region too small for the components asked of it, or a basis that
            # does not fit this layer, can still be ranked and compared; say why
            # under Features rather than blanking the view. Whatever else is wrong
            # is refused before anything is read, so asking again costs nothing.
            no_features = str(exc)
            result = _analyse(path, mask_variable, {**settings, "n_components": 0, "basis": None})
    except RequestError as exc:
        st.warning(str(exc))
        return
    if reason := why_no_coastlines():
        st.caption(f"Maps are drawn without coastlines: {reason}.")

    region = tuple(settings["region"].values())
    layer, time = settings["layer"], settings["time"]
    tab_channels, tab_similarity, tab_features, tab_field, tab_time, tab_reproduce = st.tabs(
        ["Channels", "Similarity", "Features", "Field", "Through time", "Reproduce"]
    )

    with tab_channels:
        ranked = [int(c) for c in result.ranking.channels]
        pinned = set(result.ranking.pinned)
        st.markdown(
            f"The **{len(ranked)} channels** that respond most strongly among the "
            f"**{result.nodes.size} nodes** of the region, ranked at layer "
            f"{settings['rank_layer']} and mapped at layer {layer}."
        )
        table = [
            {"rank": i, "channel": c, "peak |activation|": float(s), "pinned": c in pinned}
            for i, (c, s) in enumerate(zip(ranked, result.ranking.scores, strict=True), start=1)
        ]
        st.dataframe(table, hide_index=True, width="stretch")
        shared = st.toggle("One colour scale for every map", value=False)
        fields = _channels(path, mask_variable, time, layer, tuple(ranked), settings["centred"])
        limit = float(np.nanmax(np.abs(fields))) if shared else None
        _gallery(
            path, mask_variable, fields.T, [f"Channel {c} · layer {layer}" for c in ranked],
            "activation", region, limit,
        )  # fmt: skip

    with tab_similarity:
        st.markdown(
            "Cosine similarity between each node's latent vector and the region's. "
            "Both maps share the fixed scale −1 to 1."
        )
        _gallery(
            path, mask_variable,
            [result.similarity_top, result.similarity],
            [f"The {len(ranked)} ranked channels · layer {layer}", f"All channels · layer {layer}"],
            "cosine similarity", region, 1.0, across=2,
        )  # fmt: skip

    with tab_features:
        if no_features:
            st.warning(no_features)
        elif result.scores is None:
            st.info("Ask for at least one feature in the sidebar.")
        else:
            st.markdown(
                "Features of the basis file that respond most strongly in the region, "
                "mapped everywhere."
                if settings["basis"]
                else "Components are fitted on the region's nodes, area-weighted, and then "
                "projected onto every node."
            )
            rows = _feature_rows(result)
            st.dataframe(rows, hide_index=True, width="stretch")
            _gallery(
                path, mask_variable, result.scores.T, [row["feature"] for row in rows],
                "activation" if settings["basis"] else "score", region, None,
            )  # fmt: skip

    with tab_field:
        _field_tab(path, mask_variable, settings, field, lead, region)

    with tab_time:
        _through_time(path, mask_variable, settings, result)

    with tab_reproduce:
        _reproduce(path, mask_variable, result)
