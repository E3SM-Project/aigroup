"""The latent explorer: pick a region, see what the model's channels do there.

Widgets and layout only. The analysis is ``xaig.daig.latent.analyse_region`` and
the maps are ``xaig.faig.map_figure``; the last tab hands back the settings,
command and code that reproduce what is on screen.

Streamlit reruns this function on every interaction, so whatever is slow is
cached -- and bounded, since a cache of per-node arrays grows without being asked
to. The 100 MB layers themselves are never cached: the analysis reads them and
lets them go.
"""

from __future__ import annotations

import json
import shlex

import numpy as np
import streamlit as st

from xaig.core.errors import XaigError
from xaig.daig.latent import Region, RegionAnalysis, analyse_region, load_channels, open_source
from xaig.faig import have_coastlines, map_figure, to_png
from xaig.waig.config import configured_latents

_CACHED = 16
_COLUMNS = 3


@st.cache_resource(max_entries=8, show_spinner=False)
def _open(path: str, mask_variable: str | None):
    options = {"mask_variable": mask_variable} if mask_variable else {}
    return open_source(path, **options)


@st.cache_data(max_entries=_CACHED, show_spinner="Analysing the region…")
def _analyse(path: str, mask_variable: str | None, settings: dict) -> RegionAnalysis:
    kwargs = dict(settings)
    region = Region(**kwargs.pop("region"))
    return analyse_region(_open(path, mask_variable), region=region, **kwargs)


@st.cache_data(max_entries=_CACHED, show_spinner="Reading channels…")
def _channels(path: str, mask_variable: str | None, time: str, layer: int, channels, centred):
    return load_channels(
        _open(path, mask_variable), time=time, layer=layer, channels=channels, centred=centred
    )


@st.cache_data(max_entries=6 * _CACHED, show_spinner=False)
def _map(path: str, mask_variable: str | None, values, title, label, region, limit, dark):
    fig = map_figure(
        _open(path, mask_variable).grid(),
        values,
        title=title,
        label=label,
        region=Region(*region),
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
        if token.isdigit() and int(token) < n_channels:
            wanted.append(int(token))
        else:
            refused.append(token)
    return list(dict.fromkeys(wanted)), refused


def _pick_archive() -> tuple[str | None, str | None]:
    known = st.session_state.setdefault("latent_archives", configured_latents())
    with st.sidebar:
        st.subheader("Archive")
        added = st.text_input("Open an archive", placeholder="/path/to/latents/atmosphere")
        if added and added not in known:
            known.append(added)
        if not known:
            return None, None
        path = st.selectbox("Archive", known, index=len(known) - 1 if added else 0)
        mask_variable = st.text_input(
            "Mask variable",
            help="A reference-file variable that is missing where nodes mean nothing, "
            "such as sst for an ocean model. Leave empty when the archive has its own mask.",
        )
    return path, mask_variable.strip() or None


def _controls(info) -> dict:
    layers = {f"{x.index} · {x.label}" if x.label else str(x.index): x.index for x in info.layers}
    names = list(layers)
    n_channels = info.layer(info.last_layer).n_channels
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
        n_components = st.number_input("Principal components", 0, 16, 4)
    return {
        "time": time,
        "layer": layer,
        "region": {"lat": lat, "lon": lon, "radius_km": float(radius_km)},
        "rank_layer": rank_layer,
        "top": top,
        "pinned": pinned,
        "centred": centred,
        "reference": reference,
        "n_components": int(n_components),
    }


def _gallery(path, mask_variable, fields, titles, label, region, limit, across=_COLUMNS) -> None:
    columns = st.columns(across)
    dark = _dark_page()
    for i, (values, title) in enumerate(zip(fields, titles, strict=True)):
        png = _map(path, mask_variable, values, title, label, region, limit, dark)
        columns[i % across].image(png, width="stretch")


def reproduction(path: str, mask_variable: str | None, settings: dict) -> tuple[str, str]:
    """The shell command and the Python that redo an analysis, one option a line."""
    s, r = settings, settings["region"]
    options = [
        ("--time", s["time"]), ("--layer", s["layer"]), ("--rank-layer", s["rank_layer"]),
        ("--lat", f"{r['lat']:g}"), ("--lon", f"{r['lon']:g}"),
        ("--radius-km", f"{r['radius_km']:g}"), ("--top", s["top"]),
        ("--reference", s["reference"]), ("--pcs", s["n_components"]),
        *(("--pin", c) for c in s["pinned"]),
    ]  # fmt: skip
    if mask_variable:
        options.append(("--mask-variable", mask_variable))
    lines = [shlex.join(["xaig", "daig", "latent", "region", path])]
    lines += [shlex.join([flag, str(value)]) for flag, value in options]
    lines += ["--centred"] if s["centred"] else []
    command = " \\\n    ".join(lines)

    opened = f"open_source({path!r}" + (
        f", mask_variable={mask_variable!r})" if mask_variable else ")"
    )
    region = ", ".join(f"{k}={v!r}" for k, v in r.items())
    arguments = [f"region=Region({region})"]
    arguments += [f"{k}={v!r}" for k, v in s.items() if k != "region"]
    python = (
        "from xaig.daig.latent import Region, analyse_region, open_source\n\n"
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
    settings = _controls(info)
    no_pca = None
    try:
        try:
            result = _analyse(path, mask_variable, settings)
        except ValueError as exc:
            if not settings["n_components"]:
                raise
            # A region too small for the components asked of it can still be ranked
            # and compared; say why under Components rather than blanking the view.
            no_pca = str(exc)
            result = _analyse(path, mask_variable, {**settings, "n_components": 0})
    except (ValueError, KeyError) as exc:
        st.warning(str(exc.args[0]))
        return
    if not have_coastlines():
        st.caption("Maps are drawn without coastlines; install `xaig[maps]` for them.")

    region = tuple(settings["region"].values())
    layer, time = settings["layer"], settings["time"]
    tab_channels, tab_similarity, tab_pca, tab_reproduce = st.tabs(
        ["Channels", "Similarity", "Components", "Reproduce"]
    )

    with tab_channels:
        ranked = [int(c) for c in result.ranking.channels]
        st.markdown(
            f"The **{len(ranked)} channels** that respond most strongly among the "
            f"**{result.nodes.size} nodes** of the region, ranked at layer "
            f"{settings['rank_layer']} and mapped at layer {layer}."
        )
        table = [
            {"rank": i, "channel": c, "peak |activation|": float(s), "pinned": c in set(
                result.ranking.pinned)}
            for i, (c, s) in enumerate(zip(ranked, result.ranking.scores, strict=True), start=1)
        ]  # fmt: skip
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

    with tab_pca:
        if no_pca:
            st.warning(no_pca)
        elif result.pca is None:
            st.info("Ask for at least one principal component in the sidebar.")
        else:
            st.markdown(
                "Components are fitted on the region's nodes, area-weighted, and then "
                "projected onto every node."
            )
            rows = [
                {
                    "component": f"PC{k}",
                    "explained variance (%)": round(100 * float(ratio), 2),
                    "strongest channels (loading)": "  ".join(
                        f"{c} ({v:+.2f})" for c, v in loadings
                    ),
                }
                for k, (ratio, loadings) in enumerate(
                    zip(result.pca.explained_variance_ratio, result.pca.top_loadings(), strict=True)
                )
            ]
            st.dataframe(rows, hide_index=True, width="stretch")
            _gallery(
                path, mask_variable, result.scores.T,
                [f"PC{k} · {row['explained variance (%)']}%" for k, row in enumerate(rows)],
                "score", region, None,
            )  # fmt: skip

    with tab_reproduce:
        _reproduce(path, mask_variable, result)
