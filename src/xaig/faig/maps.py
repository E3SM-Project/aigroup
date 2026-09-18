"""Per-node values on a map.

Colour follows the job the numbers do. Everything drawn so far is *signed* --
an activation about its mean, a cosine similarity, a principal-component score --
so the default is a diverging blue-red scale whose neutral midpoint sits exactly
on zero, with limits symmetric about it: zero has to read as "nothing", and equal
magnitudes of either sign have to look equally strong. Invalid nodes are a flat
grey that belongs to no value. Pass ``symmetric=False`` with a single-hue ``cmap``
for a magnitude.

A dark surface gets a scale of its own rather than the light one on a black card:
the diverging midpoint becomes dark, so "nothing" still recedes into the page and
magnitude reads as brightness, where a near-white midpoint would glare.

Figures are built on ``matplotlib.figure.Figure`` directly, never through
``pyplot``: no global state, nothing to close, and safe to call from a server
handling several sessions at once.

Coastlines come from cartopy when it is installed (the ``maps`` extra) and its
Natural Earth data can be had; otherwise the map is still drawn, on plain axes,
with the grid's own mask outlined where it has one.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Protocol

from xaig.core.extras import missing_extra

# Checked before anything of daig's is imported: whoever wants figures should be
# pointed at the one extra that brings everything, not at numpy's and then ours.
try:
    import numpy as np
    from matplotlib.figure import Figure
except ImportError as exc:
    raise missing_extra(exc.name or "matplotlib", "faig") from exc

from xaig.daig.grid import Grid, small_circle

log = logging.getLogger(__name__)

SIGNED_CMAP = "RdBu_r"
MAGNITUDE_CMAP = "Blues"
INVALID_COLOUR = "#c9c9c4"
_INK = "#33332f"

# The dark counterparts. "berlin" (blue - black - red) arrived in matplotlib 3.10;
# before that the light scale is kept, which is legible if not ideal.
DARK_SIGNED_CMAP = "berlin"
DARK_MAGNITUDE_CMAP = "Blues_r"
DARK_SURFACE = "#0e1117"
DARK_INVALID_COLOUR = "#4b4b47"
_DARK_INK = "#d9d9d3"


class _Cap(Protocol):
    lat: float
    lon: float
    radius_km: float


@lru_cache(maxsize=1)
def _cartopy() -> Any | None:
    """``cartopy.crs`` if coastlines can actually be drawn, else None.

    Cartopy fetches its coastline data on first use, which fails on a compute
    node with no network -- and fails late, in the middle of rendering. Asking for
    the file up front turns that into a quiet fallback. ``XAIG_NO_COASTLINES=1``
    skips the attempt altogether.
    """
    if os.environ.get("XAIG_NO_COASTLINES"):
        return None
    try:
        import cartopy.crs as ccrs
        from cartopy.io import shapereader

        shapereader.natural_earth(resolution="110m", category="physical", name="coastline")
    except Exception as exc:  # absent, or present with no way to get its data
        log.info("drawing maps without coastlines: %s", exc)
        return None
    return ccrs


def have_coastlines() -> bool:
    return _cartopy() is not None


def _limits(values: np.ndarray, symmetric: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        top = float(np.max(np.abs(finite))) or 1.0
        return -top, top
    low, high = float(finite.min()), float(finite.max())
    return (low, high) if high > low else (low, low + 1.0)


def _break_at_dateline(lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN wherever a line would otherwise be drawn the long way round the map."""
    jump = np.flatnonzero(np.abs(np.diff(lon)) > 180.0) + 1
    return np.insert(lat, jump, np.nan), np.insert(lon, jump, np.nan)


def map_figure(
    grid: Grid,
    values: np.ndarray,
    *,
    title: str | None = None,
    label: str | None = None,
    region: _Cap | None = None,
    symmetric: bool = True,
    cmap: str | None = None,
    limit: float | None = None,
    coastlines: bool = True,
    dark: bool = False,
    figsize: tuple[float, float] = (7.0, 3.7),
) -> Figure:
    """One per-node field on a global map.

    A structured grid is drawn as cells, a mesh as points. ``region`` (anything
    with ``lat``, ``lon`` and ``radius_km``) is outlined. ``limit`` pins the colour
    range to ``±limit`` so several maps can be compared by eye; without it each
    map scales to its own data. ``dark`` draws for a dark page.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (grid.n_nodes,):
        raise ValueError(f"expected one value per node {(grid.n_nodes,)}, got {values.shape}")
    values = np.where(grid.valid, values, np.nan)
    vmin, vmax = (-limit, limit) if limit else _limits(values, symmetric)
    ink, invalid = (_DARK_INK, DARK_INVALID_COLOUR) if dark else (_INK, INVALID_COLOUR)
    if cmap is None and dark:
        from matplotlib import colormaps

        cmap = DARK_SIGNED_CMAP if symmetric else DARK_MAGNITUDE_CMAP
        cmap = cmap if cmap in colormaps else None
    cmap = cmap or (SIGNED_CMAP if symmetric else MAGNITUDE_CMAP)

    ccrs = _cartopy() if coastlines else None
    fig = Figure(figsize=figsize, layout="constrained")
    if dark:
        fig.set_facecolor(DARK_SURFACE)
    if ccrs is not None:
        ax = fig.add_subplot(projection=ccrs.PlateCarree())
        ax.set_global()
        on_map: dict[str, Any] = {"transform": ccrs.PlateCarree()}
    else:
        ax = fig.add_subplot()
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-90, 91, 30))
        ax.tick_params(labelsize=7, colors=ink, length=2)
        ax.grid(color=ink, alpha=0.15, linewidth=0.5)
        on_map = {}
    ax.set_facecolor(invalid)

    lon = (grid.lon + 180.0) % 360.0 - 180.0
    if grid.shape is not None:
        # Archives keep longitude in model order (0..360 wrapped to -180..180 is
        # not monotonic), and cells must be drawn west to east.
        order = np.argsort(lon.reshape(grid.shape)[0])
        lon_1d = lon.reshape(grid.shape)[0][order]
        lat_1d = grid.lat.reshape(grid.shape)[:, 0]
        cells = values.reshape(grid.shape)[:, order]
        drawn = ax.pcolormesh(
            lon_1d, lat_1d, cells, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", **on_map
        )
        if ccrs is None and grid.mask is not None:
            outline = grid.valid.reshape(grid.shape)[:, order].astype(float)
            ax.contour(lon_1d, lat_1d, outline, levels=[0.5], colors=ink, linewidths=0.4)
    else:
        keep = np.isfinite(values)
        size = max(1.0, 12000.0 / max(grid.n_nodes, 1))
        drawn = ax.scatter(
            lon[keep], grid.lat[keep], c=values[keep], s=size, cmap=cmap, vmin=vmin, vmax=vmax,
            linewidths=0, **on_map,
        )  # fmt: skip
    if ccrs is not None:
        ax.coastlines(linewidth=0.5, color=ink)
        ax.spines["geo"].set_edgecolor(ink)
    else:
        for spine in ax.spines.values():
            spine.set_edgecolor(ink)

    if region is not None:
        ring_lat, ring_lon = _break_at_dateline(
            *small_circle(region.lat, region.lon, region.radius_km)
        )
        ax.plot(ring_lon, ring_lat, color=ink, linewidth=1.2, **on_map)

    bar = fig.colorbar(drawn, ax=ax, shrink=0.85, pad=0.02, aspect=28)
    bar.ax.tick_params(labelsize=7, colors=ink, length=2)
    bar.outline.set_visible(False)
    if label:
        bar.set_label(label, fontsize=8, color=ink)
    if title:
        ax.set_title(title, fontsize=9, color=ink, loc="left")
    return fig


def to_png(fig: Figure, dpi: int = 130) -> bytes:
    """A figure as PNG bytes, for a report, a notebook or a web page."""
    from io import BytesIO

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    return buffer.getvalue()
