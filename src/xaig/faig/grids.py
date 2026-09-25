"""Values on a grid of two axes: layers against time, longitude against time.

Two pictures recur whenever something is followed through a network and through
time. A layer-by-time panel shows where a signal lives and how it spreads -- a
storyline's correlations, or how far a perturbed run has drifted from its control.
A Hovmoller diagram shows one quantity along a latitude band, longitude against
time, where anything travelling draws tilted stripes.

Both take plain arrays, so any analysis that produces one can be drawn. Built on
``matplotlib.figure.Figure`` directly, like every figure in ``faig``.
"""

from __future__ import annotations

from collections.abc import Sequence

from xaig.core.extras import missing_extra

try:
    import numpy as np
    from matplotlib.figure import Figure
except ImportError as exc:
    raise missing_extra(exc.name or "matplotlib", "faig") from exc

from xaig.faig.maps import _DARK_INK, _INK, DARK_SURFACE


def _time_ticks(ax, tick_labels: Sequence[str] | None, n_times: int, axis: str) -> None:
    if tick_labels is None:
        return
    shown = np.unique(np.linspace(0, n_times - 1, min(n_times, 8)).round().astype(int))
    if axis == "x":
        ax.set_xticks(shown)
        ax.set_xticklabels([tick_labels[i] for i in shown], rotation=30, ha="right")
    else:
        ax.set_yticks(shown)
        ax.set_yticklabels([tick_labels[i] for i in shown])


def _frame(figsize: tuple[float, float], dark: bool):
    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot()
    if dark:
        fig.set_facecolor(DARK_SURFACE)
        ax.set_facecolor(DARK_SURFACE)
    return fig, ax


def _finish(fig, ax, image, *, ink, title, label, x_label, y_label) -> Figure:
    ax.tick_params(labelsize=7, colors=ink, length=2)
    for spine in ax.spines.values():
        spine.set_edgecolor(ink)
    if x_label:
        ax.set_xlabel(x_label, fontsize=8, color=ink)
    if y_label:
        ax.set_ylabel(y_label, fontsize=8, color=ink)
    if title:
        ax.set_title(title, fontsize=9, color=ink, loc="left")
    bar = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
    bar.ax.tick_params(labelsize=7, colors=ink)
    bar.outline.set_edgecolor(ink)
    if label:
        bar.set_label(label, fontsize=8, color=ink)
    return fig


def layer_time_figure(
    values: np.ndarray,
    *,
    layers: Sequence[int] | None = None,
    tick_labels: Sequence[str] | None = None,
    title: str | None = None,
    label: str | None = None,
    vmin: float | None = 0.0,
    vmax: float | None = None,
    cmap: str = "magma",
    dark: bool = False,
    figsize: tuple[float, float] = (7.0, 3.2),
) -> Figure:
    """``values`` is ``(n_times, n_layers)``, drawn with time across and layers up.

    Unsigned quantities (a correlation's size, a difference's RMS) suit the
    default sequential map from zero; pass ``vmin``/``vmax`` to compare panels."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected (n_times, n_layers), got {values.shape}")
    n_times, n_layers = values.shape
    names = [str(i) for i in range(n_layers)] if layers is None else [str(v) for v in layers]
    if len(names) != n_layers:
        raise ValueError(f"expected {n_layers} layer name(s), got {len(names)}")
    ink = _DARK_INK if dark else _INK
    fig, ax = _frame(figsize, dark)
    image = ax.imshow(
        values.T, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="nearest",
    )  # fmt: skip
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels(names)
    _time_ticks(ax, tick_labels, n_times, "x")
    return _finish(fig, ax, image, ink=ink, title=title, label=label, x_label=None, y_label="layer")


def hovmoller_figure(
    values: np.ndarray,
    lon: np.ndarray,
    *,
    tick_labels: Sequence[str] | None = None,
    title: str | None = None,
    label: str | None = None,
    symmetric: bool = True,
    limit: float | None = None,
    cmap: str | None = None,
    dark: bool = False,
    figsize: tuple[float, float] = (7.0, 4.2),
) -> Figure:
    """``values`` is ``(n_times, n_lon)``: longitude across, time running up.

    Columns are sorted by longitude in 0..360, so a band that crosses the
    dateline or the prime meridian draws unbroken. Signed data is centred on zero
    unless ``symmetric`` is False; ``limit`` pins the range to ``+-limit``."""
    values = np.asarray(values, dtype=np.float64)
    lon = np.asarray(lon, dtype=np.float64)
    if values.ndim != 2 or lon.shape != (values.shape[1],):
        raise ValueError(f"expected (n_times, n_lon) and (n_lon,), got {values.shape}, {lon.shape}")
    order = np.argsort(np.mod(lon, 360.0), kind="stable")
    lon_sorted = np.mod(lon, 360.0)[order]
    data = values[:, order]
    finite = data[np.isfinite(data)]
    if symmetric:
        bound = (
            limit if limit is not None else (float(np.abs(finite).max()) if finite.size else 1.0)
        )
        vmin, vmax = -bound, bound
        cmap = cmap or "RdBu_r"
    else:
        vmin = float(finite.min()) if finite.size else 0.0
        vmax = float(finite.max()) if finite.size else 1.0
        cmap = cmap or "viridis"
    ink = _DARK_INK if dark else _INK
    fig, ax = _frame(figsize, dark)
    step = float(np.median(np.diff(lon_sorted))) if lon_sorted.size > 1 else 1.0
    extent = (lon_sorted[0] - step / 2, lon_sorted[-1] + step / 2, -0.5, values.shape[0] - 0.5)
    image = ax.imshow(
        data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
        extent=extent, interpolation="nearest",
    )  # fmt: skip
    _time_ticks(ax, tick_labels, values.shape[0], "y")
    return _finish(fig, ax, image, ink=ink, title=title, label=label,
                   x_label="longitude (degrees east)", y_label=None)  # fmt: skip
