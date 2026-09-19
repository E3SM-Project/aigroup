"""Values through time, one line per channel or feature.

Lines are categories, so they get a categorical palette -- Okabe and Ito's, which
survives the common colour-vision deficiencies -- and, past its eight colours, a
change of dash rather than a ninth hue nobody can tell from the first. Zero is
drawn, because what is plotted here is signed and "did it change sign" is usually
the question.

Built on ``matplotlib.figure.Figure`` directly, like every figure in ``faig``.
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

CATEGORICAL = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#999999",
)  # fmt: skip
_DASHES = ("-", "--", ":", "-.")


def series_figure(
    values: np.ndarray,
    *,
    x: Sequence[float] | None = None,
    tick_labels: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    dark: bool = False,
    figsize: tuple[float, float] = (7.0, 3.2),
) -> Figure:
    """``values`` is ``(n_times, n_lines)``. ``x`` places the times (hours since
    the first, say); without it they are evenly spaced and named by
    ``tick_labels``, which is the honest picture when the real spacing is unknown.
    Markers are drawn because the times are samples, and may have gaps between them.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError(f"expected (n_times, n_lines), got {values.shape}")
    n_times, n_lines = values.shape
    where = np.arange(n_times, dtype=np.float64) if x is None else np.asarray(x, dtype=np.float64)
    if where.shape != (n_times,):
        raise ValueError(f"expected {n_times} x value(s), got {where.shape}")
    names = [str(i) for i in range(n_lines)] if labels is None else [str(n) for n in labels]
    ink = _DARK_INK if dark else _INK

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot()
    if dark:
        fig.set_facecolor(DARK_SURFACE)
        ax.set_facecolor(DARK_SURFACE)
    ax.axhline(0.0, color=ink, linewidth=0.6, alpha=0.5)
    for i, name in enumerate(names):
        ax.plot(
            where, values[:, i], label=name, linewidth=1.3, marker="o", markersize=2.5,
            color=CATEGORICAL[i % len(CATEGORICAL)],
            linestyle=_DASHES[(i // len(CATEGORICAL)) % len(_DASHES)],
        )  # fmt: skip
    if x is None and tick_labels is not None:
        shown = np.unique(np.linspace(0, n_times - 1, min(n_times, 6)).round().astype(int))
        ax.set_xticks(shown)
        ax.set_xticklabels([tick_labels[i] for i in shown], rotation=20, ha="right")
    ax.tick_params(labelsize=7, colors=ink, length=2)
    ax.grid(color=ink, alpha=0.15, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(ink)
    if x_label:
        ax.set_xlabel(x_label, fontsize=8, color=ink)
    if y_label:
        ax.set_ylabel(y_label, fontsize=8, color=ink)
    if title:
        ax.set_title(title, fontsize=9, color=ink, loc="left")
    legend = ax.legend(
        fontsize=7, ncols=min(n_lines, 6), frameon=False, loc="upper left",
        bbox_to_anchor=(0.0, -0.22 if x_label or tick_labels is not None else -0.12),
    )  # fmt: skip
    for text in legend.get_texts():
        text.set_color(ink)
    return fig
