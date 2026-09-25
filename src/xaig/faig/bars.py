"""Named, signed values side by side: what a feature goes with.

A profile sets one feature against many fields at once, each as a difference in
units of its own spread, so the picture is a bar per field, longest first, and the
sign is the story -- more of this field where the feature is active, or less. Two
hues from the diverging pair the maps use keep "above" and "below" apart without
a legend, and the value is printed at each bar's end for anyone reading closely.

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

_ABOVE, _BELOW = "#B2182B", "#2166AC"


def profile_figure(
    names: Sequence[str],
    values: Sequence[float],
    *,
    top: int | None = 12,
    title: str | None = None,
    label: str | None = "difference, in units of the field's spread",
    dark: bool = False,
    figsize: tuple[float, float] | None = None,
) -> Figure:
    """One horizontal bar per name, the ``top`` largest by absolute value, largest
    at the top. NaN values are left out: they say nothing about the feature."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(names),):
        raise ValueError(f"expected {len(names)} value(s), got {values.shape}")
    keep = np.flatnonzero(np.isfinite(values))
    keep = keep[np.argsort(np.abs(values[keep]), kind="stable")[::-1]]
    if top is not None:
        keep = keep[: max(top, 0)]
    shown = values[keep][::-1]
    labels = [str(names[i]) for i in keep][::-1]
    ink = _DARK_INK if dark else _INK

    fig = Figure(figsize=figsize or (5.5, 0.9 + 0.28 * max(len(keep), 1)), layout="constrained")
    ax = fig.add_subplot()
    if dark:
        fig.set_facecolor(DARK_SURFACE)
        ax.set_facecolor(DARK_SURFACE)
    where = np.arange(len(shown))
    ax.barh(where, shown, height=0.7, color=[_ABOVE if v > 0 else _BELOW for v in shown])
    ax.axvline(0.0, color=ink, linewidth=0.6, alpha=0.6)
    reach = float(np.max(np.abs(shown))) if shown.size else 1.0
    ax.set_xlim(-1.25 * reach, 1.25 * reach)
    for y, v in zip(where, shown, strict=True):
        ax.text(
            v + np.sign(v) * 0.03 * reach, y, f"{v:+.2f}", va="center",
            ha="left" if v > 0 else "right", fontsize=7, color=ink,
        )  # fmt: skip
    ax.set_yticks(where)
    ax.set_yticklabels(labels)
    ax.tick_params(labelsize=7, colors=ink, length=2)
    ax.grid(axis="x", color=ink, alpha=0.15, linewidth=0.5)
    for side, spine in ax.spines.items():
        spine.set_edgecolor(ink)
        spine.set_visible(side in ("left", "bottom"))
    if label:
        ax.set_xlabel(label, fontsize=8, color=ink)
    if title:
        ax.set_title(title, fontsize=9, color=ink, loc="left")
    return fig
