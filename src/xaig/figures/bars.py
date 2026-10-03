"""Named values side by side: what a feature goes with, and what it does against chance.

A profile sets one feature against many fields at once, each as a difference in
units of its own spread, so the picture is a bar per field, longest first, and the
sign is the story -- more of this field where the feature is active, or less. Two
hues from the diverging pair the maps use keep "above" and "below" apart without
a legend, and the value is printed at each bar's end for anyone reading closely.

A steering response is drawn against the random directions it is compared with
(``response_figure``): a histogram of what they did, and a line for the feature.

Built on ``matplotlib.figure.Figure`` directly, like every figure in ``figures``.
"""

from __future__ import annotations

from collections.abc import Sequence

from xaig.core.extras import missing_extra

try:
    import numpy as np
    from matplotlib.figure import Figure
except ImportError as exc:
    raise missing_extra(exc.name or "matplotlib", "figures") from exc

from xaig.figures.maps import _DARK_INK, _INK, DARK_SURFACE

_ABOVE, _BELOW = "#B2182B", "#2166AC"
# Okabe and Ito, as the lines in ``series`` are: the feature, the reconstruction, the draws.
_FEATURE, _RECON, _DRAWS = "#D55E00", "#0072B2", "#999999"


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


def response_figure(
    random_responses: Sequence[float],
    feature: float,
    reconstruction: float | None = None,
    *,
    title: str | None = None,
    label: str = "paired response, magnitude",
    dark: bool = False,
    figsize: tuple[float, float] = (5.5, 2.4),
) -> Figure:
    """A steering response against chance: the magnitudes of what random directions of the
    same size did, as a histogram, with the feature's response and (if given) the
    reconstruction arm's marked on the same axis. A feature that stands apart from the
    histogram did something a random direction does not; one inside it did not.

    Magnitudes, because the comparison is of size and not of sign; the signed values are
    in the labels."""
    draws = np.abs(np.asarray(random_responses, dtype=np.float64))
    draws = draws[np.isfinite(draws)]
    marks = [(float(abs(feature)), f"feature {feature:+.3g}", _FEATURE)]
    if reconstruction is not None:
        marks.append(
            (float(abs(reconstruction)), f"reconstruction only {reconstruction:+.3g}", _RECON)
        )
    ink = _DARK_INK if dark else _INK

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot()
    if dark:
        fig.set_facecolor(DARK_SURFACE)
        ax.set_facecolor(DARK_SURFACE)
    reach = max([draws.max() if draws.size else 0.0, *(m[0] for m in marks)]) or 1.0
    ax.hist(
        draws, bins=np.linspace(0.0, 1.05 * reach, 21), color=_DRAWS, alpha=0.8,
        label=f"{draws.size} random directions",
    )  # fmt: skip
    for at, text, color in marks:
        ax.axvline(at, color=color, linewidth=2.0, label=text)
    ax.set_xlim(0.0, 1.05 * reach)
    ax.tick_params(labelsize=7, colors=ink, length=2)
    ax.set_xlabel(label, fontsize=8, color=ink)
    ax.set_ylabel("draws", fontsize=8, color=ink)
    ax.yaxis.get_major_locator().set_params(integer=True)
    for side, spine in ax.spines.items():
        spine.set_edgecolor(ink)
        spine.set_visible(side in ("left", "bottom"))
    legend = ax.legend(fontsize=7, frameon=False)
    for text in legend.get_texts():
        text.set_color(ink)
    if title:
        ax.set_title(title, fontsize=9, color=ink, loc="left")
    return fig
