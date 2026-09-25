"""How a fit went: the loss curve a ``taig`` fit records in its basis file.

What is drawn is the fraction of variance left unexplained, not the raw squared
error, so that curves from different layers -- differently sized, differently
standardised -- sit on one scale, and 0.1 means the same thing on each. The
training curve is per batch and noisy, so it is drawn faint with a running mean
over it; the held-out curve is a fixed sample of times never trained on, and is
the one to believe.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from xaig.core.extras import missing_extra

try:
    import numpy as np
    from matplotlib.figure import Figure
except ImportError as exc:
    raise missing_extra(exc.name or "matplotlib", "faig") from exc

from xaig.faig.maps import _DARK_INK, _INK, DARK_SURFACE
from xaig.faig.series import CATEGORICAL


def _running_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean, shorter at the start rather than padded."""
    sums = np.cumsum(np.insert(values, 0, 0.0))
    ends = np.arange(1, values.size + 1)
    starts = np.maximum(0, ends - window)
    return (sums[ends] - sums[starts]) / (ends - starts)


def loss_figure(
    history: Mapping[str, Any],
    *,
    title: str | None = None,
    smooth: int = 40,
    dark: bool = False,
    figsize: tuple[float, float] = (7.0, 3.4),
) -> Figure:
    """``history`` is ``meta["history"]`` of a fitted dictionary: ``step`` and
    ``fraction_unexplained`` per logged batch, ``epoch`` for each, and optionally
    ``holdout`` with its own ``step`` and ``fraction_unexplained``. The y axis is
    logarithmic, where the slow end of training is still visible."""
    steps = np.asarray(history.get("step", ()), dtype=np.float64)
    unexplained = np.asarray(history.get("fraction_unexplained", ()), dtype=np.float64)
    if steps.size == 0 or steps.shape != unexplained.shape:
        raise ValueError("the history holds no training curve to draw")
    ink = _DARK_INK if dark else _INK

    fig = Figure(figsize=figsize, layout="constrained")
    ax = fig.add_subplot()
    if dark:
        fig.set_facecolor(DARK_SURFACE)
        ax.set_facecolor(DARK_SURFACE)
    epochs = np.asarray(history.get("epoch", ()), dtype=np.int64)
    if epochs.size == steps.size:
        for boundary in steps[1:][np.diff(epochs) != 0]:
            ax.axvline(boundary, color=ink, linewidth=0.6, alpha=0.35, linestyle=":")
    train = CATEGORICAL[0]
    ax.plot(steps, unexplained, color=train, linewidth=0.6, alpha=0.3)
    ax.plot(
        steps, _running_mean(unexplained, max(1, smooth)), color=train, linewidth=1.6,
        label=f"training (per batch, mean of {max(1, smooth)})",
    )  # fmt: skip
    held = history.get("holdout")
    if held and held.get("step"):
        ax.plot(
            held["step"], held["fraction_unexplained"], color=CATEGORICAL[1], linewidth=1.4,
            marker="o", markersize=3, label="held out (fixed sample)",
        )  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlabel("step", fontsize=8, color=ink)
    ax.set_ylabel("fraction of variance unexplained", fontsize=8, color=ink)
    ax.tick_params(which="both", labelsize=7, colors=ink, length=2)
    ax.grid(which="both", color=ink, alpha=0.12, linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor(ink)
    if title:
        ax.set_title(title, fontsize=9, color=ink, loc="left")
    legend = ax.legend(fontsize=7, frameon=False, loc="upper right")
    for text in legend.get_texts():
        text.set_color(ink)
    return fig
