"""faig -- reusable figures, independent of any web framework.

    from xaig.faig import map_figure

    fig = map_figure(source.grid(), result.similarity, region=region, title="similarity")
    fig.savefig("similarity.png")

Needs the ``faig`` extra (matplotlib, and cartopy for coastlines).
"""

from __future__ import annotations

from xaig.faig.bars import profile_figure
from xaig.faig.grids import hovmoller_figure, layer_time_figure
from xaig.faig.maps import have_coastlines, map_figure, to_png, why_no_coastlines
from xaig.faig.series import series_figure
from xaig.faig.training import loss_figure

__all__ = [
    "have_coastlines",
    "hovmoller_figure",
    "layer_time_figure",
    "loss_figure",
    "map_figure",
    "profile_figure",
    "series_figure",
    "to_png",
    "why_no_coastlines",
]
