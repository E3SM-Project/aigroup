"""faig -- reusable figures, independent of any web framework.

    from xaig.faig import map_figure

    fig = map_figure(source.grid(), result.similarity, region=region, title="similarity")
    fig.savefig("similarity.png")

Needs the ``faig`` extra (matplotlib); add ``maps`` (cartopy) for coastlines.
"""

from __future__ import annotations

from xaig.faig.maps import have_coastlines, map_figure, to_png

__all__ = ["have_coastlines", "map_figure", "to_png"]
