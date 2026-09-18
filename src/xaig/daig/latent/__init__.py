"""Latent-space diagnostics: what a model's internal channels respond to.

    from xaig.daig.latent import Region, analyse_region, open_source

    source = open_source("latents/atmosphere")
    result = analyse_region(
        source, time=0, layer=8, region=Region(lat=5, lon=-140, radius_km=1500), n_components=4
    )
    result.ranking.channels      # which channels respond in the region
    result.similarity            # per node: where else the model looks like this
    source.grid().to_map(result.scores[:, 0])   # the first component, as a map

Reading is an adapter's job (see ``LatentSource``); this package computes.
"""

from __future__ import annotations

from xaig.daig.latent.analysis import (
    PCA,
    ChannelRanking,
    Region,
    RegionAnalysis,
    analyse_region,
    cosine_similarity,
    fit_pca,
    rank_channels,
)
from xaig.daig.latent.source import LatentInfo, LatentSource, LayerInfo, open_source

__all__ = [
    "PCA",
    "ChannelRanking",
    "LatentInfo",
    "LatentSource",
    "LayerInfo",
    "Region",
    "RegionAnalysis",
    "analyse_region",
    "cosine_similarity",
    "fit_pca",
    "open_source",
    "rank_channels",
]
