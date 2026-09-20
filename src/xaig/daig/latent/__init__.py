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

- ``source``    the contract: ``LatentSource``, and the optional ``ReferenceFields``
- ``basis``     ``Decomposition``: PCA, a sparse ``Dictionary``, and their file
- ``analysis``  one region at one time: ranking, similarity, a decomposition
"""

from __future__ import annotations

from xaig.daig.latent.analysis import (
    ChannelRanking,
    Region,
    RegionAnalysis,
    analyse_region,
    cosine_similarity,
    load_channels,
    rank_channels,
)
from xaig.daig.latent.basis import (
    PCA,
    Decomposition,
    Dictionary,
    bspline_activation,
    fit_pca,
    load_basis,
    save_basis,
    spline_knots,
    top_loadings,
)
from xaig.daig.latent.source import (
    LatentInfo,
    LatentSource,
    LayerInfo,
    ReferenceFields,
    open_source,
    parse_time,
)

__all__ = [
    "PCA",
    "ChannelRanking",
    "Decomposition",
    "Dictionary",
    "LatentInfo",
    "LatentSource",
    "LayerInfo",
    "ReferenceFields",
    "Region",
    "RegionAnalysis",
    "analyse_region",
    "bspline_activation",
    "cosine_similarity",
    "fit_pca",
    "load_basis",
    "load_channels",
    "open_source",
    "parse_time",
    "rank_channels",
    "save_basis",
    "spline_knots",
    "top_loadings",
]
