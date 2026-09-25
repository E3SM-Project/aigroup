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
- ``samples``   many times at once: moments, a global PCA, batches to train on
- ``through``   through time and between runs: series, differences, field correlation,
                storylines, Hovmoller diagrams
- ``features``  what a feature is, without a field in mind: a census of a layer, and one
                feature's profile against every field
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
    node_normalise,
    save_basis,
    spline_knots,
    top_loadings,
)
from xaig.daig.latent.features import (
    FeatureCensus,
    FeatureProfile,
    feature_census,
    feature_profile,
)
from xaig.daig.latent.samples import (
    Moments,
    NodeNormalised,
    accumulate_moments,
    iter_batches,
    pca_from_moments,
)
from xaig.daig.latent.source import (
    LatentInfo,
    LatentSource,
    LayerInfo,
    ReferenceFields,
    check_comparable,
    open_source,
    parse_time,
)
from xaig.daig.latent.through import (
    DifferenceGrowth,
    FieldRanking,
    FieldStoryline,
    Hovmoller,
    PairedDifference,
    RegionSeries,
    correlate_field,
    difference,
    difference_growth,
    field_storyline,
    hovmoller,
    rank_by_field,
    region_series,
)

__all__ = [
    "PCA",
    "ChannelRanking",
    "Decomposition",
    "Dictionary",
    "DifferenceGrowth",
    "FeatureCensus",
    "FeatureProfile",
    "FieldRanking",
    "FieldStoryline",
    "Hovmoller",
    "LatentInfo",
    "LatentSource",
    "LayerInfo",
    "Moments",
    "NodeNormalised",
    "PairedDifference",
    "ReferenceFields",
    "Region",
    "RegionAnalysis",
    "RegionSeries",
    "accumulate_moments",
    "analyse_region",
    "bspline_activation",
    "check_comparable",
    "correlate_field",
    "cosine_similarity",
    "difference",
    "difference_growth",
    "feature_census",
    "feature_profile",
    "field_storyline",
    "hovmoller",
    "fit_pca",
    "iter_batches",
    "load_basis",
    "load_channels",
    "node_normalise",
    "open_source",
    "parse_time",
    "pca_from_moments",
    "rank_by_field",
    "rank_channels",
    "region_series",
    "save_basis",
    "spline_knots",
    "top_loadings",
]
