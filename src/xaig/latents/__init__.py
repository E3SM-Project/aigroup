"""latents -- what a model's internal channels respond to, on a grid.

    from xaig.latents import Region, analyse_region, open_source

    source = open_source("latents/atmosphere")
    result = analyse_region(
        source, time=0, layer=8, region=Region(lat=5, lon=-140, radius_km=1500), n_components=4
    )
    result.ranking.channels      # which channels respond in the region
    result.similarity            # per node: where else the model looks like this
    source.grid().to_map(result.scores[:, 0])   # the first component, as a map

Reading is an adapter's job (see ``LatentSource``); this package computes.

- ``grid``      nodes on a sphere: masks, area weights, regions, maps
- ``source``    the contract: ``LatentSource``, and the optional ``ReferenceFields``
- ``basis``     ``Decomposition``: PCA, a sparse ``Dictionary``, and their file
- ``analysis``  one region at one time: ranking, similarity, a decomposition
- ``samples``   many times at once: moments, a global PCA, batches to train on
- ``through``   through time and between runs: series, differences, field correlation,
                storylines, Hovmoller diagrams
- ``features``  what a feature is, without a field in mind: a census of a layer, and one
                feature's profile against every field
- ``toy``       a toy emulator in numpy, so an archive can be made with no model

Needs the ``latents`` extra (numpy). The names above are loaded on first use, not
when this package is imported: ``xaig --help`` imports ``xaig.latents.cli`` on a base
install, and that must not bring numpy with it. The modules say which extra they
need the moment they are imported.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from xaig.latents.analysis import (
        AnyRegion,
        Box,
        ChannelRanking,
        Region,
        RegionAnalysis,
        analyse_region,
        cosine_similarity,
        load_channels,
        rank_channels,
    )
    from xaig.latents.basis import (
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
    from xaig.latents.features import (
        FeatureCensus,
        FeatureProfile,
        feature_census,
        feature_profile,
    )
    from xaig.latents.samples import (
        Moments,
        accumulate_moments,
        iter_batches,
        pca_from_moments,
    )
    from xaig.latents.source import (
        LatentInfo,
        LatentSource,
        LayerInfo,
        ReferenceFields,
        check_comparable,
        open_source,
        parse_time,
        read_latents,
    )
    from xaig.latents.through import (
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

# name -> the module that defines it, imported when the name is first asked for
_LAZY = {
    "AnyRegion": "analysis",
    "Box": "analysis",
    "ChannelRanking": "analysis",
    "Region": "analysis",
    "RegionAnalysis": "analysis",
    "analyse_region": "analysis",
    "cosine_similarity": "analysis",
    "load_channels": "analysis",
    "rank_channels": "analysis",
    "Decomposition": "basis",
    "Dictionary": "basis",
    "PCA": "basis",
    "bspline_activation": "basis",
    "fit_pca": "basis",
    "load_basis": "basis",
    "save_basis": "basis",
    "spline_knots": "basis",
    "top_loadings": "basis",
    "FeatureCensus": "features",
    "FeatureProfile": "features",
    "feature_census": "features",
    "feature_profile": "features",
    "Moments": "samples",
    "accumulate_moments": "samples",
    "iter_batches": "samples",
    "pca_from_moments": "samples",
    "LatentInfo": "source",
    "LatentSource": "source",
    "LayerInfo": "source",
    "ReferenceFields": "source",
    "check_comparable": "source",
    "open_source": "source",
    "parse_time": "source",
    "read_latents": "source",
    "DifferenceGrowth": "through",
    "FieldRanking": "through",
    "FieldStoryline": "through",
    "Hovmoller": "through",
    "PairedDifference": "through",
    "RegionSeries": "through",
    "correlate_field": "through",
    "difference": "through",
    "difference_growth": "through",
    "field_storyline": "through",
    "hovmoller": "through",
    "rank_by_field": "through",
    "region_series": "through",
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f"{__name__}.{_LAZY[name]}"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_LAZY])


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
    "PairedDifference",
    "ReferenceFields",
    "AnyRegion",
    "Box",
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
    "open_source",
    "parse_time",
    "pca_from_moments",
    "rank_by_field",
    "rank_channels",
    "read_latents",
    "region_series",
    "save_basis",
    "spline_knots",
    "top_loadings",
]
