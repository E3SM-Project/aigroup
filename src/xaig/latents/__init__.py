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
- ``evaluate``  a frozen basis on held-out times: splits, fidelity and sparsity, dead and
                redundant features, stability across seeds, PCA against a dictionary
- ``steering`` steering: edit a feature inside a running model, four arms, paired noise,
                the response against random directions
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
        basis_hash,
        basis_provenance,
        bspline_activation,
        fit_pca,
        load_basis,
        result_provenance,
        save_basis,
        spline_knots,
        top_loadings,
    )
    from xaig.latents.evaluate import (
        Evaluation,
        FidelityCurve,
        Split,
        Stability,
        evaluate_basis,
        fidelity_curve,
        save_result,
        seed_stability,
        split_archives,
        split_groups,
        split_time_blocks,
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
    from xaig.latents.steering import (
        Effect,
        Hook,
        Intervenable,
        Pairing,
        Rollout,
        Run,
        Steer,
        SteeringResult,
        feature_direction,
        open_intervenable,
        run_steering,
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
    "basis_hash": "basis",
    "basis_provenance": "basis",
    "bspline_activation": "basis",
    "result_provenance": "basis",
    "fit_pca": "basis",
    "load_basis": "basis",
    "save_basis": "basis",
    "spline_knots": "basis",
    "top_loadings": "basis",
    "Effect": "steering",
    "Hook": "steering",
    "Intervenable": "steering",
    "Pairing": "steering",
    "Rollout": "steering",
    "Run": "steering",
    "Steer": "steering",
    "SteeringResult": "steering",
    "feature_direction": "steering",
    "open_intervenable": "steering",
    "run_steering": "steering",
    "Evaluation": "evaluate",
    "FidelityCurve": "evaluate",
    "Split": "evaluate",
    "Stability": "evaluate",
    "evaluate_basis": "evaluate",
    "fidelity_curve": "evaluate",
    "save_result": "evaluate",
    "seed_stability": "evaluate",
    "split_archives": "evaluate",
    "split_groups": "evaluate",
    "split_time_blocks": "evaluate",
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
    "Effect",
    "Evaluation",
    "FeatureCensus",
    "FeatureProfile",
    "FidelityCurve",
    "FieldRanking",
    "FieldStoryline",
    "Hovmoller",
    "Hook",
    "Intervenable",
    "LatentInfo",
    "LatentSource",
    "LayerInfo",
    "Moments",
    "PairedDifference",
    "Pairing",
    "ReferenceFields",
    "AnyRegion",
    "Box",
    "Region",
    "RegionAnalysis",
    "RegionSeries",
    "Rollout",
    "Run",
    "Split",
    "Stability",
    "Steer",
    "SteeringResult",
    "accumulate_moments",
    "analyse_region",
    "basis_hash",
    "basis_provenance",
    "bspline_activation",
    "check_comparable",
    "correlate_field",
    "cosine_similarity",
    "difference",
    "difference_growth",
    "evaluate_basis",
    "feature_census",
    "feature_direction",
    "feature_profile",
    "fidelity_curve",
    "field_storyline",
    "hovmoller",
    "fit_pca",
    "iter_batches",
    "load_basis",
    "load_channels",
    "open_intervenable",
    "open_source",
    "parse_time",
    "pca_from_moments",
    "rank_by_field",
    "rank_channels",
    "read_latents",
    "region_series",
    "result_provenance",
    "run_steering",
    "save_basis",
    "save_result",
    "seed_stability",
    "spline_knots",
    "split_archives",
    "split_groups",
    "split_time_blocks",
    "top_loadings",
]
