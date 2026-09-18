from __future__ import annotations

import json
import warnings

import pytest

np = pytest.importorskip("numpy")

from conftest import BUMP, LATENT_TIMES, write_latent_archive  # noqa: E402
from xaig.daig.latent import (  # noqa: E402
    Region,
    analyse_region,
    cosine_similarity,
    fit_pca,
    open_source,
    rank_channels,
)

HERE = Region(lat=BUMP[0], lon=BUMP[1], radius_km=2500.0)

# -- the pure pieces -------------------------------------------------------


def test_channels_rank_by_peak_absolute_activation():
    latents = np.array([[0.1, -9.0, 2.0], [0.2, 1.0, -3.0]])
    ranking = rank_channels(latents, top=2)
    assert ranking.channels.tolist() == [1, 2]
    assert ranking.scores.tolist() == [9.0, 3.0]


def test_pinned_channels_lead_without_duplicating_or_growing_the_list():
    latents = np.array([[0.1, -9.0, 2.0, 5.0]])
    ranking = rank_channels(latents, top=3, pinned=[0, 0, 3])
    assert ranking.channels.tolist() == [0, 3, 1]
    assert ranking.pinned == (0, 3)
    with pytest.raises(ValueError, match="outside 0..3"):
        rank_channels(latents, pinned=[7])


def test_cosine_similarity_and_the_node_with_no_direction():
    latents = np.array([[1.0, 0.0], [-2.0, 0.0], [0.0, 3.0], [0.0, 0.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a zero vector must not leak a divide warning
        out = cosine_similarity(latents, np.array([1.0, 0.0]))
    assert out[:3].tolist() == [1.0, -1.0, 0.0]
    assert np.isnan(out[3])


def test_pca_recovers_a_planted_direction_with_a_fixed_sign():
    rng = np.random.default_rng(1)
    direction = np.array([3.0, -4.0, 0.0]) / 5.0
    latents = np.outer(rng.normal(size=400), direction) + rng.normal(scale=0.01, size=(400, 3))
    for data in (latents, -latents):  # the sign of the data must not flip the component
        pca = fit_pca(data + 10.0, n_components=2)
        assert pca.components[0] == pytest.approx(-direction, abs=1e-2)  # largest loading > 0
        assert pca.explained_variance_ratio[0] > 0.99
        assert pca.mean == pytest.approx([10.0] * 3, abs=0.2)
    assert pca.transform(data + 10.0).shape == (400, 2)
    assert pca.top_loadings(1)[0][0][0] == 1  # channel 1 weighs most


def test_a_node_weighted_twice_counts_as_two_nodes():
    rng = np.random.default_rng(2)
    latents = rng.normal(size=(30, 4))
    weights = np.ones(30)
    weights[0] = 2.0
    weighted = fit_pca(latents, 3, weights=weights)
    doubled = fit_pca(np.vstack([latents[:1], latents]), 3)
    assert weighted.components == pytest.approx(doubled.components, abs=1e-9)
    assert weighted.explained_variance_ratio == pytest.approx(doubled.explained_variance_ratio)


def test_asking_for_more_components_than_exist_says_what_to_do():
    with pytest.raises(ValueError, match="at most 3 exist -- widen the region"):
        fit_pca(np.zeros((3, 8)), n_components=5)


# -- the routine, on an archive with planted structure ---------------------


def test_the_channel_with_a_bump_in_the_region_ranks_first_once_centred(latent_archive):
    source = open_source(latent_archive)
    raw = analyse_region(source, time=0, layer=2, region=HERE, top=3)
    assert raw.ranking.channels[0] == 1  # the constant-offset channel wins uncentred...
    centred = analyse_region(source, time=0, layer=2, region=HERE, top=3, centred=True)
    assert centred.ranking.channels[0] == 4  # ...and the real response wins once it is removed


def test_channels_rank_at_the_last_layer_unless_told_otherwise(latent_archive):
    source = open_source(latent_archive)
    last = analyse_region(source, time=0, layer=0, region=HERE, centred=True)
    first = analyse_region(source, time=0, layer=0, region=HERE, centred=True, rank_layer=0)
    assert (last.settings["rank_layer"], first.settings["rank_layer"]) == (2, 0)
    assert last.ranking.scores[0] == pytest.approx(3 * first.ranking.scores[0], rel=0.1)


def test_similarity_is_highest_where_the_model_looks_like_the_region(latent_archive):
    source = open_source(latent_archive)
    result = analyse_region(source, time=0, layer=2, region=HERE, centred=True, top=2)
    grid = source.grid()
    centre = grid.nearest(*BUMP)
    assert result.similarity[centre] == pytest.approx(1.0)
    far = grid.distance_km(*BUMP) > 8000
    assert np.nanmean(result.similarity[far]) < 0.5
    assert result.similarity_top.shape == result.similarity.shape == (grid.n_nodes,)


@pytest.mark.parametrize("reference", ["nearest", "mean"])
def test_the_reference_is_a_stated_policy_not_whichever_node_came_first(latent_archive, reference):
    source = open_source(latent_archive)
    kwargs = dict(time=0, layer=2, region=HERE, centred=True, reference=reference)
    a, b = analyse_region(source, **kwargs), analyse_region(source, **kwargs)
    assert a.settings["reference"] == reference
    assert np.array_equal(a.similarity, b.similarity, equal_nan=True)
    with pytest.raises(ValueError, match="nearest, mean"):
        analyse_region(source, time=0, layer=2, region=HERE, reference="first")


def test_pca_is_fitted_in_the_region_and_projected_everywhere(latent_archive):
    source = open_source(latent_archive)
    result = analyse_region(source, time=0, layer=2, region=HERE, centred=True, n_components=2)
    assert result.scores.shape == (source.grid().n_nodes, 2)
    assert result.pca.top_loadings(1)[0][0][0] == 4  # the bump is the leading pattern here
    assert analyse_region(source, time=0, layer=2, region=HERE).scores is None


def test_a_result_says_how_to_get_it_again(latent_archive):
    source = open_source(latent_archive)
    result = analyse_region(
        source, time=LATENT_TIMES[1], layer=1, region=HERE, pinned=[5], n_components=2
    )
    summary = json.loads(json.dumps(result.summary()))  # plain data, all of it
    assert summary["settings"]["time"] == LATENT_TIMES[1]
    assert summary["settings"]["region"] == {"lat": 7.5, "lon": 45.0, "radius_km": 2500.0}
    assert summary["provenance"]["checkpoint"] == "toy.ckpt"
    assert summary["ranking"][0] == {
        "channel": 5,
        "peak_abs": pytest.approx(summary["ranking"][0]["peak_abs"]),
        "pinned": True,
    }
    assert len(summary["pca"]) == 2


def test_invalid_nodes_are_excluded_from_the_region_and_blank_in_the_maps(tmp_path):
    n = 12 * 24
    mask = np.ones(n, dtype=bool)
    mask[: n // 2] = False  # the southern half is "land"
    source = open_source(write_latent_archive(tmp_path / "ocean", mask=mask))
    result = analyse_region(source, time=0, layer=2, region=HERE, centred=True, n_components=1)
    assert mask[result.nodes].all()
    assert np.isnan(result.similarity[~mask]).all() and np.isnan(result.scores[~mask]).all()
    assert np.isfinite(result.similarity[mask]).all()


def test_a_mesh_works_like_a_grid_except_for_maps(tmp_path):
    source = open_source(write_latent_archive(tmp_path / "mesh", mesh=True))
    assert source.grid().shape is None
    result = analyse_region(source, time=0, layer=2, region=HERE, centred=True)
    assert result.ranking.channels[0] == 4


def test_an_empty_region_and_an_unknown_layer_are_explained(latent_archive):
    source = open_source(latent_archive)
    with pytest.raises(ValueError, match="widen the region"):
        analyse_region(source, time=0, layer=2, region=Region(0.0, 7.0, 1.0))
    with pytest.raises(KeyError, match="layers are 0, 1, 2"):
        analyse_region(source, time=0, layer=9, region=HERE)


def test_ranking_reads_only_the_regions_nodes(latent_archive):
    """Selective reads are the point of the contract: never a whole layer for a region."""
    real = open_source(latent_archive)
    asked = []

    class Spy:
        info, grid = real.info, real.grid

        def load(self, time, layer, channels=None, nodes=None):
            asked.append((layer, None if nodes is None else len(nodes)))
            return real.load(time, layer, channels=channels, nodes=nodes)

    result = analyse_region(Spy(), time=0, layer=1, region=HERE)
    assert asked == [(2, result.nodes.size), (1, None)]
