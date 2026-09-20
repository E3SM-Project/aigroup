"""Bases: a PCA and a sparse dictionary are used the same way."""

from __future__ import annotations

import json
import warnings

import pytest

np = pytest.importorskip("numpy")

from conftest import BUMP  # noqa: E402
from xaig.core.errors import RequestError  # noqa: E402
from xaig.daig.latent import (  # noqa: E402
    Decomposition,
    Dictionary,
    Region,
    accumulate_moments,
    analyse_region,
    bspline_activation,
    cosine_similarity,
    fit_pca,
    load_basis,
    open_source,
    pca_from_moments,
    rank_channels,
    save_basis,
    spline_knots,
    top_loadings,
)

HERE = Region(lat=BUMP[0], lon=BUMP[1], radius_km=2500.0)


def _dictionary(**kwargs) -> Dictionary:
    """Three features over two channels, small enough to work out by hand:
    inputs are standardised as (x - [1, 0]) / 2."""
    weights = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    given = dict(
        encoder=weights,
        encoder_bias=np.array([0.0, -1.0, 0.0]),
        decoder=weights,
        decoder_bias=np.zeros(2),
        input_mean=np.array([1.0, 0.0]),
        input_scale=2.0,
    )
    return Dictionary(**{**given, **kwargs})


def test_a_pca_and_a_dictionary_are_both_decompositions():
    pca = fit_pca(np.random.default_rng(0).normal(size=(20, 3)), 2)
    for basis, shape in ((pca, (2, 3)), (_dictionary(), (3, 2))):
        assert isinstance(basis, Decomposition)
        assert (basis.n_features, basis.n_channels) == shape == basis.directions().shape


def test_a_dictionary_standardises_encodes_and_rebuilds():
    x = np.array([[5.0, 4.0]])  # standardised: (2, 2); pre-activations: 2, 1, -2
    relu = _dictionary()
    assert relu.transform(x).tolist() == [[2.0, 1.0, 0.0]]
    assert relu.reconstruct(x).tolist() == [[5.0, 2.0]]
    assert relu.transform(x, features=[1]).tolist() == [[1.0]]
    assert top_loadings(relu, [2], n=1) == [[(0, -1.0)]]


def test_topk_features_compete_so_a_subset_is_not_computed_alone():
    top1 = _dictionary(activation="topk", k=1)
    assert top1.transform(np.array([[5.0, 4.0]])).tolist() == [[2.0, 0.0, 0.0]]
    assert top1.transform(np.array([[5.0, 4.0]]), features=[1]).tolist() == [[0.0]]


def test_a_spline_at_its_knots_is_a_relu_and_can_be_bent_from_there():
    z = np.array([[-2.0, 0.0, 0.3, 2.71, 5.999, 6.0, 9.0]]).T @ np.ones((1, 3))
    knots = np.tile(spline_knots(8, 6.0), (3, 1))
    assert bspline_activation(z, knots, 6.0) == pytest.approx(np.maximum(z, 0.0), abs=1e-12)
    flat = np.zeros_like(knots)  # a dead zone up to 6, a line of slope one beyond
    assert bspline_activation(np.array([[3.0, 8.0, -1.0]]), flat, 6.0).tolist() == [[0.0, 2.0, 0.0]]
    splined = _dictionary(activation="bspline", spline=knots, spline_upper=6.0)
    assert splined.transform(np.array([[5.0, 4.0]])) == pytest.approx(np.array([[2.0, 1.0, 0.0]]))


def test_an_inconsistent_dictionary_is_refused():
    with pytest.raises(ValueError, match="needs 1 <= k"):
        _dictionary(activation="topk")
    with pytest.raises(ValueError, match="spline coefficients"):
        _dictionary(activation="bspline")
    with pytest.raises(RequestError, match="reads 2 channel"):
        _dictionary().transform(np.zeros((4, 5)))
    with pytest.raises(RequestError, match=r"outside 0\.\.2"):
        _dictionary().transform(np.zeros((4, 2)), features=[3])


@pytest.mark.parametrize("kind", ["pca", "relu", "topk", "bspline"])
def test_a_basis_survives_its_file(tmp_path, kind):
    if kind == "pca":
        basis = fit_pca(np.random.default_rng(0).normal(size=(20, 3)), 2)
    else:
        extra = {"topk": {"k": 2}, "bspline": {"spline": np.ones((3, 11)), "spline_upper": 4.0}}
        basis = _dictionary(activation=kind, **extra.get(kind, {}))
    path = save_basis(tmp_path / "deep" / "basis.npz", basis, note="why it exists")
    back = load_basis(path)
    x = np.random.default_rng(1).normal(size=(7, basis.n_channels))
    assert type(back) is type(basis)
    assert np.array_equal(back.transform(x), basis.transform(x))
    assert back.meta == {"note": "why it exists", "path": str(path)}
    # The model's environment needs only numpy to find a direction to steer along.
    with np.load(path) as plain:
        name = "components" if kind == "pca" else "decoder"
        assert np.array_equal(plain[name], basis.directions())


def test_what_is_not_a_basis_file_says_so(tmp_path):
    with pytest.raises(RequestError, match="no basis file"):
        load_basis(tmp_path / "nothing.npz")
    other = tmp_path / "other.npz"
    np.savez(other, lat=np.zeros(3))
    with pytest.raises(RequestError, match="not a basis file written by xaig"):
        load_basis(other)


# -- the things the first review found --------------------------------------


def test_the_mean_costs_a_degree_of_freedom():
    """Three nodes span two directions; a third component would be arbitrary."""
    rng = np.random.default_rng(3)
    assert fit_pca(rng.normal(size=(3, 8)), 2).explained_variance_ratio.sum() == pytest.approx(1.0)
    with pytest.raises(RequestError, match="at most 2 exist"):
        fit_pca(rng.normal(size=(3, 8)), 3)
    weights = np.array([1.0, 1.0, 1.0, 0.0])  # and a node of no weight is no node
    with pytest.raises(RequestError, match="3 node"):
        fit_pca(rng.normal(size=(4, 8)), 3, weights=weights)


def test_a_channel_that_is_nowhere_a_number_ranks_last_not_first():
    latents = np.array([[np.nan, 1.0, -3.0], [np.nan, 2.0, np.nan]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ranking = rank_channels(latents, top=3)
    assert ranking.channels.tolist() == [2, 1, 0]
    assert ranking.scores[:2].tolist() == [3.0, 2.0] and np.isnan(ranking.scores[2])


def test_half_precision_is_widened_before_it_can_overflow():
    """An archive keeps float16, and 384 squares of 50 do not fit in one."""
    latents = np.full((2, 384), 50.0, dtype=np.float16)
    assert cosine_similarity(latents, latents[0]) == pytest.approx([1.0, 1.0])
    pca = fit_pca(np.random.default_rng(0).normal(size=(10, 384)), 2)
    assert np.isfinite(pca.transform(latents)).all()


# -- a basis in the routine ---------------------------------------------------


@pytest.fixture
def global_pca(latent_archive):
    return pca_from_moments(accumulate_moments(open_source(latent_archive), layer=2), 3)


def test_a_given_basis_maps_its_features_that_respond_in_the_region(latent_archive, global_pca):
    source = open_source(latent_archive)
    result = analyse_region(source, time=0, layer=2, region=HERE, n_components=2, basis=global_pca)
    first = result.feature_info[0]
    assert first["label"] == "F0" and first["top_loadings"][0]["channel"] == 4  # the bump
    assert first["peak_abs"] >= result.feature_info[1]["peak_abs"]
    assert result.scores.shape == (source.grid().n_nodes, 2) and result.pca is None
    assert result.summary()["features"] == list(result.feature_info)


def test_a_basis_sees_raw_latents_whatever_the_analysis_centres(latent_archive, global_pca):
    """It carries the standardisation it was fitted with; centring twice is wrong."""
    source = open_source(latent_archive)
    kwargs = dict(time=0, layer=2, region=HERE, n_components=2, basis=global_pca)
    raw, centred = analyse_region(source, **kwargs), analyse_region(source, centred=True, **kwargs)
    assert np.array_equal(raw.scores, centred.scores)
    assert raw.ranking.channels[0] == 1 and centred.ranking.channels[0] == 4  # unlike the ranking


def test_a_basis_for_another_layer_is_refused_before_anything_is_read(latent_archive):
    source = open_source(latent_archive)
    narrow = fit_pca(np.random.default_rng(0).normal(size=(10, 3)), 2)
    with pytest.raises(RequestError, match="reads 3 channel"):
        analyse_region(source, time=0, layer=2, region=HERE, n_components=1, basis=narrow)
    wide = pca_from_moments(accumulate_moments(source, layer=2), 3)
    with pytest.raises(RequestError, match="at most 3 exist"):
        analyse_region(source, time=0, layer=2, region=HERE, n_components=4, basis=wide)


# -- what a second review found ---------------------------------------------


def test_topk_keeps_exactly_k_and_a_tie_goes_to_the_lowest_index():
    from xaig.daig.latent.basis import topk_mask

    z = np.array([[3.0, 3.0, 3.0, 3.0], [5.0, 3.0, 3.0, 3.0], [0.0, 2.0, 0.0, 2.0]])
    assert topk_mask(z, 1).tolist() == [[1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0]]
    assert topk_mask(z, 2).tolist() == [[1, 1, 0, 0], [1, 1, 0, 0], [0, 1, 0, 1]]
    eye = np.eye(4)
    top1 = Dictionary(
        encoder=eye, encoder_bias=np.zeros(4), decoder=eye, decoder_bias=np.zeros(4),
        input_mean=np.zeros(4), activation="topk", k=1,
    )  # fmt: skip
    assert top1.transform(np.full((1, 4), 3.0)).tolist() == [[3.0, 0.0, 0.0, 0.0]]


def test_a_basis_is_refused_where_it_was_not_fitted(latent_archive):
    """Every layer of a model is as wide as the next, and so is every seed of a
    campaign: width matching was the only check, and is no check."""
    from dataclasses import replace

    source = open_source(latent_archive)
    here = {"fitted_on": {"layer": 2, "provenance": source.info().provenance()}}
    signed = replace(fit_pca(np.random.default_rng(0).normal(size=(40, 6)), 2), meta=here)
    kwargs = dict(time=0, region=HERE, n_components=1, basis=signed)
    assert analyse_region(source, layer=2, **kwargs).scores is not None
    with pytest.raises(RequestError, match="layer 1 against the layer 2 it was fitted on"):
        analyse_region(source, layer=1, rank_layer=1, **kwargs)

    manifest = json.loads((latent_archive / "manifest.json").read_text())
    manifest["checkpoint"] = "another-seed.ckpt"
    (latent_archive / "manifest.json").write_text(json.dumps(manifest))
    elsewhere = open_source(latent_archive)
    with pytest.raises(RequestError, match="checkpoint 'another-seed.ckpt' against 'toy.ckpt'"):
        analyse_region(elsewhere, layer=2, **kwargs)
    unsigned = replace(signed, meta={})
    with pytest.raises(RequestError, match="compatibility is unverified"):
        analyse_region(elsewhere, layer=2, **{**kwargs, "basis": unsigned})
    overridden = analyse_region(
        elsewhere, layer=2, **{**kwargs, "basis": unsigned}, allow_unverified_basis=True
    )
    assert overridden.scores is not None
    assert overridden.settings["allow_unverified_basis"] is True
    with pytest.raises(RequestError, match="checkpoint"):
        analyse_region(elsewhere, layer=2, **kwargs, allow_unverified_basis=True)


def test_features_rank_by_what_they_contribute_not_by_a_dictionarys_bookkeeping(latent_archive):
    """Two features read the bump's channel. One is loud with a short direction,
    the other quiet with a long one; what each adds to the layer is what counts."""
    loud, quiet = np.zeros((2, 6)), np.zeros((2, 6))
    loud[0, 4], quiet[0, 4] = 10.0, 0.01  # feature 0: activation x10, direction /100
    loud[1, 4], quiet[1, 4] = 1.0, 1.0  # feature 1: as it comes
    basis = Dictionary(
        encoder=loud, encoder_bias=np.zeros(2), decoder=quiet, decoder_bias=np.zeros(6),
        input_mean=np.zeros(6),
    )  # fmt: skip
    result = analyse_region(
        open_source(latent_archive),
        time=0,
        layer=2,
        region=HERE,
        n_components=2,
        basis=basis,
        allow_unverified_basis=True,
    )
    first, second = result.feature_info
    assert (first["feature"], second["feature"]) == (1, 0)
    assert second["peak_abs"] == pytest.approx(10 * first["peak_abs"], rel=1e-5)
    assert first["peak_contribution"] == pytest.approx(10 * second["peak_contribution"], rel=1e-5)


@pytest.mark.parametrize("centred", [False, True])
def test_region_basis_round_trips_raw_inputs_with_identity(latent_archive, tmp_path, centred):
    source = open_source(latent_archive)
    kwargs = dict(time=0, layer=2, region=HERE, n_components=2)
    result = analyse_region(source, **kwargs, centred=centred)
    raw = source.load(0, 2)
    valid = source.grid().valid
    restored = load_basis(save_basis(tmp_path / "region.npz", result.pca))
    np.testing.assert_allclose(restored.transform(raw)[valid], result.scores[valid], atol=1e-12)
    raw_result = analyse_region(source, **kwargs, centred=False)
    np.testing.assert_allclose(result.scores, raw_result.scores, atol=1e-12)
    fitted = restored.meta["fitted_on"]
    assert fitted["layer"] == 2
    assert fitted["time"] == source.info().times[0]
    assert fitted["region"] == result.settings["region"]
    assert fitted["provenance"] == source.info().provenance()
    reused = analyse_region(source, **kwargs, basis=restored, centred=centred)
    order = [feature["feature"] for feature in reused.feature_info]
    np.testing.assert_allclose(reused.scores, result.scores[:, order], atol=1e-12)
    with pytest.raises(RequestError, match="layer 0"):
        analyse_region(source, **{**kwargs, "layer": 0}, basis=restored)


def test_pca_projection_retains_small_float32_variation_across_blocks():
    from xaig.daig.latent.basis import PCA

    direction = np.array([[1.0, 1.0]]) / np.sqrt(2.0)
    basis = PCA(np.array([1e8, 1e8]), direction, np.ones(1))
    x = np.tile(np.array([[1e8, 1e8], [1e8 + 8, 1e8]], dtype=np.float32), (2050, 1))
    before = x.copy()
    expected = (x.astype(np.float64) - basis.mean) @ direction.T
    np.testing.assert_allclose(basis.transform(x), expected, atol=1e-12)
    np.testing.assert_allclose(basis.transform(x, features=[0]), expected, atol=1e-12)
    np.testing.assert_array_equal(x, before)


def test_pca_ignores_nonfinite_zero_weight_nodes():
    x = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])
    expected = fit_pca(x, 2)
    padded = np.vstack([x, [np.nan, np.inf]])
    actual = fit_pca(padded, 2, weights=np.array([1.0, 1.0, 1.0, 0.0]))
    np.testing.assert_allclose(actual.mean, expected.mean)
    np.testing.assert_allclose(actual.components, expected.components)
    np.testing.assert_allclose(actual.explained_variance_ratio, expected.explained_variance_ratio)
    with pytest.raises(RequestError, match="positive-weight"):
        fit_pca(padded, 2)


@pytest.mark.parametrize("missing", ["layer", "model", "component", "checkpoint"])
def test_incomplete_basis_identity_refused_before_loading(latent_archive, missing):
    from dataclasses import replace

    source = open_source(latent_archive)
    fitted = {"layer": 2, "provenance": source.info().provenance()}
    if missing == "layer":
        del fitted[missing]
    else:
        del fitted["provenance"][missing]
    basis = replace(fit_pca(np.arange(24.0).reshape(4, 6), 1), meta={"fitted_on": fitted})

    class Unreadable:
        info, grid = source.info, source.grid

        def load(self, *args, **kwargs):
            pytest.fail("invalid request loaded latents")

    with pytest.raises(RequestError, match=f"missing {missing}"):
        analyse_region(Unreadable(), time=0, layer=2, region=HERE, basis=basis)


def test_missing_source_identity_needs_override(latent_archive):
    from dataclasses import replace

    from xaig.daig.latent.source import check_basis_fits

    source = open_source(latent_archive)
    result = analyse_region(source, time=0, layer=2, region=HERE, n_components=1)
    info = replace(source.info(), checkpoint=None)
    with pytest.raises(RequestError, match="missing checkpoint"):
        check_basis_fits(result.pca, info, 2)
    check_basis_fits(result.pca, info, 2, allow_unverified=True)
    with pytest.raises(RequestError, match="layer 0"):
        check_basis_fits(result.pca, info, 0, allow_unverified=True)


def test_cli_requires_explicit_unverified_basis_override(latent_archive, tmp_path):
    from click.testing import CliRunner

    from xaig._cli import cli

    basis = fit_pca(np.random.default_rng(0).normal(size=(20, 6)), 2)
    path = save_basis(tmp_path / "anonymous.npz", basis)
    args = [
        "daig",
        "latent",
        "region",
        str(latent_archive),
        "--basis",
        str(path),
        "--lat",
        str(BUMP[0]),
        "--lon",
        str(BUMP[1]),
        "--radius-km",
        "2500",
        "--pcs",
        "1",
        "--json",
    ]
    rejected = CliRunner().invoke(cli, args)
    assert rejected.exit_code == 1 and "--allow-unverified-basis" in rejected.output
    allowed = CliRunner().invoke(cli, [*args, "--allow-unverified-basis"])
    assert allowed.exit_code == 0, allowed.output
    assert json.loads(allowed.output)["settings"]["allow_unverified_basis"] is True
