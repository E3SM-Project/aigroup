"""Many times at once: moments, a global PCA, and batches to train on."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from conftest import N_CHANNELS, N_LAT, N_LON, N_TIMES, write_latent_archive  # noqa: E402
from xaig.core.errors import RequestError  # noqa: E402
from xaig.daig.latent import (  # noqa: E402
    accumulate_moments,
    fit_pca,
    iter_batches,
    open_source,
    pca_from_moments,
)


def _stacked(source, layer):
    """Every node of every time, with its weight: what moments must agree with."""
    grid, times = source.grid(), source.info().times
    latents = np.vstack([source.load(t, layer) for t in times]).astype(np.float64)
    return latents, np.tile(grid.weights(), len(times)) / len(times)


def test_moments_are_area_weighted_over_every_time(latent_archive):
    source = open_source(latent_archive)
    moments = accumulate_moments(source, layer=2)
    latents, weights = _stacked(source, 2)
    mean = weights @ latents
    assert moments.mean == pytest.approx(mean, abs=1e-9)
    assert moments.mean[1] == pytest.approx(50.0, abs=0.01)  # the planted offset
    assert moments.std == pytest.approx(np.sqrt(weights @ (latents - mean) ** 2), abs=1e-7)
    assert moments.times == source.info().times
    assert moments.scale == pytest.approx(np.sqrt(np.mean(moments.std**2)))


def test_a_global_pca_from_moments_is_the_pca_of_all_the_nodes(latent_archive):
    source = open_source(latent_archive)
    latents, weights = _stacked(source, 2)
    direct = fit_pca(latents, 3, weights=weights)
    streamed = pca_from_moments(accumulate_moments(source, layer=2), 3)
    assert streamed.components == pytest.approx(direct.components, abs=1e-6)
    assert streamed.explained_variance_ratio == pytest.approx(direct.explained_variance_ratio)
    assert streamed.meta["fitted_on"] == {"layer": 2, "times": list(source.info().times)}
    with pytest.raises(RequestError, match=f"at most {N_CHANNELS} exist"):
        pca_from_moments(accumulate_moments(source, layer=2), N_CHANNELS + 1)


def test_moments_can_be_taken_over_some_times(latent_archive):
    source = open_source(latent_archive)
    one = accumulate_moments(source, layer=0, times=[-1])
    assert one.times == (source.info().times[-1],)
    assert one.mean == pytest.approx(source.grid().mean(source.load(-1, 0)), abs=1e-9)


def test_batches_cover_each_time_once_an_epoch_and_are_repeatable(latent_archive):
    source = open_source(latent_archive)
    batches = list(iter_batches(source, layer=1, batch_size=100, seed=7))
    assert sum(len(b) for b in batches) == N_TIMES * N_LAT * N_LON
    assert batches[0].shape == (100, N_CHANNELS) and batches[0].dtype == np.float32
    again = list(iter_batches(source, layer=1, batch_size=100, seed=7))
    assert all(np.array_equal(a, b) for a, b in zip(batches, again, strict=True))
    other = next(iter_batches(source, layer=1, batch_size=100, seed=8))
    assert not np.array_equal(batches[0], other)
    assert len(list(iter_batches(source, layer=1, batch_size=100, epochs=3))) == 3 * len(batches)


def test_nodes_are_drawn_by_area_and_never_where_the_grid_is_invalid(tmp_path):
    """The planted offset marks a row: give each node its own latitude instead."""
    mask = np.ones(N_LAT * N_LON, dtype=bool)
    mask[: N_LON * 2] = False  # the two southernmost rows are "land"
    path = write_latent_archive(tmp_path / "a", mask=mask)
    lat = np.repeat(np.linspace(-82.5, 82.5, N_LAT), N_LON)
    stored = np.load(path / "step_00.npy")
    stored[:, :, 0] = lat.astype(np.float16)
    np.save(path / "step_00.npy", stored)

    source = open_source(path)
    drawn = np.concatenate([b[:, 0] for b in iter_batches(source, layer=0, epochs=40)])
    assert drawn.min() > -60.0  # nothing from the masked rows
    polar, tropical = np.sum(np.abs(drawn) > 80.0), np.sum(np.abs(drawn) < 10.0)
    weights = source.grid().weights()
    expected = weights[np.abs(lat) > 80.0].sum() / weights[np.abs(lat) < 10.0].sum()
    assert polar / tropical == pytest.approx(expected, rel=0.15)
    assert expected < 0.2  # where unweighted draws would make it 0.5: one row against two


def test_a_transcoder_gets_the_same_nodes_at_two_layers(latent_archive):
    """Channel 4's bump grows with depth by a known factor, node for node."""
    source = open_source(latent_archive)
    inputs, targets = next(iter_batches(source, layer=0, target_layer=2, batch_size=288))
    strong = inputs[:, 4] > 1.0
    assert strong.any()
    assert targets[strong, 4] / inputs[strong, 4] == pytest.approx(3.0, abs=0.2)


def test_batches_need_something_to_draw_from(latent_archive):
    with pytest.raises(RequestError, match="batch_size"):
        next(iter_batches(open_source(latent_archive), layer=0, batch_size=0))
