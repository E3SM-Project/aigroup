from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from xaig.daig.grid import Grid, cell_area_weights, great_circle_km  # noqa: E402


def _regular(step: float = 1.0, mask=None) -> Grid:
    lat_1d = np.arange(-90 + step / 2, 90, step)
    lon_1d = np.arange(0, 360, step)
    lon, lat = np.meshgrid(lon_1d, lat_1d)
    return Grid(lat.ravel(), lon.ravel(), shape=(lat_1d.size, lon_1d.size), mask=mask)


def test_an_unweighted_mean_on_a_latlon_grid_is_simply_wrong():
    """The sphere's mean of sin^2(lat) is exactly 1/3; rows crowd the poles."""
    grid = _regular()
    field = np.sin(np.radians(grid.lat)) ** 2
    assert grid.mean(field) == pytest.approx(1 / 3, abs=1e-4)
    assert field.mean() == pytest.approx(1 / 2, abs=1e-4)


def test_band_areas_cover_the_sphere_in_either_latitude_order():
    lat = np.linspace(-82.5, 82.5, 12)
    assert cell_area_weights(lat).sum() == pytest.approx(2.0)
    assert cell_area_weights(lat[::-1]) == pytest.approx(cell_area_weights(lat)[::-1])


def test_a_row_on_the_pole_still_has_area():
    """cos(lat) gives it none, and whatever sits there then never counts."""
    assert cell_area_weights(np.linspace(-90, 90, 19))[0] > 0.0


def test_band_areas_track_gaussian_quadrature_weights():
    x, exact = np.polynomial.legendre.leggauss(64)
    ours = cell_area_weights(np.degrees(np.arcsin(x)))
    relative = np.abs(ours / ours.sum() - exact / exact.sum()) / (exact / exact.sum())
    assert relative[1:-1].max() < 1e-3  # the two polar rows are the documented exception
    assert relative.max() < 0.07


def test_invalid_nodes_never_count_whatever_they_hold():
    mask = np.ones(180 * 360, dtype=bool)
    mask[:1000] = False
    grid = _regular(mask=mask)
    field = np.where(mask, 2.0, np.nan)
    assert grid.mean(field) == pytest.approx(2.0)
    assert grid.weights()[~mask].sum() == 0.0 and grid.weights().sum() == pytest.approx(1.0)


def test_the_mean_runs_along_nodes_and_keeps_channels():
    grid = _regular(10.0)
    field = np.stack([np.full(grid.n_nodes, 3.0), np.full(grid.n_nodes, -1.0)], axis=1)
    assert grid.mean(field) == pytest.approx([3.0, -1.0])


def test_a_mesh_weights_uniformly_unless_told_its_areas():
    lat, lon = np.array([0.0, 10.0, 20.0]), np.array([0.0, 0.0, 0.0])
    assert Grid(lat, lon).weights() == pytest.approx([1 / 3] * 3)
    assert Grid(lat, lon, area=np.array([2.0, 1.0, 1.0])).weights() == pytest.approx(
        [0.5, 0.25, 0.25]
    )


def test_great_circle_distance():
    quarter = great_circle_km(np.array([90.0]), np.array([0.0]), 0.0, 0.0)[0]
    assert quarter == pytest.approx(np.pi * 6371.0 / 2)


def test_a_region_crosses_the_dateline_in_either_longitude_convention():
    for lon_1d in (np.arange(0, 360, 5.0), np.arange(-180, 180, 5.0)):
        lon, lat = np.meshgrid(lon_1d, np.array([-5.0, 0.0, 5.0]))
        grid = Grid(lat.ravel(), lon.ravel(), shape=lat.shape)
        found = np.sort(grid.lon[grid.within(0.0, 180.0, 700.0)] % 360)
        assert {175.0, 180.0, 185.0} <= set(found.tolist())


def test_regions_and_nearest_skip_invalid_nodes():
    lat, lon = np.array([0.0, 1.0, 2.0]), np.zeros(3)
    grid = Grid(lat, lon, mask=np.array([False, True, True]))
    assert grid.within(0.0, 0.0, 500.0).tolist() == [1, 2]
    assert grid.nearest(0.0, 0.0) == 1


def test_maps_are_for_structured_grids_and_blank_out_invalid_nodes():
    mask = np.array([True, False, True, True])
    grid = Grid(np.array([0.0, 0, 1, 1]), np.array([0.0, 1, 0, 1]), shape=(2, 2), mask=mask)
    out = grid.to_map(np.arange(4.0))
    assert out.shape == (2, 2) and np.isnan(out[0, 1]) and out[1, 1] == 3.0
    with pytest.raises(ValueError, match="mesh"):
        Grid(np.zeros(4), np.zeros(4)).to_map(np.zeros(4))


def test_inconsistent_grids_are_refused():
    with pytest.raises(ValueError, match="does not hold"):
        Grid(np.zeros(5), np.zeros(5), shape=(2, 2))
    with pytest.raises(ValueError, match="mask has shape"):
        Grid(np.zeros(4), np.zeros(4), mask=np.ones(3, dtype=bool))


def test_nodes_stored_the_other_way_round_are_refused_not_misweighted():
    """Reshaped without complaint, a (lon, lat) ordering would give every row the
    weight of a meridian."""
    lon, lat = np.meshgrid(np.arange(0, 360, 30.0), np.linspace(-75, 75, 6))
    assert Grid(lat.ravel(), lon.ravel(), shape=lat.shape).weights().sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match=r"must be \(n_lat, n_lon\) in C order"):
        Grid(lat.T.ravel(), lon.T.ravel(), shape=lat.shape)
    with pytest.raises(ValueError, match="a mesh that should have no shape"):
        Grid(np.random.default_rng(0).uniform(-90, 90, 72), lon.ravel(), shape=lat.shape)
