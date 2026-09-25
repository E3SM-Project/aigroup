from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("matplotlib")

from conftest import BUMP, write_latent_archive  # noqa: E402
from xaig.daig.grid import Grid, small_circle  # noqa: E402
from xaig.daig.latent import Region, load_channels, open_source  # noqa: E402
from xaig.faig import map_figure, maps, to_png  # noqa: E402


@pytest.fixture(autouse=True)
def no_coastlines(monkeypatch):
    """Cartopy would reach for the network; the fallback is what is under test."""
    monkeypatch.setenv("XAIG_NO_COASTLINES", "1")
    maps._coastlines.cache_clear()
    yield
    maps._coastlines.cache_clear()


def _mesh(figure):
    return figure.axes[0].collections[0]


def test_a_map_without_coastlines_can_say_why(monkeypatch):
    from xaig.faig import have_coastlines, why_no_coastlines

    assert not have_coastlines() and "XAIG_NO_COASTLINES" in why_no_coastlines()

    # Installed, but with no way to get its data: a compute node.
    monkeypatch.delenv("XAIG_NO_COASTLINES")
    maps._coastlines.cache_clear()
    shapereader = pytest.importorskip("cartopy.io.shapereader")

    def offline(*args, **kwargs):
        raise OSError("no route to host")

    monkeypatch.setattr(shapereader, "natural_earth", offline)
    assert not have_coastlines()
    assert "could not get its coastline data (OSError: no route to host)" in why_no_coastlines()


def test_signed_fields_get_a_scale_symmetric_about_zero(latent_archive):
    grid = open_source(latent_archive).grid()
    values = np.linspace(-1.0, 3.0, grid.n_nodes)
    drawn = _mesh(map_figure(grid, values))
    assert drawn.get_clim() == (-3.0, 3.0)
    assert drawn.get_cmap().name == maps.SIGNED_CMAP


def test_a_magnitude_gets_one_hue_and_its_own_range(latent_archive):
    grid = open_source(latent_archive).grid()
    drawn = _mesh(map_figure(grid, np.linspace(2.0, 5.0, grid.n_nodes), symmetric=False))
    assert drawn.get_clim() == (2.0, 5.0)
    assert drawn.get_cmap().name == maps.MAGNITUDE_CMAP


def test_a_dark_page_gets_a_scale_of_its_own_not_the_light_one_on_black(latent_archive):
    from matplotlib import colormaps
    from matplotlib.colors import to_rgb

    grid = open_source(latent_archive).grid()
    figure = map_figure(grid, np.linspace(-1.0, 1.0, grid.n_nodes), dark=True)
    assert to_rgb(figure.get_facecolor()) == to_rgb(maps.DARK_SURFACE)
    if maps.DARK_SIGNED_CMAP in colormaps:  # matplotlib >= 3.10
        middle = _mesh(figure).get_cmap()(0.5)
        assert _mesh(figure).get_cmap().name == maps.DARK_SIGNED_CMAP
        assert sum(middle[:3]) / 3 < 0.2  # zero recedes into the page instead of glaring


def test_a_pinned_limit_makes_maps_comparable(latent_archive):
    grid = open_source(latent_archive).grid()
    assert _mesh(map_figure(grid, np.ones(grid.n_nodes), limit=4.0)).get_clim() == (-4.0, 4.0)


def test_columns_are_drawn_west_to_east_whatever_order_the_archive_keeps():
    """Archives wrap 0..360 to -180..180 without reordering; drawn as stored, the two
    hemispheres would swap."""
    lon_1d = np.array([0.0, 90.0, -180.0, -90.0])
    lon, lat = np.meshgrid(lon_1d, np.array([-45.0, 45.0]))
    grid = Grid(lat.ravel(), lon.ravel(), shape=(2, 4))
    cells = _mesh(map_figure(grid, grid.lon.copy())).get_array().reshape(2, 4)
    assert cells[0].tolist() == [-180.0, -90.0, 0.0, 90.0]


def test_invalid_nodes_are_blank_and_the_mask_is_outlined(tmp_path):
    mask = np.ones(12 * 24, dtype=bool)
    mask[:100] = False
    grid = open_source(write_latent_archive(tmp_path / "ocean", mask=mask)).grid()
    figure = map_figure(grid, np.ones(grid.n_nodes))
    assert np.isnan(np.ma.filled(_mesh(figure).get_array(), np.nan)).sum() == 100
    assert len(figure.axes[0].collections) == 2  # the field, and the mask's outline


def test_a_mesh_is_drawn_as_points(tmp_path):
    grid = open_source(write_latent_archive(tmp_path / "mesh", mesh=True)).grid()
    assert _mesh(map_figure(grid, np.ones(grid.n_nodes))).get_offsets().shape == (grid.n_nodes, 2)


def test_a_region_is_outlined_without_a_line_the_long_way_round(latent_archive):
    grid = open_source(latent_archive).grid()
    figure = map_figure(grid, np.zeros(grid.n_nodes), region=Region(10.0, 179.0, 1500.0))
    ring_lon = figure.axes[0].lines[0].get_xdata()
    jumps = np.abs(np.diff(ring_lon))
    assert np.isnan(ring_lon).any() and np.nanmax(jumps) < 180.0


def test_the_outline_is_a_circle_on_the_sphere():
    from xaig.daig.grid import great_circle_km

    lat, lon = small_circle(60.0, -170.0, 2000.0)
    assert great_circle_km(lat, lon, 60.0, -170.0) == pytest.approx(2000.0)
    assert lon.min() >= -180.0 and lon.max() < 180.0


def test_a_missing_dependency_names_the_one_extra_that_brings_everything():
    """Not numpy's extra first and ours second."""
    import subprocess
    import sys

    code = (
        "import sys\n"
        "sys.modules['numpy'] = None\n"  # as if it were not installed
        "try:\n    import xaig.faig\nexcept ImportError as exc:\n    print(exc)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
    assert "'faig' extra" in out and "'daig' extra" not in out


def test_values_must_be_one_per_node(latent_archive):
    with pytest.raises(ValueError, match="one value per node"):
        map_figure(open_source(latent_archive).grid(), np.zeros(5))


def test_a_figure_becomes_png_bytes(latent_archive):
    grid = open_source(latent_archive).grid()
    assert to_png(map_figure(grid, np.zeros(grid.n_nodes), title="t", label="l"))[:4] == b"\x89PNG"


def test_load_channels_is_ready_to_map(tmp_path):
    mask = np.arange(12 * 24) % 5 != 0
    source = open_source(write_latent_archive(tmp_path / "a", mask=mask))
    raw = load_channels(source, time=0, layer=2, channels=[1, 4])
    assert raw.shape == (12 * 24, 2) and np.isnan(raw[~mask]).all()
    assert np.nanmean(raw[:, 0]) == pytest.approx(50.0, abs=0.1)  # the planted offset...
    centred = load_channels(source, time=0, layer=2, channels=[1, 4], centred=True)
    assert abs(source.grid().mean(np.nan_to_num(centred))[0]) < 1e-6  # ...gone
    assert np.nanargmax(centred[:, 1]) == source.grid().nearest(*BUMP)


# -- series ---------------------------------------------------------------------


def test_a_series_is_one_line_a_column_placed_in_time():
    from xaig.faig import series_figure

    values = np.array([[0.0, 1.0], [1.0, -1.0], [4.0, 0.5]])
    figure = series_figure(values, x=[0.0, 6.0, 42.0], labels=["ch 4", "ch 1"], x_label="hours")
    axes = figure.axes[0]
    zero, first, second = axes.lines
    assert first.get_xdata().tolist() == [0.0, 6.0, 42.0]  # the gap is drawn as a gap
    assert second.get_ydata().tolist() == [1.0, -1.0, 0.5] and zero.get_ydata()[0] == 0.0
    assert [t.get_text() for t in axes.get_legend().get_texts()] == ["ch 4", "ch 1"]
    assert first.get_color() != second.get_color()
    assert to_png(figure)[:4] == b"\x89PNG"


def test_without_real_times_a_series_is_evenly_spaced_and_says_which_is_which():
    from xaig.faig import series_figure

    figure = series_figure(np.zeros((3, 9)), tick_labels=["a", "b", "c"], dark=True)
    axes = figure.axes[0]
    assert [t.get_text() for t in axes.get_xticklabels()] == ["a", "b", "c"]
    styles = {(line.get_color(), line.get_linestyle()) for line in axes.lines[1:]}
    assert len(styles) == 9  # a ninth line changes its dash, not to a ninth hue
    with pytest.raises(ValueError, match="x value"):
        series_figure(np.zeros((3, 2)), x=[0.0, 1.0])


def test_fetching_coastlines_has_a_deadline_and_gives_the_old_one_back(monkeypatch):
    """Cartopy's fetch has none, and a node that drops packets never refuses."""
    import socket

    shapereader = pytest.importorskip("cartopy.io.shapereader")
    seen = []
    monkeypatch.delenv("XAIG_NO_COASTLINES")
    maps._coastlines.cache_clear()
    monkeypatch.setattr(
        shapereader, "natural_earth", lambda **kwargs: seen.append(socket.getdefaulttimeout())
    )
    before = socket.getdefaulttimeout()
    assert maps.have_coastlines()
    assert seen == [maps._FETCH_TIMEOUT_SECONDS] and socket.getdefaulttimeout() == before


def test_a_layer_by_time_panel_puts_layers_up_and_time_across():
    from xaig.faig import layer_time_figure

    values = np.arange(12.0).reshape(4, 3) / 12  # 4 times, 3 layers
    fig = layer_time_figure(values, layers=[0, 4, 8], tick_labels=list("abcd"), label="|r|")
    image = fig.axes[0].images[0]
    assert image.get_array().shape == (3, 4) and image.get_clim()[0] == 0.0
    assert [t.get_text() for t in fig.axes[0].get_yticklabels()] == ["0", "4", "8"]
    assert to_png(fig).startswith(b"\x89PNG")
    with pytest.raises(ValueError):
        layer_time_figure(values, layers=[0, 1])


def test_a_hovmoller_runs_west_to_east_across_the_prime_meridian():
    from xaig.faig import hovmoller_figure

    lon = np.array([90.0, 180.0, 270.0, 0.0])  # the archive's order, not the map's
    values = np.tile(lon, (5, 1))
    fig = hovmoller_figure(values, lon, symmetric=False)
    drawn = fig.axes[0].images[0].get_array()
    assert list(drawn[0]) == [0.0, 90.0, 180.0, 270.0]
    left, right, _, _ = fig.axes[0].images[0].get_extent()
    assert left < 0.0 < 270.0 < right
    signed = hovmoller_figure(values - 135.0, lon)
    low, high = signed.axes[0].images[0].get_clim()
    assert low == -high
    with pytest.raises(ValueError):
        hovmoller_figure(values, lon[:3])
