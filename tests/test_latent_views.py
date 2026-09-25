"""Storylines, Hovmoller diagrams, and a difference set against the model's own noise."""

from __future__ import annotations

import json
import warnings

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from conftest import BUMP, LATENT_TIMES, N_LAT, N_LON, write_latent_archive  # noqa: E402
from xaig._cli import cli  # noqa: E402
from xaig.core.errors import RequestError  # noqa: E402
from xaig.daig.latent import (  # noqa: E402
    accumulate_moments,
    difference_growth,
    field_storyline,
    hovmoller,
    open_source,
    pca_from_moments,
    rank_by_field,
)


@pytest.fixture
def fields(tmp_path):
    """``warmth`` has the bump's shape, doubling each time, so channel 4 tracks it at
    every layer. ``rain`` is a diagnostic: missing at the first latent time, as a
    model's outputs are before its first step."""
    xarray = pytest.importorskip("xarray")
    path = write_latent_archive(tmp_path / "a", reference="reference.nc")
    with np.load(path / "grid.npz") as grid:
        distance = np.hypot(grid["lat"] - BUMP[0], grid["lon"] - BUMP[1])
    shape = np.exp(-((distance / 20.0) ** 2)).reshape(1, N_LAT, N_LON)
    warmth = shape * np.array([1.0, 2.0, 4.0])[:, None, None]
    rain = shape * np.array([np.nan, np.nan, 3.0])[:, None, None]
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["reference_times"] = ["0425-01-01T00:00:00", *LATENT_TIMES]
    (path / "manifest.json").write_text(json.dumps(manifest))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        xarray.Dataset(
            {"warmth": (("time", "lat", "lon"), warmth), "rain": (("time", "lat", "lon"), rain)}
        ).to_netcdf(path / "reference.nc")
    return path


# -- a difference against the model's own noise ---------------------------------


def test_a_difference_is_measured_against_what_noise_alone_does(tmp_path):
    control = open_source(write_latent_archive(tmp_path / "control"))
    steered = open_source(write_latent_archive(tmp_path / "steered", shift=(2, 1.0)))
    reseeded = open_source(write_latent_archive(tmp_path / "reseeded", shift=(3, 0.5)))
    growth = difference_growth(control, steered, noise=reseeded)
    assert growth.noise_rms[1] == pytest.approx(np.sqrt(0.25 / 6), abs=1e-3)
    assert growth.signal_to_noise[1] == pytest.approx(2.0, abs=1e-2)
    assert np.isnan(growth.signal_to_noise[0]).all()  # no noise yet, nothing to divide by
    summary = json.loads(json.dumps(growth.summary()))
    assert summary["settings"]["noise"] == str(tmp_path / "reseeded")
    assert "noise" in summary["provenance"]
    assert difference_growth(control, steered).signal_to_noise is None


def test_a_noise_run_must_be_comparable_too(tmp_path):
    control = open_source(write_latent_archive(tmp_path / "control"))
    other = write_latent_archive(tmp_path / "other")
    manifest = json.loads((other / "manifest.json").read_text())
    manifest["checkpoint"] = "another.ckpt"
    (other / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RequestError):
        difference_growth(control, control, noise=open_source(other))


# -- a field with no values yet ---------------------------------------------------


def test_an_empty_field_is_explained_not_divided_by_zero(fields):
    source = open_source(fields)
    with pytest.raises(RequestError, match="no values at .* later time"):
        rank_by_field(source, time=0, layer=2, field="rain")
    assert rank_by_field(source, time=1, layer=2, field="rain").ranking.channels[0] == 4


# -- storylines --------------------------------------------------------------------


def test_a_storyline_finds_the_field_at_every_layer_and_time(fields):
    story = field_storyline(open_source(fields), field="warmth")
    assert story.layers == (0, 1, 2) and story.times == tuple(LATENT_TIMES)
    assert (story.best > 0.98).all() and (story.column == 4).all() and (story.sign == 1).all()
    json.dumps(story.summary())


def test_a_storyline_skips_the_times_a_field_is_missing(fields):
    story = field_storyline(open_source(fields), field="rain", layers=[2])
    assert np.isnan(story.best[0, 0]) and story.column[0, 0] == -1
    assert story.best[1, 0] > 0.99


def test_a_storyline_reads_a_layer_through_its_basis(fields):
    source = open_source(fields)
    basis = pca_from_moments(accumulate_moments(source, layer=2), 2)
    story = field_storyline(source, field="warmth", layers=[1, 2], bases={2: basis})
    assert story.column[0, 0] == 4 and story.column[0, 1] == 0  # a channel, then a component
    assert story.settings["bases"] == {2: None}
    with pytest.raises(RequestError, match="does not read"):
        field_storyline(source, field="warmth", layers=[1], bases={2: basis})
    with pytest.raises(RequestError, match="no field"):
        field_storyline(source, field="snow")


# -- Hovmoller diagrams ------------------------------------------------------------


def test_a_hovmoller_of_a_field_peaks_at_the_bump(fields):
    diagram = hovmoller(open_source(fields), lat_min=0.0, lat_max=15.0, field="warmth")
    assert diagram.values.shape == (2, N_LON) and diagram.settings["rows"] == 1
    column = int(np.argmin(np.abs(diagram.lon - BUMP[1])))
    assert diagram.values[:, column] == pytest.approx([2.0, 4.0])  # the latents' own times
    assert np.argmax(diagram.values[0]) == column
    json.dumps(diagram.summary())


def test_a_hovmoller_of_a_channel_and_of_a_feature(fields):
    source = open_source(fields)
    by_channel = hovmoller(source, lat_min=0.0, lat_max=15.0, layer=2, channel=4)
    column = int(np.argmin(np.abs(by_channel.lon - BUMP[1])))
    assert np.argmax(by_channel.values[0]) == column
    basis = pca_from_moments(accumulate_moments(source, layer=2), 2)
    by_feature = hovmoller(source, lat_min=0.0, lat_max=15.0, layer=2, basis=basis, feature=0)
    assert np.argmax(np.abs(by_feature.values[0])) == column


def test_a_hovmoller_is_asked_for_one_thing_on_a_structured_grid(fields, tmp_path):
    source = open_source(fields)
    for kw in ({}, {"field": "warmth", "layer": 2, "channel": 4}, {"layer": 2}):
        with pytest.raises(RequestError, match="exactly one"):
            hovmoller(source, lat_min=0.0, lat_max=15.0, **kw)
    with pytest.raises(RequestError, match="no grid rows"):
        hovmoller(source, lat_min=88.0, lat_max=89.0, field="warmth")
    with pytest.raises(RequestError, match="below"):
        hovmoller(source, lat_min=15.0, lat_max=0.0, field="warmth")
    with pytest.raises(RequestError, match="no channel"):
        hovmoller(source, lat_min=0.0, lat_max=15.0, layer=2, channel=99)
    mesh = open_source(write_latent_archive(tmp_path / "mesh", mesh=True))
    with pytest.raises(RequestError, match="mesh"):
        hovmoller(mesh, lat_min=0.0, lat_max=15.0, layer=2, channel=4)


def test_a_masked_column_is_left_empty(tmp_path):
    lat = np.repeat(np.linspace(-82.5, 82.5, N_LAT), N_LON)
    lon = np.tile(np.arange(N_LON) * 15.0, N_LAT)
    mask = ~((np.abs(lat - 7.5) < 1) & (lon == 90.0))
    source = open_source(write_latent_archive(tmp_path / "masked", mask=mask))
    diagram = hovmoller(source, lat_min=0.0, lat_max=15.0, layer=2, channel=4)
    assert np.isnan(diagram.values[:, 6]).all() and diagram.settings["columns_without_nodes"] == 1


# -- the commands -------------------------------------------------------------------


def _invoke(*args):
    result = CliRunner().invoke(cli, ["daig", "latent", *(str(a) for a in args)])
    assert "Traceback" not in result.output
    return result


def test_cli_storyline(fields, tmp_path):
    result = _invoke("storyline", fields, "--field", "warmth")
    assert result.exit_code == 0 and "best |r| of any channel" in result.output
    basis = pca_from_moments(accumulate_moments(open_source(fields), layer=2), 2)
    from xaig.daig.latent import save_basis

    save_basis(tmp_path / "pca_L02.npz", basis, provenance=open_source(fields).info().provenance())
    template = str(tmp_path / "pca_L{layer:02d}.npz")
    with_bases = _invoke("storyline", fields, "--field", "warmth", "--bases", template, "--json")
    assert with_bases.exit_code == 0
    assert json.loads(with_bases.output)["column"][0] == [4, 4, 0]
    missing = _invoke(
        "storyline", fields, "--field", "warmth", "--bases", str(tmp_path / "x{layer}")
    )
    assert missing.exit_code != 0 and "no basis file" in missing.output


def test_cli_hovmoller(fields, tmp_path):
    out = tmp_path / "hov.npz"
    result = _invoke(
        "hovmoller", fields, "--lat-min", 0, "--lat-max", 15, "--field", "warmth", "--out", out
    )
    assert result.exit_code == 0 and "30E" in result.output
    with np.load(out) as stored:
        assert stored["values"].shape == (2, N_LON)


def test_cli_diff_with_noise(tmp_path):
    control = write_latent_archive(tmp_path / "control")
    steered = write_latent_archive(tmp_path / "steered", shift=(2, 1.0))
    reseeded = write_latent_archive(tmp_path / "reseeded", shift=(3, 0.5))
    result = _invoke("diff", control, steered, "--growth", "--noise", reseeded)
    assert result.exit_code == 0 and "multiple of what the noise run" in result.output
    assert "2" in result.output.splitlines()[-1]
    refused = _invoke("diff", control, steered, "--noise", reseeded)
    assert refused.exit_code != 0 and "--growth" in refused.output


# -- a pass against what it wrote -----------------------------------------------------


def test_lead_sets_a_pass_against_the_field_it_produced(fields):
    """``rain`` is empty at the first two reference times and bump-shaped at the third:
    the output of the pass that starts at the first latent time."""
    source = open_source(fields)
    with pytest.raises(RequestError, match="try a later time, or lead=1"):
        rank_by_field(source, time=0, layer=2, field="rain")
    led = rank_by_field(source, time=0, layer=2, field="rain", lead=1)
    assert led.ranking.channels[0] == 4 and led.correlation[4] > 0.99
    assert led.settings["lead"] == 1
    with pytest.raises(RequestError, match="ends before"):
        rank_by_field(source, time=1, layer=2, field="rain", lead=1)
    same = field_storyline(source, field="rain", layers=[2])
    ahead = field_storyline(source, field="rain", layers=[2], lead=1)
    assert np.isnan(same.best[0, 0]) and same.best[1, 0] > 0.99
    assert ahead.best[0, 0] > 0.99 and np.isnan(ahead.best[1, 0])  # past the file's end


def test_cli_lead(fields):
    ranked = _invoke("fields", fields, "--field", "rain", "--time", 0, "--layer", 2, "--lead", 1)
    assert ranked.exit_code == 0 and "+1 time(s) later" in ranked.output
    story = _invoke("storyline", fields, "--field", "rain", "--layer", 2, "--lead", 1, "--json")
    assert story.exit_code == 0 and json.loads(story.output)["settings"]["lead"] == 1
