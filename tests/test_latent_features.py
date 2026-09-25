"""A census of a layer's features, and one feature's profile against every field."""

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
    Region,
    accumulate_moments,
    feature_census,
    feature_profile,
    open_source,
    pca_from_moments,
    save_basis,
)

# Channel 4 carries the bump, three times its height and more at each layer: above
# one only near it. Channel 1 sits at 50 everywhere; the rest are faint noise.
NEAR_THE_BUMP = 1.0


@pytest.fixture
def fields(tmp_path):
    """``warmth`` has the bump's shape and ``coolth`` its opposite; ``flat`` is the
    same everywhere, always; ``rain`` is missing at the first latent time."""
    xarray = pytest.importorskip("xarray")
    path = write_latent_archive(tmp_path / "a", reference="reference.nc")
    with np.load(path / "grid.npz") as grid:
        distance = np.hypot(grid["lat"] - BUMP[0], grid["lon"] - BUMP[1])
    shape = np.exp(-((distance / 20.0) ** 2)).reshape(1, N_LAT, N_LON)
    warmth = shape * np.array([1.0, 2.0, 4.0])[:, None, None] + 280.0
    rain = shape * np.array([np.nan, np.nan, 3.0])[:, None, None]
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["reference_times"] = ["0425-01-01T00:00:00", *LATENT_TIMES]
    (path / "manifest.json").write_text(json.dumps(manifest))
    dims = ("time", "lat", "lon")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        xarray.Dataset(
            {
                "warmth": (dims, warmth),
                "coolth": (dims, -warmth),
                "flat": (dims, np.full(warmth.shape, 1e5)),
                "rain": (dims, rain),
            }
        ).to_netcdf(path / "reference.nc")
    return path


# -- a census ----------------------------------------------------------------------


def test_a_census_says_how_much_of_the_world_each_channel_covers(fields):
    census = feature_census(open_source(fields), time=0, layer=2, threshold=NEAR_THE_BUMP)
    assert census.coverage[1] == pytest.approx(1.0)
    assert 0.0 < census.coverage[4] < 0.1
    assert census.coverage[[0, 2, 3, 5]].max() == 0.0 and np.isnan(census.strength[0])
    assert (census.peak_lat[4], census.peak_lon[4]) == BUMP
    assert census.peak[4] == pytest.approx(9.0, abs=0.2)
    assert list(census.ranked("coverage")[:2]) == [1, 4]
    assert list(census.ranked("peak")[:2]) == [1, 4]
    assert set(census.ranked("coverage")[2:]) == {0, 2, 3, 5}  # never active: last
    assert [c["column"] for c in census.summary(top=1)["columns"]] == [1]
    json.dumps(census.summary())
    with pytest.raises(RequestError, match="order by"):
        census.ranked("loudness")


def test_a_census_of_a_region_and_of_a_basis(fields):
    source = open_source(fields)
    near = feature_census(
        source, time=0, layer=2, region=Region(*BUMP, radius_km=1500), threshold=NEAR_THE_BUMP
    )
    assert near.coverage[4] == pytest.approx(1.0)
    basis = pca_from_moments(accumulate_moments(source, layer=2), 2)
    census = feature_census(source, time=-1, layer=2, basis=basis)
    assert census.coverage.shape == (2,) and census.settings["columns"] == "features"


# -- a profile ----------------------------------------------------------------------


def test_a_profile_ranks_the_fields_a_channel_goes_with(fields):
    profile = feature_profile(open_source(fields), layer=2, column=4, threshold=NEAR_THE_BUMP)
    assert set(profile.fields[:3]) == {"warmth", "coolth", "rain"}
    effect = dict(zip(profile.fields, profile.effect, strict=True))
    assert effect["warmth"] > 1.0 and effect["coolth"] == pytest.approx(-effect["warmth"])
    assert profile.fields[-1] == "flat" and np.isnan(effect["flat"])  # no spread
    assert effect["rain"] > 1.0  # from the one time it has values
    warm = profile.fields.index("warmth")
    assert profile.active_mean[warm] > profile.inactive_mean[warm] > 280.0
    assert 0.0 < profile.coverage < 0.1 and len(profile.times) == 2
    json.dumps(profile.summary())


def test_a_profile_can_be_narrowed_and_read_through_a_basis(fields):
    source = open_source(fields)
    one = feature_profile(source, layer=2, column=4, times=[1], fields=["warmth"], threshold=1.0)
    assert one.fields == ("warmth",) and one.times == (LATENT_TIMES[1],)
    # Everywhere in the region is active: nothing to set it against.
    near = feature_profile(
        source, layer=2, column=4, region=Region(*BUMP, radius_km=1500), threshold=1.0
    )
    assert np.isnan(near.effect).all() and near.coverage == pytest.approx(1.0)
    basis = pca_from_moments(accumulate_moments(source, layer=2), 2)
    by_feature = feature_profile(source, layer=2, column=0, basis=basis)
    assert by_feature.settings["columns"] == "features" and len(by_feature.fields) == 4


def test_a_profile_says_what_it_cannot_do(fields, tmp_path):
    source = open_source(fields)
    with pytest.raises(RequestError, match="no field"):
        feature_profile(source, layer=2, column=4, fields=["snow"])
    with pytest.raises(RequestError, match="no channel 9"):
        feature_profile(source, layer=2, column=9)
    with pytest.raises(RequestError, match="no physical fields"):
        feature_profile(open_source(write_latent_archive(tmp_path / "bare")), layer=2, column=4)
    with pytest.raises(RequestError, match="widen"):
        feature_census(source, time=0, layer=2, region=Region(-80.0, 200.0, radius_km=10))


# -- the commands -------------------------------------------------------------------


def _invoke(*args):
    result = CliRunner().invoke(cli, ["daig", "latent", *(str(a) for a in args)])
    assert "Traceback" not in result.output
    return result


def test_cli_census(fields):
    result = _invoke("census", fields, "--layer", 2, "--threshold", 1, "--top", 2)
    assert result.exit_code == 0 and "channels of layer 2" in result.output
    as_json = _invoke("census", fields, "--layer", 2, "--threshold", 1, "--by", "peak", "--json")
    assert [c["column"] for c in json.loads(as_json.output)["columns"]][:2] == [1, 4]
    half = _invoke("census", fields, "--lat", 7.5)
    assert half.exit_code != 0 and "both --lat and --lon" in half.output


def test_cli_profile(fields, tmp_path):
    result = _invoke("profile", fields, "--layer", 2, "--channel", 4, "--threshold", 1)
    assert result.exit_code == 0 and "warmth" in result.output and "active over" in result.output
    source = open_source(fields)
    basis = pca_from_moments(accumulate_moments(source, layer=2), 2)
    save_basis(tmp_path / "pca.npz", basis, provenance=source.info().provenance())
    by_feature = _invoke(
        "profile", fields, "--layer", 2, "--basis", tmp_path / "pca.npz", "--feature", 0, "--json"
    )
    assert by_feature.exit_code == 0 and json.loads(by_feature.output)["settings"]["column"] == 0
    neither = _invoke("profile", fields, "--layer", 2)
    assert neither.exit_code != 0 and "name one" in neither.output


def test_a_profile_can_set_a_pass_against_what_it_wrote(fields):
    source = open_source(fields)
    same = feature_profile(source, layer=2, column=4, times=[0], threshold=NEAR_THE_BUMP)
    ahead = feature_profile(source, layer=2, column=4, times=[0], threshold=NEAR_THE_BUMP, lead=1)
    rain = [dict(zip(p.fields, p.effect, strict=True))["rain"] for p in (same, ahead)]
    assert np.isnan(rain[0]) and rain[1] > 1.0 and ahead.settings["lead"] == 1
