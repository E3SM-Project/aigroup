"""Through time, between two runs, and against a physical field."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from conftest import BUMP, LATENT_TIMES, N_LAT, N_LON, write_latent_archive  # noqa: E402
from xaig._cli import cli  # noqa: E402
from xaig.core.errors import AdapterError, RequestError  # noqa: E402
from xaig.daig.latent import (  # noqa: E402
    ReferenceFields,
    Region,
    accumulate_moments,
    correlate_field,
    difference,
    difference_growth,
    load_basis,
    open_source,
    pca_from_moments,
    rank_by_field,
    region_series,
    save_basis,
)

HERE = Region(lat=BUMP[0], lon=BUMP[1], radius_km=2500.0)

# -- time ----------------------------------------------------------------------


def test_elapsed_time_follows_the_calendar_and_shows_the_gaps(latent_archive):
    info = open_source(latent_archive).info()
    assert info.elapsed_seconds() == (0.0, 6 * 3600.0)
    # The real exporter keeps the calls it is asked to, so positions are not leads;
    # and a no-leap February has 28 days in a year any other calendar gives 29.
    gappy = replace(info, times=("0424-02-28T18:00:00", "0424-03-01T00:00:00", "0424-03-02T12:00"))
    assert gappy.elapsed_seconds() == (0.0, 6 * 3600.0, 42 * 3600.0)
    assert replace(gappy, calendar="standard").elapsed_seconds()[1] == 30 * 3600.0
    thirty = replace(info, calendar="360_day", times=("0001-02-30", "0001-03-01"))
    assert thirty.elapsed_seconds() == (0.0, 86400.0)


def test_nothing_is_guessed_about_time(latent_archive):
    info = open_source(latent_archive).info()
    assert replace(info, calendar=None).elapsed_seconds() is None
    assert replace(info, calendar="julian").elapsed_seconds() is None
    assert replace(info, times=("step 0", "step 1")).elapsed_seconds() is None


def test_a_regions_series_is_its_weighted_mean_at_every_time(latent_archive):
    source = open_source(latent_archive)
    grid = source.grid()
    series = region_series(source, layer=2, region=HERE, channels=[4, 1])
    nodes = grid.within(HERE.lat, HERE.lon, HERE.radius_km)
    weights = grid.weights()[nodes] / grid.weights()[nodes].sum()
    expected = [weights @ source.load(t, 2)[nodes][:, [4, 1]] for t in LATENT_TIMES]
    assert series.values == pytest.approx(np.array(expected), abs=1e-6)
    assert series.times == tuple(LATENT_TIMES) and series.columns == (4, 1)
    assert series.elapsed_seconds == (0.0, 21600.0)
    assert series.values[0, 1] == pytest.approx(50.0, abs=0.1)  # the offset...
    centred = region_series(source, layer=2, region=HERE, channels=[4, 1], centred=True)
    assert abs(centred.values[0, 1]) < 0.05  # ...gone, and the bump less its own global mean
    removed = grid.mean(source.load(0, 2))[4]
    assert centred.values[0, 0] == pytest.approx(series.values[0, 0] - removed, abs=1e-5)
    json.dumps(series.summary())


def test_an_uncentred_series_reads_only_the_region(latent_archive):
    real = open_source(latent_archive)
    asked = []

    class Spy:
        info, grid = real.info, real.grid

        def load(self, time, layer, channels=None, nodes=None):
            asked.append((None if channels is None else len(channels), nodes is not None))
            return real.load(time, layer, channels=channels, nodes=nodes)

    region_series(Spy(), layer=2, region=HERE, channels=[4])
    assert asked == [(1, True), (1, True)]


def test_a_series_of_a_bases_features(latent_archive):
    source = open_source(latent_archive)
    basis = pca_from_moments(accumulate_moments(source, layer=2), 2)
    series = region_series(source, layer=2, region=HERE, basis=basis, features=[0], times=[1])
    assert series.values.shape == (1, 1) and series.settings["columns"] == "features"
    nodes = source.grid().within(HERE.lat, HERE.lon, HERE.radius_km)
    weights = source.grid().weights()[nodes]
    expected = (weights / weights.sum()) @ basis.transform(source.load(1, 2)[nodes])[:, 0]
    assert series.values[0, 0] == pytest.approx(expected)
    with pytest.raises(RequestError, match=r"channel\(s\) \[9\] outside"):
        region_series(source, layer=2, region=HERE, channels=[9])


# -- a run against its control ----------------------------------------------


@pytest.fixture
def pair(tmp_path):
    control = write_latent_archive(tmp_path / "control")
    steered = write_latent_archive(
        tmp_path / "steered", shift=(2, 1.0), experiment={"steer": {"channel": 2, "by": 1.0}}
    )
    return control, steered


def test_the_channel_that_was_pushed_is_the_one_that_moved(pair):
    control, steered = (open_source(p) for p in pair)
    result = difference(control, steered, time=1, layer=2, top=3)
    assert result.ranking.channels[0] == 2
    assert result.rms == pytest.approx([0, 0, 1, 0, 0, 0], abs=2e-3)
    assert result.maps[:, 0] == pytest.approx(1.0, abs=2e-3)
    assert difference(control, steered, time=0, layer=2).rms.max() == 0.0  # before the push
    summary = json.loads(json.dumps(result.summary()))
    assert summary["provenance"]["experiment"]["experiment"] == {"steer": {"channel": 2, "by": 1.0}}
    assert "experiment" not in summary["provenance"]["control"]


def test_a_difference_is_followed_through_layers_and_time(pair):
    control, steered = (open_source(p) for p in pair)
    growth = difference_growth(control, steered)
    assert growth.layers == (0, 1, 2) and growth.times == tuple(LATENT_TIMES)
    assert growth.rms[0] == pytest.approx(0.0)
    assert growth.rms[1] == pytest.approx(np.sqrt(1 / 6), abs=1e-3)  # one channel of six, by 1
    assert np.isfinite(growth.relative).all() and growth.elapsed_seconds == (0.0, 21600.0)
    json.dumps(growth.summary())


def test_runs_that_cannot_be_set_against_each_other_are_refused(pair, tmp_path):
    control = open_source(pair[0])
    mesh = open_source(write_latent_archive(tmp_path / "mesh", mesh=True))
    with pytest.raises(RequestError, match="different grids"):
        difference(control, mesh, time=0, layer=2)

    elsewhere = write_latent_archive(tmp_path / "elsewhere")
    with np.load(elsewhere / "grid.npz") as stored:
        moved = {name: stored[name] for name in stored.files}
    moved["lon"] = moved["lon"] + 5.0
    np.savez(elsewhere / "grid.npz", **moved)
    with pytest.raises(RequestError, match="in different places"):
        difference(control, open_source(elsewhere), time=0, layer=2)

    later = write_latent_archive(tmp_path / "later")
    manifest = json.loads((later / "manifest.json").read_text())
    manifest["latent_times"] = ["0425-01-02T06:00:00", "0425-01-02T12:00:00"]
    (later / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(RequestError, match="no latents at"):
        difference(control, open_source(later), time=0, layer=2)
    with pytest.raises(RequestError, match="any time the control has"):
        difference_growth(control, open_source(later))


# -- what the exporter said about the run -----------------------------------


def test_the_experiment_and_the_way_it_was_read_travel_with_every_result(tmp_path):
    from xaig.daig.latent import analyse_region

    mask = np.arange(N_LAT * N_LON) % 3 != 0
    path = write_latent_archive(tmp_path / "a", mask=mask, experiment={"seed": 7})
    plain = open_source(path).info().provenance()
    assert plain["experiment"] == {"seed": 7} and "options" not in plain
    result = analyse_region(open_source(path), time=0, layer=2, region=HERE)
    assert result.provenance["experiment"] == {"seed": 7} and result.provenance["xaig"]
    assert open_source(path).info().name == "toy-emulator · atmosphere"

    manifest = json.loads((path / "manifest.json").read_text())
    manifest["experiment"] = "a note"
    (path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AdapterError, match="'experiment' must be a mapping"):
        open_source(path)


# -- against a physical field -------------------------------------------------


def test_correlation_is_weighted_and_leaves_out_what_is_missing():
    rng = np.random.default_rng(0)
    field = rng.normal(size=500)
    latents = np.stack([2.0 * field + 1.0, -field, rng.normal(size=500), np.ones(500)], axis=1)
    field[:50] = np.nan  # land
    weights = rng.uniform(0.5, 1.5, size=500)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = correlate_field(latents, field, weights)
    assert r[:2] == pytest.approx([1.0, -1.0]) and abs(r[2]) < 0.2 and np.isnan(r[3])
    # Against numpy's own, on the nodes that count.
    keep = slice(50, None)
    expected = np.cov(latents[keep, 2], field[keep], aweights=weights[keep])
    assert r[2] == pytest.approx(expected[0, 1] / np.sqrt(expected[0, 0] * expected[1, 1]))
    with pytest.raises(RequestError, match="fewer than two"):
        correlate_field(latents, np.full(500, np.nan))


@pytest.fixture
def with_fields(tmp_path):
    """An archive whose reference file holds ``warmth`` -- the bump's own shape,
    so channel 4 must track it -- and a level-resolved variable that is no field."""
    xarray = pytest.importorskip("xarray")
    path = write_latent_archive(tmp_path / "a", reference="reference.nc")
    with np.load(path / "grid.npz") as grid:
        distance = np.hypot(grid["lat"] - BUMP[0], grid["lon"] - BUMP[1])
    warmth = (
        np.exp(-((distance / 20.0) ** 2)).reshape(1, N_LAT, N_LON)
        * np.array([1.0, 2.0, 4.0])[:, None, None]
    )
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["reference_times"] = ["0425-01-01T00:00:00", *LATENT_TIMES]  # one more than latents
    (path / "manifest.json").write_text(json.dumps(manifest))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        xarray.Dataset(
            {
                "warmth": (("time", "lat", "lon"), warmth),
                "profile": (("time", "level"), np.zeros((3, 4))),
            }
        ).to_netcdf(path / "reference.nc")
    return path


def test_reference_fields_are_read_at_the_latents_own_times(with_fields, latent_archive):
    source = open_source(with_fields)
    assert isinstance(source, ReferenceFields) and source.field_names() == ("warmth",)
    first, last = source.field("warmth", 0), source.field("warmth", LATENT_TIMES[1])
    assert first.shape == (N_LAT * N_LON,) and first.dtype == np.float64
    assert last.max() / first.max() == pytest.approx(2.0)  # times 1 and 2 of the file, not 0 and 1
    with pytest.raises(RequestError, match="fields are warmth"):
        source.field("sst", 0)
    assert open_source(latent_archive).field_names() == ()  # no reference file: no fields


def test_the_channel_that_tracks_a_field_is_found(with_fields, latent_archive):
    source = open_source(with_fields)
    ranking = rank_by_field(source, time=0, layer=2, field="warmth", top=2)
    assert ranking.ranking.channels[0] == 4 and ranking.ranking.scores[0] > 0.99
    basis = pca_from_moments(accumulate_moments(source, layer=2), 2)
    by_feature = rank_by_field(source, time=0, layer=2, field="warmth", basis=basis)
    assert by_feature.ranking.channels[0] == 0 and abs(by_feature.ranking.scores[0]) > 0.99
    assert by_feature.summary()["settings"]["columns"] == "features"
    with pytest.raises(RequestError, match="keeps no physical fields|no field"):
        rank_by_field(open_source(latent_archive), time=0, layer=2, field="warmth")


# -- the commands ---------------------------------------------------------------


def _invoke(*args):
    result = CliRunner().invoke(cli, ["daig", "latent", *(str(a) for a in args)])
    assert "Traceback" not in result.output
    return result


def test_cli_series(latent_archive):
    here = ["--lat", "7.5", "--lon", "45", "--radius-km", "2500"]
    table = _invoke("series", latent_archive, *here, "--channel", "4", "--channel", "1")
    assert table.exit_code == 0, table.output
    assert table.output.splitlines()[0].split() == ["TIME", "HOURS", "4", "1"]
    assert table.output.splitlines()[2].split()[:2] == [LATENT_TIMES[1], "6"]
    summary = json.loads(
        _invoke("series", latent_archive, *here, "--channel", "4", "--json").output
    )
    assert summary["columns"] == [4] and len(summary["values"]) == 2
    nothing = _invoke("series", latent_archive, *here)
    assert nothing.exit_code == 2 and "name what to follow" in nothing.output


def test_cli_pca_writes_a_basis_the_other_commands_take(latent_archive, tmp_path):
    out = tmp_path / "global.npz"
    fitted = _invoke("pca", latent_archive, "--components", "3", "--out", out)
    assert fitted.exit_code == 0, fitted.output
    assert "3 component(s) of layer 2 over 2 time(s)" in fitted.output
    basis = load_basis(out)
    assert basis.n_features == 3
    assert basis.meta["fitted_on"]["provenance"]["model"] == "toy-emulator"

    args = ["region", latent_archive, "--lat", "7.5", "--lon", "45", "--radius-km", "2500"]
    shown = _invoke(*args, "--features", "2", "--basis", out)
    assert shown.exit_code == 0 and "F0  peak" in shown.output
    summary = json.loads(_invoke(*args, "--features", "2", "--basis", out, "--json").output)
    assert summary["settings"]["basis"] == str(out) and len(summary["features"]) == 2
    followed = _invoke("series", latent_archive, *args[2:], "--basis", out, "--feature", "0")
    assert followed.exit_code == 0 and followed.output.splitlines()[0].split()[-1] == "0"


def test_cli_diff(pair):
    ranked = _invoke("diff", *pair, "--top", "2")
    assert ranked.exit_code == 0, ranked.output
    assert ranked.output.splitlines()[3].split()[:2] == ["1", "2"]  # rank 1 is channel 2
    growth = _invoke("diff", *pair, "--growth")
    assert growth.exit_code == 0 and "LAYER 2" in growth.output
    assert len(json.loads(_invoke("diff", *pair, "--growth", "--json").output)["rms"]) == 2


def test_cli_fields(with_fields, latent_archive):
    assert _invoke("fields", with_fields).output.split() == ["warmth"]
    assert "no reference fields" in _invoke("fields", latent_archive).output
    ranked = _invoke("fields", with_fields, "--field", "warmth", "--top", "1")
    assert ranked.exit_code == 0, ranked.output
    assert ranked.output.splitlines()[-1].split()[:2] == ["1", "4"]
    assert "reference field(s)" in _invoke("info", with_fields).output


def test_cli_info_shows_what_was_done_to_the_run(pair):
    assert "experiment.steer" in _invoke("info", pair[1]).output


def test_a_basis_written_here_is_usable_there(tmp_path, latent_archive):
    """The round trip a basis file is for: fit, save, load, analyse."""
    source = open_source(latent_archive)
    path = save_basis(tmp_path / "b.npz", pca_from_moments(accumulate_moments(source, layer=2), 2))
    assert region_series(source, layer=2, region=HERE, basis=load_basis(path)).values.shape == (
        2,
        2,
    )


# -- what a second review found ---------------------------------------------


def test_a_difference_weighs_only_the_nodes_valid_in_both_runs(tmp_path):
    """One run's invalid node holds whatever it holds -- here NaN -- and used to
    decide the whole difference: the control's mask alone was consulted."""
    mask = np.arange(N_LAT * N_LON) % 4 != 0
    control = write_latent_archive(tmp_path / "control")
    masked = write_latent_archive(tmp_path / "masked", mask=mask, shift=(2, 1.0))
    for step in masked.glob("step_*.npy"):
        stored = np.load(step)
        stored[:, ~mask, :] = np.nan
        np.save(step, stored)
    a, b = open_source(control), open_source(masked)
    for first, second in ((a, b), (b, a)):  # whichever of the two carries the mask
        result = difference(first, second, time=1, layer=2)
        assert result.rms == pytest.approx([0, 0, 1, 0, 0, 0], abs=2e-3)
        assert result.settings["n_nodes_compared"] == int(mask.sum())
        assert np.isnan(result.maps[~mask]).all() and np.isfinite(result.maps[mask]).all()
        growth = difference_growth(first, second)
        assert growth.rms[1] == pytest.approx(np.sqrt(1 / 6), abs=1e-3)
        assert growth.settings["n_nodes_compared"] == int(mask.sum())


def _renamed(path, **declared):
    manifest = json.loads((path / "manifest.json").read_text())
    (path / "manifest.json").write_text(json.dumps({**manifest, **declared}))
    return path


def test_two_networks_are_not_compared_channel_by_channel(pair, tmp_path):
    """Equal widths on one grid used to be enough. Channel 2 of another checkpoint
    is not channel 2 of this one, whatever number it carries."""
    control = open_source(pair[0])
    other = open_source(_renamed(write_latent_archive(tmp_path / "seed2"), checkpoint="seed2.ckpt"))
    with pytest.raises(RequestError, match="different networks .*'toy.ckpt' against 'seed2.ckpt'"):
        difference(control, other, time=1, layer=2)
    with pytest.raises(RequestError, match="different networks"):
        difference_growth(control, other)
    insisted = difference(control, other, time=1, layer=2, across_models=True)
    assert insisted.settings["across_models"] is True  # and the result says that it was
    # What a source does not declare is not held against it.
    anonymous = _renamed(write_latent_archive(tmp_path / "anon"), checkpoint=None, model=None)
    assert difference(control, open_source(anonymous), time=1, layer=2).rms.max() == 0.0

    refused = _invoke("diff", pair[0], tmp_path / "seed2")
    assert refused.exit_code == 1 and "different networks" in refused.output
    assert _invoke("diff", pair[0], tmp_path / "seed2", "--across-models").exit_code == 0


@pytest.mark.parametrize("offset", [0.0, 1e3, 1e6])
def test_a_channels_offset_takes_nothing_from_its_correlation(offset):
    """Moments about zero lost a variation of 1 beside a mean of 1e6, and the
    tolerance meant to catch a constant channel then called this one constant."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    latents = np.stack([x + offset, -(x + offset), np.full(5, offset + 7.0)], axis=1)
    r = correlate_field(latents.astype(np.float32), x)
    assert r[:2] == pytest.approx([1.0, -1.0], abs=1e-12) and np.isnan(r[2])


# -- a basis that does not say where it was fitted --------------------------------


def test_an_anonymous_basis_needs_the_override_through_time_and_against_fields(tmp_path):
    """The same rule as in a region: what is missing may be allowed, what is wrong may not."""
    from xaig.daig.latent import fit_pca
    from xaig.daig.latent.toy import STORM_CHANNEL, write_toy

    source = open_source(write_toy(tmp_path / "toy"), mask_variable="sst")
    layer = source.info().last_layer
    here = Region(lat=10.0, lon=-150.0 + 12.0 * 3, radius_km=1500.0)
    anonymous = fit_pca(source.load(2, layer).astype(np.float64)[source.grid().valid], 2)
    for run in (
        lambda **kw: region_series(source, layer=layer, region=here, basis=anonymous, **kw),
        lambda **kw: rank_by_field(
            source, time=2, layer=layer, field="precipitation", basis=anonymous, **kw
        ),
    ):
        with pytest.raises(RequestError, match="compatibility is unverified"):
            run()
        assert run(allow_unverified_basis=True).settings["allow_unverified_basis"] is True
    elsewhere = replace(anonymous, meta={"fitted_on": {"layer": 0}})
    with pytest.raises(RequestError, match="layer 0 it was fitted on"):
        rank_by_field(
            source,
            time=2,
            layer=layer,
            field="precipitation",
            basis=elsewhere,
            allow_unverified_basis=True,
        )
    # and the toy's planted answer, by channel: nothing follows the rain like the storm
    assert rank_by_field(source, time=2, layer=layer, field="precipitation").ranking.channels[
        0
    ] == (STORM_CHANNEL)


def test_cli_series_and_fields_take_the_override(tmp_path):
    from xaig.daig.latent import fit_pca, save_basis
    from xaig.daig.latent.toy import write_toy

    archive = write_toy(tmp_path / "toy")
    basis = save_basis(
        tmp_path / "anonymous.npz", fit_pca(np.random.default_rng(0).normal(size=(40, 16)), 2)
    )
    where = ["--lat", "10", "--lon", "-114"]
    for args in (
        ["series", str(archive), *where, "--feature", "0", "--basis", str(basis)],
        ["fields", str(archive), "--field", "precipitation", "--basis", str(basis)],
    ):
        refused = CliRunner().invoke(cli, ["daig", "latent", *args])
        assert refused.exit_code == 1 and "--allow-unverified-basis" in refused.output
        allowed = CliRunner().invoke(
            cli, ["daig", "latent", *args, "--allow-unverified-basis", "--json"]
        )
        assert allowed.exit_code == 0, allowed.output
        assert json.loads(allowed.output)["settings"]["allow_unverified_basis"] is True
