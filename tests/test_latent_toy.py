"""The toy emulator, the archive writer and the reader, against one another.

The toy is shaped like the real SamudrACE archives where that has bitten or could:
kept steps with a gap between them, fields that start one step before the latents
and hold more times, Gaussian latitudes, longitudes in -180..180, a mask that only
a reference variable knows, and a calendar without leap days.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("xarray")

from click.testing import CliRunner  # noqa: E402

from xaig._cli import cli  # noqa: E402
from xaig.adapters.latent_archive import write_archive  # noqa: E402
from xaig.core.errors import RequestError  # noqa: E402
from xaig.daig.latent import ReferenceFields, open_source  # noqa: E402
from xaig.daig.latent.toy import (  # noqa: E402
    OFFSET_CHANNEL,
    STORM_CHANNEL,
    noleap_label,
    parse_steps,
    toy_grid,
    toy_run,
    write_toy,
)


@pytest.fixture(scope="module")
def run():
    return toy_run()


@pytest.fixture(scope="module")
def archive(tmp_path_factory):
    return write_toy(tmp_path_factory.mktemp("toy") / "control")


# -- what was written is what is read ----------------------------------------


def test_the_reader_gives_back_what_the_toy_held(run, archive):
    source = open_source(archive)
    info = source.info()
    assert info.identity() == {
        "model": "xaig-toy",
        "component": "atmosphere",
        "checkpoint": "seed-0",
    }
    assert [(x.index, x.n_channels, x.label) for x in info.layers][::3] == [
        (0, 16, "encoder output"),
        (3, 16, "block 3 output"),
    ]
    assert info.times == run.times and info.experiment["kept_steps"] == [1, 2, 3, 4, 9, 10, 11, 12]
    for index, (_, held) in enumerate(run.layers):
        for time in (0, -1):
            # float16 on disk, as the real exporter writes: three decimal places here
            assert np.abs(source.load(time, index) - held[time]).max() < 2e-3


def test_the_grid_is_gaussian_with_longitudes_either_side_of_greenwich(archive):
    grid = open_source(archive).grid()
    lat = grid.lat.reshape(grid.shape)[:, 0]
    assert grid.shape == (24, 48) and grid.lon.min() == -176.25 and grid.lon.max() == 176.25
    assert np.ptp(np.diff(lat)) > 0.05  # not evenly spaced: these are quadrature nodes
    _, quadrature = np.polynomial.legendre.leggauss(24)
    rows = grid.weights().reshape(grid.shape).sum(axis=1)
    assert np.allclose(rows, quadrature / 2.0, atol=2e-3)


def test_kept_steps_leave_a_gap_and_the_calendar_has_no_leap_day(archive):
    info = open_source(archive).info()
    # Steps 5-8 were not kept. 0424 is a leap year by the usual rule and this calendar
    # has no 29 February, so the gap is 30 hours where the usual rule would say 54.
    assert info.times[3:5] == ("0424-02-28T00:00:00", "0424-03-01T06:00:00")
    hours = [seconds / 3600 for seconds in info.elapsed_seconds()]
    assert hours == [0, 6, 12, 18, 48, 54, 60, 66]


def test_labels_carry_over_a_year_end_without_a_leap_day():
    assert noleap_label(0) == "0424-02-27T00:00:00"
    assert noleap_label(24 * 365) == "0425-02-27T00:00:00"
    assert noleap_label(18, start=(424, 12, 31, 12)) == "0425-01-01T06:00:00"


# -- fields beside the latents -------------------------------------------------


def test_fields_are_found_by_their_time_not_by_their_position(run, archive):
    """The fields begin one step before the latents and hold more times, as the real
    archives do: the first latent time is the *second* field time."""
    source = open_source(archive)
    assert isinstance(source, ReferenceFields)
    assert source.field_names() == ("precipitation", "sst", "temperature")
    assert len(run.field_times) == 10 and len(run.times) == 8
    assert run.field_times[1] == run.times[0]
    for time in (0, 4, -1):
        label = run.times[time]
        held = run.fields["temperature"][run.field_times.index(label)]
        assert np.allclose(source.field("temperature", time), held, atol=1e-3)
        assert np.allclose(source.field("temperature", label), held, atol=1e-3)
    by_position = run.fields["temperature"][0]
    assert not np.allclose(source.field("temperature", 0), by_position, atol=1e-3)
    with pytest.raises(RequestError, match="no field 'rain'.*precipitation, sst, temperature"):
        source.field("rain", 0)


def test_a_mask_nobody_asked_for_is_not_applied_and_land_is_not_zero(archive):
    """The real hazard: nothing in an archive without a mask says it needs one."""
    unmasked, masked = open_source(archive).grid(), open_source(archive, mask_variable="sst").grid()
    assert unmasked.mask is None and int(unmasked.valid.sum()) == 1152
    assert int(masked.valid.sum()) == 1062
    over_land = open_source(archive).load(0, 3)[~masked.valid]
    assert np.sqrt((over_land.astype(np.float64) ** 2).mean()) > 1.0


def test_a_mask_can_travel_in_the_archive_instead(tmp_path, run):
    sea = np.isfinite(run.fields["sst"][0])
    grid = toy_grid()
    grid = type(grid)(lat=grid.lat, lon=grid.lon, shape=grid.shape, mask=sea)
    path = write_archive(tmp_path / "masked", grid=grid, times=run.times, layers=run.layers)
    assert np.array_equal(open_source(path).grid().valid, sea)


# -- what is planted, so that an analysis has a right answer --------------------


def test_the_storm_is_found_once_the_layer_is_centred(archive):
    source = open_source(archive, mask_variable="sst")
    grid = source.grid()
    time = 2  # kept step 3: the storm has drifted to 114W
    nodes = grid.within(10.0, -150.0 + 12.0 * 3, 1200.0)
    layer = source.load(time, 3).astype(np.float64)
    assert np.abs(layer[nodes]).max(axis=0).argmax() == OFFSET_CHANNEL
    centred = layer - grid.mean(layer)
    assert np.abs(centred[nodes]).max(axis=0).argmax() == STORM_CHANNEL
    rain = source.field("precipitation", time)
    assert np.corrcoef(layer[grid.valid, STORM_CHANNEL], rain[grid.valid])[0, 1] > 0.999


def test_a_steered_run_leaves_its_control_where_it_was_pushed_and_then_everywhere(tmp_path):
    control = open_source(write_toy(tmp_path / "control"))
    steered = open_source(write_toy(tmp_path / "steered", steer=(2, 7, 3.0)))
    assert steered.info().experiment["steer"] == {"layer": 2, "channel": 7, "by": 3.0}
    assert "steer" not in control.info().experiment
    assert steered.info().identity() == control.info().identity()  # one network, twice

    def apart(time, layer):
        return np.abs(steered.load(time, layer) - control.load(time, layer)).max(axis=0)

    assert apart(0, 1).max() == 0.0  # upstream of the push, on the first step
    pushed = apart(0, 2)
    assert abs(pushed[7] - 3.0) < 2e-3 and np.delete(pushed, 7).max() == 0.0
    assert apart(-1, 1).max() > 0.01  # and then the temperature carries it round


def test_the_same_seed_is_the_same_run_and_another_is_not(tmp_path, archive):
    again = write_toy(tmp_path / "again")
    other = write_toy(tmp_path / "other", seed=1)
    first = (archive / "step_03.npy").read_bytes()
    assert (again / "step_03.npy").read_bytes() == first
    assert (other / "step_03.npy").read_bytes() != first


# -- what the writer and the toy refuse ------------------------------------------


def test_the_writer_checks_everything_before_it_writes_anything(tmp_path, run):
    grid, out = run.grid, tmp_path / "bad"
    cases = [
        (dict(times=run.times[:3]), "layer 0 .* expected \\(3 times"),
        (dict(times=(run.times[0],) * 8), "each only once"),
        (dict(layers=()), "at least one layer"),
        (dict(fields=run.fields, field_times=run.field_times[2:]), "the fields hold no time"),
        (dict(fields={"t": run.fields["sst"][:, :5]}, field_times=run.field_times), "field 't'"),
    ]
    for changed, message in cases:
        contents = dict(grid=grid, times=run.times, layers=run.layers) | changed
        with pytest.raises(RequestError, match=message):
            write_archive(out, **contents)
        assert not out.exists()


def test_an_archive_is_not_written_over_unless_asked(tmp_path):
    path = write_toy(tmp_path / "a")
    with pytest.raises(RequestError, match="is not empty"):
        write_toy(path)
    assert open_source(write_toy(path, overwrite=True, seed=3)).info().checkpoint == "seed-3"


def test_steps_and_steering_are_checked():
    assert parse_steps("1-3, 7") == (1, 2, 3, 7)
    with pytest.raises(RequestError, match="steps look like"):
        parse_steps("first four")
    with pytest.raises(RequestError, match="must lie in 1..12"):
        toy_run(keep=(0, 1))
    with pytest.raises(RequestError, match="steer is"):
        toy_run(steer=(4, 0, 1.0))


# -- xaig daig latent toy ------------------------------------------------------------


def test_cli_toy_then_info(tmp_path):
    out = tmp_path / "steered"
    made = CliRunner().invoke(
        cli, ["daig", "latent", "toy", str(out), "--steer", "2:7:3", "--keep", "1-2,6"]
    )
    assert made.exit_code == 0, made.output
    assert json.loads((out / "manifest.json").read_text())["latent_times"][-1].endswith("T12:00:00")
    shown = CliRunner().invoke(cli, ["daig", "latent", "info", str(out), "--mask-variable", "sst"])
    assert shown.exit_code == 0, shown.output
    assert "24x48, 1152 nodes, 1062 valid" in shown.output and "xaig-toy" in shown.output
    assert "experiment.steer" in shown.output and "3 reference field(s)" in shown.output
    bad = CliRunner().invoke(cli, ["daig", "latent", "toy", str(tmp_path / "x"), "--steer", "2:7"])
    assert bad.exit_code == 1 and "LAYER:CHANNEL:AMOUNT" in bad.output
