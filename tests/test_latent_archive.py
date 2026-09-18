from __future__ import annotations

import json
import warnings

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from conftest import LATENT_TIMES, N_CHANNELS, N_LAT, N_LON, write_latent_archive  # noqa: E402
from xaig._cli import cli  # noqa: E402
from xaig.core import registry  # noqa: E402
from xaig.core.errors import AdapterError  # noqa: E402
from xaig.daig.latent import LatentSource, open_source  # noqa: E402


def test_describes_the_archive_without_loading_it(latent_archive):
    source = open_source(latent_archive)
    assert isinstance(source, LatentSource)
    info = source.info()
    assert info.times == tuple(LATENT_TIMES)
    assert [(x.index, x.n_channels) for x in info.layers] == [(0, 6), (1, 6), (2, 6)]
    assert [x.index for x in info.off_grid_layers] == [100]
    assert info.provenance()["model"] == "toy-emulator"
    assert source.grid().shape == (N_LAT, N_LON)


def test_a_selective_load_equals_a_slice_of_the_full_one(latent_archive):
    source = open_source(latent_archive)
    full = source.load(0, 2)
    assert full.shape == (N_LAT * N_LON, N_CHANNELS) and full.dtype == np.float32
    part = source.load(0, 2, channels=[4, 1], nodes=[7, 3, 200])
    assert np.array_equal(part, full[[7, 3, 200]][:, [4, 1]])


def test_times_are_addressed_by_label_or_position(latent_archive):
    source = open_source(latent_archive)
    assert np.array_equal(source.load(LATENT_TIMES[1], 0), source.load(-1, 0))
    with pytest.raises(KeyError, match="times are"):
        source.load("0425-02-01T00:00:00", 0)
    with pytest.raises(KeyError, match="out of range"):
        source.load(5, 0)


def test_a_directory_that_is_not_an_archive(tmp_path):
    with pytest.raises(AdapterError, match="not a latent archive"):
        open_source(tmp_path)


def test_an_archive_that_contradicts_its_manifest_is_refused(latent_archive):
    manifest = json.loads((latent_archive / "manifest.json").read_text())
    manifest["steps"][0]["n_channels"] = 99
    (latent_archive / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(AdapterError, match=r"shape .* but the manifest implies"):
        open_source(latent_archive).load(0, 0)


def test_a_manifest_missing_what_the_reader_needs(latent_archive):
    (latent_archive / "manifest.json").write_text(json.dumps({"latent_times": []}))
    with pytest.raises(AdapterError, match="unreadable manifest"):
        open_source(latent_archive)


def test_the_mask_can_travel_in_the_archive(tmp_path):
    mask = np.arange(N_LAT * N_LON) % 3 != 0
    grid = open_source(write_latent_archive(tmp_path / "a", mask=mask)).grid()
    assert np.array_equal(grid.valid, mask)


def test_or_be_taken_from_a_reference_variable(tmp_path):
    netcdf = pytest.importorskip("netCDF4")
    mask = np.arange(N_LAT * N_LON) % 3 != 0
    path = write_latent_archive(tmp_path / "a", mask=mask, reference="reference.nc")
    with netcdf.Dataset(path / "reference.nc", "w") as ds, warnings.catch_warnings():
        # netCDF4 1.7's *write* path trips a NumPy 2.5 deprecation of its own. Writing
        # is only this fixture's business; the reader is clean under -W error.
        warnings.simplefilter("ignore", DeprecationWarning)
        for name, size in (("time", 1), ("lat", N_LAT), ("lon", N_LON)):
            ds.createDimension(name, size)
        sst = ds.createVariable("sst", "f4", ("time", "lat", "lon"), fill_value=1e20)
        sst[0] = np.where(mask.reshape(N_LAT, N_LON), 1.0, 1e20)
    assert open_source(path).grid().mask is None  # nothing is guessed
    assert np.array_equal(open_source(path, mask_variable="sst").grid().valid, mask)
    with pytest.raises(AdapterError, match="no variable 'ssh'"):
        open_source(path, mask_variable="ssh").grid()


def test_options_are_validated_like_any_other_adapters(latent_archive):
    with pytest.raises(AdapterError, match="mask_variabel; accepted: mask_variable"):
        open_source(latent_archive, mask_variabel="sst")


def test_an_adapter_that_cannot_supply_latents_is_refused(toy_table_path):
    assert "latent-archive" in registry.available()
    with pytest.raises(AdapterError, match="does not implement LatentSource"):
        open_source(toy_table_path, adapter="table")


# -- xaig daig latent ------------------------------------------------------


def test_cli_info(latent_archive):
    result = CliRunner().invoke(cli, ["daig", "latent", "info", str(latent_archive)])
    assert result.exit_code == 0, result.output
    assert "toy.ckpt" in result.output and "12x24, 288 nodes, 288 valid" in result.output
    assert "1 more layer(s) on coarser grids" in result.output


def test_cli_region_as_a_table_and_as_json(latent_archive):
    args = ["daig", "latent", "region", str(latent_archive), "--lat", "7.5", "--lon", "45"]
    args += ["--radius-km", "2500", "--centred", "--top", "3", "--pcs", "2"]
    table = CliRunner().invoke(cli, args)
    assert table.exit_code == 0, table.output
    assert table.output.splitlines()[2].split()[:2] == ["RANK", "CHANNEL"]
    assert table.output.splitlines()[3].split()[:2] == ["1", "4"]
    assert "PC0" in table.output
    summary = json.loads(CliRunner().invoke(cli, [*args, "--json"]).output)
    assert summary["settings"]["centred"] is True and summary["settings"]["layer"] == 2
    assert summary["ranking"][0]["channel"] == 4


def test_cli_explains_a_bad_request_in_one_line(latent_archive):
    args = ["daig", "latent", "region", str(latent_archive), "--lat", "0", "--lon", "0"]
    result = CliRunner().invoke(cli, [*args, "--layer", "9"])
    assert result.exit_code != 0 and "layers are 0, 1, 2" in result.output
    assert "Traceback" not in result.output


def test_a_loaded_array_is_the_callers_to_modify(tmp_path):
    """Analyses centre in place, so load() must never hand out a read-only view of
    the file -- which is what a float32 archive would otherwise give."""
    path = write_latent_archive(tmp_path / "a")
    stored = np.load(path / "step_00.npy").astype(np.float32)
    np.save(path / "step_00.npy", stored)
    loaded = open_source(path).load(0, 0)
    loaded -= 1.0
    assert np.array_equal(open_source(path).load(0, 0), stored[0])
