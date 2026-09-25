from __future__ import annotations

import json
import warnings

import pytest

np = pytest.importorskip("numpy")

from click.testing import CliRunner  # noqa: E402

from conftest import LATENT_TIMES, N_CHANNELS, N_LAT, N_LON, write_latent_archive  # noqa: E402
from xaig._cli import cli  # noqa: E402
from xaig.core import registry  # noqa: E402
from xaig.core.errors import AdapterError, RequestError  # noqa: E402
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
    with pytest.raises(RequestError, match="times are"):
        source.load("0425-02-01T00:00:00", 0)
    with pytest.raises(RequestError, match="out of range"):
        source.load(5, 0)


def test_a_directory_that_is_not_an_archive(tmp_path):
    with pytest.raises(AdapterError, match="not a latent archive"):
        open_source(tmp_path)


def test_a_path_that_is_not_there_says_so():
    """An unset shell variable turns `$A/atmosphere` into `/atmosphere`; calling that
    "not an archive" sends the reader looking for a manifest instead of a typo."""
    with pytest.raises(AdapterError, match="no such directory: /atmosphere"):
        open_source("/atmosphere")


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
    xarray = pytest.importorskip("xarray")
    mask = np.arange(N_LAT * N_LON) % 3 != 0
    path = write_latent_archive(tmp_path / "a", mask=mask, reference="reference.nc")
    sst = np.where(mask.reshape(1, N_LAT, N_LON), 1.0, np.nan)
    with warnings.catch_warnings():
        # netCDF4 1.7's *write* path trips a NumPy 2.5 deprecation of its own. Writing
        # is only this fixture's business; the reader is clean under -W error.
        warnings.simplefilter("ignore", DeprecationWarning)
        xarray.Dataset({"sst": (("time", "lat", "lon"), sst)}).to_netcdf(path / "reference.nc")
    assert open_source(path).grid().mask is None  # nothing is guessed
    assert np.array_equal(open_source(path, mask_variable="sst").grid().valid, mask)
    with pytest.raises(AdapterError, match="no variable 'ssh'"):
        open_source(path, mask_variable="ssh").grid()


def test_options_are_validated_like_any_other_adapters(latent_archive):
    with pytest.raises(AdapterError, match="mask_variabel; accepted: mask_variable"):
        open_source(latent_archive, mask_variabel="sst")


def test_an_adapter_that_cannot_supply_latents_is_refused(tmp_path):
    assert "latent-archive" in registry.available()
    registry.register("not-latents", lambda source: object())
    try:
        with pytest.raises(AdapterError, match="does not implement LatentSource"):
            open_source(tmp_path, adapter="not-latents")
    finally:
        registry.unregister("not-latents")


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


# -- written a time at a time ----------------------------------------------------


def _small_grid():
    from xaig.daig.grid import Grid

    lat, lon = np.meshgrid(np.linspace(-60.0, 60.0, 3), np.arange(4) * 90.0, indexing="ij")
    return Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(3, 4))


def _start(path, **kw):
    from xaig.adapters.latent_archive import start_archive

    options = {
        "grid": _small_grid(),
        "times": ["1950-01-01T06:00:00", "1950-01-08T12:00:00", "1950-01-15T18:00:00"],
        "layers": [("block 0 input", 5), ("block 1 input", 5)],
        "directories": ["block_00_input", "block_01_input"],
        "model": "toy",
        "checkpoint": "toy.tar",
        "calendar": "noleap",
    }
    return start_archive(path, **{**options, **kw})


def test_an_archive_filled_by_two_writers_reads_back_exactly(tmp_path):
    from xaig.adapters.latent_archive import ArchiveFiller, finish_archive

    out = _start(tmp_path / "a")
    rng = np.random.default_rng(0)
    expected = rng.normal(size=(3, 2, 12, 5)).astype(np.float32)
    # Two fillers, as two GPUs would be: disjoint times, one archive.
    first, second = ArchiveFiller(out), ArchiveFiller(out)
    for time, filler in ((0, first), (2, first), (1, second)):
        for layer in range(2):
            filler.put(time, layer, expected[time, layer])
    first.flush()
    second.flush()
    finish_archive(out)

    source = open_source(out)
    assert [x.label for x in source.info().layers] == ["block 0 input", "block 1 input"]
    assert (out / "block_01_input" / "latents.npy").is_file()
    for time in range(3):
        for layer in range(2):
            assert np.array_equal(source.load(time, layer), expected[time, layer])
    assert source.grid().shape == (3, 4)
    assert not (out / "written.npy").exists()


def test_an_archive_laid_out_with_network_layers_says_where_each_sits(tmp_path):
    from xaig.adapters.latent_archive import ArchiveFiller, finish_archive

    out = _start(tmp_path / "a", network_layers=[3, 8])
    filler = ArchiveFiller(out)
    for time in range(3):
        for layer in range(2):
            filler.put(time, layer, np.ones((12, 5)))
    filler.flush()
    finish_archive(out)
    assert [x.position for x in open_source(out).info().layers] == [3, 8]
    with pytest.raises(RequestError, match="give one each"):
        _start(tmp_path / "b", network_layers=[3])


def test_an_unfinished_archive_is_refused_and_says_what_is_left(tmp_path):
    from xaig.adapters.latent_archive import ArchiveFiller, finish_archive

    out = _start(tmp_path / "a")
    filler = ArchiveFiller(out)
    filler.put(0, 0, np.ones((12, 5)))
    filler.put(0, 1, np.ones((12, 5)))
    filler.put(1, 0, np.ones((12, 5)))  # one layer of time 1 only
    assert filler.missing() == [1, 2]
    with pytest.raises(AdapterError, match="still being written"):
        open_source(out)
    with pytest.raises(RequestError, match="2 time"):
        finish_archive(out)
    assert ArchiveFiller(out).missing() == [1, 2]  # a new filler resumes, not restarts


def test_a_filler_refuses_what_does_not_fit(tmp_path):
    from xaig.adapters.latent_archive import ArchiveFiller

    filler = ArchiveFiller(_start(tmp_path / "a", dtype="float16"))
    with pytest.raises(RequestError, match="takes"):
        filler.put(0, 0, np.ones((12, 4)))
    with pytest.raises(RequestError, match="overflows float16"):
        filler.put(0, 0, np.full((12, 5), 1e6))
    with pytest.raises(RequestError, match="no layer"):
        filler.put(0, 7, np.ones((12, 5)))
    with pytest.raises(RequestError, match="out of range"):
        filler.put(3, 0, np.ones((12, 5)))


def test_layer_directories_must_be_plain_and_distinct(tmp_path):
    with pytest.raises(RequestError, match="its own directory"):
        _start(tmp_path / "a", directories=["same", "same"])
    with pytest.raises(RequestError, match="plain relative"):
        _start(tmp_path / "b", directories=["../escape", "fine"])
    _start(tmp_path / "c")
    with pytest.raises(RequestError, match="not empty"):
        _start(tmp_path / "c")


def test_a_read_of_many_nodes_in_any_order_equals_a_slice(latent_archive):
    source = open_source(latent_archive)
    full = source.load(1, 0)
    everywhere = np.random.default_rng(0).integers(0, full.shape[0], size=2 * full.shape[0])
    assert np.array_equal(source.load(1, 0, nodes=everywhere), full[everywhere])
    assert np.array_equal(source.load(1, 0, nodes=everywhere, channels=[3, 0]),
                          full[everywhere][:, [3, 0]])  # fmt: skip
    with pytest.raises(RequestError, match="nodes x"):
        source.load(1, 0, nodes=np.append(everywhere, full.shape[0]))
