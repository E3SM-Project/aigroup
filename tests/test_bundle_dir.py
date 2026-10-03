"""The bundle adapter: what is particular to its layout, and the proof that a second layout
costs a module and an entry point.

What every adapter owes is asked in ``test_adapter_contracts.py``; here the layout is
looked at (channels first, one file per level and time, its own words), a read is shown to
be selective, the writer's refusals and the reader's are asserted, and the commands that
know nothing of bundles are run on one.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("xarray")

from click.testing import CliRunner  # noqa: E402

from xaig._cli import cli  # noqa: E402
from xaig.adapters.bundle_dir import BundleDir  # noqa: E402
from xaig.core import registry  # noqa: E402
from xaig.core.errors import AdapterError, RequestError  # noqa: E402
from xaig.latents import load_record, open_source  # noqa: E402
from xaig.latents.grid import Grid  # noqa: E402
from xaig.latents.toy import write_toy  # noqa: E402


def _write(path, **overrides):
    lat, lon = np.meshgrid(np.array([-45.0, 45.0]), np.array([0.0, 90.0, 180.0]), indexing="ij")
    grid = Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(2, 3))
    layer = np.arange(2 * 6 * 4, dtype=np.float32).reshape(2, 6, 4)  # (time, node, channel)
    contents = {"grid": grid, "times": ["a", "b"], "layers": [("only", layer)], **overrides}
    return BundleDir.write(path, **contents), layer


def test_the_layout_is_channels_first_in_its_own_words(tmp_path):
    path, layer = _write(tmp_path / "b", model="m", component="c", checkpoint="k")
    manifest = json.loads((path / "bundle.json").read_text())
    assert manifest["format"] == "xaig-bundle" and manifest["grid"] == [2, 3]
    assert manifest["system"] == {"name": "m", "part": "c", "weights": "k"}
    assert manifest["levels"] == [{"id": 0, "name": "only", "width": 4}]
    archive_words = {"steps", "latent_times", "n_nodes", "extra_steps"}
    assert not archive_words & manifest.keys()  # it has its own words
    assert sorted(p.name for p in path.iterdir()) == ["bundle.json", "coords.npz", "levels"]
    block = np.load(path / "levels" / "L00" / "T0001.npy")
    assert block.shape == (4, 2, 3)  # channels, latitudes, longitudes
    assert np.array_equal(block[:, 1, 2], layer[1, 5])  # node 5 is the last of the second row
    with np.load(path / "coords.npz") as coords:
        assert coords["lat"].tolist() == [-45.0, 45.0]
        assert coords["lon"].tolist() == [0.0, 90.0, 180.0]


def test_a_read_takes_the_channels_and_nodes_asked_for_from_a_memory_map(tmp_path):
    path, layer = _write(tmp_path / "b")
    source = BundleDir(path)
    opened = source._open("levels/L00/T0000.npy", (4, 2, 3))
    assert isinstance(opened, np.memmap)  # nothing is read until it is asked for
    assert np.array_equal(source.load(0, 0, channels=[3], nodes=[5, 0]), layer[0][[5, 0]][:, [3]])


def test_the_unmasked_option_ignores_the_mask_and_says_so(tmp_path):
    lat, lon = np.meshgrid(np.array([-45.0, 45.0]), np.array([0.0, 90.0, 180.0]), indexing="ij")
    land = np.array([True, False, False, True, True, True])
    grid = Grid(lat=lat.ravel(), lon=lon.ravel(), shape=(2, 3), mask=land)
    path, _ = _write(tmp_path / "b", grid=grid)
    assert BundleDir(path).grid().valid.tolist() == land.tolist()
    open_ = BundleDir(path, unmasked=True)
    assert open_.grid().valid.all() and open_.info().provenance()["options"] == {"unmasked": True}
    assert "options" not in BundleDir(path).info().provenance()


def test_the_writer_refuses_what_the_layout_cannot_hold(tmp_path):
    mesh = Grid(lat=np.zeros(6), lon=np.arange(6.0))
    with pytest.raises(RequestError, match="mesh"):
        _write(tmp_path / "m", grid=mesh)
    with pytest.raises(RequestError, match="shape"):
        _write(tmp_path / "s", layers=[("x", np.zeros((2, 5, 4)))])
    with pytest.raises(RequestError, match="only"):
        _write(tmp_path / "t", times=["a", "a"])
    with pytest.raises(RequestError, match="network layer"):
        _write(tmp_path / "n", network_layers=[1, 2])
    assert not any((tmp_path / name).exists() for name in "mstn")  # nothing written
    path, _ = _write(tmp_path / "b")
    with pytest.raises(RequestError, match="not empty"):
        _write(path)


def test_a_bundle_that_is_broken_is_an_adapter_error_naming_what_is_wrong(tmp_path):
    path, _ = _write(tmp_path / "b")
    manifest = json.loads((path / "bundle.json").read_text())

    def rewrite(**change):
        (path / "bundle.json").write_text(json.dumps({**manifest, **change}))

    rewrite(version=2)
    with pytest.raises(AdapterError, match="version"):
        BundleDir(path)
    rewrite(format="something-else")
    with pytest.raises(AdapterError, match="format"):
        BundleDir(path)
    (path / "bundle.json").write_text("{not json")
    with pytest.raises(AdapterError, match="unreadable manifest"):
        BundleDir(path)
    rewrite(levels=[{"id": 0, "name": "x", "width": 4}, {"id": 0, "name": "y", "width": 4}])
    with pytest.raises(AdapterError, match="more than once"):
        BundleDir(path)
    rewrite(grid=[3, 3])
    with pytest.raises(AdapterError, match="grid of"):
        BundleDir(path).grid()
    rewrite()
    (path / "levels" / "L00" / "T0001.npy").unlink()
    with pytest.raises(AdapterError, match="is missing"):
        BundleDir(path).load(1, 0)
    np.save(path / "levels" / "L00" / "T0001.npy", np.zeros((3, 2, 3), dtype=np.float32))
    with pytest.raises(AdapterError, match="manifest implies"):
        BundleDir(path).load(1, 0)
    (path / "coords.npz").unlink()
    with pytest.raises(AdapterError, match="no coords.npz"):
        BundleDir(path).grid()


def test_the_toy_emulator_writes_a_bundle_that_reads_as_its_archive_does(tmp_path):
    """One toy run through two writers: the same numbers back, to the archive's float16
    (half precision is about three decimal digits)."""
    archive = write_toy(tmp_path / "archive")
    bundle = write_toy(tmp_path / "bundle", adapter="bundle-dir")
    one = open_source(archive, mask_variable="sst")
    other = open_source(bundle, adapter="bundle-dir")
    a, b = one.info(), other.info()
    assert (a.times, a.calendar, a.timestep_seconds) == (b.times, b.calendar, b.timestep_seconds)
    assert [(x.label, x.n_channels) for x in a.layers] == [
        (x.label, x.n_channels) for x in b.layers
    ]
    assert a.experiment == b.experiment and a.identity() == b.identity()
    for layer in a.layers:
        stored, archived = other.load(2, layer.index), one.load(2, layer.index)
        assert np.allclose(archived, stored, rtol=2e-3, atol=1e-3, equal_nan=True)
    assert np.array_equal(
        one.field("precipitation", 1), other.field("precipitation", 1), equal_nan=True
    )


def test_a_new_layout_is_reached_by_every_command_without_editing_any_of_them(tmp_path):
    """The point of the adapter: nothing that consumes adapters knows this one exists."""
    assert "bundle-dir" in registry.available()
    run, bundle = CliRunner(), str(write_toy(tmp_path / "bundle", adapter="bundle-dir"))
    base = ["--adapter", "bundle-dir"]

    info = run.invoke(cli, ["latents", "info", bundle, *base])
    assert info.exit_code == 0, info.output
    assert "xaig-toy" in info.output and "noleap" in info.output

    fits = []
    for name in ("a.npz", "b.npz"):
        fit = ["latents", "pca", bundle, *base, "--components", "4", "--out", str(tmp_path / name)]
        done = run.invoke(cli, [*fit, *(x for i in range(5) for x in ("--time", str(i)))])
        assert done.exit_code == 0, done.output
        fits.append(str(tmp_path / name))
    record = tmp_path / "record.json"
    args = ["latents", "evaluate", bundle, *base, "--blocks", "4", "--stability"]
    for basis in fits:
        args += ["--basis", basis]
    done = run.invoke(cli, [*args, "--record", str(record)])
    assert done.exit_code == 0, done.output

    kept = load_record(record)
    assert kept.provenance["model"] == "xaig-toy" and kept.settings["adapter"] == "bundle-dir"
    assert "--adapter bundle-dir" in kept.command  # and the command to redo it names it
    again = run.invoke(cli, [*kept.command.split()[1:], "--json"])
    assert again.exit_code == 0, again.output
    assert (
        json.loads(again.output)["evaluations"][0]["test"] == kept.results["evaluations"][0]["test"]
    )
