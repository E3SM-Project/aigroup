"""A latent archive read from a Hugging Face dataset repository, a file at a time.

The hub is faked: the suite never touches the network. The fake keeps a list of
what was downloaded, so laziness is asserted, not assumed.
"""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from conftest import URL  # noqa: E402
from xaig.core.errors import AdapterError  # noqa: E402
from xaig.latents import open_source  # noqa: E402


def test_a_hub_archive_downloads_only_what_is_read(hub):
    source = open_source(URL)
    assert hub.fetched == ["control/manifest.json"] and hub.asked["repo"][1] == "dataset"
    assert source.info().source == URL  # provenance names the repository, not the cache
    source.grid()
    source.load(0, 1, nodes=[0, 1])
    assert hub.fetched == ["control/manifest.json", "control/grid.npz", "control/step_01.npy"]
    source.load(1, 1)
    assert len(hub.fetched) == 3  # a layer is downloaded once


def test_other_files_of_a_hub_archive(hub):
    source = open_source(URL)
    assert source.file("bases/pca_L02.npz").is_file()
    assert source.file("bases/pca_L07.npz") is None


def test_a_hub_folder_is_listed_without_downloading_it(hub):
    source = open_source(URL)
    assert source.files("bases") == ("bases/pca_L02.npz",)  # files only, not the folder in it
    assert source.files("/bases/") == ("bases/pca_L02.npz",)
    assert source.files("nothing") == ()
    assert hub.fetched == ["control/manifest.json"]
    assert source.field_names() == ()  # this archive keeps no reference file


def test_every_file_comes_from_one_revision(hub):
    source = open_source(URL)
    source.load(0, 2)
    assert hub.asked["revisions"] == {"abc"}  # the one the repository was at when opened


def test_the_default_revision_is_resolved_and_recorded(hub):
    info = open_source(URL).info()
    assert info.provenance()["revision"] == {"requested": None, "commit": "abc"}
    assert info.commit == "abc"


def test_a_tag_is_resolved_to_its_commit_and_both_are_recorded(hub):
    source = open_source(URL, revision="v1")
    source.grid()
    # The files come from the commit, not from a name that can be moved under us.
    assert hub.asked["revisions"] == {"def"}
    assert source.info().provenance()["revision"] == {"requested": "v1", "commit": "def"}

    hub.heads["v1"] = "ghi"  # the tag is moved after the fact
    assert open_source(URL, revision="v1").info().commit == "ghi"
    assert source.info().commit == "def"  # a result already made still says what it read


def test_a_full_commit_is_not_asked_of_the_hub(hub):
    sha = "0123456789abcdef0123456789abcdef01234567"
    source = open_source(URL, revision=sha)
    source.grid()  # still readable: the files come from that commit
    assert hub.asked["lookups"] == []
    assert source.info().provenance()["revision"] == {"requested": sha, "commit": sha}
    assert hub.asked["revisions"] == {sha}
    assert open_source(URL, revision="ghi").info().commit == "ghi"
    assert hub.asked["lookups"] == ["ghi"]  # a short commit, like a branch or a tag, is asked


def test_a_revision_the_hub_does_not_know_is_refused(hub):
    with pytest.raises(AdapterError, match="cannot reach.*not found"):
        open_source(URL, revision="nope")


def test_a_hub_with_no_commit_to_name_is_refused(hub):
    hub.heads["main"] = ""
    with pytest.raises(AdapterError, match="named no commit"):
        open_source(URL)


def test_the_commit_reaches_every_result_a_hub_source_makes(hub):
    from xaig.latents import Region, analyze_region, difference, region_series

    here = Region(lat=7.5, lon=45.0, radius_km=2500.0)
    source = open_source(URL, revision="v1")
    expected = {"requested": "v1", "commit": "def"}
    analyzed = analyze_region(source, time=0, layer=2, region=here)
    assert analyzed.summary()["provenance"]["revision"] == expected
    assert region_series(source, layer=2, region=here).provenance["revision"] == expected
    # A comparison names the commit of each side: the two may be different revisions.
    paired = difference(source, open_source(URL), layer=2, time=0, allow_unverified=True)
    assert paired.provenance["control"]["revision"] == expected
    assert paired.provenance["experiment"]["revision"] == {"requested": None, "commit": "abc"}


def test_the_command_line_pins_a_revision_and_reports_the_commit(hub):
    from click.testing import CliRunner

    from xaig._cli import cli

    run = CliRunner().invoke(cli, ["latents", "info", URL, "--revision", "v1"])
    assert run.exit_code == 0, run.output
    assert "def" in run.output and "v1" in run.output
    region = ["latents", "region", URL, "--time", "0", "--layer", "2", "--lat", "7.5"]
    region += ["--lon", "45", "--radius-km", "2500", "--json", "--revision", "v1"]
    run = CliRunner().invoke(cli, region)
    assert run.exit_code == 0, run.output
    assert json.loads(run.output)["provenance"]["revision"] == {"requested": "v1", "commit": "def"}
    assert hub.asked["revisions"] == {"def"}


def test_a_revision_means_nothing_to_a_local_archive(latent_archive):
    assert "revision" not in open_source(latent_archive).info().provenance()


def test_what_a_hub_source_cannot_be(hub, latent_archive):
    with pytest.raises(AdapterError, match="expected hf://datasets/<owner>/<repo>/<folder>"):
        open_source("hf://datasets/owner/latents")
    with pytest.raises(AdapterError, match="cannot reach"):
        open_source("hf://datasets/someone/else/control")
    with pytest.raises(AdapterError, match="not a latent archive"):
        open_source("hf://datasets/owner/latents/nothing")
    with pytest.raises(AdapterError, match="hf:// sources only"):
        open_source(latent_archive, revision="abc")


def test_a_local_archive_is_unchanged(latent_archive):
    source = open_source(latent_archive)
    assert source.info().source == str(latent_archive)
    assert source.file("grid.npz") == latent_archive / "grid.npz"
    assert source.file("nothing.npz") is None
    (latent_archive / "bases" / "old").mkdir(parents=True)
    (latent_archive / "bases" / "sae_L02.npz").write_bytes(b"")
    assert source.files("bases") == ("bases/sae_L02.npz",) and source.files("nothing") == ()


def test_a_revision_needs_a_hub_source_on_the_command(hub, latent_archive):
    from click.testing import CliRunner

    from xaig._cli import cli

    run = CliRunner().invoke(cli, ["latents", "info", str(latent_archive), "--revision", "v1"])
    assert run.exit_code != 0 and "--revision" in run.output and "hf://" in run.output
    diff = ["latents", "diff", URL, str(latent_archive), "--layer", "2", "--time", "0"]
    diff += ["--revision", "v1"]
    run = CliRunner().invoke(cli, diff)
    assert "--revision applies to hf://" not in run.output  # one hub source is enough


def test_a_basis_hash_needs_a_basis(latent_archive):
    from click.testing import CliRunner

    from xaig._cli import cli

    region = ["latents", "region", str(latent_archive), "--time", "0", "--layer", "2"]
    region += ["--lat", "7.5", "--lon", "45", "--radius-km", "2500", "--basis-sha256", "ab"]
    run = CliRunner().invoke(cli, region)
    assert run.exit_code != 0 and "--basis-sha256" in run.output and "--basis" in run.output
