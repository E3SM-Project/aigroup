"""A latent archive read from a Hugging Face dataset repository, a file at a time.

The hub is faked: the suite never touches the network. The fake keeps a list of
what was downloaded, so laziness is asserted, not assumed.
"""

from __future__ import annotations

import shutil
import sys
import types

import pytest

np = pytest.importorskip("numpy")

from conftest import write_latent_archive  # noqa: E402
from xaig.core.errors import AdapterError  # noqa: E402
from xaig.daig.latent import open_source  # noqa: E402


class _EntryNotFound(Exception):
    pass


@pytest.fixture
def hub(tmp_path, monkeypatch):
    """A fake ``huggingface_hub`` serving ``tmp_path/remote`` as the dataset
    ``owner/latents`` at revision ``abc``, into a cache under ``tmp_path/cache``."""
    remote = tmp_path / "remote"
    write_latent_archive(remote / "control")
    (remote / "control" / "bases").mkdir()
    np.savez(remote / "control" / "bases" / "pca_L02.npz", placeholder=np.zeros(1))
    cache = tmp_path / "cache"
    fetched: list[str] = []
    asked = {}

    class HfApi:
        def repo_info(self, repo_id, repo_type):
            asked["repo"] = (repo_id, repo_type)
            if repo_id != "owner/latents":
                raise RuntimeError("404 Client Error")
            return types.SimpleNamespace(sha="abc")

    def hf_hub_download(repo_id, filename, repo_type, revision):
        assert (repo_id, repo_type) == ("owner/latents", "dataset")
        asked.setdefault("revisions", set()).add(revision)
        source = remote / filename
        if not source.is_file():
            raise _EntryNotFound(filename)
        target = cache / "snapshots" / revision / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy(source, target)
            fetched.append(filename)
        return str(target)

    module = types.ModuleType("huggingface_hub")
    module.HfApi, module.hf_hub_download = HfApi, hf_hub_download
    utils = types.ModuleType("huggingface_hub.utils")
    utils.EntryNotFoundError = _EntryNotFound
    module.utils = utils
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", utils)
    return types.SimpleNamespace(fetched=fetched, asked=asked)


URL = "hf://datasets/owner/latents/control"


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
    assert source.field_names() == ()  # this archive keeps no reference file


def test_every_file_comes_from_one_revision(hub):
    source = open_source(URL)
    source.load(0, 2)
    assert hub.asked["revisions"] == {"abc"}  # the one the repository was at when opened


def test_a_pinned_revision_needs_no_lookup(hub):
    open_source(URL, revision="v1").grid()
    assert "repo" not in hub.asked and hub.asked["revisions"] == {"v1"}


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
