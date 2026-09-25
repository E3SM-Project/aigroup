from __future__ import annotations

import json
import os
import shlex

import pytest
from click.testing import CliRunner

from xaig._cli import cli
from xaig.waig import cli as waig_cli
from xaig.waig.config import LATENTS_ENV, configured_latents, discover_archives

# -- the launcher works on a base install ----------------------------------


def test_launcher_names_the_extra_when_streamlit_is_absent(monkeypatch):
    monkeypatch.setattr(waig_cli, "find_spec", lambda name: None)
    result = CliRunner().invoke(cli, ["waig"])
    assert result.exit_code != 0 and "'waig' extra" in result.output


def test_launcher_hands_the_app_absolute_paths_through_the_environment(monkeypatch, tmp_path):
    seen = {}

    def call(command, env):
        seen.update(command=command, env=env)
        return 0

    monkeypatch.setattr(waig_cli, "find_spec", lambda name: object())
    monkeypatch.setattr(waig_cli.subprocess, "call", call)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "atm").mkdir()
    (tmp_path / "ocn").mkdir()
    args = ["waig", "--latents", "atm", "--latents", "ocn", "--port", "9000"]
    result = CliRunner().invoke(cli, [*args, "--headless"])
    assert result.exit_code == 0, result.output
    # The app runs elsewhere, so a path relative to here would mean nothing to it.
    assert seen["env"][LATENTS_ENV].splitlines() == [
        str((tmp_path / name).resolve()) for name in ("atm", "ocn")
    ]
    command = seen["command"]
    assert command[1:4] == ["-m", "streamlit", "run"] and command[4].endswith("waig/app.py")
    assert command[command.index("--server.port") + 1] == "9000"
    # Streamlit's default is every interface; the app reads any path it is given.
    assert command[command.index("--server.address") + 1] == "localhost"
    assert command[command.index("--server.headless") + 1] == "true"


def test_a_directory_of_archives_fills_the_drop_down(tmp_path):
    for name in ("sfno/atmosphere", "sfno/ocean", "graphcast"):
        (tmp_path / name).mkdir(parents=True)
        (tmp_path / name / "manifest.json").write_text("{}")
    sfno = tmp_path.resolve() / "sfno"
    assert discover_archives(tmp_path / "sfno") == [str(sfno / "atmosphere"), str(sfno / "ocean")]
    assert discover_archives(tmp_path / "graphcast") == [str(tmp_path.resolve() / "graphcast")]
    (tmp_path / "empty").mkdir()  # kept, so the app can say what is wrong with it
    assert discover_archives(tmp_path / "empty") == [str(tmp_path.resolve() / "empty")]


# -- the app itself, run headlessly ----------------------------------------

pytest.importorskip("streamlit")
pytest.importorskip("matplotlib")
from streamlit.testing.v1 import AppTest  # noqa: E402

from xaig.faig import maps  # noqa: E402

APP = str(waig_cli.Path(waig_cli.__file__).with_name("app.py"))


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("XAIG_NO_COASTLINES", "1")
    maps._coastlines.cache_clear()
    monkeypatch.delenv(LATENTS_ENV, raising=False)


def _run(**env) -> AppTest:
    for name, value in env.items():
        os.environ[name] = value
    return AppTest.from_file(APP, default_timeout=60).run()


def _widget(widgets, label):
    return next(w for w in widgets if w.label == label)


def _at_the_bump(at: AppTest) -> AppTest:
    """Point the controls at the toy archive's planted bump, with room for a PCA."""
    _widget(at.sidebar.number_input, "Latitude").set_value(7.5)
    _widget(at.sidebar.number_input, "Longitude").set_value(45.0)
    _widget(at.sidebar.slider, "Radius (km)").set_value(2500)
    return at.run()


def test_with_nothing_to_open_it_says_how(monkeypatch):
    at = _run()
    assert not at.exception
    assert "xaig waig --latents" in at.info[0].value


def test_the_latent_view_shows_the_analysis(monkeypatch, latent_archive):
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _at_the_bump(_run())
    assert not at.exception, at.exception
    assert "toy-emulator" in at.caption[0].value and "288 of 288 nodes valid" in at.caption[0].value
    ranking = at.dataframe[0].value
    assert list(ranking.columns) == ["rank", "channel", "peak |activation|", "pinned"]
    assert len(ranking) == 6  # the default number of channels to rank
    assert "PC0" in at.dataframe[1].value["feature"].tolist()


def test_changing_a_control_changes_the_result(monkeypatch, latent_archive):
    """Centred by default, the planted bump wins; uncentred, the constant offset does."""
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _at_the_bump(_run())
    assert at.dataframe[0].value["channel"].iloc[0] == 4
    _widget(at.sidebar.checkbox, "Centre channels").set_value(False)
    at.run()
    assert not at.exception
    assert at.dataframe[0].value["channel"].iloc[0] == 1


def test_what_is_on_screen_can_be_reproduced_off_screen(monkeypatch, latent_archive):
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _at_the_bump(_run())
    command, python = (block.value for block in at.code[:2])
    shown = json.loads(at.json[0].value)
    assert shown["settings"]["centred"] is True and shown["provenance"]["checkpoint"] == "toy.ckpt"

    # Not "looks like a command": run it, and it must say what the app says.
    argv = shlex.split(command.replace("\\\n", " "))
    assert argv[0] == "xaig"
    rerun = CliRunner().invoke(cli, [*argv[1:], "--json"])
    assert rerun.exit_code == 0, rerun.output
    assert json.loads(rerun.output) == shown

    # Likewise the Python.
    scope: dict = {}
    exec(python, scope)  # noqa: S102
    assert json.loads(json.dumps(scope["result"].summary())) == shown


def test_a_region_too_small_for_a_pca_still_shows_everything_else(monkeypatch, latent_archive):
    """The default region holds 3 of the toy grid's nodes, so two directions of variance:
    4 components cannot exist."""
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _run()
    assert not at.exception
    assert len(at.dataframe) == 1 and len(at.dataframe[0].value) == 6  # the ranking survives
    assert "at most 2 exist" in at.warning[0].value


def test_a_region_with_no_nodes_is_a_warning_not_a_crash(monkeypatch, latent_archive):
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _run()
    _widget(at.sidebar.number_input, "Longitude").set_value(7.0)
    _widget(at.sidebar.slider, "Radius (km)").set_value(100)
    at.run()
    assert not at.exception
    assert "widen the region" in at.warning[0].value


def test_a_bad_archive_is_an_error_message(monkeypatch, tmp_path):
    monkeypatch.setenv(LATENTS_ENV, str(tmp_path))
    at = _run()
    assert not at.exception
    assert "not a latent archive" in at.error[0].value


def test_the_drop_down_names_the_model_not_just_the_path(monkeypatch, latent_archive):
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _run()
    chosen = _widget(at.sidebar.selectbox, "Archive")
    assert chosen.format_func(chosen.value).startswith("toy-emulator · atmosphere")


@pytest.fixture
def basis_file(latent_archive, tmp_path):
    from xaig.daig.latent import accumulate_moments, open_source, pca_from_moments, save_basis

    moments = accumulate_moments(open_source(latent_archive), layer=2)
    return str(save_basis(tmp_path / "global.npz", pca_from_moments(moments, 3)))


def test_a_basis_file_is_a_method_like_any_other_and_reproduces(
    monkeypatch, latent_archive, basis_file
):
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _at_the_bump(_run())
    _widget(at.sidebar.selectbox, "Method").set_value("a basis file (global PCA, SAE)…")
    at.run()
    _widget(at.sidebar.text_input, "Basis file").set_value(basis_file)
    _widget(at.sidebar.number_input, "Features to map").set_value(2)
    at.run()
    assert not at.exception, at.exception
    features = at.dataframe[1].value
    assert features["feature"].tolist()[0] == "F0" and len(features) == 2
    assert "peak |activation| in the region" in features.columns

    command, python = (block.value for block in at.code[:2])
    shown = json.loads(at.json[0].value)
    assert shown["settings"]["basis"] == basis_file and "--basis" in command
    argv = shlex.split(command.replace("\\\n", " "))
    assert json.loads(CliRunner().invoke(cli, [*argv[1:], "--json"]).output) == shown
    scope: dict = {}
    exec(python, scope)  # noqa: S102
    assert json.loads(json.dumps(scope["result"].summary())) == shown


def test_an_unverified_basis_is_refused_until_it_is_allowed_and_then_reproduces(
    monkeypatch, latent_archive, tmp_path
):
    """One fitted from plain arrays says nothing of where: the app asks for a tick, as
    the command asks for a flag, and what it then shows says that it was allowed."""
    import numpy as np

    from xaig.daig.latent import fit_pca, save_basis

    anonymous = str(
        save_basis(tmp_path / "anon.npz", fit_pca(np.random.default_rng(0).normal(size=(40, 6)), 2))
    )
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _at_the_bump(_run())
    _widget(at.sidebar.selectbox, "Method").set_value("a basis file (global PCA, SAE)…")
    at.run()
    _widget(at.sidebar.text_input, "Basis file").set_value(anonymous)
    _widget(at.sidebar.number_input, "Features to map").set_value(2)
    at.run()
    assert not at.exception, at.exception
    assert any("compatibility is unverified" in note.value for note in [*at.info, *at.warning])
    refused = json.loads(at.json[0].value)  # the rest of the view survives
    assert "features" not in refused and refused["ranking"]

    _widget(at.sidebar.checkbox, "Allow an unverified basis").set_value(True)
    at.run()
    assert not at.exception, at.exception
    command, python = (block.value for block in at.code[:2])
    shown = json.loads(at.json[0].value)
    assert shown["settings"]["allow_unverified_basis"] is True and len(shown["features"]) == 2
    assert "--allow-unverified-basis" in command and "allow_unverified_basis=True" in python
    argv = shlex.split(command.replace("\\\n", " "))
    assert json.loads(CliRunner().invoke(cli, [*argv[1:], "--json"]).output) == shown
    scope: dict = {}
    exec(python, scope)  # noqa: S102
    assert json.loads(json.dumps(scope["result"].summary())) == shown


def test_a_basis_that_does_not_fit_is_said_under_features_and_the_rest_survives(
    monkeypatch, latent_archive, tmp_path
):
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _at_the_bump(_run())
    _widget(at.sidebar.selectbox, "Method").set_value("a basis file (global PCA, SAE)…")
    at.run()
    _widget(at.sidebar.text_input, "Basis file").set_value(str(tmp_path / "no-such.npz"))
    at.run()
    assert not at.exception
    assert "no basis file" in at.warning[0].value and len(at.dataframe[0].value) == 6


def test_the_region_is_followed_through_time_on_request(monkeypatch, latent_archive):
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _at_the_bump(_run())
    before = len(at.dataframe)
    _widget(at.toggle, "Follow them through time").set_value(True)
    at.run()
    assert not at.exception, at.exception
    assert len(at.dataframe) == before + 1
    through = at.dataframe[before].value
    assert through["time"].tolist() == ["0425-01-01T06:00:00", "0425-01-01T12:00:00"]
    assert "4" in through.columns  # the bump's channel is among those followed


def test_a_basis_refitted_to_the_same_name_is_the_one_shown(monkeypatch, latent_archive, tmp_path):
    """Caches were keyed on the path alone, so the app went on showing the first
    dictionary written there until it was restarted."""
    from dataclasses import replace

    from xaig.daig.latent import accumulate_moments, open_source, pca_from_moments, save_basis

    pca = pca_from_moments(accumulate_moments(open_source(latent_archive), layer=2), 3)
    path = str(save_basis(tmp_path / "refit.npz", pca))
    monkeypatch.setenv(LATENTS_ENV, str(latent_archive))
    at = _at_the_bump(_run())
    _widget(at.sidebar.selectbox, "Method").set_value("a basis file (global PCA, SAE)…")
    at.run()
    _widget(at.sidebar.text_input, "Basis file").set_value(path)
    _widget(at.sidebar.number_input, "Features to map").set_value(1)
    at.run()
    assert at.dataframe[1].value["feature"].tolist() == ["F0"]

    backwards = replace(
        pca, components=pca.components[::-1].copy(),
        explained_variance_ratio=pca.explained_variance_ratio[::-1].copy(),
    )  # fmt: skip
    save_basis(path, backwards)
    os.utime(path, ns=(1, os.stat(path).st_mtime_ns + 10**9))  # a clock too coarse to notice
    at.run()
    assert not at.exception, at.exception
    assert at.dataframe[1].value["feature"].tolist() == ["F2"]  # the bump's component moved


def test_a_url_is_kept_as_given_for_its_adapter(monkeypatch):
    url = "hf://datasets/owner/repo/atmosphere"
    assert discover_archives(url) == [url]
    monkeypatch.setenv(LATENTS_ENV, f"{url}\n/data/ocean")
    assert configured_latents() == [url, "/data/ocean"]


def test_a_basis_kept_in_an_archive_is_offered_where_it_fits(monkeypatch, latent_archive, tmp_path):
    from conftest import write_latent_archive
    from xaig.daig.latent import accumulate_moments, open_source, pca_from_moments, save_basis

    source = open_source(latent_archive)
    for layer in (1, 2):
        pca = pca_from_moments(accumulate_moments(source, layer=layer), 2)
        save_basis(latent_archive / "bases" / f"pca_L{layer:02d}.npz", pca)
    (latent_archive / "bases" / "notes.txt").write_text("not a basis")
    # A twin run of the same network is offered the control's basis files as well.
    twin = write_latent_archive(tmp_path / "twin", shift=(2, 1.0))
    monkeypatch.setenv(LATENTS_ENV, f"{latent_archive}\n{twin}")
    at = _at_the_bump(_run())
    method = _widget(at.sidebar.selectbox, "Method")
    offered = [o for o in method.options if o.startswith("pca_")]
    assert offered == ["pca_L02.npz · PCA, 2 features · latents"]  # the last layer's only
    method.set_value(offered[0])
    _widget(at.sidebar.number_input, "Features to map").set_value(2)
    at.run()
    assert not at.exception, at.exception
    assert at.dataframe[1].value["feature"].tolist() == ["F0", "F1"]
    _widget(at.sidebar.selectbox, "Archive").set_value(str(twin))
    at.run()
    assert offered[0] in _widget(at.sidebar.selectbox, "Method").options


@pytest.fixture
def with_fields(latent_archive):
    """The toy archive, with ``warmth`` (the bump's shape) and ``flat`` beside it."""
    np = pytest.importorskip("numpy")
    xarray = pytest.importorskip("xarray")
    from conftest import BUMP, LATENT_TIMES, N_LAT, N_LON

    with np.load(latent_archive / "grid.npz") as grid:
        distance = np.hypot(grid["lat"] - BUMP[0], grid["lon"] - BUMP[1])
    shape = np.exp(-((distance / 20.0) ** 2)).reshape(1, N_LAT, N_LON)
    manifest = json.loads((latent_archive / "manifest.json").read_text())
    manifest["reference_file"] = "reference.nc"
    manifest["reference_times"] = ["0425-01-01T00:00:00", *LATENT_TIMES]
    (latent_archive / "manifest.json").write_text(json.dumps(manifest))
    dims = ("time", "lat", "lon")
    xarray.Dataset(
        {"warmth": (dims, np.repeat(shape, 3, axis=0)), "flat": (dims, np.ones((3, N_LAT, N_LON)))}
    ).to_netcdf(latent_archive / "reference.nc")
    return latent_archive


def test_a_field_ranks_the_layer_and_profiles_what_follows_it(monkeypatch, with_fields):
    monkeypatch.setenv(LATENTS_ENV, str(with_fields))
    at = _run()
    assert not at.exception, at.exception
    assert "Pick a physical field" in at.info[0].value
    _widget(at.sidebar.selectbox, "Field").set_value("warmth")
    at.run()
    assert not at.exception, at.exception
    ranking = next(d.value for d in at.dataframe if "correlation" in d.value.columns)
    assert ranking["channel"].iloc[0] == 4 and ranking["correlation"].iloc[0] > 0.99
    _widget(at.toggle, "Profile it (reads 2 times)").set_value(True)
    at.run()
    assert not at.exception, at.exception
    assert any("Channel 4 is active over" in c.value for c in at.caption)

    # What the tab shows, a terminal says too.
    fields_command = next(c.value for c in at.code if "latent fields" in c.value)
    argv = shlex.split(fields_command.replace("\\\n", " "))
    rerun = CliRunner().invoke(cli, [*argv[1:], "--json"])
    assert rerun.exit_code == 0, rerun.output
    assert json.loads(rerun.output)["ranking"][0]["column"] == 4
    profile_command = next(c.value for c in at.code if "latent profile" in c.value)
    argv = shlex.split(profile_command.replace("\\\n", " "))
    rerun = CliRunner().invoke(cli, [*argv[1:], "--json"])
    assert rerun.exit_code == 0, rerun.output
    assert json.loads(rerun.output)["fields"][0]["field"] == "warmth"
