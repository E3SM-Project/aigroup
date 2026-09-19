from __future__ import annotations

import json
import os
import shlex

import pytest
from click.testing import CliRunner

from xaig._cli import cli
from xaig.waig import cli as waig_cli
from xaig.waig.config import LATENTS_ENV, SOURCE_ENV, SPEC_ENV, discover_archives

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
    args = ["waig", "--latents", "atm", "--latents", "ocn", "--spec", "aug26", "--port", "9000"]
    result = CliRunner().invoke(cli, [*args, "--headless"])
    assert result.exit_code == 0, result.output
    # The app runs elsewhere, so a path relative to here would mean nothing to it.
    assert seen["env"][LATENTS_ENV].split(os.pathsep) == [
        str((tmp_path / name).resolve()) for name in ("atm", "ocn")
    ]
    assert seen["env"][SPEC_ENV] == "aug26" and SOURCE_ENV not in seen["env"]  # a name, as it came
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


def test_launcher_makes_paths_absolute_and_leaves_the_rest_alone(monkeypatch, tmp_path):
    seen = {}
    monkeypatch.setattr(waig_cli, "find_spec", lambda name: object())
    monkeypatch.setattr(waig_cli.subprocess, "call", lambda command, env: seen.update(env) or 0)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mine.yaml").write_text("name: mine")
    args = ["waig", "--spec", "mine.yaml", "--source", "https://example.org/runs.tsv"]
    assert CliRunner().invoke(cli, args).exit_code == 0
    assert seen[SPEC_ENV] == str((tmp_path / "mine.yaml").resolve())
    assert seen[SOURCE_ENV] == "https://example.org/runs.tsv"  # not a path: not mangled into one


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
    for name in (LATENTS_ENV, SPEC_ENV, SOURCE_ENV):
        monkeypatch.delenv(name, raising=False)


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


def test_the_campaign_view_shows_runs_and_what_does_not_add_up(
    monkeypatch, toy_spec_path, tmp_path
):
    table = tmp_path / "runs.csv"
    table.write_text("name,seed\nrun001-fast-d1-w08-s1,1\nrun002-fast-d2-w08-s1,99\n")
    monkeypatch.setenv(SPEC_ENV, str(toy_spec_path))
    monkeypatch.setenv(SOURCE_ENV, str(table))
    # Views are functions handed to st.navigation, so one is reached by calling it.
    script = "from xaig.waig import campaign\ncampaign.page()"
    at = AppTest.from_string(script, default_timeout=60).run()
    assert not at.exception, at.exception
    assert {m.label: m.value for m in at.metric} == {"runs": "2", "unknown": "2", "problems": "1"}
    assert at.dataframe[0].value["depth"].tolist() == [1, 2]
    assert "disagrees with the id" in at.dataframe[1].value["problem"].iloc[0]


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
