from __future__ import annotations

import json

import click
import pytest
from click.testing import CliRunner

from xaig import _cli
from xaig._cli import cli


@pytest.fixture
def runner():
    return CliRunner()


@pytest.fixture
def toy(toy_spec_path, toy_table_path):
    return ["--spec", str(toy_spec_path), "--source", str(toy_table_path)]


def test_ls_lists_runs(runner, toy):
    result = runner.invoke(cli, ["caig", "ls", *toy])
    assert result.exit_code == 0, result.output
    assert "run001-fast-d1-w08-s1" in result.output
    assert "3 run(s)" in result.output


def test_ls_shows_spec_factor_columns(runner, toy):
    result = runner.invoke(cli, ["caig", "ls", *toy])
    assert "DEPTH" in result.output and "WIDTH" in result.output


def test_ls_select_filters(runner, toy):
    result = runner.invoke(cli, ["caig", "ls", *toy, "--select", "mode=slow"])
    assert "1 run(s)" in result.output
    assert "run003" in result.output


def test_ls_select_rejects_malformed_filter(runner, toy):
    result = runner.invoke(cli, ["caig", "ls", *toy, "--select", "modeslow"])
    assert result.exit_code != 0
    assert "KEY=VALUE" in result.output


def test_ls_reports_no_match_rather_than_an_empty_table(runner, toy):
    result = runner.invoke(cli, ["caig", "ls", *toy, "--select", "mode=nope"])
    assert result.exit_code == 0
    assert "no runs matched" in result.output


def test_ls_explicit_columns(runner, toy):
    result = runner.invoke(cli, ["caig", "ls", *toy, "-c", "mode"])
    assert "MODE" in result.output and "DEPTH" not in result.output


def test_show_prints_attributes(runner, toy):
    result = runner.invoke(cli, ["caig", "show", "run001-fast-d1-w08-s1", *toy])
    assert result.exit_code == 0, result.output
    assert "baseline" in result.output


def test_show_unknown_run_is_an_error(runner, toy):
    result = runner.invoke(cli, ["caig", "show", "nope", *toy])
    assert result.exit_code != 0
    assert "no run 'nope'" in result.output


def test_check_passes_on_a_consistent_campaign(runner, toy):
    result = runner.invoke(cli, ["caig", "check", *toy])
    assert result.exit_code == 0, result.output
    assert "3 run id(s) round-trip" in result.output


def test_check_fails_loudly_on_a_bad_id(runner, toy_spec_path, tmp_path):
    """A run id disagreeing with its factors mislabels every later plot."""
    table = tmp_path / "bad.csv"
    table.write_text("name,note\nrun001-fast-d1-z09-s1,x\n")
    result = runner.invoke(
        cli, ["caig", "check", "--spec", str(toy_spec_path), "--source", str(table)]
    )
    assert result.exit_code != 0
    assert "unknown factor key" in result.output


def test_specs_lists_bundled_campaigns(runner):
    result = runner.invoke(cli, ["caig", "specs"])
    assert result.exit_code == 0
    assert "aug26" in result.output


def test_unknown_spec_names_the_alternatives(runner, toy_table_path):
    result = runner.invoke(cli, ["caig", "ls", "--spec", "nope", "--source", str(toy_table_path)])
    assert result.exit_code != 0
    assert "bundled specs" in str(result.output) + str(result.exception)


def test_default_columns_come_from_the_spec_not_from_aug26(runner, toy):
    """The id's own fields lead, in id order; nothing is hard-coded to exp/realm/seed."""
    header = runner.invoke(cli, ["caig", "ls", *toy]).output.splitlines()[0].split()
    assert header == ["ID", "STATUS", "NUM", "MODE", "SEED", "DEPTH", "WIDTH"]


def test_ls_sorts_numbers_numerically(runner, toy_spec_path, tmp_path):
    table = tmp_path / "t.csv"
    ids = [f"run00{i}-fast-d1-w{w:02d}-s1" for i, w in ((1, 16), (2, 8), (3, 32))]
    table.write_text("name\n" + "\n".join(ids) + "\n")
    args = ["caig", "ls", "--spec", str(toy_spec_path), "--source", str(table)]
    rows = json.loads(runner.invoke(cli, [*args, "--sort", "width", "--json"]).output)
    assert [r["width"] for r in rows] == [8, 16, 32]


def test_ls_can_select_on_status(runner, toy):
    result = runner.invoke(cli, ["caig", "ls", *toy, "--select", "status=unknown"])
    assert "3 run(s)" in result.output
    result = runner.invoke(cli, ["caig", "ls", *toy, "--select", "status=finished"])
    assert "no runs matched" in result.output


def test_ls_and_show_point_at_tolerated_problems(runner, toy_spec_path, tmp_path):
    table = tmp_path / "t.csv"
    table.write_text("name,seed\nrun001-fast-d1-w08-s1,99\n")
    args = ["--spec", str(toy_spec_path), "--source", str(table)]
    assert "with issues" in runner.invoke(cli, ["caig", "ls", *args]).output
    shown = runner.invoke(cli, ["caig", "show", "run001-fast-d1-w08-s1", *args]).output
    assert "metadata: column seed=99 disagrees" in shown


def test_check_fails_on_metadata_that_contradicts_the_id(runner, toy_spec_path, tmp_path):
    """An id that round-trips is not enough when the manifest says something else."""
    table = tmp_path / "t.csv"
    table.write_text("name,seed\nrun001-fast-d1-w08-s1,99\n")
    result = runner.invoke(
        cli, ["caig", "check", "--spec", str(toy_spec_path), "--source", str(table)]
    )
    assert result.exit_code != 0
    assert "metadata:" in result.output and "0 run id problem(s) and 1 metadata" in result.output


# -- the top-level command loads its subcommands lazily --------------------


class _FakeEntryPoint:
    def __init__(self, name, target):
        self.name, self.value, self._target = name, f"fake:{name}", target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


def test_a_separate_distribution_can_add_a_command(runner, monkeypatch):
    @click.command()
    def hello():
        click.echo("hello from a plugin")

    eps = [_FakeEntryPoint("hello", hello), _FakeEntryPoint("broken", ImportError("no torch"))]
    monkeypatch.setattr(_cli, "entry_points", lambda group: eps)
    assert "hello from a plugin" in runner.invoke(cli, ["hello"]).output
    listing = runner.invoke(cli, ["--help"])
    assert listing.exit_code == 0 and "caig" in listing.output and "hello" in listing.output


def test_a_plugin_cannot_replace_a_shipped_command(runner, monkeypatch):
    impostor = click.Command("caig", callback=lambda: click.echo("impostor"))
    monkeypatch.setattr(_cli, "entry_points", lambda group: [_FakeEntryPoint("caig", impostor)])
    assert "impostor" not in runner.invoke(cli, ["caig", "specs"]).output
