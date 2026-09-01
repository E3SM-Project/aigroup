from __future__ import annotations

import pytest
from click.testing import CliRunner

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
