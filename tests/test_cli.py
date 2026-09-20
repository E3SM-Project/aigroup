from __future__ import annotations

import click
import pytest
from click.testing import CliRunner

from xaig import _cli
from xaig._cli import cli


@pytest.fixture
def runner():
    return CliRunner()


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
    assert listing.exit_code == 0 and "daig" in listing.output and "hello" in listing.output


def test_a_plugin_cannot_replace_a_shipped_command(runner, monkeypatch):
    impostor = click.Command("daig", callback=lambda: click.echo("impostor"))
    monkeypatch.setattr(_cli, "entry_points", lambda group: [_FakeEntryPoint("daig", impostor)])
    assert "impostor" not in runner.invoke(cli, ["daig", "latent", "--help"]).output


# -- one way out for deliberate errors ---------------------------------------


def test_deliberate_errors_are_one_line_and_debug_keeps_the_traceback(runner, tmp_path):
    """Whatever the tier: with no numpy the missing extra is the deliberate error,
    and with it the directory that holds no archive is."""
    from xaig.core.errors import XaigError

    args = ["daig", "latent", "info", str(tmp_path / "nothing-here")]
    plain = runner.invoke(cli, args)
    assert plain.exit_code == 1 and plain.output.startswith("Error: ")
    assert isinstance(runner.invoke(cli, ["--debug", *args]).exception, XaigError)


def test_an_exit_code_asked_for_inside_a_command_reaches_the_shell(monkeypatch):
    """Not standalone, click returns the code of a `ctx.exit(n)` rather than exiting."""
    monkeypatch.setattr(_cli.cli, "main", lambda **kwargs: 3)
    with pytest.raises(SystemExit) as stopped:
        _cli.main()
    assert stopped.value.code == 3
    monkeypatch.setattr(_cli.cli, "main", lambda **kwargs: None)
    _cli.main()  # and nothing to report is not an exit at all
