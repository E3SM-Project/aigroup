"""Top-level ``xaig`` command.

Subcommands load lazily, so ``xaig --help`` and ``xaig caig ...`` never pay for
(or require) the scientific stack behind a sibling command.
"""

from __future__ import annotations

import logging
import sys
from importlib import import_module
from importlib.metadata import entry_points

import click

from xaig import __version__
from xaig.core.errors import XaigError

log = logging.getLogger(__name__)

# Shipped commands, by import path. Listing them for --help imports every cli
# module, so each must stay importable on the base tier: anything heavier is
# imported inside the command that needs it (tests/test_purity.py holds them to it).
_COMMANDS = {"caig": "xaig.caig.cli:caig"}

# A separate distribution adds a command by registering a ``click.Command``
# under this entry-point group. Shipped names win a clash.
_GROUP = "xaig.commands"


class _LazyGroup(click.Group):
    def list_commands(self, ctx: click.Context) -> list[str]:
        return sorted(set(_COMMANDS) | {ep.name for ep in entry_points(group=_GROUP)})

    def get_command(self, ctx: click.Context, name: str) -> click.Command | None:
        if name in _COMMANDS:
            module, _, attr = _COMMANDS[name].partition(":")
            return getattr(import_module(module), attr)
        for ep in entry_points(group=_GROUP):
            if ep.name == name:
                try:
                    return ep.load()
                except Exception as exc:  # one broken plugin must not take --help down
                    log.warning("could not load command %r from %s: %s", name, ep.value, exc)
        return None


@click.group(cls=_LazyGroup)
@click.version_option(__version__, prog_name="xaig")
@click.option("--debug", is_flag=True, help="Verbose logging.")
def cli(debug: bool) -> None:
    """Tooling for E3SM AI campaigns: tracking, diagnostics, toys."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    try:
        cli.main(standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except click.Abort:
        sys.exit(130)
    except XaigError as exc:
        # Deliberate errors read as one clear line, not a traceback.
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
