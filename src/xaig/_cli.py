"""Top-level ``xaig`` command.

Subcommands load lazily, so ``xaig --help`` and one command never pay for (or
require) what stands behind another.
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
_COMMANDS = {
    "daig": "xaig.daig.cli:daig",
}

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

    def invoke(self, ctx: click.Context) -> object:
        """Deliberate errors read as one clear line, not a traceback -- here, so
        that a test runner or an embedding program sees what a terminal does.
        ``--debug`` keeps the traceback."""
        try:
            return super().invoke(ctx)
        except XaigError as exc:
            if ctx.params.get("debug"):
                raise
            raise click.ClickException(str(exc)) from exc


@click.group(cls=_LazyGroup)
@click.version_option(__version__, prog_name="xaig")
@click.option("--debug", is_flag=True, help="Verbose logging.")
def cli(debug: bool) -> None:
    """Tooling for E3SM AI campaigns."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


def main() -> None:
    try:
        # Not standalone, so that an interrupt exits 130 without click's "Aborted!".
        # click then *returns* the code of a `ctx.exit(n)` instead of exiting with it.
        code = cli.main(standalone_mode=False)
    except click.ClickException as exc:
        exc.show()
        sys.exit(exc.exit_code)
    except click.Abort:
        sys.exit(130)
    if isinstance(code, int) and code:
        sys.exit(code)


if __name__ == "__main__":
    main()
