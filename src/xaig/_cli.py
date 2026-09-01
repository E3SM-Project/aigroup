"""Top-level ``xaig`` command."""

from __future__ import annotations

import logging
import sys

import click

from xaig import __version__
from xaig.caig.cli import caig
from xaig.core.errors import XaigError


@click.group()
@click.version_option(__version__, prog_name="xaig")
@click.option("--debug", is_flag=True, help="Verbose logging.")
def cli(debug: bool) -> None:
    """Tooling for E3SM AI campaigns: tracking, diagnostics, toys."""
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


cli.add_command(caig)


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
