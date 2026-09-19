"""``xaig waig``: start the local web app. Must import on a base install."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import click

from xaig.core.extras import missing_extra
from xaig.waig.config import LATENTS_ENV, SOURCE_ENV, SPEC_ENV, discover_archives


def _absolute_if_a_path(text: str) -> str:
    """What exists here is made absolute, since the app need not share this
    directory. Anything else -- a bundled spec's name, a URL an adapter reads --
    is handed on exactly as it came."""
    return str(Path(text).resolve()) if Path(text).exists() else text


@click.command(name="waig")
@click.option(
    "--latents",
    multiple=True,
    type=click.Path(exists=True, file_okay=False),
    help="A latent archive, or a directory of them, to offer in the explorer; repeatable.",
)
@click.option("--spec", help="Campaign spec to open: a bundled name or a path.")
@click.option("--source", help="What that campaign's adapter reads (e.g. a manifest).")
@click.option("--port", type=int, default=8501, show_default=True)
@click.option(
    "--address",
    default="localhost",
    show_default=True,
    help="Interface to listen on. The default is reachable from this machine only, which "
    "is all an SSH tunnel or a Jupyter proxy needs.",
)
@click.option(
    "--headless", is_flag=True, help="Do not open a browser (a remote or proxied session)."
)
def waig(latents, spec, source, port, address, headless) -> None:
    """Explore campaigns and latents in a local web app.

    Runs on this machine and reads what is on disk; nothing is uploaded and no
    tracking service is involved. On a remote system, pass --headless and reach
    the port through your usual tunnel or Jupyter proxy.
    """
    if find_spec("streamlit") is None:
        raise missing_extra("streamlit", "waig")
    env = dict(os.environ)
    found = [archive for path in latents for archive in discover_archives(path)]
    env[LATENTS_ENV] = os.pathsep.join(dict.fromkeys(found))
    if spec:
        env[SPEC_ENV] = _absolute_if_a_path(spec)
    if source:
        env[SOURCE_ENV] = _absolute_if_a_path(source)
    command = [
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("app.py")),
        "--server.port", str(port),
        # Streamlit's own default is every interface, and the app opens any path typed
        # into it: on a shared login node that is everyone's view of your files.
        "--server.address", address,
        "--server.headless", "true" if headless else "false",
        "--browser.gatherUsageStats", "false",
    ]  # fmt: skip
    raise SystemExit(subprocess.call(command, env=env))


__all__ = ["waig"]
