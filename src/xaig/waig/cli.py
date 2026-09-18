"""``xaig waig``: start the local web app. Must import on a base install."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import click

from xaig.core.extras import missing_extra
from xaig.waig.config import LATENTS_ENV, SOURCE_ENV, SPEC_ENV


@click.command(name="waig")
@click.option(
    "--latents",
    multiple=True,
    type=click.Path(exists=True, file_okay=False),
    help="A latent archive to offer in the explorer; repeatable.",
)
@click.option("--spec", help="Campaign spec to open: a bundled name or a path.")
@click.option("--source", help="What that campaign's adapter reads (e.g. a manifest).")
@click.option("--port", type=int, default=8501, show_default=True)
@click.option(
    "--headless", is_flag=True, help="Do not open a browser (a remote or proxied session)."
)
def waig(latents, spec, source, port, headless) -> None:
    """Explore campaigns and latents in a local web app.

    Runs on this machine and reads what is on disk; nothing is uploaded and no
    tracking service is involved. On a remote system, pass --headless and reach
    the port through your usual tunnel or Jupyter proxy.
    """
    if find_spec("streamlit") is None:
        raise missing_extra("streamlit", "waig")
    env = dict(os.environ)
    env[LATENTS_ENV] = os.pathsep.join(str(Path(p).resolve()) for p in latents)
    if spec:
        env[SPEC_ENV] = spec
    if source:
        env[SOURCE_ENV] = str(Path(source).resolve())
    command = [
        sys.executable, "-m", "streamlit", "run", str(Path(__file__).with_name("app.py")),
        "--server.port", str(port),
        "--server.headless", "true" if headless else "false",
        "--browser.gatherUsageStats", "false",
    ]  # fmt: skip
    raise SystemExit(subprocess.call(command, env=env))


__all__ = ["waig"]
