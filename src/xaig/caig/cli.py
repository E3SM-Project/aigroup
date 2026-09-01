"""Campaign tracking, the command half. Parsing and formatting only -- no logic."""

from __future__ import annotations

import click

from xaig import _render
from xaig.caig import api
from xaig.core import spec as spec_module
from xaig.core.errors import XaigError

_spec_option = click.option(
    "-s", "--spec", "spec_name", required=True, help="Bundled spec name or path to a spec YAML."
)
_source_option = click.option(
    "-f",
    "--source",
    type=click.Path(dir_okay=True),
    help="What the adapter reads (e.g. a manifest).",
)
_adapter_option = click.option("--adapter", help="Override the adapter named by the spec.")


def _load(spec_name: str, source: str | None, adapter: str | None):
    spec = spec_module.load(spec_name)
    return spec, api.load_campaign(spec, source=source, adapter=adapter)


def _default_columns(spec, campaign) -> list[str]:
    """Prefer the spec's own vocabulary; fall back to whatever the source had."""
    if spec.factors:
        head = [c for c in ("exp", "realm", "seed") if c in campaign.attr_names()]
        return head + [f.name for f in spec.factors]
    return campaign.attr_names()


@click.group(name="caig")
def caig() -> None:
    """Track training campaigns offline."""


@caig.command("specs")
def specs_cmd() -> None:
    """List the campaign specs bundled with xaig."""
    for name in spec_module.bundled_names():
        spec = spec_module.load(name)
        summary = spec.description.strip().splitlines()[0] if spec.description else ""
        click.echo(f"{name}  {summary}")


@caig.command("ls")
@_spec_option
@_source_option
@_adapter_option
@click.option("--select", multiple=True, metavar="KEY=VALUE", help="Filter, repeatable.")
@click.option("-c", "--columns", help="Comma-separated columns to show.")
@click.option("--sort", help="Comma-separated attributes to sort by.")
def ls_cmd(spec_name, source, adapter, select, columns, sort) -> None:
    """List runs in a campaign."""
    spec, campaign = _load(spec_name, source, adapter)
    for item in select:
        key, _, value = item.partition("=")
        if not _:
            raise click.BadParameter(f"--select expects KEY=VALUE, got {item!r}")
        campaign = campaign.filter(**{key: value})
    if sort:
        campaign = campaign.sorted_by(*[s.strip() for s in sort.split(",")])
    cols = [c.strip() for c in columns.split(",")] if columns else _default_columns(spec, campaign)
    rows = campaign.table(cols)
    if not rows:
        click.echo("no runs matched")
        return
    click.echo(_render.table(rows, ["id", "status", *cols]))
    click.echo(f"\n{len(rows)} run(s)")


@caig.command("show")
@click.argument("run_id")
@_spec_option
@_source_option
@_adapter_option
def show_cmd(run_id, spec_name, source, adapter) -> None:
    """Show one run in full."""
    _, campaign = _load(spec_name, source, adapter)
    run = campaign.get(run_id)
    if run is None:
        raise click.ClickException(f"no run {run_id!r} in campaign {campaign.name!r}")
    click.echo(_render.pairs({"id": run.id, "status": str(run.status), "location": run.location}))
    if run.attrs:
        click.echo("\nattributes")
        click.echo(_render.pairs(run.attrs))
    if run.metrics:
        click.echo("\nmetrics")
        click.echo(_render.pairs({n: f"{len(m)} point(s)" for n, m in run.metrics.items()}))


@caig.command("check")
@_spec_option
@_source_option
@_adapter_option
def check_cmd(spec_name, source, adapter) -> None:
    """Round-trip every run id through the spec and report disagreements."""
    spec, campaign = _load(spec_name, source, adapter)
    failures = api.check_ids(campaign, spec)
    if failures:
        for run_id, reason in failures:
            click.echo(f"FAIL  {run_id}: {reason}", err=True)
        raise click.ClickException(f"{len(failures)} of {len(campaign)} run id(s) disagree")
    click.echo(f"OK  {len(campaign)} run id(s) round-trip through spec {spec.name!r}")


__all__ = ["caig", "XaigError"]
