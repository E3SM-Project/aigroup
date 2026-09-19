"""Campaign tracking, the command half: a thin client of ``api``.

Parsing and formatting only. Anything worth testing without a terminal belongs
in ``api``, where a notebook can reach it too.
"""

from __future__ import annotations

import json

import click
import yaml

from xaig import _render
from xaig.caig import api
from xaig.caig import spec as spec_module

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
_options_option = click.option(
    "-o",
    "--option",
    "options",
    multiple=True,
    metavar="KEY=VALUE",
    help="An adapter option, over the spec's discovery block. VALUE is YAML, so "
    "'status_map={done: finished}' is a mapping. Repeatable.",
)

# Arguments of api.load_campaign itself, which have flags of their own.
_NOT_OPTIONS = {"spec": "--spec", "source": "--source", "adapter": "--adapter", "metrics": None}


def _adapter_options(items: tuple[str, ...]) -> dict:
    options = {}
    for item in items:
        key, equals, value = item.partition("=")
        key = key.strip()
        if not equals or not key:
            raise click.BadParameter(f"--option expects KEY=VALUE, got {item!r}")
        if key in _NOT_OPTIONS:
            use = f"; use {_NOT_OPTIONS[key]}" if _NOT_OPTIONS[key] else ""
            raise click.BadParameter(f"{key!r} is not an adapter option{use}")
        try:
            options[key] = yaml.safe_load(value)
        except yaml.YAMLError as exc:
            raise click.BadParameter(
                f"--option {key}: the value is not valid YAML ({exc})"
            ) from exc
    return options


def _load(spec_name: str, source: str | None, adapter: str | None, options: tuple[str, ...] = ()):
    spec = spec_module.load(spec_name)
    campaign = api.load_campaign(spec, source=source, adapter=adapter, **_adapter_options(options))
    return spec, campaign


def _default_columns(spec, campaign) -> list[str]:
    """Prefer the spec's own vocabulary; fall back to whatever the source had."""
    if spec.factors or spec.id_fields:
        present = campaign.attr_names()
        head = [c for c in spec.id_fields if c in present]
        return head + [f.name for f in spec.factors]
    return campaign.attr_names()


def _select(campaign, items: tuple[str, ...]):
    for item in items:
        key, equals, value = item.partition("=")
        if not equals or not key:
            raise click.BadParameter(f"--select expects KEY=VALUE, got {item!r}")
        campaign = campaign.filter(**{key.strip(): value.strip()})
    return campaign


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
@_options_option
@click.option(
    "--select", multiple=True, metavar="KEY=VALUE", help="Filter (also id, status); repeatable."
)
@click.option("-c", "--columns", help="Comma-separated columns to show.")
@click.option("--sort", help="Comma-separated attributes to sort by.")
@click.option("--json", "as_json", is_flag=True, help="Emit rows as JSON, for other tools.")
def ls_cmd(spec_name, source, adapter, options, select, columns, sort, as_json) -> None:
    """List runs in a campaign."""
    spec, campaign = _load(spec_name, source, adapter, options)
    campaign = _select(campaign, select)
    if sort:
        campaign = campaign.sorted_by(*[s.strip() for s in sort.split(",")])
    cols = [c.strip() for c in columns.split(",")] if columns else _default_columns(spec, campaign)
    rows = campaign.table(cols)
    if as_json:
        click.echo(json.dumps(rows, indent=2))
        return
    if not rows:
        click.echo("no runs matched")
        return
    click.echo(_render.table(rows))
    click.echo(f"\n{len(rows)} run(s)")
    flagged = sum(1 for run in campaign if run.issues)
    if flagged:
        click.echo(f"{flagged} of them with issues; see `xaig caig check`", err=True)


@caig.command("show")
@click.argument("run_id")
@_spec_option
@_source_option
@_adapter_option
@_options_option
def show_cmd(run_id, spec_name, source, adapter, options) -> None:
    """Show one run in full."""
    _, campaign = _load(spec_name, source, adapter, options)
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
    if run.issues:
        click.echo("\nissues")
        click.echo("\n".join(f"{issue}" for issue in run.issues))


@caig.command("check")
@_spec_option
@_source_option
@_adapter_option
@_options_option
def check_cmd(spec_name, source, adapter, options) -> None:
    """Check run ids against the spec, and the source's metadata against the ids."""
    spec, campaign = _load(spec_name, source, adapter, options)
    findings = api.check_campaign(campaign, spec)
    if not findings:
        grammar = "round-trip through" if spec.id_pattern else "are unique under"
        click.echo(f"OK  {len(campaign)} run id(s) {grammar} spec {spec.name!r}")
        return
    ids = [f for f in findings if f.kind == "id"]
    rest = [f for f in findings if f.kind != "id"]
    for heading, group in (("run ids", ids), ("metadata", rest)):
        if group:
            click.echo(f"{heading}:", err=True)
            for f in group:
                click.echo(f"  FAIL  {f.run_id}: {f.message}", err=True)
    raise click.ClickException(
        f"{len(ids)} run id problem(s) and {len(rest)} metadata problem(s) "
        f"across {len(campaign)} run(s)"
    )


__all__ = ["caig"]
