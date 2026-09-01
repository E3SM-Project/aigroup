# xaig

Light, framework-agnostic tooling for AI campaigns. Three sibling subpackages:

| Subpackage | Scope |
|---|---|
| `caig` | campaign tracking — offline, no server, no live streaming |
| `daig` | emulator-vs-reference diagnostics |
| `taig` | reusable neural blocks |

## The one rule that matters

**Core knows nothing.** We expect to move to systems profoundly unlike ACE/FME/Samudra,
so no durable code may assume today's framework or artifact layout.

```
core/       protocols + data model + generic algorithms   (stdlib + pyyaml ONLY)
   ^
   | implements
adapters/   everything framework-specific lives here, and nowhere else
```

Core reaches the outside world through exactly four protocols — `Discoverer`,
`StatusProbe`, `MetricSource`, `ArtifactStore` (see `core/protocols.py`). Supporting a
new system means **writing a new adapter module, never editing core**.

What belongs in core is the reusable science, not the plumbing: the seed-spread
decision rule is statistics over a metric table and works for any framework. Log
parsing is disposable by design.

## Dependency tiers

| Install | Pulls | Gets you |
|---|---|---|
| `xaig` | pyyaml, click | core, CLI, specs |
| `xaig[fme]` | numpy, xarray, netCDF4 | the ACE/FME adapter |
| `xaig[viz]` | matplotlib | plots and reports |
| `xaig[toys]` | torch | `taig` |

Heavy imports belong in adapters, behind an extra. Never in the base tier.

## CLI and API are peers

Each subpackage has `api.py` (returns objects, prints nothing) and `cli.py` (parses and
formats, no domain logic). Neither is built on the other; both sit on core.

All of the above is enforced by `tests/test_purity.py`, not by convention alone.
