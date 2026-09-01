# core — the durable half

## Purity contract

**Imports: standard library and `pyyaml`. Nothing else. Ever.**

No numpy, xarray, netCDF4, torch, matplotlib, pandas, click. No `fme`. No knowledge of
NetCDF, Slurm, log formats, directory layouts, or any run-id grammar.

Core must also not import `xaig.caig`, `xaig.daig`, `xaig.taig` or `xaig.adapters` — it
may not know its consumers exist.

Enforced by `tests/test_purity.py`. If a change needs a heavier import, that change
belongs in an adapter.

## What lives here

| Module | Holds |
|---|---|
| `model.py` | `Run`, `Campaign`, `MetricSeries`, `RunStatus` |
| `protocols.py` | the four protocols, and `resolve_status` |
| `spec.py` | campaign specs: id grammar, factors, parents, metric |
| `registry.py` | adapter lookup via the `xaig.adapters` entry-point group |
| `errors.py` | `XaigError` and friends |

## Design notes worth keeping

- A `Run` is an id plus a `dict` of attributes. Core never parses an id — a spec
  declares the grammar, or an adapter attaches attributes directly. A system with
  opaque UUID ids works with zero core changes.
- `RunStatus.INTERRUPTED` is distinct from `FAILED`. A preempted-and-requeued job is
  the normal state of a live campaign; collapsing the two misreads it badly.
- `MetricSeries.at()` does not interpolate. A missing epoch reads as missing.
- Parent-map keys are compared as strings: unquoted `2:` in YAML is an int and would
  otherwise miss silently.
