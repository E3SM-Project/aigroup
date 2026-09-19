# core — the durable, shared half

## Purity contract

**Imports: the standard library. Nothing else. Ever.**

No numpy, xarray, netCDF4, torch, matplotlib, pandas, click, yaml. No `fme`. No knowledge
of NetCDF, Slurm, log formats, directory layouts, or any run-id grammar.

Core must also not know its consumers exist: it may not import `xaig.caig`, `xaig.daig`,
`xaig.taig` or `xaig.adapters`, **nor name them in a string**. (`resources.files("xaig.caig…")`
and a table of built-in adapter paths were both real leaks; adapters are now found
through entry points alone.) Enforced by `tests/test_purity.py`, statically and at runtime.

## What lives here

| Module | Holds |
|---|---|
| `model.py` | `Run`, `Campaign`, `MetricSeries`, `RunStatus`, `Issue`, `coerce_attr` |
| `protocols.py` | the run-shaped protocols, and `resolve_status` |
| `registry.py` | adapter lookup and construction; the factory contract |
| `extras.py` | naming the extra that provides a missing dependency |
| `errors.py` | `XaigError` and friends: `SpecError`, `AdapterError`, `RequestError`, `MissingExtraError` |

Campaign specs are a campaign concept and live in `caig/spec.py`.

## Design notes worth keeping

- A `Run` is an id plus a `dict` of attributes. Core never parses an id — a spec
  declares the grammar, or an adapter attaches attributes directly. A system with
  opaque UUID ids works with zero core changes.
- `id` and `status` are **reserved**: addressable like attributes in filters, sorts and
  tables, but no attribute may take those names. A manifest column called `status` used
  to replace a run's real status in every table.
- Tolerated is not the same as forgotten. A reader that declines to raise records an
  `Issue` on the run (`id`, `metadata`, `status`), and `caig check` reports them.
- `RunStatus.INTERRUPTED` is distinct from `FAILED`. A preempted-and-requeued job is
  the normal state of a live campaign; collapsing the two misreads it badly.
- `MetricSeries.at()` does not interpolate. A missing epoch reads as missing.
- Attributes sort numerically when they are numbers (8 before 16), text after, missing
  last. They *compare* as strings, so `--select batch=16` matches the int 16.
- `RequestError` is for what was asked and cannot be had. It is also a `ValueError`, but
  raise it rather than one wherever a person's input is at fault: clients show an
  `XaigError` in one line and let everything else keep its traceback.
- The registry validates what it passes to a factory. A misspelt option is an error
  naming the accepted ones; it must never silently mean "use the default".
