# core — the durable, shared half

## Purity contract

**Imports: the standard library. Nothing else. Ever.**

No numpy, xarray, netCDF4, torch, matplotlib, pandas, click, yaml. No `fme`. No knowledge
of NetCDF, Slurm, log formats or directory layouts.

Core must also not know its consumers exist: it may not import any other part of `xaig`,
**nor name one in a string**. (A `resources.files("xaig.…")` naming a consumer and a table
of built-in adapter paths were both real leaks; adapters are now found through entry
points alone.) Enforced by `tests/test_purity.py`, statically and at runtime.

## What lives here

| Module | Holds |
|---|---|
| `registry.py` | adapter lookup and construction; the factory contract |
| `extras.py` | naming the extra that provides a missing dependency |
| `errors.py` | `XaigError` and friends: `AdapterError`, `RequestError`, `MissingExtraError` |

## Design notes worth keeping

- `RequestError` is for what was asked and cannot be had. It is also a `ValueError`, but
  raise it rather than one wherever a person's input is at fault: clients show an
  `XaigError` in one line and let everything else keep its traceback.
- The registry validates what it passes to a factory. A misspelt option is an error
  naming the accepted ones; it must never silently mean "use the default".
