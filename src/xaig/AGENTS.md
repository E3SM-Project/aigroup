# xaig

Light, framework-agnostic tooling for AI campaigns. One installable toolkit, grown by
adding subpackages and adapters rather than by widening what exists.

| Subpackage | Scope | Needs |
|---|---|---|
| `core` | errors, adapter registry, the hint for a missing extra | stdlib |
| `daig` | diagnostics of emulators' internals: the latent space, on a grid | `xaig[daig]` (numpy, xarray) |
| `taig` | neural blocks trained on `daig`'s latents: a sparse autoencoder | `xaig[taig]` (torch, and `daig`'s) |
| `faig` | figures, with no web framework in them | `xaig[faig]` (matplotlib, cartopy) |
| `waig` | a local web app: presentation only | `xaig[waig]` (streamlit, and `faig`'s) |
| `adapters` | everything that knows a framework or a file layout | per adapter |

## Three concerns, kept apart

They are easy to conflate, and the first version of these rules did. Each has its own
answer.

1. **Framework coupling → `adapters/`.** We expect to move to systems profoundly unlike
   ACE/FME/Samudra, so no durable code may assume today's framework or artifact layout.
   Anything that does lives in an adapter, behind a protocol. Supporting a new system
   means **writing a new adapter module, never editing the code that uses it**.
2. **Scientific responsibility → the domain subpackages.** Reusable science lives where
   it is used, *with the dependencies it honestly needs*: needing numpy does not make
   something an adapter. What a domain may not do is know a file format or a user
   interface.
3. **Dependency weight → extras.** `import xaig` and `xaig --help` stay on Click alone.
   Anything heavier sits behind an extra named after the subpackage that needs it, and
   says so when it is missing (`core/extras.py`).

## Who may import whom

```
core        <-  everything; imports nothing of xaig, and no third party
adapters    ->  core, and the domain contract each one implements
daig        ->  core (and _render, for its cli)
taig        ->  core, daig
faig        ->  core, daig
waig        ->  core, daig, faig;  nothing imports waig
```

Domains never import `adapters` — they ask `core.registry` for one by name. `taig` reads
`daig` and is read by nothing: it trains on `daig.latent`'s batches and hands back a
`daig.latent.Dictionary`, so what it learns is used wherever a PCA is, by code that has
never heard of torch. Presentation sits downstream of the science: `faig` draws what
`daig` computes, and `waig` puts widgets on both.

The authoritative version is the `ALLOWED` and `THIRD_PARTY` tables in
`tests/test_purity.py`. Adding a subpackage without declaring its edges there fails the
suite, so the decision is always made on purpose.

## Contracts

Keep shared contracts few and small. A contract whose consumers all sit on one subpackage
lives in it — `daig.latent.LatentSource`, `ReferenceFields`, `Decomposition` — and is
promoted to core when something that does not import that subpackage needs it, not
before.

## Errors

What xaig raises on purpose is an `XaigError` (`core/errors.py`): `AdapterError`,
`MissingExtraError`, and `RequestError` for something asked that cannot be had — no such
layer, an empty region. A client shows those to whoever asked, in one line; **any other
exception is a bug and must keep its traceback**, so never catch `ValueError` or
`KeyError` wholesale to tidy a message. The top-level command does the one-line part for
every subcommand (`xaig --debug` does not).

## API first; everything else is a client

Each subpackage has plain modules that return objects and print nothing, and a `cli.py`
that parses, calls them and formats. A notebook, a batch job and the web app are clients
in exactly the same way. If it is worth testing without a terminal, it belongs in the
API.

`cli.py` modules must import on the base tier — `xaig --help` imports every one of them —
so heavy imports happen inside the command that needs them.

## Extending

| To add | Do |
|---|---|
| support for a framework | a module in `adapters/` + an entry point in `pyproject.toml` |
| a diagnostic | a module in `daig/`, on `daig.grid` |
| a way of finding features | something satisfying `daig.latent.Decomposition`; if torch finds it, the block and its loop in `taig/` |
| a figure | a function in `faig/` that returns a `Figure` |
| a view in the app | a module with a `page()` in `waig/`, and a line in `waig/app.py` |
| a command | a `cli.py`, named in `_cli._COMMANDS` (or the `xaig.commands` entry-point group, from another distribution) |
| a subpackage | the directory, an extra, its row in `tests/test_purity.py`, an `AGENTS.md` |

Entry points are read from installed metadata: after editing them, rerun `uv sync`.

## Releasing

`xaig` is published to PyPI by `.github/workflows/release.yml`, through trusted
publishing: no token exists, the indexes trust that workflow by name.

1. Bump `__version__` in `src/xaig/__init__.py` in a PR, and merge it.
2. Rehearse: run the *release* workflow by hand ("Run workflow") on `main`. It builds,
   checks the metadata, installs the wheel alone and publishes to TestPyPI.
3. Tag the merged commit `vX.Y.Z` and push the tag. The same job runs, refuses a tag
   that is not the version, and publishes to PyPI.

An index never lets a version be replaced: a bad release is fixed by the next number.
`src/xaig/README.md` is the page PyPI shows. A release holds the package, its tests and
its licences; the guide site, the run configs and these notes stay out of it.
