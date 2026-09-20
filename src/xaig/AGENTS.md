# xaig

Light, framework-agnostic tooling for AI campaigns. One installable toolkit, grown by
adding subpackages and adapters rather than by widening what exists.

| Subpackage | Scope | Needs |
|---|---|---|
| `core` | errors, adapter registry, the hint for a missing extra | stdlib |
| `daig` | diagnostics of emulators' internals: the latent space, on a grid | `xaig[daig]` (numpy, xarray) |
| `taig` | neural blocks trained on `daig`'s latents: a sparse autoencoder | `xaig[taig]` (torch, and `daig`'s) |
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
```

Domains never import `adapters` — they ask `core.registry` for one by name. `taig` reads
`daig` and is read by nothing: it trains on `daig.latent`'s batches and hands back a
`daig.latent.Dictionary`, so what it learns is used wherever a PCA is, by code that has
never heard of torch.

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
that parses, calls them and formats. A notebook and a batch job are clients in exactly the
same way. If it is worth testing without a terminal, it belongs in the API.

`cli.py` modules must import on the base tier — `xaig --help` imports every one of them —
so heavy imports happen inside the command that needs them.

## Extending

| To add | Do |
|---|---|
| support for a framework | a module in `adapters/` + an entry point in `pyproject.toml` |
| a diagnostic | a module in `daig/`, on `daig.grid` |
| a way of finding features | something satisfying `daig.latent.Decomposition`; if torch finds it, the block and its loop in `taig/` |
| a command | a `cli.py`, named in `_cli._COMMANDS` (or the `xaig.commands` entry-point group, from another distribution) |
| a subpackage | the directory, an extra, its row in `tests/test_purity.py`, an `AGENTS.md` |

Entry points are read from installed metadata: after editing them, rerun `uv sync`.
