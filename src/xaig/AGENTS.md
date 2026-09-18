# xaig

Light, framework-agnostic tooling for AI campaigns. One installable toolkit, grown by
adding subpackages and adapters rather than by widening what exists.

| Subpackage | Scope | Needs |
|---|---|---|
| `core` | run-shaped data model, shared protocols, adapter registry | stdlib |
| `caig` | campaign tracking — offline, no server, no live streaming | base tier |
| `daig` | diagnostics of emulators: their outputs and their internals | `xaig[daig]` |
| `taig` | reusable neural blocks | `xaig[taig]` |
| `adapters` | everything that knows a framework, a file layout or a scheduler | per adapter |

## Three concerns, kept apart

They are easy to conflate, and the first version of these rules did. Each has its own
answer.

1. **Framework coupling → `adapters/`.** We expect to move to systems profoundly unlike
   ACE/FME/Samudra, so no durable code may assume today's framework or artifact layout.
   Anything that does lives in an adapter, behind a protocol. Supporting a new system
   means **writing a new adapter module, never editing the code that uses it**.
2. **Scientific responsibility → the domain subpackages.** Reusable science lives where
   it is used, *with the dependencies it honestly needs*: PCA needs numpy and is not an
   adapter; a torch block belongs in `taig`. What a domain may not do is know a file
   format or a user interface.
3. **Dependency weight → extras.** `import xaig`, `xaig.caig` and `xaig --help` stay on
   PyYAML + Click. Anything heavier sits behind an extra named after the subpackage that
   needs it, and says so when it is missing (`core/extras.py`).

## Who may import whom

```
core        <-  everything; imports nothing of xaig, and no third party
adapters    ->  core, and the domain contract each one implements
caig, daig, taig  ->  core (and _render, for their cli)
```

Domains never import `adapters` — they ask `core.registry` for one by name — and never
import each other. A future front end (`waig`) sits downstream: it may import `caig` and
`daig`; nothing may import it.

The authoritative version is the `ALLOWED` and `THIRD_PARTY` tables in
`tests/test_purity.py`. Adding a subpackage without declaring its edges there fails the
suite, so the decision is always made on purpose.

## Contracts

Keep shared contracts few and small. `core/protocols.py` holds the run-shaped ones
(`Discoverer`, `StatusProbe`, `MetricSource`, `ArtifactStore`, `IdParser`). A contract with
one consumer lives next to it — `daig.latent.LatentSource` — and is promoted to core when
a second consumer appears, not before.

## API first; everything else is a client

Each subpackage has `api.py` (or plain modules) that return objects and print nothing,
and a `cli.py` that parses, calls the API and formats. A notebook, a batch job and a web
app are clients in exactly the same way. If it is worth testing without a terminal, it
belongs in the API.

`cli.py` modules must import on the base tier — `xaig --help` imports every one of them —
so heavy imports happen inside the command that needs them.

## Extending

| To add | Do |
|---|---|
| a campaign | a YAML file in `caig/campaigns/` |
| support for a framework | a module in `adapters/` + an entry point in `pyproject.toml` |
| a diagnostic | a module in `daig/`, on `daig.grid` |
| a command | a `cli.py`, named in `_cli._COMMANDS` (or the `xaig.commands` entry-point group, from another distribution) |
| a subpackage | the directory, an extra, its row in `tests/test_purity.py`, an `AGENTS.md` |

Entry points are read from installed metadata: after editing them, rerun
`uv pip install -e .`.
