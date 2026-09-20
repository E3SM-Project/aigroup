# xaig

Light, framework-agnostic tooling for AI campaigns. One installable toolkit, grown by
adding subpackages rather than by widening what exists.

| Subpackage | Scope | Needs |
|---|---|---|
| `core` | errors, adapter registry, the hint for a missing extra | stdlib |

## Who may import whom

```
core        <-  everything; imports nothing of xaig, and no third party
```

The authoritative version is the `ALLOWED` and `THIRD_PARTY` tables in
`tests/test_purity.py`. Adding a subpackage without declaring its edges there fails the
suite, so the decision is always made on purpose.

## Errors

What xaig raises on purpose is an `XaigError` (`core/errors.py`): `AdapterError`,
`MissingExtraError`, and `RequestError` for something asked that cannot be had. A client
shows those to whoever asked, in one line; **any other exception is a bug and must keep
its traceback**, so never catch `ValueError` or `KeyError` wholesale to tidy a message.
The top-level command does the one-line part for every subcommand (`xaig --debug` does
not).

## Extending

| To add | Do |
|---|---|
| a command | a `cli.py`, named in `_cli._COMMANDS` (or the `xaig.commands` entry-point group, from another distribution) |
| a subpackage | the directory, its row in `tests/test_purity.py`, an `AGENTS.md` |

Entry points are read from installed metadata: after editing them, rerun `uv sync`.
