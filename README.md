# aigroup

Docs, scripts, examples, and prototypes for E3SM AI efforts.

This repo hosts two peers:

- **`docs/`** — the guide site, published at <https://e3sm-project.github.io/aigroup>
- **`src/xaig/`** — `xaig`, a light Python package for campaign tracking (`caig`),
  emulator diagnostics (`daig`), and reusable neural blocks (`taig`)

## Install

```console
$ uv venv --python 3.11 .venv
$ uv pip install -e '.[dev]'
```

The base install pulls only PyYAML and Click. Framework-specific readers live behind
extras (`fme`, `viz`, `toys`) so the core stays nimble.

## Use

```console
$ xaig caig specs
$ xaig caig ls --spec aug26 --source /path/to/runs/MANIFEST.tsv
$ xaig caig check --spec aug26 --source /path/to/runs/MANIFEST.tsv
```

## Develop

```console
$ uv run ruff check
$ uv run pytest
$ uv run mkdocs build --strict
```

See `AGENTS.md` for how the pieces fit together, and the `AGENTS.md` in each
subdirectory for that directory's rules.
