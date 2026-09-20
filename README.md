# aigroup

Docs, scripts, examples, and prototypes for E3SM AI efforts.

This repo hosts two peers:

- **`docs/`** — the guide site, published at <https://e3sm-project.github.io/aigroup>
- **`src/xaig/`** — `xaig`, a light Python package for campaign tracking (`caig`)

## Install

```console
$ uv sync
$ uv run xaig --help
```

In a checkout, `uv sync` (or the first `uv run`) installs the package with pytest and
ruff. `xaig` is not on PyPI; to use it from another project, install it from this
repository:

```console
$ uv pip install 'xaig @ git+https://github.com/E3SM-Project/aigroup'
```

The install pulls only PyYAML and Click.

## Use

```console
$ xaig caig specs
$ xaig caig ls --spec aug26 --source /path/to/runs/MANIFEST.tsv
$ xaig caig check --spec aug26 --source /path/to/runs/MANIFEST.tsv
```

## Develop

```console
$ uv run ruff check && uv run ruff format --check
$ uv run pytest
$ uv run --group docs mkdocs build --strict   # MKDOCS_SOCIAL=false without cairo
```

See `AGENTS.md` for how the pieces fit together, and the `AGENTS.md` in each
subdirectory for that directory's rules.
