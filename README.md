# aigroup

Docs, scripts, examples, and prototypes for E3SM AI efforts.

This repo hosts two peers:

- **`docs/`** — the guide site, published at <https://e3sm-project.github.io/aigroup>
- **`src/xaig/`** — `xaig`, a light Python package for campaign tracking (`caig`),
  emulator diagnostics, latent space included (`daig`), reusable neural blocks (`taig`),
  and a local web app over them (`waig`)

## Install

```console
$ uv venv --python 3.11 .venv
$ uv pip install -e '.[dev]'
```

The base install pulls only PyYAML and Click. Anything heavier sits behind an extra named
after the subpackage that needs it (`daig`, `faig`, `taig`, `waig`, plus `maps` and `netcdf`), so the core
stays nimble: `uv pip install -e '.[dev,daig]'`.

## Use

```console
$ xaig caig specs
$ xaig caig ls --spec aug26 --source /path/to/runs/MANIFEST.tsv
$ xaig caig check --spec aug26 --source /path/to/runs/MANIFEST.tsv
$ xaig daig latent info /path/to/latents/atmosphere
$ xaig daig latent region /path/to/latents/atmosphere --lat 5 --lon -140 --centred --pcs 3
$ xaig waig --latents /path/to/latents/atmosphere   # the same, in a local web app
```

## Develop

```console
$ uv run ruff check
$ uv run pytest
$ uv run mkdocs build --strict
```

See `AGENTS.md` for how the pieces fit together, and the `AGENTS.md` in each
subdirectory for that directory's rules.
