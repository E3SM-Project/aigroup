# aigroup

Docs, scripts, examples, and prototypes for E3SM AI efforts.

This repo hosts two peers:

- **`docs/`** — the guide site, published at <https://e3sm-project.github.io/aigroup>
- **`src/xaig/`** — `xaig`, a light Python package for working with AI campaigns: so far,
  emulator diagnostics, latent space included (`daig`)

## Install

```console
$ uv sync
$ uv run xaig --help
```

In a checkout, `uv sync` (or the first `uv run`) installs every extra, plus pytest and
ruff. `xaig` is not on PyPI; to use it from another project, install it from this
repository, asking for the extras you need:

```console
$ uv pip install 'xaig[daig] @ git+https://github.com/E3SM-Project/aigroup'
```

The base install pulls only Click. Anything heavier sits behind an extra named
after the subpackage that needs it, so the core stays nimble.

## Use

```console
$ xaig daig latent toy scratch/toy/control        # no model to hand? make an archive with numpy
$ xaig daig latent info scratch/toy/control --mask-variable sst
$ xaig daig latent info /path/to/latents/atmosphere
```

## Develop

```console
$ uv run ruff check && uv run ruff format --check
$ uv run pytest
$ uv run --group docs mkdocs build --strict   # MKDOCS_SOCIAL=false without cairo
```

See `AGENTS.md` for how the pieces fit together, and the `AGENTS.md` in each
subdirectory for that directory's rules.
