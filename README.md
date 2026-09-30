# aigroup

Docs, scripts, examples, and prototypes for E3SM AI efforts.

This repo hosts two peers:

- **`docs/`** — the guide site, published at <https://e3sm-project.github.io/aigroup>
- **`src/xaig/`** — `xaig`, a light Python package for working with AI campaigns: so far,
  emulator diagnostics, latent space included (`diagnostics`), sparse autoencoders
  trained on it (`blocks`), figures (`figures`), and a local web app over them
  (`widgets`)

## Install

```console
$ uv sync
$ uv run xaig --help
```

In a checkout, `uv sync` (or the first `uv run`) installs every extra but torch, plus
pytest and ruff; `uv sync --extra blocks` adds torch. To use it from another project,
install it from [PyPI](https://pypi.org/project/xaig/), asking for the extras you need:

```console
$ uv pip install 'xaig[diagnostics]'
$ uv pip install 'xaig[diagnostics] @ git+https://github.com/E3SM-Project/aigroup'   # what main holds and no release does yet
```

The base install pulls only Click. Anything heavier sits behind an extra named after the
subpackage that needs it (`diagnostics`, `figures`, `blocks`, `widgets`), so the core
stays nimble.

## Use

```console
$ xaig diagnostics latent toy scratch/toy/control        # no model to hand? make an archive with numpy
$ xaig diagnostics latent info scratch/toy/control --mask-variable sst
$ xaig diagnostics latent info /path/to/latents/atmosphere
$ xaig diagnostics latent region /path/to/latents/atmosphere --lat 5 --lon -140 --centred --pcs 3
$ xaig blocks sae /path/to/latents/atmosphere --out sae.npz      # needs `uv sync --extra blocks`
$ xaig diagnostics latent region /path/to/latents/atmosphere --lat 5 --lon -140 --features 3 --basis sae.npz
$ xaig widgets --latents /path/to/latents/   # the same, in a local web app
```

## Develop

```console
$ uv run ruff check && uv run ruff format --check
$ uv run pytest
$ uv run --group docs mkdocs build --strict   # MKDOCS_SOCIAL=false without cairo
```

See `AGENTS.md` for how the pieces fit together, and the `AGENTS.md` in each
subdirectory for that directory's rules.
