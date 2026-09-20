# The `xaig` package

`xaig` is a light, framework-agnostic package for working with AI campaigns. It lives in
this repo alongside the guides, as a peer rather than an appendix.

| Subpackage | Scope |
| --- | --- |
| `daig` | diagnostics of emulators: [what they hold inside](latents.md) |

!!! warning "research tool"

    `xaig` is early. What this page describes works; expect it to change.

## Install

```console
$ uv sync
$ uv run xaig --help
```

In a checkout, `uv sync` (or the first `uv run`) installs every extra below, plus
pytest and ruff. `xaig` is not on PyPI; to use it from another project, install it from
this repository, asking for the extras you need:

```console
$ uv pip install 'xaig[daig] @ git+https://github.com/E3SM-Project/aigroup'
```

The base install pulls only Click. Anything heavier sits behind an extra named after the
subpackage that needs it. A missing one says so, with the command that fits how this
`xaig` was installed:

```console
$ xaig daig latent info latents/atmosphere
Error: numpy is not installed; it comes with the 'daig' extra: uv pip install -e '/path/to/aigroup[daig]'  (in that checkout: `uv sync --extra daig`)
```

| Extra | Pulls | Gets you |
| --- | --- | --- |
| `daig` | numpy, xarray, netCDF4 | `xaig.daig` |

!!! tip "uv cache"

    On NERSC, keep the cache off `$HOME`:

    ```console
    $ export UV_CACHE_DIR="$PSCRATCH/.cache/uv"
    ```

## Why it is built this way

Three concerns are kept apart, because each has a different answer:

- **Framework coupling lives in adapters.** The group expects to move to systems
  profoundly unlike ACE/FME/Samudra, so everything that knows a real file layout, log
  format or scheduler lives in `xaig.adapters`, behind a small protocol. Supporting a new
  system means writing a new adapter, never editing the code that uses it. Adapters are
  found through the `xaig.adapters` entry-point group and nothing else, so one can ship
  from a completely separate package.
- **Science lives in the subpackage that uses it**, with the dependencies it honestly
  needs: needing numpy does not make something an adapter. What it may not know is a file
  format or a user interface.
- **Weight lives behind extras.** `xaig.core` depends on the standard library alone, and
  `import xaig` never pulls in the scientific stack.

Every API returns objects and prints nothing; the CLI is one client of it, a notebook
another. These rules are enforced by `tests/test_purity.py`, not by convention.

Errors xaig raises on purpose are `XaigError`s and reach a terminal as one line; anything
else is a bug and keeps its traceback, as does everything under `xaig --debug`.

## Looking inside a model

`daig.latent` reads activations recorded from inside a model, through an adapter, and
analyses them; [latent diagnostics](latents.md) is the guide to it.

```console
$ xaig daig latent info /path/to/latents/atmosphere
```

With no model and no data to hand, make an archive with the toy emulator — an MLP with a
residual stream on a small Gaussian grid, in numpy, in a few seconds — and read it back:

```console
$ xaig daig latent toy scratch/toy/control
$ xaig daig latent toy scratch/toy/steered --steer 2:7:3     # +3 on channel 7 of layer 2, every step
$ xaig daig latent info scratch/toy/steered --mask-variable sst
model                  xaig-toy
calendar               noleap
grid                   24x48, 1152 nodes, 1062 valid
times                  8: 0424-02-27T06:00:00 .. 0424-03-02T00:00:00
experiment.steer       {'layer': 2, 'channel': 7, 'by': 3.0}
...
```

The toy is shaped like a real archive where that matters to a reader: kept steps with a
gap between them, fields that begin one step before the latents, a continent the mask has
to come from, a calendar without leap days. One channel is planted to follow its storm, so
an analysis has a right answer to find.

The CLI is a thin client of the API; anything it can do, a notebook can.

```python
from xaig.daig.latent import open_source

source = open_source("latents/atmosphere")
source.info().layers  # what was recorded, without loading any of it
nodes = source.grid().within(5, -140, 1500)
source.load(0, 8, nodes=nodes)  # one region of one layer, and nothing else
```

## Reusing a fitted basis

A PCA fitted by `analyse_region(..., n_components=2)` accepts raw latents, even
when the analysis uses `centred=True`. Centring changes channel ranking and similarity;
the basis carries its own mean. Save `result.pca` with `save_basis` and reload it with
`load_basis` to reuse it through `basis=` or the region command's `--basis` option.
The file retains the fitting layer, time, region and source provenance automatically.

Reusing a basis checks its layer, model, component and checkpoint against the source.
Incomplete identity on either side requires `allow_unverified_basis=True` in the API
or `--allow-unverified-basis` in the CLI. This choice appears in the result settings;
known identity mismatches and channel-width mismatches still fail. A PCA fitted directly
from arrays with `fit_pca` has no source identity unless the caller supplies fitting
metadata when saving it.

## Adding an adapter

An adapter is one class implementing one or more protocols — for latents, `info()`,
`grid()` and `load(time, layer, …)` — constructed as `Adapter(source, **options)`.
Register it, from this repo or any other package:

```toml
[project.entry-points."xaig.adapters"]
myframework = "mypkg.adapter:MyAdapter"
```

```console
$ uv sync   # entry points are read from installed metadata
$ xaig daig latent info /path/to/export --adapter myframework
```

## Remaining tasks

- [ ] `daig`: bias and time-mean maps, spectra, zonal means (a `FieldSource` beside
      `LatentSource`, on the same `daig.grid`)
- [ ] `daig.latent`: a GraphCast mesh adapter; the activation exporter as an adapter,
      with a steering hook
