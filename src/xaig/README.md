# xaig

Light, framework-agnostic tooling for E3SM AI campaigns: what an emulator holds inside,
sparse autoencoders trained on it, figures, and a local app over them.

> **Research tool.** `xaig` is early. What is described here works; expect it to change.

## Install

The base install pulls only Click. Anything heavier sits behind an extra named after the
subpackage that needs it:

```console
$ uv pip install 'xaig[latents]'     # or: pip install 'xaig[latents]'
```

| Extra | Pulls | Gets you |
| --- | --- | --- |
| `latents` | numpy, xarray, netCDF4 | `xaig.latents`: latent diagnostics on a grid |
| `figures` | matplotlib, cartopy | `xaig.figures`: maps and series figures (brings `latents`) |
| `app` | streamlit | `xaig app`: a local web app (brings `figures`) |
| `nn` | torch | `xaig.nn` and `xaig nn`: sparse autoencoders (brings `latents`) |
| `hf` | huggingface_hub | archives read from a Hugging Face repository, `hf://datasets/...` (brings `latents`) |

A missing extra says so, with the command that fits how `xaig` was installed.

## Try it, with no model and no data

A toy emulator writes a latent archive with nothing but numpy; everything else reads it
like any other:

```console
$ xaig latents toy scratch/toy/control
$ xaig latents info scratch/toy/control --mask-variable sst
$ xaig latents region scratch/toy/control --lat 10 --lon -114 --time 2 --centred --pcs 2
$ xaig latents fields scratch/toy/control --field precipitation --top 3
$ xaig nn sae scratch/toy/control --features 64 --k 4 --out scratch/toy/sae.npz   # needs xaig[nn]
$ xaig app --latents scratch/toy                                                  # needs xaig[app]
```

The CLI is a thin client of the Python API; anything it can do, a notebook can:

```python
from xaig.latents import Region, analyse_region, open_source

source = open_source("scratch/toy/control", mask_variable="sst")
result = analyse_region(
    source, time=2, layer=3, region=Region(lat=10, lon=-114, radius_km=1500), centred=True
)
result.ranking.channels  # the channels that respond most strongly there
```

## More

- Guides: <https://e3sm-project.github.io/aigroup/package/>
- Source and issues: <https://github.com/E3SM-Project/aigroup>

BSD-3-Clause. `xaig.latents`, `xaig.figures` and `xaig.app` grew out of
the [latent space visualiser for weather models](https://github.com/ktempestuous/latent_space_visualiser_weather_models)
(Tempest, Beylich & Craig 2026, arXiv:2604.20467, doi:10.1007/978-3-032-29915-4_10); see
`NOTICE`. The sparse autoencoders follow MacMillan & Ouellette (2025, arXiv:2512.24440);
the B-spline autoencoder of Cheon (2026, arXiv:2605.17493) is what `xaig.nn` is
heading for and does not implement yet.
