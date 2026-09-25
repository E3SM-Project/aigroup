# xaig

Light, framework-agnostic tooling for E3SM AI campaigns: what an emulator holds inside,
sparse autoencoders trained on it, figures, and a local app over them.

> **Research tool.** `xaig` is early. What is described here works; expect it to change.

## Install

The base install pulls only Click. Anything heavier sits behind an extra named after the
subpackage that needs it:

```console
$ uv pip install 'xaig[daig]'     # or: pip install 'xaig[daig]'
```

| Extra | Pulls | Gets you |
| --- | --- | --- |
| `daig` | numpy, xarray, netCDF4 | `xaig.daig`: latent diagnostics on a grid |
| `faig` | matplotlib, cartopy | `xaig.faig`: maps and series figures (brings `daig`) |
| `waig` | streamlit | `xaig waig`: a local web app (brings `faig`) |
| `taig` | torch | `xaig.taig` and `xaig taig`: sparse autoencoders (brings `daig`) |
| `hf` | huggingface_hub | archives read from a Hugging Face repository, `hf://datasets/...` (brings `daig`) |

A missing extra says so, with the command that fits how `xaig` was installed.

## Try it, with no model and no data

A toy emulator writes a latent archive with nothing but numpy; everything else reads it
like any other:

```console
$ xaig daig latent toy scratch/toy/control
$ xaig daig latent info scratch/toy/control --mask-variable sst
$ xaig daig latent region scratch/toy/control --lat 10 --lon -114 --time 2 --centred --pcs 2
$ xaig daig latent fields scratch/toy/control --field precipitation --top 3
$ xaig taig sae scratch/toy/control --features 64 --k 4 --out scratch/toy/sae.npz   # needs xaig[taig]
$ xaig waig --latents scratch/toy                                                  # needs xaig[waig]
```

The CLI is a thin client of the Python API; anything it can do, a notebook can:

```python
from xaig.daig.latent import Region, analyse_region, open_source

source = open_source("scratch/toy/control", mask_variable="sst")
result = analyse_region(
    source, time=2, layer=3, region=Region(lat=10, lon=-114, radius_km=1500), centred=True
)
result.ranking.channels  # the channels that respond most strongly there
```

## More

- Guides: <https://e3sm-project.github.io/aigroup/package/>
- Source and issues: <https://github.com/E3SM-Project/aigroup>

BSD-3-Clause. `xaig.daig.latent`, `xaig.faig` and `xaig.waig` grew out of the
[latent space visualiser for weather models](https://github.com/ktempestuous/latent_space_visualiser_weather_models)
(Tempest, Beylich & Craig 2026, arXiv:2604.20467, doi:10.1007/978-3-032-29915-4_10); see
`NOTICE`. The sparse autoencoders follow MacMillan & Ouellette (2025, arXiv:2512.24440);
the B-spline autoencoder of Cheon (2026, arXiv:2605.17493) is what `xaig.taig` is heading
for and does not implement yet.
