# Latent diagnostics

`xaig.daig.latent` asks what a model's internal channels respond to: which ones light up
over a region, where else the model looks the same, and what the main patterns are. It
grew out of the [latent space visualiser](https://github.com/ktempestuous/latent_space_visualiser_weather_models)
(Tempest, Beylich & Craig 2026,
[doi:10.1007/978-3-032-29915-4_10](https://doi.org/10.1007/978-3-032-29915-4_10); cite it
if you use this), with the science lifted out of the app so that a notebook, a batch job
and the CLI all run the same code.

!!! tip "two environments, one directory between them"

    Recording activations needs the model's own environment — torch, the framework, a
    checkpoint, usually a pinned Python. Studying them needs numpy. The
    [latent archive](#the-latent-archive) is the hand-off, so neither side installs the
    other's stack.

## Install

```console
$ uv pip install -e '.[dev,daig]'
```

Add `netcdf` to read a mask out of a reference file (see [masks](#masks)).

## What is in an archive

```console
$ xaig daig latent info latents/atmosphere
source      latents/atmosphere
model       SamudrACE-E3SMv3
component   atmosphere
checkpoint  SamudrACE-E3SMv3.tar
calendar    noleap
timestep_s  21600
grid        180x360, 64800 nodes, 64800 valid
times       17: 0425-01-03T18:00:00 .. 0425-01-09T00:00:00

layers
LAYER  CHANNELS  LABEL
0      384       encoder output (input to block 1)
1      384       Fourier block 1 output
...
8      384       Fourier block 8 output
```

## What responds in a region

Rank channels by their peak absolute activation within 1500 km of a point in the
equatorial Pacific, and fit three principal components there:

```console
$ xaig daig latent region latents/atmosphere --lat 5 --lon -140 --radius-km 1500 \
    --centred --top 6 --pcs 3
572 node(s) at layer 8, time 0425-01-03T18:00:00

RANK  CHANNEL  PEAK_ABS
1     45       2.021
2     107      1.626
3     108      1.577
4     248      1.545
5     224      1.352
6     217      1.314

PC0   25.0%  45(+0.21)  109(-0.18)  193(-0.16)  330(-0.16)  56(-0.15)  224(-0.14)

PC1   14.4%  108(+0.19)  287(+0.15)  45(-0.14)  135(-0.14)  148(-0.14)  356(+0.14)

PC2   10.6%  326(+0.21)  351(-0.19)  179(+0.19)  336(+0.15)  349(-0.14)  344(+0.14)
```

This takes 0.4 s and peaks near 210 MB on an Apple M1 Max, for an archive whose layers
total 7.6 GB: one layer at one time is ever in memory. `--json` emits the settings,
provenance and results, which is enough to rerun an analysis and to check that the rerun
agrees.

| Option | Meaning |
| --- | --- |
| `--time` | a time label, or a position (`0`, `-1`) |
| `--layer` | the layer similarity and PCA are computed at; the last by default |
| `--rank-layer` | the layer channels are ranked at; the last by default — what the network ends up emphasising |
| `--centred` | remove each channel's area-weighted global mean first |
| `--pin` | list a channel first whatever it scores, to follow it across layers |
| `--reference` | what "the region" is as one vector: the `nearest` node to its centre, or its area-weighted `mean` |

## Python API

```python
from xaig.daig.latent import Region, analyse_region, open_source

source = open_source("latents/atmosphere")
result = analyse_region(
    source,
    time="0425-01-03T18:00:00",
    layer=8,
    region=Region(lat=5, lon=-140, radius_km=1500),
    centred=True,
    n_components=3,
)

result.ranking.channels  # which channels respond in the region
result.similarity  # per node: where else the model looks like this
result.pca.top_loadings()  # per component: the channels that weigh most
result.summary()  # settings + provenance + results, JSON-ready

grid = source.grid()
first_pc = grid.to_map(result.scores[:, 0])  # (n_lat, n_lon), NaN where invalid
```

To draw any of it, `xaig.viz.map_figure(grid, values, region=...)` returns a matplotlib
figure, and [`xaig waig`](waig.md) puts the whole routine behind widgets.

The pieces are plain functions over `(n_nodes, n_channels)` arrays — `rank_channels`,
`cosine_similarity`, `fit_pca` — for when the routine above is not the question being
asked. `source.load(time, layer, channels=..., nodes=...)` reads only what it is asked for.

## Where this departs from the app, on purpose

Checked on the real SamudrACE-E3SMv3 atmosphere latents: region selection and the
uncentred ranking are identical to the app's, node for node, and the unweighted PCA
matches scikit-learn's to 1e-7. Three things differ because they should:

- **Means are area-weighted.** Rows of a lat-lon grid crowd the poles, so an unweighted
  "global mean" over-counts them. Weighting moves channel 321's global mean by 0.35 —
  40% of its standard deviation — and changes the centred top five.
- **The similarity reference is a stated policy.** The app compares against whichever
  region node comes first in the array, which for this region is its south-west corner
  at (7.5°S, 144.5°W), 13° from the centre asked for. Here it is the node nearest the
  centre, or the region's mean.
- **Invalid nodes are left out.** Over land an ocean model's activations are all zero:
  they are excluded from regions and means and read NaN in every map, with no divide
  warnings and no borrowed numbers.

PCA signs are also fixed (each component's largest loading is positive), so a map does not
flip colour between two runs of the same analysis.

## Masks

A grid's mask travels in `grid.npz` when the archive carries one. For an archive that
does not, name a variable of its reference file that is missing exactly where nodes mean
nothing — `sst` for the ocean:

```console
$ xaig daig latent info latents/ocean --mask-variable sst
...
grid        180x360, 64800 nodes, 44892 valid
```

That is 30.7% of points over land, excluded from everything that follows.

## The latent archive

A directory per model component. Any exporter that writes this layout can be read by the
`latent-archive` adapter; the SamudrACE one is the visualiser's
`scripts/extract_samudrace_latents.py`.

| File | Content |
| --- | --- |
| `manifest.json` | times, layers and provenance (below) |
| `grid.npz` | `lat`, `lon` per node, flat. Optional: `grid_shape` `(n_lat, n_lon)` for a structured grid in C order (absent for a mesh), `mask` (true where a node means something), `area` (per-node area, for meshes with uneven cells) |
| `step_XX.npy` | `(n_times, n_nodes, n_channels)`, any float dtype (float16 halves the disk), one file per layer, read memory-mapped |
| `reference.nc` | optional: physical fields on the same grid |

```json
{
  "model": "SamudrACE-E3SMv3",
  "component": "atmosphere",
  "checkpoint": "SamudrACE-E3SMv3.tar",
  "calendar": "noleap",
  "timestep_seconds": 21600,
  "n_nodes": 64800,
  "latent_times": ["0425-01-03T18:00:00", "..."],
  "steps": [
    {"index": 0, "label": "encoder output", "file": "step_00.npy", "n_channels": 384}
  ],
  "extra_steps": [],
  "reference_file": "reference.nc"
}
```

`n_nodes`, `latent_times` and `steps` (each with `index`, `file`, `n_channels`) are
required; the rest is provenance, carried into every result. Times are labels, kept as
text: emulators run on calendars (no-leap, year 425) that the usual datetime types cannot
hold. `extra_steps` are layers recorded on a coarser grid than `grid.npz` describes — the
inner levels of a U-Net — and are listed but not loadable. A file whose shape contradicts
the manifest is refused rather than misread.

## Another source of latents

`LatentSource` is three methods — `info()`, `grid()` and `load(time, layer, channels,
nodes)` — and an adapter for a different layout registers exactly like any other:

```toml
[project.entry-points."xaig.adapters"]
graphcast-latents = "mypkg.latents:GraphCastLatents"
```

```python
source = open_source("/path/to/latents", adapter="graphcast-latents")
```

Meshes need no special handling: without a `grid_shape` everything works except
`to_map`, and weights are uniform unless the adapter supplies `area`.

## Remaining tasks

- [ ] A GraphCast mesh adapter, including the app's translator
- [ ] The activation exporter as an adapter of its own, behind a framework extra
- [ ] Ensemble members and per-layer grids in `LatentSource`
