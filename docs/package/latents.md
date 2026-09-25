# Latent diagnostics

`xaig.daig.latent` asks what a model's internal channels respond to: which ones light up
over a region, where else the model looks the same, what the main patterns are, how they
evolve from one physics step to the next, and what a perturbation did to them. It
grew out of the [latent space visualiser](https://github.com/ktempestuous/latent_space_visualiser_weather_models)
(Tempest, Beylich & Craig 2026, [arXiv:2604.20467](https://arxiv.org/abs/2604.20467),
[doi:10.1007/978-3-032-29915-4_10](https://doi.org/10.1007/978-3-032-29915-4_10); cite it
if you use this), with the science lifted out of the app so that a notebook, a batch job
and the CLI all run the same code. Finding features with a learned dictionary, and testing
one by setting a steered run against its control, follows
[MacMillan & Ouellette (2025)](https://arxiv.org/abs/2512.24440).

!!! tip "two environments, one directory between them"

    Recording activations needs the model's own environment — torch, the framework, a
    checkpoint, usually a pinned Python. Studying them needs numpy. The
    [latent archive](#the-latent-archive) is the hand-off, so neither side installs the
    other's stack.

## Install

```console
$ uv sync                      # in a checkout: everything
$ uv pip install 'xaig[daig]'   # elsewhere, from PyPI
```

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

This takes 0.3 s and peaks near 275 MB resident (`/usr/bin/time -l`) on an Apple M1 Max,
for an archive whose layers total 7.6 GB: one layer at one time is ever in memory, and
everything that can be refused — no such layer, an empty region, more components than the
region's nodes can carry — is refused before the first read. `--json` emits the settings,
provenance and results, which is enough to rerun an analysis and to check that the rerun
agrees.

| Option | Meaning |
| --- | --- |
| `--time` | a time label, or a position (`0`, `-1`) |
| `--layer` | the layer similarity and the features are computed at; the last by default |
| `--rank-layer` | the layer channels are ranked at; the last by default — what the network ends up emphasising. It must be as wide as `--layer`: a channel is followed from one to the other by its index, which only means something along a residual stream |
| `--centred` | remove each channel's area-weighted global mean first |
| `--pin` | list a channel first whatever it scores, to follow it across layers |
| `--reference` | what "the region" is as one vector: the `nearest` node to its centre, or its area-weighted `mean` |
| `--pcs`, `--features` | how many features to map: principal components fitted in the region or, with `--basis`, the features of that basis which respond most strongly there |
| `--basis` | a [basis file](#methods-a-basis-is-a-value): a global PCA, a sparse autoencoder |

## Methods: a basis is a value

A PCA fitted in the region is one way to turn channels into features. A PCA fitted over
the whole globe and every time is another, and a [sparse autoencoder](taig.md) a third.
They differ in how they are found and agree in what is done with them afterwards, so all
of them are a `Decomposition` — `transform`, `directions`, `describe` — and every analysis,
the CLI and the [web app](waig.md) take one wherever they take another.

A basis is fitted once and used many times, so it has a file: one `.npz` of plain arrays
and a JSON record of how it was made.

```console
$ xaig daig latent pca latents/atmosphere --components 32 --out pca8.npz
wrote pca8.npz: 32 component(s) of layer 8 over 17 time(s), 69.1% of the variance
```

That is an area-weighted PCA over all 1.1 million node-times of the layer, from moments
accumulated a block at a time: 2.9 s, and the sums are 384 × 384 however many times there
are. It is the baseline a learned dictionary has to beat — on this layer, 32 components
hold 69.1% of the variance, and [a top-32 sparse autoencoder](taig.md) 82.1%.

```console
$ xaig daig latent region latents/atmosphere --lat 5 --lon -140 --radius-km 1500 \
    --centred --top 6 --features 3 --basis sae8.npz
...
F676  peak 27.5  45(-0.18)  107(+0.14)  351(-0.13)  124(+0.13)  129(+0.13)  326(+0.13)
F48   peak 12.3  124(+0.16)  326(-0.15)  108(+0.14)  104(+0.14)  380(+0.13)  196(+0.13)
F500  peak 11.5  280(+0.17)  351(-0.16)  332(+0.16)  211(+0.15)  22(+0.14)  114(-0.14)
```

A basis is given the raw latents whatever `--centred` says: it carries the standardisation
it was fitted with, and centring twice is simply wrong. Only the features asked for are
computed, so a map of three features out of 1,024 does not cost the other 1,021. They are
ranked by what each *contributes* in the region — its activation times the length of its
direction — because a dictionary is free to trade one for the other.

!!! warning "an index is not an identity"

    Every layer of this model is 384 channels wide, and so is every seed of a campaign, so
    a basis *fits* anywhere and means something in one place. Its file records the
    network and layer it was fitted on, and it is refused anywhere else:

    ```console
    $ xaig daig latent region latents/atmosphere --lat 5 --lon -140 --layer 4 --rank-layer 4 \
        --features 3 --basis sae8.npz
    Error: the basis (sae8.npz) was not fitted here: layer 4 against the layer 8 it was fitted on
    ```

    A basis that does not say where it was fitted — one made with `fit_pca` from plain
    arrays, say — is refused too, until you pass `--allow-unverified-basis`
    (`allow_unverified_basis=True` from Python, on `analyse_region`, `region_series` and
    `rank_by_field` alike); the choice is recorded in the result's settings. To put a basis
    to a layer it says it was *not* fitted on, strip what it says first,
    `dataclasses.replace(basis, meta={})`, and then allow it: along a residual stream, a
    dictionary from one layer can be a fair question to put to the next.

!!! tip "the way back to the model"

    The same file is the hand-off to a steering experiment. The model's environment
    needs nothing but numpy to read it — `np.load("sae8.npz")["decoder"][676]` is the
    direction feature 676 writes, and `["components"][0]` the first principal component —
    and the file's `record` says which archive, layer and times it was fitted on.

## Through time

`analyse_region` is a snapshot. To follow a region from one physics step to the next:

```console
$ xaig daig latent series latents/atmosphere --lat 5 --lon -140 --radius-km 1500 \
    --channel 45 --channel 107
TIME                 HOURS  45        107
0425-01-03T18:00:00  0      0.04793   4.694
0425-01-04T00:00:00  6      0.1133    4.762
...
0425-01-05T12:00:00  42     0.1648    4.633
0425-01-07T00:00:00  78     0.1314    4.9
```

`HOURS` comes from the archive's own calendar, by hand: positions are not lead times. This
exporter kept two runs of steps a day and a half apart, and a plot against position would
hide that. Uncentred, only the region's nodes are read — a series over every time of the
7.6 GB archive is a few MB — and `--basis … --feature N` follows a feature instead.

## A run against its control

A perturbed or steered run is set against its control node for node and channel for
channel, so the two must be one network on one grid; that is checked, not assumed. Two
archives that declare different models or checkpoints are refused — channel 42 of another
seed is not channel 42 of this one, whatever number it carries — unless `--across-models`
says the index does carry over (a fine-tune of the same weights); comparing networks
trained apart needs a different tool altogether. Only nodes valid in *both* runs are
weighed: a node one of them masks holds whatever it holds there.

```console
$ xaig daig latent diff latents/control latents/steered --layer 8
$ xaig daig latent diff latents/control latents/steered --growth
```

The first ranks channels by the area-weighted RMS of `steered − control` at one time —
which channels the change reached. `--growth` follows its size through every layer and
every time the two share, relative to the control's own spread across the globe.

### Against the model's own noise

A stochastic model -- one that draws noise at every step -- makes a node-for-node
comparison mean something only when both runs drew the *same* noise: give the exporter one
seed for the control and every experiment. Even then, a perturbation grows as the runs
drift apart, and some of what `--growth` shows after a few steps is that drift. The
yardstick is a third run: the control again, with another seed.

```console
$ xaig daig latent diff latents/control latents/steered --growth --noise latents/control-seed1
```

The second table it prints is the experiment's difference as a multiple of the one a new
noise draw makes, layer by layer and time by time: above 1, the intervention moved the
layer more than chance does. From Python, `difference_growth(control, steered,
noise=reseeded)` returns the same as `noise_rms` and `signal_to_noise`.

For two archives to be told apart at all, the exporter has to say what it did. Anything
under `experiment` in the [manifest](#the-latent-archive) — a seed, a perturbed input, a
steered channel — is shown by `latent info` and carried into the provenance of every
result, as is the way the archive was read (a mask variable).

## Feature-finding

An archive that keeps physical fields beside its latents (`reference.nc`) can say which
channels, or which features of a basis, track one of them:

```console
$ xaig daig latent fields latents/atmosphere          # the 62 fields this archive keeps
$ xaig daig latent fields latents/atmosphere --field surface_precipitation_rate --top 3
channels of layer 8 against surface_precipitation_rate at 0425-01-03T18:00:00

RANK  CHANNEL  CORRELATION
1     45       -0.588
2     248      -0.540
3     224      +0.509
$ xaig daig latent fields latents/atmosphere --field surface_precipitation_rate --top 3 \
    --basis sae8.npz
features of layer 8 against surface_precipitation_rate at 0425-01-03T18:00:00

RANK  FEATURE  CORRELATION
1     676      +0.966
2     48       +0.490
3     360      +0.469
```

No single channel of this layer follows precipitation better than |r| = 0.59; one feature
of a [sparse autoencoder trained in eleven seconds](taig.md) follows it at 0.97 — the
same feature 676 that answered most strongly in the equatorial Pacific above, built mostly
from the same channel 45. Correlation is area-weighted over valid nodes, leaves out nodes
where the field is missing, is taken at one time, and says nothing about cause: it is
where an expedition starts, and a [steering experiment](#a-run-against-its-control) is
where it ends.

## Browsing features, and what one goes with

A correlation asks about one field you already had in mind, and one number stands for a
whole map: a feature that is on over sunlit land and off everywhere else correlates only
modestly with sunlight, although that is exactly what it is. Two commands start from the
features instead.

A **census** lists every channel of a layer at one time, or every feature of a basis, with
how much of the area it is active over, its mean, its mean where active (*strength*), and
where it peaks. It is a catalogue to browse:

```console
$ xaig daig latent census latents/atmosphere --time 4 --layer 4 --basis bases/sae_L04.npz --top 3
features of layer 4 at 2015-01-04T12:00:00, by coverage

FEATURE  COVERAGE  MEAN   STRENGTH  PEAK  AT
982      0.239     2      8.38      24.5  79, 244
121      0.189     0.456  2.41      6.86  8, 134
379      0.177     1.31   7.4       24.6  -85, 288
```

A **profile** takes one of them and sets every physical field the archive keeps where it
is active against where it is not, as a difference in units of each field's own spread, so
fields of any unit share one axis. Here is the feature that followed sunlight at layer 4
with |r| = 0.42 only:

```console
$ xaig daig latent profile latents/atmosphere --layer 4 --basis bases/sae_L04.npz --feature 883 \
    --time 1 --time 6 --time 11 --time 16
feature 883 of layer 4: active over 4.3% of the area and 4 time(s)

FIELD                       EFFECT  ACTIVE     INACTIVE
SOLIN                       +1.66   994.4      323.3
LANDFRAC                    +1.17   0.7735     0.272
OCNFRAC                     -1.06   0.2253     0.6942
T_1                         -0.68   198.4      205.7
TS_input                    +0.65   297.4      286.5
```

Sunlit land: a sharper description than the correlation gave. Pool times that differ in hour
as well as day. Every fourth time of a 6-hourly run is the same hour each day, and a profile
of those alone describes the feature at that hour only: over five 18Z times, this one comes
out as land whose sensible heat flux is high.
"Active" means above `--threshold`, zero by default: *firing*, for a sparse autoencoder,
whose activations are mostly exactly zero; *positive*, for a channel or a PCA score, where
another threshold may say more. Both commands read the whole grid or, with `--lat`,
`--lon` and `--radius-km`, a region: the day side only, say. A profile says what a feature
goes with, not what it does.

## Storylines and travelling things

Two views follow something through the network and through time at once.

A **storyline** asks, for one physical field, how closely each layer follows it at each
time: the best absolute correlation any channel reaches (or any feature, for layers given
a basis). Bright at the first layer means the field comes in with the inputs; bright only
deep in the network means the network builds it.

```console
$ xaig daig latent storyline latents/atmosphere --field surface_precipitation_rate
$ xaig daig latent storyline latents/atmosphere --field surface_precipitation_rate \
    --bases 'bases/sae_L{layer:02d}.npz'
```

A time at which the field has no values -- a diagnostic output, before the model's first
step -- is left empty rather than refused.

### What a pass reads, and what it writes

Latents at a time belong to the forward pass that *starts* there: it reads the state at that
time and writes the next. A field at the same time is what the pass read -- right for an
input such as sunlight -- but for an output it is the *previous* pass's, which this one never
saw. `--lead 1` (`lead=1` in Python) sets the latents against the field one reference time
later, what the pass itself produced; `fields`, `storyline` and `profile` all take it.
Precipitation, the best SAE feature per layer:

```console
$ xaig daig latent storyline latents/atmosphere --field surface_precipitation_rate \
    --layer 0 --layer 4 --layer 6 --layer 8 --time 4 --time 7 --time 19 \
    --bases 'bases/sae_L{layer:02d}.npz' --lead 1
best |r| of any feature (basis) with surface_precipitation_rate, by layer and time

TIME                 LAYER 0  LAYER 4  LAYER 6  LAYER 8
2015-01-04T12:00:00  0.51     0.464    0.755    0.86
2015-01-05T06:00:00  0.47     0.411    0.773    0.86
2015-01-08T06:00:00  0.492    0.46     0.746    0.842
```

Without `--lead` the last column reads 0.65, 0.52 and 0.55: the rain the network builds deep
down is set against rain it did not build. The reference axis keeps every forward step, so a
lead is exact even where latents were kept for fewer, and the last latent time's output is
there too. Which fields are inputs a manifest does not say; the exporter that wrote it does.

A **Hovmoller diagram** averages one quantity over a latitude band, per longitude and
time: anything that travels draws tilted stripes, whose slope is its speed. The quantity
is a field, one channel of a layer, or one feature of a basis.

```console
$ xaig daig latent hovmoller latents/atmosphere --lat-min 40 --lat-max 60 --field V_3
$ xaig daig latent hovmoller latents/atmosphere --lat-min 40 --lat-max 60 --layer 8 --channel 45 \
    --out hovmoller.npz
```

The command prints the diagram in coarse longitude bins; `--out` keeps it whole, and
`xaig.faig.hovmoller_figure` draws it. `xaig.faig.layer_time_figure` draws a storyline,
or a `--growth` table: anything that is layers against time.

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
result.feature_info  # per feature: its label, size, and the channels that weigh most
result.summary()  # settings + provenance + results, JSON-ready

grid = source.grid()
first_pc = grid.to_map(result.scores[:, 0])  # (n_lat, n_lon), NaN where invalid
```

To draw any of it, `xaig.faig.map_figure(grid, values, region=...)` returns a matplotlib
figure, and [`xaig waig`](waig.md) puts the whole routine behind widgets.

The pieces are plain functions over `(n_nodes, n_channels)` arrays — `rank_channels`,
`cosine_similarity`, `fit_pca`, `correlate_field` — for when the routine above is not the
question being asked. `source.load(time, layer, channels=..., nodes=...)` reads only what
it is asked for. The rest of what this page shows:

```python
from xaig.daig.latent import (
    accumulate_moments,
    difference,
    difference_growth,
    iter_batches,
    load_basis,
    pca_from_moments,
    rank_by_field,
    region_series,
    save_basis,
)

moments = accumulate_moments(source, layer=8)  # area-weighted, over every time
save_basis("pca8.npz", pca_from_moments(moments, 32), provenance=source.info().provenance())

basis = load_basis("sae8.npz")
result = analyse_region(source, time=0, layer=8, region=region, n_components=3, basis=basis)
series = region_series(source, layer=8, region=region, basis=basis, features=[676])
series.elapsed_seconds  # under the archive's calendar; None when it cannot say

control, steered = open_source("latents/control"), open_source("latents/steered")
difference(control, steered, time=-1, layer=8).ranking.channels  # what the change reached
difference_growth(control, steered).relative  # (n_times, n_layers)

for batch in iter_batches(source, layer=8, batch_size=4096):  # to train on
    ...  # float32 (4096, 384): valid nodes only, drawn in proportion to area

from xaig.daig.latent import field_storyline, hovmoller

story = field_storyline(source, field="SOLIN", bases={8: basis})
story.best  # (n_times, n_layers): the best |r| of any channel, or of layer 8's features
band = hovmoller(source, lat_min=40, lat_max=60, layer=8, channel=45)
band.values  # (n_times, n_lon), with band.lon
growth = difference_growth(control, steered, noise=open_source("latents/control-seed1"))
growth.signal_to_noise  # (n_times, n_layers)

from xaig.daig.latent import feature_census, feature_profile

census = feature_census(source, time=4, layer=4, basis=basis)
census.ranked("coverage", top=20)  # or "mean", "strength", "peak"
profile = feature_profile(source, layer=4, column=883, basis=basis, times=range(1, 20, 5))
profile.fields, profile.effect  # largest effect first; xaig.faig.profile_figure draws it
```

!!! warning "area, again"

    A 1° grid has as many nodes in its last row as on the equator, covering 1/115 of the
    area, and a third of an ocean model's nodes are land. `iter_batches` draws nodes by
    area and never where the grid is invalid, so a plain mean over a batch is already the
    area-weighted loss. Anything trained on `source.load(...)` directly should do the same.

What a request cannot have — a layer that is not there, an empty region, a basis for a
different width — is a `RequestError` (an `XaigError` and a `ValueError`), raised before
anything is read; any other exception is a bug and keeps its traceback.

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
- **Invalid nodes are left out.** Over land an ocean model's activations mean nothing —
  and are not zero: at the last layer of the SamudrACE-E3SMv3 ocean they are larger than
  over the sea (RMS 0.60 against 0.50), so nothing in the latents gives them away:
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
| `bases/` | optional: basis files fitted on this archive (`xaig daig latent pca`, `xaig taig sae`), under any names. `source.files("bases")` lists them and `source.file(name)` reads one; the [app](waig.md) offers every one that fits the layer shown |

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
  "reference_file": "reference.nc",
  "reference_times": ["0425-01-03T12:00:00", "..."],
  "experiment": {"seed": 0, "steer": {"layer": 4, "basis": "sae4.npz", "feature": 12, "by": 3.0}}
}
```

`n_nodes`, `latent_times` and `steps` (each with `index`, `file`, `n_channels`) are
required; the rest is provenance, carried into every result. `experiment` is free-form:
whatever distinguishes this run from a plain one. `reference_times` labels the reference
file's time axis, which usually holds the state each forward call started from as well;
without it the file is taken to share the latents' times. Times are labels, kept as text:
emulators run on calendars (no-leap, year 425) that the usual datetime types cannot hold,
and `grid_shape` is `(n_lat, n_lon)` in C order — latitude constant along a row — which
the reader checks, because nodes stored the other way round reshape without complaint
and weight wrongly. `extra_steps` are layers recorded on a coarser grid than `grid.npz` describes — the
inner levels of a U-Net — and are listed but not loadable. A file whose shape contradicts
the manifest is refused rather than misread.

## From a Hugging Face repository

An archive kept in a Hugging Face dataset repository opens in place, with the `hf`
extra:

```console
$ uv pip install 'xaig[hf]'
$ xaig daig latent info hf://datasets/<owner>/<repo>/<folder>
```

Nothing is downloaded until something needs it, and then one file at a time, into the
Hugging Face cache: opening an archive fetches its manifest, a map its grid, a layer its
one `step_XX.npy`. A notebook that looks at one layer of a nine-layer archive downloads
one layer. Every file comes from the revision the repository was at when the archive was
opened; `open_source(url, revision="v1")` pins one. `source.file("bases/sae_L08.npz")`
fetches any other file kept in the folder, and says None when there is none. A private or
gated repository reads the token `huggingface_hub` finds (`HF_TOKEN`, or `hf auth login`).

## Another source of latents

`LatentSource` is three methods — `info()`, `grid()` and `load(time, layer, channels,
nodes)` — plus, optionally, `ReferenceFields` (`field_names()`, `field(name, time)`) for
the physical fields kept beside them. An adapter for a different layout registers exactly
like any other:

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

- [ ] A GraphCast mesh adapter, including the *translator* of Tempest et al. (2026), which
      puts an intermediate processor step in the basis of the last one
- [ ] The activation exporter as an adapter of its own, behind a framework extra — with
      the hooks that *write* a layer, for steering along a basis file's direction, and an
      `experiment` block (the SamudrACE exporter takes `--seed` and does not record it)
- [ ] Ensemble members and per-layer grids in `LatentSource`
- [ ] Differences and series in a basis's features between two runs
- [ ] How redundant a basis is: the pairwise correlation of its features' activations over
      time, the measure Cheon (2026) reports beside explained variance
- [ ] A probe for a labelled phenomenon on features against one on channels (MacMillan &
      Ouellette 2025 find a tropical-cyclone feature a probe on neurons cannot)
