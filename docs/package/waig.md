# The web app

`xaig waig` is a local web app over the rest of the package: an explorer for
[latent archives](latents.md) and a view of a [campaign](index.md#tracking-a-campaign).
It is presentation only — every number on screen comes from `xaig.daig` or `xaig.caig`
and every figure from `xaig.viz` — so anything you see there can be redone in a notebook
or a batch job, and the app tells you how.

!!! warning "local, not a service"

    The app runs on your machine and reads what is on disk. Nothing is uploaded, there
    are no accounts, and no tracking service is involved.

## Install

```console
$ uv pip install -e '.[waig,maps]'
```

`waig` pulls Streamlit, matplotlib and numpy. `maps` adds cartopy for coastlines; without
it, or on a compute node where cartopy cannot fetch its coastline data, maps are still
drawn — on plain axes, with the archive's own mask outlined where it has one. Set
`XAIG_NO_COASTLINES=1` to skip the attempt.

## Start it

```console
$ xaig waig --latents latents/atmosphere --latents latents/ocean \
    --spec aug26 --source /path/to/runs/MANIFEST.tsv
```

Every option is optional: archives and campaigns can also be opened from the sidebar.

| Option | Meaning |
| --- | --- |
| `--latents` | a latent archive to offer in the explorer; repeatable |
| `--spec`, `--source` | the campaign to open: a bundled spec name or a path, and what its adapter reads |
| `--port` | 8501 by default |
| `--headless` | do not open a browser |

!!! tip "on a remote system"

    Start it with `--headless` and reach the port the way you reach a notebook. Through a
    JupyterHub proxy that is `https://<hub>/user/<you>/proxy/8501/`; through SSH,
    `ssh -L 8501:localhost:8501 <host>`.

## Latents

Pick a time, a layer and a region in the sidebar. The view is
[`analyse_region`](latents.md#python-api) with widgets on it:

- **Channels** — the channels that respond most strongly in the region, and a map of
  each. *One colour scale for every map* makes them comparable by eye; otherwise each
  scales to its own range.
- **Similarity** — where else the model looks like the region, over the ranked channels
  and over all of them, on the fixed scale −1 to 1.
- **Components** — principal components fitted in the region and projected everywhere,
  with the channels that weigh most in each. A region too small for the components asked
  of it still shows everything else, and says why here.
- **Reproduce** — the settings, the `xaig daig latent region` command and the Python
  that produce exactly what is on screen, and a JSON download of all three. The test
  suite runs that command and that code and checks they agree with the app.

Maps follow the page's theme. Signed quantities use a diverging blue–red scale symmetric
about zero — with a light midpoint on a light page and a dark one on a dark page, so that
zero always recedes — and nodes the grid marks invalid are a flat grey that belongs to no
value.

On the SamudrACE-E3SMv3 atmosphere archive (9 layers × 17 times × 64,800 nodes × 384
channels, 7.6 GB) the analysis takes 0.1 s and each map 0.07 s on an Apple M1 Max, so a
change of region redraws the default twelve maps in about a second. One layer is read per
analysis, and results and rendered maps are cached, bounded.

## Campaigns

Runs by status, the run table with the spec's own columns, and — under *What does not add
up* — the same findings as [`xaig caig check`](index.md#checking-a-campaign): ids that fail
the grammar or occur twice, and metadata that contradicts an id.

## The same figures without the app

```python
from xaig.daig.latent import Region, analyse_region, open_source
from xaig.viz import map_figure

source = open_source("latents/atmosphere")
region = Region(lat=5, lon=-140, radius_km=1500)
result = analyse_region(source, time=0, layer=8, region=region, centred=True)

fig = map_figure(
    source.grid(),
    result.similarity,
    region=region,
    title="Cosine similarity · layer 8",
    label="cosine",
    limit=1.0,
)
fig.savefig("similarity.png", dpi=150)
```

`map_figure` returns a plain `matplotlib.figure.Figure` and never touches `pyplot`.

## Remaining tasks

- [ ] A reference-field panel beside the latent maps (needs `FieldSource`)
- [ ] Click on a map to move the region
- [ ] Compare two archives side by side (a second noise seed; atmosphere against ocean)
- [ ] Metric curves and the seed-spread comparison in the campaign view
- [ ] A PDF report of a session
