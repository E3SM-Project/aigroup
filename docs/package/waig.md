# The web app

`xaig waig` is a local web app over the rest of the package: an explorer for
[latent archives](latents.md) — pick a model, a region and a method.
It is presentation only — every number on screen comes from `xaig.daig` and every figure
from `xaig.faig` — so anything you see there can be redone in a notebook or a batch job,
and the app tells you how.

!!! warning "local, not a service"

    The app runs on your machine and reads what is on disk. Nothing is uploaded, there
    are no accounts, and no tracking service is involved.

## Install

```console
$ uv sync                      # in a checkout: everything
$ uv pip install 'xaig[waig] @ git+https://github.com/E3SM-Project/aigroup'   # elsewhere
```

The `waig` extra brings Streamlit and everything `faig` and `daig` need.

!!! tip "coastlines on a compute node"

    Cartopy downloads its coastline data the first time it draws, and a compute node has
    no network. Maps are then still drawn — on plain axes, with the archive's own mask
    outlined where it has one — and the app says why, after at most ten seconds: the
    fetch has a deadline, because a node that drops packets never refuses. To have
    coastlines there, fetch the data once from a login node:

    ```console
    $ python -c "from cartopy.io import shapereader as s; s.natural_earth('110m', 'physical', 'coastline')"
    ```

    Set `XAIG_NO_COASTLINES=1` to skip the attempt altogether.

## Start it

```console
$ uv run xaig waig --latents latents/atmosphere --latents latents/ocean
$ uv run xaig waig --latents latents/        # every archive directly inside it
```

Every option is optional: archives can also be opened from the sidebar.

| Option | Meaning |
| --- | --- |
| `--latents` | a latent archive, or a directory of them, to offer in the explorer; repeatable |
| `--port` | 8501 by default |
| `--address` | the interface to listen on; `localhost` by default, so the app is reachable from this machine only |
| `--headless` | do not open a browser |

!!! tip "on a remote system"

    Start it with `--headless` and reach the port the way you reach a notebook. It
    listens on `localhost` only — on a shared login node anything wider would show your
    files to everyone — which is all a tunnel or a proxy needs. Through a
    JupyterHub proxy that is `https://<hub>/user/<you>/proxy/8501/`; through SSH,
    `ssh -L 8501:localhost:8501 <host>`.

## Latents

Pick a model in the sidebar — the drop-down names each archive by the model and component
its manifest declares, so SFNO's atmosphere, its ocean and a steered twin of either are
told apart — then a time, a layer and a region. The view is
[`analyse_region`](latents.md#python-api) with widgets on it:

- **Channels** — the channels that respond most strongly in the region, and a map of
  each. *One colour scale for every map* makes them comparable by eye; otherwise each
  scales to its own range.
- **Similarity** — where else the model looks like the region, over the ranked channels
  and over all of them, on the fixed scale −1 to 1.
- **Features** — by the *Method* chosen in the sidebar: principal components fitted in
  the region and projected everywhere, or the features of a
  [basis file](latents.md#methods-a-basis-is-a-value) — a global PCA, a
  [sparse autoencoder](taig.md) — that respond most strongly there. Either way, with the
  channels that weigh most in each. A region too small for the components asked of it, or
  a basis that does not fit the layer, still shows everything else, and says why here.
  A basis that does not say which model and layer it was fitted on is refused until
  *Allow an unverified basis* is ticked, as `--allow-unverified-basis` does for the command.
- **Through time** — the region's mean of the ranked channels (or the basis's features)
  at every time the archive holds, placed by its own calendar so that a gap between kept
  steps looks like one. On request, since centred it reads every time once.
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

## The same figures without the app

```python
from xaig.daig.latent import Region, analyse_region, open_source
from xaig.faig import map_figure

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
- [ ] A run against its control (`latent diff`), and two archives side by side
- [ ] Rank channels and features against a reference field (`latent fields`)
- [ ] A PDF report of a session
