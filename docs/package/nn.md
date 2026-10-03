# Sparse autoencoders

`xaig.nn` holds a sparse autoencoder over a model's [latents](latents.md): one node's
vector of channels goes in, a wide and mostly-zero vector of *features* comes out, and the
input is rebuilt from it. Channels are entangled; features, being few at a time, are
easier to name. It is the tool
[MacMillan & Ouellette (2025)](https://arxiv.org/abs/2512.24440) turned on GraphCast,
where it found tropical cyclones, atmospheric rivers and sea ice among the features, and
let them steer a hurricane.

What it produces is a [basis file](latents.md#methods-a-basis-is-a-value), used wherever
a PCA is — `xaig latents region`, `series`, `fields`, and the
[web app](app.md) — and readable with nothing but numpy from the environment that
runs the model.

## Install

`uv sync` leaves torch out: it is large, and whether it should be the CPU or a CUDA build
is the machine's business.

```console
$ uv sync --extra nn
$ uv pip install 'xaig[nn]'   # elsewhere, from PyPI
```

## Fit one

```console
$ xaig nn sae latents/atmosphere --layer 8 --features 1024 --out sae8.npz
  step     25  reconstruction 153.5
  step    250  reconstruction 78.66
  step    500  reconstruction 68.42
wrote sae8.npz: 1024 feature(s) of layer 8; 82.1% of the variance explained, 32.0 active per node, 0.0% dead
```

That is 11 s on an Apple M1 Max (MPS), over the layer's 1.1 million node-times, twice.
(That run, and the 82.1% above, predate the metrics being measured on the exported
dictionary; rerun it before quoting the number.) Set
against the [global PCA](latents.md#methods-a-basis-is-a-value) of the same layer, where
32 components hold 69.1% of the variance, 32 active features out of 1,024 hold 82.1%.

| Option | Meaning |
| --- | --- |
| `--layer` | the layer read; the last by default |
| `--target-layer` | write this layer instead of rebuilding the input: a *transcoder* |
| `--features` | the width of the dictionary |
| `--activation` | `topk` (the default), `relu` or `bspline` |
| `--k` | active features per node, for `topk` |
| `--l1` | the sparsity penalty, for `relu` and `bspline` |
| `--epochs`, `--batch-size`, `--lr`, `--seed` | the loop |
| `--time` | fit on some times only; repeatable |
| `--device` | `cpu`, `cuda` or `mps`; the best there is by default |

### Three ways of being sparse

- **`topk`** keeps each node's `k` largest features and needs no penalty, so the sparsity
  is exactly what was asked for from the first step. It is the default for that reason, and
  the form MacMillan & Ouellette (2025) use: TopK, with decoder directions of unit length.
  Exactly `k`: a tie at the cut goes to the lowest index, in torch and in numpy alike.
- **`relu`** gets its sparsity from an L1 penalty on the features, weighted by their decoder
  norms: the standard form, and the baseline Cheon (2026) calls LIN-SAE. The penalty takes more steps than the two-epoch default to settle: at two epochs
  this layer is rebuilt to 53.2% with 55 features active per node; with `--epochs 10`,
  64.5% with 40, in 28 s. At equal sparsity `topk` rebuilds better.
- **`bspline`** replaces the ReLU with a learnable activation per feature: zero for
  `z ≤ 0`, a uniform cubic B-spline on `(0, 6]`, a line of slope one beyond. It *starts* as
  a ReLU exactly — a B-spline whose coefficients sit at its Greville abscissae is the
  identity — and training bends each feature's own curve from there: a threshold, a
  saturation, a dead zone. The hard zero is kept whatever is learned. It takes the same
  penalty as `relu` and, as measured, answers it far more slowly: 57.9% with 213 active
  per node at two epochs, 66.5% with 156 at ten (82 s). The penalty can be paid by bending
  a curve down near zero instead of switching the feature off, and a small positive
  activation still counts as active. It is a starting point for the experiment, not a
  result; the design is one class, `BSplineActivation`, and meant to be changed.

!!! warning "`bspline` is not KAN-SAE"

    [Cheon (2026)](https://arxiv.org/abs/2605.17493) replaces the ReLU with a
    learnable cubic B-spline per feature and nothing else: no hard zero, nine control
    points that start at zero, one knot vector over the 1st–99th percentile of the
    pre-activations measured on a calibration sample, sparsity from the L1 penalty alone
    (annealed), decoder directions renormalised every step, and a feature counted alive by
    the size of its control points. The activation above was written before we read that
    paper and differs on every one of those points. Do not report it as KAN-SAE.

!!! warning "a toy loop, on purpose"

    Adam, a fixed learning rate, no resampling of dead features, no held-out times. It
    trains a useful dictionary on a laptop in seconds and says how good it is; making it
    better is what the modules being separate is for.

Inputs are centred on the layer's area-weighted mean over the times used and divided by
one number, so a node's vector has unit mean square per channel and the channels keep
their relative sizes. That standardisation travels in the file: an analysis hands the
basis raw latents. Every feature's direction is kept at unit length throughout training,
so an activation is in the same units for every feature — how much of the standardised
layer it accounts for at that node — and two features can be compared by it. Batches come from
[`iter_batches`](latents.md#python-api) — valid nodes only, drawn in proportion to area —
so the plain mean the loop takes is already the area-weighted loss.

## Python API

```python
from xaig.latents import open_source, save_basis
from xaig.nn.train import fit_sae

source = open_source("latents/atmosphere")
dictionary = fit_sae(source, layer=8, n_features=1024, activation="topk", k=32)
dictionary.meta["metrics"]  # explained_variance, mean_active_features, dead_fraction
dictionary.meta["training_metrics"]  # the same three, as the loop saw them while learning
save_basis("sae8.npz", dictionary)
```

`metrics` describes the dictionary you were given: after the last update, the loop makes one
more pass over the last epoch's batches with the final weights frozen (no gradients) and
measures that. `training_metrics` is the running tally from that epoch, taken while the
weights were still changing; after a single update it describes the model *before* it. Use
the first for results and the second to watch a run. Before training starts the moments are
checked: a layer with no variance to standardise by (every channel constant), or a
covariance that is not finite, is a `RequestError` and not a NaN loss.

The modules are plain `nn.Module`s that take and return tensors and know nothing of
archives, grids or loops, so they can be lifted into any harness:

```python
from xaig.nn.sae import BSplineActivation, SparseAutoencoder

sae = SparseAutoencoder(384, 1024, activation="bspline")
rebuilt, features = sae(x)  # x standardised, (n, 384)
total, reconstruction, features = sae.loss(x, l1=5.0)
sae.to_dictionary(input_mean=mean, input_scale=scale)  # plain arrays, for xaig.latents
```

Trained against another layer (`target_layer=`, or `sae.loss(x, target)`), the same module
is a transcoder: it reads one layer and writes a later one, and its features are steps of
computation rather than directions of representation. A cross-layer transcoder is several
of these sharing an encoder.

## From a feature to an experiment

1. Fit a dictionary, and find a feature worth naming: where it answers
   (`latents region --basis`), what it tracks (`latents fields --basis`), how it evolves
   (`latents series --basis`).
2. Read its direction in the model's environment, which needs only numpy:
   `np.load("sae8.npz")["decoder"][676]` is the unit direction feature 676 writes to
   layer 8; an activation of `a` adds `a × input_scale` of it, in the layer's own units.
3. Add a multiple of it to that layer in a forward hook, export the run as a latent
   archive whose manifest says so under `experiment`, and set it against its control:
   `xaig latents diff control steered --growth`.

Step 3's hook is the exporter's to grow; see [remaining tasks](#remaining-tasks).

## Remaining tasks

In the order we mean to take them, and after whom:

- [ ] The B-spline autoencoder as Cheon (2026) has it, replacing `bspline`, beside the
      `relu` baseline it is measured against, with that paper's table: explained variance,
      features alive and dead, mean L1 norm, redundancy between features. Two things its
      text leaves open need an answer first: what a spline does outside its knots, and
      whether the knot vector is extended past the measured range
- [ ] Held-out times: fit on some, report on others. Every number on this page is in-sample
- [ ] An auxiliary loss that revives dead features (MacMillan & Ouellette 2025, after
      [Gao et al. 2024](https://arxiv.org/abs/2406.04093)): what has not fired in a long while is made to rebuild the residual
- [ ] Steering a feature as MacMillan & Ouellette do it: keep the autoencoder's
      reconstruction error, scale one feature's activation, add the error back and let the
      model run on. The [toy emulator](latents.md) can do this in numpy today; a real model
      needs a hook in its exporter
- [ ] A sweep over `k` and the number of features, for the trade between sparsity and fidelity
- [ ] A cross-layer transcoder (several decoders on one encoder), and tracing a
      feature to its antecedents in an earlier layer, as Cheon (2026) does by correlation
- [ ] Features compared across seeds of the ablation campaign
