# daig — diagnostics

What an emulator outputs, and what it holds inside.

| Module | Holds |
|---|---|
| `grid.py` | nodes on a sphere: masks, area weights, regions, maps |
| `latent/source.py` | the contract: `LatentSource`, the optional `ReferenceFields`, `LatentInfo` |
| `latent/basis.py` | `Decomposition`: `PCA`, a sparse `Dictionary`, and the basis file |
| `latent/analysis.py` | one region at one time: ranking, similarity, a decomposition |
| `latent/samples.py` | many times at once: moments, a global PCA, batches to train on |
| `latent/through.py` | through time and between runs: series, differences, field correlation |
| *(to come)* | emulator-vs-reference: bias and time-mean maps, spectra, zonal means, drift |

Field and latent diagnostics share one subpackage on purpose: both stand on `grid.py`,
and siblings may not import each other.

## Rules that must not be lost

- **Area-weight everything.** Unweighted means on a lat-lon grid are simply wrong. On
  real SamudrACE latents, unweighted centring moved one channel's global mean by 40% of
  its standard deviation and changed the top-5 ranking.
- **Be NaN- and mask-aware.** Ocean channels are undefined over land (30.7% of points in
  the E3SMv3 configuration); a plain `mean` silently returns NaN or a biased number.
  Invalid nodes are left out of regions and read NaN in every map.
- **No user interface and no foreign file formats here.** Functions take arrays or a
  source and return structured results carrying their settings and provenance. Reading
  what a framework wrote is an adapter's job (`LatentSource`); drawing is a client's.
  Third-party imports are capped at numpy and click by `tests/test_purity.py`. The one
  file daig itself reads and writes is its own: the basis `.npz` (`basis.py`), plain
  arrays and a JSON record, because a basis is fitted once — often by `taig`, with torch
  — and used many times, including from the model's environment, with numpy alone.
- **A method is a value.** Anything that turns channels into features is a
  `Decomposition`; routines take one as `basis=` rather than growing an argument per
  method. A basis is handed *raw* latents: its standardisation is its own and travels
  with it, so centring an analysis must not centre its input twice.
- **Area and mask hold for training data too.** `iter_batches` draws nodes by area and
  never where the grid is invalid, so a plain mean over a batch is the area-weighted
  loss. Do not train on `source.load()` directly.
- **Refuse before reading.** Whatever can be wrong with a request is checked before the
  first 100 MB is loaded, and raised as `RequestError`.
- **Positions are not lead times.** An exporter keeps the forward calls it was asked to.
  Anything plotted through time uses `LatentInfo.elapsed_seconds()`, and falls back to
  positions knowingly when that is None.
- **Mind the memory.** One layer of a 1-degree, 384-channel model is 100 MB; nine layers
  at one time is 0.9 GB. Ask a source for the nodes and channels you need, keep a layer
  in its own precision, and never make a second copy of one. `xaig daig latent region`
  on the real atmosphere archive peaks near 210 MB; the first draft took 650.
- **Deterministic results.** No dependence on node order; PCA signs are fixed; batches
  are a function of their seed.
- Plotting requires the `faig` extra. Do not import matplotlib at package import time.
- May not import `caig` or `taig` (`taig` imports this, never the reverse).
