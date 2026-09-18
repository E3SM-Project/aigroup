# daig — diagnostics

What an emulator outputs, and what it holds inside.

| Module | Holds |
|---|---|
| `grid.py` | nodes on a sphere: masks, area weights, regions, maps |
| `latent/` | which internal channels respond where: ranking, similarity, PCA |
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
- **No user interface and no file formats here.** Functions take arrays or a source and
  return structured results carrying their settings and provenance. Reading is an
  adapter's job (`LatentSource`); drawing is a client's. Third-party imports are capped
  at numpy and click by `tests/test_purity.py`.
- **Mind the memory.** One layer of a 1-degree, 384-channel model is 100 MB; nine layers
  at one time is 0.9 GB. Ask a source for the nodes and channels you need, keep a layer
  in its own precision, and never make a second copy of one. `xaig daig latent region`
  on the real atmosphere archive peaks near 210 MB; the first draft took 650.
- **Deterministic results.** No dependence on node order; PCA signs are fixed.
- Plotting requires the `viz` extra. Do not import matplotlib at package import time.
- May not import `caig` or `taig`.
