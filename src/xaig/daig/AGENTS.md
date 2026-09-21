# daig — diagnostics

What an emulator holds inside.

| Module | Holds |
|---|---|
| `grid.py` | nodes on a sphere: masks, area weights, regions, maps |
| `latent/source.py` | the contract: `LatentSource`, the optional `ReferenceFields`, `LatentInfo` |
| `latent/toy.py` | a toy emulator in numpy, so an archive can be made with no model and no data |
| `latent/basis.py` | `Decomposition`: `PCA`, a sparse `Dictionary`, and the basis file |
| `latent/analysis.py` | one region at one time: ranking, similarity, a decomposition |

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
  arrays and a JSON record, because a basis is fitted once and used many times,
  including from the model's environment, with numpy alone.
- **An index is not an identity.** Channel 42 of one trained network is not channel 42
  of another, and every layer of a model is as wide as the next. Whatever lines two
  things up by index checks who they are first: `check_basis_fits` for a basis against
  a layer (the network and layer its file says it was fitted on). Widths matching is
  never the check. Region-fitted bases record their source and layer automatically.
  Missing identity requires `allow_unverified_basis=True` (CLI:
  `--allow-unverified-basis`); known mismatches are always refused.
- **A feature's size is what it contributes.** Activation times the length of its
  direction: a dictionary may trade one for the other, so rank and compare by the
  product.
- **A method is a value.** Anything that turns channels into features is a
  `Decomposition`; routines take one as `basis=` rather than growing an argument per
  method. A basis is handed *raw* latents: its standardisation is its own and travels
  with it, so centring an analysis must not centre its input twice.
- **Refuse before reading.** Whatever can be wrong with a request is checked before the
  first 100 MB is loaded, and raised as `RequestError`.
- **Mind the memory.** One layer of a 1-degree, 384-channel model is 100 MB; nine layers
  at one time is 0.9 GB. Ask a source for the nodes and channels you need, keep a layer
  in its own precision, and never make a second copy of one. `xaig daig latent region`
  on the real atmosphere archive peaks near 275 MB resident (`/usr/bin/time -l`), a
  quarter of it the archive's own mapped pages; the first draft took 650.
- **Deterministic results.** No dependence on node order; PCA signs are fixed.
