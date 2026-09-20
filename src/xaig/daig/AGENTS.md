# daig — diagnostics

What an emulator holds inside.

| Module | Holds |
|---|---|
| `grid.py` | nodes on a sphere: masks, area weights, regions, maps |
| `latent/source.py` | the contract: `LatentSource`, the optional `ReferenceFields`, `LatentInfo` |
| `latent/toy.py` | a toy emulator in numpy, so an archive can be made with no model and no data |

## Rules that must not be lost

- **Area-weight everything.** Unweighted means on a lat-lon grid are simply wrong.
- **Be NaN- and mask-aware.** Ocean channels are undefined over land (30.7% of points in
  the E3SMv3 configuration); a plain `mean` silently returns NaN or a biased number.
  Invalid nodes are left out of regions and read NaN in every map.
- **No user interface and no foreign file formats here.** Functions take arrays or a
  source and return structured results carrying their settings and provenance. Reading
  what a framework wrote is an adapter's job (`LatentSource`); drawing is a client's.
  Third-party imports are capped at numpy and click by `tests/test_purity.py`.
- **Refuse before reading.** Whatever can be wrong with a request is checked before the
  first 100 MB is loaded, and raised as `RequestError`.
- **Mind the memory.** One layer of a 1-degree, 384-channel model is 100 MB; nine layers
  at one time is 0.9 GB. Ask a source for the nodes and channels you need, keep a layer
  in its own precision, and never make a second copy of one.
- **Deterministic results.** No dependence on node order.
