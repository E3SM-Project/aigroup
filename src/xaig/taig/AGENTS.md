# taig — toys

Reusable neural blocks, and small architectures built from them for testing concepts.
Needs the `taig` extra (torch), which a plain `uv sync` leaves out: `uv sync --extra taig`.

| Module | Holds |
|---|---|
| `sae.py` | `SparseAutoencoder` (also a transcoder), `BSplineActivation` — plain `nn.Module`s |
| `train.py` | `fit_sae`: the loop over `daig.latent.iter_batches`, returning a `Dictionary` |
| `cli.py` | `xaig taig sae …`; writes a basis file |

## Rules

- **`taig` reads `daig`; nothing reads `taig`.** It trains on `daig.latent`'s batches and
  returns a `daig.latent.Dictionary`, so what it learns is used by the analyses, the CLI
  and the app without any of them importing torch. It may not import `caig`, `faig` or
  `waig`, and the edge to `daig` is declared in `tests/test_purity.py`.
- **Blocks are framework-plain:** take and return tensors, no config objects, no
  dependency on a training harness, no knowledge of archives or grids. The loop that has
  those lives in `train.py`, apart, so a block can be lifted into anything.
- **Whatever torch evaluates, numpy must too.** A trained block is exported as plain
  arrays and applied by `daig.latent.Dictionary`. A new activation therefore needs a
  numpy twin there, and a test that the two agree (`test_taig.py` has the pattern).
- **Standardisation travels with the result.** Inputs are centred and scaled before
  training; the mean and scale go into the `Dictionary`, which is handed raw latents.
- **Train on `iter_batches`, not on `source.load()`:** valid nodes only, drawn by area, so
  the plain mean in the loss is the area-weighted one.
- **Say how good it is.** A fit reports explained variance, mean active features and the
  dead fraction in `meta["metrics"]`, with everything needed to refit it beside them.
- Import torch behind the extra, at the top of the module that needs it and never in
  `__init__.py` or `cli.py`: `xaig --help` imports every cli module on a base install.
- The loop is a toy on purpose. Quote what it measures; do not tune it in secret.
