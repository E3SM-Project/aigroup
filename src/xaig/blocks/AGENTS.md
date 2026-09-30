# blocks — neural blocks

Neural blocks trained on the latents of `diagnostics`: a sparse autoencoder, and the loop
that fits it. Needs the `blocks` extra (torch), which a plain `uv sync` leaves out:
`uv sync --extra blocks`.

| Module | Holds |
|---|---|
| `sae.py` | `SparseAutoencoder` (also a transcoder), `BSplineActivation` — plain `nn.Module`s |
| `train.py` | `fit_sae`: the loop over `diagnostics.latent.iter_batches`, returning a `Dictionary` |
| `cli.py` | `xaig blocks sae …`; writes a basis file |

## Rules

- **`blocks` reads `diagnostics`; nothing reads `blocks`.** It trains on
  `diagnostics.latent`'s batches and returns a `diagnostics.latent.Dictionary`, so what
  it learns is used by the analyses and the CLI without either importing torch. The edge
  to `diagnostics` is declared in `tests/test_purity.py`.
- **Blocks are framework-plain:** take and return tensors, no config objects, no
  dependency on a training harness, no knowledge of archives or grids. The loop that has
  those lives in `train.py`, apart, so a block can be lifted into anything.
- **Whatever torch evaluates, numpy must too.** A trained block is exported as plain
  arrays and applied by `diagnostics.latent.Dictionary`. A new activation therefore needs
  a numpy twin there, and a test that the two agree (`test_blocks.py` has the pattern).
- **Standardisation travels with the result.** Inputs are centred and scaled before
  training; the mean and scale go into the `Dictionary`, which is handed raw latents.
- **Train on `iter_batches`, not on `source.load()`:** valid nodes only, drawn by area, so
  the plain mean in the loss is the area-weighted one.
- **Say how good it is.** A fit reports explained variance, mean active features and the
  dead fraction in `meta["metrics"]`, with everything needed to refit it beside them.
- Import torch behind the extra, at the top of the module that needs it and never in
  `__init__.py` or `cli.py`: `xaig --help` imports every cli module on a base install.
- The loop is a toy on purpose. Quote what it measures; do not tune it in secret.
