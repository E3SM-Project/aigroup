# nn — torch modules

Torch modules trained on a model's latents: a sparse autoencoder, and the loop that fits
it. Needs the `nn` extra (torch), which a plain `uv sync` leaves out:
`uv sync --extra nn`.

| Module | Holds |
|---|---|
| `sae.py` | `SparseAutoencoder` (also a transcoder), `BSplineActivation` — plain `nn.Module`s |
| `train.py` | `fit_sae`: the loop over `latents.iter_batches`, returning a `Dictionary` |
| `cli.py` | `xaig nn sae …`; writes a basis file |

## Rules

- **`nn` reads `latents`; nothing reads `nn`.** It trains on the batches of `latents`
  and returns a `latents.Dictionary`, so what it learns is used by the analyses and the
  CLI without either importing torch. The edge to `latents` is declared in
  `tests/test_purity.py`.
- **Modules are framework-plain:** take and return tensors, no config objects, no
  dependency on a training harness, no knowledge of archives or grids. The loop that has
  those lives in `train.py`, apart, so a module can be lifted into anything.
- **Whatever torch evaluates, numpy must too.** A trained module is exported as plain
  arrays and applied by `latents.Dictionary`. A new activation therefore needs a numpy
  twin there, and a test that the two agree (`test_nn.py` has the pattern).
- **Standardization travels with the result.** Inputs are centered and scaled before
  training; the mean and scale go into the `Dictionary`, which is handed raw latents.
- **Train on `iter_batches`, not on `source.load()`:** valid nodes only, drawn by area, so
  the plain mean in the loss is the area-weighted one.
- **Say how good it is, of what it returns.** A fit reports explained variance, mean active
  features and the dead fraction in `meta["metrics"]`, with everything needed to refit it
  beside them, measured on the *exported* dictionary (`training_metrics` is a monitor). A
  test holds `metrics` to an independent numpy evaluation of the `Dictionary`.
- Import torch behind the extra, at the top of the module that needs it and never in
  `__init__.py` or `cli.py`: `xaig --help` imports every cli module on a base install.
- The loop is a toy on purpose. Quote what it measures; do not tune it in secret.
