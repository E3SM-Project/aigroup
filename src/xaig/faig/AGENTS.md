# faig — reusable figures

Plotting functions with no web framework in them, so a notebook, a report and `waig`
draw the same picture. May import `core` and `daig`; needs the `faig` extra (matplotlib),
and `maps` (cartopy) for coastlines.

## Rules

- **Return a figure; never show or save one.** The caller decides.
- **`matplotlib.figure.Figure`, never `pyplot`.** No global state, nothing to close,
  safe under a server that handles several sessions at once.
- **Colour follows the job.** Signed quantities (an activation about its mean, a
  similarity, a component score) get a diverging blue-red scale symmetric about zero, so
  zero reads as nothing and equal magnitudes look equally strong. Magnitudes get one
  hue, light to dark. Never a rainbow. Invalid nodes are a grey that belongs to no value.
- **Degrade, do not fail.** Without cartopy, or without network for its coastline data
  (a compute node), maps are still drawn — on plain axes, with the grid's own mask
  outlined. `XAIG_NO_COASTLINES=1` skips the attempt.
- Archives keep longitude in model order; sort columns west to east before drawing.
- Render it and look at it. A test can check that a figure is produced, not that it is
  right.
