# faig — reusable figures

Plotting functions with no web framework in them, so a notebook and a report draw the
same picture. May import `core` and `daig`; needs the `faig` extra (matplotlib,
and cartopy for coastlines).

## Rules

- **Return a figure; never show or save one.** The caller decides.
- **`matplotlib.figure.Figure`, never `pyplot`.** No global state, nothing to close,
  safe under a server that handles several sessions at once.
- **Colour follows the job.** Signed quantities (an activation about its mean, a
  similarity, a component score) get a diverging blue-red scale symmetric about zero, so
  zero reads as nothing and equal magnitudes look equally strong. Magnitudes get one
  hue, light to dark. Never a rainbow. Invalid nodes are a grey that belongs to no value.
  Lines are categories: Okabe and Ito's eight colours, then a change of dash rather than
  a ninth hue, and zero is drawn.
- **Degrade, do not fail.** Cartopy fetches its coastline data on first use, and a
  compute node has no network. Maps are then still drawn — on plain axes, with the
  grid's own mask outlined — and `why_no_coastlines()` says what happened. The fetch has
  a deadline, because cartopy's has none and a node that drops packets never refuses.
  `XAIG_NO_COASTLINES=1` skips the attempt.
- **Time is placed, not counted.** `series_figure` takes real `x` (hours) when the source
  can give it, so a gap between kept steps looks like one; without it, points are evenly
  spaced and labelled, which is the honest picture when the spacing is unknown.
- Archives keep longitude in model order; sort columns west to east before drawing.
- Render it and look at it. A test can check that a figure is produced, not that it is
  right.
