# tests

## Rules

- **Synthetic fixtures only.** Nothing may touch a real campaign, a real scratch
  filesystem, or the network. The suite has to pass on a laptop with nothing mounted.
- `test_purity.py` enforces the architecture from two tables (`ALLOWED`: who may import
  whom; `THIRD_PARTY`: dependency ceilings), plus runtime checks that core imports no
  consumer and that `import xaig` / `xaig --help` stay light. A new subpackage must be
  added to the tables or the suite fails. These are cheap and unglamorous; without them
  the boundary erodes in a month.
- The app is tested headlessly with `streamlit.testing.v1.AppTest`: what a view shows,
  that a widget changes it, and that the command and code it offers reproduce it. That
  cannot see layout; look at the app before calling a view done.
- Three tiers. Tests needing numpy start with `pytest.importorskip("numpy")` and those
  needing torch with `pytest.importorskip("torch")`, so the suite passes on a base
  install and on a full one without torch; CI runs all three.
- What torch computes and numpy applies is tested for agreement: a block against the
  `Dictionary` exported from it, the torch spline against the numpy one.
- Give a fixture a known right answer. The synthetic latent archive plants a bump in one
  channel and a constant offset in another, so ranking and centring can be asserted, not
  just exercised. Its noise is seeded, so two archives are twins node for node, and
  `shift=` makes a perturbed one whose difference from its control is known exactly.
- Where the real thing has a shape, give the fixture that shape. `daig.latent.toy` writes an
  archive the way the real exporter does — kept steps with a gap, fields that begin a step
  before the latents, a mask only a field knows, no leap days — and `test_latent_toy.py`
  reads it back.
- Test the behaviour that would actually bite: a misspelt adapter option that is refused
  rather than ignored, an install hint that keeps the commit it was installed from.
