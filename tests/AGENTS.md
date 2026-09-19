# tests

## Rules

- **Synthetic fixtures only.** Nothing may touch a real campaign, a real scratch
  filesystem, or the network. The suite has to pass on a laptop with nothing mounted.
- The `toy` fixture in `conftest.py` is deliberately unlike `aug26` — different
  separator, different field order, a non-default parent key. If core is quietly shaped
  by one campaign, that fixture is what fails. Keep it different on purpose.
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
- Test the behaviour that would actually bite: an id that round-trips, a status that is
  interrupted rather than failed, a missing epoch that reads as missing.
