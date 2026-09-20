# tests

## Rules

- **Synthetic fixtures only.** Nothing may touch a real campaign, a real scratch
  filesystem, or the network. The suite has to pass on a laptop with nothing mounted.
- `test_purity.py` enforces the architecture from two tables (`ALLOWED`: who may import
  whom; `THIRD_PARTY`: dependency ceilings), plus runtime checks that core imports no
  consumer and that `import xaig` / `xaig --help` stay light. A new subpackage must be
  added to the tables or the suite fails. These are cheap and unglamorous; without them
  the boundary erodes in a month.
- Two tiers. Tests needing numpy start with `pytest.importorskip("numpy")`, so the suite
  passes on a base install; CI runs both.
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
