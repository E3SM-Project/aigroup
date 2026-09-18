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
- Two tiers. Tests needing numpy start with `pytest.importorskip("numpy")`, so the suite
  passes on a base install; CI runs both.

- Test the behaviour that would actually bite: an id that round-trips, a status that is
  interrupted rather than failed, a missing epoch that reads as missing.
