# tests

## Rules

- **Synthetic fixtures only.** Nothing may touch a real campaign, a real scratch
  filesystem, or the network. The suite has to pass on a laptop with nothing mounted.
- `test_purity.py` enforces the architecture from two tables (`ALLOWED`: who may import
  whom; `THIRD_PARTY`: dependency ceilings), plus runtime checks that core imports no
  consumer and that `import xaig` / `xaig --help` stay light. A new subpackage must be
  added to the tables or the suite fails. These are cheap and unglamorous; without them
  the boundary erodes in a month.
- Test the behaviour that would actually bite: a misspelt adapter option that is refused
  rather than ignored, an install hint that keeps the commit it was installed from.
