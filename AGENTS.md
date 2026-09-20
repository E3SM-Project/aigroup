# aigroup

Two peers live here. Neither exists to serve the other.

- `docs/` — the MkDocs guide site, published to gh-pages
- `src/xaig/` — the `xaig` Python package

## Rules

- Work on `user/topic` branches; merge to `main` via PR. Short lowercase imperative
  commit subjects.
- `uv run --group docs mkdocs build --strict` must pass. New pages must be added to `nav`
  in `mkdocs.yml` by hand.
- `ruff check`, `ruff format --check` and `pytest` must pass. All run in
  `.github/workflows/ci.yml`.
- `uv` is the tool of record: `uv sync` once, then `uv run …`. A checkout gets pytest and
  ruff by default (the `dev` dependency group). ACE itself pins Python 3.11.
- Ship in ~1000-line increments. Each increment leaves the repo working and useful.

## Where to look

| Path | AGENTS.md covers |
|---|---|
| `src/xaig/` | package architecture, who may import whom, how to extend it |
| `src/xaig/core/` | the purity contract |
| `tests/` | fixture rules |
| `docs/` | prose and nav conventions |
