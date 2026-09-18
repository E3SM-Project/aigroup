# aigroup

Two peers live here. Neither exists to serve the other.

- `docs/` — the MkDocs guide site, published to gh-pages
- `src/xaig/` — the `xaig` Python package

## Rules

- Work on `user/topic` branches; merge to `main` via PR. Short lowercase imperative
  commit subjects.
- `mkdocs build --strict` must pass. New pages must be added to `nav` in `mkdocs.yml`
  by hand.
- `ruff check` and `pytest` must pass. Both run in `.github/workflows/ci.yml`, on a base
  install (PyYAML + Click only) and on a full one; numpy-backed tests skip on the former.
- `uv` is the tool of record. ACE itself pins Python 3.11.
- Ship in ~1000-line increments. Each increment leaves the repo working and useful.

## Where to look

| Path | AGENTS.md covers |
|---|---|
| `src/xaig/` | package architecture, who may import whom, how to extend it |
| `src/xaig/core/` | the purity contract |
| `src/xaig/adapters/` | the factory contract; writing a new adapter |
| `src/xaig/{caig,daig,taig,viz,waig}/` | each subpackage's scope, rules and non-goals |
| `tests/` | fixture rules |
| `docs/` | prose and nav conventions |
