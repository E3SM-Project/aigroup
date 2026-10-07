# aigroup

The E3SM AI guide site: `docs/`, built with MkDocs and published to gh-pages. The
`xaig` package that lived beside it is now [exege](https://github.com/mahf708/exege);
work on it there.

## Rules

- Work on `user/topic` branches; merge to `main` via PR. Short lowercase imperative
  commit subjects.
- `uv run --group docs mkdocs build --strict` must pass. New pages must be added to `nav`
  in `mkdocs.yml` by hand.
- `uv` is the tool of record: `uv run --group docs …`. ACE itself pins Python 3.11.
- `scratch/` is ignored by git: write throwaway output there, never beside the docs.

## Where to look

| Path | AGENTS.md covers |
|---|---|
| `docs/` | prose and nav conventions |
