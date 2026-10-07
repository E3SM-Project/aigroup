# docs

MkDocs Material, published to gh-pages. `uv run --group docs mkdocs build --strict` must
pass — broken links fail CI. The `social` plugin draws cards with cairo, a system library
the `docs` group cannot install for you; without it, build with `MKDOCS_SOCIAL=false`.

## Conventions

- Shell blocks are fenced as `console`, with `$` prompts.
- Annotated YAML uses ` ```{ .yaml .annotate } ` with `(1)!` markers and a matching
  ordered list beneath.
- Long configs and scripts are collapsed in `??? example "title"`.
- Admonitions use lowercase titles: `!!! tip "uv cache"`.
- Guides end with a `## Remaining tasks` unchecked list.
- Cross-links are relative (`python-envs.md`).
- Quote real measured numbers and real NERSC paths rather than genericizing them.

## Adding a page

Add the file, then add it to `nav` in `mkdocs.yml` by hand — nav is explicit, not
inferred.
