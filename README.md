# aigroup

Docs, scripts, examples, and prototypes for E3SM AI efforts.

- **`docs/`** — the guide site, published at <https://e3sm-project.github.io/aigroup>
- **`archive/`** — the configs of past runs

The Python package that used to live here as `xaig` is now
[exege](https://github.com/mahf708/exege) — tools for understanding and evaluating
scientific machine-learning models — with its history, open issues and documentation.

## Develop

```console
$ uv run --group docs mkdocs build --strict   # MKDOCS_SOCIAL=false without cairo
$ uv run --group docs mkdocs serve
```

See `AGENTS.md` and `docs/AGENTS.md` for the conventions.
