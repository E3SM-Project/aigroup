# aigroup

> [!IMPORTANT]
> This repository is archived and read-only. It was `E3SM-Project/aigroup` until
> October 2026.
>
> - The guides now live on Confluence:
>   [archive of E3SM-Project/aigroup](https://e3sm.atlassian.net/wiki/spaces/p3ai/pages/6753845256).
> - The guide site at `docs.e3sm.org/aigroup` is no longer published.
> - The Python package is [exege](https://github.com/mahf708/exege).

Docs, scripts, examples, and prototypes for E3SM AI efforts.

- **`docs/`** — the guide site's sources
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
