# adapters — the disposable half

Every assumption about a specific framework, scheduler, file layout or log format lives
here. This code is expected to be thrown away when the group changes systems; what uses
it is not.

## The factory contract

```python
factory(source, **options) -> adapter
```

- **`source`** is whatever the adapter reads — a path, a directory, a URL. It is passed
  positionally, so **the first positional parameter is the source**, whatever it is
  called. An adapter that reads nothing declares its options keyword-only.
- **`options`** are keyword arguments: a spec's `discovery` block, or `**kwargs` from the
  API. An option the factory does not declare is an error listing the ones it does.
- **context** is offered by the caller, not the user — `caig` offers `spec`. It reaches
  only a factory that declares a parameter of that name.

The adapter returned is **one object implementing one or more protocols**. Callers ask
with `isinstance`, so a framework's discovery, status and metrics travel under one name:

| Protocol | Method | Consumer |
|---|---|---|
| `core.Discoverer` | `discover()` → iterator of `Run` | `caig` |
| `core.StatusProbe` | `probe(run)` → `RunStatus` | `caig` |
| `core.MetricSource` | `metrics(run, names=None)` → `dict[str, MetricSeries]` | `caig` |
| `core.ArtifactStore` | `artifacts(run)` → `dict[str, str]` of handles | — |
| `daig.latent.LatentSource` | `info()`, `grid()`, `load(time, layer, …)` | `daig`, `taig` |
| `daig.latent.ReferenceFields` | `field_names()`, `field(name, time)` → per-node values | `daig` |

## Registering

Entry points are the only mechanism, for the adapters shipped here and for one living in
a completely separate distribution alike:

```toml
[project.entry-points."xaig.adapters"]
myframework = "mypkg.adapter:MyAdapter"
```

Then **rerun `uv sync`**: entry points are read from installed
metadata, and a stale install is the usual reason a new adapter "is not found". xaig's
own names win a clash, so a plugin can add adapters but never silently replace one.

## Rules

- An adapter may import `xaig.core` and the domain contract it implements
  (`xaig.daig.latent`). Nothing imports an adapter; it is reached through the registry.
- Heavy dependencies go behind an extra and are imported by the adapter, which loads
  lazily — never at `import xaig` time.
- Tolerate partial data. A run being written right now has truncated logs and missing
  epochs; that is a normal reading, not an error. Missing means `None`, not an exception.
- What you tolerate, record: attach an `Issue` to the run instead of swallowing it.
- Return `RunStatus.UNKNOWN` rather than guess.
- A source that is broken is an `AdapterError`; a request it cannot meet (a node index out
  of range, a field it does not hold) is a `RequestError`. Never a bare `IndexError`.
- Carry what the exporter said. A latent adapter puts the manifest's free-form
  `experiment` block, and the options it was opened with, into `LatentInfo`, so they
  reach the provenance of every result.
- Read selectively. A `LatentSource` asked for a region must not load the layer.

## Present adapters

- `table.py` — runs from a delimited table (stdlib `csv`, base tier). The most
  framework-neutral source there is.
- `latent_archive.py` — activations recorded from a model, as a directory of
  memory-mapped arrays, with the physical fields kept beside them (`xaig[daig]`; format
  in `docs/package/latents.md`).
