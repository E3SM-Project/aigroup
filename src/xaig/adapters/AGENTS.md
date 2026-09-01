# adapters — the disposable half

Every assumption about a specific framework, scheduler, file layout or log format lives
here. This code is expected to be thrown away when the group changes systems; core is not.

## Writing a new adapter

Implement one or more of the protocols in `core/protocols.py`:

| Protocol | Method | Returns |
|---|---|---|
| `Discoverer` | `discover()` | iterator of `Run` |
| `StatusProbe` | `probe(run)` | `RunStatus` |
| `MetricSource` | `metrics(run, names=None)` | `dict[str, MetricSeries]` |
| `ArtifactStore` | `artifacts(run)` | `dict[str, str]` of handles |

Then register it in `pyproject.toml`:

```toml
[project.entry-points."xaig.adapters"]
myframework = "mypkg.adapter:MyDiscoverer"
```

A separate distribution can do this too — that is the point. An adapter never needs to
live in this repo.

## Rules

- Heavy dependencies go behind an extra and are imported inside the adapter, never at
  package import time.
- Tolerate partial data. A run being written right now has truncated logs and missing
  epochs; that is a normal reading, not an error. Missing means `None`, not an exception.
- Return `RunStatus.UNKNOWN` rather than guess. Probes are tried in order and the first
  confident answer wins.
- Adapter constructors may take any arguments; `caig.api` passes only what a signature
  accepts, so the spec's `discovery` block never becomes a lowest-common-denominator.

## Present adapters

- `table.py` — runs from a delimited table (stdlib `csv`, base tier). The most
  framework-neutral source there is.
