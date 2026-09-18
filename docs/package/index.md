# The `xaig` package

`xaig` is a light, framework-agnostic package for working with AI campaigns. It lives in
this repo alongside the guides, as a peer rather than an appendix.

| Subpackage | Scope |
| --- | --- |
| `caig` | campaign tracking — offline, no server, no live streaming |
| `daig` | diagnostics of emulators: their outputs and [their internals](latents.md) |
| `taig` | reusable neural blocks *(skeleton)* |
| `viz` | figures, with no web framework in them |
| `waig` | [a local web app](waig.md) over `caig` and `daig` |

!!! warning "research tool"

    `xaig` is early. The `caig` surface described here and `daig`'s
    [latent diagnostics](latents.md) work; emulator-vs-reference diagnostics and `taig`
    are still to come.

## Install

```console
$ uv venv --python 3.11 .venv
$ uv pip install -e '.[dev]'
```

The base install pulls only PyYAML and Click, and is all `caig` needs. Anything heavier
sits behind an extra named after the subpackage that needs it, and a missing one says so:

```console
$ xaig daig latent info latents/atmosphere
error: numpy is not installed; it comes with the 'daig' extra: uv pip install 'xaig[daig]'
```

| Extra | Pulls | Gets you |
| --- | --- | --- |
| `daig` | numpy | `xaig.daig` |
| `netcdf` | netCDF4 | masks read from a reference file |
| `viz` | matplotlib | `xaig.viz`: maps and figures |
| `maps` | cartopy | coastlines on those maps |
| `waig` | streamlit | [`xaig waig`](waig.md) |
| `taig` | torch | `xaig.taig` |

!!! tip "uv cache"

    On NERSC, keep the cache off `$HOME`:

    ```console
    $ export UV_CACHE_DIR="$PSCRATCH/.cache/uv"
    ```

## Why it is built this way

Three concerns are kept apart, because each has a different answer:

- **Framework coupling lives in adapters.** The group expects to move to systems
  profoundly unlike ACE/FME/Samudra, so everything that knows a real file layout, log
  format or scheduler lives in `xaig.adapters`, behind a small protocol. Supporting a new
  system means writing a new adapter, never editing the code that uses it. Adapters are
  found through the `xaig.adapters` entry-point group and nothing else, so one can ship
  from a completely separate package.
- **Science lives in the subpackage that uses it**, with the dependencies it honestly
  needs: PCA needs numpy and is not an adapter. What it may not know is a file format or
  a user interface.
- **Weight lives behind extras.** `xaig.core` depends on the standard library alone, and
  `import xaig` never pulls in the scientific stack.

Every API returns objects and prints nothing; the CLI is one client of it, a notebook
another. These rules are enforced by `tests/test_purity.py`, not by convention.

A *campaign spec* is a YAML file describing one campaign's conventions — its run-id
grammar, its factors, its parent map, its metric. Adding a campaign does not mean writing
Python.

## Tracking a campaign

Specs bundled with the package:

```console
$ xaig caig specs
aug26  35 runs (25 atmosphere, 10 ocean) ablating aerosol, CO2, batch size, ...
```

List the runs of the `e3sm_hist_v20260812` campaign, reading its manifest:

```console
$ xaig caig ls --spec aug26 \
    --source /pscratch/sd/m/mahf708/ace/configs/experiments/e3sm_hist_v20260812/runs/MANIFEST.tsv
ID                                       STATUS   EXP  REALM  SEED  AEROSOL  BATCH  CO2  LR  OCEAN_STEP  WEIGHTS  AMP
E01.aug26.atm.A0_B16_C0_L0_O5_W0_X0.S01  unknown  E01  atm    1     0        16     0    0   5           0        0
E02.aug26.atm.A0_B16_C1_L0_O5_W0_X0.S01  unknown  E02  atm    1     0        16     1    0   5           0        0
...

35 run(s)
```

The factor word in each run id is decomposed into named attributes, so you can filter on
them:

```console
$ xaig caig ls --spec aug26 --source .../MANIFEST.tsv --select realm=ocn -c exp,seed,ocean_step
```

`STATUS` reads `unknown` above because status probing arrives with the FME adapter — see
[remaining tasks](#remaining-tasks). Until then a manifest can carry it: name the column,
and map its vocabulary onto xaig's, in the spec's `discovery` block.

```yaml
discovery:
  adapter: table
  id_column: runid
  status_column: state
  status_map: { done: finished, crashed: failed }
```

`id` and `status` can be selected and sorted on like any attribute, numbers sort
numerically, and `--json` hands the rows to another tool:

```console
$ xaig caig ls --spec aug26 --source .../MANIFEST.tsv --select status=running --sort batch,seed
$ xaig caig ls --spec aug26 --source .../MANIFEST.tsv --json | jq '.[].id'
```

A misspelt option is an error, not a silent default:

```console
$ xaig caig ls --spec ./typo.yaml --source .../MANIFEST.tsv
error: adapter 'table' does not accept option(s) status_colum; accepted: delimiter, id_column, status_column, status_map, strict
```

Show one run:

```console
$ xaig caig show E05.aug26.atm.A3_B16_C1_L0_O5_W0_X0.S01 --spec aug26 --source .../MANIFEST.tsv
```

## Checking a campaign

A run id that disagrees with the config it names is the worst failure a campaign can
have, because every table and plot is labelled by the id. `check` asks two questions and
reports them apart, because they have different fixes:

- **run ids** — does every id fit the spec's grammar, round-trip through it byte for
  byte, and occur once?
- **metadata** — does the source say something that contradicts the id? The id wins (it
  is the primary key), but the disagreement is reported rather than dropped. A column
  that repeats the id's own notation (`S01` for seed 1) agrees with it.

```console
$ xaig caig check --spec aug26 --source .../MANIFEST.tsv
OK  35 run id(s) round-trip through spec 'aug26'
```

```console
$ xaig caig check --spec aug26 --source broken.tsv
run ids:
  FAIL  not-an-id: run id 'not-an-id' does not match spec 'aug26'
metadata:
  FAIL  E05.aug26.atm.A3_B08_C1_L0_O5_W0_X0.S02: column seed=99 disagrees with the id, which says 2; the id wins
Error: 1 run id problem(s) and 1 metadata problem(s) across 4 run(s)
```

`ls` stays usable on such a campaign, and says how many of its runs have issues.

## Python API

The CLI is a thin client of this API; anything it can do, a notebook can.

```python
from xaig.caig import check_campaign, load_campaign, load_spec

spec = load_spec("aug26")
campaign = load_campaign(spec, source=".../runs/MANIFEST.tsv")

ocean = campaign.filter(realm="ocn")
for run in ocean.sorted_by("exp", "seed"):
    print(run.id, run.attrs["ocean_step"])

for finding in check_campaign(campaign, spec):
    print(finding.kind, finding.run_id, finding.message)
```

## Adding a campaign

Write a spec. This one describes a grammar with nothing in common with `aug26`:

```yaml
name: toy
id_pattern: '^run(?P<num>\d+)-(?P<mode>fast|slow)-(?P<knobs>[a-z0-9-]+)-s(?P<seed>\d+)$'
id_template: 'run{num:03d}-{mode}-{knobs}-s{seed:d}'
factor_field: knobs
factor_separator: "-"
factors:
  - { key: d, name: depth, width: 1 }
  - { key: w, name: width, width: 2 }
discovery:
  adapter: table
  id_column: name
```

```console
$ xaig caig ls --spec ./toy.yaml --source runs.csv
```

A campaign whose run ids carry no structure simply omits `id_pattern`; its adapter
attaches attributes instead. A key the spec does not know is an error, so `id_patern:`
cannot quietly mean "this campaign has no grammar".

## Adding an adapter

An adapter is one class implementing one or more protocols — `discover()` to find runs,
`probe(run)` for status, `metrics(run, names)` for scalar series — constructed as
`Adapter(source, **options)`. Register it, from this repo or any other package:

```toml
[project.entry-points."xaig.adapters"]
myframework = "mypkg.adapter:MyAdapter"
```

```console
$ uv pip install -e .   # entry points are read from installed metadata
$ xaig caig ls --spec ./mine.yaml --adapter myframework --source /path/to/runs
```

## Remaining tasks

- [ ] FME adapter: run-directory discovery, `out.log` parsing, joblog segment chains,
      optional Slurm probe
- [ ] Incremental scan index, so polling a live campaign is cheap
- [ ] `caig compare`: the pre-registered seed-spread decision rule
- [ ] `caig doctor`: machine-checked campaign guardrails
- [ ] `daig`: bias and time-mean maps, spectra, zonal means (a `FieldSource` beside
      `LatentSource`, on the same `daig.grid`)
- [ ] `daig.latent`: a GraphCast mesh adapter; the activation exporter as an adapter
- [ ] `taig`: the first reusable blocks
