# The `xaig` package

`xaig` is a light, framework-agnostic package for working with AI campaigns. It lives in
this repo alongside the guides, as a peer rather than an appendix.

| Subpackage | Scope |
| --- | --- |
| `caig` | campaign tracking — offline, no server, no live streaming |
| `daig` | emulator-vs-reference diagnostics *(skeleton)* |
| `taig` | reusable neural blocks *(skeleton)* |

!!! warning "research tool"

    `xaig` is early. The `caig` surface described here works; `daig` and `taig` are
    skeletons with their intended APIs sketched in their module docstrings.

## Install

```console
$ uv venv --python 3.11 .venv
$ uv pip install -e '.[dev]'
```

The base install pulls only PyYAML and Click. Anything that knows about a specific
training framework lives behind an extra.

| Extra | Pulls | Gets you |
| --- | --- | --- |
| `fme` | numpy, xarray, netCDF4 | the ACE/FME adapter |
| `viz` | matplotlib | plots and reports |
| `toys` | torch | `taig` |

!!! tip "uv cache"

    On NERSC, keep the cache off `$HOME`:

    ```console
    $ export UV_CACHE_DIR="$PSCRATCH/.cache/uv"
    ```

## Why it is built this way

The group expects to move to systems profoundly unlike ACE/FME/Samudra. So `xaig.core`
knows nothing about any framework: it holds a data model, four protocols, and generic
algorithms, and depends only on the standard library and PyYAML. Everything that knows
about a real file layout, log format or scheduler lives in `xaig.adapters`.

Supporting a new system means writing a new adapter, never editing core. Adapters are
found through the `xaig.adapters` entry-point group, so one can ship from a completely
separate package.

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
[remaining tasks](#remaining-tasks).

Show one run:

```console
$ xaig caig show E05.aug26.atm.A3_B16_C1_L0_O5_W0_X0.S01 --spec aug26 --source .../MANIFEST.tsv
```

## Checking a campaign

A run id that disagrees with the config it names is the worst failure a campaign can
have, because every table and plot is labelled by the id. `check` round-trips every id
through the spec:

```console
$ xaig caig check --spec aug26 --source .../MANIFEST.tsv
OK  35 run id(s) round-trip through spec 'aug26'
```

## Python API

The API is a peer of the CLI, not a layer beneath it. Both sit on `xaig.core`.

```python
from xaig.caig import load_campaign
from xaig.core import spec as spec_module

spec = spec_module.load("aug26")
campaign = load_campaign(spec, source=".../runs/MANIFEST.tsv")

ocean = campaign.filter(realm="ocn")
for run in ocean.sorted_by("exp", "seed"):
    print(run.id, run.attrs["ocean_step"])
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
attaches attributes instead.

## Remaining tasks

- [ ] FME adapter: run-directory discovery, `out.log` parsing, joblog segment chains,
      optional Slurm probe
- [ ] Incremental scan index, so polling a live campaign is cheap
- [ ] `caig compare`: the pre-registered seed-spread decision rule
- [ ] `caig doctor`: machine-checked campaign guardrails
- [ ] `daig`: bias and time-mean maps, spectra, zonal means
- [ ] `taig`: the first reusable blocks
