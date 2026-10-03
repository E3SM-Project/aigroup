# Experiment records

A result of `latents evaluate` or `latents steer` is complete, and large: the steering
result of the [steering guide](steering.md) is 110 KB of runs and pairings, most
of it time series nobody reads twice. What you come back to a week later is smaller: who
produced it, from what, how, what it found, and the command that makes it again. That is
an **experiment record**: one versioned JSON file, 4.8 KB for the same experiment, that
the commands write, a notebook can load, and [the app](app.md) opens.

```console
$ xaig latents steer --adapter toy-dynamics --adapter-option masked=3 \
    --basis scratch/rec/planted.npz --layer 1 --feature 0 --amount 1.5 --time 1 \
    --steps 5 --seeds 0,1,2 --random-draws 20 --record scratch/rec/steer.json
$ xaig latents evaluate scratch/rec/toy --blocks 4 --basis scratch/rec/a.npz --pca 1,4 \
    --record scratch/rec/eval.json
$ xaig app --record scratch/rec/steer.json --record scratch/rec/eval.json
```

`--record` is beside `--out`, which still writes the whole result. A record cannot be asked
of `--split-only`, which computes nothing to keep.

## What is in one

```json
{
  "format": "xaig.experiment-record",
  "schema_version": 1,
  "kind": "steering",
  "provenance": {"source": "toy-dynamics", "model": "toy-dynamics", "component": "linear",
                 "checkpoint": "planted-0", "options": {"masked": 3, "…": "…"}, "xaig": "0.4.0"},
  "bases": [{"path": "scratch/rec/planted.npz", "sha256": "72ac4908…0960", "status": "verified"}],
  "settings": {"adapter": "toy-dynamics", "adapter_options": {"masked": 3}, "…": "…"},
  "split": null,
  "results": {"spec": {"…": "…"}, "effects": {"temperature": {"feature_response": 1.40625,
              "reconstruction_response": 0.3335, "random_responses": ["20 numbers"],
              "rank": 1, "n_draws": 20, "p_value": 0.0476, "…": "…"}}},
  "command": "xaig latents steer --adapter toy-dynamics … --basis-sha256 72ac4908…0960 …"
}
```

| Key | Holds |
| --- | --- |
| `provenance` | the source's model, component and checkpoint; for a hub archive the **commit** it was opened at and the revision that was asked for; the options it was opened with; what the exporter recorded about the run (`experiment`); the version of xaig. The same as `result_provenance` writes into every result ([provenance](latents.md#provenance-what-a-result-was-made-from)). |
| `bases` | every basis used, by file and **content hash** |
| `settings` | what the command was asked: enough to ask it again |
| `split` | for an evaluation, the time labels of both sides and the buffer, held once. `null` for a steering record |
| `results` | an evaluation: per basis, the held-out and training numbers, dead and redundant features; the [fidelity-sparsity curve](evaluation.md) and the [seed stability](evaluation.md) if asked for. A steering run: its spec, and for each field the feature's response, the reconstruction arm's, and **every random draw**, which is what the feature is set against |
| `command` | the shell command that reproduces it |

What a record leaves out is what a result keeps and a reader rarely wants: every run and
every pairing of a steering experiment, and the evaluation's split a second time under each
basis. `SteeringResult.to_dict` and `--out` still have them.

An evaluation's command pins what it can: the archive's commit (`--revision`) and each
basis by content, as repeated `--basis PATH --basis-sha256 HASH` pairs in the order the
bases were given, the same hashes that are in `bases`. A basis file that has changed since
is refused by the command the record holds. A steering record's command pins its one basis.

## Versions

`schema_version` is 1. `load_record` reads the versions it knows and refuses the rest with
a one-line `RequestError`: a record from a later xaig says to upgrade, and a file that is
not a record, is not JSON, lacks a part a view relies on, has one of the wrong JSON type
(a number the view formats must be finite, or `null`), or is of a kind no view knows is
refused too. Within a version a writer may add keys and a reader ignores what it does not
know; removing or renaming one, or changing what one means, is a new version.

```python
from xaig.latents import load_record

record = load_record("scratch/rec/steer.json")
record.kind  # "steering"
record.commit  # the archive's commit, or None for a local source
record.results["effects"]["temperature"]["rank"]  # 1
record.command  # the command that redoes it
```

From Python, `evaluation_record(info, split, evaluations=…, curve=…, stability=…,
bases=…, settings=…, command=…)` and `steering_record(result, settings=…, command=…)` build
one and `save_record` writes it. The commands are one client of these, as the app is
another.

## In the app

Under **Records**, the app opens a record from `--record`, or from a path typed in the
sidebar. It shows *where it came from* (source, network, commit, options, bases by hash),
then, for an evaluation, the split, the held-out table, the curve and the stability, and
for a steering run each field's response beside the random directions' — as a table, and
for the chosen field as a histogram of the draws' magnitudes with the feature and the
reconstruction arm marked on it. The last section is the `command` and a download of the
record. The test suite runs that command and checks it agrees with what is on screen. A file the
validator lets through that the view still cannot show is a warning on the page, not a
traceback.

## Remaining tasks

- [ ] Check a record against the files it names (`bases`, the archive's commit) and say
      which have changed
- [ ] Records of the other commands (`diff`, `growth`), if anyone comes back to them
- [ ] Two records side by side in the app
