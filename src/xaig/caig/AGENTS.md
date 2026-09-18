# caig — campaign tracking

A minimal, offline substitute for a tracking service. Answers "what ran, how far did it
get, and which arm won" from a directory or table of runs.

## Non-goals

- **No live tracking.** No tracking server, no daemon, no streaming. You run a command.
  (A local viewer that reads the same on-disk artifacts is a client, not a tracker.)
- **No network and no credentials** in the base path. Metrics are read from what the
  runs already wrote on disk.
- No dependency on any particular training framework — that is what adapters are for.

## Must work on a running campaign *and* a finished one

This constraint shapes everything here:

1. Never assume an artifact exists. Missing reads as `None`.
2. Derive status; do not read it from one field. Combine evidence and prefer `UNKNOWN`
   over a guess.
3. Tolerate partial and truncated files.
4. Re-scanning must be cheap and incremental, so polling a live campaign is practical.
5. The index is a plain on-disk artifact, so a finished campaign can be archived with it
   and re-analyzed later without the original run tree.

## Layout

- `spec.py` — campaign specs: id grammar, factors, parents, metric; loading and the
  bundled ones. Unknown keys are errors, and nothing defaults to one campaign's words
  (`parent_key` has no default; CLI columns come from the id's own groups).
- `api.py` — importable; returns objects, prints nothing
- `cli.py` — `xaig caig ...`; a thin client of `api`
- `campaigns/*.yaml` — bundled campaign specs (data, not code)

Adding a campaign means adding a YAML file here, not writing Python.

## `check` asks two questions

- **run ids** — does each id fit the grammar, round-trip byte for byte, and occur once?
- **metadata** — did the source say something that contradicts the id (`seed` 99 on a
  run named `…S01`), or that could not be read? The id wins, and the disagreement is
  reported rather than dropped. A column repeating the id's notation (`S01` for 1) agrees.

The pre-registered seed-spread decision rule (`caig compare`) is deliberately not
implemented yet: it must be transcribed from the campaign's own protocol, not invented.
