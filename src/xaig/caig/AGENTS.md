# caig — campaign tracking

A minimal, offline substitute for a tracking service. Answers "what ran, how far did it
get, and which arm won" from a directory or table of runs.

## Non-goals

- **No live tracking.** No server, no daemon, no streaming. You run a command.
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

- `api.py` — importable; returns objects, prints nothing
- `cli.py` — `xaig caig ...`; parses and formats, no domain logic
- `campaigns/*.yaml` — bundled campaign specs (data, not code)

Adding a campaign means adding a YAML file here, not writing Python.
