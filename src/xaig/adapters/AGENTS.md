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
- **`options`** are keyword arguments: what a command was given (`--mask-variable`), or
  `**kwargs` from the API. An option the factory does not declare is an error listing the
  ones it does.
- **context** is offered by the caller, not the user (a campaign spec, say). It reaches
  only a factory that declares a parameter of that name. `**kwargs` is not a declaration
  (what a factory hands on to another library must not hold xaig's objects), and the
  source parameter never receives it.

The adapter returned is **one object implementing one or more protocols**. Callers ask
with `isinstance`, so everything one framework can supply travels under one name:

| Protocol | Method | Consumer |
|---|---|---|
| `latents.LatentSource` | `info()`, `grid()`, `load(time, layer, …)` | `latents`, `nn` |
| `latents.ReferenceFields` | `field_names()`, `field(name, time)` → per-node values | `latents` |
| `latents.Intervenable` | `info()`, `grid()`, `initial_state()`, `run(state, steps, hooks, noise_seed=, record=)` → `Rollout` | `latents.intervene` |

`Intervenable` is the one protocol that *writes into a model*: a hook at `(layer, time)`
receives the latents the forward pass produced and returns what it continues with. It is
separate from `LatentSource` because running a model is neither cheap nor repeatable, and
most sources cannot. Its adapter must make noise a function of `noise_seed` alone (not of
what the hooks did), or arms cannot be paired; must refuse a place that does not exist
with a `RequestError`; and hands only numpy across, so the model's framework stays on its
side. An adapter that reads nothing declares its options keyword-only.

An adapter may also *write* what it reads: a `write(path, **contents)` on the class the
registry hands out. `latents.toy` reaches the archive writer that way, by
name, and a writer is always tested against its own reader.

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
  (`xaig.latents`). Nothing imports an adapter; it is reached through the
  registry.
- Heavy dependencies go behind an extra and are imported by the adapter, which loads
  lazily — never at `import xaig` time.
- A source that is broken is an `AdapterError`; a request it cannot meet (a node index out
  of range, a field it does not hold) is a `RequestError`. Never a bare `IndexError`.
- Carry what the exporter said. A latent adapter puts the manifest's free-form
  `experiment` block, and the options it was opened with, into `LatentInfo`, so they
  reach the provenance of every result.
- Resolve what moves. A source in a versioned store resolves a branch, a tag or the
  default to an immutable commit when it is opened (a full commit is taken as given),
  reads every file at that commit, and puts both what was asked and what it became into
  `LatentInfo.revision`.
- Read selectively. A `LatentSource` asked for a region must not load the layer.

## Present adapters

- `toy_dynamics.py` — `toy-dynamics`: a small linear system with latent layers, noise seeds
  and a planted feature, an `Intervenable` that reads nothing. For tests and docs only;
  nothing in it is a model of anything (`docs/package/steering.md`).
- `latent_archive.py` — activations recorded from a model, as a directory of
  memory-mapped arrays, with the physical fields kept beside them; `write_archive` writes
  one (`xaig[latents]`; format in `docs/package/latents.md`). The same reader takes an
  `hf://datasets/<owner>/<repo>/<folder>` source and downloads one file at a time
  (`xaig[hf]`).
