# Interventions

Everything in `xaig` so far *reads*: an adapter hands over activations a model produced,
and the package computes on them. A steering experiment asks a different question, "what
does the model do if this feature is changed?", and answering it means the first time an
adapter **writes into** the model. This page is the design note for that, written before
the code and kept as its record. Each section says what is **implemented** and what is
**planned**.

!!! note "status"
    Planned: everything below. This commit adds the note only; the protocol, the runner,
    the command and the toy system follow it, and this page is updated to say what
    shipped.

## What an intervention is

An intervention edits the model's latents at one layer, along a feature's direction, and
then lets the forward pass continue from the edited value. Three things name it:

- **where**: a layer, and the forward steps (times) at which it acts, optionally on a
  subset of nodes;
- **along what**: a feature of a `latents.Decomposition`, i.e. a direction in that layer's
  channel space (`Dictionary.decoder[f]` in the layer's units; a PCA component);
- **by how much**: `add` an amount of the direction, `scale` the feature's own activation
  by a factor, or `clamp` it to a value. Every mode reduces to a per-node change `delta`
  in the feature's activation, and the layer moves by `delta * direction`.

The edit is made on the model's own latents (`h' = h + delta * direction`), not on a
reconstruction of them, so whatever the dictionary fails to explain is left as it was. That
is what makes the arms below separable.

## Why this changes the adapter contract

The contract in `adapters/AGENTS.md` is a set of read protocols (`LatentSource`,
`ReferenceFields`): the adapter is passive, and a result is a function of files that exist.
An intervention is a *run*: the adapter owns the model, and the experiment owns what
happens to the latents at a point inside it. Three consequences:

1. **A new protocol, not a new method on `LatentSource`.** Reading a recorded archive is
   cheap, repeatable and safe; running a model is none of those, and most sources (an
   archive on a hub) cannot do it. One adapter object may implement both, and callers ask
   with `isinstance`, as everywhere else.
2. **The model's side stays in the model's environment.** The protocol hands the adapter
   a function to call and takes arrays back. No torch, no framework object crosses it, so
   the code that designs an experiment still runs on numpy alone.
3. **Determinism becomes the adapter's promise.** A recorded archive is the same every
   time it is read. A run is the same only if the adapter makes its noise a function of a
   seed it is given, which the paired design below needs.

Proposed protocol (`latents.Intervenable`; its consumers all sit on `latents`, so by the
rule in `src/xaig/AGENTS.md` it lives there and moves to `core` only when something
outside needs it):

```python
class Intervenable(Protocol):
    def info(self) -> LatentInfo: ...
    def grid(self) -> Grid: ...
    def initial_state(self, start: str | int = 0) -> Any: ...
    def run(
        self, initial_state, steps: int, hooks: Sequence[Hook] = (), *,
        noise_seed: int | None = None, record: Sequence[tuple[int, int]] = (),
    ) -> Rollout: ...

@dataclass(frozen=True)
class Hook:
    layer: int
    time: int                       # the forward step, from 0
    edit: Callable[[np.ndarray], np.ndarray]   # (n_nodes, n_channels) -> same shape
```

A hook at `(layer, time)` receives the latent tensor the forward pass produced there and
returns what it continues with. `Rollout` holds the physical fields each step wrote
(`(steps, n_nodes)` per name, step `t` being what the pass starting at `t` produced, the
`lead=1` convention of `ReferenceFields`) and the latents at the `record`ed
`(layer, time)` pairs, taken *after* any hook there. The initial state is opaque to xaig:
it is whatever the adapter's `initial_state` returned.

Planned. The factory contract is unchanged; an adapter that reads nothing declares its
options keyword-only, as the synthetic one does.

## The four arms

Each is one run from the same initial state, with the same noise.

| Arm | The run | What it isolates |
|---|---|---|
| control | no hooks | the baseline every difference is taken against |
| reconstruction-only | at the hook point, `h` is replaced by `decode(encode(h))` | the effect of passing through the dictionary at all: its reconstruction error, with no feature touched |
| feature | `h' = h + delta * direction` for the chosen feature | the effect of the feature, plus nothing the dictionary dropped |
| random direction | the same `delta`, along a random unit direction scaled to the feature's direction length; repeated draws | what an edit of this size does to the model in a direction that means nothing |

The reconstruction arm is not subtracted from the feature arm: the feature edit is applied
to `h`, so the feature arm's difference from control *is* the feature's effect. The
reconstruction arm says how large the effect of simply using the dictionary would be, so a
reader can see whether a claimed feature effect is bigger than that. The random arm is the
yardstick for "bigger than an arbitrary push of the same size": the feature's response is
reported as an effect size and an empirical rank within the random draws.

## Paired noise

A stochastic model run twice from one state differs for reasons that have nothing to do
with the edit. All arms of one repetition therefore share a `noise_seed`, so the
difference against control contains the edit and nothing else; with no intervention it is
exactly zero. Seeds are then repeated (`--seeds`) to get an uncertainty: the paired
difference is computed per seed, and the spread across seeds says how far to trust the
mean. The random draws are fixed across seeds, so a draw is one direction tested under
every noise. Pairing is only as good as the adapter's promise that its noise does not
depend on what the hooks did (the noise is drawn from the seed, not from the state).

## What is measured

- **Latent differences**: per arm, the area-weighted RMS of `latents - control latents` at
  every recorded `(layer, time)`, so the spread of an edit through the layers and steps is
  visible, not just its outcome.
- **Physical outcomes**: the fields the model writes (`Rollout.fields`), reduced to an
  area-weighted, mask-aware global mean per step. The *response* of an arm is the mean of
  its paired difference over the steps from the first edit on. The feature's `|response|`
  is set against the `|response|` of each random draw.

Nonfinite latents or fields on a valid node, a hook that changes the shape, and empty
selections (no seeds, no fields, no steps, no draws) are `RequestError`s, as for every
other read in the package.

## Provenance

A result carries `result_provenance(info, basis=...)` of the system it ran (source,
commit, options, the basis's content hash) and the full specification of the experiment,
so a printed reproduction command pins what moves.

## Non-goals

- No real model is run, loaded or downloaded by `xaig`; a real adapter is a later piece
  of work and lives in its own environment.
- No search over features or amounts, and no claim that a feature "causes" anything beyond
  what the arms show; the numbers are evidence for a reader to weigh.
- No intervention on several layers or features at once (a hook list can hold them, but
  the runner varies one feature).
- No gradients: an edit is a forward-pass substitution.
- No plotting here; figures and the app may draw a result later.

## Remaining tasks

- [ ] The protocol, the runner and a toy system to test it against.
- [ ] `xaig latents steer`.
- [ ] A real adapter, in the model's environment.
