# Steering

Everything in `xaig` so far *reads*: an adapter hands over activations a model produced,
and the package computes on them. A steering experiment asks a different question, "what
does the model do if this feature is changed?", and answering it means the first time an
adapter **writes into** the model. This page is the design note for that, written before
the code and kept as its record, then updated as it shipped. Each section says what is
**implemented** and what is **planned**.

!!! note "status"
    Implemented: the protocol (`Intervenable`, `Hook`, `Steer`), the runner
    (`run_steering`), the `xaig latents steer` command and a toy system to test them
    against. Planned: a real adapter, a view and a figure in the app, and fitted
    dictionaries on the toy system; see [Remaining tasks](#remaining-tasks).

## What a steer is

A steer edits the model's latents at one layer, along a feature's direction, and
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
A steer is a *run*: the adapter owns the model, and the experiment owns what
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

The protocol (`latents.Intervenable`; its consumers all sit on `latents`, so by the
rule in `src/xaig/AGENTS.md` it lives there and moves to `core` only when something
outside needs it):

```python
class Intervenable(Protocol):
    def info(self) -> LatentInfo: ...
    def grid(self) -> Grid: ...
    def initial_state(self, start: str | int = 0) -> Any: ...
    def run(
        self,
        initial_state,
        steps: int,
        hooks: Sequence[Hook] = (),
        *,
        noise_seed: int | None = None,
        record: Sequence[tuple[int, int]] = (),
    ) -> Rollout: ...


@dataclass(frozen=True)
class Hook:
    layer: int
    time: int  # the forward step, from 0
    edit: Callable[[np.ndarray], np.ndarray]  # (n_nodes, n_channels) -> same shape
```

A hook at `(layer, time)` receives the latent tensor the forward pass produced there and
returns what it continues with. `Rollout` holds the physical fields each step wrote
(`(steps, n_nodes)` per name, step `t` being what the pass starting at `t` produced, the
`lead=1` convention of `ReferenceFields`) and the latents at the `record`ed
`(layer, time)` pairs, taken *after* any hook there. The initial state is opaque to xaig:
it is whatever the adapter's `initial_state` returned.

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
difference against control contains the edit and nothing else; with no steer it is
exactly zero. Seeds are then repeated (`--seeds`) to get an uncertainty: the paired
difference is computed per seed, and the spread across seeds says how far to trust the
mean. The random draws are fixed across seeds, so a draw is one direction tested under
every noise. Pairing is only as good as the adapter's promise that its noise does not
depend on what the hooks did (the noise is drawn from the seed, not from the state).

## What is measured

- **Latent differences**: per arm, the area-weighted RMS of `latents - control latents`
  (over nodes and channels) at every recorded `(layer, time)`, so the spread of an edit
  through the layers and steps is visible, not just its outcome.
- **Physical outcomes**: the fields the model writes (`Rollout.fields`), reduced to an
  area-weighted, mask-aware global mean per step. The *response* of an arm is the mean of
  its paired difference over the steps from the first edit on. The feature's `|response|`
  is set against the `|response|` of each random draw.

Nonfinite latents or fields on a valid node, a hook that changes the shape or returns a
nonfinite value, and empty selections (no seeds, no fields, no steps, no draws) are
`RequestError`s, as for every other read in the package. An edit never changes a masked
node.

Implemented: `run_steering` returns a `SteeringResult` holding every `Run` (arm, seed,
draw, outcome series, latent RMS), every `Pairing` against the control of its seed, and
an `Effect` per field: the feature's signed response and its standard error across
seeds, the reconstruction arm's, the random draws', and the feature's `effect_size`,
`rank` (1 is a larger magnitude than every draw) and `p_value` among them.
The edit's size is worked out once per seed from the control's latents, so every arm of a
seed changes the same nodes by the same amounts however the runs drift apart; `scale` and
`clamp` are therefore relative to the control, not to the arm's own activations.

## Provenance

A result carries `result_provenance(info, basis=...)` of the system it ran (source,
commit, options, the basis's content hash) and the full specification of the experiment,
so a printed reproduction command pins what moves: `xaig latents steer` ends with a
`reproduce:` line that carries the adapter and its options, `--basis-sha256`, and every
setting. A result saved with `save_result` is the same record as JSON.

## Trying it

The synthetic system is linear and its latents do not read its state back, so the answer is
known: adding `a` to channel 0 of layer 1 moves temperature by `gain * a` on that step and by
`decay` times as much on each step after, and nothing else. The planted dictionary reads
three of its four channels, so it has a reconstruction error too. The numbers below are
from running exactly this (`masked=3` leaves three nodes with NaN fields, like land).

```console
$ python -c "
from xaig.adapters.toy_dynamics import ToyDynamics
from xaig.latents import save_basis
save_basis('scratch/steer/planted.npz', ToyDynamics(masked=3).planted_dictionary())"
$ xaig latents steer --adapter toy-dynamics --adapter-option masked=3 \
    --basis scratch/steer/planted.npz --layer 1 --feature 0 --amount 1.5 --time 1 \
    --steps 5 --seeds 0,1,2 --random-draws 20
add 1.5 on feature 0 of layer 1 at time(s) 1; 5 step(s), seed(s) 0, 1, 2, 20 random direction(s)

FIELD        RESPONSE  +-       RECON_ONLY  RANDOM_|RESP|  EFFECT_SIZE  RANK   PERCENTILE
moisture     0         0        0.3107      0.2964         -1.83        21/21  0
pressure     0         0        0.1266      0.4251         -2.07        21/21  0
temperature  1.406     1.2e-09  0.3335      0.4719         2.52         1/21   100

reproduce: xaig latents steer --adapter toy-dynamics --adapter-option masked=3 --basis scratch/steer/planted.npz --basis-sha256 72ac49085d99b99ce37574a829a851708590fb99c1501497f907d09db4440960 --layer 1 --feature 0 --mode add --amount 1.5 --time 1 --steps 5 --seeds 0,1,2 --random-draws 20 --random-seed 0
```

- **response** is the mean over steps 1 to 4 of the paired temperature difference:
  `1.5 * 2.0 * (1 + 1/2 + 1/4 + 1/8) / 4 = 1.406`, to rounding. Moisture and pressure do
  not move at all. The standard error across the three noise seeds is `1e-9`: the paired
  difference holds the edit and nothing else.
- **random** directions of the same length move temperature by `gain * 1.5 * r[0]` on the
  first step with `|r[0]| < 1`, so none matches the feature (rank 1 of 21), while on
  moisture and pressure, which the feature does not reach, it is the weakest of the draws
  (rank 21 of 21): an effect size is only meaningful against the field it is about.
- **recon_only** is what substituting `decode(encode(h))` does with nothing edited. Here
  it is a quarter of the feature's response (0.33 against 1.41), which is the reason the
  arm exists: a feature effect near that size would not be distinguishable from the
  dictionary being lossy.

`scale` and `clamp` change the activation the feature has, so their size depends on the
noise and the standard error is no longer zero:

```console
$ xaig latents steer --adapter toy-dynamics --adapter-option masked=3 \
    --basis scratch/steer/planted.npz --layer 1 --feature 0 --mode scale --amount 3 \
    --time 1 --steps 5 --seeds 0,1,2 --random-draws 20 --field temperature
FIELD        RESPONSE  +-      RECON_ONLY  RANDOM_|RESP|  EFFECT_SIZE  RANK  PERCENTILE
temperature  0.4099    0.0045  0.3335      0.1375         2.52         1/21  100
```

From Python, `run_steering(system, basis, Steer(layer=1, feature=0, amount=1.5,
times=(1,)), steps=5, seeds=(0, 1, 2))` returns the same as an object, and `save_result`
writes it.

## Non-goals

- No real model is run, loaded or downloaded by `xaig`; a real adapter is a later piece
  of work and lives in its own environment.
- No search over features or amounts, and no claim that a feature "causes" anything beyond
  what the arms show; the numbers are evidence for a reader to weigh.
- No steer on several layers or features at once (a hook list can hold them, but
  the runner varies one feature).
- No gradients: an edit is a forward-pass substitution.
- No plotting here; figures and the app may draw a result later.

## Remaining tasks

- [x] The protocol, the runner and a toy system to test it against.
- [x] `xaig latents steer`.
- [ ] A real adapter, in the model's environment.
- [ ] A view in the app, and a figure of the feature's response against the random draws.
- [ ] Fitted dictionaries on the toy system, to show a learned feature that is not planted.
