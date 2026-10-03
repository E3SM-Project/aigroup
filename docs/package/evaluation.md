# Evaluating a dictionary

A sparse dictionary scored on the nodes it was fitted to says how well it memorised them.
`xaig.latents.evaluate` scores a *frozen* basis on times it never saw, and says what each
number does and does not tell you. It is an API first (`evaluate_basis`, `seed_stability`,
`fidelity_curve` return objects and print nothing) and `xaig latents evaluate` is a client
of it. It reads `latents.Dictionary`s, which are plain numpy, so nothing here needs torch;
training the seeds and sweeps it compares is [`xaig.nn`](nn.md)'s.

## The protocol

1. **Split by time.** Nodes of one step, and steps next to each other, are nearly one
   sample: a random split of either leaks. The times are cut into contiguous blocks, some
   blocks are held out, and the training times within `--gap` positions of a held-out block
   are left out as a buffer, on both sides (at least one: no training step touches a
   held-out one). The alternatives are whole trajectories (`split_groups`) and a whole other
   archive of the same network (`split_archives`). A split is a list of time labels, and
   every result carries it.
2. **Fit on the training times only.** The standardisation, a PCA, a dictionary:
   `fit_sae(times=split.train)`, `accumulate_moments(times=split.train)`. The basis file
   records the times it was fitted on, and `evaluate_basis` refuses one that lists a
   held-out time. A basis that records none is scored and marked `fitted_on: unknown`:
   the split cannot vouch for it.
3. **Freeze and measure both sides.** Area-weighted, valid nodes only, every time counting
   equally.

```console
$ xaig latents toy scratch/ev/control
$ xaig latents evaluate scratch/ev/control --blocks 4 --split-only
layer 3: fit on 5 time(s), hold out 2 (0424-03-01T18:00:00, 0424-03-02T00:00:00), 1 in the buffer

train  0424-02-27T06:00:00
       0424-02-27T12:00:00
       0424-02-27T18:00:00
       0424-02-28T00:00:00
       0424-03-01T06:00:00
$ for k in 2 4 8; do xaig nn sae scratch/ev/control --features 32 --activation topk --k $k \
    --epochs 20 --batch-size 256 --lr 3e-3 --time 0 --time 1 --time 2 --time 3 --time 4 \
    --out scratch/ev/sae_k$k.npz; done
$ xaig latents evaluate scratch/ev/control --blocks 4 \
    --basis scratch/ev/sae_k2.npz --basis scratch/ev/sae_k4.npz --basis scratch/ev/sae_k8.npz \
    --pca 1,2,4,8,16
BASIS       FEATURES  FITTED  EV_TRAIN  EV_TEST  ACTIVE_TRAIN  ACTIVE_TEST  DEAD_TEST  DUPLICATES
sae_k2.npz  32        train   97.7%     93.1%    2.00          2.00         59.4%      0
sae_k4.npz  32        train   99.1%     97.7%    4.00          4.00         46.9%      0
sae_k8.npz  32        train   99.3%     98.2%    7.99          8.00         15.6%      0

held out, against mean active features
POINT   ACTIVE  EV_TEST  PCA_THERE
pca-1   1.00    45.7%    -
pca-2   2.00    72.2%    -
pca-4   4.00    98.3%    -
pca-8   8.00    100.0%   -
pca-16  16.00   100.0%   -
sae_k2  2.00    93.1%    72.2%
sae_k4  4.00    97.7%    98.3%
sae_k8  8.00    98.2%    100.0%
```

That is the [toy emulator](latents.md), eight times, so these are numbers about the
command, not about any model. They do show the shape of a result: a dictionary with two
features active rebuilds 93% of a held-out layer where two principal components rebuild
72%, and from four active on the toy's low-rank layer the PCA is as good or better.
Nothing is hidden by the training columns being higher: that gap is what a held-out
split is for. `--out FILE` (and `--json`) write all of it with its provenance.

**Pinning what was scored.** Each `--basis` may be followed by a `--basis-sha256 HASH` (the
`xaig nn sae` and `latents pca` commands print it when they write a file): give one per
`--basis`, matched by position, or none; a different count is refused, and a file whose
content has another hash is refused when it is loaded. The command ends with a
`reproduce:` line that repeats the whole invocation with the hash of every basis scored,
so a reader can assert which file produced a table.

## What each number tells you, and what it does not

**Explained variance** is one minus the area-weighted squared error over the variance of
the target about the *training* mean the basis carries, summed over channels, so channels
with the largest variance decide it. It says how much of the layer the features rebuild;
it does not say the features mean anything, and a dictionary can explain 99% of a layer
while every feature is a smear of all its channels. `mse` is the same error per channel
in the layer's own units. For a transcoder the target is the layer it writes
(`target_layer=`, read from the file when it says). Where the target has no variance the
number is undefined and the JSON says `null`.

**Sparsity** is the area-weighted mean number of features active at a node, and `l0_fraction`
that over the width. *Active* means an activation of magnitude above `--active-above`
(zero: anything at all). A relu or top-k dictionary writes exact zeros, but float32
arithmetic leaves 1e-8 where a feature is "off" in a rotated frame, so a threshold of
1e-4 is the honest setting for a hand-built fixture; for a trained one, read the
`firing_rate` summary before choosing. A PCA has every component active wherever its score
clears the threshold, so its sparsity is its rank.

**Dead features** never exceed the threshold at any valid node of that side. On a short
held-out stretch a feature can be dead for want of an occasion: `dormant` lists those
alive in training and silent held out, and the held-out dead fraction above is larger than
the one `xaig nn` reports for training, because it is counted over two times, not the
last epoch's draws. Dead is a fact about a feature and a stretch of data, not about the
dictionary; a feature that the training data also never used is a wasted one, and one
that only the held-out data wakes is evidence the splits differ.

**Redundancy** is the largest signed cosine between a feature's decoder direction and any
other's; `n_near_duplicates` counts features with one at or above `--duplicate-above`
(0.95), and `n_pairs` the pairs. It is a property of the directions, computed blockwise,
and it includes dead features, whose directions are whatever initialisation left. An
opposite direction is not a duplicate: activations are not negative. It finds copies, not
features that are *combinations* of others.

**Stability across seeds** (`seed_stability`, `--stability`) matches the features of two
trainings one to one by decoder cosine (the Hungarian method, in numpy), and reports the
matched cosines and the share at or above `--recur-above`. Read it against
`chance_similarity`, the same matching against random directions: forced pairings of
unrelated unit vectors in few channels find high cosines by luck. On the toy, two seeds of
the same dictionary (`--seed 0` and `--seed 1` of the `xaig nn sae` command above, `--k 4`)
give a median matched cosine of 0.585 where random directions give 0.488 (chance), so the
real seeds are only a little above chance and 6.2% of their features recur at 0.9:

```console
$ xaig latents evaluate scratch/ev/control --blocks 4 \
    --basis scratch/ev/seed0.npz --basis scratch/ev/seed1.npz --stability
...
stability of 2 bases: 6.2% of matched features at cosine >= 0.9 (median 0.585; random directions 0.488)
```

The bases must say they were
fitted to one place in one network, and `same_training_times` says whether they saw the
same data. A feature that does not recur may still be a good one: a dictionary has many
equally good bases, and a seed that finds another is not a failure. A feature that does
recur is not thereby *right*, only reproducible.

**The fidelity-sparsity curve** (`fidelity_curve`, `--pca`) puts a PCA at each `k` (fitted
here, on the training times) beside the dictionaries you supply, as held-out explained
variance against held-out mean active features. `pca_at_same_sparsity` interpolates the
PCA curve at a dictionary's sparsity, and is empty where the PCA curve does not reach.
It compares *at equal L0*, which is not equal cost, equal interpretability or equal
width: a dictionary has more features than channels and a PCA has not. It says
nothing about which is better for steering or naming a feature.

## Python API

```python
from xaig.latents import (
    evaluate_basis,
    fidelity_curve,
    open_source,
    save_result,
    seed_stability,
    split_time_blocks,
)
from xaig.nn.train import fit_seeds, fit_sweep

source = open_source("latents/atmosphere")
split = split_time_blocks(source.info().times, n_blocks=5, test_blocks=[-1], gap=1)

fit = dict(layer=8, n_features=4096, activation="topk", times=list(split.train))
sweep = fit_sweep(source, [{"k": 8}, {"k": 32}, {"k": 128}], **fit)
curve = fidelity_curve(source, split, layer=8, pca_components=[8, 32, 128, 384], dictionaries=sweep)
save_result("curve.json", curve)

seeds = fit_seeds(source, [0, 1, 2], k=32, **fit)
seed_stability(seeds).fraction_recurring
evaluate_basis(seeds[0], source, split, layer=8).test.explained_variance
```

`evaluate_basis` checks the basis against the layer as every other analysis does
(`check_basis_fits`), and puts `result_provenance(info, basis=basis)` in the result: the
commit the archive was opened at and the content hash of the basis. A `fidelity_curve` and
a `seed_stability` carry the hash of every basis in them. With `split_archives` the
held-out archive is passed as `test_source` and its provenance is recorded beside the
first's.

## Remaining tasks

- [ ] Block-bootstrapped intervals on the held-out numbers: eight held-out times is a
      point estimate, not a distribution
- [ ] Splits that buffer by elapsed time (`LatentInfo.elapsed_seconds`) rather than by
      position, for archives that keep irregular steps
- [ ] The same evaluation for a cross-layer transcoder, when there is one
