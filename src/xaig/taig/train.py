"""Fit a sparse autoencoder to a model's latents.

This is the one place ``taig`` reaches into ``daig``: the batches come from
``daig.latent.iter_batches`` -- valid nodes only, drawn by area, so a plain mean
over a batch is already the area-weighted loss -- and what comes back is a
``daig.latent.Dictionary``, which every analysis and the CLI take wherever they
take a PCA.

It is a toy loop on purpose: Adam, a fixed learning rate, no resampling of dead
features. It trains a useful dictionary on a laptop in minutes, and says how good
it is; making it better is what the blocks being separate is for.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from xaig import __version__
from xaig.core.errors import RequestError
from xaig.core.extras import missing_extra

try:
    import numpy as np
    import torch
except ImportError as exc:
    raise missing_extra("torch", "taig") from exc

from xaig.daig.latent import (
    Dictionary,
    LatentSource,
    NodeNormalised,
    accumulate_moments,
    iter_batches,
)
from xaig.taig.sae import SparseAutoencoder


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def fit_sae(
    source: LatentSource,
    *,
    layer: int,
    n_features: int,
    target_layer: int | None = None,
    activation: str = "relu",
    k: int | None = None,
    l1: float = 5.0,
    epochs: int = 2,
    batch_size: int = 4096,
    lr: float = 1e-3,
    seed: int = 0,
    times: Sequence[str | int] | None = None,
    holdout_times: Sequence[str | int] | None = None,
    node_norm: bool = False,
    log_every: int = 25,
    eval_every: int = 500,
    device: str | None = None,
    progress=None,
) -> Dictionary:
    """Train on ``layer`` of ``source`` and return the dictionary.

    Inputs are centred on the layer's area-weighted mean over the times used and
    divided by one number, so that a node's vector has unit mean square per
    channel; the channels keep their relative sizes. ``l1`` is on that scale.
    Feature directions are kept at unit length throughout, so an activation is in
    the same units for every feature: how much of the (standardised) layer it
    accounts for at that node.
    With ``target_layer`` the block learns to write that layer instead of
    rebuilding its input: a transcoder.

    With ``node_norm`` each node is first centred over its channels and scaled to
    unit RMS (``daig.latent.node_normalise``), and the standardisation above is
    then taken over normalised nodes: the geometry a pre-norm block reads, as
    MacMillan & Ouellette (2025) trained on. The dictionary records it, and is
    still handed raw latents.

    ``meta["metrics"]`` reports the last epoch: the fraction of variance
    explained, the mean number of active features per node, and the fraction of
    features that never fired. ``progress(step, reconstruction)`` is called every
    ``log_every`` steps, if given.

    ``meta["history"]`` is the loss curve: every ``log_every`` steps, that
    batch's reconstruction error and the fraction of its variance left
    unexplained. With ``holdout_times``, times left out of training, a fixed
    sample of their nodes (drawn by area, as training is) is scored every
    ``eval_every`` steps and at the end, as ``meta["history"]["holdout"]``: a
    rising held-out curve under a falling training one is overfitting, which the
    training curve alone cannot show.
    """
    if node_norm:
        if target_layer is not None:
            raise RequestError("node_norm is for an autoencoder, not a transcoder")
        source = NodeNormalised(source)
    info = source.info()
    n_inputs = info.layer(layer).n_channels
    if n_features < 1:
        raise RequestError("n_features must be at least 1")
    if epochs < 1:
        raise RequestError("epochs must be at least 1")
    if log_every < 1 or eval_every < 1:
        raise RequestError("log_every and eval_every must be at least 1")
    held: list[str] = []
    if holdout_times:
        held = [info.times[info.time_index(t)] for t in holdout_times]
        training = [info.times[info.time_index(t)] for t in times] if times else [
            t for t in info.times if t not in set(held)
        ]  # fmt: skip
        both = sorted(set(held) & set(training))
        if both:
            raise RequestError(f"{len(both)} held-out time(s) are also trained on, e.g. {both[0]}")
        if not training:
            raise RequestError("every time is held out; none is left to train on")
        times = training
    moments = accumulate_moments(source, layer=layer, times=times)
    goal = (
        moments
        if target_layer is None
        else accumulate_moments(source, layer=target_layer, times=times)
    )
    n_outputs = goal.mean.size

    torch.manual_seed(seed)
    where = pick_device(device)
    model = SparseAutoencoder(
        n_inputs, n_features, n_outputs=n_outputs, activation=activation, k=k
    ).to(where)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    mean_in = torch.as_tensor(moments.mean, dtype=torch.float32, device=where)
    mean_out = torch.as_tensor(goal.mean, dtype=torch.float32, device=where)

    def standardised(batch):
        if target_layer is None:
            x = (torch.as_tensor(batch, device=where) - mean_in) / moments.scale
            return x, None
        x = (torch.as_tensor(batch[0], device=where) - mean_in) / moments.scale
        return x, (torch.as_tensor(batch[1], device=where) - mean_out) / goal.scale

    holdout = _holdout_sample(source, layer, target_layer, held, batch_size, seed) if held else None
    history: dict[str, Any] = {
        "step": [], "epoch": [], "reconstruction": [], "fraction_unexplained": [],
        "holdout": None if holdout is None else {"step": [], "reconstruction": [],
                                                 "fraction_unexplained": []},
    }  # fmt: skip

    def score_holdout() -> None:
        if holdout is None or step in history["holdout"]["step"]:
            return
        error = power = 0.0
        with torch.no_grad():
            for batch in holdout:
                x, target = standardised(batch)
                _, part, _ = model.loss(x, target)
                wanted = x if target is None else target
                error += float(part) * x.shape[0]
                power += float((wanted**2).sum(-1).mean()) * x.shape[0]
        n = sum(len(b if target_layer is None else b[0]) for b in holdout)
        history["holdout"]["step"].append(step)
        history["holdout"]["reconstruction"].append(error / n)
        history["holdout"]["fraction_unexplained"].append(error / power if power else float("nan"))

    step = 0
    score_holdout()
    for epoch in range(epochs):
        last = epoch == epochs - 1
        error_sum = power_sum = active_sum = 0.0
        n_seen = 0
        fired = torch.zeros(n_features, dtype=torch.bool, device=where)
        batches = iter_batches(
            source,
            layer=layer,
            target_layer=target_layer,
            batch_size=batch_size,
            times=times,
            seed=seed + epoch,
        )
        for batch in batches:
            x, target = standardised(batch)
            total, error, features = model.loss(x, target, l1=l1)
            optimiser.zero_grad()
            total.backward()
            optimiser.step()
            model.normalise_decoder()
            step += 1
            if step % log_every == 0:
                wanted = x if target is None else target
                reconstruction = float(error.detach())
                power = float((wanted**2).sum(-1).mean())
                history["step"].append(step)
                history["epoch"].append(epoch)
                history["reconstruction"].append(reconstruction)
                history["fraction_unexplained"].append(
                    reconstruction / power if power else float("nan")
                )
                if progress is not None:
                    progress(step, reconstruction)
            if step % eval_every == 0:
                score_holdout()
            if last:
                wanted = x if target is None else target
                n = x.shape[0]
                error_sum += float(error.detach()) * n
                power_sum += float((wanted**2).sum(-1).mean()) * n
                active_sum += float((features > 0).sum(-1).float().mean()) * n
                fired |= (features > 0).any(0)
                n_seen += n

    score_holdout()
    metrics = {
        "explained_variance": 1.0 - error_sum / power_sum if power_sum else float("nan"),
        "mean_active_features": active_sum / max(n_seen, 1),
        "dead_fraction": float((~fired).float().mean()),
    }
    if holdout is not None:
        unexplained = history["holdout"]["fraction_unexplained"][-1]
        metrics["holdout_explained_variance"] = 1.0 - unexplained
    meta: dict[str, Any] = {
        "fitted_on": {
            "provenance": info.provenance(),
            "layer": layer,
            "network_layer": info.layer(layer).position,
            "target_layer": target_layer,
            "times": list(moments.times),
            "holdout_times": held or None,
        },
        "training": {
            "n_features": n_features, "activation": activation, "k": k, "l1": l1,
            "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed, "steps": step,
            "log_every": log_every, "eval_every": eval_every, "node_norm": node_norm,
        },
        "metrics": metrics,
        "history": history,
        "xaig": __version__,
    }  # fmt: skip
    return model.to_dictionary(
        input_mean=moments.mean.astype("float32"),
        input_scale=moments.scale,
        output_mean=None if target_layer is None else goal.mean.astype("float32"),
        output_scale=None if target_layer is None else goal.scale,
        node_norm=node_norm,
        **meta,
    )


_HOLDOUT_NODES = 65536  # enough for a curve steady to a few parts in a thousand


def _holdout_sample(source, layer, target_layer, times, batch_size, seed):
    """A fixed sample of nodes from ``times``, drawn by area as training batches
    are, kept in memory as batches: the same nodes are scored at every check."""
    labels = list(times)
    per_time = max(1, _HOLDOUT_NODES // len(labels))
    batches, kept = [], 0
    for batch in iter_batches(
        source, layer=layer, target_layer=target_layer, batch_size=per_time,
        times=labels, seed=seed + 7919,
    ):  # fmt: skip
        n = len(batch if target_layer is None else batch[0])
        take = min(n, per_time)
        if target_layer is None:
            batches.append(batch[:take])
        else:
            batches.append((batch[0][:take], batch[1][:take]))
        kept += take
        if kept >= per_time * len(labels):
            break
    # score in training-sized pieces
    if target_layer is None:
        whole = np.concatenate(batches)
        return [whole[i : i + batch_size] for i in range(0, len(whole), batch_size)]
    inputs = np.concatenate([b[0] for b in batches])
    targets = np.concatenate([b[1] for b in batches])
    return [
        (inputs[i : i + batch_size], targets[i : i + batch_size])
        for i in range(0, len(inputs), batch_size)
    ]
