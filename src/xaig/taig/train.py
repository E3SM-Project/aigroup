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
    import torch
except ImportError as exc:
    raise missing_extra("torch", "taig") from exc

from xaig.daig.latent import Dictionary, LatentSource, accumulate_moments, iter_batches
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

    ``meta["metrics"]`` reports the last epoch: the fraction of variance
    explained, the mean number of active features per node, and the fraction of
    features that never fired. ``progress(step, reconstruction)`` is called now
    and then, if given.
    """
    info = source.info()
    n_inputs = info.layer(layer).n_channels
    if n_features < 1:
        raise RequestError("n_features must be at least 1")
    if epochs < 1:
        raise RequestError("epochs must be at least 1")
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

    step = 0
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
            if progress is not None and step % 25 == 0:
                progress(step, float(error.detach()))
            if last:
                wanted = x if target is None else target
                n = x.shape[0]
                error_sum += float(error.detach()) * n
                power_sum += float((wanted**2).sum(-1).mean()) * n
                active_sum += float((features > 0).sum(-1).float().mean()) * n
                fired |= (features > 0).any(0)
                n_seen += n

    metrics = {
        "explained_variance": 1.0 - error_sum / power_sum if power_sum else float("nan"),
        "mean_active_features": active_sum / max(n_seen, 1),
        "dead_fraction": float((~fired).float().mean()),
    }
    meta: dict[str, Any] = {
        "fitted_on": {
            "provenance": info.provenance(),
            "layer": layer,
            "network_layer": info.layer(layer).position,
            "target_layer": target_layer,
            "times": list(moments.times),
        },
        "training": {
            "n_features": n_features, "activation": activation, "k": k, "l1": l1,
            "epochs": epochs, "batch_size": batch_size, "lr": lr, "seed": seed, "steps": step,
        },
        "metrics": metrics,
        "xaig": __version__,
    }  # fmt: skip
    return model.to_dictionary(
        input_mean=moments.mean.astype("float32"),
        input_scale=moments.scale,
        output_mean=None if target_layer is None else goal.mean.astype("float32"),
        output_scale=None if target_layer is None else goal.scale,
        **meta,
    )
