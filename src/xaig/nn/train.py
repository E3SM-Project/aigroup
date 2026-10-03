"""Fit a sparse autoencoder to a model's latents.

This is the one place ``nn`` reaches into ``latents``: the batches come from
``latents.iter_batches`` -- valid nodes only, drawn by area, so a plain mean
over a batch is already the area-weighted loss -- and what comes back is a
``latents.Dictionary``, which every analysis and the CLI take wherever they
take a PCA.

It is a toy loop on purpose: Adam, a fixed learning rate, no resampling of dead
features. It trains a useful dictionary on a laptop in minutes, and says how good
it is; making it better is what the modules being separate is for.
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
    raise missing_extra("torch", "nn") from exc

from xaig.latents import (
    Dictionary,
    LatentSource,
    accumulate_moments,
    iter_batches,
)
from xaig.nn.sae import SparseAutoencoder


def pick_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class _Tally:
    """Running sums over batches: what ``metrics`` of a fit are made of."""

    def __init__(self, n_features: int, device) -> None:
        self.error = self.power = self.active = 0.0
        self.n = 0
        self.fired = torch.zeros(n_features, dtype=torch.bool, device=device)

    def add(self, wanted: torch.Tensor, error: torch.Tensor, features: torch.Tensor) -> None:
        n = wanted.shape[0]
        self.error += float(error) * n
        self.power += float((wanted**2).sum(-1).mean()) * n
        self.active += float((features > 0).sum(-1).float().mean()) * n
        self.fired |= (features > 0).any(0)
        self.n += n

    def metrics(self) -> dict[str, float]:
        return {
            "explained_variance": 1.0 - self.error / self.power if self.power else float("nan"),
            "mean_active_features": self.active / max(self.n, 1),
            "dead_fraction": float((~self.fired).float().mean()),
        }


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

    ``meta["metrics"]`` describes the dictionary returned: the fraction of variance
    explained, the mean number of active features per node, and the fraction of
    features that never fired, measured in a pass of the frozen final model over
    the last epoch's batches. ``meta["training_metrics"]`` holds the same three as
    the loop saw them during that epoch, while the weights were still moving; it
    is a monitor, not a result. ``progress(step, reconstruction)`` is called now
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
    for each in (moments,) if target_layer is None else (moments, goal):
        if not each.scale > 0.0:  # every channel constant
            raise RequestError(f"layer {each.layer} has no variance; there is nothing to fit")
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

    def batches_of(epoch: int):
        return iter_batches(
            source,
            layer=layer,
            target_layer=target_layer,
            batch_size=batch_size,
            times=times,
            seed=seed + epoch,
        )

    step = 0
    for epoch in range(epochs):
        last = epoch == epochs - 1
        tally = _Tally(n_features, where)
        for batch in batches_of(epoch):
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
                tally.add(x if target is None else target, error.detach(), features.detach())
    training_metrics = tally.metrics()

    # What is exported is the model after its last update, so that is what is measured:
    # the last epoch's batches again, through the frozen model.
    model.eval()
    tally = _Tally(n_features, where)
    with torch.no_grad():
        for batch in batches_of(epochs - 1):
            x, target = standardised(batch)
            _, error, features = model.loss(x, target, l1=0.0)
            tally.add(x if target is None else target, error, features)
    metrics = tally.metrics()
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
        "training_metrics": training_metrics,
        "xaig": __version__,
    }  # fmt: skip
    return model.to_dictionary(
        input_mean=moments.mean.astype("float32"),
        input_scale=moments.scale,
        output_mean=None if target_layer is None else goal.mean.astype("float32"),
        output_scale=None if target_layer is None else goal.scale,
        **meta,
    )
