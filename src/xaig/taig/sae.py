"""A sparse autoencoder over a model's latents, as plain torch blocks.

One node's vector of channels goes in; a wide, mostly-zero vector of *features*
comes out, from which the input is rebuilt. The hope is the usual one: channels
are entangled, and features, being few at a time, are easier to name.

Blocks take and return tensors and know nothing of archives, grids or training
loops (``xaig.taig.train`` has those), so they can be lifted into any harness.
Inputs are expected standardised; that is the caller's business.

Trained against another layer instead of its own input, the same block is a
*transcoder*: it reads one layer and writes a later one, and its features are
then steps of computation rather than directions of representation. A
cross-layer transcoder is several of these sharing an encoder.
"""

from __future__ import annotations

from xaig.core.extras import missing_extra

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise missing_extra("torch", "taig") from exc

from xaig.daig.latent.basis import ACTIVATIONS, Dictionary, spline_knots


class BSplineActivation(nn.Module):
    """A learnable activation per feature: zero for ``z <= 0``, a uniform cubic
    B-spline on ``(0, upper]``, a line of slope one beyond.

    It starts as a ReLU exactly -- a B-spline whose coefficients sit at its
    Greville abscissae is the identity -- and training bends each feature's own
    curve from there: a threshold, a saturation, a dead zone. The hard zero is
    kept whatever is learned, so the code stays sparse. This is the twin of
    ``xaig.daig.latent.bspline_activation``, which evaluates a trained one with
    numpy; the two are tested to agree.
    """

    def __init__(self, n_features: int, n_intervals: int = 8, upper: float = 6.0) -> None:
        super().__init__()
        self.n_intervals, self.upper = n_intervals, float(upper)
        knots = torch.as_tensor(spline_knots(n_intervals, upper), dtype=torch.float32)
        self.coefficients = nn.Parameter(knots.repeat(n_features, 1))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        step = self.upper / self.n_intervals
        inside = z.clamp(0.0, self.upper)
        cell = (inside / step).floor().long().clamp(max=self.n_intervals - 1)
        u = inside / step - cell
        blend = (
            (1.0 - u) ** 3 / 6.0,
            (3.0 * u**3 - 6.0 * u**2 + 4.0) / 6.0,
            (-3.0 * u**3 + 3.0 * u**2 + 3.0 * u + 1.0) / 6.0,
            u**3 / 6.0,
        )
        feature = torch.arange(z.shape[-1], device=z.device).expand_as(cell)
        spline = sum(b * self.coefficients[feature, cell + m] for m, b in enumerate(blend))
        return torch.where(z > 0.0, spline + (z - self.upper).clamp(min=0.0), 0.0)


class SparseAutoencoder(nn.Module):
    """``x -> features -> x_hat``, with one of three ways of being sparse.

    - ``"relu"``: a ReLU, kept sparse by an L1 penalty on the features (weighted
      by their decoder norms, so shrinking a feature and growing its decoder
      buys nothing).
    - ``"topk"``: only each node's ``k`` largest features survive; no penalty,
      and the sparsity is exactly what was asked for.
    - ``"bspline"``: a ``BSplineActivation`` per feature, with the L1 penalty.

    With ``n_outputs`` it writes something other than what it read: a transcoder.
    """

    def __init__(
        self,
        n_inputs: int,
        n_features: int,
        *,
        n_outputs: int | None = None,
        activation: str = "relu",
        k: int | None = None,
        spline_intervals: int = 8,
        spline_upper: float = 6.0,
    ) -> None:
        super().__init__()
        if activation not in ACTIVATIONS:
            raise ValueError(f"activation must be one of {', '.join(ACTIVATIONS)}")
        if activation == "topk" and not (k and 1 <= k <= n_features):
            raise ValueError(f"a topk autoencoder needs 1 <= k <= {n_features}")
        self.activation, self.k = activation, k
        self.encoder = nn.Linear(n_inputs, n_features)
        self.decoder = nn.Linear(n_features, n_outputs or n_inputs)
        self.spline = (
            BSplineActivation(n_features, spline_intervals, spline_upper)
            if activation == "bspline"
            else None
        )
        with torch.no_grad():
            # Unit decoder directions, and an encoder that starts as their transpose
            # where the shapes allow: each feature first reads what it writes.
            directions = torch.randn(self.decoder.weight.shape)
            self.decoder.weight.copy_(directions / directions.norm(dim=0, keepdim=True))
            if self.decoder.weight.shape[0] == n_inputs:
                self.encoder.weight.copy_(self.decoder.weight.T)
            self.encoder.bias.zero_()
            self.decoder.bias.zero_()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.spline is not None:
            return self.spline(z)
        z = torch.relu(z)
        if self.activation == "topk" and self.k < z.shape[-1]:
            cut = z.topk(self.k, dim=-1).values[..., -1:]
            z = torch.where(z >= cut, z, 0.0)
        return z

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return self.decoder(features), features

    def loss(self, x: torch.Tensor, target: torch.Tensor | None = None, l1: float = 0.0):
        """``(total, reconstruction, features)``. Reconstruction is the squared
        error summed over channels and averaged over nodes."""
        rebuilt, features = self(x)
        error = ((rebuilt - (x if target is None else target)) ** 2).sum(-1).mean()
        if self.activation == "topk" or not l1:
            return error, error, features
        penalty = (features.abs() * self.decoder.weight.norm(dim=0)).sum(-1).mean()
        return error + l1 * penalty, error, features

    def to_dictionary(
        self,
        input_mean,
        input_scale: float = 1.0,
        output_mean=None,
        output_scale: float | None = None,
        **meta,
    ) -> Dictionary:
        """The trained block as plain arrays, for ``xaig.daig`` to use without
        torch. The standardisation the inputs were given travels with it."""

        def array(tensor: torch.Tensor):
            return tensor.detach().cpu().numpy().astype("float32")

        spline = self.spline
        return Dictionary(
            encoder=array(self.encoder.weight),
            encoder_bias=array(self.encoder.bias),
            decoder=array(self.decoder.weight.T),
            decoder_bias=array(self.decoder.bias),
            input_mean=input_mean,
            input_scale=float(input_scale),
            output_mean=output_mean,
            output_scale=None if output_scale is None else float(output_scale),
            activation=self.activation,
            k=self.k,
            spline=None if spline is None else array(spline.coefficients),
            spline_upper=1.0 if spline is None else spline.upper,
            meta=meta,
        )
