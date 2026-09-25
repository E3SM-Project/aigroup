"""A sparse autoencoder over a model's latents, as plain torch blocks.

One node's vector of channels goes in; a wide, mostly-zero vector of *features*
comes out, from which the input is rebuilt. The hope is the usual one: channels
are entangled, and features, being few at a time, are easier to name.

With ``topk`` this is the k-sparse autoencoder MacMillan & Ouellette (2025,
arXiv:2512.24440) trained on GraphCast's node embeddings; with ``relu`` and an L1
penalty, the standard one.

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

    Not the KAN-SAE of Cheon (2026, arXiv:2605.17493), which has no hard zero,
    starts its control points at zero and places its knots over the measured range
    of pre-activations: this was written before that paper was read, and is to be
    replaced by it. Do not report results from it under that name.

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


def topk_mask(z: torch.Tensor, k: int) -> torch.Tensor:
    """True at each row's ``k`` largest entries -- exactly ``k``, a tie at the cut
    going to the lowest index. The twin of ``xaig.daig.latent.basis.topk_mask``,
    spelt the same way on purpose; ``torch.topk`` alone leaves ties unspecified."""
    cut = z.topk(k, dim=-1).values[..., -1:]
    above, tied = z > cut, z == cut
    room = k - above.sum(dim=-1, keepdim=True)
    return above | (tied & (tied.cumsum(dim=-1) <= room))


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

    @torch.no_grad()
    def normalise_decoder(self) -> None:
        """Bring every feature's direction back to unit length; call after each
        optimiser step. A feature's activation and the length of its direction can
        be traded for one another at no cost to the reconstruction, so without
        this an activation is in units of its own: not comparable between two
        features, which is the first thing anyone does with them."""
        norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-12)
        self.decoder.weight.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        if self.spline is not None:
            return self.spline(z)
        z = torch.relu(z)
        if self.activation == "topk" and self.k < z.shape[-1]:
            z = torch.where(topk_mask(z, self.k), z, 0.0)
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

    @classmethod
    def from_dictionary(cls, dictionary: Dictionary) -> SparseAutoencoder:
        """The block a dictionary was exported from, to go on training it or to
        run it in torch. The standardisation stays with the dictionary: this block
        takes standardised inputs, as it did when it was trained -- and, if the
        dictionary says ``node_norm``, nodes normalised before that."""
        n_features, n_inputs = dictionary.encoder.shape
        n_outputs = dictionary.decoder.shape[1]
        spline = dictionary.spline
        block = cls(
            n_inputs,
            n_features,
            n_outputs=n_outputs,
            activation=dictionary.activation,
            k=dictionary.k,
            spline_intervals=8 if spline is None else spline.shape[1] - 3,
            spline_upper=6.0 if spline is None else dictionary.spline_upper,
        )
        with torch.no_grad():
            block.encoder.weight.copy_(torch.as_tensor(dictionary.encoder))
            block.encoder.bias.copy_(torch.as_tensor(dictionary.encoder_bias))
            block.decoder.weight.copy_(torch.as_tensor(dictionary.decoder.T))
            block.decoder.bias.copy_(torch.as_tensor(dictionary.decoder_bias))
            if block.spline is not None:
                block.spline.coefficients.copy_(torch.as_tensor(spline))
        return block

    def to_dictionary(
        self,
        input_mean,
        input_scale: float = 1.0,
        output_mean=None,
        output_scale: float | None = None,
        node_norm: bool = False,
        **meta,
    ) -> Dictionary:
        """The trained block as plain arrays, for ``xaig.daig`` to use without
        torch. The standardisation the inputs were given travels with it, and so
        does ``node_norm``, if each node was normalised before that."""

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
            node_norm=node_norm,
            meta=meta,
        )
