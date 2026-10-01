"""nn -- torch modules trained on a model's latents: a sparse autoencoder.

- ``xaig.nn.sae``    a sparse autoencoder / transcoder, and a learnable B-spline
                     activation: plain ``nn.Module``s over tensors
- ``xaig.nn.train``  fitting one to a model's latents, and handing the result
                     to ``xaig.latents`` as a ``Dictionary``

Needs the ``nn`` extra (torch). This package module stays importable without
it, so that ``xaig --help`` can list the ``nn`` command on a base install.
"""

from __future__ import annotations

__all__: list[str] = []
