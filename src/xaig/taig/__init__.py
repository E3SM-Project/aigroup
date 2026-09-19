"""taig -- reusable neural blocks, and toy architectures built from them.

- ``xaig.taig.sae``    a sparse autoencoder / transcoder, and a learnable B-spline
                       activation: plain ``nn.Module`` blocks over tensors
- ``xaig.taig.train``  fitting one to a model's latents, and handing the result
                       to ``xaig.daig`` as a ``Dictionary``

Needs the ``taig`` extra (torch). This package module stays importable without
it, so that ``xaig --help`` can list the ``taig`` command on a base install.
"""

from __future__ import annotations

__all__: list[str] = []
