"""Latent-space diagnostics: what a model's internal channels respond to.

    from xaig.daig.latent import open_source

    source = open_source("latents/atmosphere")
    source.info().layers          # what was recorded, without loading any of it
    nodes = source.grid().within(5, -140, 1500)
    source.load(0, 8, nodes=nodes)  # one region of one layer, and nothing else

Reading is an adapter's job (see ``LatentSource``); this package computes.

- ``source``    the contract: ``LatentSource``, and the optional ``ReferenceFields``
"""

from __future__ import annotations

from xaig.daig.latent.source import (
    LatentInfo,
    LatentSource,
    LayerInfo,
    ReferenceFields,
    open_source,
    parse_time,
)

__all__ = [
    "LatentInfo",
    "LatentSource",
    "LayerInfo",
    "ReferenceFields",
    "open_source",
    "parse_time",
]
