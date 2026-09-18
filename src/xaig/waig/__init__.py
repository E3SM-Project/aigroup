"""waig -- a local web app over the rest of xaig.

    $ xaig waig --latents latents/atmosphere --latents latents/ocean

Presentation only. Every number on screen comes from ``xaig.caig`` or
``xaig.daig`` and every figure from ``xaig.viz``, so anything seen here can be
reproduced in a notebook or a batch job -- the app says how. Nothing imports
this package.

Needs the ``waig`` extra (streamlit, matplotlib, numpy); add ``maps`` for
coastlines.
"""

from __future__ import annotations

__all__: list[str] = []
