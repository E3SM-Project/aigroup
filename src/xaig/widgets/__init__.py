"""widgets -- a local web app over the rest of xaig.

    $ xaig widgets --latents latents/atmosphere --latents latents/ocean

Presentation only. Every number on screen comes from ``xaig.diagnostics`` and every
figure from ``xaig.figures``, so anything seen here can be reproduced in a notebook
or a batch job -- the app says how. Nothing imports this package.

Needs the ``widgets`` extra (streamlit, plus everything ``figures`` and ``diagnostics`` need).
"""

from __future__ import annotations

__all__: list[str] = []
