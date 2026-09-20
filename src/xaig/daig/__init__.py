"""daig -- diagnostics of emulators: what they hold inside.

- ``xaig.daig.grid``    nodes on a sphere: masks, area weights, regions
- ``xaig.daig.latent``  what a network's internal channels respond to

Needs the ``daig`` extra (numpy). This package module stays importable without
it, so that ``xaig --help`` can list the ``daig`` command on a base install; the
modules above say which extra they need the moment they are imported.
"""

from __future__ import annotations

__all__: list[str] = []
