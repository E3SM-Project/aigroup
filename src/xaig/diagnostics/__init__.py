"""diagnostics -- what emulators hold inside.

- ``xaig.diagnostics.grid``    nodes on a sphere: masks, area weights, regions
- ``xaig.diagnostics.latent``  what a network's internal channels respond to

Needs the ``diagnostics`` extra (numpy). This package module stays importable without
it, so that ``xaig --help`` can list the ``diagnostics`` command on a base install; the
modules above say which extra they need the moment they are imported.
"""

from __future__ import annotations

__all__: list[str] = []
