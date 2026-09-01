"""daig -- emulator-vs-reference diagnostics.

Skeleton only in 0.1.0. The intended API, for reference while it is built:

    from xaig.daig import reduce_epoch, compare_fields

    scalars = reduce_epoch(run, block="inference", epoch=15)  # -> dict[str, float]
    diff    = compare_fields(prediction, reference, weights="area")

Reductions must be area-weighted and NaN-aware: unweighted means on a Gaussian
grid are wrong, and ocean channels are undefined over land.

Requires the ``viz`` extra for plotting and an adapter for reading fields.
"""

from __future__ import annotations

__all__: list[str] = []
