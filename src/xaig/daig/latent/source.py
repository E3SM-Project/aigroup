"""Where latents come from: the ``LatentSource`` contract.

This is the seam between running a model and studying it. Recording activations
needs the model's own environment (torch, the framework, a checkpoint, often a
different Python); analysing them needs numpy. An exporter on one side writes
latents down, an adapter on the other reads them back through this protocol, and
nothing here ever imports a model.

The contract lives next to its only consumer. It moves to ``xaig.core`` when a
second subpackage needs it, and not before.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from xaig.core import registry
from xaig.core.errors import AdapterError
from xaig.core.extras import missing_extra
from xaig.daig.grid import Grid

try:
    import numpy as np
except ImportError as exc:
    raise missing_extra("numpy", "daig") from exc

DEFAULT_ADAPTER = "latent-archive"


@dataclass(frozen=True, slots=True)
class LayerInfo:
    """One place in a network where activations were recorded."""

    index: int
    label: str
    n_channels: int


@dataclass(frozen=True, slots=True)
class LatentInfo:
    """What a source holds, without loading any of it.

    Times are the source's own labels, kept as text: emulators run on calendars
    (no-leap, year 425) that the usual datetime types cannot hold, and nothing in
    the analysis needs to do arithmetic on them. ``off_grid_layers`` were
    recorded but live on a coarser grid than ``grid()`` describes -- the inner
    levels of a U-Net, say -- so they are listed and not loadable here.
    """

    source: str
    times: tuple[str, ...]
    layers: tuple[LayerInfo, ...]
    n_nodes: int
    model: str | None = None
    component: str | None = None
    checkpoint: str | None = None
    calendar: str | None = None
    timestep_seconds: int | None = None
    off_grid_layers: tuple[LayerInfo, ...] = ()

    def provenance(self) -> dict[str, Any]:
        """What a result must carry to be traceable to the model that produced it."""
        return {
            "source": self.source,
            "model": self.model,
            "component": self.component,
            "checkpoint": self.checkpoint,
        }

    def layer(self, index: int) -> LayerInfo:
        for layer in self.layers:
            if layer.index == index:
                return layer
        known = ", ".join(str(layer.index) for layer in self.layers)
        raise KeyError(f"no layer {index}; layers are {known}")

    @property
    def last_layer(self) -> int:
        return max(layer.index for layer in self.layers)

    def time_index(self, time: str | int) -> int:
        """Position of a time given by its label, or by position already."""
        if isinstance(time, int):
            if not -len(self.times) <= time < len(self.times):
                raise KeyError(f"time index {time} is out of range for {len(self.times)} time(s)")
            return time % len(self.times)
        if time not in self.times:
            raise KeyError(f"no latents at {time!r}; times are {', '.join(self.times)}")
        return self.times.index(time)


@runtime_checkable
class LatentSource(Protocol):
    """Supplies recorded activations, one layer at one time, selectively.

    ``load`` returns float32 ``(n_nodes, n_channels)``, narrowed to ``nodes`` and
    ``channels`` when they are given. Selection is part of the contract because
    the full array rarely fits comfortably: one time of a 9-layer, 384-channel
    1-degree model is 0.9 GB, while a region of one layer is a few hundred KB. An
    implementation should read only what was asked for.

    The array returned is new and the caller's to modify: analyses centre it in
    place rather than hold a second copy.
    """

    def info(self) -> LatentInfo: ...

    def grid(self) -> Grid: ...

    def load(
        self,
        time: str | int,
        layer: int,
        channels: Sequence[int] | None = None,
        nodes: Sequence[int] | None = None,
    ) -> np.ndarray: ...


def open_source(source: str | Path, adapter: str = DEFAULT_ADAPTER, **options: Any) -> LatentSource:
    """Open latents through the adapter registry, like every other source in xaig."""
    built = registry.create(adapter, source=str(source), options=options)
    if not isinstance(built, LatentSource):
        raise AdapterError(f"adapter {adapter!r} does not implement LatentSource")
    return built
