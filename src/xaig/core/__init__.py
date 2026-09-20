"""Durable core: the errors, the adapter registry, the hint for a missing extra.

Depends on the standard library and nothing else, and knows nothing about the
subpackages built on it.
"""

from __future__ import annotations

from xaig.core.errors import (
    AdapterError,
    MissingExtraError,
    RequestError,
    XaigError,
)

__all__ = [
    "AdapterError",
    "MissingExtraError",
    "RequestError",
    "XaigError",
]
