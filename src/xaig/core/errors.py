"""Exception types. Kept in one place so callers can catch `XaigError` broadly."""

from __future__ import annotations


class XaigError(Exception):
    """Base class for every error xaig raises deliberately."""


class SpecError(XaigError):
    """A campaign spec is malformed, or a run id disagrees with it."""


class AdapterError(XaigError):
    """An adapter could not be resolved or failed to read its source."""


class MissingExtraError(XaigError, ImportError):
    """An optional dependency is absent. Also an ``ImportError``, so code that
    guards an optional import the usual way keeps working."""
