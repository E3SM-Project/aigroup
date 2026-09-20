"""Exception types. Kept in one place so callers can catch `XaigError` broadly."""

from __future__ import annotations


class XaigError(Exception):
    """Base class for every error xaig raises deliberately."""


class AdapterError(XaigError):
    """An adapter could not be resolved or failed to read its source."""


class RequestError(XaigError, ValueError):
    """What was asked for cannot be done with what is there.

    Distinct from a plain ``ValueError`` on purpose. A client shows this to the
    person who asked, in one line; any other exception is a bug and must keep its
    traceback. Also a ``ValueError``, so code written before it existed still works.
    """


class MissingExtraError(XaigError, ImportError):
    """An optional dependency is absent. Also an ``ImportError``, so code that
    guards an optional import the usual way keeps working."""
