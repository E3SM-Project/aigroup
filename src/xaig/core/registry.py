"""Adapter lookup and construction.

Adapters are found through the ``xaig.adapters`` entry-point group and through
nothing else, so the adapters shipped here and one from a separate distribution
are resolved identically -- and core never names an adapter module. Loading is
lazy: an adapter's heavy dependencies are imported only when it is asked for.

The factory contract, in full::

    factory(source, **options) -> adapter

- ``source`` is whatever the adapter reads (a path, a directory, a URL). It is
  passed positionally, so the adapter may call its first parameter anything.
- ``options`` are keyword arguments. One the factory does not accept is an error
  naming the ones it does; a typo must not silently mean "use the default".
- ``context`` is offered by the caller rather than supplied by the user (a
  campaign spec, say) and reaches only a factory that declares a parameter of
  that name. ``**kwargs`` is not a declaration: a factory that hands its
  keywords on to another library must not find xaig's own objects among them.
  Nor is the source parameter, whatever it happens to be called.

The object returned implements one or more protocols; see ``core.protocols``.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from importlib.metadata import EntryPoint, entry_points
from typing import Any

from xaig.core.errors import AdapterError

log = logging.getLogger(__name__)

_GROUP = "xaig.adapters"
_OWN_DISTRIBUTION = "xaig"
_REGISTERED: dict[str, Callable[..., Any]] = {}

_Parameter = inspect.Parameter


def register(name: str, factory: Callable[..., Any]) -> None:
    """Register an adapter in-process. Mainly for tests and notebooks."""
    _REGISTERED[name] = factory


def unregister(name: str) -> None:
    _REGISTERED.pop(name, None)


def _entry_points() -> dict[str, EntryPoint]:
    """One entry point per name. xaig's own win a clash, so installing a plugin
    can add adapters but never silently replace a shipped one."""
    found: dict[str, EntryPoint] = {}
    for ep in entry_points(group=_GROUP):
        ours = ep.dist is not None and ep.dist.name == _OWN_DISTRIBUTION
        if ep.name in found and not ours:
            log.warning("adapter %r is registered more than once; ignoring %s", ep.name, ep.value)
            continue
        found[ep.name] = ep
    return found


def available() -> list[str]:
    return sorted(set(_REGISTERED) | set(_entry_points()))


def get(name: str) -> Callable[..., Any]:
    """Resolve an adapter factory by name.

    In-process registrations win, so a test or notebook can shadow a shipped
    adapter without uninstalling anything.
    """
    if name in _REGISTERED:
        return _REGISTERED[name]
    found = _entry_points()
    if name in found:
        try:
            return found[name].load()
        except Exception as exc:  # a broken plugin must not read as a traceback
            raise AdapterError(f"could not load adapter {name!r}: {exc}") from exc
    if not found and not _REGISTERED:
        raise AdapterError(
            f"unknown adapter {name!r}: no adapters are registered at all, which usually "
            "means xaig is not installed (in a checkout: `uv sync`)"
        )
    raise AdapterError(f"unknown adapter {name!r}; available: {', '.join(available())}")


def create(
    name: str,
    source: str | None = None,
    options: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> Any:
    """Build the adapter ``name`` under the factory contract described above."""
    factory = get(name)
    options = dict(options or {})
    context = dict(context or {})
    args = () if source is None else (source,)

    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):  # no introspectable signature: nothing to check
        return factory(*args, **options)

    kinds = {p.kind for p in params.values()}
    positional = [
        p
        for p in params.values()
        if p.kind in (_Parameter.POSITIONAL_ONLY, _Parameter.POSITIONAL_OR_KEYWORD)
    ]
    source_param = positional[0] if positional else None
    keywords = {
        n
        for n, p in params.items()
        if p.kind in (_Parameter.POSITIONAL_OR_KEYWORD, _Parameter.KEYWORD_ONLY)
        and p is not source_param
    }
    open_ended = _Parameter.VAR_KEYWORD in kinds
    offered = {k: v for k, v in context.items() if k in keywords}

    clash = sorted(set(options) & set(offered))
    if clash:
        raise AdapterError(f"adapter {name!r}: option(s) {', '.join(clash)} are set by xaig")
    if source is None and source_param is not None and source_param.default is _Parameter.empty:
        raise AdapterError(f"adapter {name!r} needs a source, and none was given")
    if source is not None and source_param is None and _Parameter.VAR_POSITIONAL not in kinds:
        raise AdapterError(f"adapter {name!r} does not take a source, but {source!r} was given")
    if source_param is not None and source_param.name in options:
        raise AdapterError(
            f"adapter {name!r}: {source_param.name!r} is the source; pass it as the source"
        )

    unknown = [] if open_ended else sorted(set(options) - keywords)
    if unknown:
        accepted = ", ".join(sorted(keywords - set(context))) or "none"
        raise AdapterError(
            f"adapter {name!r} does not accept option(s) {', '.join(unknown)}; accepted: {accepted}"
        )

    given = options.keys() | offered.keys()
    required = {n for n in keywords if params[n].default is _Parameter.empty}
    missing = sorted(required - given)
    if missing:
        raise AdapterError(f"adapter {name!r} is missing required option(s): {', '.join(missing)}")
    return factory(*args, **options, **offered)
