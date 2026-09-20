"""Plain-text rendering. Presentation only, and stdlib only."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _cell(value: Any) -> str:
    return "-" if value is None or value == "" else str(value)


def table(rows: Sequence[Mapping[str, Any]], columns: Iterable[str] | None = None) -> str:
    """Left-aligned fixed-width table. Returns '' for no rows so callers can
    distinguish 'nothing found' from 'a header with nothing under it'."""
    if not rows:
        return ""
    cols = list(columns) if columns is not None else list(rows[0].keys())
    widths = {c: max(len(c), *(len(_cell(r.get(c))) for r in rows)) for c in cols}
    out = ["  ".join(c.upper().ljust(widths[c]) for c in cols).rstrip()]
    for row in rows:
        out.append("  ".join(_cell(row.get(c)).ljust(widths[c]) for c in cols).rstrip())
    return "\n".join(out)


def pairs(mapping: Mapping[str, Any]) -> str:
    """Aligned ``key: value`` block."""
    if not mapping:
        return ""
    width = max(len(k) for k in mapping)
    return "\n".join(f"{k.ljust(width)}  {_cell(v)}" for k, v in mapping.items())
