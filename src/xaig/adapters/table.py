"""Discover runs from a delimited table (TSV/CSV).

The most framework-neutral source there is: any campaign that can list its runs
in a text table works with this, with no knowledge of how the runs were produced.
Uses only stdlib ``csv``, so it stays inside the base dependency tier.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from pathlib import Path

from xaig.core.errors import AdapterError
from xaig.core.model import AttrValue, Run
from xaig.core.spec import CampaignSpec


def _coerce(value: str) -> AttrValue:
    text = value.strip()
    if not text:
        return None
    return int(text) if text.isdigit() else text


class TableDiscoverer:
    """Yields one :class:`Run` per data row.

    Attributes come from the table's columns; when a spec supplies an id grammar,
    the attributes parsed from the id win on conflict. The id is the primary key
    and the spec gives its values a known type, so a ``seed`` column reading
    ``S01`` must not shadow the parsed integer ``1`` -- otherwise attributes stop
    round-tripping back into an id.
    """

    def __init__(
        self,
        path: str | Path,
        id_column: str = "runid",
        delimiter: str | None = None,
        spec: CampaignSpec | None = None,
        strict: bool = False,
    ) -> None:
        self.path = Path(path)
        self.id_column = id_column
        self.delimiter = delimiter
        self.spec = spec
        # strict=False lets a table with a few unparseable ids still be listed;
        # a partially-readable campaign is more useful than an exception.
        self.strict = strict

    def _sniff(self, sample: str) -> str:
        if self.delimiter:
            return self.delimiter
        try:
            return csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
        except csv.Error:
            return "\t"

    def discover(self) -> Iterator[Run]:
        if not self.path.exists():
            raise AdapterError(f"table not found: {self.path}")
        text = self.path.read_text()
        if not text.strip():
            return
        reader = csv.DictReader(text.splitlines(), delimiter=self._sniff(text[:4096]))
        if reader.fieldnames is None or self.id_column not in reader.fieldnames:
            raise AdapterError(
                f"{self.path}: no {self.id_column!r} column "
                f"(found: {', '.join(reader.fieldnames or [])})"
            )
        for row in reader:
            run_id = (row.get(self.id_column) or "").strip()
            if not run_id:
                continue
            yield self._to_run(run_id, row)

    def _to_run(self, run_id: str, row: dict[str, str | None]) -> Run:
        attrs: dict[str, AttrValue] = {
            k: _coerce(v or "") for k, v in row.items() if k and k != self.id_column
        }
        if self.spec is not None:
            try:
                attrs.update(self.spec.parse_id(run_id))
            except Exception:
                if self.strict:
                    raise
        return Run(id=run_id, attrs=attrs, location=str(self.path))
