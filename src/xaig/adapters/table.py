"""Discover runs from a delimited table (TSV/CSV).

The most framework-neutral source there is: any campaign that can list its runs
in a text table works with this, with no knowledge of how the runs were produced.
Uses only stdlib ``csv``, so it stays inside the base dependency tier.
"""

from __future__ import annotations

import csv
import logging
import re
from collections.abc import Iterator, Mapping
from pathlib import Path

from xaig.core.errors import AdapterError, SpecError
from xaig.core.model import RESERVED_ATTRS, AttrValue, Issue, Run, RunStatus, coerce_attr
from xaig.core.protocols import IdParser

log = logging.getLogger(__name__)

_NOTATION = re.compile(r"[A-Za-z_]*([0-9]+)")


def _agrees(column: AttrValue, parsed: AttrValue) -> bool:
    """Whether a column says the same thing as the id. It may repeat the id's
    value (``1``) or its notation (``S01``); a blank column says nothing."""
    if column is None or str(column) == str(parsed):
        return True
    if isinstance(parsed, int):
        match = _NOTATION.fullmatch(str(column))
        return match is not None and int(match.group(1)) == parsed
    return False


class TableDiscoverer:
    """Yields one :class:`Run` per data row.

    Attributes come from the table's columns; when a spec supplies an id grammar,
    the attributes parsed from the id win on conflict. The id is the primary key
    and the spec gives its values a known type, so a ``seed`` column reading
    ``S01`` must not shadow the parsed integer ``1`` -- otherwise attributes stop
    round-tripping back into an id. A column that genuinely disagrees with the
    id (``seed`` 99 on a run named ``...S01``) still loses, but the disagreement
    is kept on the run as a ``metadata`` issue instead of vanishing.

    Status is read only from a column named by ``status_column``, through
    ``status_map`` when the table has a vocabulary of its own (``done``,
    ``crashed``). A column that merely happens to be called ``status`` or ``id``
    is kept as ``table_status`` / ``table_id``: those names belong to the run.
    """

    def __init__(
        self,
        path: str | Path,
        id_column: str = "runid",
        delimiter: str | None = None,
        spec: IdParser | None = None,
        strict: bool = False,
        status_column: str | None = None,
        status_map: Mapping[str, str] | None = None,
    ) -> None:
        self.path = Path(path)
        self.id_column = id_column
        self.delimiter = delimiter
        self.spec = spec
        # strict=False lets a table with a few unparseable ids still be listed;
        # a partially-readable campaign is more useful than an exception.
        self.strict = strict
        self.status_column = status_column
        try:
            self.status_map = {
                str(k).strip().lower(): RunStatus(str(v).strip().lower())
                for k, v in (status_map or {}).items()
            }
        except ValueError as exc:
            known = ", ".join(s.value for s in RunStatus)
            raise AdapterError(f"status_map: {exc}; statuses are {known}") from exc

    def _sniff(self, sample: str) -> str:
        if self.delimiter:
            return self.delimiter
        try:
            return csv.Sniffer().sniff(sample, delimiters="\t,;|").delimiter
        except csv.Error:
            return "\t"

    def discover(self) -> Iterator[Run]:
        if not self.path.is_file():
            raise AdapterError(f"table not found: {self.path}")
        text = self.path.read_text()
        if not text.strip():
            return
        reader = csv.DictReader(text.splitlines(), delimiter=self._sniff(text[:4096]))
        columns = reader.fieldnames or []
        for needed in filter(None, (self.id_column, self.status_column)):
            if needed not in columns:
                raise AdapterError(
                    f"{self.path}: no {needed!r} column (found: {', '.join(columns)})"
                )
        for row in reader:
            run_id = (row.get(self.id_column) or "").strip()
            if not run_id:
                continue
            yield self._to_run(run_id, row)

    def _status(self, text: str | None, issues: list[Issue]) -> RunStatus:
        word = (text or "").strip().lower()
        if not word:
            return RunStatus.UNKNOWN
        if word in self.status_map:
            return self.status_map[word]
        try:
            return RunStatus(word)
        except ValueError:
            issues.append(Issue("status", f"unrecognised status {text!r}; add it to status_map"))
            return RunStatus.UNKNOWN

    def _to_run(self, run_id: str, row: dict[str, str | None]) -> Run:
        issues: list[Issue] = []
        status = RunStatus.UNKNOWN
        attrs: dict[str, AttrValue] = {}
        for column, text in row.items():
            if not column or column == self.id_column:
                continue
            if column == self.status_column:
                status = self._status(text, issues)
                continue
            name = f"table_{column}" if column in RESERVED_ATTRS else column
            attrs[name] = coerce_attr(text)

        if self.spec is not None:
            try:
                parsed = self.spec.parse_id(run_id)
            except SpecError as exc:
                if self.strict:
                    raise
                log.debug("%s: %s", self.path, exc)
                issues.append(Issue("id", str(exc)))
            else:
                for key, value in parsed.items():
                    if key in attrs and not _agrees(attrs[key], value):
                        issues.append(
                            Issue(
                                "metadata",
                                f"column {key}={attrs[key]!r} disagrees with the id, "
                                f"which says {value!r}; the id wins",
                            )
                        )
                attrs.update(parsed)
        return Run(
            id=run_id, attrs=attrs, status=status, location=str(self.path), issues=tuple(issues)
        )
