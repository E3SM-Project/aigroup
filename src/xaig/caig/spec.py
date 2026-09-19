"""Campaign specs: the declarative description of one campaign's conventions.

Nothing in xaig hard-codes a run-id grammar, a parent map, or a metric name. A
spec supplies them, so a new campaign is a YAML file rather than a code change.
A system that has no id grammar at all simply omits ``id_pattern`` and lets its
adapter attach attributes directly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from xaig.core.errors import SpecError
from xaig.core.model import RESERVED_ATTRS, AttrValue, coerce_attr

_SPEC_PACKAGE = "xaig.caig.campaigns"


@dataclass(frozen=True, slots=True)
class Factor:
    """One position in a compound factor word such as ``A3_B16_C1``: a ``key`` of
    one or more letters, then a whole number."""

    key: str
    name: str
    width: int = 1

    def render(self, value: AttrValue) -> str:
        return f"{self.key}{int(value):0{self.width}d}"


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    name: str
    description: str = ""
    id_pattern: str | None = None
    id_template: str | None = None
    factor_field: str | None = None
    factor_separator: str = "_"
    factors: tuple[Factor, ...] = ()
    parent_key: str | None = None
    parents: Mapping[str, str] = field(default_factory=dict)
    metric: str | None = None
    noise_floor: float | None = None
    discovery: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        groups = self._groups()
        if self.factor_field and self.id_pattern and self.factor_field not in groups:
            raise SpecError(
                f"spec {self.name!r}: factor_field {self.factor_field!r} is not a named "
                f"group of id_pattern (groups: {', '.join(groups) or 'none'})"
            )
        if self.parents and not self.parent_key:
            raise SpecError(f"spec {self.name!r}: 'parents' needs a 'parent_key' to look up by")
        for what, values in (("key", [f.key for f in self.factors]), ("name", self.names())):
            repeated = sorted({v for v in values if values.count(v) > 1})
            if repeated:
                raise SpecError(
                    f"spec {self.name!r}: factor {what}(s) {', '.join(repeated)} appear twice"
                )
        unusable = [f.key for f in self.factors if not f.key.isalpha()]
        if unusable:
            # A key ending in a digit cannot be told from the number after it.
            raise SpecError(
                f"spec {self.name!r}: factor key(s) {', '.join(map(repr, unusable))} must be "
                "letters only"
            )
        taken = [n for n in (*groups, *(f.name for f in self.factors)) if n in RESERVED_ATTRS]
        if taken:
            raise SpecError(
                f"spec {self.name!r}: {', '.join(taken)} cannot name a group or factor; a run "
                "already has an id and a status of its own"
            )

    # -- id grammar -------------------------------------------------------

    def _groups(self) -> tuple[str, ...]:
        if not self.id_pattern:
            return ()
        try:
            compiled = re.compile(self.id_pattern)
        except re.error as exc:
            raise SpecError(f"spec {self.name!r}: id_pattern does not compile: {exc}") from exc
        return tuple(compiled.groupindex)

    @property
    def id_fields(self) -> tuple[str, ...]:
        """Attributes the id names directly, in id order -- the compound factor
        word aside, since its content is what ``factors`` spells out."""
        return tuple(g for g in self._groups() if g != self.factor_field)

    def names(self) -> list[str]:
        return [f.name for f in self.factors]

    def parse_id(self, run_id: str) -> dict[str, AttrValue]:
        """Decompose a run id into attributes.

        With no ``id_pattern`` this returns ``{}`` rather than failing: an
        opaque id is a legitimate design, not an error.
        """
        if not self.id_pattern:
            return {}
        match = re.match(self.id_pattern, run_id)
        if match is None:
            raise SpecError(f"run id {run_id!r} does not match spec {self.name!r}")
        attrs: dict[str, AttrValue] = {k: coerce_attr(v) for k, v in match.groupdict().items() if v}
        if self.factor_field and self.factor_field in attrs:
            attrs.update(self._parse_factors(str(attrs[self.factor_field]), run_id))
        return attrs

    def _parse_factors(self, word: str, run_id: str) -> dict[str, AttrValue]:
        by_key = {f.key: f for f in self.factors}
        out: dict[str, AttrValue] = {}
        for token in word.split(self.factor_separator):
            if not token:
                continue
            key = token.rstrip("0123456789")
            factor = by_key.get(key)
            if factor is None:
                raise SpecError(f"{run_id!r}: unknown factor key {key!r} in {word!r}")
            out[factor.name] = coerce_attr(token[len(key) :])
        missing = [f.name for f in self.factors if f.name not in out]
        if missing:
            raise SpecError(f"{run_id!r}: factor word {word!r} is missing {', '.join(missing)}")
        return out

    def format_id(self, attrs: Mapping[str, AttrValue]) -> str:
        """Rebuild a run id from attributes -- the inverse of :meth:`parse_id`.

        Round-tripping every id in a campaign is the cheapest possible check that
        a spec actually describes it, so this is not merely a convenience.
        """
        if not self.id_template:
            raise SpecError(f"spec {self.name!r} has no id_template")
        values = dict(attrs)
        try:
            if self.factor_field and self.factors:
                values[self.factor_field] = self.factor_separator.join(
                    f.render(attrs[f.name]) for f in self.factors
                )
            return self.id_template.format(**values)
        except KeyError as exc:
            raise SpecError(
                f"spec {self.name!r}: id_template needs {exc} but it was not given"
            ) from exc
        except (TypeError, ValueError) as exc:
            raise SpecError(f"spec {self.name!r}: cannot format an id: {exc}") from exc

    def parent_of(self, run: Mapping[str, AttrValue] | str) -> str | None:
        """Parent experiment of an arm, for the seed-spread comparison.

        Keys are compared as strings because YAML turns an unquoted ``2:`` into
        an int, which would otherwise miss silently and report "no parent".
        """
        key = run if isinstance(run, str) else run.get(self.parent_key or "", "")
        return self.parents.get(str(key))


# -- loading --------------------------------------------------------------


def _factors_from(raw: Any, spec_name: str) -> tuple[Factor, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise SpecError(f"spec {spec_name!r}: 'factors' must be a list")
    try:
        return tuple(
            Factor(key=str(f["key"]), name=str(f["name"]), width=int(f.get("width", 1)))
            for f in raw
        )
    except (KeyError, TypeError, AttributeError) as exc:
        raise SpecError(
            f"spec {spec_name!r}: each factor needs a 'key' and a 'name' ({exc!r})"
        ) from exc


def from_mapping(data: Mapping[str, Any]) -> CampaignSpec:
    if "name" not in data:
        raise SpecError("campaign spec has no 'name'")
    name = str(data["name"])
    known = {f.name for f in fields(CampaignSpec)}
    unknown = sorted(set(data) - known)
    if unknown:
        # A misspelt key would otherwise mean "use the default", with no sign of it.
        raise SpecError(
            f"spec {name!r}: unknown key(s) {', '.join(unknown)}; known: {', '.join(sorted(known))}"
        )
    parent_key = data.get("parent_key")
    return CampaignSpec(
        name=name,
        description=str(data.get("description", "")),
        id_pattern=data.get("id_pattern"),
        id_template=data.get("id_template"),
        factor_field=data.get("factor_field"),
        factor_separator=str(data.get("factor_separator", "_")),
        factors=_factors_from(data.get("factors"), name),
        parent_key=None if parent_key is None else str(parent_key),
        parents={str(k): str(v) for k, v in (data.get("parents") or {}).items()},
        metric=data.get("metric"),
        noise_floor=data.get("noise_floor"),
        discovery=dict(data.get("discovery") or {}),
    )


def bundled_names() -> list[str]:
    """Specs shipped inside the package."""
    return sorted(
        p.name.removesuffix(".yaml")
        for p in resources.files(_SPEC_PACKAGE).iterdir()
        if p.name.endswith(".yaml")
    )


def load(name_or_path: str | Path) -> CampaignSpec:
    """Load a spec by bundled name (``aug26``) or filesystem path."""
    path = Path(name_or_path)
    if path.is_file():
        text = path.read_text()
    else:
        resource = resources.files(_SPEC_PACKAGE) / f"{name_or_path}.yaml"
        if not resource.is_file():
            known = ", ".join(bundled_names()) or "none"
            raise SpecError(f"no spec {str(name_or_path)!r}; bundled specs: {known}")
        text = resource.read_text()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise SpecError(f"spec {name_or_path!s} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise SpecError(f"spec {name_or_path!s} did not parse to a mapping")
    return from_mapping(data)
