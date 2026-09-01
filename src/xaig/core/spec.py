"""Campaign specs: the declarative description of one campaign's conventions.

Core never hard-codes a run-id grammar, a parent map, or a metric name. A spec
supplies them, so a new campaign is a YAML file rather than a code change. A
system that has no id grammar at all simply omits ``id_pattern`` and lets its
adapter attach attributes directly.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from xaig.core.errors import SpecError
from xaig.core.model import AttrValue

_SPEC_PACKAGE = "xaig.caig.campaigns"


def _coerce(value: str) -> AttrValue:
    """Digit strings become ints so ``batch=16`` compares numerically.
    Anything else is left alone -- ``E01`` must stay ``E01``."""
    return int(value) if value.isdigit() else value


@dataclass(frozen=True, slots=True)
class Factor:
    """One position in a compound factor word such as ``A3_B16_C1``."""

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
    parent_key: str = "exp"
    parents: Mapping[str, str] = field(default_factory=dict)
    metric: str | None = None
    noise_floor: float | None = None
    discovery: Mapping[str, Any] = field(default_factory=dict)

    # -- id grammar -------------------------------------------------------

    @property
    def _by_key(self) -> dict[str, Factor]:
        return {f.key: f for f in self.factors}

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
        attrs: dict[str, AttrValue] = {k: _coerce(v) for k, v in match.groupdict().items() if v}
        if self.factor_field and self.factor_field in attrs:
            attrs.update(self._parse_factors(str(attrs[self.factor_field]), run_id))
        return attrs

    def _parse_factors(self, word: str, run_id: str) -> dict[str, AttrValue]:
        by_key = self._by_key
        out: dict[str, AttrValue] = {}
        for token in word.split(self.factor_separator):
            if not token:
                continue
            factor = by_key.get(token[0])
            if factor is None:
                raise SpecError(f"{run_id!r}: unknown factor key {token[0]!r} in {word!r}")
            out[factor.name] = _coerce(token[1:])
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
        if self.factor_field and self.factors:
            values[self.factor_field] = self.factor_separator.join(
                f.render(attrs[f.name]) for f in self.factors
            )
        try:
            return self.id_template.format(**values)
        except KeyError as exc:
            raise SpecError(
                f"spec {self.name!r}: id_template needs {exc} but it was not given"
            ) from exc

    def parent_of(self, run: Mapping[str, AttrValue] | str) -> str | None:
        """Parent experiment of an arm, for the seed-spread comparison.

        Keys are compared as strings because YAML turns an unquoted ``2:`` into
        an int, which would otherwise miss silently and report "no parent".
        """
        key = run if isinstance(run, str) else run.get(self.parent_key, "")
        return self.parents.get(str(key))


# -- loading --------------------------------------------------------------


def _factors_from(raw: Any, spec_name: str) -> tuple[Factor, ...]:
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise SpecError(f"spec {spec_name!r}: 'factors' must be a list")
    return tuple(
        Factor(key=str(f["key"]), name=str(f["name"]), width=int(f.get("width", 1))) for f in raw
    )


def from_mapping(data: Mapping[str, Any]) -> CampaignSpec:
    if "name" not in data:
        raise SpecError("campaign spec has no 'name'")
    name = str(data["name"])
    return CampaignSpec(
        name=name,
        description=str(data.get("description", "")),
        id_pattern=data.get("id_pattern"),
        id_template=data.get("id_template"),
        factor_field=data.get("factor_field"),
        factor_separator=str(data.get("factor_separator", "_")),
        factors=_factors_from(data.get("factors"), name),
        parent_key=str(data.get("parent_key", "exp")),
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
    if path.exists():
        text = path.read_text()
    else:
        resource = resources.files(_SPEC_PACKAGE) / f"{name_or_path}.yaml"
        if not resource.is_file():
            known = ", ".join(bundled_names()) or "none"
            raise SpecError(f"no spec {str(name_or_path)!r}; bundled specs: {known}")
        text = resource.read_text()
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SpecError(f"spec {name_or_path!s} did not parse to a mapping")
    return from_mapping(data)
