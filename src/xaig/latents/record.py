"""The experiment record: what one evaluation or steering run found, kept small.

A result (``Evaluation``, ``SteeringResult``) is complete and large: every run, every
pairing, every time series. A record is what a person comes back to -- who produced it,
from what, how, and the numbers -- as one JSON file that a client can open without
rerunning anything:

- ``provenance``: the source (model, component, checkpoint, the commit a hub archive was
  opened at, the options it was opened with, what the exporter said about the run) and the
  version of xaig, as ``result_provenance`` gives it;
- ``bases``: each basis used, by file and content hash;
- ``settings``: what the command was asked, enough to ask it again;
- ``split``: for an evaluation, the time labels of both sides and the buffer;
- ``results``: the evaluation's numbers, or the steering experiment's effects with the
  random-direction draws they are set against, and not the runs behind them;
- ``command``: the shell command that reproduces it.

The file is versioned. ``load_record`` reads versions it knows and refuses the rest with a
``RequestError``: a record from a later xaig is not guessed at, and neither is a file that
is not one. A reader may rely on everything named here being present; a writer may add
keys within a version, and a reader ignores what it does not know. A change that removes
or renames a key, or changes what one means, is a new ``VERSION``.

Nothing here runs anything. It is a contract between whoever writes a record (the
``evaluate`` and ``steer`` commands, a notebook) and whoever reads one (the app).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from xaig.core.errors import RequestError
from xaig.latents.basis import Decomposition, basis_provenance, result_provenance
from xaig.latents.evaluate import Evaluation, FidelityCurve, Split, Stability, jsonable
from xaig.latents.source import LatentInfo
from xaig.latents.steering import SteeringResult

FORMAT = "xaig.experiment-record"
VERSION = 1
SUPPORTED = (1,)
EVALUATION, STEERING = "evaluation", "steering"
KINDS = (EVALUATION, STEERING)


@dataclass(frozen=True)
class ExperimentRecord:
    """A record, validated. Every field is plain JSON (mappings, lists, numbers, text)."""

    kind: str
    version: int
    provenance: Mapping[str, Any]
    bases: tuple[Mapping[str, Any], ...]
    settings: Mapping[str, Any]
    split: Mapping[str, Any] | None
    results: Mapping[str, Any]
    command: str

    @property
    def commit(self) -> str | None:
        """The commit SHA of the archive, if it came from a versioned store."""
        return (self.provenance.get("revision") or {}).get("commit")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT, "schema_version": self.version, "kind": self.kind,
            "provenance": dict(self.provenance), "bases": [dict(b) for b in self.bases],
            "settings": dict(self.settings), "split": self.split and dict(self.split),
            "results": dict(self.results), "command": self.command,
        }  # fmt: skip


# -- writing ------------------------------------------------------------------


def evaluation_record(
    info: LatentInfo,
    split: Split,
    *,
    evaluations: Sequence[Evaluation] = (),
    curve: FidelityCurve | None = None,
    stability: Stability | None = None,
    bases: Sequence[Decomposition] = (),
    settings: Mapping[str, Any],
    command: str,
) -> ExperimentRecord:
    """A record of ``evaluate_basis`` on each basis, and of the curve and stability that
    were asked for. The split is held once, at the top, as every evaluation shares it."""
    entries = []
    for evaluation in evaluations:
        entry = evaluation.to_dict()
        entry["basis"] = (entry.pop("provenance", None) or {}).get("basis")
        entry.pop("split", None)
        entries.append(entry)
    results: dict[str, Any] = {"evaluations": entries}
    if curve is not None:
        results["curve"] = curve.to_dict()
    if stability is not None:
        results["stability"] = stability.to_dict()
    used = [basis_provenance(b) for b in bases]
    return _build(
        EVALUATION, result_provenance(info), used, settings, split.to_dict(), results, command
    )


def steering_record(
    result: SteeringResult, *, settings: Mapping[str, Any], command: str
) -> ExperimentRecord:
    """A record of a steering experiment: its spec and, per field, the feature's response
    and the reconstruction arm's beside every random draw's. The runs and pairings, which
    are the bulk of a result, stay with ``SteeringResult.to_dict`` for whoever wants them."""
    provenance = dict(result.provenance)
    basis = provenance.pop("basis", None)
    results = {
        "spec": dict(result.spec),
        "effects": {name: effect.to_dict() for name, effect in result.effects.items()},
    }
    return _build(STEERING, provenance, [basis], settings, None, results, command)


def _build(
    kind: str,
    provenance: Mapping[str, Any],
    bases: Sequence[Mapping[str, Any] | None],
    settings: Mapping[str, Any],
    split: Mapping[str, Any] | None,
    results: Mapping[str, Any],
    command: str,
) -> ExperimentRecord:
    entries = [b for b in bases if b]
    # Through JSON and back: a record in hand is exactly what a reader of its file sees.
    plain = json.loads(
        json.dumps(
            jsonable(
                {
                    "provenance": provenance,
                    "bases": entries,
                    "settings": settings,
                    "split": split,
                    "results": results,
                }  # fmt: skip
            )
        )
    )
    return record_from_dict({
        "format": FORMAT, "schema_version": VERSION, "kind": kind, **plain, "command": command,
    })  # fmt: skip


def save_record(path: str | Path, record: ExperimentRecord) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record.to_dict(), indent=2) + "\n")
    return path


# -- reading ------------------------------------------------------------------


def _need(value: Any, keys: Sequence[str], where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RequestError(f"{where} must be a mapping, not {type(value).__name__}")
    absent = [key for key in keys if key not in value]
    if absent:
        raise RequestError(f"{where} lacks {', '.join(absent)}")
    return value


def _list(value: Any, where: str) -> list:
    if not isinstance(value, list):
        raise RequestError(f"{where} must be a list, not {type(value).__name__}")
    return value


def _number(value: Any, where: str, *, null: bool = False) -> None:
    """A finite number: not text, not a bool, not NaN or infinity. ``null`` admits None,
    which is how JSON holds a number that was not one (``jsonable``)."""
    if value is None and null:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        wanted = "a finite number or null" if null else "a finite number"
        raise RequestError(f"{where} must be {wanted}, not {value!r}")


def _check_evaluation(split: Any, results: Any, where: str) -> None:
    _need(split, ("train", "test", "buffer"), f"{where} split")
    for side in ("train", "test", "buffer"):
        _list(split[side], f"{where} split {side}")
    _need(results, ("evaluations",), f"{where} results")
    for i, entry in enumerate(_list(results["evaluations"], f"{where} evaluations")):
        label = f"{where} evaluation {i}"
        _need(entry, ("n_features", "train", "test", "redundancy"), label)
        for side in ("train", "test"):
            _need(entry[side], ("explained_variance", "mean_active_features"), f"{label} {side}")
            for key in ("explained_variance", "mean_active_features"):
                _number(entry[side][key], f"{label} {side} {key}", null=True)
        _number(entry["test"].get("dead_fraction"), f"{label} test dead_fraction", null=True)
        _need(entry["redundancy"], (), f"{label} redundancy")
    if (curve := results.get("curve")) is not None:
        _need(curve, ("points",), f"{where} curve")
        for i, point in enumerate(_list(curve["points"], f"{where} curve points")):
            label = f"{where} curve point {i}"
            _need(point, ("label", "mean_active_features", "explained_variance"), label)
            for key in ("mean_active_features", "explained_variance", "pca_at_same_sparsity"):
                _number(point.get(key), f"{label} {key}", null=True)
    if (stability := results.get("stability")) is not None:
        label = f"{where} stability"
        _need(
            stability,
            ("n_dictionaries", "fraction_recurring", "threshold", "chance_similarity"),
            label,
        )
        _number(stability["fraction_recurring"], f"{label} fraction_recurring", null=True)
        _number(stability["threshold"], f"{label} threshold")
        _need(stability["chance_similarity"], ("median",), f"{label} chance_similarity")
        _number(stability["chance_similarity"]["median"], f"{label} chance_similarity median")


def _check_steering(results: Any, where: str) -> None:
    _need(results, ("spec", "effects"), f"{where} results")
    spec = _need(results["spec"], ("steer", "steps", "seeds"), f"{where} spec")
    label = f"{where} spec"
    _list(spec["seeds"], f"{label} seeds")
    i = _need(
        spec["steer"],
        ("mode", "amount", "feature", "layer", "times"),
        f"{label} steer",
    )
    _number(i["amount"], f"{label} amount")
    _list(i["times"], f"{label} times")
    for name, effect in _need(results["effects"], (), f"{where} effects").items():
        label = f"{where} effect of {name!r}"
        _need(
            effect,
            ("feature_response", "reconstruction_response", "random_responses", "rank"),
            label,
        )
        for key in ("feature_response", "reconstruction_response"):
            _number(effect[key], f"{label} {key}")
        for n, draw in enumerate(_list(effect["random_responses"], f"{label} random_responses")):
            _number(draw, f"{label} random_responses[{n}]")
        for key in ("p_value", "random_mean_magnitude"):
            _number(effect.get(key), f"{label} {key}", null=True)


def record_from_dict(data: Any, *, where: str = "the record") -> ExperimentRecord:
    """Validate what ``to_dict`` wrote (or a file held) and make a record of it.

    ``RequestError`` for anything that is not a record this xaig reads: another format, a
    version it does not know (older or newer), a kind it has no view of, a missing part,
    or a part of the wrong type or shape. What the readers use is checked, types and
    finiteness included; keys they do not use are not.
    """
    top = _need(data, ("format", "schema_version"), where)
    if top["format"] != FORMAT:
        raise RequestError(f"{where} is not an xaig experiment record (format {top['format']!r})")
    version = top["schema_version"]
    if type(version) is not int or version not in SUPPORTED:  # not a bool, nor 1.0
        known = ", ".join(str(v) for v in SUPPORTED)
        newer = type(version) is int and version > max(SUPPORTED)
        raise RequestError(
            f"{where} is schema version {version!r}; this xaig reads version {known}"
            + (" -- it is from a newer xaig, upgrade to read it" if newer else "")
        )
    _need(top, ("kind", "provenance", "bases", "settings", "results", "command"), where)
    kind = top["kind"]
    if kind not in KINDS:
        raise RequestError(f"{where} is of kind {kind!r}; known kinds are {', '.join(KINDS)}")
    results, split = top["results"], top.get("split")
    if not isinstance(top["command"], str):
        raise RequestError(f"{where}: 'command' must be text")
    for i, basis in enumerate(_list(top["bases"], f"{where} bases")):
        _need(basis, (), f"{where} basis {i}")
    _need(top["provenance"], ("source",), f"{where} provenance")
    _need(top["settings"], (), f"{where} settings")
    _need(results, (), f"{where} results")
    if kind == EVALUATION:
        _check_evaluation(split, results, where)
    else:
        _check_steering(results, where)
    return ExperimentRecord(
        kind=kind,
        version=version,
        provenance=top["provenance"],
        bases=tuple(top["bases"]),
        settings=top["settings"],
        split=split,
        results=results,
        command=top["command"],
    )


def load_record(path: str | Path) -> ExperimentRecord:
    """Read a record file, as ``save_record`` wrote it."""
    path = Path(path)
    try:
        text = path.read_text()
    except OSError as exc:
        raise RequestError(f"cannot read {path}: {exc.strerror or exc}") from exc
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise RequestError(f"{path} is not JSON ({exc})") from exc
    return record_from_dict(data, where=str(path))


__all__ = [
    "EVALUATION",
    "FORMAT",
    "KINDS",
    "STEERING",
    "SUPPORTED",
    "VERSION",
    "ExperimentRecord",
    "evaluation_record",
    "load_record",
    "record_from_dict",
    "save_record",
    "steering_record",
]
