"""Experiment records: open one, see where it came from, what it found and how to redo it.

Widgets and layout only. A record is a file ``xaig latents evaluate --record`` or
``xaig latents steer --record`` wrote and ``xaig.latents.load_record`` validates; nothing
is computed here, so every number shown is one the record holds, and the command at the
bottom is the one it holds too.
"""

from __future__ import annotations

import json
import os

import streamlit as st

from xaig.app.config import configured_records
from xaig.app.theme import dark_page
from xaig.core.errors import RequestError
from xaig.figures import response_figure, to_png
from xaig.latents import ExperimentRecord, load_record

_CACHED = 8


@st.cache_data(max_entries=_CACHED, show_spinner=False)
def _load(path: str, stamp: int | None) -> ExperimentRecord:
    """Keyed on when the file changed: a record is rewritten to one name by every rerun."""
    return load_record(path)


def _stamp(path: str) -> int | None:
    try:
        return os.stat(path).st_mtime_ns
    except OSError:
        return None


def _percent(value) -> str:
    return "-" if value is None else f"{100 * value:.1f}%"


def _number(value, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}g}"


def _pick() -> str | None:
    known = st.session_state.setdefault("record_files", configured_records())
    with st.sidebar:
        st.subheader("Record")
        added = st.text_input("Open a record", placeholder="/path/to/steer-record.json")
        if added and added not in known:
            known.append(added)
        if not known:
            return None
        return st.selectbox(
            "Record file", known, index=len(known) - 1 if added else 0, format_func=_label
        )


def _label(path: str) -> str:
    return os.path.basename(path) or path


def _provenance(record: ExperimentRecord) -> None:
    st.subheader("Where it came from")
    p = record.provenance
    identity = " · ".join(str(p[k]) for k in ("model", "component", "checkpoint") if p.get(k))
    revision = p.get("revision") or {}
    rows = [
        {"": "source", "value": p["source"]},
        {"": "network", "value": identity or "not declared"},
        {
            "": "commit",
            "value": (
                f"{revision['commit']} (asked for {revision.get('requested') or 'the default'})"
                if revision.get("commit")
                else "none: a local source, which has no revision"
            ),
        },
        {"": "options", "value": json.dumps(p.get("options") or {})},
        {"": "experiment", "value": json.dumps(p.get("experiment") or {})},
        {"": "xaig", "value": str(p.get("xaig", "-"))},
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    if record.bases:
        st.markdown("**Bases**, by content")
        st.dataframe(
            [
                {"file": b.get("path"), "sha256": b.get("sha256"), "status": b.get("status")}
                for b in record.bases
            ],
            hide_index=True,
            width="stretch",
        )


def _evaluation(record: ExperimentRecord) -> None:
    st.subheader("Held-out evaluation")
    split, results = record.split or {}, record.results
    st.markdown(
        f"Fitted on **{len(split['train'])}** time(s) and scored on **{len(split['test'])}** "
        f"it was not fitted on, with **{len(split['buffer'])}** left out between them "
        f"(`{split.get('scheme', 'blocks')}`, gap {split.get('gap', 0)})."
    )
    with st.expander("The split, as time labels"):
        st.json({k: split[k] for k in ("train", "buffer", "test")}, expanded=False)
    rows = [
        {
            "basis": (e.get("basis") or {}).get("path") or f"basis {i}",
            "features": e["n_features"],
            "fitted on": e.get("fitted_on"),
            "EV train": _percent(e["train"]["explained_variance"]),
            "EV held out": _percent(e["test"]["explained_variance"]),
            "active (train)": _number(e["train"]["mean_active_features"], 3),
            "active (held out)": _number(e["test"]["mean_active_features"], 3),
            "dead (held out)": _percent(e["test"].get("dead_fraction")),
            "near-duplicates": e["redundancy"].get("n_near_duplicates"),
        }
        for i, e in enumerate(results["evaluations"])
    ]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    if curve := results.get("curve"):
        st.markdown(
            "**Fidelity against sparsity**, held out: PCA at each rank, and the dictionaries"
        )
        st.dataframe(
            [
                {
                    "point": point["label"],
                    "mean active features": _number(point["mean_active_features"], 3),
                    "EV held out": _percent(point["explained_variance"]),
                    "PCA at the same sparsity": _percent(point.get("pca_at_same_sparsity")),
                }
                for point in curve["points"]
            ],
            hide_index=True,
            width="stretch",
        )
    if stability := results.get("stability"):
        st.markdown(
            f"**Stability** of {stability['n_dictionaries']} bases: "
            f"{_percent(stability['fraction_recurring'])} of matched features recur at cosine "
            f">= {stability['threshold']:g}; random directions match at "
            f"{stability['chance_similarity']['median']:.3f} (median)."
        )


def _steering(record: ExperimentRecord) -> None:
    st.subheader("Steering, against random directions")
    spec, effects = record.results["spec"], record.results["effects"]
    i = spec["steer"]
    st.markdown(
        f"**{i['mode']} {i['amount']:g}** on feature **{i['feature']}** of layer {i['layer']} at "
        f"time(s) {', '.join(map(str, i['times']))}; {spec['steps']} step(s), seed(s) "
        f"{', '.join(map(str, spec['seeds']))}. Every arm of a seed shares the control's noise, "
        "so a response is a paired difference from it."
    )
    rows = [
        {
            "field": name,
            "feature": e["feature_response"],
            "± (seeds)": e.get("feature_se"),
            "reconstruction only": e["reconstruction_response"],
            "random |response|": _number(e.get("random_mean_magnitude")),
            "effect size": e.get("effect_size"),
            "rank": f"{e['rank']}/{len(e['random_responses']) + 1}",
            "p": e.get("p_value"),
        }
        for name, e in effects.items()
    ]
    st.dataframe(rows, hide_index=True, width="stretch")
    if not effects:
        st.info("The record holds no field.")
        return
    name = st.selectbox("Field", list(effects))
    e = effects[name]
    st.image(
        to_png(
            response_figure(
                e["random_responses"],
                e["feature_response"],
                e["reconstruction_response"],
                title=f"{name}: the feature against {len(e['random_responses'])} random directions",
                dark=dark_page(),
            )  # fmt: skip
        ),
        width="stretch",
    )
    st.caption(
        f"Rank {e['rank']} of {len(e['random_responses']) + 1} (1 is a larger response than "
        f"every draw); p = {_number(e.get('p_value'), 3)}. The reconstruction arm is what passing "
        "through the dictionary does with nothing changed: a feature effect near it is not "
        "distinguishable from the dictionary being lossy."
    )


def _reproduce(record: ExperimentRecord) -> None:
    st.subheader("Reproduce")
    st.code(record.command, language="bash")
    st.download_button(
        "Download the record (JSON)", json.dumps(record.to_dict(), indent=2), "record.json"
    )
    st.json(record.settings, expanded=False)


def page() -> None:
    st.title("Records")
    path = _pick()
    if path is None:
        st.info(
            "Open an experiment record from the sidebar, or start the app with "
            "`xaig app --record FILE`. `xaig latents evaluate --record FILE` and "
            "`xaig latents steer --record FILE` write one."
        )
        return
    try:
        record = _load(path, _stamp(path))
    except RequestError as exc:
        st.warning(str(exc))
        return
    st.caption(f"{record.kind} record · schema version {record.version} · {path}")
    _provenance(record)
    if record.kind == "evaluation":
        _evaluation(record)
    else:
        _steering(record)
    _reproduce(record)
