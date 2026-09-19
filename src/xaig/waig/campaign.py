"""The campaign view: what ran, in what state, and what does not add up.

The same answers as ``xaig caig ls`` and ``xaig caig check``, from the same API.
"""

from __future__ import annotations

import os
from collections import Counter

import streamlit as st

from xaig.caig import bundled_specs, check_campaign, load_campaign, load_spec
from xaig.core.errors import XaigError
from xaig.waig.config import SOURCE_ENV, SPEC_ENV

_OTHER = "a spec file…"


def page() -> None:
    st.title("Campaigns")
    bundled = bundled_specs()
    preset = os.environ.get(SPEC_ENV, "")
    with st.sidebar:
        st.subheader("Campaign")
        choices = [*bundled, _OTHER]
        chosen = st.selectbox(
            "Spec", choices, index=choices.index(preset) if preset in bundled else len(bundled)
        )
        if chosen == _OTHER:
            chosen = st.text_input("Path to a spec", value="" if preset in bundled else preset)
        source = st.text_input(
            "Source",
            value=os.environ.get(SOURCE_ENV, ""),
            help="What the spec's adapter reads, such as a manifest. Leave empty if the spec "
            "names its own.",
        )
    if not chosen:
        st.info("Choose a campaign spec in the sidebar.")
        return
    try:
        spec = load_spec(chosen)
        campaign = load_campaign(spec, source=source or None)
        findings = check_campaign(campaign, spec)
    except XaigError as exc:
        st.error(str(exc))
        return

    st.caption(spec.description or spec.name)
    counts = Counter(str(run.status) for run in campaign)
    tiles = st.columns(len(counts) + 2)
    tiles[0].metric("runs", len(campaign))
    for tile, (status, n) in zip(tiles[1:], sorted(counts.items()), strict=False):
        tile.metric(status, n)
    tiles[-1].metric("problems", len(findings))

    columns = [*spec.id_fields, *(f.name for f in spec.factors)] or None
    st.dataframe(campaign.table(columns), hide_index=True, width="stretch")

    if findings:
        st.subheader("What does not add up")
        st.markdown(
            "**run ids** fail the spec's grammar or occur twice; **metadata** is where the "
            "source contradicts a run's id (the id wins) or could not be read."
        )
        st.dataframe(
            [
                {
                    "kind": "run id" if f.kind == "id" else f.kind,
                    "run": f.run_id,
                    "problem": f.message,
                }
                for f in findings
            ],
            hide_index=True,
            width="stretch",
        )
    else:
        st.success(f"All {len(campaign)} run ids fit the spec, and nothing contradicts them.")
