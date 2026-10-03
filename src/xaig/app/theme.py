"""What the pages share about how they look. Streamlit is imported, so not config.py."""

from __future__ import annotations

import streamlit as st


def dark_page() -> bool:
    """Whether the viewer's theme is dark, where Streamlit is new enough to say."""
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", "light") == "dark"
