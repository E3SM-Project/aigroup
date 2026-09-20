"""The app's entry script: ``streamlit run`` executes this file.

A shell and nothing more. Each view is a function in its own module, so adding
one is a module plus a line here.
"""

from __future__ import annotations

import streamlit as st

from xaig.waig import latent

st.set_page_config(page_title="xaig", layout="wide")
st.navigation(
    [
        st.Page(latent.page, title="Latents", url_path="latents", default=True),
    ]
).run()
