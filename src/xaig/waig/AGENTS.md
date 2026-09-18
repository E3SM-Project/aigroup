# waig — the web app

A local Streamlit app over `caig` and `daig`. Optional, and downstream of everything:
it may import `core`, `caig`, `daig` and `faig`; **nothing may import it**.

## Rules

- **Presentation only.** No science here: if a number is computed in this package, it
  belongs in `daig` or `caig`, where a notebook can reach it and a test can check it
  without a browser. Figures come from `xaig.faig`.
- **Everything on screen is reproducible off screen.** A view that shows an analysis
  also shows the settings, the command and the code that produce it.
- **Cache what is slow, and bound every cache** (`max_entries`). Streamlit reruns a page
  on each interaction and `st.cache_data` copies what it returns. Open sources go in
  `cache_resource`; results and rendered maps in `cache_data`; whole layers nowhere.
- **Local and offline.** It reads what is on disk. No uploads, no accounts, no tracking
  service — `caig`'s non-goals hold here too.
- `cli.py` and `config.py` must import on a base install (`xaig --help` imports every
  cli module): standard library and click only. Streamlit is found, not imported, there.
- The framework is an adapter-grade choice. Keeping pages this thin is what makes
  replacing Streamlit a rewrite of `waig/` and nothing else.

## Layout

- `cli.py` — `xaig waig …`; passes paths to the app through the environment (`config.py`)
- `app.py` — the script Streamlit runs: page configuration and navigation only
- `latent.py`, `campaign.py` — one view each, as a `page()` function

Adding a view is a module with a `page()` and a line in `app.py`.

## Testing

`streamlit.testing.v1.AppTest` runs the app headlessly: assert on what a view shows and
that changing a widget changes it. It cannot see layout — look at the app before
calling a view done.
