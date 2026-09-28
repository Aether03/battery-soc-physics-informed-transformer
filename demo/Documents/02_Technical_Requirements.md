# Technical Requirements Document — HA-PIT Streamlit Demo

No technical guesses below — every tool/version listed was actually installed
and run in this project's environment during the build (Windows 11, Python
3.10.10).

## Scope note: this is a single-tier Python web app

The generic TRD categories of Mobile / Backend / Frontend / Database don't
map cleanly onto a Streamlit app, and forcing them would misrepresent the
architecture. What follows is the honest shape, with each conventional
category addressed explicitly (including "not applicable, because...").

## Mobile

**Not applicable — no native or PWA mobile app exists or is planned.**
Streamlit produces a responsive web page reachable from a mobile browser, but
that is not the same thing as a native/PWA mobile experience (no offline
support, no install prompt, no touch-optimized chart interactions). If a true
mobile experience is ever wanted, it would be a separate project, not an
extension of this one.

## Frontend

- **Framework:** [Streamlit](https://streamlit.io) 1.64.0 — a Python-only
  reactive UI framework. There is no separate JS/HTML/CSS frontend codebase;
  the entire UI is declared in `app.py` using Streamlit's Python API
  (`st.tabs`, `st.columns`, `st.metric`, `st.dataframe`, `st.plotly_chart`).
- **Charting:** [Plotly](https://plotly.com/python/) 6.1.1, via
  `plotly.graph_objects`. `Scattergl` (WebGL-backed) traces are used for the
  large per-timestep series (up to 136,320 points per split) instead of plain
  SVG `Scatter`, for render performance in-browser.
- **Theming:** `.streamlit/config.toml` sets a dark theme with a validated
  color palette (see `04_UIUX_Design_Brief.md`) — this is the only
  "frontend styling" surface; there is no separate CSS file.
- **Execution model:** Streamlit's rerun-on-interaction model — every widget
  interaction (e.g. changing the split dropdown) re-executes `app.py` top to
  bottom. Expensive work is cached with `st.cache_data` so this doesn't mean
  re-reading files from disk on every interaction.

## Backend

**No server process, API layer, or request/response backend exists.**
Streamlit's own built-in server (a Tornado process, bundled inside the
`streamlit` package) is the only "backend," and it isn't something this
project writes code for — the framework owns it entirely. There is no REST/
GraphQL API, no separate compute service, and no background job runner.

All "backend logic" is really **data-loading and light computation**, done
in-process by `data_loader.py`:

| Function | Reads | Returns |
|---|---|---|
| `load_metrics()` | `data/metrics.json` | `dict` |
| `load_predictions(split)` | `data/predictions.npz` | `pandas.DataFrame` (per-timestep) |
| `load_training_curves()` | `data/training_curves.npz` | `dict` of arrays |
| `load_hpo_trials()` | `data/hpo/*.csv` (3 files) | one normalized `pandas.DataFrame` |

Every function is decorated `@st.cache_data`, so each file is read from disk
at most once per server process (or until the underlying file changes).

## Data storage

**No database.** All data is static files, checked into the project
repository:

| File | Format | Size (approx.) |
|---|---|---|
| `data/metrics.json` | JSON | < 5 KB |
| `data/predictions.npz` | NumPy compressed archive | ~3.5 MB |
| `data/training_curves.npz` | NumPy compressed archive | < 50 KB |
| `data/hpo/*.csv` | CSV, 3 files | ~17 KB total |
| `data/figures/architecture.png` | PNG | ~210 KB |
| `data/hapit_pytorch.pt` | PyTorch state_dict (ported weights) | ~330 KB |
| `data/try_it/<cell>.npz` | NumPy compressed archive, 4 files (B0005/B0006/B0007/B0018) | ~13 MB total |

See `05_Data_Architecture.md` for the full field-level schema of each file
and how they relate to each other.

## Cloud deployment

**Target: [Streamlit Community Cloud](https://streamlit.io/cloud)** (free
tier), deployed directly from the project's GitHub repo — no separate
infrastructure, no containerization, no CI/CD pipeline needed beyond
Community Cloud's own git-push-to-deploy flow. This is viable specifically
*because* the app has no MATLAB dependency and no database: everything it
needs is a `pip install -r requirements.txt` away.

## Languages, tools, and confirmed versions

| Tool | Version | Role |
|---|---|---|
| Python | 3.10.10 | Runtime |
| streamlit | 1.64.0 | Web app framework |
| plotly | 6.1.1 | Charting |
| pandas | 2.2.3 | Tabular data handling |
| numpy | (bundled with above) | Array handling |
| scipy | 1.15.3 | Originally build-time only (extracting `.mat` arrays); **now also a runtime dependency** — `inference.py`'s OCV re-anchoring lookup uses `scipy.interpolate.interp1d`. Updated 2026-09-29, see the Live model inference note below. |
| torch | 2.14.0 (CPU build) | Runtime — the ported model's inference engine. Added 2026-09-29. |

## Live model inference — shipped (2026-09-29 update)

**No longer out of scope.** The MATLAB model was ported to PyTorch (see
`../PyTorch_port/Documents/` for the full planning set and verification
methodology) and is now live in the app's "Try It" tab (`app.py` +
`model.py` + `inference.py`). The port's output matches the original MATLAB
model to within 0.000001 percentage points of RMSE on real held-out data —
verified, not assumed.

Current scope: input is limited to the project's own four existing cells
(B0005/B0006/B0007/B0018) — no free-form upload yet. Deliberate, not a
technical limitation: the model isn't yet shown to generalize beyond the
NASA cycling data it was trained on (the README's own UDDS stress test shows
an order-of-magnitude accuracy drop out of distribution). See
`../PyTorch_port/Documents/03_Port_and_Integration_Flow.md` for the full
reasoning and the revisit condition.

## Explicitly out of scope for this version

- **User accounts / authentication.** The app is a public, read-only
  results viewer. There's nothing to log into and nothing per-user to store.
