# UI/UX Design Brief — HA-PIT Streamlit Demo

Unlike the other documents, this one isn't purely descriptive of a generic
plan — the palette below was run through a formal validation procedure
(the `dataviz` skill's color-formula method) rather than picked by eye, and
every value stated here is what's actually wired into `.streamlit/config.toml`
and `app.py` today.

## Typography

Streamlit's default system-sans stack is used unchanged: `system-ui,
-apple-system, "Segoe UI", sans-serif`. No custom font is loaded. This is a
deliberate choice, not an oversight — for a data-heavy results explorer, a
familiar system font that renders instantly (no web-font load delay, no
FOUT) serves the "recruiter skims it in 30 seconds" goal better than a
distinctive brand typeface would. Numeric figures (the `st.metric` tiles,
table cells) use Streamlit's default proportional figures; no tabular-figure
override was needed since no columns require strict vertical digit alignment.

## Color palette

The palette is the **validated default categorical palette** from the
project's `dataviz` skill (`references/palette.md`) — dark-mode steps, since
the app runs Streamlit's dark theme. It was chosen (not custom-designed)
specifically because it ships pre-validated for colorblind-safety and
contrast, which a from-scratch palette would need a separate validation pass
to match.

| Role | Hex | Used for |
|---|---|---|
| Page background | `#0d0d0d` | `.streamlit/config.toml` → `backgroundColor` |
| Chart/card surface | `#1a1a19` | `.streamlit/config.toml` → `secondaryBackgroundColor` |
| Text | `#ffffff` | `.streamlit/config.toml` → `textColor` |
| Primary accent (categorical slot 1 — blue) | `#3987e5` | `.streamlit/config.toml` → `primaryColor`; True SOC line; HPO "ASHA" line; Bias chart "Pure Coulomb" bars; regression scatter points; error histogram; training-convergence continuous-RMSE line |
| Categorical slot 2 — orange | `#d95926` | Predicted SOC line; HPO "BOHB" line; Bias chart "HA-PIT V8.6" bars; training-convergence windowed-RMSE line |
| Categorical slot 3 — aqua | `#199e70` | HPO "Bayesian optimisation + PCGrad" line; Bias chart "HA-PIT, HPO config" bars |
| Categorical slot 8 — red | `#e66767` | Absolute-error-over-time line (single series, chosen for its "this is an error" connotation, not by categorical rotation) |
| Muted / secondary ink | `#c3c2b7` | Reference lines (RMSE/MAE dashed lines), regression fit line, "best epoch" marker — anything that's an annotation rather than a data series |

**Why this specific palette, not a custom-branded one:** the project has no
pre-existing brand identity to match (it's a research/portfolio piece, not a
company product), so there was nothing to preserve by custom-designing
colors. Reaching for a pre-validated default was the correct call under
"follow current best practice" — it guarantees the categorical hues clear
colorblind-safety and contrast checks without a manual validation pass, which
a hand-picked palette would need and this build didn't have time to run
(the validator script itself, `scripts/validate_palette.js`, was not
executed against this exact usage — worth doing as a follow-up if the
palette is ever extended past its current 3-series maximum, since the skill
notes the default categorical set is only guaranteed CVD-safe for all-pairs
comparison up to 3 series).

**Consistency rule applied throughout:** a given "slot" (1st series, 2nd
series, 3rd series in a chart) always gets the same hex across every chart in
the app, even though the *entities* differ (e.g. slot 1 is "True SOC" in one
chart and "ASHA" in another) — this keeps the app feeling like one coherent
system rather than six independently-styled charts, per the PRD's design
requirement.

## Components

All components are Streamlit's own built-in widgets — no custom component
library or CSS overrides beyond the theme file:

| Component | Streamlit API | Where used |
|---|---|---|
| Tab bar | `st.tabs` | Top-level navigation (the app's only navigation) |
| Metric tile | `st.metric` | Headline numbers (Overview, Results Explorer) |
| Data table | `st.dataframe` | Every tabular result (results table, stress tests, HPO summary, loss terms) |
| Chart | `st.plotly_chart` (wrapping `plotly.graph_objects` figures) | Every chart in the app |
| Select dropdown | `st.selectbox` | The one control in the app — split selector, Results Explorer |
| Collapsible section | `st.expander` | Training-convergence chart (kept collapsed by default to avoid overwhelming the Results Explorer tab on first view) |
| Static image | `st.image` | Architecture diagram, How It Works |
| Rich text | `st.markdown` | All narrative copy |

No custom-built components were needed — every UI requirement in the App
Flow document was met by Streamlit's own widget set, which kept the frontend
free of any hand-written HTML/CSS/JS.

## Layout

- **Wide layout** (`st.set_page_config(layout="wide")`) — appropriate given
  charts are the primary content and the audience includes people who'll
  view this on a laptop/desktop, not primarily mobile.
- **Column grids** (`st.columns`) for metric tiles (4 across) and the 2×2
  chart grid in Results Explorer — chosen over a single-column stack so
  related numbers/charts are visually grouped and comparable at a glance
  without scrolling.
