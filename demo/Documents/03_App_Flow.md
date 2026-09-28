# App Flow — HA-PIT Streamlit Demo

The app is a **single page with six tabs** (`st.tabs`, in `app.py`). There is
no routing, no multi-page navigation, and no login — opening the app's URL is
the entire entry point. Every tab is described below screen-by-screen, control
by control, with what happens on each interaction.

## Global elements (visible on every tab)

- **Title** — "HA-PIT: Ageing-Aware Physics-Informed Transformer"
- **Caption** — one-line project description with a live link to the GitHub
  repo (opens in a new tab; standard link behavior, no in-app handling)
- **Tab bar** — Overview | Results Explorer | HPO Comparison | Bias
  Robustness | How It Works | About. Clicking a tab is a pure client-side
  switch handled by Streamlit — it does **not** trigger a full script rerun
  or any data reload; all six tabs' content is computed once per page load
  and Streamlit just shows/hides the relevant block.

## Tab 1 — Overview

1. **Four metric tiles** (`st.metric`, in a 4-column row): Test RMSE,
   Full-life RMSE, Parameters, Training time. The first two have a `?` hover
   tooltip (`help=`) giving context (which cell, what "full-life" means).
   Static — no interaction.
2. **Narrative paragraph** — fixed text explaining the residual-correction
   architecture in plain language.
3. **"Headline results" table** (`st.dataframe`) — the train/validation/test
   split metrics. Streamlit's default dataframe widget: sortable by column
   header click, but this is a display table, not a control — clicking it
   doesn't affect any other part of the app.
4. **"Stress tests" table** — the two out-of-distribution stress-test rows
   (338-cycle full life, UDDS drive cycle), same widget behavior as above.

No user input on this tab; it's pure orientation.

## Tab 2 — Results Explorer

This is the interactive core of the app.

1. **Split selector** (`st.selectbox`, options: "Test (held out) — B0018"
   [default], "Validation — B0007", "Train — B0005+B0006"). **This is the one
   control that drives real recomputation**: changing it triggers a Streamlit
   rerun of the whole script; because every data-loading function is
   `st.cache_data`-cached, the underlying files are not re-read from disk,
   but the four charts below are rebuilt against the newly selected split's
   DataFrame.
2. **Four metric tiles** — RMSE / MAE / Max error / R² for the currently
   selected split, read from `metrics.json`'s `results_table`. Updates
   immediately when the split selector changes.
3. **Caption** explaining the x-axis convention (timestep index, not literal
   seconds) — static text, always visible regardless of split.
4. **Four charts, 2×2 grid** (each `st.plotly_chart`, independently
   interactive — hover for exact values, drag to zoom, double-click to reset,
   camera icon to download as PNG):
   - **SOC trajectory** (top-left) — true vs. predicted SOC, line chart.
   - **Absolute error over time** (top-right) — with dashed/dotted reference
     lines at the split's RMSE and MAE.
   - **Predicted vs. true SOC** (bottom-left) — scatter with a linear fit
     line; slope is recomputed live via `numpy.polyfit` against whichever
     split is currently selected.
   - **Error distribution** (bottom-right) — histogram of signed error.

   All four re-render whenever the split selector changes; no other
   interaction affects them (no cross-filtering between charts).
5. **"Training convergence" expander** (`st.expander`, collapsed by
   default) — clicking it reveals a line chart of continuous vs.
   windowed-oracle validation RMSE across all 400 training epochs, with a
   dashed vertical line at the best epoch (107). This chart is **not**
   affected by the split selector — it reflects the one training run,
   regardless of which split's predictions are being viewed above.

## Tab 3 — HPO Comparison

1. **Narrative intro** — fixed text on the three search methods.
2. **Convergence chart** — one line per method (ASHA, BOHB, Bayesian
   optimisation + PCGrad), each showing that method's best-RMSE-found-so-far
   against trial number (a running minimum, computed client-side from the
   raw trial CSVs at load time — not stored pre-computed). Fully interactive
   (hover, zoom) but has no controls of its own; it's static content on this
   tab, no selectors.
3. **Caption** clarifying these trial-level numbers use the search harness's
   own settings and aren't directly comparable to the 0.41% production
   number.
4. **"Search summary" table** — trials run, pruned count, best trial RMSE,
   compute hours, per method.
5. **"Fair comparison" table** — the two directly-comparable numbers (hand-
   tuned baseline vs. best searched config, both re-run at full fidelity).
6. **Closing narrative** — explains *why* hand-tuning won (the structural
   fix, not the hyperparameters, is what mattered).

No interactive controls beyond the chart's own native hover/zoom.

## Tab 4 — Bias Robustness

1. **Narrative intro** — explains the bias-injection methodology and why it
   tests something Coulomb counting can't survive.
2. **Grouped bar chart** — three bias levels (0.00 A / 0.05 A / 0.10 A) on
   the x-axis, three bars per group (Pure Coulomb counting, HA-PIT V8.6,
   HA-PIT HPO config). Static data (from `metrics.json`), fully interactive
   chart (hover for exact values), no selector controls.
3. **Closing narrative** — states the actual finding plainly: V8.6 doesn't
   beat Coulomb counting under bias; the HPO-found config does, at a small
   accuracy cost.

## Tab 5 — How It Works

1. **"Residual formulation" section** — narrative text.
2. **Architecture diagram** (`st.image`, from `data/figures/architecture.png`)
   — static image, no interaction beyond the browser's native
   right-click-save/zoom.
3. **"Ageing awareness" section** — narrative text on capacity/resistance
   recalibration.
4. **"Five-term loss" table** — the five loss components, their weights, and
   their purpose, from `metrics.json`.
5. **Caption** on why the drift term (`L_drift`) mattered most.

Entirely static content — no widgets.

## Tab 6 — About

Static markdown block: project attribution, supervisor, GitHub link, data
provenance (NASA PCoE), and licensing (MIT for code, CC BY 4.0 for docs/
figures). One integrity statement: every number in the app is sourced from
the repo's saved evaluation output or README, not recomputed or estimated for
the demo. No interactive elements.

## What does *not* happen anywhere in the app

- No cross-tab state — switching tabs never carries a selection (like the
  Results Explorer's split choice) to another tab.
- No write operations of any kind — nothing the user does persists anywhere
  or affects what any other visitor sees. This is a stateless, read-only
  viewer.
- No error states requiring user recovery — all data is bundled with the app
  at deploy time, so there's no "loading failed, retry" path to design for
  under normal operation.
