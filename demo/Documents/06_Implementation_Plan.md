# Implementation Plan — HA-PIT Streamlit Demo

## What changed from the original plan

The project originally started (per an internal, unpublished handoff note) as a hands-on Streamlit
learning exercise: build it phase-by-phase, with the project owner typing the
code himself while Claude explained concepts first. Partway through Phase 2
(data-layer investigation — inspecting the source `.mat` files), the owner
explicitly changed the instruction: complete the app end-to-end, and produce
this set of planning documents for review instead. Everything below reflects
what was actually built under the new instruction, in the order it happened —
this is a record, not a forward-looking guess.

## Step 1 — Confirm what data actually exists (done)

Before writing any app code, the real inventory of source data was confirmed
by direct inspection (not assumed from the README):

- `results/hpo/*.csv` — three HPO trial-log files, real but with inconsistent
  column names across methods.
- `data/*.csv` — raw per-cell sensor readings (voltage/current/temperature),
  **no SOC or prediction data** — useful for context, not for the results
  charts.
- `models/hapit_v8p6_results.mat` — a curated results struct: summary metrics
  and per-epoch training curves, but **no per-timestep predictions**.
- `8V3/8v4/Baseline_8.4/8v5/8v6/workspace_8v6.mat` (outside the public repo,
  a personal working file) — a full raw MATLAB workspace dump, 106 variables,
  which **did** contain per-timestep predicted/true/error arrays for all
  three splits. This became the actual source for the Results Explorer
  charts.

This inventory step mattered: building against an assumed schema first would
have produced a `data_loader.py` that had to be rewritten once the real
column names and data availability became clear.

## Step 2 — Extract and self-contain the data (done)

Rather than have the app depend on an absolute path to a personal file
outside the repo, a one-time extraction script pulled exactly the needed
arrays out of `workspace_8v6.mat` into `data/predictions.npz` and
`data/training_curves.npz`, both committed inside `streamlit demo/`. The
three HPO CSVs and the architecture diagram were copied in alongside them.
This is what makes the app portable/deployable — see
`02_Technical_Requirements.md`, Cloud deployment.

## Step 3 — Verify before trusting (done)

Before writing `metrics.json` by hand from README prose, every number with
an independent source was cross-checked:
- Test RMSE (0.4129%) confirmed identical in the README,
  `hapit_v8p6_results.mat`, and `workspace_8v6.mat`.
- The 0.10A bias-ablation row confirmed identical between the README and the
  raw `abl` struct in the workspace file.
- The extracted `err_test` array's max value (1.9475) confirmed identical to
  the README's reported test Max Error.

This verification is what justifies the "every number is real, none
fabricated" claim in the PRD — it isn't an assumption, it was checked.

## Step 4 — Build the data layer (done)

`data_loader.py`: four cached loader functions
(`load_metrics`, `load_predictions`, `load_training_curves`,
`load_hpo_trials`). The HPO loader specifically had to normalize three
different column-naming conventions into one schema — see
`05_Data_Architecture.md` for the exact mapping.

## Step 5 — Build the app (done)

`app.py`: six tabs (Overview, Results Explorer, HPO Comparison, Bias
Robustness, How It Works, About), covering the original plan's Phases 3
(app skeleton), 4 (interactive result visuals), 5 (the HPO/bias findings),
and most of 7 (polish — captions, consistent color, an About tab). Every
chart uses live Plotly figures built from the real extracted data — no
embedded static result images (only the architecture diagram, which is
illustrative, not a result, so a static image is appropriate there).

## Step 6 — Apply a validated visual identity (done)

Rather than leave Plotly's default color cycling, the `dataviz` skill's
validated categorical palette was applied consistently across every chart,
and a matching `.streamlit/config.toml` dark theme was set. See
`04_UIUX_Design_Brief.md` for the full color mapping and rationale.

## Step 7 — Test in a real browser (done)

The app was launched (`streamlit run app.py`) and every one of the six tabs
was visually verified in a live browser session: correct headline numbers,
correctly-rendering charts against real data, no console/server errors, and
the theme/color changes confirmed after a restart. This followed the same
bar as any UI change: don't declare a UI feature done without having actually
opened it.

## Step 8 — Write this document set (done)

This file and the five before it.

## What's deliberately NOT done, and why

**Phase 6, live inference, was not built.** This isn't an oversight — the
original handoff explicitly flagged this decision ("MATLAB bridge vs. PyTorch
port... confirm with Amirul before committing real effort to either path") as
one to make consciously, not by default. Given:
- the MATLAB Engine path can't be publicly hosted (no MATLAB on free hosts),
- the PyTorch-port path requires rebuilding the architecture from scratch and
  verifying numerical parity against the original — real, error-prone
  engineering work that shouldn't be done speculatively inside a "complete
  the demo" pass,

the responsible choice was to ship a complete, honest v1 without it, and
document the decision clearly rather than either skip it silently or rush an
unvalidated implementation. If live inference is wanted next, the concrete
next step is: pick a path (the original handoff leaned toward PyTorch, for a
publicly-hostable link), and if PyTorch, port the architecture from
`src/matlab/HAPIT8v6.m`, convert `hapit_v8p6_best_checkpoint.mat`'s weights,
and verify the ported model's output matches the MATLAB original on B0018
*before* wiring it into the app at all.

## Update, 2026-09-29: Phase 6 shipped

Item 3 below is done. The PyTorch port was completed and verified — see
`../PyTorch_port/Documents/` for the full plan and
`02_Technical_Requirements.md`'s "Live model inference — shipped" section
for the result (0.000001 percentage-point RMSE match against the original
MATLAB model on real held-out data). The app now has a seventh tab, "Try
It," running genuine live inference.

## Recommended next steps (not started)

1. **Run the palette validator** (`scripts/validate_palette.js` from the
   `dataviz` skill) against the exact three-hue set used, to have a formal
   pass/fail record rather than relying on the skill's own documented
   defaults being sufficient as-is.
2. **Deploy to Streamlit Community Cloud** and confirm the app runs
   identically in that environment (this build only tested `localhost`) —
   note `requirements.txt` now includes `torch`, which is a larger install
   than the app's previous dependencies; confirm Community Cloud's build
   step handles it within its resource limits.
3. ~~Decide the Phase 6 path~~ — done, PyTorch port shipped 2026-09-29.
4. **Push this repo's changes** (the extracted data files, `app.py`,
   `data_loader.py`, `model.py`, `inference.py`, both Documents folders) to
   the existing GitHub remote — they currently exist only in the local
   `streamlit demo/` folder.
5. **Consider whether to expand "Try It" beyond the four bundled cells**
   once there's a basis for trusting out-of-distribution behavior — see the
   PyTorch port's Flow document for the deferred file-upload option.
