# Product Requirements Document — HA-PIT Streamlit Demo

## 1. Overview

An interactive Streamlit web app that presents the results of **HA-PIT**
(Ageing-Aware Physics-Informed Transformer), a final-year-project battery
state-of-charge (SOC) estimation model, built on top of the existing GitHub
repo [`battery-soc-physics-informed-transformer`](https://github.com/Aether03/battery-soc-physics-informed-transformer).

The app does not run new experiments or train anything. It is a **results
explorer**: it takes real, already-computed research output (evaluation
metrics, per-timestep predictions, HPO trial logs) and makes it interactively
explorable, in place of a static PDF/README a reader would otherwise have to
parse manually.

Secondary goal, explicitly set by the project owner: this was originally
scoped as a hands-on exercise to learn Streamlit. That goal was superseded
mid-project by a direct request to have the app completed end-to-end — see
`06_Implementation_Plan.md` for what changed and when.

## 2. Goals and Objectives

1. **Portfolio depth, not a dumbed-down showcase.** A recruiter or reviewer
   should come away understanding not just "the model is accurate" but *why*
   the two headline findings are non-trivial:
   - Hand-tuning beat three real hyperparameter-search methods (ASHA, BOHB,
     Bayesian optimisation + PCGrad).
   - A bias-robustness ablation produced a genuine trade-off, not a clean win
     — the best-clean-accuracy configuration does *not* beat plain Coulomb
     counting under sensor bias; a different configuration does, at a small
     accuracy cost.
2. **Every number shown is real.** No synthetic or illustrative data anywhere
   in the app. Where real per-timestep data existed, it's used; where it
   didn't (e.g. the bias-ablation table), the numbers are transcribed
   verbatim from the source repo's README and cross-checked against raw
   saved evaluation output where an overlap existed.
3. **Self-contained and deployable.** The app must not depend on absolute
   paths to files outside its own project folder — everything it reads lives
   under `streamlit demo/data/`.

## 3. Requirements

### Designing
- One visual identity (typography, color) applied consistently across every
  chart and every page — not each chart styled independently.
- Charts must be interactive (hover, zoom) — not static embedded images —
  wherever the underlying real data exists to support that.

### Developing
- Data loading and transformation logic lives in its own module
  (`data_loader.py`), never inline in the page-rendering code (`app.py`).
- All expensive loads (file reads, CSV parses) are cached
  (`st.cache_data`) so interacting with the app (changing the split
  selector, etc.) doesn't re-read files from disk.
- No fabricated data under any circumstance — if a chart's underlying data
  doesn't exist, the chart is either omitted, replaced with a summary table,
  or the gap is stated explicitly in-app rather than papered over.

### Deploying
- Must run with `streamlit run app.py` from a plain `pip install -r
  requirements.txt` environment — no MATLAB, no GPU, no external services.
- Target host: Streamlit Community Cloud, deployed straight from the
  project's own repo (see `02_Technical_Requirements.md`).

## 4. User Experience

**Audience:** mixed, by explicit prior decision (Phase 1 of the original, unpublished handoff plan).
A recruiter skimming the Overview tab should get the headline numbers and a
plain-language paragraph in under 30 seconds. Someone who wants to dig in —
another researcher, a technical interviewer — should be able to switch
splits, hover any chart for exact values, and read the full technical
narrative in "How It Works," without leaving the app.

**Navigation:** a single page, six tabs (`st.tabs`), left to right in
narrative order: Overview → Results Explorer → HPO Comparison → Bias
Robustness → How It Works → About. See `03_App_Flow.md` for the exact
behavior of every control.

**Tone:** plain-language captions next to every chart, real numbers and
methodology available to anyone who reads further — never one at the expense
of the other.

## 5. Success Metrics

There is no live user base or analytics for this app (it's a portfolio
artifact, not a product), so success is qualitative and reviewer-facing:

- The project owner would be comfortable sending the deployed link to a
  recruiter or attaching it to conference materials (ICEP2026) without
  caveats.
- A reader with no prior context on the project can explain, after using the
  app, why "hand-tuning beat automated search" and what the bias-ablation
  trade-off actually is — not just that the model is "accurate."
- Every number in the app matches its source (README table or saved
  evaluation output) exactly — this was verified during the build (see
  `06_Implementation_Plan.md`, Step 3) and should be re-verified after any
  future change to the underlying data.
- The app loads and every tab renders without error on a fresh
  `pip install -r requirements.txt` — the actual acceptance test used during
  the build (see Implementation Plan).
