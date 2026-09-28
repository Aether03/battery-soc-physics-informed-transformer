# Product Requirements Document — HA-PIT PyTorch Port

## 1. Overview

Port the trained HA-PIT model (currently a MATLAB `dlnetwork`, defined in
`src/matlab/HAPIT8v6.m`) to PyTorch, and integrate it into the existing
Streamlit demo as a new **"Try It"** tab — a live inference feature where a
visitor supplies input and gets a real SOC prediction from the actual trained
model, not a lookup of pre-computed results.

This is Phase 6 of the original demo plan (an internal, unpublished handoff note), picked up now
because the decision it was gated on — "does this need to be a public URL" —
is resolved: yes, a personal portfolio site will link to or embed this demo,
which makes MATLAB-based inference (license-locked, can't be publicly hosted)
a non-starter. See `../Documents/06_Implementation_Plan.md` in the parent
project and the chat decision that preceded this document for the full
MATLAB-bridge-vs-PyTorch-port comparison.

## 2. Goals and Objectives

1. **Numerical parity with the original, not "close enough."** The port's
   output on the held-out test cell (B0018) must match the original MATLAB
   model's output to a tight, explicitly stated tolerance — see Success
   Metrics. A port that "mostly works" would quietly misrepresent the
   project's actual published results.
2. **A working "Try It" feature**, not just a converted model file sitting
   unused. The port only delivers value once it's wired into the Streamlit
   app behind a real UI.
3. **A secondary, legitimate portfolio artifact in its own right**: "ported a
   custom transformer + physics-informed post-processing pipeline from
   MATLAB to PyTorch, with verified numerical parity" is worth being able to
   describe and defend in an interview — the verification process is part of
   the deliverable, not just the port itself.

## 3. Requirements

### Designing
- The "Try It" tab must fit the existing app's established visual identity
  (see the parent project's `04_UIUX_Design_Brief.md`) — no separate look.
- Inputs must be something a visitor can plausibly supply without domain
  expertise (e.g., picking from the existing held-out test data, or a
  provided sample file) — not a raw 5-channel tensor paste box.

### Developing
- The PyTorch model definition lives in its own module, separate from any
  Streamlit UI code — same separation-of-concerns principle already applied
  to `data_loader.py` in the parent app.
- Every learnable weight is converted from the original checkpoint — no
  re-initialization, no "close enough" approximation of any layer.
- A parity-verification script/test is a required deliverable, not optional
  polish — see `06_Implementation_Plan.md`.

### Deploying
- The ported model and its converted weights must be small and portable
  enough to ship inside the same Streamlit Community Cloud deployment as the
  rest of the app (see `02_Technical_Requirements.md` for the concrete size
  expectation).
- No MATLAB, no MATLAB Runtime, and no paid license may be a runtime
  dependency of the deployed app — MATLAB is needed only once, locally,
  during the one-time weight-export step (see below).

## 4. User Experience

A visitor picks or supplies an input sequence (the simplest version: choose
one of the existing held-out cells' data, already bundled with the app) and
sees a predicted SOC trajectory rendered the same way the existing Results
Explorer renders true-vs-predicted charts — reusing established visual
language rather than inventing a new one. See
`03_Port_and_Integration_Flow.md` for the exact interaction design once it's
settled.

## 5. Success Metrics

- **Primary, non-negotiable:** the ported model's continuous-reconstruction
  RMSE on B0018, computed exactly the same way the original evaluation
  computes it, matches the original 0.4129% figure within a stated tolerance
  (proposed: within 0.01 percentage points, i.e. 0.403%–0.423%; this number
  should be confirmed, not assumed, once real per-timestep parity numbers are
  in hand — see Implementation Plan Step 4).
- **Secondary:** per-timestep output values (not just the aggregate RMSE)
  match the MATLAB original within a tight absolute tolerance on a shared
  test input — an aggregate metric alone could hide compensating errors.
- The "Try It" tab is usable end-to-end in the deployed app with no MATLAB
  dependency at runtime.
