# Port & Integration Flow — HA-PIT PyTorch Port

*(Equivalent of the parent project's "App Flow" document. Reframed because
this deliverable has two distinct flows that both need documenting: the
one-time **conversion pipeline** that produces the ported model, and the
**runtime flow** inside the Streamlit app that uses it.)*

## Flow 1 — One-time conversion pipeline (build-time, run once by hand)

This never runs inside the deployed app. It runs once, locally, on this
machine, and its *output* (a converted weights file) is what gets shipped.

```
MATLAB (R2023a, local, one-time)
  1. Load hapit_v8p6_best_checkpoint.mat's dlnetwork object
  2. Walk dlnet.Learnables (layer name, parameter name, value, per row)
  3. Export every row to a plain, non-MCOS format (e.g. one .mat per layer,
     or flattened .csv/.npy files) — this is the step that gets past the
     "scipy can't read dlnetwork objects" problem documented in the TRD
        │
        ▼
Python / PyTorch (this environment)
  4. Load the exported plain-format weights
  5. Build the PyTorch nn.Module matching the confirmed layer graph
     (see 05_Model_Architecture_and_Weight_Schema.md)
  6. Assign each exported weight to its corresponding PyTorch parameter —
     this is the highest-risk step: shapes and axis conventions between
     MATLAB's Deep Learning Toolbox and PyTorch are not always identical
     (e.g. MATLAB convolution weight layout vs. PyTorch's), so this step
     needs explicit shape assertions, not silent reshaping
        │
        ▼
  7. Run the SAME input (a real B0018 window) through both the original
     MATLAB model and the new PyTorch model
  8. Compare outputs — this is the parity check, and it's a hard gate:
     nothing proceeds to Flow 2 until this passes the tolerance set in
     01_Product_Requirements.md
        │
        ▼
  9. Save the verified PyTorch model's weights (state_dict) as the artifact
     that ships with the app
```

## Flow 2 — Runtime flow inside the deployed Streamlit app

This is the only part end users ever interact with; everything in Flow 1 has
already happened before this code runs.

1. **User opens the "Try It" tab.**
2. **Input selection** — proposed simplest version: a dropdown to pick one of
   the existing held-out splits' cells (reusing data already bundled in
   `data/predictions.npz`'s source arrays, or a small held-back raw sample),
   rather than asking a visitor to construct a valid 5-channel, 480-step
   input from scratch. *(Open question for review — see below.)*
3. **On selection**, the app:
   a. Loads the cached ported model (`st.cache_resource`, loaded once)
   b. Normalizes the selected input using the same `train_scalers` constants
      the original model was trained with (already extracted, real values)
   c. Runs the forward pass through the PyTorch model → raw SOC-correction
      output
   d. Applies the same Coulomb-counting + gated-anchoring post-processing
      the original `HA_PIT_test_v8.m` uses, to turn the raw correction into
      an actual SOC trajectory
4. **Display** — a chart in the same visual language as the Results
   Explorer's SOC-trajectory chart: predicted SOC (and true SOC, if the
   selected input has known ground truth) over the sequence.
5. **A visible disclosure line** stating this is the ported model, with a
   link to how it was verified (the parity numbers from Flow 1, Step 8) —
   so a technical visitor can check the claim rather than take it on faith.

## Decisions (resolved during review, 2026-09-29)

1. **Input source: existing held-out data only, no file upload for now.**
   Deliberately deferred — the ported model isn't yet validated as
   generalizing beyond the NASA cycling data it was trained/tested on (the
   README's own UDDS stress test shows an order-of-magnitude accuracy drop
   out of distribution), so inviting arbitrary visitor-supplied input would
   risk showcasing the model's worst case rather than its real one. Revisit
   once there's a basis for trusting out-of-distribution behavior.
2. **Show ground truth alongside the prediction.** Confirmed — both series
   render on the same chart, reusing the existing true/predicted color
   convention (blue/orange) from Results Explorer.
3. **Tab placement: after "About."** Confirmed — appended as the seventh and
   final tab, preserving the existing Overview → ... → About narrative order.
