# Implementation Plan — HA-PIT PyTorch Port

Nothing in this plan has been executed yet — per instruction, this document
set is the deliverable for this round; implementation starts only after
review and explicit go-ahead.

## Step 0 — Investigation already done (this round)

Before writing this plan, the actual source was read directly rather than
assumed, to avoid planning against a guessed architecture:

- Full layer graph and connection pattern confirmed from
  `src/matlab/HAPIT8v6.m` (lines 347–420).
- Custom `PositionalEncodingLayer.m` read in full — confirmed parameter-free,
  standard sinusoidal formula.
- Real 5-feature input schema confirmed from `prepareBatteryDataV9.m`
  (corrects the README diagram's simplified 4-feature caption).
- Post-processing/anchoring logic confirmed from `HA_PIT_test_v8.m`.
- **Checkpoint format constraint discovered**: `hapit_v8p6_best_checkpoint.mat`'s
  actual network is a MATLAB MCOS object (`dlnetwork`), not readable via
  `scipy.io.loadmat` — confirmed by direct inspection, not assumed.
- **MATLAB R2023a confirmed installed on this machine** — the weight-export
  step below is actually executable here, not a blocked dependency.

## Step 1 — Export weights from MATLAB (first real implementation step)

A short MATLAB script (run once, by hand, locally):
1. Load `hapit_v8p6_best_checkpoint.mat`.
2. Access the `dlnetwork` object's `.Learnables` table (columns: `Layer`,
   `Parameter`, `Value`).
3. For each row, extract the value (a `dlarray`) to a plain numeric array
   (`extractdata`) and write it out — proposed: one `.mat` file per layer,
   containing only plain double arrays, so the existing `scipy.io.loadmat`
   pattern already used elsewhere in this project works unchanged for
   reading them back.
4. **Sanity check before moving on**: confirm the exported parameter shapes
   match what's expected from the layer graph in
   `05_Model_Architecture_and_Weight_Schema.md` (e.g., `proj`'s weight should
   be roughly 64×144). A mismatch here means the export script has a bug —
   catching it now is far cheaper than catching it after the PyTorch side is
   built.

## Step 2 — Build the PyTorch architecture (no weights yet)

Write the `nn.Module` matching the confirmed layer graph exactly, including
the **post-norm residual pattern** (not pre-norm) documented in Step 0's
findings. Initialize with random weights at this stage — the goal here is
only to confirm the module's forward pass runs and produces the right output
*shape* (`[batch, 2, seqLen]`), before any real weight is involved.

## Step 3 — Map and load the exported weights

Assign each exported MATLAB parameter to its corresponding PyTorch parameter,
per the mapping table in `05_Model_Architecture_and_Weight_Schema.md`. Two
specific risk points to check explicitly, not assume:
- **Convolution weight layout** — MATLAB and PyTorch may store 1-D
  convolution kernels with different axis ordering; verify by shape and, if
  ambiguous, by testing a single known input against a single conv layer in
  isolation before trusting the full model.
- **`selfAttentionLayer` → `nn.MultiheadAttention` parameter layout** — the
  two frameworks' internal Q/K/V and output-projection weight organization
  needs to be confirmed empirically (e.g., checking `nn.MultiheadAttention`'s
  `in_proj_weight` shape against the exported attention layer's `Learnables`
  rows) rather than assumed identical because both are "standard multi-head
  attention." This is the single highest-risk mapping in the whole port.

## Step 4 — Parity verification (hard gate before anything else proceeds)

1. Take one real input window from B0018 (already available — the same data
   used throughout the existing Streamlit demo).
2. Run it through the original MATLAB model (`HA_PIT_test_v8.m`) and record
   the raw 2-channel network output.
3. Run the identical input through the ported PyTorch model.
4. Compare the two raw outputs directly, before any post-processing —
   isolates whether the *network* matches, independent of the anchoring
   logic ported in Step 5.
5. Only once Step 4 passes: run the full pipeline (network + Coulomb-
   counting + gated re-anchoring) on the full B0018 continuous-reconstruction
   evaluation and compare the resulting RMSE against the original 0.4129%,
   against the tolerance set in `01_Product_Requirements.md`.
6. **If parity fails**, the fix is to narrow down which of Steps 2/3
   introduced the divergence — starting from the raw-output comparison in
   Step 4.4, which isolates the network from the post-processing — not to
   adjust tolerances until the numbers happen to match.

## Step 5 — Port the post-processing pipeline

Reimplement the Coulomb-counting recursion and the gated rest-anchoring
condition from `HA_PIT_test_v8.m` in Python, operating on the PyTorch model's
raw output. This is deterministic control-flow logic, not a machine-learning
component — lower risk than Steps 2–3, but still must be exact (see the gate
condition's four exact thresholds in the architecture document).

## Step 6 — Integrate into the Streamlit app

1. Add `model.py` (or similar) to `streamlit demo/`, alongside `data_loader.py`.
2. Add a `load_ported_model()` function, `st.cache_resource`-cached.
3. Add the "Try It" tab to `app.py`, per the interaction design in
   `03_Port_and_Integration_Flow.md` (pending the open questions there being
   resolved during review).
4. Extend `04_UIUX_Design_Brief.md`'s components if the file-upload path is
   chosen.

## Step 7 — Test in a real browser

Same bar as the original demo build: launch the app, exercise the new tab
end to end, confirm no console/server errors, before calling this done.

## Step 8 — Update documentation

Update the parent project's `Documents/02_Technical_Requirements.md` (the
"explicitly out of scope: live model inference" section is no longer
accurate once this ships) and `06_Implementation_Plan.md`'s "recommended next
steps" list.

## Risks carried forward explicitly (not hidden)

- **Parity may not be achievable to the proposed tolerance on the first
  attempt.** The two highest-risk mappings (convolution layout,
  attention-layer parameter layout) are flagged above precisely because
  they're the most likely sources of a first-pass mismatch. If Step 4 fails
  repeatedly, the right escalation is deeper empirical layer-by-layer
  comparison (feed the same input through each matched pair of layers in
  isolation, not just the full model end-to-end), not lowering the tolerance
  to make a wrong port look right.
- **The exact `selfAttentionLayer` parameterization is not fully confirmed
  from documentation alone** — Step 1's export and Step 3's shape-matching
  are what actually resolve this, not this planning document.
