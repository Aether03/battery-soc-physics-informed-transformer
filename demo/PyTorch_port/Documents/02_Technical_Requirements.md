# Technical Requirements Document — HA-PIT PyTorch Port

No technical guesses below where it matters most (the architecture) — the
layer graph, custom layer formula, and I/O schema in this document were read
directly out of the source repo's `.m` files, not inferred from the README.
One real open item is flagged explicitly where the checkpoint's actual weight
values are concerned (see "Checkpoint format problem" below) — that one
genuinely can't be resolved without running MATLAB once, which this document
confirms is possible on this machine.

## Source of truth

| What | File | What was confirmed |
|---|---|---|
| Layer graph (17 layers, exact connections) | `src/matlab/HAPIT8v6.m`, lines 347–420 | Read directly — see `05_Model_Architecture_and_Weight_Schema.md` |
| Custom positional encoding | `src/matlab/PositionalEncodingLayer.m` | Full file read — standard sinusoidal formula, scaled, added (not concatenated) |
| Input feature schema | `src/matlab/prepareBatteryDataV9.m` | 5 features: `[V, I, T, C_dyn, SOC_ocv_instant]`, confirmed via source comments and code |
| Inference/anchoring logic | `src/matlab/HA_PIT_test_v8.m` | Confirmed: net outputs a per-step SOC correction (channel 1) scaled by `dSOC_scale`, integrated via a Coulomb-counting recursion with gated rest-anchoring |
| Hyperparameters | `models/hapit_v8p6_best_checkpoint.mat` → `cfg` struct | Full config read directly (see architecture doc) |
| Normalization constants | same checkpoint → `train_scalers` struct | `mu3`/`sig3` (3,), `mu_c`, `sig_c`, `mu_si`, `sig_si` — all real values, already extracted |

## Target stack

| Tool | Role |
|---|---|
| Python 3.10 (already the project's runtime) | No change from the existing Streamlit app |
| **PyTorch** (CPU-only build) | The ported model's runtime. CPU-only because the original trained in 1h38m on a laptop CPU with no GPU — inference is far cheaper than training, so there's no reason to require a GPU for a single forward pass, and Streamlit Community Cloud has none anyway. |
| `torch.nn.MultiheadAttention` | PyTorch's built-in equivalent of MATLAB's `selfAttentionLayer` — both implement standard scaled dot-product multi-head self-attention with learnable in/out projections. **To be verified empirically during implementation** (matching parameter layout is a real risk — see Implementation Plan Step 3), not assumed identical by name alone. |
| `scipy.io` (already a project dependency, build-time only) | Reading the `cfg`/`train_scalers` structs (already proven to work — these are plain structs, unlike the network weights themselves) |
| MATLAB R2023a (confirmed installed on this machine, Deep Learning Toolbox required) | **One-time, local-only** use: exporting the trained `dlnetwork`'s learnable weights to a plain, Python-readable format. Not a runtime dependency of the shipped app. |

## Checkpoint format problem (confirmed, not assumed)

`models/hapit_v8p6_best_checkpoint.mat` was inspected directly with
`scipy.io.loadmat`. It contains `cfg` and `train_scalers` as plain structs
(both fully readable, values already extracted) — but the actual trained
network is stored as a MATLAB **MCOS object** (`dlnetwork`/`dlarray`), which
`scipy.io.loadmat` cannot deserialize; it surfaces only as an opaque,
unusable blob. **The learnable weights themselves are not accessible from
Python without an intermediate MATLAB-side export step.**

This is a confirmed constraint, not a guess, and it's the single most
important technical fact this document has to convey: **the port cannot
start with a Python script reading the checkpoint.** It has to start with a
short MATLAB script (run once, locally, by hand) that loads the `dlnetwork`,
walks its `Learnables` table (MATLAB's own name for the per-layer weight/bias
tensors), and writes each tensor out to a format `scipy`/`numpy` can read
(e.g., one `.mat` file per layer containing only plain numeric arrays, or a
flat set of `.csv`/`.npy` files). This step is scoped in
`06_Implementation_Plan.md`, Step 1.

## Frontend integration

- New tab added to the existing `app.py`'s `st.tabs(...)` call — "Try It,"
  appended after "About" or wherever the parent project's owner prefers in
  final review.
- Reuses the existing `data_loader.py` pattern: a new loader function
  (proposed `load_ported_model()`, cached with `st.cache_resource` — the
  correct Streamlit cache decorator for a loaded model object, distinct from
  `st.cache_data` used for plain data) that loads the converted weights once
  per server process.
- Chart rendering reuses the existing Plotly + validated-palette conventions
  from the parent app's `04_UIUX_Design_Brief.md` — no new visual system.

## Deployment

No change to the parent app's deployment target (Streamlit Community Cloud).
The converted weights (expected to be well under 1 MB, given the original
checkpoint is 314 KB and PyTorch's storage format has comparable density)
ship as a checked-in file alongside the existing `data/` folder — same
"self-contained repo, no external dependency" principle as the rest of the
app.

## Explicitly out of scope

- **GPU inference.** Not needed; see Target stack above.
- **Training in PyTorch.** This is a port of an already-trained model, not a
  retraining effort. The five-term physics-informed loss function does not
  need porting — it only matters during training, not for a forward pass
  (this was already noted as likely in the original project handoff, and
  the architecture read above confirms the loss terms aren't part of the
  `dlnetwork`'s layer graph at all — they're computed separately during
  training).
- **Retraining or fine-tuning the ported model.** The port's job is to
  reproduce the existing trained model's behavior exactly, not to improve it.
