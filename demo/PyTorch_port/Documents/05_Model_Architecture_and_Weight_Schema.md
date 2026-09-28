# Model Architecture & Weight Schema — HA-PIT PyTorch Port

*(Equivalent of "Backend Schema." Reframed the same way the parent project's
`05_Data_Architecture.md` was: there are no database tables here, but there
is a real schema — the network's layer graph and its weight tensors — and
that's what this document specifies, at the level of detail an
implementation can be built against directly.)*

## Full layer graph (read directly from `src/matlab/HAPIT8v6.m`, lines 347–420)

| # | Layer name | Type | Config |
|---|---|---|---|
| 1 | `input` | `sequenceInputLayer` | 5 channels |
| 2 | `conv_k3` | `convolution1dLayer` | kernel 3, 48 filters, same padding |
| 3 | `conv_k5` | `convolution1dLayer` | kernel 5, 48 filters, same padding |
| 4 | `conv_k9` | `convolution1dLayer` | kernel 9, 48 filters, same padding |
| 5 | `concat_cnn` | `concatenationLayer` | concatenates conv_k3/k5/k9 along channel dim → 144 channels |
| 6 | `ln_cnn` | `layerNormalizationLayer` | |
| 7 | `relu_cnn` | `reluLayer` | |
| 8 | `drop_cnn` | `dropoutLayer` | rate 0.05 |
| 9 | `proj` | `fullyConnectedLayer` | → 64 (attentionDim) |
| 10 | `pos_enc` | `PositionalEncodingLayer` (custom) | 64 channels, scale 0.1 |
| 11 | `self_att_1` | `selfAttentionLayer` | 4 heads, 64 dim |
| 12 | `drop_att_1` | `dropoutLayer` | rate 0.05 |
| 13 | `add_att_1` | `additionLayer` | `drop_att_1 + pos_enc` (residual) |
| 14 | `ln_att_1` | `layerNormalizationLayer` | |
| 15 | `ffn1_1` | `fullyConnectedLayer` | → 128 (attentionDim × ffnExpand=2) |
| 16 | `relu_ffn_1` | `reluLayer` | |
| 17 | `drop_ffn_1` | `dropoutLayer` | rate 0.05 |
| 18 | `ffn2_1` | `fullyConnectedLayer` | → 64 |
| 19 | `add_ffn_1` | `additionLayer` | `ffn2_1 + ln_att_1` (residual) |
| 20 | `ln_ffn_1` | `layerNormalizationLayer` | end of transformer block 1 |
| 21 | `self_att_2` | `selfAttentionLayer` | 4 heads, 64 dim |
| 22 | `drop_att_2` | `dropoutLayer` | rate 0.05 |
| 23 | `add_att_2` | `additionLayer` | `drop_att_2 + ln_ffn_1` (residual) |
| 24 | `ln_att_2` | `layerNormalizationLayer` | |
| 25 | `ffn1_2` | `fullyConnectedLayer` | → 128 |
| 26 | `relu_ffn_2` | `reluLayer` | |
| 27 | `drop_ffn_2` | `dropoutLayer` | rate 0.05 |
| 28 | `ffn2_2` | `fullyConnectedLayer` | → 64 |
| 29 | `add_ffn_2` | `additionLayer` | `ffn2_2 + ln_att_2` (residual) |
| 30 | `ln_ffn_2` | `layerNormalizationLayer` | end of transformer block 2 |
| 31 | `net_out` | `fullyConnectedLayer` | → 2 (SOC correction, OCV correction) |

**Pattern confirmed from the `connectLayers` calls**: both transformer blocks
use a **post-norm residual** pattern — `add` happens on the *pre-normalization*
branch output added to the *previous normalized* state, then the sum is
normalized (`x → sublayer(x) → add(sublayer(x), x) → norm`) — not the
pre-norm pattern (`norm → sublayer → add`) that's become more common in
newer transformer implementations. This distinction matters for the PyTorch
port: naively using a modern pre-norm transformer block template would
silently produce a structurally different (and non-parity) model.

**Total learnable parameters: 80,866** (from the source README, already
cross-verified elsewhere in this project) — not yet re-derived layer-by-layer
here, since the exact `selfAttentionLayer` internal parameterization needs
confirming empirically once real weights are exported (see Implementation
Plan, Step 1) rather than assumed from documentation.

## Custom layer: `PositionalEncodingLayer` (read in full, no parameters to convert)

Deterministic — **no learnable weights**, so this is a pure reimplementation
task, not a weight-conversion task. Formula, confirmed from source:

```
for pos in 0..seqLen-1:
  for i in 0, 2, 4, ... < NumChannels:
    exponent = i / NumChannels
    pos_enc[i, pos]   = sin(pos / 10000^exponent)
    pos_enc[i+1, pos] = cos(pos / 10000^exponent)   # if i+1 < NumChannels

pos_enc *= ScaleFactor   # 0.1, from cfg.pEncScale
output = input + pos_enc   # added, not concatenated; broadcast over batch
```

This is the standard sinusoidal positional encoding from the original
Transformer paper, with one project-specific addition: a `ScaleFactor` (0.1)
applied before adding, "to prevent drowning out the CNN data" per the
source's own comment. **Exactly reproducible in PyTorch with no ambiguity**
— this is the lowest-risk component of the entire port.

## Input schema (read from `src/matlab/prepareBatteryDataV9.m`)

5 input channels per timestep, **not** the 4 named in the original README's
architecture diagram caption (`[Voltage, Current, Temp, SOH]`) — the real
schema, confirmed from source:

| Channel | Meaning | Normalization (from checkpoint's `train_scalers`) |
|---|---|---|
| 1 | Voltage (measured) | joint z-score: `(x - mu3[0]) / sig3[0]` |
| 2 | Current (measured) | joint z-score: `(x - mu3[1]) / sig3[1]` |
| 3 | Temperature (measured) | joint z-score: `(x - mu3[2]) / sig3[2]` |
| 4 | `C_dyn` — dynamic capacity estimate (the ageing/SOH signal) | `(x - mu_c) / sig_c` |
| 5 | `SOC_ocv_instant` — instantaneous OCV-derived SOC estimate | `(x - mu_si) / sig_si` |

`mu3`/`sig3`/`mu_c`/`sig_c`/`mu_si`/`sig_si` are **real, already-extracted
values** (see the parent project's `05_Data_Architecture.md` — they were
pulled from `train_scalers` during the original demo build). Channel 4 and 5
are *derived* features (computed from raw V/I/T via the Coulomb-counting and
OCV logic in `prepareBatteryDataV9.m`), not raw sensor readings — the port's
preprocessing code has to replicate that derivation, not just the network.

## Output schema and post-processing (read from `src/matlab/HA_PIT_test_v8.m`)

The network itself outputs 2 raw values per timestep (`net_out`, channels
`[SOC_correction_raw, OCV_correction_raw]`). Turning that into an actual SOC
trajectory is a **separate, deterministic post-processing step**, confirmed
from source — this logic must be ported too, not just the network:

```
soc_correction = net_output[0] * cfg.dSOC_scale       # scale: 0.001
# (net_output[1] * cfg.ocv_corr_scale is used during training for the ECM
#  loss term only — not needed for a pure SOC-output inference path)

SOC[0] = <anchor: true initial SOC, or a Coulomb-derived start point>
for i in 1..N-1:
    SOC[i] = SOC[i-1] + (I[i] * cfg.dt) / C_dyn[i] + soc_correction[i]
    if gate[i]:   # settled-rest re-anchoring condition, see below
        SOC[i] = <re-anchor to instantaneous OCV-derived SOC>

SOC = clip(SOC, 0, 1) * 100   # final output, percent
```

The **gating condition** (`cont.gate`, per the parent project's README
research) fires only when: current < 0.05 A, rest duration ≥ 120 s, ≥ 30 s
past rest start, and trailing voltage slope < 2e-4 V/s. This is real
decision logic the port must reproduce exactly — an approximate version
would change *when* re-anchoring happens and directly affect parity.

## Weight-mapping table (structure only — values pending Step 1 export)

| MATLAB layer | Learnable parameters (MATLAB) | Target PyTorch module | Notes |
|---|---|---|---|
| `conv_k3`/`k5`/`k9` | Weights + bias per filter | `nn.Conv1d(5, 48, kernel_size=k, padding='same')` × 3 | MATLAB conv weight layout vs. PyTorch's — axis order must be verified, not assumed, during Step 3 |
| `ln_cnn`, `ln_att_1/2`, `ln_ffn_1/2` | Scale (γ) + offset (β) | `nn.LayerNorm(...)` × 5 | Straightforward — both frameworks use the same γ/β convention |
| `proj` | Weight + bias | `nn.Linear(144, 64)` | |
| `self_att_1`/`self_att_2` | In-projection (Q/K/V) + out-projection weights & biases | `nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=...)` × 2 | **Highest-risk mapping in this table** — needs empirical shape verification, per the TRD |
| `ffn1_1`/`ffn2_1`/`ffn1_2`/`ffn2_2` | Weight + bias | `nn.Linear` × 4 | |
| `net_out` | Weight + bias | `nn.Linear(64, 2)` | |

This table's right two columns are the actual implementation task; this
document stops at "what has to map to what," not "here is the converted
code" — per the instruction to plan fully before any port code is written.
