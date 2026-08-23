# Technical reference

How the model works and why each piece is shaped the way it is. Written to be read alongside the source. File and line references point at the code that implements each claim.

## Contents

1. [The problem, and the one decision everything follows from](#1-the-problem)
2. [Data pipeline](#2-data-pipeline)
3. [Physics model](#3-physics-model)
4. [Network](#4-network)
5. [Loss](#5-loss)
6. [Training](#6-training)
7. [Evaluation](#7-evaluation)
8. [Version history](#8-version-history)

---

## 1. The problem

Estimate a cell's state of charge, 0-100%, at every second, from three sensor channels: terminal voltage, current, temperature.

The difficulty is not the sensors. It's that terminal voltage barely moves over the flat region of the open-circuit-voltage curve, roughly 30-80% SOC on this chemistry. A 240-second window there contains almost no information that pins down absolute SOC. No amount of architecture tuning breaks that floor, because the information is not in the window.

**So the model doesn't predict absolute SOC.** It predicts per-step *increments*, and Coulomb counting already gives an excellent increment: `I·Δt/C` is accurate and window-independent. The network's job reduces to correcting that integration using the voltage channel, which is structurally a learned Kalman gain.

Since V8.4 this is strictly residual. The Coulomb term is computed directly and never learned:

```
ΔSOC[k] = I_b[k]·Δt / C_dyn[k]  +  c_k
          └──── exact, fixed ────┘   └ learned ┘
```

`src/matlab/HAPIT8v6.m`

The reason to bake it in rather than let the network reproduce it: MSE training systematically attenuates a learned mapping under noise, the standard shrinkage effect. Per step that attenuation is negligible. Over an unanchored 5,600-step startup segment it compounds into several percent of drift. With the Coulomb term outside the learned path there is nothing to attenuate. A zero-correction network reproduces the labels to about 1e-9, which was verified numerically before the design was committed to.

The correction is scaled by 1e-3 at the output, encoding the prior that it should be small.

---

## 2. Data pipeline

`src/matlab/prepareBatteryDataV9.m` is one function, called by every trainer and every search harness. No duplicated data logic exists anywhere in the system, which matters because the evaluation step has to reproduce the labelling rule *exactly*; one implementation makes that guaranteed rather than aspirational.

### Conditioning

1. De-duplicate timestamps.
2. **Hard-clip voltage to [2.0, 4.25] V** before anything else. Raw readings contain physically impossible glitches. Values from 0.003 V to 4.98 V appear in the files, and they would corrupt OCV inversion and poison the filter.
3. **Resample to a uniform 1 Hz grid.** Raw timestamps are irregular, gaps averaging ~4.7 s and ranging 1-21 s. On a uniform grid every integration step with Δt = 1 s is exact, and a fixed window length corresponds to a fixed number of real seconds.
4. **Zero-phase 4th-order Butterworth low-pass** (Wn = 0.20, via `filtfilt`) on voltage and current. Forward-backward filtering introduces no time shift; a causal filter would smear features later than they occurred and misalign input against label.

### Capacity is not a constant

Cells fade, so the Coulomb denominator can't be the nameplate value. The pipeline finds contiguous active-current runs (|I| > 0.1 A), sums amp-seconds per run, and treats any run over 1000 A·s as a fresh capacity estimate, carrying it forward between updates.

Observed ranges: B0005 [2661, 6703], B0006 [2935, 7375], B0007 [2753, 6908], B0018 [2679, 6726] A·s, against a nominal 6664 A·s. That spread is the ageing this model is built to track.

### Relaxation-gated anchoring

The single most consequential fix in the project's history.

Immediately after current drops to zero, terminal voltage is still relaxing toward open circuit. Thirty seconds into a rest it can still be off by several hundred millivolts, mean ~490 mV on one validation file. Anchoring the SOC estimate against that non-relaxed voltage doesn't correct error, it *injects* it. The symptom in earlier versions was a regression slope of 0.92, systematic compression across the whole range.

A rest step earns the right to re-anchor only if all four conditions hold:

| Condition | Threshold |
|---|---|
| Current is at rest | \|I\| < 0.05 A |
| Rest is sustained | run length ≥ 120 s |
| Past initial relaxation | ≥ 30 s into the rest |
| Voltage has settled | trailing 30-sample \|dV/dt\| < 2e-4 V/s |

The gate fires on 6.5-12.8% of steps depending on the file. The mask is computed **once** and returned as `cont.gate`, then shared by label construction, the drift loss, and evaluation reconstruction. Any independent recomputation could diverge and reintroduce exactly the mismatch the gate exists to eliminate.

### Ground truth

With α = 0.20 and gate g ∈ {0,1}:

```
SOC[i] = clip₀¹[ (1 − α·gᵢ)·(SOC[i−1] + I[i]·Δt / C_dyn[i]) + α·gᵢ·OCV⁻¹(V[i]) ]
SOC[1] = SOC₀
```

`prepareBatteryDataV9.m` lines 189-203.

Read that carefully, because it determines what the reported error means. Ground truth here is Coulomb counting with periodic voltage recalibration. That is standard practice for this dataset, which has no independent SOC sensor, but it means the evaluation metric is a reconstruction-fidelity measure, not a comparison against an external reference. See the README's *What these numbers are measured against*.

Initial SOC comes from a one-shot heuristic: average the first ten current readings, use the sign to pick the charge or discharge branch, invert that OCV polynomial at the IR-compensated starting voltage. A more elaborate rule based on the first sustained active phase was tried and reliably picked the *wrong* branch on these files, because their first sustained phase is a multi-thousand-second charge even though the cells start near-full. The heuristic was kept and the real fix applied downstream at the gate, where it can't go wrong regardless of what SOC₀ says.

### Inputs

Five channels, z-scored with scalers fitted once on the first training file and reused for validation and test, so there is no normalisation mismatch between splits.

| # | Channel |
|---|---|
| 1 | Voltage |
| 2 | Current |
| 3 | Temperature |
| 4 | Capacity-derived state-of-health signal |
| 5 | `SOC_OCV_INSTANT`, per-timestep IR-compensated OCV-inverted SOC |

Channel 5 is the fusion input: `SOC_ocv[i] = OCV⁻¹(V[i] − I[i]·R₀)`, inverted through the charge, discharge or equilibrium curve depending on current sign. It hands the network a direct voltage-based SOC reading to weigh against its Coulomb integration, rather than making it rediscover the OCV-SOC relationship implicitly from raw voltage.

Windows are 480 steps at 50% overlap. Window weights are SOC-stratified: binned into ten buckets by mean SOC, with weight inversely proportional to bucket population, so the loss isn't dominated by whichever SOC range the cycles happen to dwell in. Counts: 566 training windows, 283 validation, 327 test.

---

## 3. Physics model

Everything here is precomputed per window. No learnables.

### OCV-SOC curves

Two 5th-order polynomials fitted from B0005 and B0006, one per branch, blended by a sigmoid on current so the switch at I = 0 stays differentiable:

```
w_chg    = 1 / (1 + exp(−10·I))
OCV(s,I) = w_chg·OCV_chg(s) + (1 − w_chg)·OCV_dchg(s)
```

### Hysteresis

A Plett-style one-state hysteresis voltage, integrated per window from raw current:

```
hₖ = hₖ₋₁·exp(−|Iₖ|·γ·Δt) + H_max·sign(Iₖ)·(1 − exp(−|Iₖ|·γ·Δt))
```

H_max = 0.015 V, γ = 1e-4, h₀ = 0. Observed range on training data: [−0.0014, 0.0011] V. The state decays toward +H_max on charge and −H_max on discharge at a rate set by charge throughput.

### Ohmic resistance, temperature- and age-dependent

```
R₀(T, C) = R_ref · exp[(Ea/R)·(1/T_K − 1/T_ref)] · (C_nom / C_dyn)^0.5
```

R_ref = 0.080 Ω (measured 0.078-0.103 Ω), Ea/R = 20000/8.314 K, T_ref = 297.15 K. Arrhenius in temperature, power law in capacity fade. Observed range with ageing: 0.051-0.126 Ω.

This is the second half of "ageing-aware". Capacity fade enters the Coulomb denominator; resistance growth enters the voltage model. Without both, the physics constraint would be badly wrong by the time the cell reaches its 1.4 Ah end-of-life threshold, and the loss term built on it would actively mislead training.

### Terminal-voltage reconstruction

```
V̂ = OCV(SOC_pred, I) + ΔOCV_net + h + I_b·R₀
```

`ΔOCV_net` is the network's second output scaled by 0.1 V. `I_b` is the *sensor* current, biased under augmentation, matching what a deployed estimator would actually compute from its own reading.

A transient mask down-weights this term during fast current steps, where a static equivalent circuit without RC dynamics is least trustworthy:

```
m = 1 / (1 + (|ΔI| / 0.5)²)
```

---

## 4. Network

80,866 learnable parameters.

```
Input [5, B, 480]
   ├─→ Conv1D k=3 ─┐
   ├─→ Conv1D k=5 ─┼─→ Concat → LayerNorm → ReLU → Dropout → FC(64) → PosEnc
   └─→ Conv1D k=9 ─┘
                          ↓
              Transformer block 1  (self-attention + residual → LayerNorm
                                    → FFN + residual → LayerNorm)
                          ↓
              Transformer block 2  (identical)
                          ↓
                    FC → 2 outputs
              [SOC correction, OCV correction]
```

**Three parallel kernel widths** rather than one: short, medium and longer local transients in current and voltage get captured simultaneously, without committing to a single temporal scale before attention sees the sequence.

**Two attention blocks, not more.** The correction signal is small by design. Two blocks are enough for the long-range cues that matter, an approaching rest or a capacity-fade pattern, without over-parameterising against two cells' worth of training data.

**Two output channels.** The first is the SOC correction. The second is an OCV correction in volts, used only inside the equivalent-circuit loss and never in the SOC reconstruction path. It gives the physics term a degree of freedom to absorb curve-fit error without letting that leak into the SOC estimate.

All standard Deep Learning Toolbox layers wired through `layerGraph`, except `PositionalEncodingLayer` (`src/matlab/PositionalEncodingLayer.m`), which is a `classdef` and must be on the path under exactly that name.

Baseline configuration: `numFilters=48`, `attentionDim=64`, `numHeads=4`, `ffnExpand=2`, `dropoutRate=0.05`, `seqLen=480`, `initLR=5e-3`, `warmupEpochs=20`, `cosineEpochs=150`, `miniBatchSize=2048`, `gradClip=1.0`.

---

## 5. Loss

Five terms, fixed weights. Learned homoscedastic weighting was tried and went numerically unstable, so fixed weights were adopted.

```
L = w_data·L_data + λ_Ah·L_Ah + λ_end·L_end + λ_drift·L_drift + w_ECM·L_ECM
```

| Term | Formula | Weight | Job |
|---|---|---:|---|
| `L_data` | mean(W·(SOC_pred − Y)²) | 1.0 | Primary signal, SOC-stratified |
| `L_Ah` | mean(c²) | 10.0 | Keep the correction small: stay near Coulomb counting unless voltage evidence demands otherwise |
| `L_end` | mean((SOC_pred[L] − Y[L])²) | 0.5 | Endpoint drift within a window, which per-step MSE cannot see |
| `L_drift` | mean_B[(Σₜ(1−gₜ)(cₜ + b·Δt/Cₜ))²] | 30.0 | Accumulated unanchored bias |
| `L_ECM` | mean(m·(V̂ − V)²) | 0.5 | Thevenin voltage consistency, transient-masked, frozen for 15 epochs |

### Why `L_drift` is shaped that way

It squares the per-window **sum**, not the mean of squares. That distinction is the whole point.

A per-step bias of 2.6e-5 is invisible to any mean-of-squares term, swamped by ordinary prediction noise. But summed across a 480-step window it becomes 480 × 2.6e-5 ≈ 1.25e-2, and squared, ≈ 1.6e-4, the same order as the data loss. The term makes a bias too small to see individually impossible to ignore in aggregate.

Gated steps are excluded from the sum because there the correction is legitimately learning the label's OCV anchor pull, and penalising it would fight the label.

The bias-aware form, `cₜ + b·Δt/Cₜ` rather than just `cₜ`, is what makes it compatible with sensor-bias augmentation. On non-gated steps the label increment is exactly `I·Δt/C` computed from clean current, so the correct target for total deviation is zero. Written this way, the term demands the correction *cancel* an injected bias. Written the old way, the drift loss and the augmentation objective would pull against each other.

### Curriculum

`w_ECM = 0` for the first 15 epochs, so the increment-correction pathway stabilises before the more indirect voltage-consistency constraint switches on. Training all five terms from epoch 1 was tried and was less stable.

### Sensor-bias augmentation

With probability 0.5 per training window, a constant bias b ~ U(−0.10, +0.10) A is added everywhere a real biased sensor would show up: the normalised input channel, the physics increment, and the IR term in the voltage reconstruction. Labels stay clean.

The data loss then demands `c ≈ −b·Δt/C`, conditioned on the mismatch between what the voltage says and what the current says. That's what teaches the correction channel to detect and counteract bias, and what makes the ablation in §7 a meaningful test rather than a formality.

One known approximation: input channel 5 is not re-biased. Its shift is second-order, about −b·R₀ ≈ 8 mV at b = 0.1 A.

---

## 6. Training

Adam with global-norm gradient clipping at 1.0.

**Learning rate**: linear warmup over 20 epochs, then cosine annealing to a *fixed horizon* of 150 epochs, holding at the 1e-5 floor afterwards. Decoupling the annealing horizon from the epoch budget is the V8.6 fix. With the schedule tied to a 400-epoch budget while patience-based stopping fired around epoch 115, the learning rate never dropped below 4.3e-3 and the continuous validation metric oscillated between 1% and 8.6% epoch to epoch instead of converging.

**Batch size 2048** exceeds the 566 training windows, so this is full-batch gradient descent, one iteration per epoch. That suits the drift objective specifically: the drift statistic is a small mean over windows, and small minibatches make it noisy. V8.5 used 64 and paid for it.

**Selection and early stopping** use the *continuous* validation metric, not the windowed one, with patience 80. After each epoch the already-computed validation corrections are stitched into a full continuous reconstruction, which takes under a second, and that number decides. The windowed metric is still logged as a diagnostic, because it is structurally blind to accumulated drift.

V8.6 run: best epoch 107 at 0.3695% continuous validation RMSE, early stop at 187, wall time 5,899.6 s.

---

## 7. Evaluation

### Continuous reconstruction: the deployment metric

Byte-identical to the label rule, with the network correction added:

```
SOC[i] = clip₀¹[ (1 − α·gᵢ)·(SOC[i−1] + I[i]·Δt/C[i] + cᵢ) + α·gᵢ·SOC_ocv[i] ]
```

Same `SOC₀`, same gate mask, same α as label construction, all read from the `cont` struct the pipeline returns, not recomputed. Corrections are predicted per window in batches of 64 to bound memory, then stitched onto the global timeline with later windows overwriting the overlapping half of earlier ones, so each timestep uses exactly one correction. Then a single forward integration across the whole sequence.

### Windowed-oracle: the diagnostic

Anchors each window at its true starting SOC and measures correction quality within the window only.

The two together localise error. A large continuous-vs-oracle gap means residual drift between anchors. A small gap means the error is intrinsic to the correction itself.

This pairing is not decoration. V8.4 scored 0.39% windowed while carrying 3.05% continuous error. The windowed metric alone would have declared it finished. Test-set numbers for V8.6: 0.4129% continuous against 0.0638% windowed, a ~0.35 pp gap saying the small remaining error is mostly accumulated drift rather than per-step correction noise.

### Per-segment breakdown

Test set split into ten equal time chunks (%RMSE): 0.0327, 0.0953, 0.1262, 0.1812, 0.6716, 0.2284, 0.5817, 0.0003, 0.3682, 0.8179.

Error concentrates in chunks 5, 7 and 10 rather than spreading uniformly. These are specific late-life segments, consistent with the drift diagnosis rather than with uniform noise.

### Current-bias ablation

Documented in the README. The honest summary: V8.6 reached near-parity with pure Coulomb counting under injected bias but did not beat it, and the configuration from the hyperparameter search did beat it at a cost in clean accuracy.

---

## 8. Version history

| Version | Key change | Continuous RMSE, train/val/test |
|---|---|---|
| ≤ V8.1 | Increment reformulation; CNN→transformer front end; ECM + hysteresis physics | superseded |
| V8.3 | Full-increment learning, every-rest anchoring, evaluation mismatched to labels | ~7-8% test, startup spikes 24-54% |
| V8.4 | Three structural fixes: label-identical reconstruction, relaxation-gated anchoring, residual increments + `L_end` | 2.16 / 3.05 / 1.75 |
| V8.5 | Drift loss (λ=10) + selection on the continuous metric | 1.1066 / 0.9725 / 0.9825 |
| **V8.6** | LR horizon fix, λ_drift 10→30, sensor-bias augmentation + corrected ablation protocol | **0.4553 / 0.3695 / 0.4129** |

The arc: V8.4 removed structural error sources. V8.5 made the remaining integration-drift bias *visible* to both the loss and the selection metric. V8.6 let that bias actually converge, pushed harder on it, and made the robustness claim testable, at which point the test said the claim wasn't met.

V8.4 and V8.5 are not in this repository; V8.6 is the accepted configuration and `HAPIT8v6_hpo_config.m` is the same pipeline under the searched hyperparameters.
