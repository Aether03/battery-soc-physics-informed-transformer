# Data Architecture — HA-PIT Streamlit Demo

*(Originally scoped as "Backend Schema" in the request. Reframed here because
there is no database and no user authentication in this app — see
`02_Technical_Requirements.md` for why. What follows is the real equivalent:
the schema of every data file the app reads, how they relate, and an explicit
statement on the authentication question.)*

## Why files instead of tables

The entire dataset is small (a few MB total), fully static (nothing is
written at runtime — no inserts, no updates), and has exactly one consumer
(this app). A database would add an operational dependency (a service to
provision, connect to, and keep in sync) with no corresponding benefit —
there's no concurrent write load, no need for ad-hoc queries beyond what the
loader functions already do, and no multi-tenant access pattern. Flat files,
checked into the repo, are simpler and directly satisfy the "self-contained,
deployable to Streamlit Community Cloud with no external services" requirement
in the TRD.

## File-by-file schema

### `data/metrics.json`

Hand-transcribed from the source repo's README tables; cross-verified against
raw evaluation output where an overlap exists (see Verification section
below).

```
{
  "headline": {
    "test_rmse_pct": float, "test_cell": str, "full_life_rmse_pct": float,
    "train_time": str, "hardware": str, "params": int,
    "best_epoch": int, "early_stop_epoch": int
  },
  "results_table": [
    { "split": str, "cells": str, "rmse": float, "mae": float,
      "max_error": float, "r2": float }, ...
  ],
  "stress_tests": [
    { "name": str, "probes": str, "rmse": float, "mae": float,
      "max_error": float, "r2": float }, ...
  ],
  "bias_ablation": {
    "rows": [
      { "injected_bias_a": float, "pure_coulomb": float,
        "hapit_v86": float, "hapit_hpo_config": float }, ...
    ]
  },
  "hpo_comparison": {
    "search_rows": [
      { "method": str, "strategy": str, "trials": int, "pruned": int,
        "best_trial_rmse": float, "compute_hours": float }, ...
    ],
    "fair_comparison_rows": [
      { "configuration": str, "continuous_test_rmse": float }, ...
    ]
  },
  "loss_terms": [
    { "term": str, "weight": float, "job": str }, ...
  ]
}
```

### `data/predictions.npz` (NumPy compressed archive)

Extracted from the source project's raw MATLAB workspace dump
(`workspace_8v6.mat`), which was not itself part of the public repo. Nine
arrays, one dtype (`float64`), grouped in three sets of three:

| Array | Shape | Meaning |
|---|---|---|
| `gt_train`, `pred_train`, `err_train` | `(136320,)` each | True SOC, predicted SOC, error — train split (B0005+B0006) |
| `gt_val`, `pred_val`, `err_val` | `(68160,)` each | Same, validation split (B0007) |
| `gt_test`, `pred_test`, `err_test` | `(78720,)` each | Same, test split (B0018) |

All three arrays within a set share an index: `err_X[i] == pred_X[i] -
gt_X[i]` for every `i`. Values are flattened, concatenated 480-step windows
(matching the model's `seqLen=480`) in the order the original continuous-
reconstruction evaluation produced them — not literally timestamped against
wall-clock seconds (see `03_App_Flow.md`, Results Explorer caption).

### `data/training_curves.npz`

| Array | Shape | Meaning |
|---|---|---|
| `lossLog` | `(400, 6)` | Six loss-component values, per epoch, 400 epochs |
| `valLog` | `(400,)` | Continuous-reconstruction validation RMSE, per epoch |
| `valLogWin` | `(400,)` | Windowed-oracle validation RMSE, per epoch |
| `best_epoch` | scalar (stored as 0-d) | Epoch selected as final (107) |

### `data/hpo/{asha,bohb,bayesopt_pcgrad}_trials.csv`

Three separate files with **different native column names** (this is real,
not a design choice — they came from three different search harnesses).
`data_loader.load_hpo_trials()` normalizes all three into one schema before
use:

| Normalized column | Type | Source column (ASHA / BOHB / BayesOpt+PCGrad) |
|---|---|---|
| `method` | str | (added, not in source: literal `"ASHA"` / `"BOHB"` / `"Bayesian optimisation + PCGrad"`) |
| `trial_number` | int | `Trial_Number` / `Trial_Number` / `Trial` |
| `rmse` | float | `Final_RMSE_Pct` / `Final_RMSE` / `RMSE` |
| `status` | str | `Status` / `State` / (not present — filled `"COMPLETE"`, since the source README states 0 of 55 PCGrad trials were pruned) |
| `duration_seconds` | float | `Duration_Seconds` / `Duration_Seconds` / `EvalTime_Seconds` |

### `data/hapit_pytorch.pt` (added with the PyTorch port)

PyTorch `state_dict` for `model.HAPITTransformer` — all 44 learnable
parameter groups (80,866 parameters) converted from the MATLAB checkpoint by
`PyTorch_port/convert_weights.py`. Loaded once per server process by
`data_loader.load_ported_model()` (`st.cache_resource`).

### `data/try_it/<cell>.npz` (added with the PyTorch port)

One file per cell (B0005, B0006, B0007, B0018), produced by
`PyTorch_port/matlab/export_all_cells.m` → `PyTorch_port/consolidate_for_app.py`.
Everything `inference.run_inference` needs for one live run:

| Field | Shape / type | Meaning |
|---|---|---|
| `X` | `(5, numWindows, 480)` float32 | Normalized network input windows `[V, I, T, C_dyn, SOC_ocv_instant]` |
| `stride` | int (240) | Window stride used to stitch per-window corrections |
| `I`, `C`, `V` | `(N,)` float64 | Per-timestep current, dynamic capacity (A·s), voltage |
| `gate` | `(N,)` bool | Settled-rest re-anchoring mask |
| `SOC_0` | float | Initial SOC anchor (shared with the labels) |
| `alpha_ocv` | float | Re-anchoring blend weight |
| `eq_s`, `soc_eq_s` | 1-D float64 | OCV→SOC lookup table (voltage → SOC) |
| `SOC_true` | `(N,)` float64 | Ground-truth SOC (0–1) |
| `dt`, `dSOC_scale` | float | Timestep (s) and network-correction scale |
| `official_rmse` | float | The original MATLAB model's RMSE on this cell, for comparison |

Recursion inputs stay float64 deliberately; only `X` (consumed as float32 by
PyTorch anyway) is downcast. `PyTorch_port/verify_full_pipeline.py` checks
these exact files through the shipped `inference.py` against MATLAB's official
output (all four cells within ~1e-6 pp RMSE).

### `data/figures/architecture.png`

Static image, no schema — copied as-is from the source repo's
`results/figures/architecture.png`.

## Relationships between files

```
metrics.json (headline, results_table)
      │  cross-checked against
      ▼
predictions.npz (gt_test / pred_test / err_test)
      │
      │  met_test.rmse in the source .mat structs == metrics.json
      │  results_table["Test"].rmse == 0.4129 (all three independently agree)
      ▼
training_curves.npz (best_epoch matches metrics.json headline.best_epoch: 107)

metrics.json (hpo_comparison.search_rows)
      │  trial counts / best RMSE summarized from
      ▼
hpo/*.csv (raw per-trial data, one row per trial attempt)

metrics.json (bias_ablation.rows[2], the 0.10 A row)
      │  cross-checked against
      ▼
the source workspace's "abl" struct (bias_A=0.1, rmse_nocorr=4.3162,
rmse_corr=4.3683) — confirmed to match to 4 decimal places during the build
```

## Verification performed during the build

Three independent numbers were cross-checked against each other and found
consistent, giving real confidence the transcribed `metrics.json` values are
correct rather than transcription errors:

1. `metrics.json` test RMSE (0.4129, from README) == `met_test.rmse` in both
   `models/hapit_v8p6_results.mat` and `workspace_8v6.mat` (0.4129315306246602).
2. `metrics.json` bias_ablation 0.10A row (Pure Coulomb 4.3162, V8.6 4.3683,
   from README) == the raw `abl` struct in `workspace_8v6.mat`
   (`rmse_nocorr=4.3161773814698`, `rmse_corr=4.3683433944018475`).
3. `err_test`'s extracted max value (1.9475) == the README's reported test
   Max Error (1.9475%) exactly.

## User authentication

**None exists, and none is needed.** The app is a public, read-only results
viewer with no per-user data, no write operations, and no content that
differs by viewer. Adding authentication would be pure overhead unless a
future version needs to gate something (e.g. an unpublished result) — if
that need arises, Streamlit Community Cloud supports viewer-level access
restriction at the deployment level, which wouldn't require adding an
authentication system to the app itself.
