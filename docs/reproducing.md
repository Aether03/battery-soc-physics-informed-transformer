# Reproducing these results

## Environment

The published numbers came from MATLAB R2024a on Windows 11, on a Dell Inspiron 5415 with an AMD Ryzen 5 5500U, 8 GB RAM and integrated graphics only. Everything runs on CPU. No CUDA device is required or used.

**MATLAB toolboxes**

| Toolbox | Needed for | Required? |
|---|---|---|
| Deep Learning Toolbox | `dlarray`, `dlnetwork`, `dlfeval`, `adamupdate`, `selfAttentionLayer` | Yes |
| Statistics and Machine Learning Toolbox | `bayesopt`, for the Bayesian-optimisation search only | Only that arm |
| Signal Processing Toolbox | `butter` / `filtfilt` zero-phase filtering | No. A 3-sample moving-average fallback runs automatically |

**Python** is needed only for the ASHA and BOHB orchestrators. The baseline model and the Bayesian-optimisation arm are pure MATLAB.

```bash
pip install -r requirements.txt
```

`matlabengine` must match your MATLAB release. If the pinned version is wrong for yours, install it from your MATLAB tree instead:

```bash
cd "$MATLABROOT/extern/engines/python"
python setup.py install
```

## Path setup

Every MATLAB entry point expects three things on the path or in the working directory: `prepareBatteryDataV9.m`, `PositionalEncodingLayer.m`, and the four `Merged_B00XX_Lifecycle_Sample.csv` files.

```matlab
addpath(genpath('src/matlab'));
addpath('data');
```

`PositionalEncodingLayer` is a `classdef`, so MATLAB resolves it by filename, so it must be named exactly that and be visible on the path, or the layer graph fails to build.

## Inference on the trained checkpoint

The fast path. Roughly a minute, no training.

```matlab
cd src/matlab
HA_PIT_test_v8
```

Edit `test_file` at line 37 to point at any 4-column CSV with headers `Voltage_measured`, `Current_measured`, `Temperature_measured`, `Time`. The script loads `models/hapit_v8p6_best_checkpoint.mat` and reproduces the full evaluation: continuous and windowed-oracle RMSE, per-segment breakdown, and the current-bias ablation.

Expected on `Merged_B0018_Lifecycle_Sample.csv`: continuous RMSE 0.4129%, MAE 0.1981%, R² 0.9998.

## Retraining from scratch

```matlab
cd src/matlab
HAPIT8v6
```

Around 1 h 38 m on the reference machine. Best epoch lands near 107 and early stopping triggers around 187, with patience 80 on the continuous validation metric.

Exact reproduction of the published checkpoint is not guaranteed. Dropout, the sensor-bias augmentation draw, and the PCGrad projection order are all stochastic and the scripts do not set a global RNG seed. Expect the same behaviour and metrics within run-to-run noise, not bit-identical weights. If you need determinism, add `rng(0,'twister')` before the training loop, but note that this changes the augmentation stream, so the numbers will shift slightly from the published ones.

`HAPIT8v6_hpo_config.m` is the same pipeline with the hyperparameters the Bayesian-optimisation search found. It produces the 0.6518% clean / bias-robust configuration discussed in the README.

## Regenerating the data

The derived CSVs in `data/` are committed, so nothing below is needed to run the model. It's here so the derivation is auditable.

1. Download the NASA PCoE Li-ion Battery Aging Dataset. See [`data/README.md`](../data/README.md) for links.
2. Put each cell's per-cycle CSV exports and its `metadata_<cell>.csv` in one directory.
3. Run `src/data_prep/build_lifecycle_sample.m` from that directory. It selects four beginning-of-life, four middle-of-life and four end-of-life charge/discharge cycles and stitches them onto a continuous time axis.

For the 338-cycle full-life file used in the stress test, run `src/data_prep/merge_full_lifecycle.m` with `numFilesToMerge` set to `height(cycleData)`. The output is around 38 MB, which is why it isn't committed.

## Re-running the hyperparameter searches

Budget four days of compute for all three. Each writes a trial-level CSV matching the format in `results/hpo/`.

**ASHA.** Optuna drives, MATLAB trains in resumable chunks:

```bash
cd src/hpo/asha
python ASHA_Orchestrator_V8.py
```

**BOHB.** Same trainer mechanics, Hyperband pruner instead of successive halving, plus a 12-trial refinement study:

```bash
cd src/hpo/bohb
python BOHB_Orchestrator_V8.py
```

**Bayesian optimisation + PCGrad.** Pure MATLAB:

```matlab
cd src/hpo/bayesopt_pcgrad
BayesOpt_PCGrad_V8_master
```

Two things to know before you start a search:

- The MATLAB trainers cache the prepared dataset in `persistent` variables across trials. After any change to `prepareBatteryDataV9.m` you must run `clear train_hybrid_asha_V8` (or the equivalent) or the search silently trains on stale data.
- Interrupted searches resume. ASHA and BOHB checkpoint per trial to `checkpoint_*_<trial_id>.mat`, and the Optuna study persists in a `.db` file. Both are gitignored. Delete them to start clean.

## Known divergence between the searches and the baseline

The three HPO harnesses were built against the four-term loss at `seqLen=480`→`240` and select on the windowed validation metric. The production model adds the drift loss, selects on the continuous metric, and runs `seqLen=480`. Trial RMSEs are therefore not comparable to the baseline's 0.4129%. See [`hpo-study.md`](hpo-study.md) for how the comparison is made fairly.
