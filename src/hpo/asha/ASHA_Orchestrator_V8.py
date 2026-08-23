import optuna
import matlab.engine
import os
import time
import csv
import math

# ─────────────────────────────────────────────────────────────────────────────
#  ASHA ORCHESTRATOR — V8.3 increment pipeline
#
#  Calls train_hybrid_asha_V8.m, which trains the SAME model as the
#  BayesOpt/PCGrad method (prepareBatteryDataV9, 5 features, delta-anchoring,
#  continuous-reconstruction objective). The 8-parameter search space below is
#  IDENTICAL in type and range to that method; the only differences are the
#  ASHA-specific mechanics (multi-fidelity milestones, successive-halving
#  pruner, TPE sampler, checkpoint/resume).
#
#  MILESTONE ALIGNMENT
#  SuccessiveHalvingPruner fires at  min_resource * reduction_factor^k.
#  With min_resource=25, reduction_factor=3  ->  25, 75, 225.
#  EVAL_MILESTONES is aligned to those, with the final rung = cfg.numEpochs
#  (400) in train_hybrid_asha_V8.m so the last chunk completes full fidelity.
# ─────────────────────────────────────────────────────────────────────────────
EVAL_MILESTONES = [25, 75, 225, 400]

print("Starting MATLAB Engine in the background...")
eng = matlab.engine.start_matlab()
print("MATLAB Engine started successfully.\n")


def objective(trial):
    # ── 1. SEARCH SPACE (8 params — identical to the BayesOpt/PCGrad method) ──
    #   initLR, attentionDim, numFilters, dropoutRate, gradClip, warmupEpochs
    #   are unchanged. weightDecay and lambda_end are ADDED so all three HPO
    #   methods optimise over one shared space.
    #     • weightDecay : AdamW decoupled decay (weight matrices only).
    #     • lambda_end  : endpoint drift weight in the v8.4 residual loss —
    #                     sets how strongly the physics term is enforced.
    #   alpha_ocv / seqLen / output scales are NOT searched (label definition).
    initLR       = trial.suggest_float("initLR",       1e-4,  5e-3,  log=True)
    attentionDim = trial.suggest_categorical("attentionDim", [32, 64, 128, 192, 256])
    numFilters   = trial.suggest_categorical("numFilters",   [16, 32, 48, 64])
    dropoutRate  = trial.suggest_float("dropoutRate",  0.05,  0.35)
    gradClip     = trial.suggest_float("gradClip",     0.5,   2.0)
    warmupEpochs = trial.suggest_categorical("warmupEpochs", [20, 30, 40])
    weightDecay  = trial.suggest_float("weightDecay",  1e-6,  5e-4,  log=True)
    # V8.4: 8th param is lambda_end (endpoint drift weight). The old lambda_Ah
    # [1e2,1e6] is obsolete — under residual increments the network predicts only
    # a small correction; lambda_end is what trades off against continuous RMSE.
    lambda_end   = trial.suggest_float("lambda_end",   1e-2,  5e0,   log=True)

    val_loss = float('inf')
    ckpt_filename = f"checkpoint_asha_V8_{trial.number}.mat"

    # ── 2. RUN MATLAB IN ALIGNED CHUNKS ───────────────────────────────────
    for target_epoch in EVAL_MILESTONES:

        print(f"--> [Trial {trial.number}] Running to epoch {target_epoch}... "
              f"(LR: {initLR:.1e}, Dim: {attentionDim}, Fil: {numFilters}, "
              f"Drop: {dropoutRate:.2f}, Clip: {gradClip:.2f}, WU: {warmupEpochs}, "
              f"WD: {weightDecay:.1e}, Lend: {lambda_end:.2e})")

        try:
            # All inputs cast to float() — MATLAB must receive 'double', not int64.
            # Argument ORDER must match train_hybrid_asha_V8.m exactly:
            #   trial_id, initLR, attentionDim, numFilters, dropoutRate,
            #   gradClip, warmupEpochs, weightDecay, lambda_end, target_epoch
            val_loss = eng.train_hybrid_asha_V8(
                float(trial.number),
                float(initLR),
                float(attentionDim),
                float(numFilters),
                float(dropoutRate),
                float(gradClip),
                float(warmupEpochs),
                float(weightDecay),
                float(lambda_end),
                float(target_epoch),
            )

        except Exception as e:
            print(f"    [ERROR] Trial {trial.number} failed at epoch {target_epoch}: {e}")
            if os.path.exists(ckpt_filename):
                os.remove(ckpt_filename)
            raise optuna.TrialPruned()

        # ── OOM GUARD: MATLAB returns NaN on out-of-memory. Log it, drop the
        #    checkpoint, and prune this trial so the study continues cleanly.
        if val_loss is None or math.isnan(float(val_loss)):
            print(f"    [OOM/NaN]  Trial {trial.number} returned NaN at epoch "
                  f"{target_epoch} (likely OOM) — logging and skipping.")
            if os.path.exists(ckpt_filename):
                os.remove(ckpt_filename)
            raise optuna.TrialPruned()

        # Report continuous-reconstruction val RMSE to ASHA
        trial.report(val_loss, target_epoch)

        if trial.should_prune():
            print(f"    [PRUNED]   Trial {trial.number} cut at epoch {target_epoch} "
                  f"(Loss: {val_loss:.4f}%)")
            if os.path.exists(ckpt_filename):
                os.remove(ckpt_filename)
            raise optuna.TrialPruned()

        print(f"    [PROMOTED] Trial {trial.number} survived epoch {target_epoch}! "
              f"(Loss: {val_loss:.4f}%)")

    if os.path.exists(ckpt_filename):
        os.remove(ckpt_filename)

    return val_loss


def live_monitor_callback(study, trial):
    if trial.state == optuna.trial.TrialState.COMPLETE:
        status = "COMPLETED"
    elif trial.state == optuna.trial.TrialState.PRUNED:
        status = "PRUNED"
    else:
        status = "FAILED"

    print(f"\n[LIVE UPDATE] Trial {trial.number} -> {status}")
    if trial.value is not None:
        print(f"              Trial RMSE : {trial.value:.4f}%")

    try:
        best_val = study.best_value
        best_num = study.best_trial.number
        print(f"              Best so far: {best_val:.4f}%  (Trial {best_num})\n")
    except ValueError:
        print(f"              Best so far: N/A (no completed trials yet)\n")

    print("-" * 60)


if __name__ == "__main__":
    global_start_time = time.time()

    # ── 3. WARM-START SEEDS (must list all 8 params) ───────────────────────
    #  Two anchors so TPE can model both ends of the surface from the start:
    #  a moderate-capacity config and a higher-capacity one. lambda_end seeds
    #  span the endpoint-drift-weight range; weightDecay seeds stay light.
    warm_start_configs = [
        {
            "initLR":       3e-4,
            "attentionDim": 64,
            "numFilters":   32,
            "dropoutRate":  0.10,
            "gradClip":     1.0,
            "warmupEpochs": 20,
            "weightDecay":  1e-5,
            "lambda_end":   0.5,
        },
        {
            "initLR":       1e-3,
            "attentionDim": 128,
            "numFilters":   48,
            "dropoutRate":  0.15,
            "gradClip":     1.0,
            "warmupEpochs": 30,
            "weightDecay":  1e-4,
            "lambda_end":   2.0,
        },
    ]

    # ── 4. STUDY CREATION ──────────────────────────────────────────────────
    #  TPE sampler (models the loss surface after n_startup_trials random
    #  trials) + SuccessiveHalvingPruner (keeps top 1/3 at each rung).
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=10,   # random exploration before TPE takes over
        seed=42                # reproducibility
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=optuna.pruners.SuccessiveHalvingPruner(
            min_resource=25,            # first prune check at epoch 25
            reduction_factor=3,         # keep top 1/3 at each rung
            min_early_stopping_rate=0   # allow pruning from the first rung
        )
    )

    for cfg in warm_start_configs:
        study.enqueue_trial(cfg)

    print("Starting ASHA sweep (V8.3 pipeline, TPE sampler, warm-start seeds)...")
    print("-" * 60)

    study.optimize(objective, n_trials=40, callbacks=[live_monitor_callback])

    global_end_time = time.time()
    elapsed_seconds = global_end_time - global_start_time

    print("\n" + "=" * 55)
    print("***  OPTIMISATION COMPLETE  ***")
    print("=" * 55)
    print(f"Total Execution Time : {elapsed_seconds:.2f} s  ({elapsed_seconds/60:.2f} min)")
    try:
        print(f"Best Trial Number    : {study.best_trial.number}")
        print(f"Best Validation RMSE : {study.best_value:.4f}%")
        print(f"Best Hyperparameters : {study.best_trial.params}")
    except ValueError:
        print("All trials were pruned or failed.  No best model found.")
    print("=" * 55)

    # ── 5. EXPORT RESULTS TO CSV ───────────────────────────────────────────
    csv_filename = f"hpo_results_summary_v8_{int(global_start_time)}.csv"
    print(f"\nExporting detailed tabulated results to {csv_filename}...")

    with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        headers = [
            "Trial_Number", "Status", "Final_RMSE_Pct",
            "RMSE_Epoch_25", "RMSE_Epoch_75", "RMSE_Epoch_225", "RMSE_Epoch_400",
            "initLR", "attentionDim", "numFilters", "dropoutRate", "gradClip",
            "warmupEpochs", "weightDecay", "lambda_end", "Duration_Seconds"
        ]
        writer.writerow(headers)

        for trial in study.trials:
            rmse_25  = trial.intermediate_values.get(25,  "N/A")
            rmse_75  = trial.intermediate_values.get(75,  "N/A")
            rmse_225 = trial.intermediate_values.get(225, "N/A")
            rmse_400 = trial.intermediate_values.get(400, "N/A")

            fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else v
            rmse_25_str, rmse_75_str = fmt(rmse_25), fmt(rmse_75)
            rmse_225_str, rmse_400_str = fmt(rmse_225), fmt(rmse_400)
            final_rmse_str = f"{trial.value:.4f}" if trial.value is not None else "N/A"

            p = trial.params
            initLR_str = f"{p.get('initLR', 0):.2e}" if 'initLR' in p else "N/A"
            drop_str   = f"{p.get('dropoutRate', 0):.4f}" if 'dropoutRate' in p else "N/A"
            clip_str   = f"{p.get('gradClip', 0):.4f}" if 'gradClip' in p else "N/A"
            wd_str     = f"{p.get('weightDecay', 0):.2e}" if 'weightDecay' in p else "N/A"
            lam_str    = f"{p.get('lambda_end', 0):.2e}" if 'lambda_end' in p else "N/A"

            dur = trial.duration.total_seconds() if trial.duration else 0
            dur_str = f"{dur:.2f}" if dur else "N/A"

            row = [
                trial.number, trial.state.name, final_rmse_str,
                rmse_25_str, rmse_75_str, rmse_225_str, rmse_400_str,
                initLR_str, p.get('attentionDim', 'N/A'), p.get('numFilters', 'N/A'),
                drop_str, clip_str, p.get('warmupEpochs', 'N/A'),
                wd_str, lam_str, dur_str
            ]
            writer.writerow(row)

    print("Export complete!")
    eng.quit()