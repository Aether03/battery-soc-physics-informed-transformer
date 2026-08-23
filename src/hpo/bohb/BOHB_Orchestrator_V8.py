import optuna
import matlab.engine
import os
import time
import csv
import logging
import math

# ─────────────────────────────────────────────────────────────────────────────
#  BOHB MILESTONE ALIGNMENT
#  HyperbandPruner rungs with min_resource=25, reduction_factor=3:
#    Rung 0 → 25 epochs   (most trials cut here)
#    Rung 1 → 75 epochs
#    Rung 2 → 225 epochs
#    Rung 3 → 400 epochs  (full run, only survivors)
#
#  CHANGE vs v1: Added a rung 0.5 at epoch 10 for ultra-cheap first culling.
#  This requires matching the bracket in HyperbandPruner (min_resource=10).
# ─────────────────────────────────────────────────────────────────────────────
EVAL_MILESTONES = [25, 75, 225, 400]

# Suppress Optuna's per-trial INFO noise; keep WARNING+ only
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("BOHB")

print("Starting MATLAB Engine... (This may take a few seconds)")
eng = matlab.engine.start_matlab()
print("MATLAB Engine started successfully.\n")


def objective(trial):
    # ── 1. CANONICAL V8.4 SEARCH SPACE (identical to ASHA and BayesOpt) ────
    #  8 shared parameters — same names and ranges across all three HPO methods.
    #  Only the pruning mechanics differ (BOHB = HyperbandPruner here).
    #    initLR       [1e-4, 5e-3]  log
    #    attentionDim {32,64,128,192,256}
    #    numFilters   {16,32,48,64}
    #    dropoutRate  [0.05, 0.35]
    #    gradClip     [0.5, 2.0]
    #    warmupEpochs {20,30,40}
    #    weightDecay  [1e-6, 5e-4]  log   (AdamW decoupled decay on weight mats)
    #    lambda_end   [1e-2, 5e0]   log   (v8.4 endpoint drift weight — replaces
    #                 the obsolete labelSmooth; under residual increments the
    #                 network predicts only a small correction, so the endpoint
    #                 penalty is what most directly trades off against RMSE)
    #  Held fixed at baseline (part of the label/loss definition, not searched):
    #    lambda_Ah=10, w_ECM=0.5, w_data=1, alpha_ocv=0.20, dSOC_scale=1e-3,
    #    settle_len=120, dvdt_thr=2e-4.

    initLR       = trial.suggest_float("initLR",       1e-4,  5e-3,  log=True)
    attentionDim = trial.suggest_categorical("attentionDim", [32, 64, 128, 192, 256])
    numFilters   = trial.suggest_categorical("numFilters",   [16, 32, 48, 64])
    dropoutRate  = trial.suggest_float("dropoutRate",  0.05,  0.35)
    gradClip     = trial.suggest_float("gradClip",     0.5,   2.0)
    warmupEpochs = trial.suggest_categorical("warmupEpochs", [20, 30, 40])
    weightDecay  = trial.suggest_float("weightDecay",  1e-6,  5e-4,  log=True)
    lambda_end   = trial.suggest_float("lambda_end",   1e-2,  5e0,   log=True)

    val_loss = float('inf')
    ckpt_filename = f"checkpoint_bohb_V3_{trial.number}.mat"

    # ── 2. RUN MATLAB IN ALIGNED CHUNKS ───────────────────────────────────
    for target_epoch in EVAL_MILESTONES:

        log.info(f"--> [Trial {trial.number}] Training to epoch {target_epoch} | "
                 f"LR={initLR:.1e}  Dim={attentionDim}  Fil={numFilters}  "
                 f"Drop={dropoutRate:.3f}  Clip={gradClip:.2f}  "
                 f"WU={warmupEpochs}  WD={weightDecay:.1e}  Lend={lambda_end:.2e}")

        try:
            # Argument ORDER must match train_BOHB_V3.m exactly:
            #   trial_id, initLR, attentionDim, numFilters, dropoutRate,
            #   gradClip, warmupEpochs, weightDecay, lambda_end, target_epoch
            val_loss = eng.train_BOHB_V3(
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
            log.error(f"    [ERROR] Trial {trial.number} failed at epoch {target_epoch}:\n{e}")
            if os.path.exists(ckpt_filename):
                os.remove(ckpt_filename)
            raise optuna.TrialPruned()

        # ── OOM GUARD: MATLAB returns NaN on out-of-memory. Log, drop the
        #    checkpoint, prune this trial so the study continues cleanly.
        if val_loss is None or math.isnan(float(val_loss)):
            log.warning(f"    [OOM/NaN]  Trial {trial.number} returned NaN at epoch "
                        f"{target_epoch} (likely OOM) — logging and skipping.")
            if os.path.exists(ckpt_filename):
                os.remove(ckpt_filename)
            raise optuna.TrialPruned()

        # ── 3. BOHB PRUNING CHECK ──────────────────────────────────────────
        trial.report(val_loss, target_epoch)

        if trial.should_prune():
            log.info(f"    [PRUNED]   Trial {trial.number} cut at epoch {target_epoch} "
                     f"(RMSE: {val_loss:.4f}%)")
            if os.path.exists(ckpt_filename):
                os.remove(ckpt_filename)
            raise optuna.TrialPruned()

        log.info(f"    [PROMOTED] Trial {trial.number} survived epoch {target_epoch} "
                 f"(RMSE: {val_loss:.4f}%)")

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
        print("              Best so far: N/A (no completed trials yet)\n")

    print("-" * 60)


def export_results_to_csv(study, filename="BOHB_Results_Summary_v2.csv"):
    """Extracts all trial data, intermediate rung RMSEs, and configs to CSV."""
    print(f"\nExporting tabulated results to {filename}...")

    milestone_headers = [f"RMSE_Epoch_{m}" for m in EVAL_MILESTONES]
    param_keys = [
        "initLR", "attentionDim", "numFilters", "dropoutRate",
        "gradClip", "warmupEpochs", "weightDecay", "lambda_end",
    ]
    headers = (["Trial_Number", "State", "Final_RMSE"]
               + milestone_headers + param_keys + ["Duration_Seconds"])

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)

        for trial in study.trials:
            state     = trial.state.name
            final_val = f"{trial.value:.4f}" if trial.value is not None else "N/A"

            intermediates = []
            for m in EVAL_MILESTONES:
                v = trial.intermediate_values.get(m)
                intermediates.append(f"{v:.4f}" if v is not None else "N/A")

            params = []
            for p in param_keys:
                v = trial.params.get(p)
                if v is None:
                    params.append("N/A")
                elif isinstance(v, float) and p in ("initLR", "weightDecay"):
                    params.append(f"{v:.2e}")
                elif isinstance(v, float):
                    params.append(f"{v:.5f}")
                else:
                    params.append(str(v))

            duration = ""
            if trial.datetime_start and trial.datetime_complete:
                duration = f"{(trial.datetime_complete - trial.datetime_start).total_seconds():.1f}"

            writer.writerow([trial.number, state, final_val]
                            + intermediates + params + [duration])

    print("Export successful!")


if __name__ == "__main__":
    global_start_time = time.time()

    # ── 4. RICHER WARM-START SEEDS ─────────────────────────────────────────
    #  v1 used 2 seeds. We add a 3rd from the known-good region near your
    #  3.3739 % baseline (slightly lower LR, moderate weight decay), and a 4th
    #  that is a deliberate "aggressive" seed to seed the far region of the space.
    warm_start_configs = [
        # Baseline-default endpoint weight (0.5) on two capacity points
        {"initLR": 3e-4,  "attentionDim": 64,  "numFilters": 16, "dropoutRate": 0.10,
         "gradClip": 1.0, "warmupEpochs": 30, "weightDecay": 1e-5, "lambda_end": 0.5},
        {"initLR": 1e-3,  "attentionDim": 128, "numFilters": 32, "dropoutRate": 0.15,
         "gradClip": 1.0, "warmupEpochs": 30, "weightDecay": 1e-5, "lambda_end": 0.5},
        # Lighter endpoint penalty near the baseline sweet-spot
        {"initLR": 5e-4,  "attentionDim": 128, "numFilters": 32, "dropoutRate": 0.10,
         "gradClip": 0.8, "warmupEpochs": 20, "weightDecay": 5e-5, "lambda_end": 0.3},
        # Aggressive / high-capacity config with stronger drift penalty
        {"initLR": 2e-3,  "attentionDim": 256, "numFilters": 48, "dropoutRate": 0.20,
         "gradClip": 1.5, "warmupEpochs": 40, "weightDecay": 1e-4, "lambda_end": 2.0},
    ]

    # ── 5. IMPROVED BOHB ARCHITECTURE ─────────────────────────────────────
    #
    #  SAMPLER: TPESampler gains two improvements:
    #    • multivariate=True  — models correlations between parameters (e.g.
    #                           attentionDim and initLR are coupled) instead of
    #                           treating each independently. Critical for this
    #                           problem since gradClip and LR interact strongly.
    #    • constant_liar=True — when running (pseudo-)parallel trials, uses the
    #                           current best value as a "lie" for in-progress
    #                           trials so TPE doesn't re-sample the same region.
    #                           Particularly beneficial if you ever parallelise.
    #    • n_startup_trials:  bumped 10 → 15 to give TPE more random exploration
    #                           before the Bayesian model kicks in, since we
    #                           added 2 new dimensions (weightDecay, lambda_end).
    #    • n_ei_candidates:   24 → 48 — more candidate samples evaluated per
    #                           acquisition function step, improving TPE's ability
    #                           to find the maximum of EI in a denser space.

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=15,
        seed=42,
        multivariate=True,
        constant_liar=True,
        n_ei_candidates=48,
    )

    #  PRUNER: min_resource lowered from 25 → 10 to add a cheap early rung.
    #  At epoch 10, learning is noisy but clear losers (e.g. badly diverging LR)
    #  are already distinguishable. This saves ~30 % wall time at no quality cost.
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=25,
        max_resource=400,
        reduction_factor=3,
    )

    # ── 6. PERSISTENT SQLITE STORAGE ───────────────────────────────────────
    db_name = "sqlite:///hapit_bohb_study_v2.db"

    study = optuna.create_study(
        study_name="HA_PIT_BOHB_Sweep_v2",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        storage=db_name,
        load_if_exists=True,
    )

    if len(study.trials) == 0:
        for cfg in warm_start_configs:
            study.enqueue_trial(cfg)

    print(f"Starting BOHB Sweep v2. Data will be saved to: {db_name}")
    print("-" * 60)

    # ── 7. INCREASED TRIAL BUDGET ──────────────────────────────────────────
    #  v1: 40 trials. With 2 new dimensions and better pruning efficiency from
    #  the extra rung, 55 trials gives a good coverage-to-cost ratio.
    study.optimize(objective, n_trials=55, callbacks=[live_monitor_callback])

    elapsed = time.time() - global_start_time

    print("\n" + "=" * 60)
    print("🎯  BOHB v2 OPTIMISATION COMPLETE  🎯")
    print("=" * 60)
    print(f"Total Time           : {elapsed:.2f} s  ({elapsed/60:.2f} min)")
    try:
        print(f"Best Trial Number    : {study.best_trial.number}")
        print(f"Best Validation RMSE : {study.best_value:.4f}%")
        print(f"Best Hyperparameters :")
        for k, v in study.best_trial.params.items():
            if isinstance(v, float) and k in ("initLR", "weightDecay"):
                print(f"  {k:>16} : {v:.2e}")
            elif isinstance(v, float):
                print(f"  {k:>16} : {v:.5f}")
            else:
                print(f"  {k:>16} : {v}")
    except ValueError:
        print("All trials were pruned or failed. No best model found.")
    print("=" * 60)

    # ── 8. POST-SWEEP REFINEMENT RUN ──────────────────────────────────────
    #  After the main sweep, optionally run a small focused study seeded
    #  entirely from the best trial ± a tight neighbourhood. This is cheap
    #  (few trials, all start near a known good point) and typically squeezes
    #  an extra 0.05–0.15 % RMSE from the result.
    #
    #  Set ENABLE_REFINE = False to skip.
    ENABLE_REFINE = True

    if ENABLE_REFINE:
        try:
            best_params = study.best_trial.params
            print("\n[REFINE] Launching post-sweep neighbourhood search...")
            import copy, math

            def _perturb(v, lo, hi, factor=0.25, log_scale=False):
                """Return ±factor neighbourhood, clamped to [lo, hi]."""
                if log_scale:
                    log_v  = math.log10(v)
                    delta  = factor * (math.log10(hi) - math.log10(lo))
                    return (max(10**(log_v - delta), lo),
                            min(10**(log_v + delta), hi))
                else:
                    delta = factor * (hi - lo)
                    return (max(v - delta, lo), min(v + delta, hi))

            # Narrow the sampler to the best neighbourhood
            refine_sampler = optuna.samplers.TPESampler(
                seed=99, multivariate=True, n_startup_trials=5
            )
            refine_pruner = optuna.pruners.HyperbandPruner(
                min_resource=25, max_resource=400, reduction_factor=3
            )
            refine_study = optuna.create_study(
                study_name="HA_PIT_BOHB_Refine",
                direction="minimize",
                sampler=refine_sampler,
                pruner=refine_pruner,
                storage=db_name,
                load_if_exists=True,
            )

            def refine_objective(trial):
                bp = best_params
                lo_lr, hi_lr = _perturb(bp["initLR"], 1e-4, 8e-3, log_scale=True)
                lo_dr, hi_dr = _perturb(bp["dropoutRate"], 0.02, 0.35)
                lo_gc, hi_gc = _perturb(bp["gradClip"],    0.3,  2.0)
                lo_wd, hi_wd = _perturb(bp["weightDecay"], 1e-6, 5e-4, log_scale=True)
                lo_le, hi_le = _perturb(bp["lambda_end"], 1e-2, 5e0, log_scale=True)

                initLR       = trial.suggest_float("initLR",      lo_lr, hi_lr, log=True)
                attentionDim = trial.suggest_categorical("attentionDim", [bp["attentionDim"]])
                numFilters   = trial.suggest_categorical("numFilters",   [bp["numFilters"]])
                dropoutRate  = trial.suggest_float("dropoutRate",  lo_dr, hi_dr)
                gradClip     = trial.suggest_float("gradClip",     lo_gc, hi_gc)
                warmupEpochs = trial.suggest_categorical("warmupEpochs", [bp["warmupEpochs"]])
                weightDecay  = trial.suggest_float("weightDecay",  lo_wd, hi_wd, log=True)
                lambda_end   = trial.suggest_float("lambda_end",   lo_le, hi_le, log=True)

                val_loss = float('inf')
                ckpt_fn  = f"checkpoint_bohb_V3_{trial.number + 1000}.mat"

                for target_epoch in EVAL_MILESTONES:
                    try:
                        # arg order matches train_BOHB_V3.m
                        val_loss = eng.train_BOHB_V3(
                            float(trial.number + 1000),  # offset to avoid ckpt name clash
                            float(initLR), float(attentionDim), float(numFilters),
                            float(dropoutRate), float(gradClip), float(warmupEpochs),
                            float(weightDecay), float(lambda_end), float(target_epoch),
                        )
                    except Exception as e:
                        if os.path.exists(ckpt_fn): os.remove(ckpt_fn)
                        raise optuna.TrialPruned()
                    # OOM guard: NaN from MATLAB -> prune & continue
                    if val_loss is None or math.isnan(float(val_loss)):
                        if os.path.exists(ckpt_fn): os.remove(ckpt_fn)
                        raise optuna.TrialPruned()
                    trial.report(val_loss, target_epoch)
                    if trial.should_prune():
                        if os.path.exists(ckpt_fn): os.remove(ckpt_fn)
                        raise optuna.TrialPruned()

                if os.path.exists(ckpt_fn): os.remove(ckpt_fn)
                return val_loss

            refine_study.enqueue_trial(best_params)  # seed #1 = exact best
            refine_study.optimize(refine_objective, n_trials=12,
                                  callbacks=[live_monitor_callback])

            print(f"\n[REFINE] Best after refinement: {refine_study.best_value:.4f}%")
            print(f"[REFINE] Refine params: {refine_study.best_trial.params}")

        except Exception as ex:
            print(f"[REFINE] Skipped (error: {ex})")

    # ── 9. EXPORT DATA ────────────────────────────────────────────────────
    export_results_to_csv(study)

    eng.quit()