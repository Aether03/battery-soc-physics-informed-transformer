from pathlib import Path
import json

import numpy as np
import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"


@st.cache_data
def load_metrics() -> dict:
    with open(DATA_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_predictions(split: str) -> pd.DataFrame:
    """split: 'train', 'val', or 'test'. Returns a DataFrame with columns
    [index, true_soc, pred_soc, error], one row per timestep in the
    continuous-reconstruction evaluation (flattened, ordered windows of
    length cfg.seqLen=480 each)."""
    npz = np.load(DATA_DIR / "predictions.npz")
    gt = npz[f"gt_{split}"]
    pred = npz[f"pred_{split}"]
    err = npz[f"err_{split}"]
    return pd.DataFrame({
        "index": np.arange(len(gt)),
        "true_soc": gt,
        "pred_soc": pred,
        "error": err,
    })


@st.cache_data
def load_training_curves() -> dict:
    npz = np.load(DATA_DIR / "training_curves.npz")
    loss_log = npz["lossLog"]
    return {
        "epoch": np.arange(1, loss_log.shape[0] + 1),
        "loss_components": loss_log,
        "val_rmse_continuous": npz["valLog"],
        "val_rmse_windowed": npz["valLogWin"],
        "best_epoch": int(npz["best_epoch"]),
    }


@st.cache_resource
def load_ported_model():
    import torch
    from model import HAPITTransformer

    model = HAPITTransformer()
    model.load_state_dict(torch.load(DATA_DIR / "hapit_pytorch.pt", map_location="cpu"))
    model.eval()
    return model


@st.cache_data
def load_try_it_cell(cell: str) -> dict:
    """cell: one of 'B0005', 'B0006', 'B0007', 'B0018'. Returns everything
    needed to run live inference and the Coulomb-counting + gated-anchoring
    reconstruction: the real windowed 5-channel input, per-timestep current/
    capacity/voltage/gate arrays, the OCV lookup table, and the official
    (MATLAB) reference RMSE for comparison."""
    npz = np.load(DATA_DIR / "try_it" / f"{cell}.npz")
    return {
        "X": npz["X"],
        "stride": int(npz["stride"]),
        "I": npz["I"],
        "C": npz["C"],
        "V": npz["V"],
        "gate": npz["gate"],
        "SOC_0": float(npz["SOC_0"]),
        "alpha_ocv": float(npz["alpha_ocv"]),
        "eq_s": npz["eq_s"],
        "soc_eq_s": npz["soc_eq_s"],
        "SOC_true": npz["SOC_true"],
        "dt": float(npz["dt"]),
        "dSOC_scale": float(npz["dSOC_scale"]),
        "official_rmse": float(npz["official_rmse"]),
    }


@st.cache_data
def load_hpo_trials() -> pd.DataFrame:
    """Normalizes the three HPO CSVs (which use different column names and
    conventions) into one tidy DataFrame:
    [method, trial_number, rmse, status, duration_seconds]."""
    frames = []

    asha = pd.read_csv(DATA_DIR / "hpo" / "asha_trials.csv")
    frames.append(pd.DataFrame({
        "method": "ASHA",
        "trial_number": asha["Trial_Number"],
        "rmse": asha["Final_RMSE_Pct"],
        "status": asha["Status"],
        "duration_seconds": asha["Duration_Seconds"],
    }))

    bohb = pd.read_csv(DATA_DIR / "hpo" / "bohb_trials.csv")
    frames.append(pd.DataFrame({
        "method": "BOHB",
        "trial_number": bohb["Trial_Number"],
        "rmse": bohb["Final_RMSE"],
        "status": bohb["State"],
        "duration_seconds": bohb["Duration_Seconds"],
    }))

    bayes = pd.read_csv(DATA_DIR / "hpo" / "bayesopt_pcgrad_trials.csv")
    frames.append(pd.DataFrame({
        "method": "Bayesian optimisation + PCGrad",
        "trial_number": bayes["Trial"],
        "rmse": bayes["RMSE"],
        "status": "COMPLETE",
        "duration_seconds": bayes["EvalTime_Seconds"],
    }))

    return pd.concat(frames, ignore_index=True)
