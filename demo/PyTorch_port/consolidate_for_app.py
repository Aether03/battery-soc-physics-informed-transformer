"""Repackages matlab/export_all_cells.m output into the app's data/try_it/*.npz.
X (the network input, which PyTorch consumes as float32 anyway) is downcast to
save space; everything feeding the 78k-step Coulomb recursion stays float64
so rounding can't accumulate differently from the verified path."""
from pathlib import Path

import numpy as np
import scipy.io as sio

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "data" / "try_it"
OUT_DIR.mkdir(parents=True, exist_ok=True)

for cell in ["B0005", "B0006", "B0007", "B0018"]:
    data = sio.loadmat(HERE / f"pipeline_{cell}.mat")
    out_path = OUT_DIR / f"{cell}.npz"
    np.savez_compressed(
        out_path,
        X=np.asarray(data["X"], dtype=np.float32),
        stride=int(data["stride"].item()),
        I=np.asarray(data["I"], dtype=np.float64).ravel(),
        C=np.asarray(data["C"], dtype=np.float64).ravel(),
        V=np.asarray(data["V"], dtype=np.float64).ravel(),
        gate=np.asarray(data["gate"]).ravel().astype(bool),
        SOC_0=float(data["SOC_0"].item()),
        alpha_ocv=float(data["alpha_ocv"].item()),
        eq_s=np.asarray(data["eq_s"], dtype=np.float64).ravel(),
        soc_eq_s=np.asarray(data["soc_eq_s"], dtype=np.float64).ravel(),
        SOC_true=np.asarray(data["SOC_true"], dtype=np.float64).ravel(),
        dt=float(data["dt"].item()),
        dSOC_scale=float(data["dSOC_scale"].item()),
        official_rmse=float(data["official_rmse"].item()),
    )
    print(f"{cell}: {out_path.stat().st_size / 1024:.1f} KB")
