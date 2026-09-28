"""End-to-end parity on the code and data the app actually ships: runs
inference.run_inference with the shipped model on data/try_it/<cell>.npz and
compares against MATLAB's official continuous reconstruction (pipeline_<cell>.mat,
from matlab/export_all_cells.m)."""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
sys.path.insert(0, str(APP_DIR))

from inference import run_inference  # noqa: E402
from model import HAPITTransformer  # noqa: E402

model = HAPITTransformer()
model.load_state_dict(torch.load(APP_DIR / "data" / "hapit_pytorch.pt", map_location="cpu"))
model.eval()

for cell in ["B0005", "B0006", "B0007", "B0018"]:
    npz = np.load(APP_DIR / "data" / "try_it" / f"{cell}.npz")
    cell_data = {k: npz[k] for k in npz.files}
    for k in ("stride",):
        cell_data[k] = int(cell_data[k])
    for k in ("SOC_0", "alpha_ocv", "dt", "dSOC_scale", "official_rmse"):
        cell_data[k] = float(cell_data[k])

    result = run_inference(model, cell_data)

    official = sio.loadmat(HERE / f"pipeline_{cell}.mat")
    official_pred = np.asarray(official["official_pred_pct"]).ravel()[: len(result["pred_pct"])]
    max_diff = np.max(np.abs(result["pred_pct"] - official_pred))

    print(
        f"{cell}: live RMSE {result['rmse']:.6f}% | official {cell_data['official_rmse']:.6f}% | "
        f"RMSE diff {abs(result['rmse'] - cell_data['official_rmse']):.2e} pp | "
        f"max per-step diff {max_diff:.2e} pp"
    )
