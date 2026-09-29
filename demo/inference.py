import numpy as np
import torch
from scipy.interpolate import interp1d


def run_inference(model, cell_data: dict) -> dict:
    """Runs the ported model on a cell's real windowed input, then applies
    the exact Coulomb-counting + gated-anchoring reconstruction confirmed
    from src/matlab/HA_PIT_test_v8.m (evaluateContinuous/reconFromCorr).
    Returns predicted/true SOC (%) and the live-computed metrics, alongside
    the official MATLAB reference RMSE for direct comparison."""
    X = cell_data["X"]  # [5, numWindows, seqLen]
    stride = cell_data["stride"]
    I, C, V, gate = cell_data["I"], cell_data["C"], cell_data["V"], cell_data["gate"]
    SOC_0, alpha_ocv = cell_data["SOC_0"], cell_data["alpha_ocv"]
    eq_s, soc_eq_s = cell_data["eq_s"], cell_data["soc_eq_s"]
    SOC_true, dt, dSOC_scale = cell_data["SOC_true"], cell_data["dt"], cell_data["dSOC_scale"]

    num_features, num_windows, seq_len = X.shape
    x_torch = torch.tensor(X.transpose(1, 0, 2), dtype=torch.float32)
    # Chunked: attention scores are windows x heads x 480 x 480 floats, so all
    # 327 windows at once peaks at ~2.6 GB and gets the process OOM-killed on
    # Streamlit Community Cloud; 16 at a time peaks at ~0.4 GB.
    with torch.no_grad():
        net_out = torch.cat([model(chunk) for chunk in torch.split(x_torch, 16)]).numpy()  # [windows, 2, seqLen]
    corr_mat = net_out[:, 0, :] * dSOC_scale

    total_pts = (num_windows - 1) * stride + seq_len
    corr_seq = np.zeros(total_pts)
    corr_seq[:seq_len] = corr_mat[0]
    for b in range(1, num_windows):
        g1 = b * stride
        g2 = min(g1 + seq_len, total_pts)
        corr_seq[g1:g2] = corr_mat[b, : g2 - g1]

    N = min(total_pts, len(SOC_true))
    corr_seq[0] = 0.0

    ocv_lookup = interp1d(eq_s, soc_eq_s, kind="linear", fill_value="extrapolate")
    soc_ocv = np.clip(ocv_lookup(V[:N]), 0.0, 1.0)
    inc = (I[:N] * dt) / C[:N] + corr_seq[:N]

    SOC = np.zeros(N)
    SOC[0] = SOC_0
    for i in range(1, N):
        soc_next = SOC[i - 1] + inc[i]
        if gate[i]:
            soc_next = (1 - alpha_ocv) * soc_next + alpha_ocv * soc_ocv[i]
        SOC[i] = min(max(soc_next, 0.0), 1.0)

    pred_pct = SOC * 100
    gt_pct = SOC_true[:N] * 100
    err = pred_pct - gt_pct

    return {
        "pred_pct": pred_pct,
        "gt_pct": gt_pct,
        "error": err,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "max_error": float(np.max(np.abs(err))),
    }
