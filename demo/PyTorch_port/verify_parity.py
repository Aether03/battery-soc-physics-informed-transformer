"""Raw-network parity: fixed-seed input through the MATLAB net (reference_io.mat,
from matlab/run_reference_output.m) vs. the shipped PyTorch model."""
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
sys.path.insert(0, str(APP_DIR))

from model import HAPITTransformer  # noqa: E402

ref = sio.loadmat(HERE / "reference_io.mat")
X = np.asarray(ref["X"])  # [5, 1, 480] -- MATLAB C,B,T
Y_matlab = np.asarray(ref["Y"])  # [2, 1, 480]

model = HAPITTransformer()
model.load_state_dict(torch.load(APP_DIR / "data" / "hapit_pytorch.pt", map_location="cpu"))
model.eval()

x_torch = torch.tensor(X, dtype=torch.float32).permute(1, 0, 2)  # [1, 5, 480]
with torch.no_grad():
    y_torch = model(x_torch).numpy()  # [1, 2, 480]

diff = np.abs(y_torch - Y_matlab.transpose(1, 0, 2))
print(f"Max abs diff:  {diff.max():.8f}")
print(f"Mean abs diff: {diff.mean():.8f}")
