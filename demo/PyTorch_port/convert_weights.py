import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

HERE = Path(__file__).resolve().parent
APP_DIR = HERE.parent
sys.path.insert(0, str(APP_DIR))

from model import HAPITTransformer  # noqa: E402  (the shipped app's model.py)

WEIGHTS_PATH = HERE / "weights_raw.mat"
OUT_PATH = APP_DIR / "data" / "hapit_pytorch.pt"


def to_tensor(arr):
    return torch.tensor(np.asarray(arr), dtype=torch.float32)


def assign_linear(linear, weights_raw, weight_key, bias_key):
    w = to_tensor(weights_raw[weight_key])
    b = to_tensor(weights_raw[bias_key]).squeeze()
    assert w.shape == linear.weight.shape, f"{weight_key}: raw {w.shape} vs module {linear.weight.shape}"
    assert b.shape == linear.bias.shape, f"{bias_key}: raw {b.shape} vs module {linear.bias.shape}"
    linear.weight.data.copy_(w)
    linear.bias.data.copy_(b)


def assign_layernorm(ln, weights_raw, scale_key, offset_key):
    scale = to_tensor(weights_raw[scale_key]).squeeze()
    offset = to_tensor(weights_raw[offset_key]).squeeze()
    assert scale.shape == ln.weight.shape, f"{scale_key}: raw {scale.shape} vs module {ln.weight.shape}"
    assert offset.shape == ln.bias.shape, f"{offset_key}: raw {offset.shape} vs module {ln.bias.shape}"
    ln.weight.data.copy_(scale)
    ln.bias.data.copy_(offset)


def assign_conv(conv, weights_raw, weight_key, bias_key):
    # MATLAB convolution1dLayer weight layout: [FilterSize, InChannels, OutChannels]
    # PyTorch Conv1d weight layout:            [OutChannels, InChannels, FilterSize]
    w_matlab = np.asarray(weights_raw[weight_key])
    w = to_tensor(w_matlab.transpose(2, 1, 0))
    b = to_tensor(weights_raw[bias_key]).squeeze()
    assert w.shape == conv.weight.shape, f"{weight_key}: converted {w.shape} vs module {conv.weight.shape}"
    assert b.shape == conv.bias.shape, f"{bias_key}: raw {b.shape} vs module {conv.bias.shape}"
    conv.weight.data.copy_(w)
    conv.bias.data.copy_(b)


def assign_attention(att, weights_raw, prefix):
    assign_linear(att.query, weights_raw, f"{prefix}__QueryWeights", f"{prefix}__QueryBias")
    assign_linear(att.key, weights_raw, f"{prefix}__KeyWeights", f"{prefix}__KeyBias")
    assign_linear(att.value, weights_raw, f"{prefix}__ValueWeights", f"{prefix}__ValueBias")
    assign_linear(att.out, weights_raw, f"{prefix}__OutputWeights", f"{prefix}__OutputBias")


def main():
    raw = sio.loadmat(WEIGHTS_PATH)
    model = HAPITTransformer()
    model.eval()

    assign_conv(model.conv_k3, raw, "conv_k3__Weights", "conv_k3__Bias")
    assign_conv(model.conv_k5, raw, "conv_k5__Weights", "conv_k5__Bias")
    assign_conv(model.conv_k9, raw, "conv_k9__Weights", "conv_k9__Bias")
    assign_layernorm(model.ln_cnn, raw, "ln_cnn__Scale", "ln_cnn__Offset")
    assign_linear(model.proj, raw, "proj__Weights", "proj__Bias")

    assign_attention(model.self_att_1, raw, "self_att_1")
    assign_layernorm(model.ln_att_1, raw, "ln_att_1__Scale", "ln_att_1__Offset")
    assign_linear(model.ffn1_1, raw, "ffn1_1__Weights", "ffn1_1__Bias")
    assign_linear(model.ffn2_1, raw, "ffn2_1__Weights", "ffn2_1__Bias")
    assign_layernorm(model.ln_ffn_1, raw, "ln_ffn_1__Scale", "ln_ffn_1__Offset")

    assign_attention(model.self_att_2, raw, "self_att_2")
    assign_layernorm(model.ln_att_2, raw, "ln_att_2__Scale", "ln_att_2__Offset")
    assign_linear(model.ffn1_2, raw, "ffn1_2__Weights", "ffn1_2__Bias")
    assign_linear(model.ffn2_2, raw, "ffn2_2__Weights", "ffn2_2__Bias")
    assign_layernorm(model.ln_ffn_2, raw, "ln_ffn_2__Scale", "ln_ffn_2__Offset")

    assign_linear(model.net_out, raw, "net_out__Weights", "net_out__Bias")

    torch.save(model.state_dict(), OUT_PATH)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"All 44 parameter groups assigned with matching shapes.")
    print(f"Total PyTorch parameters: {total_params:,} (README reports 80,866 for the original)")
    print(f"Saved state_dict to {OUT_PATH}")


if __name__ == "__main__":
    main()
