from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from data_loader import (
    load_hpo_trials,
    load_metrics,
    load_ported_model,
    load_predictions,
    load_training_curves,
    load_try_it_cell,
)
from inference import run_inference

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(page_title="HA-PIT: Battery SOC Estimation", layout="wide")

# Validated categorical palette (dark-mode steps), fixed slot order per series identity.
BLUE, ORANGE, AQUA, RED, MUTED = "#3987e5", "#d95926", "#199e70", "#e66767", "#c3c2b7"



# Every tab body runs on every rerun, so without this cache any widget change
# anywhere in the app would redo the full forward pass + 78k-step recursion.
@st.cache_data(show_spinner="Running live inference...")
def live_inference(cell: str) -> dict:
    return run_inference(load_ported_model(), load_try_it_cell(cell))


metrics = load_metrics()

st.title("HA-PIT: Ageing-Aware Physics-Informed Transformer")
st.caption(
    "Lithium-ion battery state-of-charge estimation, trained on the NASA PCoE "
    "Li-ion Battery Aging Dataset. Final-year project, Universiti Teknologi MARA "
    "Shah Alam, 2026. This app is an interactive companion to the project's "
    "[GitHub repo](https://github.com/Aether03/battery-soc-physics-informed-transformer)."
)

tab_overview, tab_results, tab_hpo, tab_bias, tab_how, tab_about, tab_try_it = st.tabs([
    "Overview", "Results Explorer", "HPO Comparison", "Bias Robustness", "How It Works", "About", "Try It",
])

# ---------------------------------------------------------------- Overview
with tab_overview:
    h = metrics["headline"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test RMSE", f"{h['test_rmse_pct']:.2f}%", help=f"On {h['test_cell']}, held out from training and model selection")
    c2.metric("Full-life RMSE", f"{h['full_life_rmse_pct']:.2f}%", help="Across a cell's complete 338-cycle life, fresh to end-of-life")
    c3.metric("Parameters", f"{h['params']:,}")
    c4.metric("Training time", h["train_time"], help=h["hardware"])

    st.markdown(
        """
A multi-scale CNN feeds two self-attention blocks; a five-term physics-informed
loss built on a first-order Thevenin equivalent circuit keeps it honest. Rather
than predicting absolute state-of-charge outright, the network predicts a small
**residual correction** on top of an exact Coulomb-counting anchor — which
removes the long-horizon integration drift that a bare neural predictor would
accumulate over a multi-hour sequence.

Trained end to end in under two hours, on a laptop CPU, with no GPU.
        """
    )

    st.subheader("Headline results")
    st.dataframe(metrics["results_table"], hide_index=True, use_container_width=True)
    st.caption(
        "Continuous reconstruction: the model walks the whole sequence in one pass, "
        "with no per-window resets — this is the number that matters for deployment."
    )

    st.subheader("Stress tests beyond the main split")
    st.dataframe(metrics["stress_tests"], hide_index=True, use_container_width=True)
    st.caption(
        "The UDDS row is the honest generalisation boundary: trained on constant-current "
        "NASA cycling, the model loses an order of magnitude on an automotive drive cycle."
    )

# ---------------------------------------------------------- Results Explorer
with tab_results:
    split_label_to_key = {"Test (held out) — B0018": "test", "Validation — B0007": "val", "Train — B0005+B0006": "train"}
    split_label = st.selectbox("Split", list(split_label_to_key.keys()), index=0)
    split_key = split_label_to_key[split_label]

    df = load_predictions(split_key)
    row = next(r for r in metrics["results_table"] if split_key in r["split"].lower())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("RMSE", f"{row['rmse']:.4f}%")
    m2.metric("MAE", f"{row['mae']:.4f}%")
    m3.metric("Max error", f"{row['max_error']:.4f}%")
    m4.metric("R²", f"{row['r2']:.4f}")

    st.caption(
        "X-axis is timestep index within the continuous-reconstruction evaluation "
        "(concatenated 480-step windows), not wall-clock seconds — the underlying "
        "windows are not individually timestamped in the saved evaluation output."
    )

    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=df["index"], y=df["true_soc"], mode="lines", name="True SOC", line=dict(width=1, color=BLUE)))
        fig.add_trace(go.Scattergl(x=df["index"], y=df["pred_soc"], mode="lines", name="Predicted SOC", line=dict(width=1, color=ORANGE)))
        fig.update_layout(title="SOC trajectory: predicted vs true", xaxis_title="Timestep index", yaxis_title="SOC (%)", height=400, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=df["index"], y=df["error"].abs(), mode="lines", name="Absolute error", line=dict(width=1, color=RED)))
        fig.add_hline(y=row["rmse"], line_dash="dash", annotation_text=f"RMSE {row['rmse']:.3f}%", line_color=MUTED)
        fig.add_hline(y=row["mae"], line_dash="dot", annotation_text=f"MAE {row['mae']:.3f}%", line_color=MUTED)
        fig.update_layout(title="Absolute error over time", xaxis_title="Timestep index", yaxis_title="|Error| (percentage points)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        slope, intercept = np.polyfit(df["true_soc"], df["pred_soc"], 1)
        x_line = np.array([df["true_soc"].min(), df["true_soc"].max()])
        fig = go.Figure()
        fig.add_trace(go.Scattergl(x=df["true_soc"], y=df["pred_soc"], mode="markers", name="Predictions", marker=dict(size=2, opacity=0.3, color=BLUE)))
        fig.add_trace(go.Scatter(x=x_line, y=slope * x_line + intercept, mode="lines", name=f"Fit (slope={slope:.3f})", line=dict(color=MUTED, dash="dash")))
        fig.update_layout(title="Predicted vs true SOC", xaxis_title="True SOC (%)", yaxis_title="Predicted SOC (%)", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df["error"], nbinsx=80, name="Error", marker_color=BLUE))
        fig.update_layout(title="Error distribution", xaxis_title="Error (percentage points)", yaxis_title="Count", height=400)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("Training convergence"):
        curves = load_training_curves()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=curves["epoch"], y=curves["val_rmse_continuous"], mode="lines", name="Validation RMSE (continuous)", line=dict(color=BLUE)))
        fig.add_trace(go.Scatter(x=curves["epoch"], y=curves["val_rmse_windowed"], mode="lines", name="Validation RMSE (windowed-oracle)", line=dict(dash="dot", color=ORANGE)))
        fig.add_vline(x=curves["best_epoch"], line_dash="dash", annotation_text=f"Best epoch {curves['best_epoch']}", line_color=MUTED)
        fig.update_layout(xaxis_title="Epoch", yaxis_title="Validation RMSE (%)", height=350)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The gap between the continuous and windowed-oracle lines localises where error "
            "comes from: the windowed metric measures correction quality alone, the continuous "
            "metric measures the full deployed pipeline including accumulated drift."
        )

# ------------------------------------------------------------- HPO Comparison
with tab_hpo:
    st.subheader("Three automated searches lost to hand tuning")
    st.markdown(
        """
Three hyperparameter searches ran against one shared 8-parameter space, roughly
100 hours of compute combined: ASHA (successive halving), BOHB (Hyperband +
refinement), and Bayesian optimisation with gradient-surgery (PCGrad). None of
them beat the hand-tuned baseline.
        """
    )

    trials = load_hpo_trials()
    method_colors = {"ASHA": BLUE, "BOHB": ORANGE, "Bayesian optimisation + PCGrad": AQUA}
    fig = go.Figure()
    for method, group in trials.groupby("method"):
        group = group.sort_values("trial_number")
        best_so_far = group["rmse"].cummin()
        fig.add_trace(go.Scatter(x=group["trial_number"], y=best_so_far, mode="lines+markers", name=method, marker=dict(size=4, color=method_colors[method]), line=dict(color=method_colors[method])))
    fig.update_layout(
        title="Best RMSE found so far, by trial number",
        xaxis_title="Trial number", yaxis_title="Best RMSE so far (%, search-harness metric)",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "These trial-level RMSE figures use the search harness's own settings (shorter "
        "sequence length, an earlier four-term loss) and are not directly comparable to the "
        "0.41% production number below — see the fair re-run comparison."
    )

    st.subheader("Search summary")
    st.dataframe(metrics["hpo_comparison"]["search_rows"], hide_index=True, use_container_width=True)

    st.subheader("Fair comparison: best config re-run at full fidelity")
    st.dataframe(metrics["hpo_comparison"]["fair_comparison_rows"], hide_index=True, use_container_width=True)
    st.markdown(
        """
**Manual tuning won.** The residual error the searches were fighting was
accumulated integration drift, not a hyperparameter the search space could
reach — the fix was structural: a drift-targeted loss term, a model-selection
metric that could actually see drift, and a learning-rate schedule decoupled
from the epoch budget.
        """
    )

# ------------------------------------------------------------- Bias Robustness
with tab_bias:
    st.subheader("Bias robustness: the result that didn't go as planned")
    st.markdown(
        """
A constant current-sensor offset is the realistic failure mode for anything
built on Coulomb counting — the bias integrates and the estimate walks away.
If the network is genuinely fusing voltage evidence rather than just
integrating current with extra steps, an injected bias should hurt it less
than it hurts pure Coulomb counting. The bias was injected into both the
integration current and the network's own input channel, so the network had
a real chance to counteract it.
        """
    )

    rows = metrics["bias_ablation"]["rows"]
    biases = [f"{r['injected_bias_a']:.2f} A" for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Pure Coulomb counting", x=biases, y=[r["pure_coulomb"] for r in rows], marker_color=BLUE))
    fig.add_trace(go.Bar(name="HA-PIT V8.6 (best clean accuracy)", x=biases, y=[r["hapit_v86"] for r in rows], marker_color=ORANGE))
    fig.add_trace(go.Bar(name="HA-PIT, HPO config", x=biases, y=[r["hapit_hpo_config"] for r in rows], marker_color=AQUA))
    fig.update_layout(barmode="group", title="RMSE under injected current-sensor bias", xaxis_title="Injected bias", yaxis_title="RMSE (%)", height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
**V8.6 — the configuration with the best clean accuracy — does not beat plain
Coulomb counting under bias.** Near-parity, not a win; that was the target and
it wasn't met. The configuration found by the Bayesian-optimisation + PCGrad
search *does* beat it — 0.83 percentage points better at 0.05 A, 0.87 points
better at 0.10 A — paid for with 0.24 points of clean accuracy. The two
configurations sit at different points on a clean-accuracy vs. sensor-robustness
trade-off; which one you'd deploy depends on how much you trust your current
sensor. Not the finding this experiment set out to produce, and arguably more
useful than the one it was aiming at.
        """
    )

# ------------------------------------------------------------- How It Works
with tab_how:
    st.subheader("Residual formulation")
    st.markdown(
        """
The network never predicts SOC directly, and never predicts the per-step
increment either — it predicts a small correction on top of an exact,
non-learned Coulomb-counting term. Anchoring the output to real physics this
way means there's nothing for standard MSE training to shrink toward zero;
a zero-correction network reproduces the Coulomb-counting labels almost
exactly by construction.
        """
    )

    arch_path = DATA_DIR / "figures" / "architecture.png"
    if arch_path.exists():
        st.image(str(arch_path), caption="Model architecture", use_container_width=True)

    st.subheader("Ageing awareness")
    st.markdown(
        """
Capacity isn't treated as a constant. The pipeline tracks contiguous
active-current runs and recalibrates the Coulomb-counting denominator as the
cell fades, rather than assuming a fixed nameplate capacity. Internal
resistance is modelled with an Arrhenius (temperature) and power-law
(capacity-fade) relationship, so the physics constraint stays valid even near
end-of-life, where a fresh-cell equivalent-circuit model would be badly wrong.
        """
    )

    st.subheader("Five-term physics-informed loss")
    st.dataframe(metrics["loss_terms"], hide_index=True, use_container_width=True)
    st.caption(
        "L_drift is the term that mattered most: squaring the sum of unanchored deviations, "
        "rather than averaging squared per-step error, is what makes a bias too small to see "
        "on any single step show up clearly in the loss."
    )

# ------------------------------------------------------------- About
with tab_about:
    st.markdown(
        """
### About this project

**HA-PIT** (Ageing-Aware / Hysteresis-Aware Physics-Informed Transformer) is a
final-year project at Universiti Teknologi MARA, Shah Alam, supervised by
Dr Masoud Ahmadipour, Faculty of Electrical Engineering.

- **Source code, full results, and technical writeups:**
  [github.com/Aether03/battery-soc-physics-informed-transformer](https://github.com/Aether03/battery-soc-physics-informed-transformer)
- **Training data:** NASA Ames Prognostics Center of Excellence, Battery Data Set
  (B. Saha and K. Goebel, 2007) — public domain, no explicit licence tag; attribution expected.
- **Licence:** source code under the linked repo is MIT; documentation and figures are CC BY 4.0.

This app presents the project's own published, verified results — every number
shown here is sourced from the repository's saved evaluation output or its
README, not recomputed or estimated for this demo.
        """
    )

# ------------------------------------------------------------------ Try It
with tab_try_it:
    st.subheader("Live inference: the actual ported model")
    st.markdown(
        """
This tab runs a real, live forward pass through a **PyTorch port** of the
trained MATLAB model — not a lookup of pre-computed results. The original
checkpoint's weights were exported from MATLAB and converted; the port's
output matched the original to within 0.000001 percentage points of RMSE on
real held-out data. Full methodology, verification scripts, and planning
documents: [`demo/PyTorch_port/`](https://github.com/Aether03/battery-soc-physics-informed-transformer/tree/main/demo/PyTorch_port) in the project repo.
        """
    )

    cell = st.selectbox(
        "Cell", ["B0018", "B0005", "B0006", "B0007"], index=0,
        help="B0018 is the held-out test cell; B0005/B0006 were used for training, B0007 for validation.",
    )

    cell_data = load_try_it_cell(cell)
    result = live_inference(cell)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Live RMSE", f"{result['rmse']:.4f}%")
    m2.metric("Live MAE", f"{result['mae']:.4f}%")
    m3.metric("Live Max error", f"{result['max_error']:.4f}%")
    m4.metric(
        "Published reference RMSE", f"{cell_data['official_rmse']:.4f}%",
        help="The original MATLAB model's RMSE on this same cell, for direct comparison.",
    )

    fig = go.Figure()
    idx = np.arange(len(result["pred_pct"]))
    fig.add_trace(go.Scattergl(x=idx, y=result["gt_pct"], mode="lines", name="True SOC", line=dict(width=1, color=BLUE)))
    fig.add_trace(go.Scattergl(x=idx, y=result["pred_pct"], mode="lines", name="Predicted SOC (live, PyTorch)", line=dict(width=1, color=ORANGE)))
    fig.update_layout(
        title=f"Live PyTorch inference — {cell}", xaxis_title="Timestep index",
        yaxis_title="SOC (%)", height=420, legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Input is drawn from the project's existing, already-validated cell data — not a "
        "free-form upload. The model hasn't yet been shown to generalize beyond the NASA "
        "cycling data it was trained on (see the README's UDDS stress-test result), so "
        "arbitrary visitor-supplied input isn't offered yet."
    )
