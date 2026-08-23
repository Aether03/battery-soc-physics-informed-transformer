%% HA_PIT_v8.m  (V8.6 — LR-HORIZON FIX + SENSOR-BIAS AUGMENTATION)
% ════════════════════════════════════════════════════════════════════════
%  HYSTERESIS-AWARE PHYSICS-INFORMED TRANSFORMER  —  INCREMENT PIPELINE
% ════════════════════════════════════════════════════════════════════════
%
%  WHY V8.6 (on top of V8.5's run: 1.11 / 0.97 / 0.98 % continuous RMSE)
%  ----------------------------------------------------------------------
%  V8.5 cut val RMSE 3.05% → 0.97% but missed the <0.5% target. Its console
%  shows exactly why, and one deeper problem:
%
%   FIX 1 — LR ANNEAL HORIZON (cfg.cosineEpochs = 150).
%     The cosine period was numEpochs=400 but patience stopped the run at 115,
%     so LR never fell below 4.3e-3. ContRMSE oscillated 1–8.6% epoch-to-epoch
%     (pure high-LR bouncing) and the selected epoch 35 was a lucky bounce.
%     Annealing over a horizon the run actually reaches lets the drift bias
%     settle to a real optimum instead of being re-randomised every epoch.
%
%   FIX 2 — lambda_drift 10 → 30 (ladder step 1).
%     Residual bias at V8.5's best epoch was ~3.8e-6/step (DriftLoss≈3.3e-5);
%     <0.5% needs <~1.2e-6/step.
%
%   FIX 3 — P4 WAS MEASURING THE WRONG THING; NOW MADE REAL.
%     (a) The V8.5 ablation added the bias only to the INTEGRATION current and
%         fed the network CLEAN inputs — the network cannot counteract a bias
%         it never observes. Fixed: the bias now also corrupts input channel 2.
%     (b) The network had never been TRAINED to exploit the voltage-vs-current
%         mismatch. New sensor-bias augmentation (cfg.bias_aug_p/bias_aug_max)
%         corrupts the sensor consistently (input ch2 + physics increment +
%         ECM IR term) on a random subset of training windows while labels
%         stay clean, so the data loss teaches corr ≈ −b·dt/C conditioned on
%         the mismatch. The drift loss is made bias-aware so the two
%         objectives cooperate instead of fighting.
%     EXPECTATION: post-V8.6, the P4 table should show Network < Pure-Coulomb
%     at bias 0.05/0.10 A — the genuine robustness demonstration for the FYP.
%
%  V8.5 BACKGROUND (retained)
%  ----------------------------------------------------------------------
%  V8.4's remaining validation error is a pure integration-drift sawtooth:
%  |error| ramps to ~10% across each long UNANCHORED charge segment and
%  collapses at the next settled rest. Because recon(corr=0) == GT exactly,
%  ALL of that error is the accumulated network correction — a per-step bias
%  of only ~2.6e-5, which is:
%    • invisible to L_Ah   (10·mean(corr²) ≈ 7e-9 at that bias), and
%    • invisible to the early-stopping metric (windowed RMSE cannot see
%      accumulation: V8.4's chosen epoch had 0.39% windowed / 3.05% continuous).
%  Getting <0.5% val therefore needs the bias cut ~20×, via two changes:
%
%   CHANGE 1 — UNANCHORED-DRIFT LOSS (cfg.lambda_drift).
%     L_drift = mean_windows( sum_t[(1-gate)·corr] )².  On non-gated steps the
%     label increment is exactly I·dt/C, so the target correction there is
%     exactly zero and the per-window SUM of corrections is pure drift-rate.
%     Squaring the SUM (not the mean of squares) rescales a 2.6e-5 bias into
%     ~1.6e-4 loss — the same order as L_data — so the optimiser can act on it.
%     Gated steps are excluded: there the correction legitimately learns the
%     label's OCV anchor pull.
%
%   CHANGE 2 — MODEL SELECTION ON THE CONTINUOUS VAL METRIC.
%     The per-epoch val corrections were already being predicted; V8.5 also
%     runs the full continuous reconstruction on them (<1 s, shared helper
%     reconFromCorr with evaluateContinuous) and early-stops/selects on THAT.
%     The selection metric and the deployment metric are now the same number.
%
%  HONEST NOTE: because recon(corr=0)==GT, this metric is trivially minimised
%  by a network that outputs nothing. The scientific value of the correction
%  channel is demonstrated by the P4 current-bias ablation (network beats pure
%  Coulomb under injected sensor bias) — keep reporting it alongside the RMSE.
%
%  V8.4 STRUCTURAL FIXES (retained, see below): matched forward-only gated
%  reconstruction; relaxation-gated anchoring; residual increments + endpoint
%  loss.
% ════════════════════════════════════════════════════════════════════════
%
%  V8.4 BACKGROUND (retained)
%  ----------------------------------------------------------------------
%  V8.3 gave ~7–8% continuous test RMSE with low MAE (~3%) and huge localized
%  spikes (startup 24–54%, discharge cusps 43–47%) and slope ~0.92. Data-
%  grounded diagnosis traced this to THREE structural faults, all fixed here:
%
%   FAULT 1 — reconstruction anchored DIFFERENTLY from the labels.
%     The GT integrates forward from SOC_0 with a gentle OCV pull at rests; the
%     V8.3 eval hard-anchored 30 s into the first rest (terminal V still ~0.5 V
%     from equilibrium) and integrated backward. Mismatch ⇒ startup spike.
%   FIX: evaluateContinuous now reconstructs FORWARD from the SAME cont.SOC_0
%        and re-anchors with the SAME settled-rest gate (cont.gate) and SAME
%        alpha as prepareBatteryDataV9. A zero-correction model reproduces the
%        labels to ~1e-9 (verified), so the startup spike is structurally gone.
%
%   FAULT 2 — anchoring on NON-RELAXED rest voltage.
%     Anchoring at every |I|<thr step read voltage up to ~0.5 V from equilibrium,
%     biasing SOC toward mid-range (the slope-0.92 compression).
%   FIX: RELAXATION-GATED anchoring — a rest re-anchors only if sustained
%        (>= cfg.settle_len) AND settled (trailing |dV/dt| < cfg.dvdt_thr).
%        Applied identically to labels and reconstruction.
%
%   FAULT 3 — the network learned the FULL increment, attenuated by MSE.
%     Learning I·dt/C from scratch under MSE shrinks it a few %, so over the
%     ~5600-step unanchored startup the drift compounds to several %.
%   FIX: RESIDUAL increments. The network predicts only a CORRECTION on top of
%        the EXACT physics increment:  dSOC = I·dt/C + (net·dSOC_scale).
%        The Coulomb baseline is built in, so there is nothing to attenuate; the
%        network only refines sensor-bias / capacity / drift. An ENDPOINT loss
%        additionally penalises any accumulated correction drift per window.
%
%  WHAT THE NETWORK STILL DOES (not "just Coulomb counting"):
%    It fuses the voltage reading to correct current-sensor bias, capacity
%    error and drift. The P4 current-bias ablation at the end demonstrates this
%    directly (network beats pure Coulomb under an injected current bias).
%
%  METRICS REPORTED:
%    Continuous (deployment) RMSE  AND  windowed-oracle RMSE, for all splits.
%    A large gap ⇒ residual drift/anchoring; small gap ⇒ intrinsic to the
%    increment correction. Plus a per-segment breakdown to locate any residual.
%
%  FIGURES: exactly 5 windows — SOC Trajectory, Error Distribution, Regression,
%    Absolute Error (each Train/Val/Test stacked), and Training Convergence.
%
%  RETAINED: empirical poly5 OCV, R0=0.080 Ω + cycle-aging, SOC-stratified loss
%    weights, voltage clipping, zero-phase filtering, shared scalers, the CNN→
%    transformer architecture, and SOC_OCV_INSTANT as the 5th fusion feature.
%
%  DEPENDENCY: the custom PositionalEncodingLayer must be on the MATLAB path.
%  NOTE: labels shift slightly vs V8.3 (gated anchoring) — before/after RMSE is
%        therefore not measured against an identical target; prepareBatteryDataV9
%        prints the max label shift so this is visible.
% ════════════════════════════════════════════════════════════════════════

clear; clc; rng(42);


%% ══════════════════════════════════════════════════════════════════════
%%  0.  CONFIGURATION
%% ══════════════════════════════════════════════════════════════════════
cfg.numFeatures      = 5;        % [V, I, T, C_soh, SOC_ocv_instant]
cfg.numFilters       = 48;       % V8.2: was 32 — extra capacity for OCV-correction channel
cfg.filterSize       = 5;
cfg.numHeads         = 4;
cfg.attentionDim     = 64;
cfg.ffnExpand        = 2;
cfg.dropoutRate      = 0.05;     % V8.2: was 0.10 — reduced; network stable, less regularisation needed
cfg.seqLen           = 480;      % V8.2: was 240 — doubled for more within-segment context
                                     %   480s = 17%% of 2800s discharge vs 8.5%% before
cfg.pEncScale        = 0.1;

cfg.numEpochs        = 400;
cfg.warmupEpochs     = 20;       % V8.2: was 30 — faster warm-up, start learning sooner
% V8.6 — LR ANNEAL HORIZON. The V8.5 run early-stopped at epoch 115 while the
% cosine period was numEpochs=400, so LR never fell below 4.3e-3: ContRMSE
% oscillated 1–8.6% epoch-to-epoch and epoch 35 (0.97%) was a lucky bounce,
% not a converged optimum. Annealing over a horizon the run actually reaches
% lets the drift bias SETTLE instead of being re-randomised every epoch.
cfg.cosineEpochs     = 150;      % LR reaches minLR here; stays there after
cfg.initLR           = 5e-3;
cfg.minLR            = 1e-5;
cfg.gradClip         = 1.0;
cfg.k_blend          = 10;
cfg.curriculumEpochs = 15;
cfg.miniBatchSize    = 2048;

cfg.dt               = 1;
cfg.C_nom            = 6664;
cfg.H_max            = 0.015;
cfg.H_gamma          = 1e-4;

% --- ECM / R0 (data-driven) ---
cfg.R_ref            = 0.080;       % measured ~0.078–0.103 Ω
cfg.R0_alpha         = 0.50;        % R0 cycle-aging exponent: R0 × (C_nom/C)^α
cfg.T_ref_K          = 297.15;
cfg.E_a_R            = 20000 / 8.314;

% --- empirical OCV (fitted from B0005+B0006) ---
cfg.ocv_dchg = [24.009042, -65.838561, 68.061132, -32.312351, 7.331077, 2.749715];
cfg.ocv_chg  = [ 5.340997, -15.748829, 16.803693,  -8.069274, 2.075416, 3.801065];

% --- V8 increment-prediction parameters (V8.4: RESIDUAL increments) ---
% The network no longer predicts the FULL per-step increment. It predicts a
% small CORRECTION on top of the exact physics increment I·dt/C:
%     dSOC[k] = I[k]·dt/C[k]  +  (network_ch1 · dSOC_scale)
% The Coulomb baseline is therefore built in (matching the labels exactly), so
% a zero-correction model reproduces the ground truth to ~0% and the network
% only has to refine sensor-bias / capacity / drift. This removes the
% increment attenuation (slope 0.92) and the startup drift of the V8.3 baseline.
cfg.dSOC_scale       = 1e-3;        % channel-1 raw output → per-step CORRECTION
cfg.ocv_corr_scale   = 0.1;         % channel-2 raw output → OCV correction (V)
cfg.alpha_ocv        = 0.20;        % OCV re-anchor gain at SETTLED rests
                                           % Must match prepareBatteryDataV9 cfg.alpha_ocv exactly
cfg.I_rest_thresh    = 0.05;        % |I| below this (A) ⇒ treated as rest

% --- P2: relaxation-gated anchoring (must match prepareBatteryDataV9) ---
% A rest only re-anchors once it is sustained AND settled (relaxed voltage).
cfg.settle_len       = 120;         % min rest length (s) to qualify for anchoring
cfg.relax_skip       = 30;          % skip the first 30 s of a rest (still relaxing)
cfg.dvdt_thr         = 2e-4;        % trailing |dV/dt| (V/s) below this ⇒ settled

% --- fixed loss weights (residual model) ---
% L_data : weighted SOC RMSE (primary)
% L_Ah   : correction-magnitude regulariser (keeps the correction small so the
%          model stays close to pure Coulomb counting unless voltage demands it)
% L_end  : ENDPOINT drift penalty — punishes accumulated correction error at the
%          window end, which per-step losses do not see (P3)
% L_ECM  : voltage physics (secondary; frozen during curriculum)
cfg.w_data           = 1.0;
cfg.lambda_Ah        = 10.0;        % correction-magnitude penalty (was full-increment 5e3)
cfg.lambda_end       = 0.5;         % endpoint / integral drift penalty
cfg.w_ECM            = 0.5;
% V8.5 — DRIFT loss. The V8.4 baseline's remaining val error (3.05%) is a
% ~2.6e-5/step correction BIAS accumulating over the long UNANCHORED charge
% segments (the sawtooth in the |error| plot). L_Ah (mean corr^2) is ~4 orders
% of magnitude too small to see a bias of that size, and L_end conflates drift
% with legitimate anchor-pull matching in windows that contain gated steps.
% L_drift = mean_B ( sum_t[(1-gate)*corr] )^2 penalises exactly the net drift
% accumulated on NON-anchored steps per window — the quantity that constitutes
% essentially all of the continuous validation error.
cfg.lambda_drift     = 30.0;     % V8.6: 10→30 (ladder step 1). V8.5 landed at
                                 % 0.97% val with residual bias ~3.8e-6/step;
                                 % <0.5% needs <~1.2e-6, so push ~3× harder.

% V8.6 — SENSOR-BIAS AUGMENTATION. The V8.5 P4 ablation revealed the correction
% channel has never LEARNED to counteract a current-sensor bias (and V8.5's
% ablation fed the network clean inputs anyway — fixed below). With probability
% bias_aug_p per training window, a constant bias b~U(±bias_aug_max) is added to
% the sensor current EVERYWHERE a real biased sensor would appear: the network's
% input channel 2, the physics increment I·dt/C, and the ECM's IR term — while
% the LABELS stay clean. The data loss then demands corr ≈ −b·dt/C conditioned
% on the voltage-vs-current mismatch: this is what makes the P4 robustness
% demonstration real. Set bias_aug_p = 0 to disable.
% (Approximation: input channel 5, SOC_ocv_instant, is not re-biased; its shift
%  is second-order, ≈ −b·R0 ≈ 8 mV at b=0.1 A.)
cfg.bias_aug_p       = 0.5;      % fraction of training windows augmented
cfg.bias_aug_max     = 0.10;     % |bias| upper bound (A) — matches P4 levels

%% ══════════════════════════════════════════════════════════════════════
%%  1.  LOAD DATA
%% ══════════════════════════════════════════════════════════════════════
trainFiles = {'Merged_B0005_Lifecycle_Sample.csv', 'Merged_B0006_Lifecycle_Sample.csv'};
valFile    = 'Merged_B0007_Lifecycle_Sample.csv';
testFile   = 'Merged_B0018_Lifecycle_Sample.csv';

fprintf('Loading Training Data (B0005, B0006)...\n');
dlX_train=[]; dlY_train=[]; dlV_train=[]; dlI_train=[];
dlT_train=[]; dlC_train=[]; dlW_train=[];
train_scalers = [];
train_conts   = {};            % continuous arrays, one per training file
train_nBatch  = [];            % #windows per training file (to slice predictions)

for i = 1 : length(trainFiles)
    fprintf('  Processing %s...\n', trainFiles{i});
    [t_X,t_Y,t_V,t_I,t_T,t_C,t_W,t_cont,ret_sc] = ...
        prepareBatteryDataV9(trainFiles{i}, cfg, train_scalers);

    if isempty(train_scalers)
        train_scalers = ret_sc;
    else
        train_scalers.mu3    = (train_scalers.mu3   + ret_sc.mu3)   / 2;
        train_scalers.sig3   = (train_scalers.sig3  + ret_sc.sig3)  / 2;
        train_scalers.mu_c   = (train_scalers.mu_c  + ret_sc.mu_c)  / 2;
        train_scalers.sig_c  = (train_scalers.sig_c + ret_sc.sig_c) / 2;
        train_scalers.mu_si  = (train_scalers.mu_si + ret_sc.mu_si) / 2;
        train_scalers.sig_si = (train_scalers.sig_si+ ret_sc.sig_si)/ 2;
    end

    dlX_train = cat(2, dlX_train, t_X);
    dlY_train = cat(2, dlY_train, t_Y);
    dlV_train = cat(2, dlV_train, t_V);
    dlI_train = cat(2, dlI_train, t_I);
    dlT_train = cat(2, dlT_train, t_T);
    dlC_train = cat(2, dlC_train, t_C);
    dlW_train = cat(2, dlW_train, t_W);

    train_conts{end+1} = t_cont;            %#ok<SAGROW>
    train_nBatch(end+1) = size(t_X, 2);     %#ok<SAGROW>
end
[~, nBatchTrain, ~] = size(dlX_train);

% V8.6 — the augmentation and the (fixed) P4 ablation need to express a
% physical current bias in the NORMALISED input space of channel 2:
cfg.sig_I = double(train_scalers.sig3(2));
fprintf('Sensor-bias augmentation: p=%.2f, |b|<=%.2f A  (sig_I=%.4f)\n', ...
    cfg.bias_aug_p, cfg.bias_aug_max, cfg.sig_I);

fprintf('\nLoading Validation Data (B0007)...\n');
[dlX_val,dlY_val,dlV_val,dlI_val,dlT_val,dlC_val,~,val_cont,~] = ...
    prepareBatteryDataV9(valFile, cfg, train_scalers);

fprintf('\nLoading Testing Data (B0018)...\n');
[dlX_test,dlY_test,dlV_test,dlI_test,dlT_test,dlC_test,~,test_cont,~] = ...
    prepareBatteryDataV9(testFile, cfg, train_scalers);

fprintf('\nData Split: Train=%d | Val=%d | Test=%d batches\n\n', ...
    nBatchTrain, size(dlX_val,2), size(dlX_test,2));

%% ══════════════════════════════════════════════════════════════════════
%%  1b. PRECOMPUTE PHYSICS CONSTANTS (training windows)
%% ══════════════════════════════════════════════════════════════════════
I_raw_train = extractdata(dlI_train);
T_raw_train = extractdata(dlT_train);
C_raw_train = extractdata(dlC_train);

% Transient mask (down-weights ECM loss during fast current steps)
dI = diff(I_raw_train, 1, 3);
dI = cat(3, zeros(1, nBatchTrain, 1), dI);
transient_mask = 1 ./ (1 + (abs(dI) / 0.5).^2);
dlTransientMask_train = dlarray(transient_mask, 'CBT');

% Hysteresis state
h_data_train = computeHysteresis(I_raw_train, cfg);
dlH_train    = dlarray(h_data_train, 'CBT');

% R0 with cycle-aging
T_K_train    = T_raw_train + 273.15;
R0_arr_train = cfg.R_ref * exp(cfg.E_a_R * (1./T_K_train - 1/cfg.T_ref_K));
R0_age_train = R0_arr_train .* (cfg.C_nom ./ max(C_raw_train, 1000)).^cfg.R0_alpha;
dlR0_train   = dlarray(R0_age_train, 'CBT');

% OCV blend (sigmoid on current sign)
blend_chg_train  = dlarray(1 ./ (1 + exp(-cfg.k_blend .* I_raw_train)), 'CBT');
blend_dchg_train = 1 - blend_chg_train;

% V8.5 — windowed settled-rest gate for the DRIFT loss. Built here from each
% training file's cont.gate (same mask the labels/reconstruction use), so no
% pipeline signature change is needed. gate=1 → anchored step (correction may
% legitimately be nonzero to match the label's OCV pull); gate=0 → unanchored
% step (target correction is exactly zero; net drift here is pure error).
G_train = zeros(1, nBatchTrain, cfg.seqLen, 'single');
off_g = 0;
for f = 1 : numel(train_conts)
    tc = train_conts{f};
    for b = 1 : train_nBatch(f)
        s = (b-1)*tc.stride + 1;
        e = s + cfg.seqLen - 1;          % ≤ tc.N by window construction
        G_train(1, off_g+b, :) = single(tc.gate(s:e));
    end
    off_g = off_g + train_nBatch(f);
end
dlG_train = dlarray(G_train, 'CBT');

fprintf('Physics constants precomputed.\n');
fprintf('  R0 range (with aging): [%.4f, %.4f] Ω\n', min(R0_age_train(:)), max(R0_age_train(:)));
fprintf('  h  range            : [%.4f, %.4f] V\n', min(h_data_train(:)), max(h_data_train(:)));
fprintf('  Gate coverage (train windows): %.1f%% of steps anchored\n\n', ...
    100*mean(G_train(:)));

%% ══════════════════════════════════════════════════════════════════════
%%  2.  NETWORK ARCHITECTURE
%%      CNN (multi-scale) → LayerNorm → Projection → PosEnc
%%      → Transformer ×2 → FC(2)  [dSOC_raw, OCV_corr_raw]
%% ══════════════════════════════════════════════════════════════════════
lgraph = layerGraph();
lgraph = addLayers(lgraph, sequenceInputLayer(cfg.numFeatures, 'Name', 'input'));

lgraph = addLayers(lgraph, convolution1dLayer(3, cfg.numFilters, 'Padding','same','Name','conv_k3'));
lgraph = addLayers(lgraph, convolution1dLayer(5, cfg.numFilters, 'Padding','same','Name','conv_k5'));
lgraph = addLayers(lgraph, convolution1dLayer(9, cfg.numFilters, 'Padding','same','Name','conv_k9'));
lgraph = addLayers(lgraph, concatenationLayer(1, 3,              'Name','concat_cnn'));
lgraph = addLayers(lgraph, layerNormalizationLayer(             'Name','ln_cnn'));
lgraph = addLayers(lgraph, reluLayer(                           'Name','relu_cnn'));
lgraph = addLayers(lgraph, dropoutLayer(cfg.dropoutRate,        'Name','drop_cnn'));
lgraph = addLayers(lgraph, fullyConnectedLayer(cfg.attentionDim,'Name','proj'));
lgraph = addLayers(lgraph, PositionalEncodingLayer(cfg.attentionDim, cfg.pEncScale, 'pos_enc'));

lgraph = addLayers(lgraph, selfAttentionLayer(cfg.numHeads, cfg.attentionDim, 'Name','self_att_1'));
lgraph = addLayers(lgraph, dropoutLayer(cfg.dropoutRate,        'Name','drop_att_1'));
lgraph = addLayers(lgraph, additionLayer(2,                     'Name','add_att_1'));
lgraph = addLayers(lgraph, layerNormalizationLayer(            'Name','ln_att_1'));
lgraph = addLayers(lgraph, fullyConnectedLayer(cfg.attentionDim*cfg.ffnExpand,'Name','ffn1_1'));
lgraph = addLayers(lgraph, reluLayer(                          'Name','relu_ffn_1'));
lgraph = addLayers(lgraph, dropoutLayer(cfg.dropoutRate,       'Name','drop_ffn_1'));
lgraph = addLayers(lgraph, fullyConnectedLayer(cfg.attentionDim,'Name','ffn2_1'));
lgraph = addLayers(lgraph, additionLayer(2,                    'Name','add_ffn_1'));
lgraph = addLayers(lgraph, layerNormalizationLayer(           'Name','ln_ffn_1'));

lgraph = addLayers(lgraph, selfAttentionLayer(cfg.numHeads, cfg.attentionDim, 'Name','self_att_2'));
lgraph = addLayers(lgraph, dropoutLayer(cfg.dropoutRate,       'Name','drop_att_2'));
lgraph = addLayers(lgraph, additionLayer(2,                    'Name','add_att_2'));
lgraph = addLayers(lgraph, layerNormalizationLayer(          'Name','ln_att_2'));
lgraph = addLayers(lgraph, fullyConnectedLayer(cfg.attentionDim*cfg.ffnExpand,'Name','ffn1_2'));
lgraph = addLayers(lgraph, reluLayer(                         'Name','relu_ffn_2'));
lgraph = addLayers(lgraph, dropoutLayer(cfg.dropoutRate,      'Name','drop_ffn_2'));
lgraph = addLayers(lgraph, fullyConnectedLayer(cfg.attentionDim,'Name','ffn2_2'));
lgraph = addLayers(lgraph, additionLayer(2,                   'Name','add_ffn_2'));
lgraph = addLayers(lgraph, layerNormalizationLayer(          'Name','ln_ffn_2'));
lgraph = addLayers(lgraph, fullyConnectedLayer(2,            'Name','net_out'));

lgraph = connectLayers(lgraph,'input','conv_k3');
lgraph = connectLayers(lgraph,'input','conv_k5');
lgraph = connectLayers(lgraph,'input','conv_k9');
lgraph = connectLayers(lgraph,'conv_k3','concat_cnn/in1');
lgraph = connectLayers(lgraph,'conv_k5','concat_cnn/in2');
lgraph = connectLayers(lgraph,'conv_k9','concat_cnn/in3');
lgraph = connectLayers(lgraph,'concat_cnn','ln_cnn');
lgraph = connectLayers(lgraph,'ln_cnn','relu_cnn');
lgraph = connectLayers(lgraph,'relu_cnn','drop_cnn');
lgraph = connectLayers(lgraph,'drop_cnn','proj');
lgraph = connectLayers(lgraph,'proj','pos_enc');
lgraph = connectLayers(lgraph,'pos_enc','self_att_1');
lgraph = connectLayers(lgraph,'self_att_1','drop_att_1');
lgraph = connectLayers(lgraph,'drop_att_1','add_att_1/in1');
lgraph = connectLayers(lgraph,'pos_enc','add_att_1/in2');
lgraph = connectLayers(lgraph,'add_att_1','ln_att_1');
lgraph = connectLayers(lgraph,'ln_att_1','ffn1_1');
lgraph = connectLayers(lgraph,'ffn1_1','relu_ffn_1');
lgraph = connectLayers(lgraph,'relu_ffn_1','drop_ffn_1');
lgraph = connectLayers(lgraph,'drop_ffn_1','ffn2_1');
lgraph = connectLayers(lgraph,'ffn2_1','add_ffn_1/in1');
lgraph = connectLayers(lgraph,'ln_att_1','add_ffn_1/in2');
lgraph = connectLayers(lgraph,'add_ffn_1','ln_ffn_1');
lgraph = connectLayers(lgraph,'ln_ffn_1','self_att_2');
lgraph = connectLayers(lgraph,'self_att_2','drop_att_2');
lgraph = connectLayers(lgraph,'drop_att_2','add_att_2/in1');
lgraph = connectLayers(lgraph,'ln_ffn_1','add_att_2/in2');
lgraph = connectLayers(lgraph,'add_att_2','ln_att_2');
lgraph = connectLayers(lgraph,'ln_att_2','ffn1_2');
lgraph = connectLayers(lgraph,'ffn1_2','relu_ffn_2');
lgraph = connectLayers(lgraph,'relu_ffn_2','drop_ffn_2');
lgraph = connectLayers(lgraph,'drop_ffn_2','ffn2_2');
lgraph = connectLayers(lgraph,'ffn2_2','add_ffn_2/in1');
lgraph = connectLayers(lgraph,'ln_att_2','add_ffn_2/in2');
lgraph = connectLayers(lgraph,'add_ffn_2','ln_ffn_2');
lgraph = connectLayers(lgraph,'ln_ffn_2','net_out');

dlnet     = dlnetwork(lgraph);
numParams = sum(cellfun(@numel, dlnet.Learnables.Value));
fprintf('Network compiled. Learnable parameters: %d\n\n', numParams);

%% ══════════════════════════════════════════════════════════════════════
%%  3.  TRAINING LOOP
%% ══════════════════════════════════════════════════════════════════════
avgGradNet=[]; avgSqGradNet=[];
% FIXED (V8.1): Drop adaptive homoscedastic weights — they went negative
% (loss < -0.2) because the log-uncertainty formulation is unbounded below
% when loss terms are very small. Use fixed weights instead:
%   w_data=1.0 (SOC RMSE primary), w_Ah=1.0 (Coulomb, rescaled by lambda_Ah),
%   w_ECM=0.5 (voltage physics, secondary).
% This eliminates the -0.26 loss instability seen in V8 and makes convergence
% easier to monitor (loss is always non-negative and interpretable).

best_val_rmse    = inf;
best_epoch       = 1;
patience         = 80;           % V8.2: was 50 — give plateau more time to break
patience_counter = 0;
best_net         = dlnet;
% (no best_wts needed — fixed weights)

lossLog = zeros(cfg.numEpochs, 6);   % [total, data, Ah(corr), ECM, endpoint, drift]
valLog     = zeros(cfg.numEpochs, 1);   % CONTINUOUS val RMSE (selection metric, V8.5)
valLogWin  = zeros(cfg.numEpochs, 1);   % windowed oracle RMSE (diagnostic)
numIterations = ceil(nBatchTrain / cfg.miniBatchSize);

Yval_np = extractdata(dlY_val);                       % for cheap windowed val metric
Ival_np = extractdata(dlI_val);                       % residual recon needs I, C
Cval_np = extractdata(dlC_val);
% precompute the windowed physics increment for the cheap val metric (residual)
val_phys = zeros(size(Ival_np), 'single');
val_phys(:,:,2:end) = (Ival_np(:,:,2:end) * cfg.dt) ./ Cval_np(:,:,2:end);

fprintf('%-6s  %-9s  %-10s  %-10s  %-10s  | %-8s  %-10s  %-8s\n', ...
    'Epoch','LR','Tot_Loss','Data_Loss','DriftLoss','WinRMSE','ContRMSE','Best_Cont(%)');
fprintf('%s\n', repmat('-',1,105));

tic;
for epoch = 1 : cfg.numEpochs

    if epoch <= cfg.warmupEpochs
        currentLR = cfg.initLR * (epoch / cfg.warmupEpochs);
    else
        T_c = cfg.cosineEpochs - cfg.warmupEpochs;
        t_c = min(epoch - cfg.warmupEpochs, T_c);   % hold at minLR past horizon
        currentLR = cfg.minLR + 0.5*(cfg.initLR - cfg.minLR)*(1 + cos(pi*t_c/T_c));
    end

    in_curriculum = (epoch <= cfg.curriculumEpochs);
    idx = randperm(nBatchTrain);
    epochLosses = zeros(numIterations, 6);

    for iter = 1 : numIterations
        sIdx = (iter-1)*cfg.miniBatchSize + 1;
        eIdx = min(iter*cfg.miniBatchSize, nBatchTrain);
        mb   = idx(sIdx:eIdx);

        mb_dlX  = dlX_train(:,mb,:);  mb_dlY  = dlY_train(:,mb,:);
        mb_dlI  = dlI_train(:,mb,:);  mb_dlV  = dlV_train(:,mb,:);
        mb_dlC  = dlC_train(:,mb,:);  mb_dlH  = dlH_train(:,mb,:);
        mb_dlR0 = dlR0_train(:,mb,:); mb_dlW  = dlW_train(:,mb,:);
        mb_bc   = blend_chg_train(:,mb,:);
        mb_bd   = blend_dchg_train(:,mb,:);
        mb_tm   = dlTransientMask_train(:,mb,:);
        mb_dlG  = dlG_train(:,mb,:);

        % V8.6 — sensor-bias augmentation: corrupt the sensor everywhere a real
        % biased sensor would appear (input ch2 here; physics + ECM inside the
        % loss via mb_dlB), while the labels stay clean.
        nb_mb   = numel(mb);
        mb_bias = single((rand(1,nb_mb) < cfg.bias_aug_p) .* ...
                         (2*rand(1,nb_mb) - 1) * cfg.bias_aug_max);
        mb_dlB  = dlarray(reshape(mb_bias, 1, nb_mb, 1), 'CBT');
        if any(mb_bias ~= 0)
            bf = zeros(cfg.numFeatures, nb_mb, cfg.seqLen, 'single');
            bf(2,:,:) = repmat(mb_bias(:).' / cfg.sig_I, 1, 1, cfg.seqLen);
            mb_dlX = mb_dlX + dlarray(bf, 'CBT');
        end

        dlnet = resetState(dlnet);
        [gradsNet, netState, lv] = dlfeval(@hapitLossV8, ...
            dlnet, mb_dlX, mb_dlY, mb_dlI, mb_dlV, ...
            mb_dlH, mb_dlR0, mb_bc, mb_bd, mb_dlC, mb_tm, mb_dlW, mb_dlG, ...
            mb_dlB, cfg, in_curriculum);

        dlnet.State = netState;
        gradsNet    = clipGradients(gradsNet, cfg.gradClip);

        [dlnet, avgGradNet, avgSqGradNet] = adamupdate( ...
            dlnet, gradsNet, avgGradNet, avgSqGradNet, epoch, currentLR);
        % (no adaptive weight update — fixed weights used in loss)

        epochLosses(iter,:) = lv;
        clear mb_dlX mb_dlY mb_dlI mb_dlV mb_dlC mb_dlH mb_dlR0 mb_dlG mb_dlB ...
              mb_dlW mb_bc mb_bd mb_tm gradsNet;
    end

    avg_lv = mean(epochLosses, 1);
    lossLog(epoch,:) = avg_lv;

    % --- validation metrics ------------------------------------------------
    % (a) cheap WINDOWED residual metric (diagnostic: increment-correction quality)
    % (b) V8.5: CONTINUOUS reconstruction on B0007 — the deployment metric. The
    %     window corrections are already predicted for (a); reconstructing costs
    %     one 68k-step loop (<1 s). Early stopping / model selection now uses
    %     (b), because (a) is structurally BLIND to accumulated drift — V8.4's
    %     best-windowed epoch (0.39% windowed) carried 3.05% continuous error.
    val_N    = size(dlX_val,2);
    val_corr = zeros(1, val_N, cfg.seqLen, 'single');
    for vs = 1 : 128 : val_N
        ve = min(vs+127, val_N);
        dlnet = resetState(dlnet);
        vp    = predict(dlnet, dlX_val(:,vs:ve,:));
        val_corr(1,vs:ve,:) = extractdata(vp(1,:,:));
    end
    val_corr = val_corr * cfg.dSOC_scale;

    % (a) windowed oracle-anchored RMSE
    val_inc  = val_phys + val_corr;
    val_inc(:,:,1) = 0;
    anchor_val = Yval_np(:,:,1);
    SOC_pred_val = anchor_val + cumsum(val_inc, 3);
    SOC_pred_val = max(0, min(1, SOC_pred_val));
    win_rmse = sqrt(mean((SOC_pred_val - Yval_np).^2,'all'))*100;
    valLogWin(epoch) = win_rmse;

    % (b) continuous reconstruction RMSE (identical rule to evaluateContinuous)
    cont_rmse = contRMSEfromCorr(val_corr, val_cont, cfg);
    valLog(epoch) = cont_rmse;

    if cont_rmse < best_val_rmse
        best_val_rmse = cont_rmse; best_net = dlnet;
        best_epoch = epoch;
        patience_counter = 0;
        save('hapit_v8p6_best_checkpoint.mat','best_net','cfg','train_scalers');
    else
        patience_counter = patience_counter + 1;
    end

    if mod(epoch,10)==0 || patience_counter==0 || epoch==1
        cs = 'OFF'; if in_curriculum, cs='ON'; end
        fprintf('%-6d  %-9.2e  %-10.6f  %-10.6f  %-10.6f  | %-8.4f  %-10.4f  %-9.4f (Ep %d)  %s\n', ...
            epoch,currentLR,avg_lv(1),avg_lv(2),avg_lv(6),win_rmse,cont_rmse,best_val_rmse,best_epoch,cs);
    end

    if patience_counter >= patience
        fprintf('\n[!] EARLY STOPPING at Epoch %d.\n', epoch);
        break;
    end
end
elapsed = toc;
fprintf('\nTraining complete in %.1f s.\n\n', elapsed);

%% ══════════════════════════════════════════════════════════════════════
%%  4.  FINAL EVALUATION  (continuous reconstruction = deployment metric)
%% ══════════════════════════════════════════════════════════════════════
fprintf('Restoring best model from Epoch %d (CONTINUOUS Val RMSE=%.4f%%)...\n\n', ...
    best_epoch, best_val_rmse);
dlnet = best_net;
% (no adaptWts to restore — fixed weights)

% --- TRAIN (two files: reconstruct each continuously, then combine) -------
pred_train=[]; gt_train=[]; off=0;
for f = 1 : numel(train_conts)
    nb  = train_nBatch(f);
    dlXf = dlX_train(:, off+1:off+nb, :);
    [pp, gg] = evaluateContinuous(dlnet, dlXf, train_conts{f}, cfg);
    pred_train = [pred_train; pp]; gt_train = [gt_train; gg]; %#ok<AGROW>
    off = off + nb;
end
err_train  = pred_train - gt_train;
met_train  = metricStruct(gt_train, pred_train);

% --- VAL ----------------------------------------------------------------
[pred_val, gt_val] = evaluateContinuous(dlnet, dlX_val, val_cont, cfg);
err_val  = pred_val - gt_val;
met_val  = metricStruct(gt_val, pred_val);

% --- TEST ---------------------------------------------------------------
[pred_test, gt_test] = evaluateContinuous(dlnet, dlX_test, test_cont, cfg);
err_test = pred_test - gt_test;
met_test = metricStruct(gt_test, pred_test);

% --- diagnostic: windowed oracle-anchored increment quality (all splits) ----
% Reconstruct each TRAIN file's windowed-oracle metric, then combine.
pwo=[]; gwo=[]; off=0;
for f = 1 : numel(train_conts)
    nb  = train_nBatch(f);
    [pp, gg] = evaluateWindowedOracle(dlnet, dlX_train(:,off+1:off+nb,:), train_conts{f}, cfg);
    pwo=[pwo; pp]; gwo=[gwo; gg]; off=off+nb; %#ok<AGROW>
end
met_train_win = metricStruct(gwo, pwo);
[~,~,~,met_val_win ] = evaluateWindowedOracle(dlnet, dlX_val,  val_cont,  cfg);
[~,~,~,met_test_win] = evaluateWindowedOracle(dlnet, dlX_test, test_cont, cfg);

fprintf('\n========================================================================\n');
fprintf('     FINAL EVALUATION\n');
fprintf('========================================================================\n');
fprintf('%-22s | %-12s | %-12s | %-12s\n','Metric','Train','Validation','Test');
fprintf('------------------------------------------------------------------------\n');
fprintf('%-22s | %-11.4f%% | %-11.4f%% | %-11.4f%%\n','Continuous RMSE', met_train.rmse,met_val.rmse,met_test.rmse);
fprintf('%-22s | %-11.4f%% | %-11.4f%% | %-11.4f%%\n','Continuous MAE',  met_train.mae, met_val.mae, met_test.mae);
fprintf('%-22s | %-11.4f%% | %-11.4f%% | %-11.4f%%\n','Continuous Max',  met_train.maxe,met_val.maxe,met_test.maxe);
fprintf('%-22s | %-12.4f | %-12.4f | %-12.4f\n',   'Continuous R^2',  met_train.r2,  met_val.r2,  met_test.r2);
fprintf('------------------------------------------------------------------------\n');
fprintf('%-22s | %-11.4f%% | %-11.4f%% | %-11.4f%%\n','Windowed-oracle RMSE', met_train_win.rmse,met_val_win.rmse,met_test_win.rmse);
fprintf('========================================================================\n');
fprintf('  Continuous = deployment metric (full reconstruction).\n');
fprintf('  Windowed-oracle = increment-correction quality (anchored each window).\n');
fprintf('  A large gap (continuous >> windowed) ⇒ residual drift/anchoring;\n');
fprintf('  a small gap ⇒ error is intrinsic to the increment correction.\n');
fprintf('========================================================================\n\n');

% --- P0: per-segment error breakdown (test) — shows where error concentrates --
perSegmentReport(pred_test, gt_test, 'TEST (B0018)');

% --- P4: current-sensor-bias ablation (test) — demonstrates the network does
%     more than integrate: the voltage path corrects an injected current bias. --
fprintf('\n── P4: CURRENT-BIAS ABLATION (TEST) ─────────────────────────────────\n');
fprintf('  Inject a constant current-sensor bias; compare pure Coulomb (corr=0)\n');
fprintf('  vs the network-corrected reconstruction. If the network reduces the\n');
fprintf('  error, it is fusing voltage to fix the bias (not merely integrating).\n');
fprintf('  %-14s | %-18s | %-18s\n','Bias (A)','Pure-Coulomb RMSE','Network RMSE');
for bias = [0.0, 0.05, 0.10]
    abl = currentBiasAblation(dlnet, dlX_test, test_cont, cfg, bias);
    fprintf('  %-14.2f | %-17.4f%% | %-17.4f%%\n', bias, abl.rmse_nocorr, abl.rmse_corr);
end
fprintf('─────────────────────────────────────────────────────────────────────\n\n');

if met_test.rmse < 0.5
    fprintf('>>> TARGET ACHIEVED: continuous test RMSE %.4f%% < 0.5%%\n\n', met_test.rmse);
else
    fprintf('>>> Continuous test RMSE %.4f%%. If >0.5%%, see the tuning notes below.\n\n', met_test.rmse);
end

%% ══════════════════════════════════════════════════════════════════════
%%  5.  PLOTTING — ONE window per metric; Train/Val/Test stacked top→bottom
%% ══════════════════════════════════════════════════════════════════════
% Exactly 5 figure windows open at the end:
%   (1) SOC Trajectory      (2) Error Distribution   (3) Regression
%   (4) Absolute Error      (5) Training Convergence  (single plot)
% Each of (1)-(4) holds 3 stacked subplots: Train (top), Validation, Test.

datasets = {'Train (B0005+B0006)', 'Validation (B0007)', 'Test (B0018)'};
preds    = {pred_train, pred_val, pred_test};
gts      = {gt_train,   gt_val,   gt_test};
errs     = {err_train,  err_val,  err_test};
mets     = {met_train,  met_val,  met_test};
colours  = {[0.18 0.55 0.34], [0.15 0.35 0.70], [0.80 0.20 0.15]};

try
    % ── Window 1: SOC Trajectory (stacked) ────────────────────────────────
    figure('Name','SOC Trajectory','Position',[60 40 820 960]);
    for d = 1:3
        subplot(3,1,d);
        plot(gts{d}, 'k', 'LineWidth', 1.3); hold on;
        plot(preds{d}, '--', 'LineWidth', 1.0, 'Color', colours{d});
        xlabel('Time step (s)'); ylabel('SOC (%)');
        title(sprintf('%s  —  RMSE = %.4f%%  |  MAE = %.4f%%', ...
              datasets{d}, mets{d}.rmse, mets{d}.mae));
        if d==1, legend('Ground truth','Predicted','Location','best'); end
        grid on; set(gca,'FontSize',10); ylim([0 100]);
    end
    sgtitle('SOC Trajectory — Continuous Reconstruction');

    % ── Window 2: Error Distribution (stacked) ────────────────────────────
    figure('Name','Error Distribution','Position',[90 40 820 960]);
    for d = 1:3
        subplot(3,1,d);
        histogram(errs{d}, 80, 'Normalization','pdf', ...
                  'FaceColor', colours{d}, 'FaceAlpha', 0.70, 'EdgeColor','none'); hold on;
        plotDistributionFit(errs{d});
        xline( mets{d}.rmse, ':', 'Color',[0.5 0.5 0.5], 'LineWidth', 1.1);
        xline(-mets{d}.rmse, ':', 'Color',[0.5 0.5 0.5], 'LineWidth', 1.1);
        xlabel('Error (%)'); ylabel('PDF');
        title(sprintf('%s  —  RMSE = %.4f%%  |  Max |e| = %.4f%%', ...
              datasets{d}, mets{d}.rmse, mets{d}.maxe));
        grid on; set(gca,'FontSize',10);
    end
    sgtitle('Error Distribution');

    % ── Window 3: Regression (stacked) ────────────────────────────────────
    figure('Name','Regression','Position',[120 40 620 980]);
    for d = 1:3
        subplot(3,1,d);
        npts = numel(gts{d});
        step = max(1, floor(npts/4000));
        scatter(gts{d}(1:step:end), preds{d}(1:step:end), 5, colours{d}, ...
                'filled', 'MarkerFaceAlpha', 0.30); hold on;
        plot([0 100],[0 100],'k--','LineWidth',1.4);
        x_fit = linspace(0,100,200);
        plot(x_fit, mets{d}.slope*x_fit + mets{d}.int, '-', ...
             'Color', colours{d}, 'LineWidth', 1.7);
        xlabel('True SOC (%)'); ylabel('Predicted SOC (%)');
        title(sprintf('%s  —  R^2 = %.4f  |  slope = %.4f', ...
              datasets{d}, mets{d}.r2, mets{d}.slope));
        axis([0 100 0 100]); grid on; set(gca,'FontSize',10);
    end
    sgtitle('Regression: Predicted vs True SOC');

    % ── Window 4: Absolute Error over time (stacked) ──────────────────────
    figure('Name','Absolute Error','Position',[150 40 820 960]);
    for d = 1:3
        subplot(3,1,d);
        plot(abs(errs{d}), 'Color', colours{d}, 'LineWidth', 0.9); hold on;
        yline(mets{d}.rmse, 'k--', sprintf('RMSE %.3f%%', mets{d}.rmse), ...
              'LineWidth', 1.2, 'LabelVerticalAlignment','bottom');
        yline(mets{d}.mae,  'k:',  sprintf('MAE %.3f%%',  mets{d}.mae), ...
              'LineWidth', 1.0, 'LabelVerticalAlignment','bottom');
        xlabel('Time step (s)'); ylabel('|Error| (%)');
        title(sprintf('%s  —  RMSE = %.4f%%  |  Max = %.4f%%', ...
              datasets{d}, mets{d}.rmse, mets{d}.maxe));
        grid on; set(gca,'FontSize',10);
    end
    sgtitle('Absolute Error over Time');

    % ── Window 5: Training Convergence (single plot) ──────────────────────
    ep_end = find(valLog > 0, 1, 'last');
    if isempty(ep_end), ep_end = cfg.numEpochs; end
    ep = 1 : ep_end;

    figure('Name','Training Convergence','Position',[180 120 900 460]);
    yyaxis left;
    plot(ep, lossLog(ep,2), 'b-',  'LineWidth', 1.3); hold on;
    plot(ep, lossLog(ep,6), 'g--', 'LineWidth', 1.0);   % V8.5 drift loss
    plot(ep, lossLog(ep,4), 'm:',  'LineWidth', 1.0);   % ECM
    ylabel('Loss (log scale)');  set(gca,'YScale','log');
    legend('Data loss','Drift loss','ECM loss','Location','northeast');

    yyaxis right;
    plot(ep, valLog(ep),    'r-',  'LineWidth', 1.6); hold on;
    plot(ep, valLogWin(ep), 'r:',  'LineWidth', 1.0);
    ylabel('Val RMSE (%)  [solid=continuous, dotted=windowed]');
    xline(best_epoch, 'k--', sprintf('Best ep %d', best_epoch), ...
          'LineWidth', 1.2, 'LabelVerticalAlignment','bottom');
    xlabel('Epoch');
    title(sprintf('Training Convergence — Best CONTINUOUS Val RMSE = %.4f%% @ Epoch %d', ...
          best_val_rmse, best_epoch));
    grid on; set(gca,'FontSize',11);

catch ME
    warning('Plotting skipped: %s', ME.message);
end

%% ══════════════════════════════════════════════════════════════════════
%%  6.  SAVE
%% ══════════════════════════════════════════════════════════════════════
results = struct('met_train',met_train,'met_val',met_val,'met_test',met_test, ...
                 'met_train_win',met_train_win,'met_val_win',met_val_win, ...
                 'met_test_win',met_test_win,'best_epoch',best_epoch, ...
                 'lossLog',lossLog,'valLog',valLog,'valLogWin',valLogWin,'cfg',cfg);
save('hapit_v8p6_results.mat','results','train_scalers');
fprintf('Saved: hapit_v8p6_results.mat  and  hapit_v8p6_best_checkpoint.mat\n');

%% ══════════════════════════════════════════════════════════════════════
%%  TUNING NOTES (V8.5 — target: continuous VAL RMSE < 0.5%)
%% ══════════════════════════════════════════════════════════════════════
%  THE LADDER (if the first V8.5 run lands above 0.5% on val):
%   1. Raise cfg.lambda_drift: 10 → 30 → 100. Watch the DriftLoss column: it
%      should fall steadily; the ContRMSE column should track it down. If
%      DriftLoss collapses but ContRMSE stalls ~1%, the residual is no longer
%      bias — check the per-segment report for WHERE it lives before tuning.
%   2. Raise cfg.lambda_end (0.5 → 2): complements L_drift by also matching
%      the anchor-pull at window ends.
%   3. If WinRMSE degrades while ContRMSE improves, that trade is FINE — the
%      windowed number is a diagnostic, the continuous number is the metric.
%   4. If ContRMSE oscillates epoch-to-epoch late in training, the selection
%      will simply pick a good epoch (that is the point of Change 2); a
%      lower cfg.initLR (5e-3 → 3e-3) reduces the oscillation itself.
%
%  SANITY CHECKS EVERY RUN:
%   • P4 ablation: network RMSE must stay BELOW pure-Coulomb under injected
%     bias. If lambda_drift is pushed so hard that the network goes inert
%     (P4 rows equal), back it off — the RMSE would be "good" but meaningless.
%   • Continuous-vs-windowed gap per split: small gap = drift solved.
%
%  V8.4 NOTES (retained)
%  ----------------------------------------------------------------------
%  READ THE TWO METRICS TOGETHER:
%  • Windowed-oracle RMSE  → increment-CORRECTION quality (network only)
%  • Continuous recon RMSE → deployment metric (network + gated anchoring)
%  Because the physics increment I·dt/C is now built in, BOTH should start far
%  lower than V8.3. A zero-correction model already reproduces the GT to ~0%, so
%  any residual error is the network's correction plus the gated re-anchoring.
%
%  IF continuous RMSE >> windowed-oracle (large gap): anchoring/drift dominates
%    → Inspect the per-segment report — is it the startup chunk or the cusps?
%    → Loosen the gate if too few rests qualify: lower cfg.settle_len (120→60)
%      or raise cfg.dvdt_thr (2e-4→5e-4). Tighten it if a cusp rest is anchoring
%      on non-relaxed voltage (raise cfg.settle_len, lower cfg.dvdt_thr).
%    → cfg.alpha_ocv controls re-anchor strength (0.20; try 0.10–0.30).
%    → cfg.settle_len / cfg.dvdt_thr / cfg.alpha_ocv MUST match in BOTH
%      prepareBatteryDataV9 and here (they are read from cfg, so keep one source).
%
%  IF windowed-oracle RMSE is high: the correction itself is mislearned
%    → The correction should be SMALL. If End_Loss stays high, raise
%      cfg.lambda_end (0.5→1.0) to punish accumulated drift harder.
%    → If the correction is being over-suppressed (model can't fix a real bias),
%      LOWER cfg.lambda_Ah (10→1). If it is noisy/unstable, RAISE it (10→50).
%    → cfg.dSOC_scale caps the per-step correction (1e-3 ⇒ ±~0.5%/step at ±5
%      logits, already >> the ~3e-4 physics step); rarely needs changing.
%    → More capacity/epochs: cfg.numFilters (48→64), cfg.numEpochs (400→600).
%
%  IF training loss oscillates: cfg.initLR too high (5e-3→3e-3), restart.
%
%  P4 ABLATION CHECK: under an injected current bias the Network RMSE should be
%  BELOW the Pure-Coulomb RMSE — that is the evidence the transformer fuses
%  voltage to correct sensor bias rather than merely integrating current.

%% ══════════════════════════════════════════════════════════════════════
%%  LOCAL FUNCTIONS
%% ══════════════════════════════════════════════════════════════════════
function [gradsNet,state,lossVals] = hapitLossV8( ...
        net,X,Y_soc,I_seq,V_meas,dlH,dlR0,...
        blend_chg,blend_dchg,dlC,dlTransientMask,dlW,dlG,dlB,cfg,in_curriculum)
% RESIDUAL-increment loss (V8.4). The network predicts a CORRECTION on top of
% the exact physics increment; SOC is reconstructed as
%     dSOC[k] = I[k]·dt/C[k] + corr[k]
%     SOC_pred = anchor + cumsumTime3(dSOC)
% with step 1 pinned to the anchor (true window-start SOC). Because the
% physics baseline matches the label-construction increment exactly, a
% zero-correction network reproduces the ground truth, so training only has to
% drive the (small) correction — eliminating the attenuation/drift of V8.3.
%
% NOTE: uses the RECTANGULAR increment I[k]·dt/C[k] (not the midpoint), to be
% byte-identical to prepareBatteryDataV9's GT construction and the continuous
% reconstruction in evaluateContinuous.

    [net_out,state] = forward(net,X);
    net_out = stripdims(net_out);            % [2, B, L] ; dim3 = time
    Y   = stripdims(Y_soc);
    Iz  = stripdims(I_seq);
    Vz  = stripdims(V_meas);
    Hz  = stripdims(dlH);
    R0z = stripdims(dlR0);
    BCz = stripdims(blend_chg);
    BDz = stripdims(blend_dchg);
    Cz  = stripdims(dlC);
    TMz = stripdims(dlTransientMask);
    Wz  = stripdims(dlW);
    Gz  = stripdims(dlG);                                     % settled-rest gate [1,B,L]
    Bz  = stripdims(dlB);                                     % sensor bias (A)  [1,B,1]
    Iz_b = Iz + Bz;                                           % what a biased sensor reads

    L = size(net_out,3);

    % --- scaled outputs ---
    corr     = net_out(1,:,:) * cfg.dSOC_scale;       % per-step CORRECTION
    OCV_corr = net_out(2,:,:) * cfg.ocv_corr_scale;   % OCV correction (V)

    % --- RESIDUAL increment: SENSOR (possibly biased) physics + correction ---
    % At deployment a biased sensor corrupts the Coulomb term too; the labels
    % stay clean, so the data loss demands corr ≈ −b·dt/C — the network learns
    % to detect the voltage-vs-current mismatch and counteract it (V8.6).
    dSOC_phy  = (Iz_b(:,:,2:end) * cfg.dt) ./ Cz(:,:,2:end); % [1, B, L-1]
    dInc      = dSOC_phy + corr(:,:,2:end);                  % [1, B, L-1]
    zero1     = corr(:,:,1) * 0;                             % zeros that keep the graph
    dSOC_full = cat(3, zero1, dInc);                        % step 1 pinned to anchor
    anchor    = Y(:,:,1);                                    % true window-start SOC
    SOC_pred  = anchor + cumsumTime3(dSOC_full);            % autodiff-safe cumulative sum

    % --- weighted data loss --- (Wz is [1,B,1], broadcasts over time)
    L_data = mean(Wz .* (SOC_pred - Y).^2, 'all');

    % --- correction-magnitude regulariser ---
    % Keeps the correction small so the model stays near pure Coulomb counting
    % unless the voltage physics genuinely demands a deviation.
    L_Ah   = mean(corr(:,:,2:end).^2, 'all');

    % --- ENDPOINT drift penalty (P3) ---
    % Per-step losses do not penalise ACCUMULATED correction error; this term
    % directly punishes the integrated drift at the window end.
    L_end  = mean((SOC_pred(:,:,end) - Y(:,:,end)).^2, 'all');

    % --- V8.5/V8.6: UNANCHORED-DRIFT penalty (bias-aware) ---
    % On non-gated steps the label increment is exactly I·dt/C (CLEAN current),
    % so the correct target for the TOTAL deviation there is zero:
    %     deviation per step = corr + b·dt/C     (b = injected sensor bias)
    % For clean windows (b=0) this reduces to the V8.5 term; for augmented
    % windows it demands the correction CANCEL the bias rather than being
    % pushed to zero — the two objectives no longer fight.
    bias_inc  = (Bz * cfg.dt) ./ Cz(:,:,2:end);                   % [1,B,L-1]
    drift_sum = sum((1 - Gz(:,:,2:end)) .* (corr(:,:,2:end) + bias_inc), 3);
    L_drift   = mean(drift_sum.^2, 'all');

    % --- ECM loss with empirical OCV polynomial ---
    % IR term uses the SENSOR current (biased when augmented) — consistent with
    % what a deployed estimator would compute from its own reading.
    OCV_poly = polyOCV_dl(SOC_pred, BCz, BDz, cfg);
    V_hat    = OCV_poly + OCV_corr + Hz + (Iz_b .* R0z);
    L_ECM    = mean((V_hat - Vz).^2 .* TMz, 'all');

    % --- fixed-weight total loss; ECM frozen during curriculum ---
    w_ECM = cfg.w_ECM;
    if in_curriculum, w_ECM = 0; end
    totalLoss = cfg.w_data*L_data + cfg.lambda_Ah*L_Ah + ...
                cfg.lambda_end*L_end + cfg.lambda_drift*L_drift + w_ECM*L_ECM;

    gradsNet = dlgradient(totalLoss, net.Learnables);

    lossVals = [extractdata(totalLoss), extractdata(L_data), ...
                extractdata(L_Ah),      extractdata(L_ECM), ...
                extractdata(L_end),     extractdata(L_drift)];
end

% ─────────────────────────────────────────────────────────────────────────
function [pred_pct, gt_pct, err, metrics] = evaluateContinuous(net, dlX, cont, cfg)
% Continuous SOC reconstruction, BYTE-IDENTICAL to the label rule in
% prepareBatteryDataV9 except that the network correction is added to each
% increment:
%     dSOC[i] = I[i]·dt/C[i] + corr[i]
%     SOC[i]  = clip( (1-a·g)·(SOC[i-1] + dSOC[i]) + a·g·SOC_ocv[i] )
% where g = cont.gate[i] (settled-rest mask) and a = cont.alpha_ocv. SOC starts
% at the SAME cont.SOC_0 used to build the labels. Therefore a zero-correction
% network reproduces the ground truth exactly (verified to 1e-9) and the
% startup spike / bidirectional-anchor mismatch of V8.3 is gone. This is the
% deployment-realistic metric.

    numBatches = size(dlX, 2);
    L = cfg.seqLen;

    % 1) network CORRECTIONS per window (channel 1 × scale)
    corr_mat = zeros(1, numBatches, L);
    for s = 1 : 64 : numBatches
        e = min(s+63, numBatches);
        net = resetState(net);
        p   = predict(net, dlX(:,s:e,:));
        corr_mat(1, s:e, :) = extractdata(p(1,:,:)) * cfg.dSOC_scale;
    end

    % 2)+3) stitch + reconstruct (shared with the in-loop val metric, V8.5)
    [SOC, N] = reconFromCorr(corr_mat, cont, cfg);

    pred_pct = SOC * 100;
    gt_pct   = cont.SOC(1:N) * 100;
    err      = pred_pct - gt_pct;
    metrics  = metricStruct(gt_pct, pred_pct);
end

% ─────────────────────────────────────────────────────────────────────────
function [SOC, N] = reconFromCorr(corr_mat, cont, cfg)
% V8.5 shared core: stitch per-window corrections (ALREADY scaled) onto the
% global timeline (later windows overwrite the overlap), then run the exact
% label-rule recursion forward from cont.SOC_0 with gated re-anchoring. Used by
% BOTH evaluateContinuous and the per-epoch continuous val metric so the
% selection metric and the reported metric can never diverge.
    numBatches = size(corr_mat, 2);
    L      = cfg.seqLen;
    stride = cont.stride;

    totalPts = (numBatches-1)*stride + L;
    corr_seq = zeros(totalPts, 1);
    corr_seq(1:L) = squeeze(corr_mat(1,1,:));
    for b = 2 : numBatches
        g1 = (b-1)*stride + 1;
        g2 = min(g1 + L - 1, totalPts);
        nL = g2 - g1 + 1;
        corr_seq(g1:g2) = squeeze(corr_mat(1,b,1:nL));
    end

    N = min(totalPts, cont.N);
    a = cont.alpha_ocv;
    corr_seq(1) = 0;                          % step 1 is the anchor
    SOC = zeros(N,1);
    SOC(1) = cont.SOC_0;                       % SAME initial anchor as the labels
    for i = 2 : N
        inc = (cont.I(i) * cfg.dt) / cont.C(i) + corr_seq(i);   % residual increment
        soc_next = SOC(i-1) + inc;
        if cont.gate(i)                         % settled-rest re-anchor (same mask as GT)
            soc_ocv  = max(0, min(1, ...
                interp1(cont.eq_s, cont.soc_eq_s, cont.V(i), 'linear','extrap')));
            soc_next = (1 - a)*soc_next + a*soc_ocv;
        end
        SOC(i) = max(0, min(1, soc_next));
    end
end

% ─────────────────────────────────────────────────────────────────────────
function rmse = contRMSEfromCorr(corr_mat, cont, cfg)
% Per-epoch continuous validation RMSE from already-predicted, already-scaled
% window corrections (V8.5 selection metric).
    [SOC, N] = reconFromCorr(corr_mat, cont, cfg);
    rmse = sqrt(mean((SOC*100 - cont.SOC(1:N)*100).^2));
end

% ─────────────────────────────────────────────────────────────────────────
function [pred_pct, gt_pct, err, metrics] = evaluateWindowedOracle(net, dlX, cont, cfg)
% Diagnostic only: anchor EACH window at the true SOC at its start (oracle),
% reconstruct within-window as anchor + cumsum(increments), stitch, compare.
% Isolates how well the increments were learned, independent of drift.

    numBatches = size(dlX, 2);
    L      = cfg.seqLen;
    stride = cont.stride;

    corr_mat = zeros(1, numBatches, L);
    for s = 1 : 64 : numBatches
        e = min(s+63, numBatches);
        net = resetState(net);
        p   = predict(net, dlX(:,s:e,:));
        corr_mat(1, s:e, :) = extractdata(p(1,:,:)) * cfg.dSOC_scale;
    end

    % RESIDUAL within-window reconstruction: per window, anchor at the true SOC
    % at its start, then integrate dSOC = I·dt/C + corr (no rest anchoring — this
    % isolates raw increment-correction quality from drift/anchoring).
    totalPts = (numBatches-1)*stride + L;
    SOC_pred = zeros(totalPts,1);
    for b = 1 : numBatches
        g1 = (b-1)*stride + 1;
        g2 = min(g1 + L - 1, totalPts);
        nL = g2 - g1 + 1;
        win    = zeros(nL,1);
        win(1) = cont.SOC(min(g1, cont.N));            % oracle anchor at window start
        for j = 2 : nL
            gi = g1 + j - 1;
            if gi <= cont.N
                inc = (cont.I(gi) * cfg.dt) / cont.C(gi) + squeeze(corr_mat(1,b,j));
            else
                inc = squeeze(corr_mat(1,b,j));
            end
            win(j) = win(j-1) + inc;
        end
        SOC_pred(g1:g2) = win;
    end
    SOC_pred = max(0, min(1, SOC_pred));

    N = min(totalPts, cont.N);
    pred_pct = SOC_pred(1:N) * 100;
    gt_pct   = cont.SOC(1:N) * 100;
    err      = pred_pct - gt_pct;
    metrics  = metricStruct(gt_pct, pred_pct);
end

% ─────────────────────────────────────────────────────────────────────────
function m = metricStruct(gt_pct, pred_pct)
    gt_pct = gt_pct(:); pred_pct = pred_pct(:);
    e = pred_pct - gt_pct;
    m.rmse = sqrt(mean(e.^2));
    m.mae  = mean(abs(e));
    m.maxe = max(abs(e));
    [m.r2, m.slope, m.int] = evaluateRegression(gt_pct, pred_pct);
end

% ─────────────────────────────────────────────────────────────────────────
function y = cumsumTime3(x)
% Cumulative sum of a [1, B, L] (unformatted) dlarray along dim 3 (time),
% implemented as an upper-triangular matrix multiply. This is mathematically
% identical to cumsum(x,3) but, unlike cumsum, is supported for dlarray
% automatic differentiation in all MATLAB versions.
%
%   reshape collapses [1,B,L] -> [B,L] with (b,k) = x(1,b,k) exactly (the
%   leading singleton dim makes the column-major layout already [B,L]).
%   Multiplying by U (U(j,k)=1 for j<=k) gives y2(b,k)=sum_{j<=k} x(1,b,j).
    sz = size(x);
    B  = sz(2);
    L  = sz(3);
    x2 = reshape(x, [B, L]);                          % [B, L]
    U  = triu(ones(L, L, 'like', extractdata(x)));    % constant, matches device/type
    y2 = x2 * U;                                      % [B, L] cumulative over time
    y  = reshape(y2, [1, B, L]);
end

% ─────────────────────────────────────────────────────────────────────────
function V_ocv = polyOCV_dl(soc_dl, blend_chg, blend_dchg, cfg)
% Empirically fitted poly5 OCV (DA-1), blended by current sign.
    soc_c  = max(0.01, min(0.99, soc_dl));
    p_d    = cfg.ocv_dchg;
    p_c    = cfg.ocv_chg;
    V_dchg = p_d(1)*soc_c.^5 + p_d(2)*soc_c.^4 + p_d(3)*soc_c.^3 + ...
             p_d(4)*soc_c.^2 + p_d(5)*soc_c     + p_d(6);
    V_chg  = p_c(1)*soc_c.^5 + p_c(2)*soc_c.^4 + p_c(3)*soc_c.^3 + ...
             p_c(4)*soc_c.^2 + p_c(5)*soc_c     + p_c(6);
    V_ocv  = (V_chg .* blend_chg) + (V_dchg .* blend_dchg);
end

% ─────────────────────────────────────────────────────────────────────────
function grads = clipGradients(grads, threshold)
    totalSqNorm = 0;
    for i = 1:height(grads)
        g = extractdata(grads.Value{i});
        totalSqNorm = totalSqNorm + sum(g(:).^2);
    end
    gradNorm = sqrt(totalSqNorm);
    if gradNorm > threshold
        scale = threshold/gradNorm;
        for i = 1:height(grads)
            grads.Value{i} = grads.Value{i}*scale;
        end
    end
end

% ─────────────────────────────────────────────────────────────────────────
function h = computeHysteresis(I_raw, cfg)
    [~,nBatch,T] = size(I_raw);
    h = zeros(1,nBatch,T);
    for b = 1:nBatch
        h_k = 0.0;
        for k = 1:T
            I_k   = I_raw(1,b,k);
            decay = exp(-abs(I_k)*cfg.H_gamma*cfg.dt);
            h_k   = h_k*decay + cfg.H_max*sign(I_k)*(1-decay);
            h(1,b,k) = h_k;
        end
    end
end

% ─────────────────────────────────────────────────────────────────────────
function plotDistributionFit(err)
    try
        pd = fitdist(err(:),'Normal');
        xp = linspace(min(err),max(err),200);
        plot(xp,pdf(pd,xp),'k','LineWidth',2);
    catch; end
    xline(0,'k--','LineWidth',1.5);
end

% ─────────────────────────────────────────────────────────────────────────
function [R2,slope,intercept] = evaluateRegression(y_true,y_pred)
    y_true = y_true(:); y_pred = y_pred(:);
    p = polyfit(y_true,y_pred,1);
    slope = p(1); intercept = p(2);
    SS_tot = sum((y_true-mean(y_true)).^2);
    SS_res = sum((y_true-y_pred).^2);
    R2 = 1-(SS_res/SS_tot);
end

% ─────────────────────────────────────────────────────────────────────────
function perSegmentReport(pred_pct, gt_pct, label)
% P0 instrumentation: split the record into 10 equal time chunks and report
% RMSE per chunk, so it is visible whether error concentrates at the startup
% chunk (chunk 1) or the discharge cusps rather than being uniform.
    N    = numel(pred_pct);
    nseg = 10;
    edges = round(linspace(1, N+1, nseg+1));
    fprintf('\n── P0: PER-SEGMENT RMSE (%s) ────────────────────────────────────\n', label);
    fprintf('  %-6s %-16s %-10s\n','chunk','time-range (s)','RMSE (%)');
    for k = 1 : nseg
        i1 = edges(k); i2 = edges(k+1)-1;
        r  = sqrt(mean((pred_pct(i1:i2) - gt_pct(i1:i2)).^2));
        fprintf('  %-6d [%6d-%6d]  %-10.4f\n', k, i1, i2, r);
    end
    fprintf('─────────────────────────────────────────────────────────────────────\n');
end

% ─────────────────────────────────────────────────────────────────────────
function abl = currentBiasAblation(net, dlX, cont, cfg, bias_A)
% P4: inject a constant current-sensor bias and compare
%   (a) pure Coulomb counting (corr = 0)           → integrates the bias
%   (b) the network-corrected reconstruction        → can fight it via voltage
% V8.6 FIX: the bias is now ALSO injected into the network's current input
% channel (ch 2), exactly as a real corrupted sensor would present it. The
% V8.5 version fed the network CLEAN inputs, making it impossible in principle
% for the network to counteract a bias it never observed — those P4 rows
% measured correction noise, not robustness. Both reconstructions use the
% identical gated rest re-anchoring and SOC_0 as the labels.

    numBatches = size(dlX, 2);
    L      = cfg.seqLen;
    stride = cont.stride;

    % corrupt the sensor as seen by the network (normalised channel 2)
    if bias_A ~= 0 && isfield(cfg, 'sig_I')
        bf = zeros(cfg.numFeatures, numBatches, L, 'single');
        bf(2,:,:) = bias_A / cfg.sig_I;
        dlX = dlX + dlarray(bf, 'CBT');
    end

    corr_mat = zeros(1, numBatches, L);
    for s = 1 : 64 : numBatches
        e = min(s+63, numBatches);
        net = resetState(net);
        p   = predict(net, dlX(:,s:e,:));
        corr_mat(1, s:e, :) = extractdata(p(1,:,:)) * cfg.dSOC_scale;
    end
    totalPts = (numBatches-1)*stride + L;
    corr_seq = zeros(totalPts,1);
    corr_seq(1:L) = squeeze(corr_mat(1,1,:));
    for b = 2 : numBatches
        g1 = (b-1)*stride + 1; g2 = min(g1+L-1, totalPts); nL = g2-g1+1;
        corr_seq(g1:g2) = squeeze(corr_mat(1,b,1:nL));
    end
    corr_seq(1) = 0;

    N  = min(totalPts, cont.N);
    a  = cont.alpha_ocv;
    Ib = cont.I + bias_A;                  % current with injected sensor bias
    gt = cont.SOC(1:N) * 100;

    SOCa = zeros(N,1); SOCa(1) = cont.SOC_0;     % (a) pure Coulomb, biased current
    SOCb = zeros(N,1); SOCb(1) = cont.SOC_0;     % (b) network-corrected, biased current
    for i = 2 : N
        base = (Ib(i) * cfg.dt) / cont.C(i);
        sa = SOCa(i-1) + base;
        sb = SOCb(i-1) + base + corr_seq(i);
        if cont.gate(i)
            so = max(0, min(1, interp1(cont.eq_s, cont.soc_eq_s, cont.V(i), 'linear','extrap')));
            sa = (1-a)*sa + a*so;
            sb = (1-a)*sb + a*so;
        end
        SOCa(i) = max(0, min(1, sa));
        SOCb(i) = max(0, min(1, sb));
    end
    abl.bias_A      = bias_A;
    abl.rmse_nocorr = sqrt(mean((SOCa*100 - gt).^2));
    abl.rmse_corr   = sqrt(mean((SOCb*100 - gt).^2));
end
