%% HA_PIT_test_v8.m  —  STANDALONE INFERENCE / UNSEEN-DATA EVALUATION
% ════════════════════════════════════════════════════════════════════════
%  Loads a trained HA-PIT checkpoint and evaluates it on ANY battery CSV,
%  reproducing the SAME analysis as the training script's final evaluation:
%    • continuous reconstruction (deployment metric) + windowed-oracle
%    • RMSE / MAE / Max / R² table
%    • per-segment RMSE breakdown (10 chunks)
%    • P4 current-bias ablation (V8.6 fixed version: network SEES the bias)
%    • 4 figures: SOC trajectory, error distribution, regression, |error|
%
%  REQUIREMENTS (all on the MATLAB path / working directory):
%    1. The checkpoint .mat produced by HA_PIT_v8.m
%       (contains best_net, cfg, train_scalers)
%    2. prepareBatteryDataV9.m         — the SAME pipeline used in training
%    3. PositionalEncodingLayer.m      — REQUIRED to deserialise the network:
%       loading the checkpoint FAILS without this class on the path
%    4. The CSV to evaluate, with the standard 4 columns:
%       Voltage_measured, Current_measured, Temperature_measured, Time
%
%  IMPORTANT CAVEATS
%    • The saved train_scalers are applied to the new file (as they must be —
%      the network expects inputs in the training normalisation). Data from a
%      very different cell/chemistry will be out-of-distribution and the OCV
%      polynomials (fitted on these NASA cells) will not transfer.
%    • The "ground truth" for an unseen file is constructed by the same
%      Coulomb-counting + gated-anchoring rule as in training. The metric
%      therefore measures deviation from that reference, exactly as in the
%      training script's evaluation.
%    • The evaluation functions below are copied VERBATIM from HA_PIT_v8.m
%      so numbers here are directly comparable to the training run's report.
% ════════════════════════════════════════════════════════════════════════

clear; clc;

%% ───────────────────────── USER SETTINGS ────────────────────────────────
checkpoint_file = 'hapit_v8p6_best_checkpoint.mat';   % or hapit_v8p5_best_checkpoint.mat
test_file       = 'Merged_B0005_338_Cycles.csv'; % ANY unseen CSV (4-column format)
dataset_tag     = 'Full Run data';                       % label used in figures/prints
run_bias_ablation = true;                              % P4 table (0 / 0.05 / 0.10 A)
%% ────────────────────────────────────────────────────────────────────────

fprintf('==========================================================\n');
fprintf('  HA-PIT standalone evaluation\n');
fprintf('  Checkpoint : %s\n', checkpoint_file);
fprintf('  Test file  : %s\n', test_file);
fprintf('==========================================================\n\n');

%% 1. LOAD CHECKPOINT ─────────────────────────────────────────────────────
assert(exist(checkpoint_file,'file')==2, 'Checkpoint not found: %s', checkpoint_file);
assert(exist('PositionalEncodingLayer','class')==8 || ...
       exist('PositionalEncodingLayer.m','file')==2, ...
       'PositionalEncodingLayer.m must be on the path to load the network.');
S = load(checkpoint_file);            % best_net, cfg, train_scalers
dlnet         = S.best_net;
cfg           = S.cfg;
train_scalers = S.train_scalers;
nLearn = sum(cellfun(@numel, dlnet.Learnables.Value));
fprintf('Loaded network: %d learnable parameters.\n', nLearn);

% Back-compat: older checkpoints (pre-V8.6) lack cfg.sig_I, needed by the
% fixed P4 ablation to express a physical bias in normalised input units.
if ~isfield(cfg, 'sig_I')
    cfg.sig_I = double(train_scalers.sig3(2));
end

%% 2. PREPARE THE UNSEEN FILE (same pipeline + SAME scalers as training) ──
assert(exist(test_file,'file')==2, 'Test CSV not found: %s', test_file);
[dlX_t, ~, ~, ~, ~, ~, ~, test_cont, ~] = ...
    prepareBatteryDataV9(test_file, cfg, train_scalers);
nWin = size(dlX_t, 2);
fprintf('Prepared %d windows of %d s from %s.\n\n', nWin, cfg.seqLen, test_file);

%% 3. EVALUATE ────────────────────────────────────────────────────────────
[pred_t, gt_t, err_t, met_t] = evaluateContinuous(dlnet, dlX_t, test_cont, cfg);
[~, ~, ~, met_t_win]         = evaluateWindowedOracle(dlnet, dlX_t, test_cont, cfg);

fprintf('========================================================================\n');
fprintf('     EVALUATION — %s (%s)\n', dataset_tag, test_file);
fprintf('========================================================================\n');
fprintf('%-24s | %-12s\n', 'Metric', 'Value');
fprintf('------------------------------------------------------------------------\n');
fprintf('%-24s | %-11.4f%%\n', 'Continuous RMSE',      met_t.rmse);
fprintf('%-24s | %-11.4f%%\n', 'Continuous MAE',       met_t.mae);
fprintf('%-24s | %-11.4f%%\n', 'Continuous Max',       met_t.maxe);
fprintf('%-24s | %-12.4f\n',   'Continuous R^2',       met_t.r2);
fprintf('%-24s | %-12.4f\n',   'Regression slope',     met_t.slope);
fprintf('------------------------------------------------------------------------\n');
fprintf('%-24s | %-11.4f%%\n', 'Windowed-oracle RMSE', met_t_win.rmse);
fprintf('========================================================================\n');
fprintf('  Continuous = deployment metric; windowed-oracle = correction quality.\n');
fprintf('  Small gap => drift/anchoring solved; error intrinsic to correction.\n');
fprintf('========================================================================\n');

% per-segment breakdown
perSegmentReport(pred_t, gt_t, sprintf('%s (%s)', dataset_tag, test_file));

% P4 current-bias ablation
if run_bias_ablation
    fprintf('\n── P4: CURRENT-BIAS ABLATION ────────────────────────────────────────\n');
    fprintf('  Bias corrupts BOTH the network input (ch2) and the integration.\n');
    fprintf('  Network < Pure-Coulomb at nonzero bias  =>  learned robustness.\n');
    fprintf('  %-14s | %-18s | %-18s\n','Bias (A)','Pure-Coulomb RMSE','Network RMSE');
    for bias = [0.0, 0.05, 0.10]
        abl = currentBiasAblation(dlnet, dlX_t, test_cont, cfg, bias);
        fprintf('  %-14.2f | %-17.4f%% | %-17.4f%%\n', bias, abl.rmse_nocorr, abl.rmse_corr);
    end
    fprintf('─────────────────────────────────────────────────────────────────────\n');
end

%% 4. FIGURES (same four per-dataset views as the training script) ────────
col = [0.15 0.35 0.70];
try
    figure('Name',sprintf('SOC Trajectory — %s',dataset_tag),'Position',[60 60 900 420]);
    plot(gt_t,'k','LineWidth',1.3); hold on;
    plot(pred_t,'--','LineWidth',1.0,'Color',col);
    xlabel('Time step (s)'); ylabel('SOC (%)'); ylim([0 100]); grid on;
    legend('Ground truth','Predicted','Location','best');
    title(sprintf('%s — RMSE = %.4f%%  |  MAE = %.4f%%', dataset_tag, met_t.rmse, met_t.mae));

    figure('Name',sprintf('Error Distribution — %s',dataset_tag),'Position',[90 60 760 420]);
    histogram(err_t, 80, 'Normalization','pdf', ...
              'FaceColor',col,'FaceAlpha',0.70,'EdgeColor','none'); hold on;
    plotDistributionFit(err_t);
    xline( met_t.rmse, ':', 'Color',[0.5 0.5 0.5], 'LineWidth',1.1);
    xline(-met_t.rmse, ':', 'Color',[0.5 0.5 0.5], 'LineWidth',1.1);
    xlabel('Error (%)'); ylabel('PDF'); grid on;
    title(sprintf('%s — RMSE = %.4f%%  |  Max |e| = %.4f%%', dataset_tag, met_t.rmse, met_t.maxe));

    figure('Name',sprintf('Regression — %s',dataset_tag),'Position',[120 60 560 520]);
    step = max(1, floor(numel(gt_t)/4000));
    scatter(gt_t(1:step:end), pred_t(1:step:end), 5, col, 'filled', ...
            'MarkerFaceAlpha',0.30); hold on;
    plot([0 100],[0 100],'k--','LineWidth',1.4);
    xf = linspace(0,100,200);
    plot(xf, met_t.slope*xf + met_t.int, '-', 'Color',col,'LineWidth',1.7);
    xlabel('True SOC (%)'); ylabel('Predicted SOC (%)');
    axis([0 100 0 100]); grid on;
    title(sprintf('%s — R^2 = %.4f  |  slope = %.4f', dataset_tag, met_t.r2, met_t.slope));

    figure('Name',sprintf('Absolute Error — %s',dataset_tag),'Position',[150 60 900 380]);
    plot(abs(err_t),'Color',col,'LineWidth',0.9); hold on;
    yline(met_t.rmse,'k--',sprintf('RMSE %.3f%%',met_t.rmse),'LineWidth',1.2, ...
          'LabelVerticalAlignment','bottom');
    yline(met_t.mae, 'k:', sprintf('MAE %.3f%%', met_t.mae), 'LineWidth',1.0, ...
          'LabelVerticalAlignment','bottom');
    xlabel('Time step (s)'); ylabel('|Error| (%)'); grid on;
    title(sprintf('%s — RMSE = %.4f%%  |  Max = %.4f%%', dataset_tag, met_t.rmse, met_t.maxe));
catch ME
    warning('Plotting skipped: %s', ME.message);
end

%% 5. SAVE ────────────────────────────────────────────────────────────────
[~, base, ~] = fileparts(test_file);
res_file = sprintf('hapit_test_%s.mat', base);
test_results = struct('file',test_file,'metrics',met_t,'metrics_windowed',met_t_win, ...
                      'pred_pct',pred_t,'gt_pct',gt_t,'err',err_t,'cfg',cfg);
save(res_file, 'test_results');
fprintf('\nSaved: %s\n', res_file);

%% ════════════════════════════════════════════════════════════════════════
%%  LOCAL FUNCTIONS — copied VERBATIM from HA_PIT_v8.m (V8.6)
%% ════════════════════════════════════════════════════════════════════════

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
