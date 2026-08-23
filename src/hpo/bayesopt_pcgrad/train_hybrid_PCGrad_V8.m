function rmse_val = train_hybrid_PCGrad_V8(params)
% TRAIN_HYBRID_PCGRAD_V8
% BayesOpt objective for the V8.3 INCREMENT-PREDICTION pipeline, trained with
% PCGrad multi-task gradient surgery.
%
% ═══════════════════════════════════════════════════════════════════════════
%  WHAT THIS IS
% ═══════════════════════════════════════════════════════════════════════════
%  This ports the current production model (HA_PIT v8.3) into the
%  PCGrad + Bayesian-Optimisation hyperparameter-search harness. Versus the
%  old objective (absolute-SOC, BiLSTM, prepareBatteryDataV6) it now:
%
%   • Uses prepareBatteryDataV9 — 5 features [V,I,T,C_soh,SOC_ocv_instant],
%     SOC-stratified sample weights dlW, shared scalers, and the continuous
%     un-windowed `cont` struct for reconstruction.
%   • Predicts per-step SOC INCREMENTS (delta-anchoring): the net outputs
%     [dSOC, OCV_corr]; SOC_pred = anchor + cumsum(dSOC) inside the loss.
%   • Drops the BiLSTM (matches v8.3 architecture) and outputs size 2.
%   • Returns the CONTINUOUS-RECONSTRUCTION validation RMSE on B0007 — the
%     same deployment metric v8.3 reports — as the BayesOpt objective.
%
%  PCGrad still arbitrates the three physics losses (data / Coulomb / ECM) by
%  deconflicting their gradients (global Yu-et-al. formulation), instead of
%  the fixed weighted sum used in the standalone v8.3 script.
%
% ═══════════════════════════════════════════════════════════════════════════
%  TUNED HYPERPARAMETERS (8)  — see the master script for bounds
%   initLR, attentionDim, numFilters, dropoutRate, gradClip, warmupEpochs,
%   weightDecay, lambda_end  (lambda_end is the v8.4 endpoint-drift weight; the
%   meaningless for increment regression; it scales the Coulomb-loss gradient
%   so PCGrad sees comparable task magnitudes).
%
%  FIXED (part of the label definition / pipeline, NOT searched):
%   alpha_ocv=0.20, I_rest_thresh=0.05, seqLen=240, dSOC_scale=1e-3,
%   ocv_corr_scale=0.1, empirical OCV, R_ref=0.080 (+aging).
%
%  DEPENDENCY: prepareBatteryDataV9.m and PositionalEncodingLayer.m on path.
% ═══════════════════════════════════════════════════════════════════════════

    %% ── TRIAL COUNTER + REPRODUCIBILITY ─────────────────────────────────
    persistent trialCount
    if isempty(trialCount), trialCount = 0; end
    trialCount = trialCount + 1;
    rng(trialCount, 'twister');     % whole study reproducible on re-run

    fprintf('\n==========================================================\n');
    fprintf('  PCGRAD-V8 (increment pipeline) — TRIAL %02d\n', trialCount);
    fprintf('==========================================================\n');

    %% ══════════════════════════════════════════════════════════════════════
    %%  0.  PERSISTENT DATA + PHYSICS (executes once per MATLAB session)
    %% ══════════════════════════════════════════════════════════════════════
    persistent dataLoaded cfg_static
    persistent dlX_tr dlY_tr dlV_tr dlI_tr dlC_tr dlW_tr
    persistent dlH_tr dlR0_tr blend_chg_tr blend_dchg_tr dlTransientMask_tr
    persistent dlX_val val_cont

    if isempty(dataLoaded)
        fprintf('[DATA] Initialising persistent workspace (once only)...\n');

        % --- static config (everything not searched by BayesOpt) ---
        cfg_static.numFeatures   = 5;          % V,I,T,C_soh,SOC_ocv_instant
        cfg_static.numHeads      = 4;
        cfg_static.ffnExpand     = 2;
        cfg_static.seqLen        = 240;
        cfg_static.pEncScale     = 0.1;
        cfg_static.dt            = 1;
        cfg_static.C_nom         = 6664;
        cfg_static.H_max         = 0.015;
        cfg_static.H_gamma       = 1e-4;
        cfg_static.R_ref         = 0.080;      % measured (was 0.0447)
        cfg_static.R0_alpha      = 0.50;       % cycle-aging exponent
        cfg_static.T_ref_K       = 297.15;
        cfg_static.E_a_R         = 20000 / 8.314;
        cfg_static.k_blend       = 10;
        cfg_static.miniBatchSize = 128;
        % increment-pipeline constants (fixed — part of the label definition)
        cfg_static.dSOC_scale    = 1e-3;
        cfg_static.ocv_corr_scale= 0.1;
        cfg_static.alpha_ocv     = 0.20;
        cfg_static.I_rest_thresh = 0.05;
        % V8.4 relaxation-gated anchoring (must match prepareBatteryDataV9)
        cfg_static.settle_len    = 120;
        cfg_static.relax_skip    = 30;
        cfg_static.dvdt_thr      = 2e-4;
        % V8.4 fixed loss weights (lambda_end is the tuned one)
        cfg_static.w_data        = 1.0;
        cfg_static.lambda_Ah     = 10.0;       % correction-magnitude regulariser (FIXED)
        cfg_static.w_ECM         = 0.5;
        cfg_static.ocv_dchg = [24.009042,-65.838561,68.061132,-32.312351,7.331077,2.749715];
        cfg_static.ocv_chg  = [ 5.340997,-15.748829,16.803693, -8.069274,2.075416,3.801065];

        % --- load training cells with SHARED scalers (v8.3 scheme) ---
        [tX5,tY5,tV5,tI5,tT5,tC5,tW5,~,sc5] = ...
            prepareBatteryDataV9('Merged_B0005_Lifecycle_Sample.csv', cfg_static, []);
        train_scalers = sc5;
        [tX6,tY6,tV6,tI6,tT6,tC6,tW6,~,sc6] = ...
            prepareBatteryDataV9('Merged_B0006_Lifecycle_Sample.csv', cfg_static, train_scalers);
        % average the two cells' scalers for val/test (matches v8.3)
        train_scalers.mu3   = (sc5.mu3   + sc6.mu3)  /2;
        train_scalers.sig3  = (sc5.sig3  + sc6.sig3) /2;
        train_scalers.mu_c  = (sc5.mu_c  + sc6.mu_c) /2;
        train_scalers.sig_c = (sc5.sig_c + sc6.sig_c)/2;
        train_scalers.mu_si = (sc5.mu_si + sc6.mu_si)/2;
        train_scalers.sig_si= (sc5.sig_si+ sc6.sig_si)/2;

        dlX_tr = cat(2, tX5, tX6);   dlY_tr = cat(2, tY5, tY6);
        dlV_tr = cat(2, tV5, tV6);   dlI_tr = cat(2, tI5, tI6);
        dlC_tr = cat(2, tC5, tC6);   dlW_tr = cat(2, tW5, tW6);

        % --- precompute physics tensors for the training windows ---
        I_raw = extractdata(dlI_tr);
        T_raw = extractdata(cat(2, tT5, tT6));
        C_raw = extractdata(dlC_tr);
        [~, nTr, ~] = size(I_raw);

        dI = diff(I_raw, 1, 3);
        dI = cat(3, zeros(1, nTr, 1), dI);
        dlTransientMask_tr = dlarray(1 ./ (1 + (abs(dI)/0.5).^2), 'CBT');

        dlH_tr = dlarray(computeHysteresis(I_raw, cfg_static), 'CBT');

        T_K    = T_raw + 273.15;
        R0_arr = cfg_static.R_ref * exp(cfg_static.E_a_R * (1./T_K - 1/cfg_static.T_ref_K));
        R0_age = R0_arr .* (cfg_static.C_nom ./ max(C_raw, 1000)).^cfg_static.R0_alpha;
        dlR0_tr = dlarray(R0_age, 'CBT');

        blend_chg_tr  = dlarray(1 ./ (1 + exp(-cfg_static.k_blend .* I_raw)), 'CBT');
        blend_dchg_tr = 1 - blend_chg_tr;

        % --- validation cell (B0007) + its continuous arrays ---
        [dlX_val,~,~,~,~,~,~,val_cont,~] = ...
            prepareBatteryDataV9('Merged_B0007_Lifecycle_Sample.csv', cfg_static, train_scalers);

        dataLoaded = true;
        fprintf('[DATA] Ready. Train: %d  Val: %d windows (5 features).\n\n', ...
                size(dlX_tr,2), size(dlX_val,2));
    end

    %% ══════════════════════════════════════════════════════════════════════
    %%  1.  CONFIG (BayesOpt-injected)
    %% ══════════════════════════════════════════════════════════════════════
    % ── OOM GUARD: any out-of-memory during build/train/eval is caught and the
    %    trial returns NaN so BayesOpt records it and moves on (it does not crash).
    rmse_val = NaN;
    try
    cfg = cfg_static;
    cfg.attentionDim = str2double(char(params.attentionDim));
    cfg.numFilters   = str2double(char(params.numFilters));
    cfg.warmupEpochs = str2double(char(params.warmupEpochs));
    cfg.dropoutRate  = params.dropoutRate;
    cfg.initLR       = params.initLR;
    cfg.gradClip     = params.gradClip;
    cfg.weightDecay  = params.weightDecay;
    cfg.lambda_end   = params.lambda_end;      % NEW (continuous, log scale) — endpoint drift

    cfg.numEpochs          = 50;               % BayesOpt sprint length
    cfg.minLR              = 1e-5;
    cfg.maxTotalEpochs     = 400;              % full-training reference horizon
    cfg.cosinePeriodEpochs = cfg.numEpochs;    % anneal WITHIN the sprint (better
                                               % rank-correlation than period=400)
    cfg.curriculumEpochs   = 15;               % freeze ECM task early

    fprintf('  LR=%.1e Dim=%d Fil=%d Drop=%.3f Clip=%.2f WU=%d WD=%.1e Lend=%.2e\n\n', ...
        cfg.initLR, cfg.attentionDim, cfg.numFilters, cfg.dropoutRate, ...
        cfg.gradClip, cfg.warmupEpochs, cfg.weightDecay, cfg.lambda_end);

    %% ══════════════════════════════════════════════════════════════════════
    %%  2.  NETWORK (v8.3 architecture: CNN → LN → proj → posenc → 2×TF → FC2)
    %% ══════════════════════════════════════════════════════════════════════
    dlnet = buildIncrementNet(cfg);

    %% ══════════════════════════════════════════════════════════════════════
    %%  3.  TRAINING LOOP (PCGrad surgery on 3 physics gradients)
    %% ══════════════════════════════════════════════════════════════════════
    avgGradNet = []; avgSqGradNet = [];
    numTotal   = size(dlX_tr, 2);
    numIter    = ceil(numTotal / cfg.miniBatchSize);
    adamIter   = 0;
    lossLog    = zeros(cfg.numEpochs, 5);   % [total, data, Ah, ECM, endpoint]

    for epoch = 1 : cfg.numEpochs
        if epoch <= cfg.warmupEpochs
            currentLR = cfg.initLR * (epoch / cfg.warmupEpochs);
        else
            t_c = epoch - cfg.warmupEpochs;
            T_c = cfg.cosinePeriodEpochs - cfg.warmupEpochs;
            currentLR = cfg.minLR + 0.5*(cfg.initLR - cfg.minLR)*(1 + cos(pi*t_c/T_c));
        end
        in_curriculum = (epoch <= cfg.curriculumEpochs);

        idx = randperm(numTotal);
        epLoss = zeros(numIter, 5);   % [total, data, Ah, ECM, endpoint]

        for it = 1 : numIter
            adamIter = adamIter + 1;
            s = (it-1)*cfg.miniBatchSize + 1;
            e = min(it*cfg.miniBatchSize, numTotal);
            mb = idx(s:e);

            [g_data, g_Ah, g_ECM, st, lv] = dlfeval(@hapitLossPCGrad_V8, ...
                dlnet, dlX_tr(:,mb,:), dlY_tr(:,mb,:), dlI_tr(:,mb,:), dlV_tr(:,mb,:), ...
                dlH_tr(:,mb,:), dlR0_tr(:,mb,:), blend_chg_tr(:,mb,:), blend_dchg_tr(:,mb,:), ...
                dlC_tr(:,mb,:), dlTransientMask_tr(:,mb,:), dlW_tr(:,mb,:), cfg);
            dlnet.State = st;

            % PCGrad: 2 tasks during curriculum (data+Coulomb), 3 after
            if in_curriculum
                gradsNet = performPCGradSurgery({g_data, g_Ah});
            else
                gradsNet = performPCGradSurgery({g_data, g_Ah, g_ECM});
            end

            gradsNet = clipGradients(gradsNet, cfg.gradClip);

            % AdamW decoupled weight decay on weight matrices only
            if cfg.weightDecay > 0
                for k = 1 : height(dlnet.Learnables)
                    if contains(dlnet.Learnables.Parameter{k}, 'Weights')
                        dlnet.Learnables.Value{k} = dlnet.Learnables.Value{k} * ...
                                                    (1 - currentLR * cfg.weightDecay);
                    end
                end
            end

            [dlnet, avgGradNet, avgSqGradNet] = adamupdate( ...
                dlnet, gradsNet, avgGradNet, avgSqGradNet, adamIter, currentLR);

            epLoss(it,:) = lv;
        end
        lossLog(epoch,:) = mean(epLoss, 1);
    end

    %% ══════════════════════════════════════════════════════════════════════
    %%  4.  OBJECTIVE — continuous-reconstruction VAL RMSE (v8.3 metric)
    %% ══════════════════════════════════════════════════════════════════════
    [~, ~, ~, met_val] = evaluateContinuous(dlnet, dlX_val, val_cont, cfg);
    rmse_val = met_val.rmse;

    % also log the cheap windowed (oracle-anchored) RMSE as a diagnostic
    rmse_win = windowedOracleRMSE(dlnet, dlX_val, val_cont, cfg);

    if ~isfinite(rmse_val)
        warning('Trial %02d non-finite RMSE — penalising.', trialCount);
        rmse_val = 1e3;
    end

    %% ── per-trial log ───────────────────────────────────────────────────
    save(sprintf('PCGrad_V8_Trial_%02d_Log.mat', trialCount), ...
         'cfg', 'lossLog', 'rmse_val', 'rmse_win');
    fprintf('Trial %02d | Continuous Val RMSE: %.4f %%  (windowed diag: %.4f %%)\n', ...
            trialCount, rmse_val, rmse_win);

    catch ME
        if isOOM(ME)
            fprintf(2, ['[OOM GUARD] Trial %02d ran out of memory: %s\n' ...
                        '   -> logging NaN, BayesOpt will skip and continue.\n'], ...
                        trialCount, ME.message);
            rmse_val = NaN;
            try, reset(gpuDevice); catch, end
        else
            rethrow(ME);
        end
    end
end

% ─────────────────────────────────────────────────────────────────────────
function tf = isOOM(ME)
% True if the exception looks like an out-of-memory condition (CPU or GPU).
    idl  = lower(ME.identifier);
    msgl = lower(ME.message);
    tf = contains(idl,'nomem') || contains(idl,'oom') || ...
         contains(idl,'outofmemory') || contains(idl,'gpu:array:oom') || ...
         contains(msgl,'out of memory') || contains(msgl,'insufficient memory') || ...
         contains(msgl,'memory allocation');
end


%% ══════════════════════════════════════════════════════════════════════════
%%  LOCAL FUNCTIONS
%% ══════════════════════════════════════════════════════════════════════════
function dlnet = buildIncrementNet(cfg)
% v8.3 architecture (no BiLSTM). Output size 2: [dSOC_raw, OCV_corr_raw].
    lg = layerGraph();
    lg = addLayers(lg, sequenceInputLayer(cfg.numFeatures,'Name','input'));
    lg = addLayers(lg, convolution1dLayer(3, cfg.numFilters,'Padding','same','Name','conv_k3'));
    lg = addLayers(lg, convolution1dLayer(5, cfg.numFilters,'Padding','same','Name','conv_k5'));
    lg = addLayers(lg, convolution1dLayer(9, cfg.numFilters,'Padding','same','Name','conv_k9'));
    lg = addLayers(lg, concatenationLayer(1,3,'Name','concat_cnn'));
    lg = addLayers(lg, layerNormalizationLayer('Name','ln_cnn'));
    lg = addLayers(lg, reluLayer('Name','relu_cnn'));
    lg = addLayers(lg, dropoutLayer(cfg.dropoutRate,'Name','drop_cnn'));
    lg = addLayers(lg, fullyConnectedLayer(cfg.attentionDim,'Name','proj'));
    lg = addLayers(lg, PositionalEncodingLayer(cfg.attentionDim, cfg.pEncScale,'pos_enc'));

    lg = addLayers(lg, selfAttentionLayer(cfg.numHeads, cfg.attentionDim,'Name','self_att_1'));
    lg = addLayers(lg, dropoutLayer(cfg.dropoutRate,'Name','drop_att_1'));
    lg = addLayers(lg, additionLayer(2,'Name','add_att_1'));
    lg = addLayers(lg, layerNormalizationLayer('Name','ln_att_1'));
    lg = addLayers(lg, fullyConnectedLayer(cfg.attentionDim*cfg.ffnExpand,'Name','ffn1_1'));
    lg = addLayers(lg, reluLayer('Name','relu_ffn_1'));
    lg = addLayers(lg, dropoutLayer(cfg.dropoutRate,'Name','drop_ffn_1'));
    lg = addLayers(lg, fullyConnectedLayer(cfg.attentionDim,'Name','ffn2_1'));
    lg = addLayers(lg, additionLayer(2,'Name','add_ffn_1'));
    lg = addLayers(lg, layerNormalizationLayer('Name','ln_ffn_1'));

    lg = addLayers(lg, selfAttentionLayer(cfg.numHeads, cfg.attentionDim,'Name','self_att_2'));
    lg = addLayers(lg, dropoutLayer(cfg.dropoutRate,'Name','drop_att_2'));
    lg = addLayers(lg, additionLayer(2,'Name','add_att_2'));
    lg = addLayers(lg, layerNormalizationLayer('Name','ln_att_2'));
    lg = addLayers(lg, fullyConnectedLayer(cfg.attentionDim*cfg.ffnExpand,'Name','ffn1_2'));
    lg = addLayers(lg, reluLayer('Name','relu_ffn_2'));
    lg = addLayers(lg, dropoutLayer(cfg.dropoutRate,'Name','drop_ffn_2'));
    lg = addLayers(lg, fullyConnectedLayer(cfg.attentionDim,'Name','ffn2_2'));
    lg = addLayers(lg, additionLayer(2,'Name','add_ffn_2'));
    lg = addLayers(lg, layerNormalizationLayer('Name','ln_ffn_2'));
    lg = addLayers(lg, fullyConnectedLayer(2,'Name','net_out'));

    lg = connectLayers(lg,'input','conv_k3');
    lg = connectLayers(lg,'input','conv_k5');
    lg = connectLayers(lg,'input','conv_k9');
    lg = connectLayers(lg,'conv_k3','concat_cnn/in1');
    lg = connectLayers(lg,'conv_k5','concat_cnn/in2');
    lg = connectLayers(lg,'conv_k9','concat_cnn/in3');
    lg = connectLayers(lg,'concat_cnn','ln_cnn');
    lg = connectLayers(lg,'ln_cnn','relu_cnn');
    lg = connectLayers(lg,'relu_cnn','drop_cnn');
    lg = connectLayers(lg,'drop_cnn','proj');
    lg = connectLayers(lg,'proj','pos_enc');
    lg = connectLayers(lg,'pos_enc','self_att_1');
    lg = connectLayers(lg,'self_att_1','drop_att_1');
    lg = connectLayers(lg,'drop_att_1','add_att_1/in1');
    lg = connectLayers(lg,'pos_enc','add_att_1/in2');
    lg = connectLayers(lg,'add_att_1','ln_att_1');
    lg = connectLayers(lg,'ln_att_1','ffn1_1');
    lg = connectLayers(lg,'ffn1_1','relu_ffn_1');
    lg = connectLayers(lg,'relu_ffn_1','drop_ffn_1');
    lg = connectLayers(lg,'drop_ffn_1','ffn2_1');
    lg = connectLayers(lg,'ffn2_1','add_ffn_1/in1');
    lg = connectLayers(lg,'ln_att_1','add_ffn_1/in2');
    lg = connectLayers(lg,'add_ffn_1','ln_ffn_1');
    lg = connectLayers(lg,'ln_ffn_1','self_att_2');
    lg = connectLayers(lg,'self_att_2','drop_att_2');
    lg = connectLayers(lg,'drop_att_2','add_att_2/in1');
    lg = connectLayers(lg,'ln_ffn_1','add_att_2/in2');
    lg = connectLayers(lg,'add_att_2','ln_att_2');
    lg = connectLayers(lg,'ln_att_2','ffn1_2');
    lg = connectLayers(lg,'ffn1_2','relu_ffn_2');
    lg = connectLayers(lg,'relu_ffn_2','drop_ffn_2');
    lg = connectLayers(lg,'drop_ffn_2','ffn2_2');
    lg = connectLayers(lg,'ffn2_2','add_ffn_2/in1');
    lg = connectLayers(lg,'ln_att_2','add_ffn_2/in2');
    lg = connectLayers(lg,'add_ffn_2','ln_ffn_2');
    lg = connectLayers(lg,'ln_ffn_2','net_out');
    dlnet = dlnetwork(lg);
end

% ─────────────────────────────────────────────────────────────────────────
function [grad_data, grad_Ah, grad_ECM, state, lossVals] = hapitLossPCGrad_V8( ...
        net, X, Y_soc, I_seq, V_meas, dlH, dlR0, blend_chg, blend_dchg, ...
        dlC, dlTransientMask, dlW, cfg)
% V8.4 RESIDUAL-increment loss returning THREE per-task gradients for PCGrad.
% dSOC = I*dt/C (rectangular) + correction; SOC_pred = anchor + cumsumTime3(dSOC).
% Task grouping: (1) SOC fidelity = data + endpoint drift, (2) correction
% regulariser, (3) ECM voltage. The endpoint term is part of SOC fidelity so it
% does not conflict with itself, keeping the 3-task PCGrad structure.

    [net_out, state] = forward(net, X);
    net_out = stripdims(net_out);
    Y = stripdims(Y_soc); Iz = stripdims(I_seq); Vz = stripdims(V_meas);
    Hz = stripdims(dlH);  R0z = stripdims(dlR0);
    BCz = stripdims(blend_chg); BDz = stripdims(blend_dchg);
    Cz = stripdims(dlC); TMz = stripdims(dlTransientMask); Wz = stripdims(dlW);

    corr     = net_out(1,:,:) * cfg.dSOC_scale;       % per-step CORRECTION
    OCV_corr = net_out(2,:,:) * cfg.ocv_corr_scale;

    dSOC_phy  = (Iz(:,:,2:end) * cfg.dt) ./ Cz(:,:,2:end);
    dInc      = dSOC_phy + corr(:,:,2:end);
    zero1     = corr(:,:,1) * 0;
    dSOC_full = cat(3, zero1, dInc);
    anchor    = Y(:,:,1);
    SOC_pred  = anchor + cumsumTime3(dSOC_full);

    % task 1 — SOC fidelity: weighted data loss + endpoint drift penalty
    L_data = mean(Wz .* (SOC_pred - Y).^2, 'all');
    L_end  = mean((SOC_pred(:,:,end) - Y(:,:,end)).^2, 'all');
    L_fid  = cfg.w_data*L_data + cfg.lambda_end*L_end;

    % task 2 — correction-magnitude regulariser
    L_Ah   = cfg.lambda_Ah * mean(corr(:,:,2:end).^2, 'all');

    % task 3 — ECM voltage consistency
    OCV_poly = polyOCV_dl(SOC_pred, BCz, BDz, cfg);
    V_hat    = OCV_poly + OCV_corr + Hz + (Iz .* R0z);
    L_ECM    = cfg.w_ECM * mean((V_hat - Vz).^2 .* TMz, 'all');

    grad_data = dlgradient(L_fid, net.Learnables, 'RetainData', true);
    grad_Ah   = dlgradient(L_Ah,  net.Learnables, 'RetainData', true);
    grad_ECM  = dlgradient(L_ECM, net.Learnables);

    lossVals = [extractdata(L_fid)+extractdata(L_Ah)+extractdata(L_ECM), ...
                extractdata(L_data), extractdata(L_Ah), extractdata(L_ECM), ...
                extractdata(L_end)];
end

% ─────────────────────────────────────────────────────────────────────────
function final_grads = performPCGradSurgery(grad_cell)
% Global PCGrad (Yu et al., 2020) over a variable number of task gradients.
% Flattens each task's FULL gradient, projects once on the full vectors, then
% unflattens. (Per-tensor surgery is NOT the algorithm — see notes in V4.)
    T          = numel(grad_cell);
    num_params = height(grad_cell{1});
    epsilon    = 1e-12;

    shapes = cell(num_params,1); numels = zeros(num_params,1);
    parts  = cell(T,1);
    for t = 1:T, parts{t} = cell(num_params,1); end
    for k = 1:num_params
        v0 = extractdata(grad_cell{1}.Value{k});
        shapes{k} = size(v0); numels(k) = numel(v0);
        for t = 1:T
            vt = extractdata(grad_cell{t}.Value{k});
            parts{t}{k} = vt(:);
        end
    end
    G = zeros(sum(numels), T);
    for t = 1:T, G(:,t) = cat(1, parts{t}{:}); end

    orig = G; proj = G;
    for i = 1:T
        gi = proj(:,i);
        others = setdiff(1:T, i);
        others = others(randperm(numel(others)));
        for j = others
            gj = orig(:,j);
            dij = gi.' * gj;
            if dij < 0
                gi = gi - (dij / (gj.'*gj + epsilon)) * gj;
            end
        end
        proj(:,i) = gi;
    end
    g_sum = sum(proj, 2);

    final_grads = grad_cell{1};
    off = 0;
    for k = 1:num_params
        n = numels(k);
        final_grads.Value{k} = dlarray(reshape(g_sum(off+1:off+n), shapes{k}));
        off = off + n;
    end
end

% ─────────────────────────────────────────────────────────────────────────
function y = cumsumTime3(x)
% Autodiff-safe cumulative sum along dim 3 of a [1,B,L] dlarray (triangular
% matmul; cumsum is unsupported for dlarray on some MATLAB versions).
    sz = size(x); B = sz(2); L = sz(3);
    x2 = reshape(x, [B, L]);
    U  = triu(ones(L, L, 'like', extractdata(x)));
    y  = reshape(x2 * U, [1, B, L]);
end

% ─────────────────────────────────────────────────────────────────────────
function V_ocv = polyOCV_dl(soc_dl, blend_chg, blend_dchg, cfg)
    soc_c  = max(0.01, min(0.99, soc_dl));
    pd = cfg.ocv_dchg; pc = cfg.ocv_chg;
    V_dchg = pd(1)*soc_c.^5 + pd(2)*soc_c.^4 + pd(3)*soc_c.^3 + pd(4)*soc_c.^2 + pd(5)*soc_c + pd(6);
    V_chg  = pc(1)*soc_c.^5 + pc(2)*soc_c.^4 + pc(3)*soc_c.^3 + pc(4)*soc_c.^2 + pc(5)*soc_c + pc(6);
    V_ocv  = (V_chg .* blend_chg) + (V_dchg .* blend_dchg);
end

% ─────────────────────────────────────────────────────────────────────────
function [pred_pct, gt_pct, err, metrics] = evaluateContinuous(net, dlX, cont, cfg)
% V8.4 continuous reconstruction: forward-only from cont.SOC_0, residual
% increment (I*dt/C + correction), settled-rest gated anchoring identical to
% the GT (cont.gate, cont.alpha_ocv). Zero-correction reproduces the labels.
    numBatches = size(dlX, 2); L = cfg.seqLen; stride = cont.stride;
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

    N = min(totalPts, cont.N);
    a = cont.alpha_ocv;
    corr_seq(1) = 0;
    SOC = zeros(N,1);
    SOC(1) = cont.SOC_0;
    for i = 2 : N
        inc = (cont.I(i) * cfg.dt) / cont.C(i) + corr_seq(i);
        soc_next = SOC(i-1) + inc;
        if cont.gate(i)
            soc_ocv = max(0, min(1, interp1(cont.eq_s, cont.soc_eq_s, cont.V(i),'linear','extrap')));
            soc_next = (1-a)*soc_next + a*soc_ocv;
        end
        SOC(i) = max(0, min(1, soc_next));
    end

    pred_pct = SOC*100; gt_pct = cont.SOC(1:N)*100;
    err = pred_pct - gt_pct; metrics = metricStruct(gt_pct, pred_pct);
end

% ─────────────────────────────────────────────────────────────────────────
function rmse = windowedOracleRMSE(net, dlX, cont, cfg)
% Cheap diagnostic: anchor each window at the TRUE SOC at its start, integrate
% RESIDUAL increments (I*dt/C + correction) within-window, compare. Isolates
% increment-correction quality from drift/anchoring.
    numBatches = size(dlX,2); L = cfg.seqLen; stride = cont.stride;
    corr = zeros(1, numBatches, L);
    for s = 1:64:numBatches
        e = min(s+63, numBatches);
        net = resetState(net);
        p = predict(net, dlX(:,s:e,:));
        corr(1,s:e,:) = extractdata(p(1,:,:)) * cfg.dSOC_scale;
    end
    totalPts = (numBatches-1)*stride + L;
    SOCp = zeros(totalPts,1);
    for b = 1:numBatches
        g1 = (b-1)*stride+1; g2 = min(g1+L-1, totalPts); nL = g2-g1+1;
        win = zeros(nL,1); win(1) = cont.SOC(min(g1, cont.N));
        for j = 2:nL
            gi = g1+j-1;
            if gi <= cont.N
                win(j) = win(j-1) + (cont.I(gi)*cfg.dt)/cont.C(gi) + squeeze(corr(1,b,j));
            else
                win(j) = win(j-1) + squeeze(corr(1,b,j));
            end
        end
        SOCp(g1:g2) = win;
    end
    SOCp = max(0, min(1, SOCp));
    N = min(totalPts, cont.N);
    rmse = sqrt(mean(((SOCp(1:N) - cont.SOC(1:N))*100).^2));
end

% ─────────────────────────────────────────────────────────────────────────
function m = metricStruct(gt_pct, pred_pct)
    gt_pct = gt_pct(:); pred_pct = pred_pct(:);
    e = pred_pct - gt_pct;
    m.rmse = sqrt(mean(e.^2)); m.mae = mean(abs(e)); m.maxe = max(abs(e));
    p = polyfit(gt_pct, pred_pct, 1);
    m.slope = p(1); m.int = p(2);
    SS_tot = sum((gt_pct-mean(gt_pct)).^2); SS_res = sum((gt_pct-pred_pct).^2);
    m.r2 = 1 - SS_res/SS_tot;
end

% ─────────────────────────────────────────────────────────────────────────
function h = computeHysteresis(I_raw, cfg)
    [~, nBatch, T] = size(I_raw);
    h = zeros(1, nBatch, T);
    for b = 1:nBatch
        h_k = 0.0;
        for k = 1:T
            I_k = I_raw(1,b,k);
            decay = exp(-abs(I_k)*cfg.H_gamma*cfg.dt);
            h_k = h_k*decay + cfg.H_max*sign(I_k)*(1-decay);
            h(1,b,k) = h_k;
        end
    end
end

% ─────────────────────────────────────────────────────────────────────────
function grads = clipGradients(grads, threshold)
    tot = 0;
    for i = 1:height(grads)
        g = extractdata(grads.Value{i}); tot = tot + sum(g(:).^2);
    end
    gn = sqrt(tot);
    if gn > threshold
        sc = threshold/gn;
        for i = 1:height(grads), grads.Value{i} = grads.Value{i}*sc; end
    end
end