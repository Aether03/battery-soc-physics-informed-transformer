function val_rmse = train_BOHB_V3(trial_id, initLR, attentionDim, ...
        numFilters, dropoutRate, gradClip, warmupEpochs, weightDecay, ...
        lambda_end, target_epoch)
% TRAIN_BOHB_V3
% Chunked (checkpoint/resume) training function for BOHB (Hyperband) multi-fidelity HPO,
% ported to the V8.3 INCREMENT-PREDICTION pipeline.
%
% ═══════════════════════════════════════════════════════════════════════════
%  WHAT CHANGED vs the original BOHB (now aligned to HA_PIT v8.4)
% ═══════════════════════════════════════════════════════════════════════════
%  Architecture/pipeline brought in line with HA_PIT v8.3 (identical to the
%  PCGrad-V8 objective, so all three HPO methods optimise the SAME model):
%    • prepareBatteryDataV9 — 5 features [V,I,T,C_soh,SOC_ocv_instant],
%      SOC-stratified weights dlW, shared scalers, continuous `cont` struct.
%    • Predicts per-step INCREMENTS (delta-anchoring): net outputs
%      [dSOC, OCV_corr]; SOC_pred = anchor + cumsumTime3(dSOC).
%    • BiLSTM removed (matches v8.3). Output size 2.
%    • Empirical OCV poly5 + R_ref=0.080 with cycle-aging.
%    • V8.4 RESIDUAL loss: w_data*L_data + lambda_Ah*mean(corr^2) +
%      lambda_end*L_end + w_ECM*L_ECM (lambda_Ah fixed=10; lambda_end tuned)
%      15-epoch ECM-freeze curriculum — NOT the old homoscedastic weights
%      (which went unstable) and NOT PCGrad (that's the other method).
%    • Objective = CONTINUOUS-RECONSTRUCTION val RMSE on B0007 (v8.3 metric),
%      with the bidirectional startup anchor.
%
%  HPO mechanics retained from V3 (these are the BOHB-specific parts):
%    • Chunked train-to-`target_epoch`, checkpoint, resume.
%    • Global Adam step counter (no bias-correction spike on resume).
%    • Chunked validation forward pass (OOM-safe).
%
%  SEARCH SPACE — identical types/ranges to the BayesOpt/PCGrad method
%  (initLR, attentionDim, numFilters, dropoutRate, gradClip, warmupEpochs,
%   weightDecay, lambda_end). weightDecay + lambda_end are the shared knobs so
%   methods share one space; alpha_ocv/seqLen/output-scales stay fixed (they
%   are part of the label definition, not free knobs).
%
%  DEPENDENCY: prepareBatteryDataV9.m + PositionalEncodingLayer.m on path.
%  numEpochs is 400; the Python milestones must end at 400.
% ═══════════════════════════════════════════════════════════════════════════

    %% 1. PERSISTENT DATA + PHYSICS (loaded once per MATLAB session)
    persistent p_dlX_tr p_dlY_tr p_dlI_tr p_dlV_tr p_dlC_tr p_dlW_tr
    persistent p_dlH_tr p_dlR0_tr p_blend_chg_tr p_blend_dchg_tr p_dlTransientMask_tr
    persistent p_dlX_val p_val_cont p_cfg_base

    if isempty(p_dlX_tr)
        fprintf('--- INITIALISING PERSISTENT V8 DATASET IN RAM ---\n');

        % --- static config (v8.3) ---
        p_cfg_base.numFeatures   = 5;
        p_cfg_base.numHeads      = 4;
        p_cfg_base.ffnExpand     = 2;
        p_cfg_base.seqLen        = 240;
        p_cfg_base.pEncScale     = 0.1;
        p_cfg_base.numEpochs     = 400;       % full-fidelity horizon (cosine period)
        p_cfg_base.minLR         = 1e-5;
        p_cfg_base.curriculumEpochs = 15;
        p_cfg_base.dt            = 1;
        p_cfg_base.C_nom         = 6664;
        p_cfg_base.H_max         = 0.015;
        p_cfg_base.H_gamma       = 1e-4;
        p_cfg_base.R_ref         = 0.080;
        p_cfg_base.R0_alpha      = 0.50;
        p_cfg_base.T_ref_K       = 297.15;
        p_cfg_base.E_a_R         = 20000 / 8.314;
        p_cfg_base.k_blend       = 10;
        p_cfg_base.dSOC_scale    = 1e-3;
        p_cfg_base.ocv_corr_scale= 0.1;
        p_cfg_base.alpha_ocv     = 0.20;
        p_cfg_base.I_rest_thresh = 0.05;
        % V8.4 relaxation-gated anchoring (must match prepareBatteryDataV9)
        p_cfg_base.settle_len    = 120;
        p_cfg_base.relax_skip    = 30;
        p_cfg_base.dvdt_thr      = 2e-4;
        % V8.4 fixed loss weights (lambda_end is the tuned one; rest held at baseline)
        p_cfg_base.w_data        = 1.0;
        p_cfg_base.lambda_Ah     = 10.0;     % correction-magnitude regulariser (FIXED)
        p_cfg_base.w_ECM         = 0.5;
        p_cfg_base.ocv_dchg = [24.009042,-65.838561,68.061132,-32.312351,7.331077,2.749715];
        p_cfg_base.ocv_chg  = [ 5.340997,-15.748829,16.803693, -8.069274,2.075416,3.801065];

        % --- training cells with shared scalers (v8.3 scheme) ---
        [tX5,tY5,tV5,tI5,tT5,tC5,tW5,~,sc5] = ...
            prepareBatteryDataV9('Merged_B0005_Lifecycle_Sample.csv', p_cfg_base, []);
        sc = sc5;
        [tX6,tY6,tV6,tI6,tT6,tC6,tW6,~,sc6] = ...
            prepareBatteryDataV9('Merged_B0006_Lifecycle_Sample.csv', p_cfg_base, sc);
        sc.mu3=(sc5.mu3+sc6.mu3)/2;     sc.sig3=(sc5.sig3+sc6.sig3)/2;
        sc.mu_c=(sc5.mu_c+sc6.mu_c)/2;  sc.sig_c=(sc5.sig_c+sc6.sig_c)/2;
        sc.mu_si=(sc5.mu_si+sc6.mu_si)/2; sc.sig_si=(sc5.sig_si+sc6.sig_si)/2;

        p_dlX_tr = cat(2, tX5, tX6);  p_dlY_tr = cat(2, tY5, tY6);
        p_dlV_tr = cat(2, tV5, tV6);  p_dlI_tr = cat(2, tI5, tI6);
        p_dlC_tr = cat(2, tC5, tC6);  p_dlW_tr = cat(2, tW5, tW6);

        % --- physics tensors for the training windows ---
        I_raw = extractdata(p_dlI_tr);
        T_raw = extractdata(cat(2, tT5, tT6));
        C_raw = extractdata(p_dlC_tr);
        [~, nTr, ~] = size(I_raw);

        dI = diff(I_raw, 1, 3);
        dI = cat(3, zeros(1, nTr, 1), dI);
        p_dlTransientMask_tr = dlarray(1 ./ (1 + (abs(dI)/0.5).^2), 'CBT');
        p_dlH_tr = dlarray(computeHysteresis(I_raw, p_cfg_base), 'CBT');
        T_K    = T_raw + 273.15;
        R0_arr = p_cfg_base.R_ref * exp(p_cfg_base.E_a_R * (1./T_K - 1/p_cfg_base.T_ref_K));
        p_dlR0_tr = dlarray(R0_arr .* (p_cfg_base.C_nom ./ max(C_raw,1000)).^p_cfg_base.R0_alpha, 'CBT');
        p_blend_chg_tr  = dlarray(1 ./ (1 + exp(-p_cfg_base.k_blend .* I_raw)), 'CBT');
        p_blend_dchg_tr = 1 - p_blend_chg_tr;

        % --- validation cell + continuous arrays ---
        [p_dlX_val,~,~,~,~,~,~,p_val_cont,~] = ...
            prepareBatteryDataV9('Merged_B0007_Lifecycle_Sample.csv', p_cfg_base, sc);

        fprintf('--- PERSISTENT V8 DATA LOADED: %d train / %d val windows ---\n', ...
                size(p_dlX_tr,2), size(p_dlX_val,2));
    end

    %% 2. CHECKPOINT MANAGEMENT & ARCHITECTURE INIT
    % ── OOM GUARD: any out-of-memory (CPU or GPU) during training/eval is
    %    caught, logged, and returned as NaN so the HPO study skips this trial
    %    and continues instead of crashing the whole run.
    val_rmse = NaN;
    try
    ckpt_file = sprintf('checkpoint_bohb_V3_%d.mat', trial_id);

    if isfile(ckpt_file)
        load(ckpt_file, 'dlnet', 'avgGradNet', 'avgSqGradNet', ...
             'current_epoch', 'global_step', 'cfg');
        start_epoch = current_epoch + 1;
        fprintf('Trial %d: Resuming epoch %d -> %d\n', trial_id, start_epoch, target_epoch);
    else
        start_epoch = 1;
        global_step = 0;
        cfg = p_cfg_base;
        cfg.initLR       = initLR;
        cfg.attentionDim = round(attentionDim);
        cfg.numFilters   = round(numFilters);
        cfg.dropoutRate  = dropoutRate;
        cfg.gradClip     = gradClip;
        cfg.warmupEpochs = round(warmupEpochs);
        cfg.weightDecay  = weightDecay;     % NEW (shared space)
        cfg.lambda_end   = lambda_end;      % NEW (shared space) — endpoint drift weight

        fprintf(['Trial %d: NEW -> epoch %d  (LR:%.1e Dim:%d Fil:%d Drop:%.2f ' ...
                 'Clip:%.2f WU:%d WD:%.1e Lend:%.2e)\n'], trial_id, target_epoch, ...
                 initLR, cfg.attentionDim, cfg.numFilters, dropoutRate, gradClip, ...
                 cfg.warmupEpochs, weightDecay, lambda_end);

        dlnet = buildIncrementNet(cfg);
        avgGradNet = []; avgSqGradNet = [];
    end

    %% 3. TRAINING LOOP CHUNK (start_epoch -> target_epoch)
    cfg.miniBatchSize = 512;
    numTotal = size(p_dlX_tr, 2);
    numIter  = ceil(numTotal / cfg.miniBatchSize);

    for epoch = start_epoch : target_epoch
        % cosine-with-warmup over the FULL 400-epoch horizon (true fidelity)
        if epoch <= cfg.warmupEpochs
            currentLR = cfg.initLR * (epoch / cfg.warmupEpochs);
        else
            t_c = epoch - cfg.warmupEpochs;
            T_c = cfg.numEpochs - cfg.warmupEpochs;
            currentLR = cfg.minLR + 0.5*(cfg.initLR - cfg.minLR)*(1 + cos(pi*t_c/T_c));
        end
        in_curriculum = (epoch <= cfg.curriculumEpochs);

        idx = randperm(numTotal);
        for it = 1 : numIter
            s = (it-1)*cfg.miniBatchSize + 1;
            e = min(it*cfg.miniBatchSize, numTotal);
            mb = idx(s:e);

            [gradsNet, st, ~] = dlfeval(@hapitLoss_V8, ...
                dlnet, p_dlX_tr(:,mb,:), p_dlY_tr(:,mb,:), p_dlI_tr(:,mb,:), ...
                p_dlV_tr(:,mb,:), p_dlH_tr(:,mb,:), p_dlR0_tr(:,mb,:), ...
                p_blend_chg_tr(:,mb,:), p_blend_dchg_tr(:,mb,:), p_dlC_tr(:,mb,:), ...
                p_dlTransientMask_tr(:,mb,:), p_dlW_tr(:,mb,:), cfg, in_curriculum);
            dlnet.State = st;

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

            global_step = global_step + 1;     % monotonic across chunks
            [dlnet, avgGradNet, avgSqGradNet] = adamupdate( ...
                dlnet, gradsNet, avgGradNet, avgSqGradNet, global_step, currentLR);
        end
    end

    %% 4. VALIDATION — continuous reconstruction RMSE (v8.3 metric)
    [~, ~, ~, met_val] = evaluateContinuous(dlnet, p_dlX_val, p_val_cont, cfg);
    val_rmse = met_val.rmse;
    if ~isfinite(val_rmse), val_rmse = 1e3; end

    %% 5. SAVE CHECKPOINT FOR NEXT BOHB CHUNK
    current_epoch = target_epoch;
    save(ckpt_file, 'dlnet', 'avgGradNet', 'avgSqGradNet', ...
         'current_epoch', 'global_step', 'cfg');

    catch ME
        if isOOM(ME)
            fprintf(2, ['[OOM GUARD] Trial %d ran out of memory ' ...
                        '(epoch target %d): %s\n   -> logging NaN, continuing.\n'], ...
                        trial_id, target_epoch, ME.message);
            val_rmse = NaN;
            try, reset(gpuDevice); catch, end   % free GPU memory if present
        else
            rethrow(ME);                          % real bug -> surface it
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
%%  LOCAL FUNCTIONS  (shared with the PCGrad-V8 objective)
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
function [gradsNet, state, lossVals] = hapitLoss_V8( ...
        net, X, Y_soc, I_seq, V_meas, dlH, dlR0, blend_chg, blend_dchg, ...
        dlC, dlTransientMask, dlW, cfg, in_curriculum)
% V8.4 RESIDUAL-increment fixed-weight loss (single merged gradient).
% dSOC = I*dt/C (rectangular, matches GT) + network correction. SOC_pred =
% anchor + cumsumTime3(dSOC). Adds an endpoint drift penalty.
    [net_out, state] = forward(net, X);
    net_out = stripdims(net_out);
    Y = stripdims(Y_soc); Iz = stripdims(I_seq); Vz = stripdims(V_meas);
    Hz = stripdims(dlH);  R0z = stripdims(dlR0);
    BCz = stripdims(blend_chg); BDz = stripdims(blend_dchg);
    Cz = stripdims(dlC); TMz = stripdims(dlTransientMask); Wz = stripdims(dlW);

    corr     = net_out(1,:,:) * cfg.dSOC_scale;       % per-step CORRECTION
    OCV_corr = net_out(2,:,:) * cfg.ocv_corr_scale;

    % residual increment: exact rectangular physics + correction
    dSOC_phy  = (Iz(:,:,2:end) * cfg.dt) ./ Cz(:,:,2:end);
    dInc      = dSOC_phy + corr(:,:,2:end);
    zero1     = corr(:,:,1) * 0;
    dSOC_full = cat(3, zero1, dInc);
    anchor    = Y(:,:,1);
    SOC_pred  = anchor + cumsumTime3(dSOC_full);

    L_data = mean(Wz .* (SOC_pred - Y).^2, 'all');
    L_Ah   = mean(corr(:,:,2:end).^2, 'all');                 % correction regulariser
    L_end  = mean((SOC_pred(:,:,end) - Y(:,:,end)).^2, 'all');% endpoint drift

    OCV_poly = polyOCV_dl(SOC_pred, BCz, BDz, cfg);
    V_hat    = OCV_poly + OCV_corr + Hz + (Iz .* R0z);
    L_ECM    = mean((V_hat - Vz).^2 .* TMz, 'all');

    w_ECM = cfg.w_ECM;
    if in_curriculum, w_ECM = 0; end                          % freeze ECM early
    totalLoss = cfg.w_data*L_data + cfg.lambda_Ah*L_Ah + ...
                cfg.lambda_end*L_end + w_ECM*L_ECM;

    gradsNet = dlgradient(totalLoss, net.Learnables);
    lossVals = [extractdata(totalLoss), extractdata(L_data), ...
                extractdata(L_Ah), extractdata(L_ECM), extractdata(L_end)];
end

% ─────────────────────────────────────────────────────────────────────────
function y = cumsumTime3(x)
% Autodiff-safe cumulative sum along dim 3 of [1,B,L] (triangular matmul).
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
function m = metricStruct(gt_pct, pred_pct)
    gt_pct = gt_pct(:); pred_pct = pred_pct(:);
    e = pred_pct - gt_pct;
    m.rmse = sqrt(mean(e.^2)); m.mae = mean(abs(e)); m.maxe = max(abs(e));
    p = polyfit(gt_pct, pred_pct, 1); m.slope = p(1); m.int = p(2);
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