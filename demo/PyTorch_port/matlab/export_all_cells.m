% Set HAPIT_REPO to the battery-soc-physics-informed-transformer repo root first.
repoPath = getenv('HAPIT_REPO');
assert(~isempty(repoPath), 'Set the HAPIT_REPO environment variable to the source repo root.');
portDir = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repoPath, 'src', 'matlab'));

checkpointPath = fullfile(repoPath, 'models', 'hapit_v8p6_best_checkpoint.mat');
S = load(checkpointPath);
dlnet = S.best_net;
cfg = S.cfg;
train_scalers = S.train_scalers;

cells = {'B0005', 'B0006', 'B0007', 'B0018'};
for c = 1:numel(cells)
    cellName = cells{c};
    testFile = fullfile(repoPath, 'data', sprintf('Merged_%s_Lifecycle_Sample.csv', cellName));
    outPath = fullfile(portDir, sprintf('pipeline_%s.mat', cellName));

    fprintf('\n=== %s ===\n', cellName);
    [dlX_t, ~, ~, ~, ~, ~, ~, cont, ~] = prepareBatteryDataV9(testFile, cfg, train_scalers);

    [pred_pct_official, gt_pct_official, ~, met_official] = evaluateContinuousExport(dlnet, dlX_t, cont, cfg);
    fprintf('Official continuous RMSE on %s: %.4f%%\n', cellName, met_official.rmse);

    X = extractdata(dlX_t);
    export = struct();
    export.X = X;
    export.stride = cont.stride;
    export.I = cont.I;
    export.C = cont.C;
    export.V = cont.V;
    export.gate = double(cont.gate);
    export.SOC_0 = cont.SOC_0;
    export.alpha_ocv = cont.alpha_ocv;
    export.eq_s = cont.eq_s;
    export.soc_eq_s = cont.soc_eq_s;
    export.SOC_true = cont.SOC;
    export.dt = cfg.dt;
    export.dSOC_scale = cfg.dSOC_scale;
    export.official_rmse = met_official.rmse;
    export.official_pred_pct = pred_pct_official;
    save(outPath, '-struct', 'export', '-v7');
    fprintf('Saved %s\n', outPath);
end

function [pred_pct, gt_pct, err, metrics] = evaluateContinuousExport(net, dlX, cont, cfg)
    numBatches = size(dlX, 2);
    L = cfg.seqLen;
    corr_mat = zeros(1, numBatches, L);
    for s = 1 : 64 : numBatches
        e = min(s+63, numBatches);
        net = resetState(net);
        p = predict(net, dlX(:,s:e,:));
        corr_mat(1, s:e, :) = extractdata(p(1,:,:)) * cfg.dSOC_scale;
    end
    [SOC, N] = reconFromCorrExport(corr_mat, cont, cfg);
    pred_pct = SOC * 100;
    gt_pct = cont.SOC(1:N) * 100;
    err = pred_pct - gt_pct;
    metrics = metricStructExport(gt_pct, pred_pct);
end

function [SOC, N] = reconFromCorrExport(corr_mat, cont, cfg)
    numBatches = size(corr_mat, 2);
    L = cfg.seqLen;
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
    corr_seq(1) = 0;
    SOC = zeros(N,1);
    SOC(1) = cont.SOC_0;
    for i = 2 : N
        inc = (cont.I(i) * cfg.dt) / cont.C(i) + corr_seq(i);
        soc_next = SOC(i-1) + inc;
        if cont.gate(i)
            soc_ocv = max(0, min(1, interp1(cont.eq_s, cont.soc_eq_s, cont.V(i), 'linear','extrap')));
            soc_next = (1 - a)*soc_next + a*soc_ocv;
        end
        SOC(i) = max(0, min(1, soc_next));
    end
end

function metrics = metricStructExport(gt_pct, pred_pct)
    err = pred_pct - gt_pct;
    metrics.rmse = sqrt(mean(err.^2));
    metrics.mae = mean(abs(err));
    metrics.maxe = max(abs(err));
    ss_res = sum(err.^2);
    ss_tot = sum((gt_pct - mean(gt_pct)).^2);
    metrics.r2 = 1 - ss_res/ss_tot;
    p = polyfit(gt_pct, pred_pct, 1);
    metrics.slope = p(1);
    metrics.int = p(2);
end
