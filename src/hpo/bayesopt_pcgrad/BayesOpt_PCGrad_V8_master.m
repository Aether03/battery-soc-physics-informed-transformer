%% BayesOpt_PCGrad_V8_master.m
% MASTER SCRIPT — Bayesian Optimisation of the V8.4 RESIDUAL-INCREMENT
% pipeline trained with PCGrad multi-task gradient surgery.
%
% Objective: train_hybrid_PCGrad_V8  (returns continuous-reconstruction
% validation RMSE on B0007 — the same deployment metric HA_PIT v8.4 reports).
%
% CHANGES vs the absolute-SOC master:
%   • Objective on the v8.4 residual-increment pipeline (prepareBatteryDataV9
%     with relaxation-gated anchoring, 5 features, residual increments, forward-
%     only gated continuous reconstruction).
%   • Search dimension is lambda_end (the endpoint drift weight) — it replaces
%     the obsolete lambda_Ah [1e2,1e6] (under residual increments the network
%     predicts only a small correction, so the old Coulomb-loss scale is moot).
%   • Same 8-D space, identical ranges to the ASHA and BOHB methods.
%   • OOM GUARD: train_hybrid_PCGrad_V8 returns NaN on out-of-memory; bayesopt
%     records the NaN trace and continues to the next trial (best is taken over
%     the finite evaluations).
%
% NOTE: this is untested MATLAB (no local Deep Learning Toolbox); the v8.4
% baseline it optimises is itself still pending a first successful run.

clear; clc; close all;
clear train_hybrid_PCGrad_V8;     % reset persistent trial counter + data cache

fprintf('==========================================================\n');
fprintf('  PCGrad Bayesian Optimisation — V8.4 residual pipeline\n');
fprintf('  Objective: continuous-reconstruction Val RMSE (B0007)\n');
fprintf('==========================================================\n\n');

%% 1. SEARCH SPACE (8 parameters — identical to ASHA/BOHB)
% ─────────────────────────────────────────────────────────────────────
%  initLR:       [1e-4, 5e-3]   log
%  attentionDim: {32,64,128,192,256}   categorical
%  numFilters:   {16,32,48,64}         categorical
%  dropoutRate:  [0.05, 0.35]
%  gradClip:     [0.5, 2.0]
%  warmupEpochs: {20,30,40}            categorical
%  weightDecay:  [1e-6, 5e-4]   log
%  lambda_end:   [1e-2, 5e0]    log    <- v8.4 endpoint drift weight
% ─────────────────────────────────────────────────────────────────────
optimVars = [
    optimizableVariable('initLR',       [1e-4, 5e-3],                    'Transform','log')
    optimizableVariable('attentionDim', {'32','64','128','192','256'},   'Type','categorical')
    optimizableVariable('numFilters',   {'16','32','48','64'},           'Type','categorical')
    optimizableVariable('dropoutRate',  [0.05, 0.35])
    optimizableVariable('gradClip',     [0.5,  2.0])
    optimizableVariable('warmupEpochs', {'20','30','40'},                'Type','categorical')
    optimizableVariable('weightDecay',  [1e-6, 5e-4],                    'Transform','log')
    optimizableVariable('lambda_end',   [1e-2, 5e0],                     'Transform','log')
];

%% 2. WARM-START SEEDS (8-D; categoricals pinned to full level sets)
attDim_cats = {'32','64','128','192','256'};
numFil_cats = {'16','32','48','64'};
wu_cats     = {'20','30','40'};

warm_starts = table( ...
    [3e-4;  1e-3;  5e-4;  2e-3], ...                          % initLR
    categorical({'64';'128';'128';'192'}, attDim_cats), ...   % attentionDim
    categorical({'32';'48';'32';'48'},   numFil_cats), ...    % numFilters
    [0.10;  0.15;  0.10;  0.20], ...                          % dropoutRate
    [1.0;   1.0;   0.8;   1.5],  ...                          % gradClip
    categorical({'20';'30';'20';'30'}, wu_cats), ...          % warmupEpochs
    [1e-5;  1e-5;  5e-5;  1e-4], ...                          % weightDecay
    [0.5;   0.5;   0.3;   2.0], ...                           % lambda_end
    'VariableNames', {'initLR','attentionDim','numFilters','dropoutRate', ...
                      'gradClip','warmupEpochs','weightDecay','lambda_end'});

%% 3. EXECUTE
num_trials = 55;
fprintf('Starting Bayesian Optimisation (%d trials)...\n\n', num_trials);

results = bayesopt(@train_hybrid_PCGrad_V8, optimVars, ...
    'MaxObjectiveEvaluations',  num_trials, ...
    'InitialX',                 warm_starts, ...
    'IsObjectiveDeterministic', false, ...         % PCGrad/minibatch stochastic
    'UseParallel',              false, ...          % DO NOT enable — VRAM limit
    'AcquisitionFunctionName',  'expected-improvement-plus', ...
    'ExplorationRatio',         0.5, ...
    'PlotFcn',                  {@plotMinObjective});

%% 4. EXPORT
fprintf('\n==========================================================\n');
fprintf('  OPTIMISATION COMPLETE — EXPORTING\n');
fprintf('==========================================================\n');

tested_configs   = results.XTrace;
RMSE             = results.ObjectiveTrace;
EvalTime_Seconds = results.ObjectiveEvaluationTimeTrace;
Trial            = (1:height(tested_configs))';
summaryTable = [table(Trial), tested_configs, table(RMSE, EvalTime_Seconds)];
writetable(summaryTable, 'BayesOpt_PCGrad_V8_Summary.csv');
fprintf('Summary exported to: BayesOpt_PCGrad_V8_Summary.csv\n\n');

%% 5. REPORT BEST
best_config = results.XAtMinObjective;
best_rmse   = results.MinObjective;
fprintf('Best continuous Val RMSE : %.4f %%\n', best_rmse);
fprintf('Optimal hyperparameters:\n');
disp(best_config);

fprintf(['\nNEXT STEP: copy these 8 values into HA_PIT_v8.m cfg and run the\n' ...
         'full 400-epoch training to get the final train/val/test numbers.\n']);