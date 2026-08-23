function [dlX, dlY, dlV, dlI, dlT, dlC, dlW, cont, scalers] = prepareBatteryDataV9(filename, cfg, scalers_in)
% PREPAREBATTERYDATAV9  Data pipeline for the INCREMENT-PREDICTION overhaul.
%
% ═══════════════════════════════════════════════════════════════════════════
%  WHY V9 EXISTS — THE ARCHITECTURAL REASON
% ═══════════════════════════════════════════════════════════════════════════
%  The previous models predicted ABSOLUTE SOC inside an isolated 240 s window.
%  In the flat OCV plateau (≈3.5–3.7 V across 30–80 % SOC) a single window
%  carries almost no information about absolute SOC, so windowed absolute
%  prediction has a hard accuracy floor of several percent — no amount of
%  tuning breaks it.
%
%  The fix is to change WHAT the network predicts. HA_PIT_v8 predicts the
%  per-step SOC INCREMENT dSOC[k], supervised by Coulomb counting
%  (dSOC ≈ I·dt/C). SOC is then reconstructed by *continuous* integration of
%  those increments across the whole sequence, anchored at rest periods via
%  the open-circuit-voltage (OCV) reading. This is exactly how a production
%  BMS fuses Coulomb counting with periodic OCV recalibration, and it is the
%  only route to <0.5 % RMSE on the flat plateau.
%
%  V9 therefore additionally returns a CONTINUOUS (un-windowed) signal struct
%  `cont` that HA_PIT_v8 uses to perform the continuous SOC reconstruction at
%  evaluation time.
%
% ═══════════════════════════════════════════════════════════════════════════
%  CHANGES vs V8
% ═══════════════════════════════════════════════════════════════════════════
%  NEW-1  5th feature is now SOC_OCV_INSTANT (a per-timestep voltage-based SOC
%         reading) instead of the broadcast initial SOC. The network learns to
%         trust it when reliable (steep regions / rests) and to fall back on
%         current integration when not (flat plateau). This is the learned
%         "Kalman-gain" fusion signal. Delta-anchoring already supplies the
%         starting point, so the broadcast initial-SOC feature is redundant.
%
%  NEW-2  GENTLE OCV ANCHORING re-enabled at rest steps in the ground-truth
%         construction (cfg.alpha_ocv, default 0.05, applied only when
%         |I| < cfg.I_rest_thresh). A single gentle pull per rest second makes
%         sustained rests converge geometrically to the OCV reading, which
%         RESETS Coulomb-counting drift between charge/discharge segments and
%         bounds error accumulation to within a single segment (~2800 s).
%         The SAME anchoring is applied identically to ALL datasets (train,
%         val, test) using the same empirical equilibrium OCV curve, so the
%         labels stay mutually consistent. HA_PIT_v8 replicates this exact
%         rule in its continuous reconstruction.
%
%  NEW-3  Returns `cont` struct with continuous arrays:
%           cont.V .I .T .C .SOC .R0 .N   (column vectors, length N)
%         plus cont.stride / cont.seqLen for window↔continuous mapping.
%
%  RETAINED FROM V8 (all data-driven fixes):
%    - Empirical poly5 OCV coefficients (DA-1)
%    - SOC-stratified sample weights dlW (DA-2)
%    - Voltage hard-clip to [2.0, 4.25] (DA-4)
%    - Zero-phase Butterworth Wn=0.20 (DA-5)
%    - seqLen handled by cfg (DA-6)
%    - Shared normalisation scalers, dynamic capacity carry-forward,
%      active-phase threshold 0.1 A.
%
% ═══════════════════════════════════════════════════════════════════════════
%  OUTPUTS (dlarray 'CBT' unless noted)
%    dlX  — normalised [V, I, T, C_soh, SOC_ocv_instant]   [5, B, seqLen]
%    dlY  — ground-truth SOC ∈ [0,1]                        [1, B, seqLen]
%    dlV  — raw terminal voltage (V)                         [1, B, seqLen]
%    dlI  — raw current (A)                                  [1, B, seqLen]
%    dlT  — raw temperature (°C)                             [1, B, seqLen]
%    dlC  — raw dynamic capacity (A·s)                       [1, B, seqLen]
%    dlW  — per-window SOC stratification weight             [1, B, 1]
%    cont — struct of continuous signals for eval recon      (plain arrays)
%    scalers — normalisation struct (reuse for val/test)

    %% 1. Load & unique-sort raw CSV ──────────────────────────────────────
    raw         = readtable(filename);
    [t_raw, ui] = unique(raw.Time);
    V_raw       = raw.Voltage_measured(ui);
    I_raw       = raw.Current_measured(ui);
    T_raw       = raw.Temperature_measured(ui);

    % DA-4: hard-clip physically impossible voltage spikes before interp
    V_raw = max(2.0, min(4.25, V_raw));

    %% 2. Resample to 1 Hz ────────────────────────────────────────────────
    t_reg = (0 : cfg.dt : floor(t_raw(end)))';
    V_reg = interp1(t_raw, V_raw, t_reg, 'linear', 'extrap');
    I_reg = interp1(t_raw, I_raw, t_reg, 'linear', 'extrap');
    T_reg = interp1(t_raw, T_raw, t_reg, 'linear', 'extrap');
    N     = numel(t_reg);

    %% 2b. DA-5: Zero-phase Butterworth (Wn=0.20) ────────────────────────
    try
        [b_filt, a_filt] = butter(4, 0.20, 'low');
        V_reg = filtfilt(b_filt, a_filt, V_reg);
        I_reg = filtfilt(b_filt, a_filt, I_reg);
        filter_method = 'Butterworth filtfilt (order 4, Wn=0.20)';
    catch
        win    = 3;
        kernel = ones(win, 1) / win;
        V_reg  = conv(V_reg, kernel, 'same');
        I_reg  = conv(I_reg, kernel, 'same');
        filter_method = 'Moving-average fallback (no Signal Toolbox)';
        warning('prepareBatteryDataV9: filtfilt unavailable, using fallback.');
    end

    %% 3. Config compatibility ─────────────────────────────────────────────
    E_a_R   = 20000 / 8.314;  if isfield(cfg,'E_a_R'),   E_a_R   = cfg.E_a_R;   end
    T_ref_K = 297.15;         if isfield(cfg,'T_ref_K'), T_ref_K = cfg.T_ref_K; end
    R_ref   = 0.080;          if isfield(cfg,'R_ref'),   R_ref   = cfg.R_ref;   end
    R0_alpha= 0.50;           if isfield(cfg,'R0_alpha'),R0_alpha= cfg.R0_alpha;end
    alpha_ocv   = 0.20;       if isfield(cfg,'alpha_ocv'),    alpha_ocv   = cfg.alpha_ocv;    end
    I_rest_thr  = 0.05;       if isfield(cfg,'I_rest_thresh'),I_rest_thr  = cfg.I_rest_thresh;end
    % P2 — relaxation-gated anchoring parameters (moderate strictness)
    settle_len  = 120;        if isfield(cfg,'settle_len'),   settle_len  = cfg.settle_len;   end
    relax_skip  = 30;         if isfield(cfg,'relax_skip'),   relax_skip  = cfg.relax_skip;   end
    dvdt_thr    = 2e-4;       if isfield(cfg,'dvdt_thr'),     dvdt_thr    = cfg.dvdt_thr;     end

    %% 3b. Dynamic SOH estimator ──────────────────────────────────────────
    C_dynamic = ones(N, 1) * cfg.C_nom;
    is_active = abs(I_reg) > 0.1;
    change    = diff([0; is_active]);
    a_starts  = find(change ==  1);
    a_ends    = find(change == -1);
    min_len   = min(numel(a_starts), numel(a_ends));
    a_starts  = a_starts(1:min_len);
    a_ends    = a_ends(1:min_len);
    for k = 1 : numel(a_starts)
        s = a_starts(k); e = a_ends(k);
        phase_Ah = sum(abs(I_reg(s:e))) * cfg.dt;
        if phase_Ah > 1000
            C_dynamic(s:e) = phase_Ah;
        end
    end
    last_C = cfg.C_nom;
    for i = 1 : N
        if C_dynamic(i) ~= cfg.C_nom
            last_C = C_dynamic(i);
        else
            C_dynamic(i) = last_C;
        end
    end

    %% 3c. Per-timestep R0 (Arrhenius × cycle-aging) ──────────────────────
    % Needed for (a) the IR-compensated OCV-instant feature and (b) the
    % continuous reconstruction in HA_PIT_v8. Same model as the main script.
    T_K_cont = T_reg + 273.15;
    R0_arr   = R_ref * exp(E_a_R * (1 ./ T_K_cont - 1/T_ref_K));
    R0_cont  = R0_arr .* (cfg.C_nom ./ max(C_dynamic, 1000)).^R0_alpha;

    %% 4. OCV–SOC LUTs (empirical poly5, DA-1) ────────────────────────────
    soc_lut = linspace(0, 1, 2000)';
    p_dchg = [24.009042, -65.838561, 68.061132, -32.312351, 7.331077, 2.749715];
    p_chg  = [ 5.340997, -15.748829, 16.803693,  -8.069274, 2.075416, 3.801065];
    if isfield(cfg,'ocv_dchg'), p_dchg = cfg.ocv_dchg; end
    if isfield(cfg,'ocv_chg'),  p_chg  = cfg.ocv_chg;  end

    ocv_dchg = polyval(p_dchg, soc_lut);
    ocv_chg  = polyval(p_chg,  soc_lut);
    ocv_eq   = (ocv_dchg + ocv_chg) / 2;

    [dchg_s, di] = sort(ocv_dchg); soc_dchg_s = soc_lut(di);
    [chg_s,  ci] = sort(ocv_chg);  soc_chg_s  = soc_lut(ci);
    [eq_s,   ei] = sort(ocv_eq);   soc_eq_s   = soc_lut(ei);

    % helper: sign-dependent OCV→SOC inversion of an IR-compensated voltage
    invOCV = @(ocv, I) localInvOCV(ocv, I, ...
        chg_s, soc_chg_s, dchg_s, soc_dchg_s, eq_s, soc_eq_s);

    %% 5. Initial SOC (one-shot OCV estimate) ─────────────────────────────
    I_probe = mean(I_reg(1 : min(10, N)));
    OCV_0   = V_reg(1) - I_reg(1) * R0_cont(1);
    if abs(I_probe) < 0.1
        SOC_0 = interp1(eq_s,   soc_eq_s,   OCV_0, 'linear', 'extrap'); ctype='rest';
    elseif I_probe > 0
        SOC_0 = interp1(chg_s,  soc_chg_s,  OCV_0, 'linear', 'extrap'); ctype='charge';
    else
        SOC_0 = interp1(dchg_s, soc_dchg_s, OCV_0, 'linear', 'extrap'); ctype='discharge';
    end
    SOC_0 = max(0, min(1, SOC_0));

    %% 5b. RELAXATION-GATED REST MASK (P2) ────────────────────────────────
    % Only a rest that is BOTH sustained (>= settle_len) AND settled (trailing
    % |dV/dt| below dvdt_thr) earns the right to re-anchor SOC to its OCV. This
    % rejects brief / still-relaxing rests whose terminal voltage is far (~0.5 V
    % measured) from equilibrium and would otherwise inject large SOC errors.
    % The gate turns ON only for the settled TAIL of each qualifying rest (after
    % the first relax_skip seconds). HA_PIT_v8 reuses this exact mask (cont.gate)
    % so labels and reconstruction anchor identically.
    rest_gate = settledRestGate(I_reg, V_reg, I_rest_thr, settle_len, relax_skip, dvdt_thr, cfg.dt);

    %% 6. Ground-truth SOC: Coulomb counting + RELAXATION-GATED anchoring (P2)
    SOC_reg    = zeros(N, 1);
    SOC_reg(1) = SOC_0;
    n_clamp    = 0; n_anchor = 0;
    for i = 2 : N
        SOC_next = SOC_reg(i-1) + (I_reg(i) * cfg.dt) / C_dynamic(i);
        if rest_gate(i)
            % settled rest: terminal voltage ≈ equilibrium OCV → recalibrate
            soc_ocv  = interp1(eq_s, soc_eq_s, V_reg(i), 'linear', 'extrap');
            soc_ocv  = max(0, min(1, soc_ocv));
            SOC_next = (1 - alpha_ocv) * SOC_next + alpha_ocv * soc_ocv;
            n_anchor = n_anchor + 1;
        end
        if SOC_next > 1.0 || SOC_next < 0.0, n_clamp = n_clamp + 1; end
        SOC_reg(i) = max(0.0, min(1.0, SOC_next));
    end

    %% 6a. P0 INSTRUMENTATION — reconstruct the OLD label rule (anchor at EVERY
    % rest step, non-relaxed voltage) purely to quantify how far the labels
    % shifted under the new gated rule. Not used for training.
    SOC_old      = zeros(N, 1);
    SOC_old(1)   = SOC_0;
    n_anchor_old = 0;
    for i = 2 : N
        sn = SOC_old(i-1) + (I_reg(i) * cfg.dt) / C_dynamic(i);
        if abs(I_reg(i)) < I_rest_thr
            so = max(0, min(1, interp1(eq_s, soc_eq_s, V_reg(i), 'linear', 'extrap')));
            sn = (1 - alpha_ocv) * sn + alpha_ocv * so;
            n_anchor_old = n_anchor_old + 1;
        end
        SOC_old(i) = max(0.0, min(1.0, sn));
    end
    label_shift = max(abs(SOC_reg - SOC_old)) * 100;

    %% 6b. NEW-1: per-timestep voltage-based SOC reading (fusion feature) ──
    OCV_comp_cont = V_reg - I_reg .* R0_cont;          % IR-compensated voltage
    SOC_ocv_inst  = invOCV(OCV_comp_cont, I_reg);      % vectorised inside helper
    SOC_ocv_inst  = max(0, min(1, SOC_ocv_inst));

    %% Diagnostics ────────────────────────────────────────────────────────
    fprintf('\n── prepareBatteryDataV9 ─────────────────────────────────\n');
    fprintf('  File            : %s\n', filename);
    fprintf('  Filter          : %s\n', filter_method);
    fprintf('  SOC_0 branch    : %-9s  (I_probe = %+.3f A)\n', ctype, I_probe);
    fprintf('  OCV(t=0)        : %.4f V   R0(t=0) = %.4f Ω\n', OCV_0, R0_cont(1));
    fprintf('  Initial SOC_0   : %.2f%%   (shared by LABELS and RECONSTRUCTION)\n', SOC_0*100);
    fprintf('  Capacity range  : [%.0f, %.0f] A·s\n', min(C_dynamic), max(C_dynamic));
    fprintf('  SOC range (new) : [%.2f%%, %.2f%%]   clamp steps = %d\n', ...
        min(SOC_reg)*100, max(SOC_reg)*100, n_clamp);
    fprintf('  Rest anchoring  : NEW gated = %d steps | OLD every-rest = %d steps\n', ...
        n_anchor, n_anchor_old);
    fprintf('  Settled gate ON : %d / %d steps (%.1f%%)\n', ...
        sum(rest_gate), N, 100*sum(rest_gate)/N);
    fprintf('  >> LABEL SHIFT (new gated vs old every-rest): max |ΔSOC| = %.3f%%\n', label_shift);

    %% 7. Overlapping windows ──────────────────────────────────────────────
    stride     = floor(cfg.seqLen / 2);
    numBatches = floor((N - cfg.seqLen) / stride) + 1;

    V_batch  = zeros(1, numBatches, cfg.seqLen);
    I_batch  = zeros(1, numBatches, cfg.seqLen);
    T_batch  = zeros(1, numBatches, cfg.seqLen);
    C_batch  = zeros(1, numBatches, cfg.seqLen);
    S_batch  = zeros(1, numBatches, cfg.seqLen);
    O_batch  = zeros(1, numBatches, cfg.seqLen);   % SOC_ocv_instant feature

    for b = 1 : numBatches
        s = (b-1)*stride + 1;
        e = s + cfg.seqLen - 1;
        V_batch(1,b,:) = V_reg(s:e);
        I_batch(1,b,:) = I_reg(s:e);
        T_batch(1,b,:) = T_reg(s:e);
        C_batch(1,b,:) = C_dynamic(s:e);
        S_batch(1,b,:) = SOC_reg(s:e);
        O_batch(1,b,:) = SOC_ocv_inst(s:e);
    end

    %% 8. DA-2: SOC-stratified sample weights ─────────────────────────────
    num_bins = 10;
    window_mean_soc = squeeze(mean(S_batch, 3));   % [numBatches,1] or [1,numBatches]
    window_mean_soc = window_mean_soc(:)';
    bin_idx    = min(floor(window_mean_soc * num_bins) + 1, num_bins);
    bin_counts = histcounts(bin_idx, 1:num_bins+1);
    weights = zeros(1, numBatches);
    for b = 1 : numBatches
        weights(b) = 1.0 / (bin_counts(bin_idx(b)) + 1e-6);
    end
    weights = weights / mean(weights);
    W_batch = reshape(weights, [1, numBatches, 1]);

    fprintf('  SOC weights     : min=%.3f, max=%.3f, mean=%.3f\n', ...
        min(weights), max(weights), mean(weights));

    %% 9. Shared Z-score normalisation ────────────────────────────────────
    % Channel order: [V, I, T] (joint), C (alone), SOC_ocv_instant (alone).
    VIT_raw = cat(1, V_batch, I_batch, T_batch);

    if isempty(scalers_in)
        mu3    = mean(VIT_raw, [2, 3]);
        sig3   = std(VIT_raw,  0, [2, 3]) + 1e-8;
        mu_c   = mean(C_batch, 'all');
        sig_c  = std(C_batch,  0, 'all') + 1e-8;
        mu_si  = mean(O_batch, 'all');           % now normalises SOC_ocv_instant
        sig_si = std(O_batch,  0, 'all') + 1e-8;
        scalers = struct('mu3',mu3,'sig3',sig3,'mu_c',mu_c,'sig_c',sig_c,...
                         'mu_si',mu_si,'sig_si',sig_si);
        fprintf('  Scaler mode     : TRAINING (scalers computed)\n');
    else
        mu3    = scalers_in.mu3;   sig3   = scalers_in.sig3;
        mu_c   = scalers_in.mu_c;  sig_c  = scalers_in.sig_c;
        mu_si  = scalers_in.mu_si; sig_si = scalers_in.sig_si;
        scalers = scalers_in;
        fprintf('  Scaler mode     : VAL/TEST (training scalers applied)\n');
    end

    VIT_n  = (VIT_raw - mu3)  ./ sig3;
    C_n    = (C_batch - mu_c) /  sig_c;
    O_n    = (O_batch - mu_si)/  sig_si;
    normBatch = cat(1, VIT_n, C_n, O_n);

    fprintf('  Output          : %d batches × %d s  (stride %d s, 5 channels)\n', ...
        numBatches, cfg.seqLen, stride);
    fprintf('──────────────────────────────────────────────────────────\n\n');

    %% 10. Package dlarrays ───────────────────────────────────────────────
    dlX = dlarray(normBatch, 'CBT');
    dlY = dlarray(S_batch,   'CBT');
    dlV = dlarray(V_batch,   'CBT');
    dlI = dlarray(I_batch,   'CBT');
    dlT = dlarray(T_batch,   'CBT');
    dlC = dlarray(C_batch,   'CBT');
    dlW = dlarray(W_batch,   'CBT');

    %% 11. NEW-3: continuous signal struct for eval reconstruction ────────
    cont = struct();
    cont.V      = V_reg(:);
    cont.I      = I_reg(:);
    cont.T      = T_reg(:);
    cont.C      = C_dynamic(:);
    cont.SOC    = SOC_reg(:);
    cont.R0     = R0_cont(:);
    cont.N      = N;
    cont.stride = stride;
    cont.seqLen = cfg.seqLen;
    cont.numBatches = numBatches;
    cont.SOC_0  = SOC_0;
    % store the equilibrium OCV LUT so the reconstruction re-anchors with the
    % identical curve used to build the labels
    cont.eq_s      = eq_s;
    cont.soc_eq_s  = soc_eq_s;
    cont.gate      = rest_gate(:);     % P2: settled-rest anchor mask (shared w/ recon)
    cont.branch    = ctype;            % P0: SOC_0 OCV branch used
    cont.alpha_ocv = alpha_ocv;        % recon re-anchors with the identical pull gain
end

% ─────────────────────────────────────────────────────────────────────────
function soc = localInvOCV(ocv, I, chg_s, soc_chg_s, dchg_s, soc_dchg_s, eq_s, soc_eq_s)
% Vectorised sign-dependent OCV→SOC inversion.
%   I > +0.1  → use charge curve
%   I < -0.1  → use discharge curve
%   otherwise → use equilibrium curve
    ocv = ocv(:); I = I(:);
    soc = zeros(size(ocv));
    mc =  I >  0.1;
    md =  I < -0.1;
    mr = ~mc & ~md;
    if any(mc), soc(mc) = interp1(chg_s,  soc_chg_s,  ocv(mc), 'linear', 'extrap'); end
    if any(md), soc(md) = interp1(dchg_s, soc_dchg_s, ocv(md), 'linear', 'extrap'); end
    if any(mr), soc(mr) = interp1(eq_s,   soc_eq_s,   ocv(mr), 'linear', 'extrap'); end
end

% ─────────────────────────────────────────────────────────────────────────
function gate = settledRestGate(I_reg, V_reg, I_rest_thr, settle_len, relax_skip, dvdt_thr, dt)
% Logical mask (length N): TRUE only on the settled TAIL of a rest that is BOTH
% sustained (>= settle_len s) AND relaxed (trailing 30 s |dV/dt| < dvdt_thr V/s).
% These are the only rests whose terminal voltage is close enough to the
% equilibrium OCV to give a trustworthy SOC recalibration. Brief / still-relaxing
% rests (terminal V up to ~0.5 V from equilibrium here) are rejected.
    N    = numel(I_reg);
    gate = false(N, 1);
    rest = abs(I_reg) < I_rest_thr;
    d    = diff([0; double(rest(:)); 0]);
    s    = find(d ==  1);          % run start index (1-based)
    e    = find(d == -1);          % run is s(k) : e(k)-1  (inclusive end e(k)-1)
    for k = 1 : numel(s)
        a = s(k);
        b = e(k) - 1;              % inclusive end
        if (b - a + 1) >= settle_len
            tail0 = max(a, b - 29);                 % trailing 30 samples
            if b > tail0
                dvdt = abs(mean(diff(V_reg(tail0:b)))) / dt;
                if dvdt < dvdt_thr
                    g0 = min(a + relax_skip, b);    % skip initial relaxation
                    gate(g0:b) = true;
                end
            end
        end
    end
end