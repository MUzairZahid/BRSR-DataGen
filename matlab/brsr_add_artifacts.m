function [clean, noisy, comps, info] = brsr_add_artifacts(long_signal, snr_db, bank, cfg, composition)
% BRSR_ADD_ARTIFACTS  Cut a clean segment and corrupt it with a random blend of AWGN, echo and
%   co-channel interference (CCI) at the target SNR (BRSR-OpGAN paper, Section 3, Algorithm 2).
%
%   comps is 3 x n (rows: AWGN, echo, CCI; zeros when absent); noisy = clean + sum(comps).
%   Optional COMPOSITION (cell array, e.g. {'AWGN','CCI'}) fixes the artifact set.
n = cfg.n_samples;
combos = {{'AWGN'}, {'Echo'}, {'CCI'}, {'AWGN','Echo'}, {'AWGN','CCI'}, {'Echo','CCI'}, {'AWGN','Echo','CCI'}};
long_signal = reshape(long_signal, 1, []);
delay = randi(cfg.echo_delay);
start = randi([1, numel(long_signal) - (n + delay)]);
clean = long_signal(start:start+n-1);
delayed = long_signal(start+delay:start+delay+n-1);
row = randi(size(bank, 1));
mixing = bank(row, :);
if nargin < 5 || isempty(composition), composition = combos{randi(numel(combos))}; end
if numel(composition) > 1, w = rand(1, numel(composition)); w = w / sum(w); else, w = 1; end

p_noise = mean(abs(clean).^2) / 10^(snr_db/10);
comps = zeros(3, n);
w_all = zeros(1, 3);
names = {'AWGN', 'Echo', 'CCI'};
for i = 1:numel(composition)
    p = p_noise * w(i);
    switch composition{i}
        case 'AWGN'
            noise = randn(1, n) + 1i * randn(1, n);
            comps(1, :) = sqrt(p / mean(abs(noise).^2)) * noise;
        case 'Echo'
            comps(2, :) = sqrt(p / mean(abs(delayed).^2)) * delayed;
        case 'CCI'
            comps(3, :) = sqrt(p / mean(abs(mixing).^2)) * mixing;
    end
    w_all(strcmp(names, composition{i})) = w(i);
end
noisy = clean + sum(comps, 1);
info = struct('composition', strjoin(composition, '+'), 'w_awgn', w_all(1), 'w_echo', w_all(2), ...
    'w_cci', w_all(3), 'echo_delay', delay * any(strcmp(composition, 'Echo')) - ~any(strcmp(composition, 'Echo')), ...
    'cci_bank_row', (row - 1) * any(strcmp(composition, 'CCI')) - ~any(strcmp(composition, 'CCI')), ...
    'segment_start', start - 1);
end
