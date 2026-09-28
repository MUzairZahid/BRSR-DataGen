function P = brsr_sample_parameters(name, n, cfg)
% BRSR_SAMPLE_PARAMETERS  Parameters for n waveforms of class NAME (ranges of the BRSR v1.0 generator).
fs = cfg.fs;
P = repmat(struct(), 1, n);
switch name
    case 'LFM'
        fc = vals(fs/6, fs/5, n, cfg); B = vals(fs/20, fs/16, n, cfg); d = {'Down', 'Up'};
        for i = 1:n, P(i).fc = fc(i); P(i).bandwidth = B(i); P(i).direction = d{randi(2)}; end
    case 'Costas'
        fcmin = vals(fs/30, fs/24, n, cfg); L = [3 4 5];
        for i = 1:n, P(i).fcmin = fcmin(i); P(i).hops = randperm(L(randi(3))); end
    case 'BPSK'
        fc = vals(fs/13, fs/10, n, cfg); L = [7 11 13];
        for i = 1:n, P(i).fc = fc(i); P(i).barker_length = L(randi(3)); P(i).cycles_per_chip = 20; end
    case {'Frank', 'P1', 'P2', 'P3', 'P4'}
        fc = vals(fs/6, fs/5, n, cfg); cyc = [3 4 5];
        switch name
            case {'Frank', 'P1'}, key = 'steps'; opts = [6 7 8];
            case 'P2', key = 'steps'; opts = [6 8];
            otherwise, key = 'subcodes'; opts = [36 49 64];
        end
        for i = 1:n
            P(i).fc = fc(i); P(i).cycles_per_chip = cyc(randi(3)); P(i).(key) = opts(randi(numel(opts)));
        end
    case {'T1', 'T2'}
        fc = vals(fs/6, fs/5, n, cfg); G = [4 5 6];
        for i = 1:n, P(i).fc = fc(i); P(i).segments = G(randi(3)); P(i).phase_states = 2; end
    case {'T3', 'T4'}
        fc = vals(fs/13, fs/10, n, cfg); B = vals(fs/20, fs/10, n, cfg);
        for i = 1:n, P(i).fc = fc(i); P(i).bandwidth = B(i); P(i).phase_states = 2; end
    otherwise
        error('unknown class %s', name);
end
end

function v = vals(lo, hi, n, cfg)
if strcmp(cfg.param_sampling, 'grid')
    v = linspace(lo, hi, n); v = v(randperm(n));
else
    v = lo + (hi - lo) * rand(1, n);
end
end
