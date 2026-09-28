function wav = brsr_make_waveform(name, p, n_out, cfg)
% BRSR_MAKE_WAVEFORM  Complex waveform (row vector) of class NAME with parameters P,
%   linearly resampled to n_out samples (as in the original generator).
fs = cfg.fs; A = cfg.amplitude;
switch name
    case 'LFM',    wav = type_LFM(n_out, fs, A, p.fc, p.bandwidth, p.direction, [], cfg.legacy_lfm);
    case 'Costas', wav = type_Costas(n_out, fs, A, p.fcmin, p.hops);
    case 'BPSK'
        codes = {[0 0 0 1 1 0 1], [0 0 0 1 1 1 0 1 1 0 1], [0 0 0 0 0 1 1 0 0 1 0 1 0]};
        code = codes{find([7 11 13] == p.barker_length)} * pi;
        wav = type_Barker(p.cycles_per_chip, fs, A, p.fc, code);
    case 'Frank', wav = type_Frank(p.cycles_per_chip, fs, A, p.fc, p.steps);
    case 'P1',    wav = type_P1(p.cycles_per_chip, fs, A, p.fc, p.steps);
    case 'P2',    wav = type_P2(p.cycles_per_chip, fs, A, p.fc, p.steps);
    case 'P3',    wav = type_P3(p.cycles_per_chip, fs, A, p.fc, p.subcodes);
    case 'P4',    wav = type_P4(p.cycles_per_chip, fs, A, p.fc, p.subcodes);
    case 'T1',    wav = type_T1(fs, A, p.fc, p.phase_states, p.segments);
    case 'T2',    wav = type_T2(fs, A, p.fc, p.phase_states, p.segments);
    case 'T3',    wav = type_T3(n_out, fs, A, p.fc, p.phase_states, p.bandwidth);
    case 'T4',    wav = type_T4(n_out, fs, A, p.fc, p.phase_states, p.bandwidth);
end
wav = reshape(wav, 1, []);
if numel(wav) ~= n_out
    L = numel(wav);
    t1 = linspace(1, L, L); t2 = linspace(1, L, n_out);
    wav = interp1(t1, real(wav), t2) + 1i * interp1(t1, imag(wav), t2);
end
end
