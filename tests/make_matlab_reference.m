% Build tests/matlab_reference.mat: waveforms from the MATLAB code for fixed parameters.
% Run from the repository root:  octave --eval "run('tests/make_matlab_reference.m')"  (or in MATLAB)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root, 'matlab'), fullfile(root, 'matlab', 'waveforms'));
cfg = brsr_config();
fs = cfg.fs; A = 1; N = 2048;
R = struct();
R.LFM_up        = type_LFM(N, fs, A, 18.1e6, 5.5e6, 'Up', 0.3, true);
R.LFM_down      = type_LFM(N, fs, A, 17.2e6, 6.1e6, 'Down', -1.2, true);
R.LFM_fixed     = type_LFM(N, fs, A, 18.1e6, 5.5e6, 'Up', 0.3, false);
R.Costas        = type_Costas(N, fs, A, 3.7e6, [3 1 4 2], 0.7);
R.BPSK13        = type_Barker(20, fs, A, 8.4e6, [0 0 0 0 0 1 1 0 0 1 0 1 0] * pi);
R.BPSK_vec_cpp  = type_Barker(20:24, fs, A, 9.1e6, [0 0 0 1 1 0 1] * pi);
R.Frank         = type_Frank(4, fs, A, 17.9e6, 7);
R.P1            = type_P1(3, fs, A, 18.8e6, 8);
R.P2            = type_P2(5, fs, A, 16.9e6, 6);
R.P3            = type_P3(4, fs, A, 19.3e6, 49);
R.P4            = type_P4(3, fs, A, 17.4e6, 36);
R.T1            = type_T1(fs, A, 18.6e6, 2, 5);
R.T2            = type_T2(fs, A, 19.7e6, 2, 6);
R.T3            = type_T3(N, fs, A, 8.9e6, 2, 7.3e6);
R.T4            = type_T4(N, fs, A, 9.6e6, 2, 5.1e6);
% resampled to N as in brsr_make_waveform
f = fieldnames(R);
for i = 1:numel(f)
    w = reshape(R.(f{i}), 1, []);
    L = numel(w); t1 = linspace(1, L, L); t2 = linspace(1, L, N);
    R.([f{i} '_resampled']) = interp1(t1, real(w), t2) + 1i * interp1(t1, imag(w), t2);
end
save('-v7', fullfile(root, 'tests', 'matlab_reference.mat'), '-struct', 'R');
disp('wrote tests/matlab_reference.mat');
