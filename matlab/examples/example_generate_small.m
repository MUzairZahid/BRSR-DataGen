% Generate a small BRSR-style dataset (3 signals per class per SNR level) into ./data_small
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'));
cfg = brsr_config('mode', 'brsr', 'seed', 0, 'n_train', 3, 'n_test', 2);
files = generate_brsr_dataset(cfg, fullfile(here, 'data_small'));
disp(files);

% Full-size BRSR-style dataset (62,400 train-run + 23,400 test signals, several GB):
%   generate_brsr_dataset(brsr_config('mode', 'brsr', 'seed', 0), 'data');
% AWGN-only dataset at 13 SNR levels:
%   generate_brsr_dataset(brsr_config('mode', 'awgn', 'seed', 0), 'data');
