function files = generate_brsr_dataset(cfg, out_dir, prefix)
% GENERATE_BRSR_DATASET  Generate a BRSR-style radar signal restoration dataset.
%
%   cfg = brsr_config('mode', 'brsr', 'seed', 0);
%   generate_brsr_dataset(cfg, 'data');
%
%   Writes <prefix>_train.h5, <prefix>_validation.h5, <prefix>_test.h5 and <prefix>_metadata.csv
%   (prefix: 'brsr' or 'awgn_baseline'). The HDF5 layout matches the published BRSR benchmark
%   (Zenodo, DOI 10.5281/zenodo.23010395): read in Python, 'clean' and 'noisy' are [N, 2, 1024]
%   (channel 1 = I, channel 2 = Q), 'distortions' is [N, 3, 2, 1024] (AWGN, echo, CCI), plus
%   'label' (1..12), 'snr_db' and 'gen_index'. In MATLAB, h5read returns the reversed dimensions.
if nargin < 2 || isempty(out_dir), out_dir = 'data'; end
if nargin < 3 || isempty(prefix)
    if strcmp(cfg.mode, 'brsr'), prefix = 'brsr'; else, prefix = 'awgn_baseline'; end
end
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, 'waveforms'));
if ~exist(out_dir, 'dir'), mkdir(out_dir); end
rng(cfg.seed);
blind = strcmp(cfg.mode, 'brsr');
if blind, S = load(cfg.interference_bank, 'mixing_signals'); bank = S.mixing_signals; else, bank = []; end
n = cfg.n_samples;
if blind, n_long = n * cfg.long_factor; else, n_long = n; end
nc = numel(cfg.classes); nl = numel(cfg.snr_levels);

meta_fid = fopen(fullfile(out_dir, [prefix '_metadata.csv']), 'w');
fprintf(meta_fid, ['split,row,gen_index,label,class_name,snr_target_db,snr_measured_db,composition,' ...
    'w_awgn,w_echo,w_cci,echo_delay,cci_bank_row,segment_start,params\n']);
runs = {'train', cfg.n_train; 'test', cfg.n_test};
files = struct();
for r = 1:2
    run = runs{r, 1}; per = runs{r, 2}; total = nl * nc * per;
    is_val = false(1, total);
    if strcmp(run, 'train')
        lab = mod(floor((0:total-1) / per), nc) + 1;
        for k = 1:nc
            idx = find(lab == k);
            pick = randperm(numel(idx), round(cfg.val_fraction * numel(idx)));
            is_val(idx(pick)) = true;
        end
        splits = {'train', 'validation'}; counts = [sum(~is_val), sum(is_val)];
    else
        splits = {'test'}; counts = total;
    end
    W = struct();
    for s = 1:numel(splits)
        W.(splits{s}) = open_writer(fullfile(out_dir, [prefix '_' splits{s} '.h5']), counts(s), n, blind, cfg, splits{s});
        files.(splits{s}) = W.(splits{s}).file;
    end
    gen = 0;
    for lev = cfg.snr_levels
        for k = 1:nc
            name = cfg.classes{k};
            P = brsr_sample_parameters(name, per, cfg);
            for i = 1:per
                wav = brsr_make_waveform(name, P(i), n_long, cfg);
                if blind
                    snr = cfg.snr_min + (cfg.snr_max - cfg.snr_min) * rand;
                    [clean, noisy, comps, info] = brsr_add_artifacts(wav, snr, bank, cfg);
                else
                    snr = lev; clean = wav; noisy = brsr_awgn_measured(clean, snr); comps = [];
                    info = struct('composition', 'AWGN', 'w_awgn', 1, 'w_echo', 0, 'w_cci', 0, ...
                        'echo_delay', -1, 'cci_bank_row', -1, 'segment_start', 0);
                end
                if is_val(gen + 1), sp = 'validation'; elseif strcmp(run, 'train'), sp = 'train'; else, sp = 'test'; end
                row = W.(sp).count;
                W.(sp) = add_row(W.(sp), clean, noisy, comps, k, snr, gen);
                measured = 10 * log10(sum(abs(clean).^2) / sum(abs(noisy - clean).^2));
                fprintf(meta_fid, '%s,%d,%d,%d,%s,%.6f,%.6f,%s,%.6f,%.6f,%.6f,%d,%d,%d,"%s"\n', sp, row, gen, k, ...
                    name, snr, measured, info.composition, info.w_awgn, info.w_echo, info.w_cci, info.echo_delay, ...
                    info.cci_bank_row, info.segment_start, strrep(jsonencode(P(i)), '"', '""'));
                gen = gen + 1;
            end
        end
        fprintf('  %s: SNR loop %+d dB done (%d/%d)\n', run, lev, gen, total);
    end
    for s = 1:numel(splits), close_writer(W.(splits{s})); end
end
fclose(meta_fid);
end

% ------------------------------------------------------------------ HDF5 writer (streams blocks of 256)
function w = open_writer(file, N, n, blind, cfg, split)
if exist(file, 'file'), delete(file); end
c = min(64, N);
h5create(file, '/clean', [n 2 N], 'Datatype', 'single', 'ChunkSize', [n 2 c], 'Deflate', 4, 'Shuffle', true);
h5create(file, '/noisy', [n 2 N], 'Datatype', 'single', 'ChunkSize', [n 2 c], 'Deflate', 4, 'Shuffle', true);
if blind
    h5create(file, '/distortions', [n 2 3 N], 'Datatype', 'single', 'ChunkSize', [n 2 3 min(32, N)], ...
        'Deflate', 4, 'Shuffle', true);
end
h5create(file, '/label', N, 'Datatype', 'uint8');
h5create(file, '/snr_db', N, 'Datatype', 'double');
h5create(file, '/gen_index', N, 'Datatype', 'int32');
h5writeatt(file, '/', 'split', split);
h5writeatt(file, '/', 'generator', 'BRSR-DataGen (MATLAB)');
h5writeatt(file, '/', 'class_names', strjoin(cfg.classes, ','));
h5writeatt(file, '/', 'config', jsonencode(cfg));
w = struct('file', file, 'N', N, 'n', n, 'blind', blind, 'count', 0, 'written', 0, ...
    'clean', zeros(n, 2, 256, 'single'), 'noisy', zeros(n, 2, 256, 'single'), ...
    'dist', zeros(n, 2, 3, 256, 'single'), 'label', zeros(256, 1), 'snr', zeros(256, 1), 'gen', zeros(256, 1), 'b', 0);
end

function w = add_row(w, clean, noisy, comps, label, snr, gen)
w.b = w.b + 1; b = w.b;
w.clean(:, :, b) = single([real(clean(:)), imag(clean(:))]);
w.noisy(:, :, b) = single([real(noisy(:)), imag(noisy(:))]);
if w.blind
    for a = 1:3, w.dist(:, :, a, b) = single([real(comps(a, :)).', imag(comps(a, :)).']); end
end
w.label(b) = label; w.snr(b) = snr; w.gen(b) = gen;
w.count = w.count + 1;
if b == 256, w = flush(w); end
end

function w = flush(w)
b = w.b;
if b == 0, return; end
a = w.written + 1;
h5write(w.file, '/clean', w.clean(:, :, 1:b), [1 1 a], [w.n 2 b]);
h5write(w.file, '/noisy', w.noisy(:, :, 1:b), [1 1 a], [w.n 2 b]);
if w.blind, h5write(w.file, '/distortions', w.dist(:, :, :, 1:b), [1 1 1 a], [w.n 2 3 b]); end
h5write(w.file, '/label', uint8(w.label(1:b)), a, b);
h5write(w.file, '/snr_db', w.snr(1:b), a, b);
h5write(w.file, '/gen_index', int32(w.gen(1:b)), a, b);
w.written = w.written + b; w.b = 0;
end

function close_writer(w)
w = flush(w);
assert(w.written == w.N, 'writer %s: wrote %d of %d rows', w.file, w.written, w.N);
end
