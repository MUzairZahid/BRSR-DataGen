% Build one corrupted radar signal step by step and plot it (time signal and spectrogram).
here = fileparts(mfilename('fullpath'));
addpath(fullfile(here, '..'), fullfile(here, '..', 'waveforms'));
rng(3);
cfg = brsr_config();
S = load(cfg.interference_bank, 'mixing_signals');
name = 'LFM';                                   % any of cfg.classes
P = brsr_sample_parameters(name, 1, cfg);
wav = brsr_make_waveform(name, P, cfg.n_samples * cfg.long_factor, cfg);
[clean, noisy, comps, info] = brsr_add_artifacts(wav, -2, S.mixing_signals, cfg, {'AWGN', 'Echo', 'CCI'});

panels = {clean, 'Clean'; comps(2,:), 'Echo component'; comps(3,:), 'Interference (CCI) component'; ...
          comps(1,:), 'AWGN component'; noisy, sprintf('Received signal (target SNR -2 dB, %s)', info.composition)};
figure('Name', 'BRSR signal model', 'Color', 'w');
for k = 1:size(panels, 1)
    subplot(size(panels, 1), 2, 2*k - 1);
    plot((0:255) / cfg.fs * 1e6, real(panels{k, 1}(1:256)), 'LineWidth', 1);
    axis tight; title(panels{k, 2}); xlabel('time (\mus)'); ylabel('I');
    subplot(size(panels, 1), 2, 2*k);
    [Sdb, t, f] = brsr_spectrogram(panels{k, 1}, cfg.fs);
    imagesc(t, f, Sdb); axis xy; caxis(max(Sdb(:)) + [-45 0]);
    title([panels{k, 2} ' - spectrogram']); xlabel('time (\mus)'); ylabel('MHz');
end
colormap(flipud(bone));
