function [S_db, t_us, f_mhz] = brsr_spectrogram(x, fs, nfft, hop)
% BRSR_SPECTROGRAM  Power spectrogram (dB) of a complex signal, no toolbox needed.
if nargin < 3, nfft = 64; end
if nargin < 4, hop = 8; end
x = reshape(x, 1, []);
w = 0.5 - 0.5 * cos(2*pi*(0:nfft-1)/(nfft-1));          % Hann window
starts = 1:hop:numel(x)-nfft+1;
S = zeros(nfft, numel(starts));
for k = 1:numel(starts)
    S(:, k) = abs(fftshift(fft(x(starts(k):starts(k)+nfft-1) .* w))).^2;
end
S_db = 10*log10(S + 1e-12);
t_us = (starts - 1 + nfft/2) / fs * 1e6;
f_mhz = ((-nfft/2):(nfft/2-1)) * fs / nfft / 1e6;
end
