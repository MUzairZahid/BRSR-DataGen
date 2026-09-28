function [noisy, noise] = brsr_awgn_measured(x, snr_db)
% BRSR_AWGN_MEASURED  Complex AWGN at SNR relative to the measured signal power
%   (same as awgn(x, snr_db, 'measured') of the Communications Toolbox, without needing it).
p = mean(abs(x).^2) / 10^(snr_db/10);
noise = sqrt(p/2) * (randn(size(x)) + 1i * randn(size(x)));
noisy = x + noise;
end
