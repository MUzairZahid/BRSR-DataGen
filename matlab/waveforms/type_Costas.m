function s = type_Costas(NumberSamples, fs, A, fcmin, NumHop, phi0)
% TYPE_COSTAS  Costas frequency-hopping waveform; NumHop is a permutation, e.g. [3 1 2].
%   Optional phi0 (random in [-pi, pi) if omitted).
if nargin < 6 || isempty(phi0), phi0 = 2*pi*rand(1) - pi; end
tsub = (1:ceil(NumberSamples/length(NumHop)))/fs;
f = NumHop*fcmin;
s1 = zeros(length(tsub), length(f));
for k = 1:length(f)
    s1(:,k) = A*exp(1j*(2*pi*f(k)*tsub + phi0));
end
s = reshape(s1, [1, numel(s1)]);
end
