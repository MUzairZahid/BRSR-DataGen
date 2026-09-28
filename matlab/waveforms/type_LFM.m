function s = type_LFM(NumberSamples, fs, A, fc, Df, updown, phi0, legacy)
% TYPE_LFM  Linear frequency modulated pulse.
%   s = type_LFM(N, fs, A, fc, Df, updown) with updown = 'Up' or 'Down'.
%   Optional phi0 (random in [-pi, pi) if omitted) and legacy (default true).
%   legacy = true reproduces the expression used for BRSR v1.0, A*exp(1j*2*pi*f.*t + phi0),
%   where phi0 scales the amplitude by exp(phi0); legacy = false applies phi0 as a phase.
if nargin < 7 || isempty(phi0), phi0 = 2*pi*rand(1) - pi; end
if nargin < 8, legacy = true; end
pw = NumberSamples/fs;
t = (1:NumberSamples)/fs;
if any(strcmpi(updown, {'down'}))
    f = fc - Df/pw*t;
else
    f = fc + Df/pw*t;
end
if legacy
    s = A*exp(1j*2*pi*f.*t + phi0);
else
    s = A*exp(1j*(2*pi*f.*t + phi0));
end
end
