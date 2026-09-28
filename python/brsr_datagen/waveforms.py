"""Complex baseband radar waveforms (12 LPI classes) - Python port of the MATLAB generators.

Each function mirrors the MATLAB function of the same name in ``matlab/waveforms/`` and
returns a 1-D complex numpy array. Random start phases are drawn from ``rng`` unless given.
"""
import numpy as np

CLASS_NAMES = ["LFM", "Costas", "BPSK", "Frank", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]

BARKER_CODES = {
    7: [0, 0, 0, 1, 1, 0, 1],
    11: [0, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
    13: [0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0],
}


def matlab_colon(start, step, stop):
    """Values of MATLAB's ``start:step:stop`` (same element count rule, tolerance for rounding)."""
    n = int(np.floor((stop - start) / step + 1e-10)) + 1
    return start + step * np.arange(max(n, 0))


def resample_linear(x, n):
    """MATLAB ``interp1(linspace(1,L,L), x, linspace(1,L,n))`` on real and imaginary parts."""
    L = len(x)
    if L == n:
        return x
    t1 = np.linspace(1, L, L)
    t2 = np.linspace(1, L, n)
    return np.interp(t2, t1, x.real) + 1j * np.interp(t2, t1, x.imag)


def _random_phase(rng, phi0):
    return (2 * np.pi * rng.random() - np.pi) if phi0 is None else phi0


def phase_coded(cpp, fs, A, fc, phase_code):
    """Carrier chips of ``cpp`` cycles, one per phase in ``phase_code`` (MATLAB BPSK.m)."""
    cpp = np.atleast_1d(cpp)[0]                   # MATLAB uses the first element of a vector end point
    t = matlab_colon(0.0, 1 / fs, cpp / fc - 1 / fs)
    return np.concatenate([A * np.exp(1j * (2 * np.pi * fc * t + p)) for p in np.asarray(phase_code, float)])


def type_LFM(n_samples, fs, A, fc, Df, direction="Up", rng=None, phi0=None, legacy=True):
    """Linear frequency modulation.

    ``legacy=True`` reproduces the original MATLAB expression ``A*exp(1j*2*pi*f.*t + phi0)``, in which the
    random phase is added outside the imaginary unit, so it scales the amplitude by exp(phi0) (phi0 ~ U[-pi, pi]).
    This is how the published BRSR v1.0 data was made. ``legacy=False`` applies phi0 as a phase (unit amplitude).
    """
    rng = rng or np.random.default_rng()
    pw = n_samples / fs
    t = np.arange(1, n_samples + 1) / fs
    phi0 = _random_phase(rng, phi0)
    f = fc + Df / pw * t if direction.lower() != "down" else fc - Df / pw * t
    if legacy:
        return A * np.exp(1j * 2 * np.pi * f * t + phi0)
    return A * np.exp(1j * (2 * np.pi * f * t + phi0))


def type_Costas(n_samples, fs, A, fcmin, hops, rng=None, phi0=None):
    """Costas frequency hopping; ``hops`` is a permutation such as [3, 1, 2]."""
    rng = rng or np.random.default_rng()
    hops = np.asarray(hops)
    tsub = np.arange(1, int(np.ceil(n_samples / len(hops))) + 1) / fs
    phi0 = _random_phase(rng, phi0)
    return np.concatenate([A * np.exp(1j * (2 * np.pi * f * tsub + phi0)) for f in hops * fcmin])


def type_Barker(cpp, fs, A, fc, code_length):
    """Binary phase shift keying with a Barker code of length 7, 11 or 13."""
    return phase_coded(cpp, fs, A, fc, np.pi * np.array(BARKER_CODES[code_length]))


def type_Frank(cpp, fs, A, fc, M):
    i, j = np.meshgrid(np.arange(M), np.arange(M), indexing="ij")
    code = 2 * np.pi / M * i * j
    return phase_coded(cpp, fs, A, fc, code.flatten(order="F"))


def type_P1(cpp, fs, A, fc, M):
    i, j = np.meshgrid(np.arange(1, M + 1), np.arange(1, M + 1), indexing="ij")
    code = -np.pi / M * (M - (2 * j - 1)) * ((j - 1) * M + (i - 1))
    return phase_coded(cpp, fs, A, fc, code.flatten(order="F"))


def type_P2(cpp, fs, A, fc, M):
    i, j = np.meshgrid(np.arange(1, M + 1), np.arange(1, M + 1), indexing="ij")
    code = -np.pi / (2 * M) * (2 * i - 1 - M) * (2 * j - 1 - M)
    return phase_coded(cpp, fs, A, fc, code.flatten(order="F"))


def type_P3(cpp, fs, A, fc, p):
    k = np.arange(p)
    return phase_coded(cpp, fs, A, fc, np.pi / p * k ** 2)


def type_P4(cpp, fs, A, fc, p):
    k = np.arange(p)
    return phase_coded(cpp, fs, A, fc, np.pi / p * k ** 2 - np.pi * k)


def _t12_code(fs, fc, Nps, Ng, kind):
    # operation order follows the MATLAB code exactly (floor() is sensitive to rounding)
    Tc = 1 / fc
    t = matlab_colon(0.0, 1 / fs, Tc - 1 / fs)
    pw = Tc * Ng
    rows = []
    for jj in range(1, Ng):
        x = Ng * t - jj * pw
        if kind == 1:
            arg = x * jj * Nps / pw
        else:
            arg = x * (2 * jj - Ng + 1) / pw * Nps / 2
        rows.append(np.mod(2 * np.pi / Nps * np.floor(arg), 2 * np.pi))
    return np.concatenate(rows)                  # MATLAB: reshape(phaseCode', 1, []) = rows in order


def type_T1(fs, A, fc, Nps, Ng):
    return phase_coded(2, fs, A, fc, _t12_code(fs, fc, Nps, Ng, 1))


def type_T2(fs, A, fc, Nps, Ng):
    return phase_coded(2, fs, A, fc, _t12_code(fs, fc, Nps, Ng, 2))


def _t34(n_samples, fs, A, fc, Nps, B, kind):
    pw = n_samples / fs
    t = matlab_colon(0.0, 1 / fs, pw - 1 / fs)
    arg = Nps * B * t ** 2 / (2 * pw) - (Nps * B * t / 2 if kind == 4 else 0)
    phase = np.mod(2 * np.pi / Nps * np.floor(arg), 2 * np.pi)
    return A * np.exp(1j * (2 * np.pi * fc * t + phase))


def type_T3(n_samples, fs, A, fc, Nps, B):
    return _t34(n_samples, fs, A, fc, Nps, B, 3)


def type_T4(n_samples, fs, A, fc, Nps, B):
    return _t34(n_samples, fs, A, fc, Nps, B, 4)
