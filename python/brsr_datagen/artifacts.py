"""Artifact model of the BRSR benchmark: AWGN, echo and co-channel interference (CCI).

Python port of ``matlab/brsr_add_artifacts.m`` (``add_distortion.m`` in the original code).
BRSR-OpGAN paper, Section 3 / Algorithm 2.
"""
import os

import numpy as np

ARTIFACTS = ("AWGN", "Echo", "CCI")
COMPOSITIONS = [("AWGN",), ("Echo",), ("CCI",), ("AWGN", "Echo"), ("AWGN", "CCI"), ("Echo", "CCI"),
                ("AWGN", "Echo", "CCI")]
_BANK = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "interference_bank.npy")


def load_interference_bank(path=None):
    """The 50 complex signals [50, 1024] used as co-channel interference in BRSR v1.0."""
    return np.load(path or _BANK)


def add_artifacts(long_signal, snr_db, bank, rng, n_samples=1024, delay_range=(128, 512),
                  compositions=COMPOSITIONS, composition=None):
    """Cut a clean segment from ``long_signal`` and corrupt it with a random blend of artifacts.

    1. Draw an echo delay d in ``delay_range`` and a start index; the clean segment is
       long_signal[s : s+n] and the echo source is long_signal[s+d : s+d+n].
    2. Draw one interference signal from ``bank`` and one artifact ``composition`` (uniform over the 7).
    3. Draw blend weights (uniform, normalized to sum 1) for the selected artifacts.
    4. Scale each artifact to power w_i * P, with P = P_clean / 10^(snr_db/10).

    Returns clean [n], noisy [n], components [3, n] (AWGN, echo, CCI; zeros when absent) and an info dict.
    """
    L = len(long_signal)
    delay = int(rng.integers(delay_range[0], delay_range[1] + 1))
    start = int(rng.integers(0, L - (n_samples + delay)))            # MATLAB randi([1, L-(n+d)]) - 1
    clean = long_signal[start:start + n_samples]
    delayed = long_signal[start + delay:start + delay + n_samples]
    cci_row = int(rng.integers(len(bank)))
    mixing = bank[cci_row]
    combo = composition or compositions[int(rng.integers(len(compositions)))]
    weights = rng.random(len(combo)) if len(combo) > 1 else np.ones(1)
    weights = weights / weights.sum()

    p_noise = np.mean(np.abs(clean) ** 2) / 10 ** (snr_db / 10)
    comps = np.zeros((3, n_samples), complex)
    w_all = np.zeros(3)
    for w, name in zip(weights, combo):
        p = p_noise * w
        if name == "AWGN":
            noise = rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples)
            comps[0] = np.sqrt(p / np.mean(np.abs(noise) ** 2)) * noise
        elif name == "Echo":
            comps[1] = np.sqrt(p / np.mean(np.abs(delayed) ** 2)) * delayed
        else:
            comps[2] = np.sqrt(p / np.mean(np.abs(mixing) ** 2)) * mixing
        w_all[ARTIFACTS.index(name)] = w
    noisy = clean + comps.sum(axis=0)
    info = dict(composition="+".join(combo), w_awgn=w_all[0], w_echo=w_all[1], w_cci=w_all[2],
                echo_delay=delay if "Echo" in combo else -1, cci_bank_row=cci_row if "CCI" in combo else -1,
                segment_start=start)
    return clean, noisy, comps, info


def add_awgn_measured(x, snr_db, rng):
    """AWGN at ``snr_db`` relative to the measured signal power (MATLAB ``awgn(x, snr, 'measured')``)."""
    p = np.mean(np.abs(x) ** 2) / 10 ** (snr_db / 10)
    noise = np.sqrt(p / 2) * (rng.standard_normal(len(x)) + 1j * rng.standard_normal(len(x)))
    return x + noise, noise
