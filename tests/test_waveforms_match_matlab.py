"""The Python waveforms must equal the MATLAB ones (reference made with tests/make_matlab_reference.m)."""
import os

import numpy as np
import pytest
import scipy.io as sio

from brsr_datagen import waveforms as W

REF = sio.loadmat(os.path.join(os.path.dirname(__file__), "matlab_reference.mat"))
FS, A, N = 100e6, 1.0, 2048
CASES = {
    "LFM_up": lambda: W.type_LFM(N, FS, A, 18.1e6, 5.5e6, "Up", phi0=0.3, legacy=True),
    "LFM_down": lambda: W.type_LFM(N, FS, A, 17.2e6, 6.1e6, "Down", phi0=-1.2, legacy=True),
    "LFM_fixed": lambda: W.type_LFM(N, FS, A, 18.1e6, 5.5e6, "Up", phi0=0.3, legacy=False),
    "Costas": lambda: W.type_Costas(N, FS, A, 3.7e6, [3, 1, 4, 2], phi0=0.7),
    "BPSK13": lambda: W.type_Barker(20, FS, A, 8.4e6, 13),
    "BPSK_vec_cpp": lambda: W.type_Barker(np.arange(20, 25), FS, A, 9.1e6, 7),
    "Frank": lambda: W.type_Frank(4, FS, A, 17.9e6, 7),
    "P1": lambda: W.type_P1(3, FS, A, 18.8e6, 8),
    "P2": lambda: W.type_P2(5, FS, A, 16.9e6, 6),
    "P3": lambda: W.type_P3(4, FS, A, 19.3e6, 49),
    "P4": lambda: W.type_P4(3, FS, A, 17.4e6, 36),
    "T1": lambda: W.type_T1(FS, A, 18.6e6, 2, 5),
    "T2": lambda: W.type_T2(FS, A, 19.7e6, 2, 6),
    "T3": lambda: W.type_T3(N, FS, A, 8.9e6, 2, 7.3e6),
    "T4": lambda: W.type_T4(N, FS, A, 9.6e6, 2, 5.1e6),
}


@pytest.mark.parametrize("name", list(CASES))
def test_waveform_matches_matlab(name):
    py = CASES[name]()
    ref = REF[name].ravel()
    assert py.shape == ref.shape, f"{name}: length {py.shape} vs MATLAB {ref.shape}"
    np.testing.assert_allclose(py, ref, rtol=1e-9, atol=1e-9 * max(1.0, np.abs(ref).max()))
    np.testing.assert_allclose(W.resample_linear(py, N), REF[name + "_resampled"].ravel(), rtol=1e-9,
                               atol=1e-9 * max(1.0, np.abs(ref).max()))
