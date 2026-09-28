"""End-to-end checks of the Python generator on a small dataset."""
import csv

import h5py
import numpy as np
import pytest

from brsr_datagen import Config, add_artifacts, generate, load_interference_bank
from brsr_datagen.waveforms import CLASS_NAMES


@pytest.fixture(scope="module")
def small_brsr(tmp_path_factory):
    out = tmp_path_factory.mktemp("brsr")
    generate(Config(mode="brsr", seed=11, n_train=5, n_test=2), str(out), progress=False)
    return out


def test_split_sizes_and_layout(small_brsr):
    sizes = {}
    for split in ("train", "validation", "test"):
        with h5py.File(small_brsr / f"brsr_{split}.h5") as f:
            n = f["label"].shape[0]
            sizes[split] = n
            assert f["clean"].shape == (n, 2, 1024) and f["noisy"].shape == (n, 2, 1024)
            assert f["distortions"].shape == (n, 3, 2, 1024)
            assert set(np.unique(f["label"][:])) == set(range(1, 13))
            assert np.all(np.bincount(f["label"][:])[1:] == n // 12)          # stratified: equal per class
    assert sizes == {"train": 624, "validation": 156, "test": 312}


def test_noisy_is_clean_plus_components(small_brsr):
    with h5py.File(small_brsr / "brsr_train.h5") as f:
        c, n, d = f["clean"][:], f["noisy"][:], f["distortions"][:]
    assert np.abs(n - c - d.sum(axis=1)).max() < 1e-4


def test_metadata(small_brsr):
    rows = list(csv.DictReader(open(small_brsr / "brsr_metadata.csv")))
    assert len(rows) == 624 + 156 + 312
    assert {r["composition"] for r in rows} == {"AWGN", "Echo", "CCI", "AWGN+Echo", "AWGN+CCI", "Echo+CCI",
                                                 "AWGN+Echo+CCI"}
    for r in rows:
        w = float(r["w_awgn"]) + float(r["w_echo"]) + float(r["w_cci"])
        assert abs(w - 1) < 1e-5
        assert -14 <= float(r["snr_target_db"]) <= 10
        if "+" not in r["composition"]:                                     # one artifact: SNR is exact
            assert abs(float(r["snr_target_db"]) - float(r["snr_measured_db"])) < 1e-4


def test_reproducible_with_seed(tmp_path):
    a, b, c = tmp_path / "a", tmp_path / "b", tmp_path / "c"
    for p, seed in ((a, 5), (b, 5), (c, 6)):
        generate(Config(mode="awgn", seed=seed, n_train=2, n_test=1), str(p), progress=False)
    fa, fb, fc = (h5py.File(p / "awgn_baseline_test.h5") for p in (a, b, c))
    assert np.array_equal(fa["noisy"][:], fb["noisy"][:])
    assert not np.array_equal(fa["noisy"][:], fc["noisy"][:])
    assert sorted(set(fa["snr_db"][:])) == list(range(-14, 12, 2))


def test_artifact_powers():
    rng = np.random.default_rng(0)
    bank = load_interference_bank()
    x = np.exp(1j * 2 * np.pi * 0.17 * np.arange(2048))
    for combo in (("AWGN",), ("Echo",), ("CCI",), ("AWGN", "Echo", "CCI")):
        clean, noisy, comps, info = add_artifacts(x, -3.0, bank, rng, composition=combo)
        p_noise = np.mean(np.abs(clean) ** 2) / 10 ** (-3 / 10)
        powers = np.mean(np.abs(comps) ** 2, axis=1)
        weights = np.array([info["w_awgn"], info["w_echo"], info["w_cci"]])
        np.testing.assert_allclose(powers, weights * p_noise, rtol=1e-9, atol=1e-12)
        assert info["composition"] == "+".join(combo)


def test_class_names():
    assert CLASS_NAMES == ["LFM", "Costas", "BPSK", "Frank", "P1", "P2", "P3", "P4", "T1", "T2", "T3", "T4"]
