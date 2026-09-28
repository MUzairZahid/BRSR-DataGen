"""Command line: python -m brsr_datagen --mode brsr --out data/ --seed 0"""
import argparse
import time

from .generator import Config, generate


def main():
    ap = argparse.ArgumentParser(prog="python -m brsr_datagen",
                                 description="Generate a BRSR-style radar signal restoration dataset (HDF5).")
    ap.add_argument("--mode", choices=["brsr", "awgn"], default="brsr",
                    help="brsr: random blend of AWGN, echo and interference; awgn: AWGN only at discrete SNRs")
    ap.add_argument("--out", default="data", help="output folder")
    ap.add_argument("--prefix", default=None, help="file name prefix (default: brsr / awgn_baseline)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=400, help="signals per class per SNR level, train run")
    ap.add_argument("--n_test", type=int, default=150, help="signals per class per SNR level, test run")
    ap.add_argument("--snr_min", type=float, default=-14.0)
    ap.add_argument("--snr_max", type=float, default=10.0)
    ap.add_argument("--param_sampling", choices=["grid", "uniform"], default="grid")
    ap.add_argument("--fixed_lfm", action="store_true", help="unit-amplitude LFM (default reproduces BRSR v1.0)")
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    a = ap.parse_args()
    cfg = Config(mode=a.mode, seed=a.seed, n_train=a.n_train, n_test=a.n_test, snr_min=a.snr_min,
                 snr_max=a.snr_max, param_sampling=a.param_sampling, legacy_lfm=not a.fixed_lfm, dtype=a.dtype)
    t0 = time.time()
    paths = generate(cfg, a.out, a.prefix)
    print(f"Done in {time.time() - t0:.0f}s:", *paths.values(), sep="\n  ")


if __name__ == "__main__":
    main()
