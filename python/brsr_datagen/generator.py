"""BRSR-style dataset generation (Python port of matlab/generate_brsr_dataset.m).

Two modes, as in the BRSR benchmark:
  * ``brsr``: blind setting, each signal gets a random blend of AWGN, echo and CCI at an SNR drawn
    uniformly from [snr_min, snr_max];
  * ``awgn``: AWGN only, at the discrete SNR levels ``snr_levels`` (AWGN-Baseline).

Output files use the same HDF5 layout as the published benchmark (Zenodo, DOI 10.5281/zenodo.23010395),
so they load directly with the BRSR-OpGAN code.
"""
import csv
import json
import os
from dataclasses import asdict, dataclass, field

import h5py
import numpy as np

from . import waveforms as W
from .artifacts import COMPOSITIONS, add_artifacts, add_awgn_measured, load_interference_bank

__version__ = "1.0.0"


@dataclass
class Config:
    mode: str = "brsr"                         # "brsr" (blind: AWGN + echo + CCI) or "awgn" (AWGN only)
    seed: int = 0
    fs: float = 100e6                          # sampling frequency (Hz)
    amplitude: float = 1.0
    n_samples: int = 1024                      # length of each stored signal
    long_factor: int = 2                       # brsr: waveforms are generated 2x longer, then a segment is cut
    snr_min: float = -14.0
    snr_max: float = 10.0
    snr_levels: tuple = tuple(range(-14, 12, 2))   # generation loop (13 levels); awgn mode uses them as SNRs
    n_train: int = 400                         # signals per class per SNR level in the train run
    n_test: int = 150                          # ... in the test run
    val_fraction: float = 0.2                  # stratified split of the train run into train / validation
    classes: tuple = tuple(W.CLASS_NAMES)
    param_sampling: str = "grid"               # "grid" (as in BRSR v1.0) or "uniform"
    legacy_lfm: bool = True                    # reproduce the original LFM amplitude behaviour (see waveforms.py)
    echo_delay: tuple = (128, 512)
    compositions: list = field(default_factory=lambda: ["+".join(c) for c in COMPOSITIONS])
    dtype: str = "float32"

    def to_json(self):
        return json.dumps(asdict(self))


# ---------------------------------------------------------------- per-class parameter sampling
def _values(lo, hi, n, rng, sampling):
    """Grid: linspace(lo, hi, n) in random order (original code); uniform: independent draws."""
    if sampling == "grid":
        return rng.permutation(np.linspace(lo, hi, n))
    return rng.uniform(lo, hi, n)


def sample_parameters(name, n, rng, cfg):
    """Parameters for ``n`` waveforms of class ``name`` (ranges of the original BRSR generator)."""
    fs, s = cfg.fs, cfg.param_sampling
    pick = lambda options: [options[int(rng.integers(len(options)))] for _ in range(n)]
    if name == "LFM":
        fc, B = _values(fs / 6, fs / 5, n, rng, s), _values(fs / 20, fs / 16, n, rng, s)
        d = pick(["Down", "Up"])
        return [dict(fc=fc[i], bandwidth=B[i], direction=d[i]) for i in range(n)]
    if name == "Costas":
        fcmin = _values(fs / 30, fs / 24, n, rng, s)
        return [dict(fcmin=fcmin[i], hops=(rng.permutation(int(L)) + 1).tolist())
                for i, L in enumerate(pick([3, 4, 5]))]
    if name == "BPSK":
        fc = _values(fs / 13, fs / 10, n, rng, s)
        return [dict(fc=fc[i], barker_length=int(b), cycles_per_chip=20) for i, b in enumerate(pick([7, 11, 13]))]
    if name in ("Frank", "P1", "P2", "P3", "P4"):
        fc = _values(fs / 6, fs / 5, n, rng, s)
        cyc = pick([3, 4, 5])
        key, opts = ("steps", [6, 8]) if name == "P2" else (("steps", [6, 7, 8]) if name in ("Frank", "P1")
                                                          else ("subcodes", [36, 49, 64]))
        return [dict(fc=fc[i], cycles_per_chip=int(cyc[i]), **{key: int(v)}) for i, v in enumerate(pick(opts))]
    if name in ("T1", "T2"):
        fc = _values(fs / 6, fs / 5, n, rng, s)
        return [dict(fc=fc[i], segments=int(g), phase_states=2) for i, g in enumerate(pick([4, 5, 6]))]
    if name in ("T3", "T4"):
        fc, B = _values(fs / 13, fs / 10, n, rng, s), _values(fs / 20, fs / 10, n, rng, s)
        return [dict(fc=fc[i], bandwidth=B[i], phase_states=2) for i in range(n)]
    raise ValueError(name)


def make_waveform(name, p, n_out, rng, cfg):
    """Complex waveform of class ``name`` with parameters ``p``, linearly resampled to ``n_out`` samples."""
    fs, A = cfg.fs, cfg.amplitude
    if name == "LFM":
        x = W.type_LFM(n_out, fs, A, p["fc"], p["bandwidth"], p["direction"], rng, legacy=cfg.legacy_lfm)
    elif name == "Costas":
        x = W.type_Costas(n_out, fs, A, p["fcmin"], p["hops"], rng)
    elif name == "BPSK":
        x = W.type_Barker(p["cycles_per_chip"], fs, A, p["fc"], p["barker_length"])
    elif name in ("Frank", "P1", "P2"):
        x = getattr(W, f"type_{name}")(p["cycles_per_chip"], fs, A, p["fc"], p["steps"])
    elif name in ("P3", "P4"):
        x = getattr(W, f"type_{name}")(p["cycles_per_chip"], fs, A, p["fc"], p["subcodes"])
    elif name in ("T1", "T2"):
        x = getattr(W, f"type_{name}")(fs, A, p["fc"], p["phase_states"], p["segments"])
    else:
        x = getattr(W, f"type_{name}")(n_out, fs, A, p["fc"], p["phase_states"], p["bandwidth"])
    return W.resample_linear(x, n_out)


# ---------------------------------------------------------------- dataset
def _stratified_validation(n_levels, n_classes, per_class, frac, rng):
    """Boolean mask over the train-run generation order: True = validation."""
    total = n_levels * n_classes * per_class
    mask = np.zeros(total, bool)
    gen = np.arange(total)
    label = (gen % (n_classes * per_class)) // per_class
    for k in range(n_classes):
        idx = gen[label == k]
        mask[rng.choice(idx, int(round(frac * len(idx))), replace=False)] = True
    return mask


class _Writer:
    def __init__(self, path, n, cfg, split, with_dist):
        self.f = h5py.File(path, "w")
        kw = dict(compression="gzip", compression_opts=4, shuffle=True)
        L = cfg.n_samples
        self.f.create_dataset("clean", (n, 2, L), cfg.dtype, chunks=(min(64, n), 2, L), **kw)
        self.f.create_dataset("noisy", (n, 2, L), cfg.dtype, chunks=(min(64, n), 2, L), **kw)
        if with_dist:
            d = self.f.create_dataset("distortions", (n, 3, 2, L), cfg.dtype, chunks=(min(32, n), 3, 2, L), **kw)
            d.attrs["component_order"] = ["AWGN", "Echo", "CCI"]
        for k, t in (("label", "u1"), ("snr_db", "f8"), ("gen_index", "i4")):
            self.f.create_dataset(k, (n,), t)
        self.f.attrs.update(split=split, n_samples=n, class_names=list(cfg.classes), generator="BRSR-DataGen",
                            generator_version=__version__, config=cfg.to_json(),
                            signal_layout="[sample, channel, time]; channel 0 = I, channel 1 = Q")
        self.i, self.written, self.buf = 0, 0, []

    def add(self, clean, noisy, comps, label, snr, gen):
        self.buf.append((np.stack([clean.real, clean.imag]), np.stack([noisy.real, noisy.imag]),
                         None if comps is None else np.stack([comps.real, comps.imag], axis=1), label, snr, gen))
        self.i += 1
        if len(self.buf) == 256:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        a, b = self.written, self.written + len(self.buf)
        self.f["clean"][a:b] = np.stack([r[0] for r in self.buf])
        self.f["noisy"][a:b] = np.stack([r[1] for r in self.buf])
        if self.buf[0][2] is not None:
            self.f["distortions"][a:b] = np.stack([r[2] for r in self.buf])
        self.f["label"][a:b] = [r[3] for r in self.buf]
        self.f["snr_db"][a:b] = [r[4] for r in self.buf]
        self.f["gen_index"][a:b] = [r[5] for r in self.buf]
        self.written, self.buf = b, []

    def close(self):
        self.flush()
        assert self.written == self.f["label"].shape[0]
        self.f.close()


def generate(cfg, out_dir, prefix=None, progress=True):
    """Generate the train/validation/test splits and a per-sample metadata CSV into ``out_dir``."""
    os.makedirs(out_dir, exist_ok=True)
    prefix = prefix or ("brsr" if cfg.mode == "brsr" else "awgn_baseline")
    rng = np.random.default_rng(cfg.seed)
    bank = load_interference_bank() if cfg.mode == "brsr" else None
    comps_allowed = [tuple(c.split("+")) for c in cfg.compositions]
    n_long = cfg.n_samples * (cfg.long_factor if cfg.mode == "brsr" else 1)
    n_cls, n_lev = len(cfg.classes), len(cfg.snr_levels)
    meta_rows = []

    for run, per_class in (("train", cfg.n_train), ("test", cfg.n_test)):
        total = n_lev * n_cls * per_class
        val_mask = (_stratified_validation(n_lev, n_cls, per_class, cfg.val_fraction, rng)
                    if run == "train" else np.zeros(total, bool))
        splits = ["train", "validation"] if run == "train" else ["test"]
        writers = {sp: _Writer(os.path.join(out_dir, f"{prefix}_{sp}.h5"),
                               int((val_mask if sp == "validation" else ~val_mask).sum()), cfg, sp,
                               cfg.mode == "brsr") for sp in splits}
        gen = 0
        for lev in cfg.snr_levels:
            for k, name in enumerate(cfg.classes):
                for p in sample_parameters(name, per_class, rng, cfg):
                    wav = make_waveform(name, p, n_long, rng, cfg)
                    if cfg.mode == "brsr":
                        snr = cfg.snr_min + (cfg.snr_max - cfg.snr_min) * rng.random()
                        clean, noisy, comps, info = add_artifacts(wav, snr, bank, rng, cfg.n_samples,
                                                                  cfg.echo_delay, comps_allowed)
                    else:
                        snr = float(lev)
                        clean = wav
                        noisy, _ = add_awgn_measured(clean, snr, rng)
                        comps, info = None, {}
                    sp = "validation" if val_mask[gen] else ("train" if run == "train" else "test")
                    w = writers[sp]
                    row = w.i
                    w.add(clean, noisy, comps, k + 1, snr, gen)
                    measured = 10 * np.log10(np.sum(np.abs(clean) ** 2) / np.sum(np.abs(noisy - clean) ** 2))
                    meta_rows.append(dict(split=sp, row=row, gen_index=gen, label=k + 1, class_name=name,
                                          snr_target_db=round(snr, 6), snr_measured_db=round(float(measured), 6),
                                          **{k2: (round(v, 6) if isinstance(v, float) else v) for k2, v in info.items()},
                                          params=json.dumps({a: (round(b, 3) if isinstance(b, float) else b)
                                                             for a, b in p.items()})))
                    gen += 1
            if progress:
                print(f"  {run}: SNR loop {lev:+d} dB done ({gen}/{total})", flush=True)
        for w in writers.values():
            w.close()

    order = {"train": 0, "validation": 1, "test": 2}
    meta_rows.sort(key=lambda r: (order[r["split"]], r["row"]))
    keys = list(dict.fromkeys(k for r in meta_rows for k in r))
    with open(os.path.join(out_dir, f"{prefix}_metadata.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        wr.writerows(meta_rows)
    return {sp: os.path.join(out_dir, f"{prefix}_{sp}.h5") for sp in ("train", "validation", "test")}
