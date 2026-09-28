"""Compare per-class statistics of generated clean signals with the published BRSR test set.

    python download_data.py --dataset brsr --splits test           # in the BRSR-OpGAN repo, or from Zenodo
    python -m brsr_datagen --mode brsr --out gen --n_train 1 --n_test 150
    python scripts/compare_with_published.py path/to/brsr_test.h5 gen/brsr_test.h5
"""
import sys

import h5py
import numpy as np

sys.path.insert(0, "python")
from brsr_datagen.waveforms import CLASS_NAMES  # noqa: E402

FS = 100e6


def stats(path):
    with h5py.File(path, "r") as f:
        c, lab = f["clean"][:].astype(np.float64), f["label"][:]
    z = c[:, 0] + 1j * c[:, 1]
    power = np.mean(np.abs(z) ** 2, axis=1)
    spec = np.abs(np.fft.fftshift(np.fft.fft(z, axis=1), axes=1)) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(z.shape[1], 1 / FS)) / 1e6
    centroid = (spec * f).sum(1) / spec.sum(1)
    bw = np.sqrt((spec * (f - centroid[:, None]) ** 2).sum(1) / spec.sum(1))
    return {k: (np.median(power[lab == k + 1]), np.mean(centroid[lab == k + 1]), np.median(bw[lab == k + 1]))
            for k in range(12)}


a, b = stats(sys.argv[1]), stats(sys.argv[2])
print(f"{'class':7s} {'power (pub / gen)':>20s} {'mean centroid MHz':>20s} {'bandwidth MHz':>16s}")
for k, name in enumerate(CLASS_NAMES):
    print(f"{name:7s} {a[k][0]:9.3f} / {b[k][0]:7.3f} {a[k][1]:9.2f} / {b[k][1]:7.2f} {a[k][2]:7.2f} / {b[k][2]:6.2f}")
