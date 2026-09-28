"""Build the interactive radar-environment page (docs/radar_environment.html).

    pip install -e ".[figures]"
    python scripts/make_environment_page.py

The page shows how one BRSR sample is made: an emitter sends a clean pulse, a building adds a
delayed echo, a second emitter adds co-channel interference and the receiver adds AWGN. All curves
and spectrograms on the page are real output of the Python generator (fixed seeds below), at SNR -3 dB
with all three artifacts (composition AWGN+Echo+CCI).

The page is one self-contained HTML file (data and images embedded), served by GitHub Pages at
https://muzairzahid.github.io/BRSR-DataGen/radar_environment.html
The README GIF is a recording of this page: scripts/record_environment_gif.py
"""
import base64
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from brsr_datagen import waveforms as W  # noqa: E402
from brsr_datagen.artifacts import add_artifacts, load_interference_bank  # noqa: E402

TEMPLATE = os.path.join(ROOT, "scripts", "radar_environment_template.html")
OUT = os.path.join(ROOT, "docs", "radar_environment.html")
FS, N_LONG, N_SHOW, SNR = 100e6, 2048, 160, -3.0
DYN_DB = 30                                     # spectrogram colour range below the clean peak (dB)
CMAP = LinearSegmentedColormap.from_list("ivory_blue", ["#fbfaf6", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

# waveform, generator seed, how the long (2048-sample) clean waveform is made
EXAMPLES = {
    "LFM": (113, lambda rng: W.type_LFM(N_LONG, FS, 1, 18e6, 5.6e6, "Up", rng, legacy=False)),
    "Costas": (113, lambda rng: W.type_Costas(N_LONG, FS, 1, 3.7e6, [3, 1, 4, 2], rng)),
    "BPSK": (107, lambda rng: W.resample_linear(W.type_Barker(20, FS, 1, 8.8e6, 13), N_LONG)),
}


def spec_db(z, nfft=64, hop=8):
    w = np.hanning(nfft)
    frames = np.stack([z[i:i + nfft] * w for i in range(0, len(z) - nfft + 1, hop)])
    return 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)).T ** 2 + 1e-12)


def spec_png(z, vmax):
    """640 x 260 px spectrogram, no axes; time left to right, frequency -50 (bottom) to +50 MHz (top)."""
    fig = plt.figure(figsize=(6.4, 2.6), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(spec_db(z), aspect="auto", origin="lower", cmap=CMAP, vmin=vmax - DYN_DB, vmax=vmax,
              interpolation="bilinear")
    ax.axis("off")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    small = io.BytesIO()                                     # 64-colour palette keeps the page small
    Image.open(buf).convert("RGB").quantize(colors=64, method=Image.Quantize.MEDIANCUT).save(small, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(small.getvalue()).decode()


def main():
    bank = load_interference_bank()
    sig, images = {}, {}
    for name, (seed, make) in EXAMPLES.items():
        rng = np.random.default_rng(seed)
        clean, noisy, comps, info = add_artifacts(make(rng), SNR, bank, rng, composition=("AWGN", "Echo", "CCI"))
        s = np.sqrt(np.mean(np.abs(clean) ** 2))               # clean RMS = 1
        noise, echo, cci = comps / s
        c = clean / s
        show = lambda x: np.round(x.real[:N_SHOW], 3).tolist()  # I channel, first 160 samples
        stages = [c, c + echo, c + echo + cci, c + echo + cci + noise]   # what the page shows, step by step
        ymax = max(float(np.abs(z.real[:N_SHOW]).max()) for z in stages) * 1.05
        sig[name] = dict(clean=show(c), echo=show(echo), cci=show(cci), noise=show(noise),
                         delay=info["echo_delay"], bank_row=info["cci_bank_row"],
                         w=[round(float(info[k]), 2) for k in ("w_awgn", "w_echo", "w_cci")],
                         ymax=round(ymax, 2), seed=seed)
        vmax = spec_db(c).max()
        for k, z in enumerate(stages):
            images[f"__SPEC_{name}_{k}__"] = spec_png(z, vmax)
        print(f"{name}: seed {seed}, echo delay {info['echo_delay']}, bank row {info['cci_bank_row']}, "
              f"weights awgn/echo/cci {sig[name]['w']}")

    html = open(TEMPLATE, encoding="utf-8").read()
    html = html.replace("__SIG__", json.dumps(sig, separators=(",", ":")))
    for key, url in images.items():
        html = html.replace(key, url)
    assert "__S" not in html, "unreplaced placeholder"
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write(html)
    print(f"Saved {OUT} ({len(html) / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
