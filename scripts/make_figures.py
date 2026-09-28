"""Make the README figures of BRSR-DataGen with the Python generator.

    pip install -e ".[figures]"      # matplotlib; ffmpeg must be on PATH for the GIFs
    python scripts/make_figures.py

The radar-environment animation (docs/figures/radar_environment.gif) is a recording of the interactive
page docs/radar_environment.html: see make_environment_page.py and record_environment_gif.py.

Outputs (docs/figures/):
    waveform_gallery.png      the 12 waveform classes: short I/Q piece + spectrogram
    corruption_buildup.gif    clean -> + echo -> + interference -> + noise (LFM and Costas)
    compositions.png          the 7 artifact compositions of the BRSR model on one waveform
    snr_sweep.gif             one corrupted waveform from +10 dB down to -14 dB
"""
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
from brsr_datagen import COMPOSITIONS, Config, add_artifacts, load_interference_bank, make_waveform  # noqa: E402
from brsr_datagen.waveforms import CLASS_NAMES  # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
CFG = Config(legacy_lfm=False)                   # unit-amplitude LFM for display
FS, N, NLONG = CFG.fs, CFG.n_samples, CFG.n_samples * CFG.long_factor
WIN = slice(0, 128)

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0"
C_CLEAN, C_CORRUPT = "#2a78d6", "#eb6834"
SPEC = LinearSegmentedColormap.from_list("blue_seq", ["#fcfcfb", "#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                     "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE})

PARAMS = {  # fixed, mid-range parameters for the figures
    "LFM": dict(fc=18e6, bandwidth=5.6e6, direction="Up"),
    "Costas": dict(fcmin=3.7e6, hops=[3, 1, 4, 2]),
    "BPSK": dict(fc=8.8e6, barker_length=13, cycles_per_chip=20),
    "Frank": dict(fc=18e6, cycles_per_chip=4, steps=7),
    "P1": dict(fc=18e6, cycles_per_chip=4, steps=7),
    "P2": dict(fc=18e6, cycles_per_chip=4, steps=8),
    "P3": dict(fc=18e6, cycles_per_chip=4, subcodes=49),
    "P4": dict(fc=18e6, cycles_per_chip=4, subcodes=49),
    "T1": dict(fc=18e6, segments=5, phase_states=2),
    "T2": dict(fc=18e6, segments=5, phase_states=2),
    "T3": dict(fc=8.8e6, bandwidth=7.5e6, phase_states=2),
    "T4": dict(fc=8.8e6, bandwidth=7.5e6, phase_states=2),
}


def norm(z):
    x = np.stack([z.real, z.imag])
    lo, hi = x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True)
    return 2 * (x - lo) / (hi - lo) - 1


def spec_db(z, nfft=64, hop=8):
    w = np.hanning(nfft)
    frames = np.stack([z[i:i + nfft] * w for i in range(0, len(z) - nfft + 1, hop)])
    return 10 * np.log10(np.abs(np.fft.fftshift(np.fft.fft(frames, axis=1), axes=1)).T ** 2 + 1e-12)


def fmt_db(v):
    return f"{v:+.0f}".replace("-", "−")


def show_spec(ax, z, vmax=None, title=None):
    S = spec_db(z / np.sqrt(np.mean(np.abs(z) ** 2)))
    vmax = S.max() if vmax is None else vmax
    ax.imshow(S, aspect="auto", origin="lower", cmap=SPEC, vmin=vmax - 45, vmax=vmax,
              extent=[0, N / FS * 1e6, -FS / 2e6, FS / 2e6])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=10.5)


def style_time(ax):
    ax.set_xlim(0, (WIN.stop - WIN.start) / FS * 1e6)
    ax.set_ylim(-1.35, 1.35)
    ax.set_yticks([-1, 0, 1])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(length=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("time (µs)", fontsize=9)
    ax.set_ylabel("I channel (normalized)", fontsize=9)


def to_gif(frame_dir, out_path, fps=12, width=860):
    pal = os.path.join(frame_dir, "pal.png")
    src = os.path.join(frame_dir, "f%04d.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", src, "-vf",
                    f"scale={width}:-1:flags=lanczos,palettegen=max_colors=128:stats_mode=diff", pal], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", src, "-i", pal, "-lavfi",
                    f"scale={width}:-1:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    "-loop", "0", out_path], check=True)


# ---------------------------------------------------------------- 1b. waveform gallery (snippet + spectrogram)
DESCR = {"LFM": "linear chirp", "Costas": "frequency hopping", "BPSK": "Barker-13 phase code",
         "Frank": "polyphase code", "P1": "polyphase code", "P2": "polyphase code", "P3": "polyphase code",
         "P4": "polyphase code", "T1": "polytime, stepped phase", "T2": "polytime, stepped phase",
         "T3": "polytime, linear phase", "T4": "polytime, linear phase"}
C_Q = "#8a8983"


def interesting_window(z, width=56):
    """Window around the largest phase jump (chip / hop boundary); LFM: the start."""
    ph = np.unwrap(np.angle(z))
    jump = np.abs(np.diff(ph, 2))
    jump[:width] = jump[-width:] = 0
    c = int(np.argmax(jump)) + 1 if jump.max() > 0.5 else None
    a = max(0, min(len(z) - width, (c if c is not None else width // 2) - width // 2))
    return slice(a, a + width), c


def waveform_gallery(rng):
    from matplotlib.patches import Rectangle
    fig = plt.figure(figsize=(13, 11.5), dpi=120)
    outer = fig.add_gridspec(3, 4, hspace=0.42, wspace=0.12, left=0.05, right=0.99, top=0.885, bottom=0.05)
    for idx, name in enumerate(CLASS_NAMES):
        z = make_waveform(name, PARAMS[name], N, rng, CFG)
        z = z / np.sqrt(np.mean(np.abs(z) ** 2))
        w, jump_at = interesting_window(z)
        inner = outer[idx // 4, idx % 4].subgridspec(2, 1, height_ratios=[1, 1.25], hspace=0.08)
        at, asp = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        t = np.arange(w.start, w.stop) / FS * 1e6
        at.plot(t, z.imag[w], color=C_Q, lw=1.0)
        at.plot(t, z.real[w], color=C_CLEAN, lw=1.5)
        at.set_xlim(t[0], t[-1]); at.set_ylim(-1.8, 1.8)
        at.set_xticks([]); at.set_yticks([])
        for s in at.spines.values():
            s.set_visible(False)
        at.axhline(0, color=GRID, lw=0.8, zorder=0)
        if jump_at is not None:
            at.axvline(jump_at / FS * 1e6, color=INK, lw=0.9, ls=(0, (2, 2)), zorder=0)
        at.set_title(f"{name}  ·  {DESCR[name]}", loc="left", fontsize=10.5)
        show_spec(asp, z)
        asp.add_patch(Rectangle((t[0], -FS / 2e6), t[-1] - t[0], FS / 1e6, fill=False, ec=INK, lw=1.0, ls=(0, (2, 2))))
        asp.set_xticks([0, 5, 10]); asp.set_yticks([-40, 0, 40])
        if idx % 4 == 0:
            asp.set_ylabel("MHz", fontsize=9)
        else:
            asp.set_yticklabels([])
        if idx >= 8:
            asp.set_xlabel("time (µs)", fontsize=9)
        else:
            asp.set_xticklabels([])
    fig.text(0.05, 0.975, "The 12 radar waveform classes", fontsize=14, fontweight="bold", color=INK, va="top")
    fig.text(0.05, 0.948, "Top: a 0.56 µs piece of each signal (I in blue, Q in gray); the dashed line marks a phase or "
             "frequency change.\nBottom: spectrogram of the full 10.24 µs signal; the dashed box shows where the piece "
             "above comes from.", fontsize=10, color=INK2, va="top", linespacing=1.4)
    fig.savefig(os.path.join(OUT, "waveform_gallery.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 2. corruption build-up animation
def corruption_buildup(rng, bank):
    steps = ["Clean", "+ Echo", "+ Interference", "+ Noise", "Received"]
    tmp = tempfile.mkdtemp()
    fig = plt.figure(figsize=(9.6, 5.4), dpi=100)
    tt = np.arange(WIN.stop - WIN.start) / FS * 1e6
    n = 0
    for name, snr in (("LFM", -4.0), ("Costas", -4.0)):
        long_wav = make_waveform(name, PARAMS[name], NLONG, rng, CFG)
        clean, noisy, comps, info = add_artifacts(long_wav, snr, bank, rng, composition=("AWGN", "Echo", "CCI"))
        vmax = spec_db(clean / np.sqrt(np.mean(np.abs(clean) ** 2))).max()
        frames = [(0, "Clean radar signal", clean)] * 8
        acc = clean.copy()
        for stage, (c, label) in enumerate(((1, "Echo"), (2, "Interference"), (0, "Noise")), start=1):
            for k in range(6):
                frames.append((stage, f"+ {label}", acc + (k + 1) / 6 * comps[c]))
            acc = acc + comps[c]
            frames += [(stage, f"+ {label}", acc)] * 3
        frames += [(4, f"Received signal · SNR {fmt_db(snr)} dB · echo delay {info['echo_delay']} samples", acc)] * 20
        for stage, title, z in frames:
            fig.clf()
            gs = fig.add_gridspec(3, 1, height_ratios=[0.34, 1.0, 1.15], hspace=0.55, left=0.08, right=0.83,
                                  top=0.95, bottom=0.1)
            ax0, ax1, ax2 = (fig.add_subplot(gs[i]) for i in range(3))
            ax0.axis("off")
            ax0.text(0, 1.0, f"BRSR-DataGen · signal model · {name} waveform", fontsize=12.5, fontweight="bold",
                     color=INK, va="top", transform=ax0.transAxes)
            x, r = 0.0, fig.canvas.get_renderer()
            for k, s in enumerate(steps):
                col = INK if k == stage else (INK2 if k < stage else MUTED)
                t = ax0.text(x, 0.05, s, fontsize=10, color=col, fontweight="bold" if k == stage else "normal",
                             transform=ax0.transAxes, va="bottom")
                x = t.get_window_extent(r).transformed(ax0.transAxes.inverted()).x1 + 0.012
                if k < len(steps) - 1:
                    a = ax0.text(x, 0.05, "›", fontsize=10, color=MUTED, transform=ax0.transAxes, va="bottom")
                    x = a.get_window_extent(r).transformed(ax0.transAxes.inverted()).x1 + 0.012
            ax1.set_title(title, loc="left", fontsize=11, pad=4)
            col = C_CLEAN if stage == 0 else C_CORRUPT
            y = norm(z)
            ax1.plot(tt, y[0, WIN], color=col, lw=1.8 if stage == 0 else 1.4)
            ax1.plot([1.02, 1.07], [0.85, 0.85], color=col, lw=1.8, transform=ax1.transAxes, clip_on=False)
            ax1.text(1.085, 0.85, "clean" if stage == 0 else "received", color=INK2, fontsize=9, va="center",
                     transform=ax1.transAxes)
            style_time(ax1)
            show_spec(ax2, z, vmax=vmax)
            ax2.set_ylabel("frequency (MHz)", fontsize=9)
            ax2.set_xlabel("time (µs)", fontsize=9)
            ax2.set_title("Spectrogram (full 10.24 µs signal)", loc="left", fontsize=10, color=INK2, pad=4)
            fig.savefig(os.path.join(tmp, f"f{n:04d}.png"))
            n += 1
    plt.close(fig)
    to_gif(tmp, os.path.join(OUT, "corruption_buildup.gif"))
    shutil.rmtree(tmp)


# ---------------------------------------------------------------- 3. artifact compositions
def compositions(rng, bank, name="P4", snr=0.0):
    long_wav = make_waveform(name, PARAMS[name], NLONG, rng, CFG)
    state = rng.bit_generator.state
    fig, axes = plt.subplots(2, 4, figsize=(12, 5), dpi=130, sharex=True, sharey=True)
    clean = None
    for ax, combo in zip(axes.ravel()[1:], COMPOSITIONS):
        rng.bit_generator.state = state          # same segment, delay and interference signal in every panel
        clean, noisy, _, _ = add_artifacts(long_wav, snr, bank, rng, composition=combo)
        show_spec(ax, noisy, vmax=spec_db(clean / np.sqrt(np.mean(np.abs(clean) ** 2))).max(), title=" + ".join(combo))
    show_spec(axes[0, 0], clean, title="Clean")
    for ax in axes[-1]:
        ax.set_xlabel("time (µs)", fontsize=9)
    for ax in axes[:, 0]:
        ax.set_ylabel("frequency (MHz)", fontsize=9)
    fig.suptitle(f"The 7 artifact compositions of the BRSR model on one {name} signal (input SNR {fmt_db(snr)} dB)",
                 x=0.01, ha="left", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(os.path.join(OUT, "compositions.png"))
    plt.close(fig)


# ---------------------------------------------------------------- 4. SNR sweep animation
def snr_sweep(rng, bank, name="LFM"):
    long_wav = make_waveform(name, PARAMS[name], NLONG, rng, CFG)
    clean, _, comps, info = add_artifacts(long_wav, 0.0, bank, rng, composition=("AWGN", "Echo", "CCI"))
    dist = comps.sum(axis=0)
    p_c, p_d = np.mean(np.abs(clean) ** 2), np.mean(np.abs(dist) ** 2)
    vmax = spec_db(clean / np.sqrt(p_c)).max()
    levels = list(np.linspace(10, -14, 49)) + [-14.0] * 12
    tmp = tempfile.mkdtemp()
    fig = plt.figure(figsize=(9.6, 4.6), dpi=100)
    tt = np.arange(WIN.stop - WIN.start) / FS * 1e6
    for n, s in enumerate(levels):
        z = clean + np.sqrt(p_c / (p_d * 10 ** (s / 10))) * dist
        fig.clf()
        gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15], hspace=0.5, left=0.08, right=0.83, top=0.8, bottom=0.12)
        fig.text(0.08, 0.96, f"BRSR-DataGen · {name} waveform with echo + interference + noise", fontsize=12.5,
                 fontweight="bold", color=INK, va="top")
        ax1, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        ax1.set_title(f"Input SNR {fmt_db(s)} dB", loc="left", fontsize=11, pad=4)
        ax1.plot(tt, norm(clean)[0, WIN], color=C_CLEAN, lw=1.3, ls=(0, (3, 2)))
        ax1.plot(tt, norm(z)[0, WIN], color=C_CORRUPT, lw=1.4)
        for k, (lab, c, ls) in enumerate((("received", C_CORRUPT, "-"), ("clean", C_CLEAN, (0, (3, 2))))):
            ax1.plot([1.02, 1.07], [0.85 - 0.22 * k] * 2, color=c, lw=1.5, ls=ls, transform=ax1.transAxes, clip_on=False)
            ax1.text(1.085, 0.85 - 0.22 * k, lab, color=INK2, fontsize=9, va="center", transform=ax1.transAxes)
        style_time(ax1)
        show_spec(ax2, z, vmax=vmax)
        ax2.set_ylabel("frequency (MHz)", fontsize=9)
        ax2.set_xlabel("time (µs)", fontsize=9)
        fig.savefig(os.path.join(tmp, f"f{n:04d}.png"))
    plt.close(fig)
    to_gif(tmp, os.path.join(OUT, "snr_sweep.gif"), fps=12)
    shutil.rmtree(tmp)


def main():
    os.makedirs(OUT, exist_ok=True)
    bank = load_interference_bank()
    waveform_gallery(np.random.default_rng(1))
    corruption_buildup(np.random.default_rng(2), bank)
    compositions(np.random.default_rng(3), bank)
    snr_sweep(np.random.default_rng(4), bank)
    print("Saved figures to", OUT)


if __name__ == "__main__":
    main()
