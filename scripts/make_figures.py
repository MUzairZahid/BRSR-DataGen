"""Make the README figures of BRSR-DataGen with the Python generator.

    pip install -e ".[figures]"      # matplotlib; ffmpeg must be on PATH for the GIFs
    python scripts/make_figures.py

Outputs (docs/figures/):
    waveform_gallery.png      the 12 waveform classes: short I/Q piece + spectrogram
    radar_environment.gif     emitter -> echo -> interference -> receiver noise, and the received signal
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


# ---------------------------------------------------------------- 2b. radar environment animation
C_ECHO, C_CCI, C_NOISE, C_RX = "#1baf7a", "#eb6834", "#8a8983", "#52514e"


def _packet(ax, p, q, s, color, length=1.3, amp=0.18, cycles=5, lw=2.0):
    """Draw a wave packet centred at fraction s along the segment p -> q."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.linalg.norm(d)
    u = d / L
    nrm = np.array([-u[1], u[0]])
    c = p + d * s
    x = np.linspace(-length / 2, length / 2, 120)
    env = np.cos(np.pi * x / length) ** 2
    y = amp * env * np.sin(2 * np.pi * cycles * x / length)
    pts = c[None] + x[:, None] * u[None] + y[:, None] * nrm[None]
    keep = ((pts - p) @ u >= 0) & ((pts - p) @ u <= L)
    ax.plot(pts[keep, 0], pts[keep, 1], color=color, lw=lw, solid_capstyle="round", zorder=5)


def _path(ax, pts, color, alpha=0.55):
    pts = np.asarray(pts, float)
    ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.2, ls=(0, (4, 3)), alpha=alpha, zorder=2)
    a, b = pts[-2], pts[-1]
    m = a + 0.93 * (b - a)
    ax.annotate("", xy=b - 0.25 * (b - a) / np.linalg.norm(b - a), xytext=m - 0.3 * (b - a) / np.linalg.norm(b - a),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2, alpha=alpha), zorder=2)


def _wavefronts(ax, x, y, phase, color, facing=1, n=3, rmax=1.6):
    """Expanding arcs in front of an antenna (phase in [0, 1) animates them)."""
    from matplotlib.patches import Arc
    for i in range(n):
        r = rmax * ((phase + i / n) % 1.0)
        a = max(0.0, 1 - r / rmax)
        theta = 0 if facing > 0 else 180
        ax.add_patch(Arc((x, y + 0.05), 2 * r, 2 * r, angle=theta, theta1=-35, theta2=35, color=color, lw=1.6,
                         alpha=0.8 * a, zorder=3))


def _tower(ax, x, y, facing, label, sub):
    from matplotlib.patches import Arc, Polygon
    ax.add_patch(Polygon([[x - 0.35, y - 1.0], [x + 0.35, y - 1.0], [x, y - 0.05]], closed=True, fc=GRID, ec=INK2, lw=1.2))
    ax.plot([x, x], [y - 1.0, y], color=INK2, lw=1.5)
    theta = 0 if facing > 0 else 180
    ax.add_patch(Arc((x - 0.12 * facing, y + 0.05), 0.7, 0.9, angle=theta, theta1=-80, theta2=80, color=INK, lw=2.2))
    ax.plot([x - 0.12 * facing, x + 0.22 * facing], [y + 0.05, y + 0.05], color=INK, lw=1.2)
    ax.text(x, y - 1.25, label, ha="center", va="top", fontsize=10, color=INK, fontweight="bold")
    ax.text(x, y - 1.6, sub, ha="center", va="top", fontsize=8.5, color=INK2)


def radar_environment(rng, bank, name="LFM", snr=-3.0):
    from matplotlib.patches import Polygon
    long_wav = make_waveform(name, PARAMS[name], NLONG, rng, CFG)
    clean, noisy, comps, info = add_artifacts(long_wav, snr, bank, rng, composition=("AWGN", "Echo", "CCI"))
    rms = np.sqrt(np.mean(np.abs(clean) ** 2))
    clean, comps = clean / rms, comps / rms
    tx, rx, refl, intf = (1.2, 3.2), (9.0, 3.2), (5.0, 5.35), (6.3, 1.3)
    noise_pts = np.random.default_rng(7).normal(0, 1, (160, 2)) * [0.42, 0.42] + [rx[0] - 0.05, rx[1] + 0.15]
    vmax = spec_db(clean).max()
    ylim = 1.08 * np.abs((clean + comps.sum(0)).real[WIN]).max()
    tt = np.arange(WIN.stop - WIN.start) / FS * 1e6

    # (stage, caption, active packet list [(path points, s, color)], received components, noise alpha)
    frames = []
    intro = "A clean radar waveform travels to the receiver, and the environment adds to it"
    frames += [(0, intro, [], None, 0.0)] * 10
    for k in range(24):
        frames.append((1, "1 · Direct path: the transmitted waveform", [((tx, rx), k / 23, C_CLEAN)], None, 0.0))
    frames += [(1, "1 · Direct path: the transmitted waveform", [], (1, 0, 0, 0), 0.0)] * 12
    for k in range(30):
        seg = ((tx, refl), k / 14, C_ECHO) if k < 15 else ((refl, rx), (k - 15) / 14, C_ECHO)
        frames.append((2, f"2 · Echo: a delayed copy via a reflector (τ = {info['echo_delay']} samples)", [seg],
                       (1, 0, 0, 0), 0.0))
    frames += [(2, f"2 · Echo: a delayed copy via a reflector (τ = {info['echo_delay']} samples)", [], (1, 1, 0, 0), 0.0)] * 12
    for k in range(22):
        frames.append((3, "3 · Co-channel interference from another emitter", [((intf, rx), k / 21, C_CCI)],
                       (1, 1, 0, 0), 0.0))
    frames += [(3, "3 · Co-channel interference from another emitter", [], (1, 1, 1, 0), 0.0)] * 12
    for k in range(15):
        frames.append((4, "4 · Receiver noise (AWGN)", [], (1, 1, 1, (k + 1) / 15), (k + 1) / 15))
    frames += [(4, "4 · Receiver noise (AWGN)", [], (1, 1, 1, 1), 1.0)] * 8
    frames += [(5, f"Received signal · SNR {fmt_db(snr)} dB · {name} with echo, interference and noise", [],
                (1, 1, 1, 1), 1.0)] * 30

    tmp = tempfile.mkdtemp()
    fig = plt.figure(figsize=(9.6, 6.4), dpi=100)
    for n, (stage, caption, packets, mix, noise_a) in enumerate(frames):
        fig.clf()
        gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], width_ratios=[1.2, 1], hspace=0.38, wspace=0.22,
                              left=0.07, right=0.98, top=0.88, bottom=0.09)
        sc = fig.add_subplot(gs[0, :])
        sc.set_xlim(0, 10.2); sc.set_ylim(0.0, 6.0); sc.axis("off")
        fig.text(0.07, 0.975, f"BRSR-DataGen · the radar environment behind the BRSR signal model", fontsize=12.5,
                 fontweight="bold", color=INK, va="top")
        fig.text(0.07, 0.935, caption, fontsize=10.5, color=INK2 if stage == 0 else INK, va="top")
        # scene
        sc.add_patch(Polygon([[3.7, 4.95], [4.4, 5.75], [5.0, 5.45], [5.6, 5.9], [6.3, 4.95]], closed=True, fc=GRID,
                             ec=MUTED, lw=1.0))
        sc.text(6.45, 5.35, "reflector\n(building / terrain)", fontsize=8.5, color=INK2, va="center")
        _tower(sc, *tx, 1, "Radar emitter", f"transmits a clean {name} waveform")
        _tower(sc, *rx, -1, "Receiver", "records 1024 I/Q samples")
        _tower(sc, *intf, -1, "Interferer", "another emitter, same band")
        sc.plot([0.2, 10.0], [2.2, 2.2], color=GRID, lw=1.0, zorder=0)            # ground
        if stage == 1 and packets:
            _wavefronts(sc, *tx, n / 8.0 % 1.0, C_CLEAN, 1)
        if stage == 3 and packets:
            _wavefronts(sc, *intf, n / 8.0 % 1.0, C_CCI, 1, rmax=1.2)
        if stage >= 1:
            _path(sc, [tx, rx], C_CLEAN, 0.5 if stage > 1 else 0.8)
            sc.text(3.3, 3.32, "direct path", fontsize=8.5, color=INK2)
        if stage >= 2:
            sc.text(2.55, 4.45, "echo path", fontsize=8.5, color=INK2, rotation=19)
        if stage >= 3:
            sc.text(7.4, 1.95, "interference", fontsize=8.5, color=INK2, rotation=40)
        if stage >= 2:
            _path(sc, [tx, refl, rx], C_ECHO, 0.5 if stage > 2 else 0.8)
        if stage >= 3:
            _path(sc, [intf, rx], C_CCI, 0.5 if stage > 3 else 0.8)
        if noise_a > 0:
            sc.scatter(noise_pts[:, 0], noise_pts[:, 1], s=5, color=C_NOISE, alpha=0.6 * noise_a, lw=0, zorder=4)
            sc.text(rx[0] + 0.55, rx[1] + 0.75, "noise", fontsize=8.5, color=INK2, alpha=noise_a)
        for pts, s_, col in packets:
            _packet(sc, pts[0], pts[1], min(max(s_, 0), 1), col)
        if stage == 5:
            sc.text(10.1, 0.45, "→ saved as a training pair:\nclean, received, components",
                    ha="right", va="center", fontsize=9, color=INK,
                    bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=MUTED, lw=1))
        # received time signal
        at, asp = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
        at.set_title("Received signal (I channel)", loc="left", fontsize=10)
        at.set_xlim(0, tt[-1]); at.set_ylim(-ylim, ylim)
        at.grid(axis="y", color=GRID, lw=0.8); at.tick_params(length=0)
        for s in ("top", "right"):
            at.spines[s].set_visible(False)
        at.set_xlabel("time (µs)", fontsize=9); at.set_ylabel("amplitude", fontsize=9)
        asp.set_title("Spectrogram", loc="left", fontsize=10)
        if mix is None:
            at.text(0.5, 0.5, "waiting for the signal …", transform=at.transAxes, ha="center", va="center",
                    color=MUTED, fontsize=10)
            asp.axis("off")
        else:
            z = clean * mix[0] + comps[1] * mix[1] + comps[2] * mix[2] + comps[0] * mix[3]
            if stage >= 2:
                at.plot(tt, clean.real[WIN], color=C_CLEAN, lw=1.2, ls=(0, (3, 2)))
            at.plot(tt, z.real[WIN], color=C_CLEAN if stage == 1 else C_RX, lw=1.5)
            keys = [("received", C_CLEAN if stage == 1 else C_RX, "-")] + ([("clean", C_CLEAN, (0, (3, 2)))] if stage >= 2 else [])
            for k, (lab, c, ls) in enumerate(keys):
                at.plot([0.62, 0.7], [1.08 - 0.13 * k] * 2, color=c, lw=1.5, ls=ls, transform=at.transAxes, clip_on=False)
                at.text(0.72, 1.08 - 0.13 * k, lab, fontsize=8.5, color=INK2, va="center", transform=at.transAxes)
            show_spec(asp, z, vmax=vmax)
            asp.set_xlabel("time (µs)", fontsize=9); asp.set_ylabel("MHz", fontsize=9)
            asp.set_yticks([-40, 0, 40])
        fig.savefig(os.path.join(tmp, f"f{n:04d}.png"))
    plt.close(fig)
    to_gif(tmp, os.path.join(OUT, "radar_environment.gif"), fps=15, width=900)
    shutil.rmtree(tmp)


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
    radar_environment(np.random.default_rng(5), bank)
    corruption_buildup(np.random.default_rng(2), bank)
    compositions(np.random.default_rng(3), bank)
    snr_sweep(np.random.default_rng(4), bank)
    print("Saved figures to", OUT)


if __name__ == "__main__":
    main()
