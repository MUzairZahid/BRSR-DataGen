"""Make the v2 README figures in light and dark mode from the shared design system.

    pip install -e ".[figures]"      # matplotlib; ffmpeg must be on PATH for the GIFs
    python scripts/make_figures_v2.py [--no-gifs]

This reuses the plotting code of scripts/make_figures.py unchanged (it is imported,
not copied) and only swaps its colours for the palette in scripts/brsr_palette.py,
so every figure uses the same component colours as the Signal Observatory v2 page:
clean = blue, echo = green, interference = orange, AWGN = neutral, received = ink.
The seeds are fixed for reproducibility. Costas uses a validated hop sequence.

Outputs (docs/figures/v2/):
    waveform_gallery_{light,dark}.png
    compositions_{light,dark}.png
    corruption_buildup_{light,dark}.gif   (about a third of the originals' size)
    snr_sweep_{light,dark}.gif

The original figures in docs/figures/ are not touched.
"""
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures", "v2")


def load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


palette = load("brsr_palette")
mf = load("make_figures")
mf.PARAMS['Costas'] = dict(mf.PARAMS['Costas'], hops=[1, 2, 4, 3])
assert all(len({mf.PARAMS['Costas']['hops'][i + d] - mf.PARAMS['Costas']['hops'][i]
                for i in range(4 - d)}) == 4 - d for d in range(1, 4))


def to_gif_small(frame_dir, out_path, fps=12, width=640):
    """Same ffmpeg pipeline as make_figures.to_gif, but 640 px wide, 48 colours and every second
    frame (played at half the frame rate, so timing is unchanged). About a third of the original size."""
    pal = os.path.join(frame_dir, "pal.png")
    src = os.path.join(frame_dir, "f%04d.png")
    pick = f"select='not(mod(n\\,2))',setpts=N/({fps / 2}*TB),scale={width}:-1:flags=lanczos"
    ffmpeg = os.environ.get('BRSR_FFMPEG', 'ffmpeg')
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", src, "-vf",
                    f"{pick},palettegen=max_colors=48:stats_mode=diff", pal], check=True)
    subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps), "-i", src, "-i", pal, "-lavfi",
                    f"{pick}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle",
                    "-r", str(fps / 2), "-loop", "0", out_path], check=True)


def apply_theme(theme):
    t = palette.THEMES[theme]
    plt.rcParams.update(palette.matplotlib_rc(theme))
    mf.SURFACE, mf.INK, mf.INK2, mf.MUTED, mf.GRID = t["surface"], t["ink"], t["ink2"], t["muted"], t["grid"]
    mf.C_CLEAN, mf.C_Q = t["clean"], t["awgn"]
    mf.C_CORRUPT = t["received"]          # the received mixture is drawn in ink, like the page
    mf.SPEC = palette.matplotlib_cmap(theme)
    mf.to_gif = to_gif_small


def build(theme, gifs=True):
    apply_theme(theme)
    tmp = tempfile.mkdtemp()
    mf.OUT = tmp
    bank = mf.load_interference_bank()
    mf.waveform_gallery(np.random.default_rng(1))
    mf.compositions(np.random.default_rng(3), bank)
    if gifs:
        mf.corruption_buildup(np.random.default_rng(2), bank)
        mf.snr_sweep(np.random.default_rng(4), bank)
    os.makedirs(OUT, exist_ok=True)
    for name in os.listdir(tmp):
        stem, ext = os.path.splitext(name)
        dest = os.path.join(OUT, f"{stem}_{theme}{ext}")
        shutil.move(os.path.join(tmp, name), dest)
        print(f"  {dest} ({os.path.getsize(dest) / 1024:.0f} KB)")
    shutil.rmtree(tmp)


def main():
    gifs = "--no-gifs" not in sys.argv
    for theme in ("light", "dark"):
        print(f"{theme}:")
        build(theme, gifs)
    print("Saved v2 figures to", OUT)


if __name__ == "__main__":
    main()
