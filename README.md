# BRSR-DataGen: Radar Signal Dataset Generator for Blind Radar Signal Restoration

[![Tests](https://github.com/MUzairZahid/BRSR-DataGen/actions/workflows/tests.yml/badge.svg)](https://github.com/MUzairZahid/BRSR-DataGen/actions/workflows/tests.yml)
[![BRSR dataset](https://zenodo.org/badge/DOI/10.5281/zenodo.23010395.svg)](https://doi.org/10.5281/zenodo.23010395)
[![Paper](https://img.shields.io/badge/Neural%20Networks-2025-blue)](https://doi.org/10.1016/j.neunet.2025.107709)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**BRSR-DataGen** generates paired clean/corrupted **radar signal datasets** for **radar signal restoration, denoising, interference suppression and modulation recognition**. It produces **12 LPI radar waveforms** (LFM, Costas, Barker BPSK, Frank, P1–P4 polyphase codes and T1–T4 polytime codes) as complex I/Q signals. It corrupts them with random blends of **additive white Gaussian noise (AWGN)**, **echo** and **co-channel interference (CCI)** at a chosen SNR.

This is the generator of the **BRSR benchmark** (Blind Radar Signal Restoration), used in **BRSR-OpGAN**, **CoRe-Net** and **XCoRe-Net**. It comes in two equivalent implementations: **MATLAB** (no toolboxes needed) and **Python** (NumPy). Both write HDF5 files in the same layout as the published benchmark, so the output works directly with the [BRSR-OpGAN code](https://github.com/MUzairZahid/BRSR-OpGAN).

<p align="center">
  <a href="https://muzairzahid.github.io/BRSR-DataGen/radar_environment.html"><img src="docs/figures/radar_environment.gif" width="900" alt="Radar environment animation: an emitter sends a clean LFM radar pulse to a receiver; a building reflects a delayed echo, a second emitter adds co-channel interference and the receiver adds noise, while the received waveform and its spectrogram build up"></a>
</p>
<p align="center"><em>How one BRSR sample is made. The clean pulse on the direct path is the training target. A building adds a delayed echo, a second emitter adds co-channel interference and the receiver adds AWGN. The waveform and spectrogram are real generator output (SNR −3 dB).</em><br>
<strong><a href="https://muzairzahid.github.io/BRSR-DataGen/radar_environment.html">Open the interactive version</a></strong>: switch between LFM, Costas and BPSK, pause, or jump to any stage.</p>

> **Looking for the BRSR benchmark itself?** Download it from Zenodo: [DOI 10.5281/zenodo.23010395](https://doi.org/10.5281/zenodo.23010395). The published files are the benchmark. This generator produces *new* data with the same process. It does not re-create the published files, because the original generation was not seeded.

---

## Quick start

### Python

```bash
pip install git+https://github.com/MUzairZahid/BRSR-DataGen.git

# BRSR-style dataset: random blend of AWGN, echo and interference, SNR uniform in [-14, 10] dB
python -m brsr_datagen --mode brsr --out data --seed 0

# AWGN-only dataset at 13 SNR levels (-14:2:10 dB)
python -m brsr_datagen --mode awgn --out data --seed 0

# A small dataset to try things out (5 signals per class per SNR level)
python -m brsr_datagen --mode brsr --out data_small --n_train 5 --n_test 2
```

```python
import numpy as np
from brsr_datagen import Config, make_waveform, sample_parameters, add_artifacts, load_interference_bank

cfg, rng = Config(), np.random.default_rng(0)
params = sample_parameters("Frank", 1, rng, cfg)[0]            # random parameters from the BRSR ranges
wave = make_waveform("Frank", params, 2048, rng, cfg)          # complex, 2x longer for the echo
clean, noisy, components, info = add_artifacts(wave, snr_db=-5, bank=load_interference_bank(), rng=rng)
print(info)   # composition, blend weights, echo delay, interference signal
```

### MATLAB

```matlab
addpath('matlab');
cfg = brsr_config('mode', 'brsr', 'seed', 0);          % or 'mode', 'awgn'
generate_brsr_dataset(cfg, 'data');
```

See `matlab/examples/` for a small dataset and a plot of one signal and its artifact components. No MATLAB toolboxes are required.

## Waveform classes

<p align="center"><img src="docs/figures/waveform_gallery.png" width="900" alt="The 12 LPI radar waveform classes (LFM, Costas, BPSK, Frank, P1, P2, P3, P4, T1, T2, T3, T4): a short I/Q waveform piece and the spectrogram of each"></p>

All signals are complex baseband at fs = 100 MHz. Parameters are drawn from the ranges of the BRSR generator:

| Class | Waveform | Parameters |
|---|---|---|
| LFM | linear frequency modulation | fc ∈ [fs/6, fs/5], bandwidth ∈ [fs/20, fs/16], up or down sweep |
| Costas | frequency hopping | fmin ∈ [fs/30, fs/24], random Costas sequence of length 3–5 |
| BPSK | Barker binary phase code | fc ∈ [fs/13, fs/10], Barker length 7, 11 or 13, 20 cycles per chip |
| Frank, P1 | polyphase codes | fc ∈ [fs/6, fs/5], M ∈ {6, 7, 8}, 3–5 cycles per chip |
| P2 | polyphase code | fc ∈ [fs/6, fs/5], M ∈ {6, 8}, 3–5 cycles per chip |
| P3, P4 | polyphase codes | fc ∈ [fs/6, fs/5], 36, 49 or 64 subcodes, 3–5 cycles per chip |
| T1, T2 | polytime codes (stepped phase) | fc ∈ [fs/6, fs/5], 4–6 segments, 2 phase states |
| T3, T4 | polytime codes (linear phase) | fc ∈ [fs/13, fs/10], bandwidth ∈ [fs/20, fs/10], 2 phase states |

## Signal model

For each clean waveform `x` (BRSR-OpGAN paper, Section 3):

1. Draw a target SNR (uniform in [−14, 10] dB), so the total artifact power is `P = P_x / 10^(SNR/10)`.
2. Pick one of the **7 artifact compositions** uniformly: each of AWGN, echo and CCI alone, the three pairs, or all three.
3. Draw random blend weights that sum to 1, and scale each artifact to power `w_i · P`:
   - **AWGN**: complex Gaussian noise.
   - **Echo**: a delayed copy of the same waveform, `x(t − τ)`, with τ ∈ [128, 512] samples. The waveform is generated 2× longer so the delayed copy is real signal, not padding.
   - **CCI**: one of 50 interference signals (`interference_bank`).
4. `noisy = clean + AWGN + echo + CCI`. The individual components are stored as well.

<p align="center"><img src="docs/figures/corruption_buildup.gif" width="860" alt="A clean LFM and a Costas radar signal corrupted step by step by echo, co-channel interference and noise"></p>
<p align="center"><em>The same steps on the signal itself: echo, interference and noise added one by one (top: I channel; bottom: spectrogram).</em></p>

<p align="center"><img src="docs/figures/compositions.png" width="900" alt="The 7 artifact compositions: AWGN, echo, CCI, AWGN+echo, AWGN+CCI, echo+CCI and all three"></p>

<p align="center"><img src="docs/figures/snr_sweep.gif" width="860" alt="An LFM radar signal with echo, interference and noise as the input SNR goes from +10 dB to -14 dB"></p>
<p align="center"><em>The same corrupted LFM signal from +10 dB down to −14 dB input SNR.</em></p>

## Output format

`generate` / `generate_brsr_dataset` write:

| File | Contents |
|---|---|
| `brsr_train.h5`, `brsr_validation.h5`, `brsr_test.h5` | one split per file; `awgn_baseline_*` in AWGN mode |
| `brsr_metadata.csv` | one row per signal |

The run sizes are those of the BRSR benchmark by default:

- **Train run:** 400 signals per class per SNR level, 62,400 in total. It is split 80/20 into train and validation, stratified by class.
- **Test run:** 150 per class per SNR level, 23,400 in total.

Each HDF5 file holds:

- `clean` and `noisy`: [N, 2, 1024] float32, with channels I and Q;
- `distortions`: [N, 3, 2, 1024], the AWGN, echo and CCI components (BRSR mode);
- `label` (1–12), `snr_db` and `gen_index`;
- the generator configuration, as a file attribute.

MATLAB's `h5read` returns the dimensions in reverse order.

The metadata CSV has, per signal:

- `split` and `row`, to find the signal in the HDF5 files;
- the class;
- target and measured SNR;
- the composition and blend weights (`w_awgn`, `w_echo`, `w_cci`);
- `echo_delay`, `cci_bank_row` and `segment_start`;
- all waveform parameters as JSON (`params`).

## Options

| Option (Python `Config` / MATLAB `brsr_config`) | Default | Meaning |
|---|---|---|
| `mode` | `brsr` | `brsr`: AWGN + echo + CCI blends; `awgn`: AWGN only at the discrete SNR levels |
| `seed` | 0 | random seed (the output is reproducible) |
| `n_train`, `n_test` | 400, 150 | signals per class per SNR level |
| `snr_min`, `snr_max` / `snr_levels` | −14, 10 / −14:2:10 | SNR range (BRSR) / levels (AWGN) |
| `param_sampling` | `grid` | `grid`: shuffled evenly spaced parameter values, as in BRSR v1.0; `uniform`: independent random values |
| `legacy_lfm` | true | reproduce the LFM amplitude behaviour of BRSR v1.0 (see below) |
| `echo_delay` | [128, 512] | echo delay range in samples |
| `compositions` | all 7 | restrict the artifact combinations |

## Faithfulness to BRSR v1.0

The generator keeps the choices of the original BRSR code, so new data follows the published distribution. The defaults reproduce them:

- **Resampling.** Waveforms whose natural length differs from the target are linearly resampled to it, which slightly shifts their frequencies.
- **LFM amplitude.** In the original LFM code the random start phase φ₀ is added outside the imaginary unit, `A·exp(j2πft + φ₀)`. It therefore scales the amplitude by `exp(φ₀)` with φ₀ ∈ [−π, π], a factor between about 0.04 and 23. The published data has this property. Use `--fixed_lfm` / `legacy_lfm=false` for unit-amplitude LFM with a proper phase.
- **Parameter grids.** Parameters come from evenly spaced grids that are reshuffled at every SNR level (`param_sampling = grid`). In AWGN mode, without the random segment cut, this repeats clean waveforms across SNR levels, as in the AWGN-Baseline set. Use `uniform` to avoid it.
- **Interference bank.** The bank holds 50 signals, of which 35 are distinct.

## Validation

- **Python matches MATLAB.** `tests/test_waveforms_match_matlab.py` checks that every Python waveform function equals the MATLAB one (relative error < 1e-9) on reference signals made with `tests/make_matlab_reference.m`. The MATLAB code is also run under GNU Octave to make that reference.
- **The generator is consistent.** `tests/test_generator.py` checks split sizes, class balance, `noisy = clean + components`, exact target SNR for single artifacts, blend weights, artifact powers and seed reproducibility.
- **New data matches the published data.** A test run generated with the Python port matches the published BRSR test set, per class: signal power, spectral centroid and bandwidth, LFM sweep rates, and the LFM amplitude distribution.

```bash
pip install -e ".[test]"
pytest
```

## Repository structure

```
BRSR-DataGen/
├── python/brsr_datagen/      # Python package: waveforms.py, artifacts.py, generator.py, CLI
├── matlab/                   # MATLAB: brsr_config, generate_brsr_dataset, brsr_add_artifacts, ...
│   ├── waveforms/            # type_LFM, type_Costas, type_Barker, type_Frank, type_P1..P4, type_T1..T4
│   └── examples/             # small dataset, plot of one signal
├── tests/                    # pytest suite, MATLAB reference waveforms, Octave shims
├── scripts/
│   ├── make_figures.py            # README figures and animations
│   ├── make_environment_page.py   # builds the interactive radar-environment page
│   └── record_environment_gif.py  # records that page as the README GIF
├── legacy/                   # original scripts used to make BRSR v1.0 (for reference)
└── docs/
    ├── radar_environment.html     # interactive page, served by GitHub Pages
    └── figures/
```

## Related

- **BRSR dataset** (the published benchmark): [Zenodo, DOI 10.5281/zenodo.23010395](https://doi.org/10.5281/zenodo.23010395)
- **BRSR-OpGAN** code and pre-trained models: [github.com/MUzairZahid/BRSR-OpGAN](https://github.com/MUzairZahid/BRSR-OpGAN)
- **CoRe-Net**: Co-Operational Regressor Network with Progressive Transfer Learning for Blind Radar Signal Restoration, *Machine Learning with Applications* 25, 100939 (2026). [doi:10.1016/j.mlwa.2026.100939](https://doi.org/10.1016/j.mlwa.2026.100939)
- **XCoRe-Net**: Expert Co-Operational Regressor Networks for High-Fidelity Restoration of Radar Signals.

## Citation

If you use this generator or data made with it, please cite:

```bibtex
@article{zahid2025brsropgan,
  title   = {{BRSR-OpGAN}: Blind radar signal restoration using operational generative adversarial network},
  author  = {Zahid, Muhammad Uzair and Kiranyaz, Serkan and Yildirim, Alper and Gabbouj, Moncef},
  journal = {Neural Networks},
  volume  = {190},
  pages   = {107709},
  year    = {2025},
  doi     = {10.1016/j.neunet.2025.107709}
}

@dataset{zahid2026brsr_dataset,
  title     = {{BRSR} Dataset: Blind Radar Signal Restoration Benchmark (v1.0)},
  author    = {Zahid, Muhammad Uzair and Kiranyaz, Serkan and Yildirim, Alper and Gabbouj, Moncef},
  publisher = {Zenodo},
  version   = {1.0},
  year      = {2026},
  doi       = {10.5281/zenodo.23010395}
}
```

## License

MIT (see [LICENSE](LICENSE)).

---

**Keywords:** radar signal generator, radar dataset generator, LPI radar waveforms, radar waveform dataset, LFM, Costas code, Barker code, BPSK, Frank code, P1 P2 P3 P4 polyphase codes, T1 T2 T3 T4 polytime codes, I/Q signals, AWGN, echo, co-channel interference, radar signal restoration, radar signal denoising, blind radar signal restoration, BRSR dataset, BRSR-OpGAN, CoRe-Net, XCoRe-Net, MATLAB, Python.
