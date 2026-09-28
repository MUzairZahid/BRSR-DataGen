# Legacy scripts (BRSR v1.0)

These are the original scripts used to create the published BRSR benchmark
([Zenodo, DOI 10.5281/zenodo.23010395](https://doi.org/10.5281/zenodo.23010395)), kept unchanged for reference.

- `DataGeneration_Extended.m`: BRSR (blind) dataset: 12 waveforms, random blend of AWGN, echo and CCI (`add_distortion.m`).
- `DataGeneration_Baseline.m`: AWGN-Baseline dataset (uses `awgn` from the Communications Toolbox).
- `DataPreparation_*.py`: MATLAB output to train/validation/test pickles (80/20 stratified split, `random_state=42`).
- `plot_signals_*.m`: plotting helpers.

The scripts were not seeded, so they do not reproduce the published files. To run them, add the waveform
functions and the interference bank to the path first:

```matlab
addpath('../matlab/waveforms');
copyfile('../matlab/interference_bank.mat', 'mixing_signals.mat');   % the scripts load mixing_signals.mat
```

For new data, use the maintained generator instead: `matlab/generate_brsr_dataset.m` or the Python package.
