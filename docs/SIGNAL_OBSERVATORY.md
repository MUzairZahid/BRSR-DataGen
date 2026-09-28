# Signal Observatory

An independent redesign of the radar-environment animation. Open
[`signal_observatory.html`](signal_observatory.html) in a browser. It is a
self-contained page and also works without a server or an internet connection.

The original `radar_environment.html`, its builder, template and GIF are preserved.
The new page links back to the original. The README keeps the original in an expandable section.

## Explore

- Choose any of the 12 classes: LFM, Costas, Barker BPSK, Frank, P1–P4 or T1–T4.
- Cycle through three independently seeded observations per class with Next sample.
  Waveform parameters remain fixed within a class; source slices and artifact draws vary.
- Select the transmitter, reflector, interferer or noise source to highlight its path
  and overlay its actual contribution on the received waveform. The Overlay buttons
  beside the plots select the same source. Inspecting a source does not enable it.
- Play the four-stage reveal, pause it, or select a completed stage directly.
- Adjust the full-blend target SNR from -14 to +10 dB.
- Toggle echo, interference and AWGN independently.
- Compare clean and received waveforms and complex STFTs side by side. The clean
  reference stays visible; both panels use matching amplitude and colour scales.
- Choose I or Q and the first 160 or all 1,024 samples for the waveform plots.
  Spectrograms always cover the complete complex observation.
- Read measured SNR and artifact-to-clean power ratio for the actual mixture.
- Expand the provenance panel for seeds, segment indices and benchmark conventions.

The mobile layout preserves the environment illustration and stacks the comparison cards. Buttons and the range control
support keyboard navigation. Reduced-motion preference starts with a paused final
sample with looping disabled. Otherwise playback begins when the diagram enters view and repeats; use Loop off for a single pass. Any stage, SNR or component adjustment pauses the guided reveal. A visible startup notice remains if scripts cannot initialize.

## Build

From the repository root, with NumPy installed:

```sh
python scripts/make_signal_observatory.py
```

Edit `scripts/signal_observatory_template.html` for layout or interaction changes.
The builder imports the repository's waveform and artifact implementations directly
and verifies their additive identity and power allocations before embedding data.

## Scientific conventions

- The browser receives all 1,024 complex samples of the clean waveform and each
  component, rounded to eight decimal places after common clean-RMS normalization.
- The examples are newly seeded generator realizations, not published test-set rows.
- LFM retains the generator's legacy phase/amplitude convention. Costas uses a
  validated `[1, 2, 4, 3]` permutation. Barker BPSK uses the generator's resampling.
- Echo is explicitly described as the benchmark's positive source-segment offset,
  not a causal physical reflection. No propagation geometry is simulated.
- Slider gain is `10 ** ((base_snr - target_snr) / 20)` for the existing artifact
  components. Switching a component off does not redistribute its power.
- Reported SNR uses the actual combined complex error, including cross-terms.
- The STFT uses a 64-point symmetric Hann window, 8-sample hop and complex FFT.
  Its 121 frame centres span 0.315 to 9.915 microseconds; frequency-bin centres
  span -50 to +48.4375 MHz. Colours are fixed from -40 to 0 dB relative to the
  clean STFT peak for the selected waveform, with out-of-range values clipped.
- The environment paths and particles illustrate component summation, not signal propagation time.
  The building is a visual metaphor, not a simulated reflector. Path lengths do not
  determine the benchmark echo offset, gains or arrival times.
- Each stored sample is generated at -3 dB with all three artifacts. The slider
  and switches explore that realization; Next sample loads a fresh stored realization.
- The 36 observations are embedded in the self-contained HTML (about 3.4 MB).

The separate restoration demonstration follows the published evaluation protocol;
its physical-scale restored output uses clean-reference channel extrema. It is
linked for context and is not changed by this redesign.
