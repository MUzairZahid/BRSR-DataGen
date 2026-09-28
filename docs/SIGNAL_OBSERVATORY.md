# Signal Observatory

An independent redesign of the radar-environment animation. Open
[`signal_observatory.html`](signal_observatory.html) in a browser. It is a
self-contained page and also works without a server or an internet connection.

The original `radar_environment.html`, its builder, template and GIF are preserved.
The new page links back to the original. The README keeps the original in an expandable section.

## Explore

- Choose LFM, Costas or Barker BPSK.
- Play the four-stage reveal, pause it, or select a completed stage directly.
- Adjust the full-blend target SNR from -14 to +10 dB.
- Toggle echo, interference and AWGN independently.
- Inspect the true summed I-channel waveform, complex STFT, measured SNR and
  artifact-to-clean power ratio. The plots change together without image crossfades.
- Expand the provenance panel for seeds, segment indices and benchmark conventions.

The mobile layout stacks component cards and charts. Buttons and the range control
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
- The signal-flow particles illustrate summation, not signal propagation time.

The separate restoration demonstration follows the published evaluation protocol;
its physical-scale restored output uses clean-reference channel extrema. It is
linked for context and is not changed by this redesign.
