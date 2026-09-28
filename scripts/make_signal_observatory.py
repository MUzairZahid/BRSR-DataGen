"""Generate the reproducible Signal Observatory sample library.

Requires only NumPy. Run: python scripts/make_signal_observatory.py
The embedded records are generated here, not invented in the browser. Browser
controls rescale these fixed realizations using the generator's power rule.
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'python' / 'brsr_datagen' / f'{name}.py')
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def build_data():
    w, a = module('waveforms'), module('artifacts')
    fs, n = 100e6, 2048
    hops = [1, 2, 4, 3]
    assert all(len(set(hops[i+d]-hops[i] for i in range(len(hops)-d))) == len(hops)-d
               for d in range(1, len(hops)))
    examples = {
        'LFM': (113, 'Linear frequency modulation', 'Rising chirp · legacy benchmark convention',
                lambda rng: w.type_LFM(n, fs, 1, 18e6, 5.6e6, 'Up', rng, legacy=True)),
        'Costas': (113, 'Costas frequency hopping', 'Validated hop sequence 1–2–4–3',
                   lambda rng: w.type_Costas(n, fs, 1, 3.7e6, hops, rng)),
        'BPSK': (107, 'Binary phase-shift keying', 'Barker-13 · linearly resampled',
                 lambda rng: w.resample_linear(w.type_Barker(20, fs, 1, 8.8e6, 13), n)),
    }
    # Use the same waveform functions and resampling as generator.make_waveform.
    # Parameters are explicit, within the benchmark ranges, and recorded below.
    for name in ('Frank', 'P1', 'P2', 'P3', 'P4'):
        count = 36 if name in ('P3', 'P4') else 6
        examples[name] = (211 + len(examples), name + ' polyphase code',
                          f'{count} ' + ('subcodes' if count == 36 else 'steps') + ' · 4 cycles per chip · 18 MHz carrier',
                          lambda rng, name=name, count=count: w.resample_linear(
                              getattr(w, 'type_' + name)(4, fs, 1, 18e6, count), n))
    for name in ('T1', 'T2'):
        examples[name] = (311 + len(examples), name + ' time code',
                          '5 segments · 2 phase states · 18 MHz carrier',
                          lambda rng, name=name: w.resample_linear(
                              getattr(w, 'type_' + name)(fs, 1, 18e6, 2, 5), n))
    for name in ('T3', 'T4'):
        examples[name] = (411 + len(examples), name + ' time code',
                          '2 phase states · 8.8 MHz carrier · 7.5 MHz bandwidth',
                          lambda rng, name=name: getattr(w, 'type_' + name)(n, fs, 1, 8.8e6, 2, 7.5e6))
    records = {}
    bank = a.load_interference_bank()
    pack = lambda z: [np.round(z.real, 8).tolist(), np.round(z.imag, 8).tolist()]
    for name, (base_seed, title, subtitle, make) in examples.items():
        records[name] = []
        # Three independent seeded observations of each waveform configuration.
        for sample in range(3):
            seed = base_seed + sample * 1000
            rng = np.random.default_rng(seed)
            clean, noisy, components, info = a.add_artifacts(make(rng), -3, bank, rng,
                                                           composition=('AWGN', 'Echo', 'CCI'))
            scale = np.sqrt(np.mean(np.abs(clean)**2))
            assert np.allclose(noisy, clean + components.sum(axis=0))
            power = np.mean(np.abs(components)**2, axis=1)
            weights = np.array([info['w_awgn'], info['w_echo'], info['w_cci']])
            assert np.allclose(power / scale**2, weights * 10**.3)
            records[name].append(dict(title=title, subtitle=subtitle, seed=seed, clean=pack(clean/scale),
                                 noise=pack(components[0]/scale), echo=pack(components[1]/scale),
                                 cci=pack(components[2]/scale), weights=weights.tolist(),
                                 offset=info['echo_delay'], start=info['segment_start'],
                                 bankRow=info['cci_bank_row'], originalRms=float(scale)))
    return dict(fs=fs, baseSnr=-3, bankSize=len(bank), records=records)


def build():
    # Preserve the established command while using the current public renderer.
    import runpy
    runpy.run_path(str(ROOT / 'scripts' / 'make_observatory_v2.py'), run_name='__main__')


if __name__ == '__main__':
    build()
