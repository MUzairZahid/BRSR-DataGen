"""Build the separate Signal Observatory without modifying the original animation.

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


def build():
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
    records = {}
    bank = a.load_interference_bank()
    pack = lambda z: [np.round(z.real, 8).tolist(), np.round(z.imag, 8).tolist()]
    for name, (seed, title, subtitle, make) in examples.items():
        rng = np.random.default_rng(seed)
        clean, noisy, components, info = a.add_artifacts(make(rng), -3, bank, rng,
                                                       composition=('AWGN', 'Echo', 'CCI'))
        scale = np.sqrt(np.mean(np.abs(clean)**2))
        assert np.allclose(noisy, clean + components.sum(axis=0))
        power = np.mean(np.abs(components)**2, axis=1)
        weights = np.array([info['w_awgn'], info['w_echo'], info['w_cci']])
        assert np.allclose(power / scale**2, weights * 10**.3)
        records[name] = dict(title=title, subtitle=subtitle, seed=seed, clean=pack(clean/scale),
                             noise=pack(components[0]/scale), echo=pack(components[1]/scale),
                             cci=pack(components[2]/scale), weights=weights.tolist(),
                             offset=info['echo_delay'], start=info['segment_start'],
                             bankRow=info['cci_bank_row'], originalRms=float(scale))
        print(f'{name}: verified 1024 complex samples, seed {seed}, offset {info["echo_delay"]}')
    data = dict(fs=fs, baseSnr=-3, bankSize=len(bank), records=records)
    template = (ROOT / 'scripts' / 'signal_observatory_template.html').read_text(encoding='utf-8')
    html = template.replace('__SIGNAL_DATA__', json.dumps(data, separators=(',', ':')))
    assert '__SIGNAL_DATA__' not in html
    out = ROOT / 'docs' / 'signal_observatory.html'
    out.write_text(html, encoding='utf-8', newline='\n')
    print(f'Built {out} ({len(html):,} characters). Original animation untouched.')


if __name__ == '__main__':
    build()
