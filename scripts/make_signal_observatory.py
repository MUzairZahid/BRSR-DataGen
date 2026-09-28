"""Build the Signal Observatory page (docs/signal_observatory.html).

    python scripts/make_signal_observatory.py

Builds 36 seeded observations from the repository's waveform and artifact code.
The page stores quantised components with a documented error bound and embeds
its fonts for offline use. Colours and line styles come from brsr_palette.py.

Requires only NumPy.
"""
import base64
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INT16_MAX = 32767


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


observatory = load('observatory_samples', ROOT / 'scripts' / 'observatory_samples.py')
palette = load('brsr_palette', ROOT / 'scripts' / 'brsr_palette.py')


def encode_component(values):
    """[2, 1024] float list -> {'scale': float, 'b64': str} holding int16 little-endian samples (I then Q)."""
    arr = np.asarray(values, dtype=np.float64)
    assert arr.shape == (2, 1024)
    scale = float(np.max(np.abs(arr))) or 1.0
    q = np.round(arr / scale * INT16_MAX).astype('<i2')
    return {'scale': scale, 'b64': base64.b64encode(q.tobytes()).decode('ascii')}


def decode_component(packed):
    """Inverse of encode_component (used by the tests); returns a [2, 1024] float64 array."""
    q = np.frombuffer(base64.b64decode(packed['b64']), dtype='<i2').reshape(2, 1024)
    return q.astype(np.float64) / INT16_MAX * packed['scale']


def build_data():
    """Same records as the original page; signal arrays replaced by compact encodings."""
    data = observatory.build_data()
    out = dict(fs=data['fs'], baseSnr=data['baseSnr'], bankSize=data['bankSize'],
               encoding='int16-le-base64, value = int / 32767 * scale, layout [I(1024), Q(1024)]',
               records={})
    worst = 0.0
    for name, samples in data['records'].items():
        out['records'][name] = []
        for d in samples:
            rec = {k: v for k, v in d.items() if k not in ('clean', 'noise', 'echo', 'cci')}
            for key in ('clean', 'noise', 'echo', 'cci'):
                rec[key] = encode_component(d[key])
                err = np.max(np.abs(decode_component(rec[key]) - np.asarray(d[key]))) / rec[key]['scale']
                worst = max(worst, float(err))
            # Composition summary used by the page and the CSV export.
            w = d['weights']  # [awgn, echo, cci]
            rec['composition'] = 'AWGN + Echo + CCI'
            rec['weightsByName'] = {'awgn': w[0], 'echo': w[1], 'cci': w[2]}
            out['records'][name].append(rec)
    out['maxQuantError'] = worst
    assert worst < 3.1e-5, worst
    return out


def inline_font(filename):
    return base64.b64encode((ROOT / 'docs' / 'fonts' / filename).read_bytes()).decode('ascii')


def build():
    data = build_data()
    n = sum(map(len, data['records'].values()))
    print(f'Verified {n} observations across {len(data["records"])} classes; '
          f'max quantisation error {data["maxQuantError"]:.2e} of component peak.')
    template = (ROOT / 'scripts' / 'signal_observatory_template.html').read_text(encoding='utf-8')
    fonts = (
        "@font-face{font-family:'Inter';font-style:normal;font-weight:100 900;font-display:swap;"
        "src:url(data:font/woff2;base64," + inline_font('inter-latin-wght-normal.woff2') + ") format('woff2')}\n"
        "@font-face{font-family:'JetBrains Mono';font-style:normal;font-weight:100 800;font-display:swap;"
        "src:url(data:font/woff2;base64," + inline_font('jetbrains-mono-latin-wght-normal.woff2') + ") format('woff2')}\n"
    )
    html = (template
            .replace('__THEME_CSS__', palette.css_tokens())
            .replace('__FONT_CSS__', fonts)
            .replace('__COMPONENT_DASHES__', json.dumps({name: palette.COMPONENT_DASHES[key] for name, key in [('Clean','clean'),('Echo','echo'),('Cci','cci'),('Noise','awgn')]}))
            .replace('__SIGNAL_DATA__', json.dumps(data, separators=(',', ':'))))
    for marker in ('__THEME_CSS__', '__FONT_CSS__', '__SIGNAL_DATA__', '__COMPONENT_DASHES__'):
        assert marker not in html, marker
    out = ROOT / 'docs' / 'signal_observatory.html'
    out.write_text(html, encoding='utf-8', newline='\n')
    print(f'Built {out} ({len(html.encode("utf-8")):,} bytes). Ready for static hosting.')


if __name__ == '__main__':
    build()
