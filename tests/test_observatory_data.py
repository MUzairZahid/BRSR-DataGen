"""Verify the sample library and complex mixing functions."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location('observatory', ROOT / 'scripts/make_signal_observatory.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gallery_data():
    return builder().build_data()


def test_sample_library_is_reproducible():
    data = gallery_data()
    assert data == builder().build_data()
    assert list(data['records']) == builder().module('waveforms').CLASS_NAMES
    for samples in data['records'].values():
        assert len(samples) == 3
        assert len({r['seed'] for r in samples}) == 3
        assert samples[0]['noise'] != samples[1]['noise'] != samples[2]['noise']
        for d in samples:
            for key in ('clean', 'noise', 'echo', 'cci'):
                values = np.asarray(d[key])
                assert values.shape == (2, 1024)
                assert np.isfinite(values).all()
            clean_power = np.mean(np.sum(np.asarray(d['clean']) ** 2, axis=0))
            np.testing.assert_allclose(clean_power, 1, atol=1e-8)
            for key, weight in zip(('noise', 'echo', 'cci'), d['weights']):
                power = np.mean(np.sum(np.asarray(d[key]) ** 2, axis=0))
                np.testing.assert_allclose(power / clean_power, weight * 10 ** .3, rtol=1e-6)


def test_browser_mixtures_match_numpy(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required for browser math regression')
    data = gallery_data()
    source = (ROOT / 'scripts/signal_observatory_template.html').read_text(encoding='utf-8')
    math_functions = source[source.index('function mix(){'):source.index('function canvasSetup(')]
    cases = []
    for name, samples in data['records'].items():
        for sample, d in enumerate(samples):
            clean = np.asarray(d['clean'])
            for snr in (-14, -3, 10):
                for mask in range(8):
                    enabled = [bool(mask & (1 << k)) for k in range(3)]
                    z = clean.copy()
                    for on, key in zip(enabled, ('echo', 'cci', 'noise')):
                        if on:
                            z += 10 ** ((-3 - snr) / 20) * np.asarray(d[key])
                    ratio = np.sum((z - clean) ** 2) / np.sum(clean ** 2)
                    cases.append(dict(wf=name, sample=sample, snr=snr, enabled=enabled,
                                      expected=z.tolist(), ratio=float(ratio)))
    payload = tmp_path / 'cases.json'
    payload.write_text(json.dumps(dict(data=data, cases=cases)), encoding='utf-8')
    script = tmp_path / 'verify.cjs'
    script.write_text("""
const assert=require('node:assert/strict');
const payload=JSON.parse(require('node:fs').readFileSync(process.argv[2],'utf8'));
const DATA=payload.data, keys=['echo','cci','noise'];
let state;
const record=()=>DATA.records[state.wf][state.sample];
""" + math_functions + """
const original=JSON.stringify(DATA);
for(const c of payload.cases){
  state=c; const z=mix(), m=measured(z);
  for(let ch=0;ch<2;ch++)for(let n=0;n<1024;n++)
    assert.ok(Math.abs(z[ch][n]-c.expected[ch][n])<1e-11);
  assert.ok(Math.abs(m.ratio-c.ratio)<1e-10);
  if(c.ratio===0)assert.equal(m.snr,Infinity);
  else assert.ok(Math.abs(m.snr+10*Math.log10(c.ratio))<1e-10);
}
assert.equal(JSON.stringify(DATA),original,'exploring mixtures never mutates the clean reference');
console.log('Verified '+payload.cases.length+' complex mixtures and immutable source data');
""", encoding='utf-8')
    subprocess.run([node, str(script), str(payload)], check=True, timeout=30)
