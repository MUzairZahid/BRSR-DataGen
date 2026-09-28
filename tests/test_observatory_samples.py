"""Verify the sample library and artifact power."""
import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def builder():
    spec = importlib.util.spec_from_file_location('observatory', ROOT / 'scripts/observatory_samples.py')
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
