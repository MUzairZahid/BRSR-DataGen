"""Verify the Signal Observatory page: embedded samples, the browser's own maths and playback."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def builder():
    return load('observatory', ROOT / 'scripts/make_signal_observatory.py')


def page_source():
    return (ROOT / 'docs/signal_observatory.html').read_text(encoding='utf-8')


def embedded_raw():
    return json.loads(page_source().split('const RAW = ', 1)[1].split(';\n', 1)[0])


def pure_math(source):
    """The contiguous block of data/maths functions the page runs (decode, mix, measured, fft, spectrum)."""
    return source[source.index('function decodeComponent('):source.index('/* ---- end pure math ---- */')]


def test_embedded_samples_match_generator():
    b = builder()
    raw = embedded_raw()
    assert raw == b.build_data(), 'page must be rebuilt with scripts/make_signal_observatory.py'
    original = b.observatory.build_data()
    assert list(raw['records']) == list(original['records']) == b.observatory.module('waveforms').CLASS_NAMES
    worst = 0.0
    for name, samples in raw['records'].items():
        assert len(samples) == 3
        for d, ref in zip(samples, original['records'][name]):
            for key in ('seed', 'weights', 'offset', 'start', 'bankRow', 'title', 'subtitle'):
                assert d[key] == ref[key]
            for key in ('clean', 'noise', 'echo', 'cci'):
                decoded = b.decode_component(d[key])
                expected = np.asarray(ref[key])
                assert decoded.shape == (2, 1024) and np.isfinite(decoded).all()
                worst = max(worst, float(np.max(np.abs(decoded - expected)) / d[key]['scale']))
    assert worst < 3.1e-5
    assert raw['maxQuantError'] == pytest.approx(worst)


def test_page_is_self_contained_and_small():
    html = page_source()
    assert 'data:font/woff2;base64,' in html
    assert '__SIGNAL_DATA__' not in html and '__THEME_CSS__' not in html and '__FONT_CSS__' not in html
    assert 'href="http' not in html.split('<style>')[0].split('<link rel="canonical"')[0].replace('content="http', '')
    assert len(html.encode('utf-8')) < 1.3e6
    # the palette in the page is the palette in brsr_palette.py
    palette = load('brsr_palette', ROOT / 'scripts/brsr_palette.py')
    for theme in palette.THEMES.values():
        for key in ('clean', 'echo', 'cci', 'awgn', 'received', 'surface', 'ink'):
            assert theme[key] in html


def test_browser_decode_and_mixtures_match_numpy(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required for browser math regression')
    b = builder()
    raw = embedded_raw()
    cases = []
    for name, samples in raw['records'].items():
        for sample, d in enumerate(samples):
            comps = {k: b.decode_component(d[k]) for k in ('clean', 'echo', 'cci', 'noise')}
            for snr in (-14, -3, 10):
                for mask in (0, 1, 2, 4, 7):
                    enabled = [bool(mask & (1 << k)) for k in range(3)]
                    z = comps['clean'].copy()
                    for on, key in zip(enabled, ('echo', 'cci', 'noise')):
                        if on:
                            z += 10 ** ((-3 - snr) / 20) * comps[key]
                    ratio = np.sum((z - comps['clean']) ** 2) / np.sum(comps['clean'] ** 2)
                    cases.append(dict(wf=name, sample=sample, snr=snr, enabled=enabled,
                                      expected=z.tolist(), ratio=float(ratio)))
    payload = tmp_path / 'cases.json'
    payload.write_text(json.dumps(dict(raw=raw, cases=cases)), encoding='utf-8')
    script = tmp_path / 'verify.cjs'
    script.write_text("""
const assert=require('node:assert/strict');
const payload=JSON.parse(require('node:fs').readFileSync(process.argv[2],'utf8'));
let state;
const record=()=>DATA.records[state.wf][state.sample];
""" + pure_math(page_source()) + """
const DATA=decodeAll(payload.raw);
const original=JSON.stringify(DATA);
for(const c of payload.cases){
  state=c; const z=mix(), m=measured(z);
  for(let ch=0;ch<2;ch++)for(let n=0;n<1024;n++)
    assert.ok(Math.abs(z[ch][n]-c.expected[ch][n])<1e-9);
  assert.ok(Math.abs(m.ratio-c.ratio)<1e-9);
  if(c.ratio===0)assert.equal(m.snr,Infinity);
  else assert.ok(Math.abs(m.snr+10*Math.log10(c.ratio))<1e-9);
}
// the STFT has 241 frames of 64 bins and a Hann-windowed single tone peaks at its bin
const S=spectrum(DATA.records.LFM[0].clean);
assert.equal(S.length,241);assert.equal(S[0].length,64);
const tone=[Array.from({length:1024},(_,n)=>Math.cos(2*Math.PI*8*n/64)),Array.from({length:1024},(_,n)=>Math.sin(2*Math.PI*8*n/64))];
const T=spectrum(tone)[100];assert.equal(T.indexOf(Math.max(...T)),8);
assert.equal(JSON.stringify(DATA),original,'exploring mixtures never mutates the decoded data');
console.log('Verified '+payload.cases.length+' mixtures, STFT layout and immutable source data');
""", encoding='utf-8')
    subprocess.run([node, str(script), str(payload)], check=True, timeout=60)


def test_observatory_playback():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js is required for the JavaScript timeline regression')
    source = (ROOT / 'scripts' / 'signal_observatory_template.html').read_text(encoding='utf-8')
    stage = source.split('function setStage(stage){', 1)[1].split('\n', 1)[0]
    frame = source.split('function frame(now){', 1)[1].split('\nif(reduced.matches)', 1)[0]
    functions = 'function setStage(stage){' + stage + '\nfunction frame(now){' + frame
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const context = {
  state: {playing:true, loop:true, t:0, stage:0, enabled:[false,false,false]},
  last:0, reduced:{matches:true}, document:{hidden:false}, sceneVisible:true,
  update(){}, progress(){}, setPlotLimit(){}, requestAnimationFrame(){}
};
vm.createContext(context);
vm.runInContext(FUNCTIONS, context);
let now = 0;
const tick = count => {for(let i=0;i<count;i++) context.frame(now += 100);};
tick(40);
assert.equal(context.state.stage, 1, 'echo is revealed after clean');
assert.equal(JSON.stringify(context.state.enabled), '[true,false,false]');
tick(110);
assert.equal(context.state.playing, true, 'default playback continues after 14 seconds');
assert.equal(context.state.stage, 0, 'a completed sequence starts again');
context.state.playing = false;
const paused = context.state.t;
tick(40);
assert.equal(context.state.t, paused, 'manual pause freezes the clock');
context.state.playing = true;
context.document.hidden = true;
tick(40);
assert.equal(context.state.t, paused, 'background tab must not consume the reveal');
context.document.hidden = false;
context.sceneVisible = false;
tick(10);
assert.ok(context.state.t > paused, 'plots keep updating when the scene scrolls out of view');
context.state.loop = false;
tick(150);
assert.equal(context.state.playing, false, 'single pass stops');
assert.equal(context.state.stage, 3);
assert.equal(JSON.stringify(context.state.enabled), '[true,true,true]');
assert.equal(context.state.t, 14);
console.log('PASS: loop, stage timing, pause, background visibility, scrolled plots and single pass');
""".replace('FUNCTIONS', json.dumps(functions))
    subprocess.run([node, '-e', harness], check=True, timeout=15)


def test_component_bounds_and_atomic_sample_switch():
    """Cancellation must not hide an overlay; a fade must not mix two records."""
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required for chart state regression')
    source = (ROOT / 'scripts/signal_observatory_template.html').read_text(encoding='utf-8')
    bounds = source[source.index('function setPlotLimit(){'):source.index('function drawTime(){')]
    nice = source[source.index('function niceStep('):source.index('const M=')]
    switch = source[source.index('function setWaveform(name,animate){'):source.index('/* ---- table & CSV ---- */')]
    script = """
const assert=require('node:assert/strict');
const values=n=>[Array(1024).fill(n),Array(1024).fill(n)];
const first={title:'first',seed:1,clean:values(1),echo:values(12),cci:values(0),noise:values(-12),
  weights:[.5,.5,0],subtitle:'first',offset:128,start:0,bankRow:1};
const next={...first,title:'second',seed:2,clean:values(3)};
const DATA={records:{LFM:[first],P1:[next]},bankSize:50};
const state={wf:'LFM',sample:0,channel:0,range:[0,16],enabled:[true,true,true]};
const keys=['echo','cci','noise'],record=()=>DATA.records[state.wf][state.sample],gain=()=>1;
let plotLimit,cleanSpec,cleanPeak,fadeTimer,currentSeed=1;
const $=()=>({classList:{add(){},remove(){}}}),$$=()=>[];
const reduced={matches:false},spectrum=()=>[[1]],clearTimeout=()=>{},setTimeout=()=>1;
const update=()=>{currentSeed=record().seed;};
""" + nice + bounds + switch + """
setPlotLimit();
assert.ok(plotLimit>=12,'components that cancel still fit within the plotted axes');
setWaveform('P1',true);
assert.equal(state.wf,'P1');
assert.equal(currentSeed,2,'new waveform and mixture must commit before the fade timer');
console.log('PASS: cancellation-safe bounds and atomic sample changes');
"""
    subprocess.run([node, '-e', script], check=True, timeout=15)


def test_public_entry_points_and_navigation():
    """The public entry point opens the current page and omits retired demos."""
    html = page_source()
    from html.parser import HTMLParser

    class Links(HTMLParser):
        hrefs = []

        def handle_starttag(self, tag, attrs):
            if tag == 'a':
                self.hrefs.append(dict(attrs).get('href', ''))

    links = Links()
    links.feed(html)
    assert links.hrefs.count('https://github.com/MUzairZahid/BRSR-DataGen') == 1
    assert 'radar_environment.html' not in links.hrefs
    assert 'signal_observatory.html' not in links.hrefs
    assert 'url=signal_observatory.html' in (ROOT / 'docs/index.html').read_text(encoding='utf-8')
    assert (ROOT / 'docs/signal_observatory.html').exists()
    assert not (ROOT / 'docs/radar_environment.html').exists()
    readme = (ROOT / 'README.md').read_text(encoding='utf-8')
    assert 'radar_environment.gif' not in readme
    assert 'radar_environment.html' not in readme


def test_csv_export_is_consistent_and_documents_precision(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js required for CSV regression')
    source = (ROOT / 'scripts/signal_observatory_template.html').read_text(encoding='utf-8')
    rows = source[source.index('function rows(){'):source.index('function renderTable(){')]
    export = source[source.index("$('downloadCsv').addEventListener"):source.index('/* ---- hover, cursor, brush ---- */')]
    payload = tmp_path / 'raw.json'
    payload.write_text(json.dumps(embedded_raw()), encoding='utf-8')
    script = tmp_path / 'csv.cjs'
    script.write_text("""
const assert=require('node:assert/strict');
const RAW=JSON.parse(require('node:fs').readFileSync(process.argv[2],'utf8'));
const state={wf:'LFM',sample:0,snr:-3,enabled:[true,false,true]};
const record=()=>DATA.records[state.wf][state.sample];
""" + pure_math(page_source()) + """
const DATA=decodeAll(RAW),current=mix();
let handler,downloadName,csv;
const $=()=>({addEventListener:(_,f)=>{handler=f;}});
const Blob=class {constructor(parts){csv=parts.join('');}};
const URL={createObjectURL:()=>'',revokeObjectURL(){}};
const setTimeout=()=>{};
const document={body:{appendChild(){}},createElement:()=>({click(){downloadName=this.download;},remove(){}})};
""" + rows + export + r"""
handler();
assert.match(downloadName,/brsr_LFM_seed113_snr-3dB.csv/);
assert.match(csv,/# encoding: int16-le-base64/);
assert.match(csv,/max quantisation error\/component peak=/);
const lines=csv.split('\n').filter(l=>!l.startsWith('#'));
assert.equal(lines.length,1025);
for(const line of lines.slice(1)){
  const a=line.split(',').map(Number);assert.equal(a.length,12);
  assert.ok(a.every(Number.isFinite));
  assert.equal(a[6],0);assert.equal(a[7],0);
  for(const c of [0,1])assert.ok(Math.abs(a[2+c]+a[4+c]+a[6+c]+a[8+c]-a[10+c])<1e-6);
}
console.log('PASS: CSV metadata, 1,024 rows, muted components and additive reconstruction');
""", encoding='utf-8')
    subprocess.run([node, str(script), str(payload)], check=True, timeout=15)
