"""Exercise the actual browser timeline functions without a browser or wall-clock waits."""
import json
from pathlib import Path
import shutil
import subprocess

import pytest


def test_observatory_playback():
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js is required for the JavaScript timeline regression")
    source = (Path(__file__).resolve().parents[1] / "scripts" /
              "signal_observatory_template.html").read_text(encoding="utf-8")
    stage = source.split("function setStage(stage){", 1)[1].split("\n", 1)[0]
    frame = source.split("function frame(now){", 1)[1].split("\nif(reduced.matches)", 1)[0]
    functions = "function setStage(stage){" + stage + "\nfunction frame(now){" + frame
    harness = r"""
const assert = require('node:assert/strict');
const vm = require('node:vm');
const context = {
  state: {playing:true, loop:true, t:0, stage:0, enabled:[false,false,false]},
  last:0, reduced:{matches:true}, document:{hidden:false},
  update(){}, progress(){}, requestAnimationFrame(){}
};
vm.createContext(context);
vm.runInContext(FUNCTIONS, context);
let now = 0;
const tick = count => {for(let i=0;i<count;i++) context.frame(now += 100);};
tick(40);
assert.equal(context.state.stage, 1, 'echo is revealed after clean');
tick(110);
assert.equal(context.state.playing, true, 'default playback must continue after 14 seconds');
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
assert.ok(context.state.t > paused, 'plots keep updating when the diagram scrolls out of view');
context.state.loop = false;
tick(150);
assert.equal(context.state.playing, false, 'single pass stops');
assert.equal(context.state.stage, 3);
assert.equal(context.state.t, 14);
console.log('PASS: loop, stage timing, pause, background visibility, scrolled plots and single pass');
""".replace("FUNCTIONS", json.dumps(functions))
    subprocess.run([node, "-e", harness], check=True, timeout=15)
