'use strict';
// Logic checks for the replay viewer, run in Node against a built replay page.
// No browser: the page's script runs in a bare vm context (no document), so the
// viewer exposes its pure helpers on window.__replayLogic and stops before the DOM.
//
// Usage: node scripts/replay/test_viewer_logic.js [built_page.html]
// Default page: notes/replay/replay_sample.html (build it with build_replay_page.py first).

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const ROOT = path.resolve(__dirname, '..', '..');
const pagePath = process.argv[2] || path.join(ROOT, 'notes', 'replay', 'replay_sample.html');
const html = fs.readFileSync(pagePath, 'utf8');
const blocks = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map((m) => m[1]);
assert.strictEqual(blocks.length, 2, 'expected the data block and the viewer block');

const sandbox = { window: {}, console };
const ctx = vm.createContext(sandbox);
blocks.forEach((src, i) => new vm.Script(src, { filename: `page-block-${i}.js` }).runInContext(ctx));
const L = sandbox.window.__replayLogic;
const REPLAY = sandbox.window.REPLAY;
assert.ok(L, 'viewer did not expose __replayLogic (is the DOM guard present?)');
assert.ok(REPLAY, 'data block did not set window.REPLAY');

const clone = (o) => JSON.parse(JSON.stringify(o));
const close = (a, b, tol = 1e-9) => Math.abs(a - b) <= tol;

const tests = [];
function test(name, fn) { tests.push({ name, fn }); }

test('sample passes checkReplay', () => {
  assert.strictEqual(L.checkReplay(REPLAY), null);
});

test('checkReplay reports the first problem in plain words', () => {
  const v2 = clone(REPLAY); v2.version = 2;
  assert.match(L.checkReplay(v2), /Unsupported replay version/);
  assert.match(L.checkReplay(undefined), /No replay data/);
  const short = clone(REPLAY); short.rounds[0].frames.p[3].hp.pop();
  assert.match(L.checkReplay(short), /player 3 field "hp" must have n=/);
  const nine = clone(REPLAY); nine.rounds[0].players.pop();
  assert.match(L.checkReplay(nine), /expected 10 players/);
});

test('prepareRounds reads winners, kills, bomb events and running score', () => {
  const rds = L.prepareRounds(REPLAY);
  assert.strictEqual(rds.length, 2);
  assert.deepStrictEqual([rds[0].winner, rds[1].winner], ['CT', 'T']);
  assert.strictEqual(rds[0].ctBefore, 0);
  assert.strictEqual(rds[1].ctBefore, 1);
  assert.strictEqual(rds[1].tBefore, 0);
  assert.strictEqual(rds[0].kills.length, 5);
  assert.strictEqual(rds[1].kills.length, 5);
  assert.strictEqual(rds[0].plant.t, 41);
  assert.strictEqual(rds[0].plant.player, 6);
  assert.strictEqual(rds[0].defuse.t, 76.5);
  assert.strictEqual(rds[0].duration, 76.5);
  assert.strictEqual(rds[1].explode.t, 92);
  assert.strictEqual(rds[1].duration, 92);
  assert.strictEqual(rds[0].marks.length, 7);  // 5 kills + plant + defuse
  assert.strictEqual(rds[1].marks.length, 7);  // 5 kills + plant + explode
});

test('bomb marker sits where the planter stood at the plant frame', () => {
  const rd = L.prepareRounds(REPLAY)[0];
  const i = Math.round(41 / REPLAY.rounds[0].frames.dt);  // frame 328
  const p = REPLAY.rounds[0].frames.p[6];
  assert.strictEqual(rd.bombPos.x, p.x[i]);
  assert.strictEqual(rd.bombPos.y, p.y[i]);
});

test('probAt interpolates the 1 Hz series and returns null when absent', () => {
  const rd = L.prepareRounds(REPLAY)[0];
  const P = REPLAY.rounds[0].prob.p;
  assert.ok(close(L.probAt(rd, 0), P[0]));
  assert.ok(close(L.probAt(rd, 10.5), (P[10] + P[11]) / 2, 1e-9));
  assert.ok(close(L.probAt(rd, 1e9), P[P.length - 1]));

  const none = clone(REPLAY); none.rounds[0].prob = null;
  assert.strictEqual(L.probAt(L.prepareRounds(none)[0], 5), null);

  const gap = clone(REPLAY); gap.rounds[0].prob.p[20] = null;
  const rg = L.prepareRounds(gap)[0];
  assert.strictEqual(L.probAt(rg, 20), null);   // touches the null sample
  assert.strictEqual(L.probAt(rg, 19.5), null); // interval ends on the null sample
  assert.notStrictEqual(L.probAt(rg, 18.5), null);
  assert.notStrictEqual(L.probAt(rg, 5), null);
});

test('bombState follows plant, defuse and explosion', () => {
  const rds = L.prepareRounds(REPLAY);
  assert.strictEqual(L.bombState(rds[0], 40.9).s, 'none');
  let bs = L.bombState(rds[0], 41);
  assert.strictEqual(bs.s, 'planted');
  assert.ok(close(bs.left, 40));
  bs = L.bombState(rds[0], 60);
  assert.ok(close(bs.left, 21));
  assert.strictEqual(L.bombState(rds[0], 76.5).s, 'defused');
  assert.strictEqual(L.bombState(rds[1], 52).s, 'planted');
  assert.strictEqual(L.bombState(rds[1], 62).s, 'planted');
  assert.ok(close(L.bombState(rds[1], 62).left, 30));
  assert.strictEqual(L.bombState(rds[1], 92).s, 'exploded');
});

test('fmtClock formats m:ss and clamps below zero', () => {
  assert.strictEqual(L.fmtClock(115), '1:55');
  assert.strictEqual(L.fmtClock(40), '0:40');
  assert.strictEqual(L.fmtClock(59.9), '0:59');
  assert.strictEqual(L.fmtClock(0), '0:00');
  assert.strictEqual(L.fmtClock(-3), '0:00');
});

test('sample() gives interpolated, finite player state', () => {
  const rd = L.prepareRounds(REPLAY)[0];
  const P = REPLAY.rounds[0].frames.p;

  L.sample(rd, 0);
  assert.ok(L.pstate.every((p) => p.alive), 'everyone alive at t=0');

  L.sample(rd, 0.0625);  // halfway between frames 0 and 1
  assert.ok(close(L.pstate[0].x, (P[0].x[0] + P[0].x[1]) / 2, 1e-9));

  L.sample(rd, 30);      // slots 7 (12.4 s) and 5 (25.8 s) are dead by now
  assert.strictEqual(L.pstate.filter((p) => p.alive).length, 8);
  assert.strictEqual(L.pstate[7].alive, false);
  assert.strictEqual(L.pstate[7].hp, 0);
  assert.strictEqual(L.pstate[5].alive, false);

  L.sample(rd, rd.duration);  // last frame
  L.pstate.forEach((p, s) => {
    assert.ok(Number.isFinite(p.x) && Number.isFinite(p.y), `slot ${s} position finite`);
    assert.ok(p.yaw >= -180 && p.yaw <= 180, `slot ${s} yaw in range`);
    assert.ok(p.hp >= 0 && p.hp <= 100, `slot ${s} hp in range`);
  });
});

let passed = 0;
const failures = [];
for (const t of tests) {
  try {
    t.fn();
    passed += 1;
    console.log(`ok   ${t.name}`);
  } catch (err) {
    failures.push(t.name);
    console.log(`FAIL ${t.name}\n     ${err.message.split('\n').join('\n     ')}`);
  }
}
console.log(`\n${passed}/${tests.length} passed (page: ${path.relative(ROOT, pagePath)})`);
console.log('Not covered here: DOM, canvas drawing, layout, keyboard and pointer handling.');
process.exitCode = failures.length ? 1 : 0;
