// Unit tests for the quality-tier auto-degrade monitor (src/render/quality.ts).
//
// Feeds the controller a synthetic frame stream (no DOM/browser) and proves:
//   - sustained low FPS degrades the tier one rung at a time and fires onChange
//   - it never degrades below the worst tier
//   - fast frames do NOT degrade (and, sustained, recover one rung)
//   - warmup frames are ignored
//   - the degrade threshold and streak length are honored (hysteresis)
//
// quality.ts is pure TS, transpiled in-memory with the project's own
// `typescript` dep (no new deps), like scripts/contract_gate_test.mjs. Run:
//   node scripts/quality_test.mjs

import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import ts from 'typescript';

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, '..', 'src', 'render');
const outDir = mkdtempSync(join(tmpdir(), 'z2r-quality-test-'));

async function loadTs(name) {
  const src = readFileSync(join(srcDir, `${name}.ts`), 'utf8');
  const js = ts.transpileModule(src, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const outPath = join(outDir, `${name}.mjs`);
  writeFileSync(outPath, js);
  return import(pathToFileURL(outPath).href);
}

const { QualityController, TIERS, DEFAULT_QUALITY_CONFIG } = await loadTs('quality');

let passed = 0;
let failed = 0;
const ok = (name) => {
  passed += 1;
  console.log(`  ok   ${name}`);
};
const bad = (name, detail) => {
  failed += 1;
  console.error(`  FAIL ${name}: ${detail}`);
};
const assert = (cond, name, detail) => (cond ? ok(name) : bad(name, detail));

const SLOW = 50; // 20 fps — below the 24 fps degrade threshold
const FAST = 8; // 125 fps — above the 55 fps recover threshold

// Tight config so a few synthetic frames drive tier changes deterministically.
// smoothing=1 makes the EMA track each sample exactly.
const tight = {
  degradeFps: 24,
  recoverFps: 55,
  degradeFrames: 3,
  recoverFrames: 4,
  warmupFrames: 2,
  smoothing: 1,
};

// --- sustained slow frames degrade, one rung, and fire onChange --------------
{
  const changes = [];
  const q = new QualityController({ config: tight, onChange: (s) => changes.push(s.tier) });
  q.sample(SLOW); // warmup 1
  q.sample(SLOW); // warmup 2 (still warming)
  assert(q.tier === 'full' && changes.length === 0, 'warmup frames ignored', `${q.tier} ${changes}`);
  // 3 sub-threshold frames past warmup -> one degrade to 'reduced'
  const c1 = q.sample(SLOW);
  const c2 = q.sample(SLOW);
  const c3 = q.sample(SLOW);
  assert(c3 === true && q.tier === 'reduced', 'sustained low FPS degrades to reduced', `${q.tier} c=${[c1, c2, c3]}`);
  assert(changes.length === 1 && changes[0] === 'reduced', 'onChange fired once with reduced', JSON.stringify(changes));
  // another 3 -> degrade to 'minimal'
  q.sample(SLOW);
  q.sample(SLOW);
  q.sample(SLOW);
  assert(q.tier === 'minimal', 'further low FPS degrades to minimal', q.tier);
  // never worse than the worst tier
  for (let i = 0; i < 10; i++) q.sample(SLOW);
  assert(q.tier === 'minimal' && q.tier === TIERS[TIERS.length - 1].tier, 'never degrades below worst tier', q.tier);
}

// --- fast frames never degrade ------------------------------------------------
{
  const changes = [];
  const q = new QualityController({ config: tight, onChange: (s) => changes.push(s.tier) });
  for (let i = 0; i < 30; i++) q.sample(FAST);
  assert(q.tier === 'full' && changes.length === 0, 'fast frames never degrade', `${q.tier} ${changes}`);
}

// --- recovery: after degrading, sustained fast frames climb back one rung ----
{
  const q = new QualityController({ config: tight });
  for (let i = 0; i < 8; i++) q.sample(SLOW); // degrade twice -> minimal
  assert(q.tier === 'minimal', 'setup: at minimal before recovery', q.tier);
  for (let i = 0; i < 4; i++) q.sample(FAST); // 4 fast frames -> recover one rung
  assert(q.tier === 'reduced', 'sustained fast FPS recovers one rung', q.tier);
}

// --- hysteresis: a couple of slow frames (< degradeFrames) do NOT degrade ----
{
  const q = new QualityController({ config: tight });
  q.sample(SLOW);
  q.sample(SLOW); // warmup
  q.sample(SLOW);
  q.sample(SLOW); // only 2 past warmup (< degradeFrames=3)
  assert(q.tier === 'full', 'brief slowness below streak does not degrade', q.tier);
}

// --- default config sanity ----------------------------------------------------
assert(
  DEFAULT_QUALITY_CONFIG.degradeFps < DEFAULT_QUALITY_CONFIG.recoverFps &&
    DEFAULT_QUALITY_CONFIG.recoverFrames > DEFAULT_QUALITY_CONFIG.degradeFrames,
  'default config has degrade<recover thresholds and hysteresis',
  JSON.stringify(DEFAULT_QUALITY_CONFIG),
);
assert(TIERS.length === 3 && TIERS[0].tier === 'full', 'three tiers, best first', JSON.stringify(TIERS.map((t) => t.tier)));

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
