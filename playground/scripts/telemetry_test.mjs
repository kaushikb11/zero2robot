// Unit tests for the telemetry WHITELIST + emitter (src/telemetry/events.ts).
//
// Proves the privacy contract playground/CLAUDE.md relies on:
//   - only whitelisted event TYPES emit; an unknown type throws (fail-closed)
//   - only whitelisted FIELD names survive; extra/PII keys are DROPPED
//   - non-primitive values for a whitelisted key are dropped
//   - the default emitter records to an in-memory buffer with NO network sink
//   - disabled emitter is a no-op
//
// events.ts is pure TS with no DOM access at import time, so we transpile it
// in-memory with the project's own `typescript` dep (no new deps, no browser),
// exactly like scripts/contract_gate_test.mjs. Run:
//   node scripts/telemetry_test.mjs

import { readFileSync, writeFileSync, mkdtempSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import ts from 'typescript';

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, '..', 'src', 'telemetry');
const outDir = mkdtempSync(join(tmpdir(), 'z2r-telemetry-test-'));

async function loadTs(name) {
  const src = readFileSync(join(srcDir, `${name}.ts`), 'utf8');
  const js = ts.transpileModule(src, {
    compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
  }).outputText;
  const outPath = join(outDir, `${name}.mjs`);
  writeFileSync(outPath, js);
  return import(pathToFileURL(outPath).href);
}

const {
  EVENT_TYPES,
  EVENT_FIELDS,
  sanitizeEvent,
  Telemetry,
  MemorySink,
  localStorageSink,
  noopSink,
} = await loadTs('events');

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

// --- the whitelist is exactly the four progress events -----------------------
assert(
  EVENT_TYPES.length === 4 &&
    ['sim-booted', 'policy-loaded', 'episode-recorded', 'policy-drove'].every((t) =>
      EVENT_TYPES.includes(t),
    ),
  'EVENT_TYPES is the four progress events',
  JSON.stringify(EVENT_TYPES),
);

// --- unknown type throws (fail-closed) ---------------------------------------
try {
  sanitizeEvent('exfiltrate', { email: 'a@b.com' });
  bad('unknown type refused', 'did not throw — FAILED OPEN');
} catch (e) {
  assert(/not a whitelisted event type/.test(e.message), 'unknown type refused', e.message);
}

// --- extra / PII fields are dropped ------------------------------------------
const ev = sanitizeEvent(
  'policy-loaded',
  {
    obsDim: 10,
    actDim: 2,
    contractVersion: 'v1',
    // hostile extras that must NOT survive:
    email: 'learner@example.com',
    ip: '10.0.0.1',
    userAgent: 'evil',
  },
  1234,
);
const keys = Object.keys(ev).sort();
assert(
  JSON.stringify(keys) === JSON.stringify(['actDim', 'contractVersion', 'obsDim', 't', 'type']),
  'policy-loaded keeps only whitelisted keys (+type,t)',
  JSON.stringify(keys),
);
assert(ev.type === 'policy-loaded' && ev.t === 1234, 'type + timestamp preserved', JSON.stringify(ev));
assert(!('email' in ev) && !('ip' in ev) && !('userAgent' in ev), 'PII keys dropped', JSON.stringify(ev));

// --- non-primitive value for a whitelisted key is dropped --------------------
const ev2 = sanitizeEvent('episode-recorded', { steps: { toString: () => 'x' }, withImages: true }, 1);
assert(!('steps' in ev2) && ev2.withImages === true, 'non-primitive whitelisted value dropped', JSON.stringify(ev2));

// --- every event type sanitizes to its declared fields -----------------------
for (const type of EVENT_TYPES) {
  const payload = Object.fromEntries(EVENT_FIELDS[type].map((f) => [f, f === 'contractVersion' ? 'v1' : 1]));
  const out = sanitizeEvent(type, payload, 0);
  const got = Object.keys(out).filter((k) => k !== 'type' && k !== 't').sort();
  const want = [...EVENT_FIELDS[type]].sort();
  assert(JSON.stringify(got) === JSON.stringify(want), `${type} -> ${want.join(',')}`, JSON.stringify(got));
}

// --- default emitter: in-memory buffer, NO network sink ----------------------
const t = new Telemetry();
t.emit('sim-booted', { loadMs: 42 });
t.emit('policy-drove', { episodes: 3, successes: 2 });
assert(t.events().length === 2, 'default emitter buffers events', String(t.events().length));
assert(
  t.events().every((e) => EVENT_TYPES.includes(e.type)),
  'buffered events are all whitelisted types',
  JSON.stringify(t.events()),
);
// localStorageSink() under Node (no localStorage) must be the no-op sink.
assert(localStorageSink() === noopSink, 'localStorageSink is no-op without localStorage', 'expected noopSink');

// --- custom sink receives only sanitized events; disabled = no-op ------------
const sink = new MemorySink();
const t2 = new Telemetry({ sinks: [sink] });
t2.emit('episode-recorded', { steps: 120, withImages: false, secret: 'nope' });
assert(sink.events.length === 1 && !('secret' in sink.events[0]), 'custom sink gets sanitized event', JSON.stringify(sink.events));
t2.setEnabled(false);
assert(t2.emit('sim-booted', { loadMs: 1 }) === null, 'disabled emit returns null (no-op)', 'expected null');
assert(sink.events.length === 1, 'disabled emit reaches no sink', String(sink.events.length));

console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed === 0 ? 0 : 1);
