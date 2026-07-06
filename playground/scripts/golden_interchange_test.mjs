// Golden test: a recorded interchange CONFORMS to the frozen z2r-teleop-1
// contract (decision 008 §"Interchange contract"), and its SCHEMA matches the
// reference writer scripts/ref_interchange_writer.mjs.
//
// Reuses the PURE serializer from src/teleop/lerobot_writer.ts (+ pusht_obs, zip)
// — transpiled in-memory with the project's own `typescript` dep, no new deps,
// no browser/WASM — exactly the pattern of scripts/contract_gate_test.mjs. Run:
//   node scripts/golden_interchange_test.mjs

import { readFileSync, writeFileSync, mkdtempSync, mkdirSync, existsSync } from 'node:fs';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, resolve, join } from 'node:path';
import { tmpdir } from 'node:os';
import { execFileSync } from 'node:child_process';
import ts from 'typescript';

const here = dirname(fileURLToPath(import.meta.url));
const srcDir = resolve(here, '..', 'src', 'teleop');
const outDir = mkdtempSync(join(tmpdir(), 'z2r-golden-'));

// Transpile a set of sibling .ts modules to .mjs (rewriting relative specifiers
// to add .mjs so they resolve in the temp dir), then import the entry module.
function transpileGraph(names) {
  for (const name of names) {
    const src = readFileSync(join(srcDir, `${name}.ts`), 'utf8');
    let js = ts.transpileModule(src, {
      compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 },
    }).outputText;
    js = js.replace(/from ['"](\.\/[^'"]+)['"]/g, (m, spec) =>
      spec.endsWith('.mjs') ? m : `from '${spec}.mjs'`,
    );
    writeFileSync(join(outDir, `${name}.mjs`), js);
  }
}
transpileGraph(['pusht_obs', 'zip', 'lerobot_writer']);
const writer = await import(pathToFileURL(join(outDir, 'lerobot_writer.mjs')).href);
const { InterchangeRecorder, serializeInterchange } = writer;

let failed = 0;
const ok = (name) => console.log(`  ok   ${name}`);
const bad = (name, detail) => {
  failed += 1;
  console.error(`  FAIL ${name}: ${detail}`);
};
const assert = (cond, name, detail) => (cond ? ok(name) : bad(name, detail));

const STATE_NAMES = [
  'pusher_x', 'pusher_y', 'tee_x', 'tee_y', 'sin_tee_yaw', 'cos_tee_yaw',
  'target_x', 'target_y', 'sin_target_yaw', 'cos_target_yaw',
];
const ONE_PX_PNG = new Uint8Array(
  Buffer.from(
    'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
    'base64',
  ),
);

// --- record a short scripted episode set THROUGH the writer ------------------
function syntheticObs(e, i) {
  const s = new Array(10);
  for (let k = 0; k < 10; k++) s[k] = Math.sin(0.1 * (e * 13 + i * 7 + k)) * 0.3;
  s[6] = 0; s[7] = 0; s[8] = 0; s[9] = 1; // fixed target cols per pusht_env
  return s;
}
function syntheticAction(e, i) {
  return [Math.cos(0.2 * (e + i)) * 0.5, Math.sin(0.2 * (e + i)) * 0.5];
}

const rec = new InterchangeRecorder();
const lengths = [6, 4];
for (let e = 0; e < lengths.length; e++) {
  rec.startEpisode();
  for (let i = 0; i < lengths[e]; i++) {
    rec.recordStep(syntheticObs(e, i), syntheticAction(e, i), i / 10, ONE_PX_PNG);
  }
  rec.finishEpisode();
}
const bundle = await rec.buildBundle();
const manifest = bundle.manifest;

// --- write the bundle to disk (folder form the converter reads) --------------
for (const f of bundle.files) {
  const p = join(outDir, 'bundle', f.path);
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, f.bytes);
}

// --- conformance checks against z2r-teleop-1 ---------------------------------
assert(manifest.interchange_version === 'z2r-teleop-1', 'interchange_version', manifest.interchange_version);
for (const k of ['repo_id', 'robot_type', 'fps', 'task', 'features', 'episodes']) {
  assert(k in manifest, `manifest has "${k}"`, 'missing');
}
assert(manifest.robot_type === 'pusher_2d', 'robot_type == pusher_2d', manifest.robot_type);
assert(manifest.fps === 10, 'fps == CONTROL_HZ (10)', String(manifest.fps));

const fs = manifest.features['observation.state'];
assert(fs && fs.dtype === 'float32', 'observation.state dtype float32', JSON.stringify(fs));
assert(fs && fs.shape.length === 1 && fs.shape[0] === 10, 'observation.state shape [10]', JSON.stringify(fs?.shape));
assert(
  fs && JSON.stringify(fs.names) === JSON.stringify(STATE_NAMES),
  'observation.state names == gen_demos STATE_NAMES',
  JSON.stringify(fs?.names),
);
const fa = manifest.features.action;
assert(fa && fa.dtype === 'float32' && fa.shape[0] === 2, 'action f32[2]', JSON.stringify(fa));
assert(
  fa && JSON.stringify(fa.names) === JSON.stringify(['pusher_vx', 'pusher_vy']),
  'action names [pusher_vx, pusher_vy]',
  JSON.stringify(fa?.names),
);
const fi = manifest.features['observation.image'];
assert(
  fi && fi.dtype === 'video' && JSON.stringify(fi.shape) === JSON.stringify([96, 96, 3]),
  'observation.image video[96,96,3]',
  JSON.stringify(fi),
);
assert(
  fi && JSON.stringify(fi.names) === JSON.stringify(['height', 'width', 'channel']),
  'observation.image names [height,width,channel]',
  JSON.stringify(fi?.names),
);

const filePaths = new Set(bundle.files.map((f) => f.path));
manifest.episodes.forEach((ep, e) => {
  const n = ep.length;
  assert(ep['observation.state'].length === n, `ep${e} state length == ${n}`, String(ep['observation.state'].length));
  assert(ep.action.length === n, `ep${e} action length == ${n}`, String(ep.action.length));
  assert(ep.timestamp.length === n, `ep${e} timestamp length == ${n}`, String(ep.timestamp.length));
  assert(ep['observation.state'].every((r) => r.length === 10), `ep${e} every state row is 10-dim`, 'bad row');
  assert(ep.action.every((r) => r.length === 2), `ep${e} every action row is 2-dim`, 'bad row');
  const mono = ep.timestamp.every((t, i) => i === 0 || t >= ep.timestamp[i - 1]);
  assert(mono, `ep${e} timestamps monotone non-decreasing`, JSON.stringify(ep.timestamp));
  assert(ep['observation.image'].length === n, `ep${e} image path per frame`, String(ep['observation.image']?.length));
  const allRef = ep['observation.image'].every(
    (rel) => filePaths.has(rel) && existsSync(join(outDir, 'bundle', rel)),
  );
  assert(allRef, `ep${e} every frame path is in the bundle AND on disk`, JSON.stringify(ep['observation.image']));
});

// --- zip round-trips (EOCD present, entry count matches) ---------------------
const { zipStore } = await import(pathToFileURL(join(outDir, 'zip.mjs')).href);
const zipped = zipStore(bundle.files);
const dv = new DataView(zipped.buffer, zipped.byteOffset, zipped.byteLength);
const eocdOffset = zipped.byteLength - 22;
assert(dv.getUint32(eocdOffset, true) === 0x06054b50, 'zip EOCD signature present', 'no EOCD');
assert(dv.getUint16(eocdOffset + 10, true) === bundle.files.length, 'zip entry count == files', 'mismatch');

// --- SCHEMA parity vs the independent reference writer -----------------------
const spikeOut = join(outDir, 'ref');
execFileSync('node', [resolve(here, 'ref_interchange_writer.mjs'), '--out', spikeOut, '--frames'], {
  stdio: 'ignore',
});
const spike = JSON.parse(readFileSync(join(spikeOut, 'interchange.json'), 'utf8'));
const canon = (v) =>
  JSON.stringify(v, (_k, val) =>
    val && typeof val === 'object' && !Array.isArray(val)
      ? Object.fromEntries(Object.keys(val).sort().map((k) => [k, val[k]]))
      : val,
  );
assert(canon(spike.features) === canon(manifest.features), 'features schema == spike writer', 'features differ');
assert(
  JSON.stringify(Object.keys(spike).sort()) === JSON.stringify(Object.keys(manifest).sort()),
  'manifest top-level keys == spike writer',
  `${Object.keys(spike)} vs ${Object.keys(manifest)}`,
);
const epKeys = (o) => JSON.stringify(Object.keys(o).sort());
assert(epKeys(spike.episodes[0]) === epKeys(manifest.episodes[0]), 'episode keys == spike writer', 'episode keys differ');

// --- also verify a STATE-ONLY recording omits image features/paths -----------
const soBundle = serializeInterchange([
  { state: [syntheticObs(0, 0), syntheticObs(0, 1)], action: [syntheticAction(0, 0), syntheticAction(0, 1)], timestamp: [0, 0.1] },
]);
assert(!('observation.image' in soBundle.manifest.features), 'state-only: no observation.image feature', 'present');
assert(soBundle.files.length === 1, 'state-only: only interchange.json in files', String(soBundle.files.length));
assert(!('observation.image' in soBundle.manifest.episodes[0]), 'state-only: no image paths on episode', 'present');

console.log(`\n${failed === 0 ? 'PASS' : 'FAIL'} — ${failed} failure(s). bundle: ${join(outDir, 'bundle')}`);
process.exit(failed === 0 ? 0 : 1);
