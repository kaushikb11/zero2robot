// SPIKE (milestone 0.4, Option 2 proof) — NOT production code.
//
// Fabricates a trivial, *browser-shaped* teleop interchange bundle using only
// plain JS (no python, no npm deps) — exactly what src/teleop/lerobot_writer.ts
// will emit in Phase 1. Two episodes of a handful of frames each. State-only by
// default; pass --frames to also emit tiny PNG-ish image frames to exercise the
// image-carrying path of the schema.
//
// Output: a directory bundle the browser would ZIP and hand to the user:
//   <out>/interchange.json         (manifest + inline episode arrays)
//   <out>/frames/ep{e}/f{i}.png    (only when --frames; one file per frame)
//
// The interchange is deliberately format-STABLE and library-agnostic: it mirrors
// the (observation.state, action, timestamp[, image]) shape the sim produces and
// nothing about lerobot's on-disk v3 layout. The canonical v3 write is done in
// Python by the pinned lerobot library (see spike_convert.py).

import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const args = process.argv.slice(2);
const outDir = args[args.indexOf('--out') + 1] ?? '/tmp/interchange';
const withFrames = args.includes('--frames');

// Feature contract — mirrors curriculum/common/envs/pusht/gen_demos.py
// build_features() + pusht_env.py OBS/ACT semantics. The browser knows these
// because it runs the SAME scene; it declares them so the converter stays
// env-agnostic (reads the spec instead of hardcoding it).
const STATE_NAMES = [
  'pusher_x', 'pusher_y', 'tee_x', 'tee_y', 'sin_tee_yaw', 'cos_tee_yaw',
  'target_x', 'target_y', 'sin_target_yaw', 'cos_target_yaw',
];
const OBS_DIM = 10;
const ACT_DIM = 2;
const FPS = 10;              // == PushTEnv.CONTROL_HZ
const IMG_HW = 96;

const features = {
  'observation.state': { dtype: 'float32', shape: [OBS_DIM], names: STATE_NAMES },
  action: { dtype: 'float32', shape: [ACT_DIM], names: ['pusher_vx', 'pusher_vy'] },
};
if (withFrames) {
  features['observation.image'] = {
    dtype: 'video', shape: [IMG_HW, IMG_HW, 3], names: ['height', 'width', 'channel'],
  };
}

// Deterministic pseudo-data so the round-trip is reproducible. Values are within
// the env's ranges but otherwise meaningless — the spike proves FORMAT, not data.
function fakeObs(e, i) {
  const s = [];
  for (let k = 0; k < OBS_DIM; k++) s.push(Math.sin(0.1 * (e * 13 + i * 7 + k)) * 0.3);
  s[6] = 0.0; s[7] = 0.0; s[8] = 0.0; s[9] = 1.0; // fixed target cols, per pusht_env
  return s;
}
function fakeAction(e, i) {
  return [Math.cos(0.2 * (e + i)) * 0.5, Math.sin(0.2 * (e + i)) * 0.5];
}

// A minimal valid PNG (1x1 opaque pixel) — stands in for a browser
// canvas.toBlob('image/png') frame. Phase 1 emits the real 96x96 frame; the
// converter path (decode PNG -> HxWx3 uint8) is identical.
const ONE_PX_PNG = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
  'base64',
);

const episodeLengths = [6, 4]; // two trivial episodes
const episodes = [];
mkdirSync(outDir, { recursive: true });

for (let e = 0; e < episodeLengths.length; e++) {
  const n = episodeLengths[e];
  const observationState = [];
  const action = [];
  const timestamp = [];
  const imagePaths = [];
  if (withFrames) mkdirSync(join(outDir, 'frames', `ep${e}`), { recursive: true });
  for (let i = 0; i < n; i++) {
    observationState.push(fakeObs(e, i));
    action.push(fakeAction(e, i));
    timestamp.push(i / FPS);                 // seconds; converter can derive too
    if (withFrames) {
      const rel = `frames/ep${e}/f${i}.png`;
      writeFileSync(join(outDir, rel), ONE_PX_PNG);
      imagePaths.push(rel);
    }
  }
  const ep = { length: n, 'observation.state': observationState, action, timestamp };
  if (withFrames) ep['observation.image'] = imagePaths;
  episodes.push(ep);
}

const manifest = {
  interchange_version: 'z2r-teleop-1',
  repo_id: 'zero2robot/pusht_teleop',
  robot_type: 'pusher_2d',
  fps: FPS,
  task: 'Push the T-shaped block to the target pose.',
  features,
  episodes,
};

writeFileSync(join(outDir, 'interchange.json'), JSON.stringify(manifest, null, 2));
console.log(
  `wrote interchange (${episodes.length} eps, ${episodeLengths.reduce((a, b) => a + b, 0)} frames` +
  `${withFrames ? ', +png frames' : ', state-only'}) to ${outDir}`,
);
