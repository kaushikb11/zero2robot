// Teleop hero entry point: load the REAL PushT scene into MuJoCo-WASM, let the
// learner drag-to-push the pusher, run the env at 10 Hz control / 100 Hz
// physics, record episodes, and download them as a z2r-teleop-1 interchange.
// Keeps the spike's graceful-degradation bottom rung (never a blank screen) and
// the ONNX policy-hello (loaded + contract-checked, not driving this phase).
import { createSim, type Sim } from './sim/mujoco_sim';
import { PUSHT_XML } from './sim/scene';
import { renderScene } from './render/canvas2d';
import { worldToPx } from './teleop/viewport';
import { Meters } from './ui/meters';
// The ONNX runtime (onnxruntime-web) is heavy (~13 MB wasm + its JS glue). It is
// only needed once a learner actually loads a policy, so infer.ts is imported
// LAZILY (dynamic import inside loadPolicyBytes) — first-interactive pays only
// for mujoco.wasm. The Policy TYPE is import-type-only, so it is erased at build
// time and pulls no ORT code into the eager chunk.
import type { Policy } from './policy/infer';
import { assertDrivesPushT } from './policy/contracts';
import { QualityController, type QualityState } from './render/quality';
import { telemetry } from './telemetry/events';
import { BrowserPushTEnv } from './teleop/pusht_env';
import { DragController } from './teleop/controller';
import { buildObs, CONTROL_DT, CONTROL_HZ, FRAME_SKIP, IMG_HW } from './teleop/pusht_obs';
import { InterchangeRecorder, downloadInterchange, pngFromCanvas } from './teleop/lerobot_writer';
import { zipStore } from './teleop/zip';

const canvas = document.getElementById('scene') as HTMLCanvasElement;
const stageEl = document.getElementById('stage') as HTMLElement;
const metersEl = document.getElementById('meters') as HTMLElement;
const policyEl = document.getElementById('policy') as HTMLElement;
const statusEl = document.getElementById('status') as HTMLElement;
const teleopEl = document.getElementById('teleop') as HTMLElement;
const loadingEl = document.getElementById('loading') as HTMLElement | null;
const qualityEl = document.getElementById('quality') as HTMLElement | null;
const btnRecord = document.getElementById('btn-record') as HTMLButtonElement;
const btnReset = document.getElementById('btn-reset') as HTMLButtonElement;
const btnDownload = document.getElementById('btn-download') as HTMLButtonElement;
const chkImages = document.getElementById('chk-images') as HTMLInputElement;
const btnPolicy = document.getElementById('btn-policy') as HTMLButtonElement;
const btnPerturb = document.getElementById('btn-perturb') as HTMLButtonElement;
const policyFile = document.getElementById('policy-file') as HTMLInputElement;

const MAX_CONTROL_STEPS_PER_FRAME = 4; // cap catch-up so slow frames can't spiral

function fail(stage: string, err: unknown): void {
  // Graceful-degradation ladder, bottom rung: never a blank screen — report what
  // failed and point at the Colab path.
  const msg = err instanceof Error ? err.message : String(err);
  loadingEl?.classList.add('hidden'); // uncover the canvas so the error is visible
  statusEl.classList.add('error');
  statusEl.textContent =
    `${stage} failed: ${msg}\n` +
    `This browser cannot run the simulation. The Colab notebook path covers ` +
    `the same material without WASM.`;
  console.error(`[teleop] ${stage} failed`, err);
}

// --- keyboard teleop (arrow keys) as an accessibility fallback to drag ---------
const keys = new Set<string>();
window.addEventListener('keydown', (e) => {
  if (e.key.startsWith('Arrow')) {
    keys.add(e.key);
    e.preventDefault();
  }
});
window.addEventListener('keyup', (e) => keys.delete(e.key));

function keyboardAction(): [number, number] | null {
  if (keys.size === 0) return null;
  const vx = (keys.has('ArrowRight') ? 1 : 0) - (keys.has('ArrowLeft') ? 1 : 0);
  const vy = (keys.has('ArrowUp') ? 1 : 0) - (keys.has('ArrowDown') ? 1 : 0);
  if (vx === 0 && vy === 0) return null;
  return [vx, vy];
}

function nextFrame(): Promise<number> {
  return new Promise((r) => requestAnimationFrame(r));
}

async function main(): Promise<void> {
  statusEl.textContent = 'loading MuJoCo WASM + PushT scene…';
  const t0 = performance.now();

  let sim: Sim;
  try {
    sim = await createSim(PUSHT_XML);
  } catch (err) {
    fail('MuJoCo WASM load', err);
    return;
  }
  const loadMs = performance.now() - t0;
  loadingEl?.classList.add('hidden'); // sim is interactive — drop the loading screen
  statusEl.textContent =
    `PushT scene ready in ${loadMs.toFixed(0)} ms — ` +
    `timestep ${(sim.timestep * 1000).toFixed(1)} ms, nu=${sim.nu}, ` +
    `${CONTROL_HZ} Hz control`;
  console.log(`[teleop] scene ready in ${loadMs.toFixed(0)} ms`);
  telemetry.emit('sim-booted', { loadMs });

  const ctx = canvas.getContext('2d');
  if (!ctx) {
    fail('canvas 2d context', new Error('getContext("2d") returned null'));
    return;
  }

  const env = new BrowserPushTEnv(sim);
  const controller = new DragController(canvas);
  const recorder = new InterchangeRecorder();
  const meters = new Meters(metersEl);

  // --- quality-tier auto-degrade ----------------------------------------------
  // Pin the ON-SCREEN size to the base backing-store size so that when a slow
  // device forces a lower render tier we shrink the pixel COUNT (cheaper fills)
  // without shrinking the visible canvas. worldToPx uses canvas.width and
  // eventToWorld normalizes by rect.width, so both render and pointer mapping
  // stay correct at any backing-store resolution.
  const BASE_W = canvas.width;
  const BASE_H = canvas.height;
  canvas.style.width = `${BASE_W}px`;
  canvas.style.height = `${BASE_H}px`;
  let renderFrameSkip = 0; // repaint every (renderFrameSkip + 1)th frame
  let renderSkipCounter = 0;
  const quality = new QualityController({
    onChange: (s: QualityState) => {
      // Resize the backing store only when the scale actually changes (this
      // clears the canvas; the next frame repaints). Display size is unchanged.
      const w = Math.max(1, Math.round(BASE_W * s.renderScale));
      const h = Math.max(1, Math.round(BASE_H * s.renderScale));
      if (canvas.width !== w) {
        canvas.width = w;
        canvas.height = h;
      }
      renderFrameSkip = s.frameSkip;
      renderSkipCounter = 0; // force a repaint on the frame after a tier change
      (window as unknown as { __quality?: QualityState }).__quality = s;
      if (qualityEl) {
        if (s.tier === 'full') {
          qualityEl.classList.add('hidden');
        } else {
          qualityEl.classList.remove('hidden');
          qualityEl.textContent =
            `render tier: ${s.tier} — low FPS detected (~${s.fps.toFixed(0)} fps), ` +
            `lowered canvas detail to stay interactive on this device.`;
        }
      }
      console.log(`[quality] tier -> ${s.tier} (scale ${s.renderScale}, skip ${s.frameSkip})`);
    },
  });
  (window as unknown as { __quality?: QualityState }).__quality = quality.state;

  // Offscreen 96x96 canvas: observation.image frames are rendered from the exact
  // pre-step sim state here (not a downscale of the big canvas), so a recorded
  // frame matches its (obs, action) row.
  const frameCanvas = document.createElement('canvas');
  frameCanvas.width = IMG_HW;
  frameCanvas.height = IMG_HW;
  const frameCtx = frameCanvas.getContext('2d')!;

  let seedCounter = 0;
  let recording = false;
  let lastSuccess = false;

  env.reset(seedCounter);

  // Headless-observability hook: the whitelisted telemetry buffer (no network),
  // so browse/CI can assert only whitelisted event shapes were emitted.
  (window as unknown as { __telemetry?: unknown }).__telemetry = {
    events: (): unknown => telemetry.events().map((e) => ({ ...e })),
  };

  // Headless-measurement hook (kept from the spike): raw stepping throughput.
  (window as unknown as { __spikeBench?: (n: number) => number }).__spikeBench = (n: number) => {
    const t = performance.now();
    sim.step([0, 0], n);
    return performance.now() - t;
  };

  // Headless-verification hook (mirrors __spikeBench/__spikeMetrics): drive a
  // scripted drag and read back a serialized-bundle summary, so CI/browse can
  // exercise the real in-browser record -> toBlob PNG -> zip path end-to-end.
  (window as unknown as { __teleop?: unknown }).__teleop = {
    setDrag(worldX: number, worldY: number): void {
      // Simulate a held drag toward a world point (bypasses pointer plumbing).
      (controller as unknown as { dragging: boolean; pointerWorld: [number, number] }).dragging = true;
      (controller as unknown as { pointerWorld: [number, number] }).pointerWorld = [worldX, worldY];
    },
    clearDrag(): void {
      (controller as unknown as { dragging: boolean }).dragging = false;
    },
    async summary(): Promise<unknown> {
      const bundle = await recorder.buildBundle();
      const zipBytes = zipStore(bundle.files).length;
      return {
        version: bundle.manifest.interchange_version,
        fps: bundle.manifest.fps,
        robot_type: bundle.manifest.robot_type,
        featureKeys: Object.keys(bundle.manifest.features),
        stateShape: bundle.manifest.features['observation.state'].shape,
        actionShape: bundle.manifest.features.action.shape,
        episodes: bundle.manifest.episodes.map((e) => ({
          length: e.length,
          hasImages: 'observation.image' in e,
          firstAction: e.action[0],
        })),
        fileCount: bundle.files.length,
        zipBytes,
      };
    },
  };

  function setTeleopStatus(): void {
    const { posErr, angErr } = env.errors();
    const recTxt = recording
      ? `● REC ep${recorder.episodeCount} · ${recorder.currentLength} steps`
      : recorder.episodeCount > 0
        ? `${recorder.episodeCount} episode(s) buffered`
        : 'not recording';
    const successTxt = env.success ? '  <span class="ok">✔ SUCCESS</span>' : '';
    teleopEl.innerHTML =
      `seed ${env.lastSeed} · step ${env.steps}/300 · ` +
      `pos_err ${posErr.toFixed(3)} m · ang_err ${angErr.toFixed(3)} rad · ${recTxt}${successTxt}`;
  }

  function startRecording(): void {
    env.reset(nextSeed());
    recorder.startEpisode();
    recording = true;
    btnRecord.classList.add('active');
    btnRecord.textContent = '■ Stop recording';
  }
  function stopRecording(): void {
    if (recording) {
      const steps = recorder.currentLength; // capture before finishEpisode resets it
      recorder.finishEpisode();
      telemetry.emit('episode-recorded', { steps, withImages: chkImages.checked });
    }
    recording = false;
    btnRecord.classList.remove('active');
    btnRecord.textContent = '● Record episode';
    btnDownload.disabled = recorder.episodeCount === 0;
  }
  function nextSeed(): number {
    return ++seedCounter;
  }

  btnRecord.addEventListener('click', () => (recording ? stopRecording() : startRecording()));
  btnReset.addEventListener('click', () => {
    env.reset(nextSeed());
    lastSuccess = false;
    if (recording) {
      // Roll to a fresh episode so back-to-back demos each become one episode.
      const steps = recorder.currentLength;
      recorder.finishEpisode();
      telemetry.emit('episode-recorded', { steps, withImages: chkImages.checked });
      recorder.startEpisode();
    }
  });
  btnDownload.addEventListener('click', async () => {
    if (recording) stopRecording();
    if (recorder.episodeCount === 0) return;
    btnDownload.disabled = true;
    try {
      const bundle = await recorder.buildBundle();
      await downloadInterchange(bundle);
      console.log(
        `[teleop] downloaded interchange: ${bundle.manifest.episodes.length} eps, ` +
          `${bundle.files.length - 1} frame file(s)`,
      );
      recorder.clear();
    } catch (err) {
      fail('interchange download', err);
    } finally {
      btnDownload.disabled = recorder.episodeCount === 0;
    }
  });

  // --- policy-drive state ------------------------------------------------------
  // mode toggles between the learner's hand (teleop drag) and the loaded policy.
  // The policy drives EXACTLY as bc.py's eval rollout does:
  //   action = policy.act(env.obs());  env.step(action)
  // env.obs() is buildObs(sim) — the raw obs[10] the parity check pins equal to
  // pusht_env._obs(); the policy's raw action[2] (denorm baked into the ONNX)
  // is stepped with no browser-side transform.
  type DriveMode = 'teleop' | 'policy';
  let mode: DriveMode = 'teleop';
  let drivePolicy: Policy | null = null;
  let policyDriving = false; // async-driver reentry guard
  let physicsStepsPending = 0; // physics steps taken since the last meter tick
  let policyEpisodes = 0;
  let policySuccesses = 0;
  const NO_POLICY_MSG =
    'policy: none loaded — drag your bc_policy.onnx onto the scene (or “Load .onnx”) to drive.';
  policyEl.textContent = NO_POLICY_MSG;

  let accumulator = 0;
  let lastT = performance.now();
  let simTime = 0;

  function controlStep(): void {
    // Teleop control step. (obs-before, action) pair — same convention as
    // gen_demos.add_frame.
    const obs = env.obs();
    const [px, py] = [obs[0], obs[1]];
    const action = keyboardAction() ?? controller.action(px, py);

    let framePromise: Promise<Uint8Array> | undefined;
    if (recording && chkImages.checked) {
      renderScene(frameCtx, sim); // pre-step state into the 96x96 canvas
      framePromise = pngFromCanvas(frameCanvas);
    }
    if (recording) {
      const ts = recorder.currentLength / CONTROL_HZ; // advisory seconds
      recorder.recordStep(obs, action, ts, framePromise);
    }

    const result = env.step(action);
    physicsStepsPending += FRAME_SKIP;
    if (result.success && !lastSuccess) console.log(`[teleop] success at step ${result.stepCount}`);
    lastSuccess = result.success;

    if (result.done) {
      // Episode ends at success or MAX_STEPS. When recording, close the episode
      // like the env; when idle, recycle to a fresh seed so the demo stays live
      // and the step counter stays bounded.
      if (recording) stopRecording();
      else {
        env.reset(nextSeed());
        lastSuccess = false;
      }
    }
  }

  // --- policy driver (async; the "watch it drive/recover" loop) ----------------
  // Steps the env from the loaded policy, real-time paced to CONTROL_HZ so it is
  // watchable. Yields a rAF between checks so the render loop stays smooth. The
  // policyDriving guard makes overlapping start() calls a no-op.
  async function runPolicyDriver(): Promise<void> {
    if (policyDriving) return;
    policyDriving = true;
    let last = performance.now();
    let acc = 0;
    try {
      while (mode === 'policy' && drivePolicy) {
        await nextFrame();
        const now = performance.now();
        acc += Math.min(now - last, 100) / 1000;
        last = now;
        let n = 0;
        while (acc >= CONTROL_DT && n < MAX_CONTROL_STEPS_PER_FRAME && mode === 'policy' && drivePolicy) {
          const obs = env.obs(); // raw buildObs(sim) — byte-identical to training obs
          const action = await drivePolicy.act(obs); // raw action[2]; denorm baked in
          const result = env.step(action); // mirrors bc.py eval: env.step(policy(obs))
          physicsStepsPending += FRAME_SKIP;
          simTime += CONTROL_DT;
          acc -= CONTROL_DT;
          n += 1;
          if (result.success && !lastSuccess) console.log(`[policy] success at step ${result.stepCount}`);
          lastSuccess = result.success;
          if (result.done) {
            // Honest tally: a ~62% policy pushes it home sometimes, times out
            // sometimes. Latch the outcome, then recycle to a fresh scene so it
            // keeps driving live.
            policyEpisodes += 1;
            if (result.success) policySuccesses += 1;
            telemetry.emit('policy-drove', { episodes: policyEpisodes, successes: policySuccesses });
            env.reset(nextSeed());
            lastSuccess = false;
          }
        }
        if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0; // shed backlog
      }
    } catch (err) {
      // Inference blew up (bad model / runtime): fall back to teleop honestly.
      drivePolicy = null;
      stopPolicyDrive();
      policyEl.textContent = `policy: inference error — ${err instanceof Error ? err.message : err}`;
      console.error('[policy] inference error', err);
    } finally {
      policyDriving = false;
    }
  }

  function startPolicyDrive(): void {
    if (!drivePolicy || mode === 'policy') return;
    if (recording) stopRecording(); // driving is not recording
    mode = 'policy';
    btnRecord.disabled = true;
    btnPolicy.classList.add('active');
    btnPolicy.textContent = '■ Stop policy';
    void runPolicyDriver();
  }
  function stopPolicyDrive(): void {
    mode = 'teleop';
    btnRecord.disabled = false;
    lastT = performance.now(); // don't let the paused gap burst the teleop accumulator
    btnPolicy.classList.remove('active');
    btnPolicy.textContent = '▶ Run policy';
  }

  // --- policy loading (file picker / drag-drop) --------------------------------
  // Runs the SAME fail-closed contract gate as the toy path (loadPolicyFromBytes)
  // then the PushT drive gate (obs[10]/action[2]); a mismatch (e.g. the 4-dim
  // toy) is refused here with a human-readable reason and never drives.
  async function loadPolicyBytes(bytes: Uint8Array, label: string): Promise<Policy> {
    policyEl.classList.remove('error');
    try {
      // Lazy-load the ONNX runtime here, the first time a policy is loaded —
      // this is when the ~13 MB ort wasm + its JS glue are actually needed. The
      // sim keeps running (this is async and off the render loop). Surface the
      // fetch so a slow first load reads as progress, not a hang; a load failure
      // (offline/blocked) falls through to the catch and back to teleop.
      policyEl.textContent = `policy: loading inference runtime (first policy, ~13 MB)…`;
      const { loadPolicyFromBytes } = await import('./policy/infer');
      policyEl.textContent = `policy: validating ${label}…`;
      const p = await loadPolicyFromBytes(bytes);
      assertDrivesPushT(p.contract); // obs[10]/act[2] or refuse to drive
      drivePolicy = p;
      policyEpisodes = 0;
      policySuccesses = 0;
      btnPolicy.disabled = false;
      console.log(
        `[policy] loaded ${label}: contract ${p.contract.contractVersion}, ` +
          `obs_dim=${p.contract.obsDim}, act_dim=${p.contract.actDim}`,
      );
      telemetry.emit('policy-loaded', {
        obsDim: p.contract.obsDim,
        actDim: p.contract.actDim,
        contractVersion: p.contract.contractVersion,
      });
      return p;
    } catch (err) {
      drivePolicy = null;
      btnPolicy.disabled = true;
      if (mode === 'policy') stopPolicyDrive();
      policyEl.classList.add('error');
      policyEl.textContent = `policy rejected — ${err instanceof Error ? err.message : err}`;
      console.warn('[policy] rejected', err);
      throw err;
    }
  }
  async function loadPolicyFile(file: File): Promise<void> {
    const bytes = new Uint8Array(await file.arrayBuffer());
    await loadPolicyBytes(bytes, file.name).catch(() => {});
  }

  policyFile.addEventListener('change', () => {
    const f = policyFile.files?.[0];
    if (f) void loadPolicyFile(f);
    policyFile.value = ''; // allow re-picking the same file
  });
  stageEl.addEventListener('dragover', (e) => {
    e.preventDefault();
    stageEl.classList.add('drop');
  });
  stageEl.addEventListener('dragleave', () => stageEl.classList.remove('drop'));
  stageEl.addEventListener('drop', (e) => {
    e.preventDefault();
    stageEl.classList.remove('drop');
    const f = e.dataTransfer?.files?.[0];
    if (f) void loadPolicyFile(f);
  });
  btnPolicy.addEventListener('click', () => (mode === 'policy' ? stopPolicyDrive() : startPolicyDrive()));

  // --- perturb (the hero: knock the block askew, watch the policy recover) -----
  function perturbBlock(): Float32Array {
    // A HARD start: block far from the target (harder than reset's 0.1–0.24 m)
    // with a random yaw, pusher left where it is. Kept inside the walls.
    const ang = Math.random() * 2 * Math.PI;
    const r = 0.24 + Math.random() * 0.06; // 0.24–0.30 m from the goal
    const teeYaw = (Math.random() * 2 - 1) * Math.PI;
    lastSuccess = false;
    return env.perturbBlock(r * Math.cos(ang), r * Math.sin(ang), teeYaw);
  }
  btnPerturb.addEventListener('click', () => {
    perturbBlock();
    if (drivePolicy && mode !== 'policy') startPolicyDrive(); // one click = knock + recover
  });

  function setPolicyStatus(): void {
    if (!drivePolicy) return; // keep the load/rejection message visible
    const tally = policyEpisodes > 0 ? ` · drove ${policyEpisodes} (${policySuccesses} ✓)` : '';
    const state = mode === 'policy' ? '▶ DRIVING' : 'ready';
    policyEl.classList.remove('error');
    policyEl.textContent =
      `policy: ${state} · contract ${drivePolicy.contract.contractVersion} ` +
      `obs=${drivePolicy.contract.obsDim} act=${drivePolicy.contract.actDim} · ` +
      `${drivePolicy.meanLatencyMs().toFixed(2)} ms/call×${drivePolicy.calls}${tally}`;
  }

  function frame(now: number): void {
    const dtMs = Math.min(now - lastT, 100); // clamp tab-switch gaps
    lastT = now;

    // Frame-time monitor: sustained low FPS degrades the render tier (see the
    // onChange above). Physics/obs/recording are untouched by this.
    quality.sample(dtMs);

    let steps = 0;
    if (mode === 'teleop') {
      accumulator += dtMs / 1000;
      while (accumulator >= CONTROL_DT && steps < MAX_CONTROL_STEPS_PER_FRAME) {
        controlStep();
        accumulator -= CONTROL_DT;
        steps += 1;
        simTime += CONTROL_DT;
      }
      if (steps === MAX_CONTROL_STEPS_PER_FRAME) accumulator = 0; // shed backlog
    } else {
      accumulator = 0; // the async policy driver owns stepping in policy mode
    }

    // Render-tier frame skipping: at reduced/minimal tiers we repaint every
    // (renderFrameSkip + 1)th frame. Skipped frames keep the previous image
    // (renderScene owns the clearRect), so nothing blanks. Physics still steps.
    if (renderFrameSkip === 0 || renderSkipCounter === 0) {
      renderScene(ctx!, sim);
      drawDragCue(ctx!);
    }
    renderSkipCounter =
      renderFrameSkip === 0 ? 0 : (renderSkipCounter + 1) % (renderFrameSkip + 1);
    // Meters count physics steps taken this frame in EITHER mode.
    const stepped = physicsStepsPending;
    physicsStepsPending = 0;
    meters.tick(stepped, simTime);
    setTeleopStatus();
    setPolicyStatus();

    requestAnimationFrame(frame);
  }

  function drawDragCue(c: CanvasRenderingContext2D): void {
    if (!controller.isDragging) return;
    const px = env.obs()[0];
    const py = env.obs()[1];
    const [x0, y0] = worldToPx(canvas, px, py);
    const [x1, y1] = worldToPx(canvas, controller.target[0], controller.target[1]);
    c.strokeStyle = 'rgba(240, 200, 90, 0.85)';
    c.lineWidth = 2;
    c.beginPath();
    c.moveTo(x0, y0);
    c.lineTo(x1, y1);
    c.stroke();
    c.beginPath();
    c.arc(x1, y1, 5, 0, Math.PI * 2);
    c.fillStyle = 'rgba(240, 200, 90, 0.85)';
    c.fill();
  }

  // Headless policy-drive hook (mirrors __teleop/__spikeBench): load a real
  // bc.py ONNX, prove it passes the fail-closed gate, feed it buildObs(sim), and
  // step the env — so CI/browse can verify the full end-to-end drive without a
  // human. buildObs parity to training obs is asserted by obs_parity_check.mjs;
  // obsParity() below re-confirms env.obs() (what act() is fed) == buildObs(sim).
  (window as unknown as { __policy?: unknown }).__policy = {
    /** Fetch + load a model expected to PASS the gate. Returns its contract. */
    async loadUrl(url: string): Promise<unknown> {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`fetch ${url}: HTTP ${resp.status}`);
      const bytes = new Uint8Array(await resp.arrayBuffer());
      const p = await loadPolicyBytes(bytes, url.split('/').pop() ?? url);
      return { ...p.contract };
    },
    /** Attempt a load expected to be REFUSED. Returns the human-readable error
     *  instead of throwing, so the negative control is easy to assert. */
    async tryLoadUrl(url: string): Promise<{ accepted: boolean; error?: string; errorName?: string }> {
      const resp = await fetch(url);
      if (!resp.ok) throw new Error(`fetch ${url}: HTTP ${resp.status}`);
      const bytes = new Uint8Array(await resp.arrayBuffer());
      try {
        await loadPolicyBytes(bytes, url.split('/').pop() ?? url);
        return { accepted: true };
      } catch (err) {
        return {
          accepted: false,
          error: err instanceof Error ? err.message : String(err),
          errorName: err instanceof Error ? err.name : undefined,
        };
      }
    },
    contract(): unknown {
      return drivePolicy ? { ...drivePolicy.contract } : null;
    },
    /** env.obs() == buildObs(sim): the exact array act() is fed. */
    obsNow(): number[] {
      return Array.from(env.obs());
    },
    obsParity(): { equal: boolean; maxErr: number; obs: number[] } {
      const fed = env.obs(); // BrowserPushTEnv.obs() -> buildObs(sim)
      const direct = buildObs(sim); // build it again independently
      let maxErr = 0;
      for (let k = 0; k < fed.length; k++) maxErr = Math.max(maxErr, Math.abs(fed[k] - direct[k]));
      return { equal: maxErr === 0, maxErr, obs: Array.from(fed) };
    },
    /** One forward pass on the live obs — proves act() returns float32[act_dim]. */
    async actOnce(): Promise<unknown> {
      if (!drivePolicy) throw new Error('no policy loaded');
      const obs = env.obs();
      const action = await drivePolicy.act(obs);
      return {
        obs: Array.from(obs),
        action: Array.from(action),
        actionCtor: action.constructor.name,
        actionLen: action.length,
      };
    },
    perturb(): number[] {
      return Array.from(perturbBlock());
    },
    /** Drive N control steps from the loaded policy and report that the pusher
     *  actually moved (i.e. the policy drives), plus the pos-error delta. */
    async drive(nSteps: number): Promise<unknown> {
      if (!drivePolicy) throw new Error('no policy loaded');
      const before = env.obs();
      const errBefore = env.errors();
      for (let i = 0; i < nSteps; i++) {
        const obs = env.obs();
        const action = await drivePolicy.act(obs);
        env.step(action);
      }
      const after = env.obs();
      const errAfter = env.errors();
      return {
        steps: nSteps,
        pusherMoved: Math.hypot(after[0] - before[0], after[1] - before[1]),
        posErrBefore: errBefore.posErr,
        posErrAfter: errAfter.posErr,
        success: env.success,
        meanLatencyMs: drivePolicy.meanLatencyMs(),
        calls: drivePolicy.calls,
      };
    },
    reset(): number[] {
      return Array.from(env.reset(nextSeed()));
    },
    fps(): number | undefined {
      return window.__spikeMetrics?.fps;
    },
  };

  requestAnimationFrame(frame);
}

main().catch((err) => fail('startup', err));
