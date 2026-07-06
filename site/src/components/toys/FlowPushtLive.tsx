/**
 * ch1.5 "Flow Matching" — the LIVE generative-policy toy (`demo: flow_pusht_live`,
 * also stacked under `flow_multimodality` beneath the 2D flow-ring).
 *
 * The money shot the flow chapter's PushT panel was blocked on (contract v1 could
 * not express a sampler): a REAL flow_velocity.onnx drives a REAL MuJoCo-WASM
 * PushT sim, and the visitor watches a GENERATIVE policy act live. Every control
 * step the runtime SAMPLES an action — starts from seeded noise, integrates the
 * learned velocity field forward with `flow_steps` forward-Euler evals (the
 * sampler-aware contract-v2 runtime, sampler.ts's eulerSample), un-standardizes,
 * and steps the env. It samples the SAME actions as flow.py's Python sampler for
 * the same obs+noise (proven by scripts/flow_sampler_parity_check.mjs).
 *
 * The ONE control is `flow_steps` — 2 vs many — the chapter's decoupling lesson
 * made live: the SAME net, re-integrated, still lands the block with only a few
 * Euler steps (straight conditional paths need few), while 2 under-resolves the
 * curved marginal field and the pusher wanders (the chapter's Break-It).
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT at the top of ../PlateIsland.tsx:
 *   1. SSR poster == the JS-off fallback (<Poster/> is pure static JSX).
 *   2. Lazy, hydration-gated sim: MuJoCo-WASM + onnxruntime pulled by *dynamic*
 *      import() inside the post-hydration effect (mounted client:visible).
 *   3. Reuse the primitives verbatim: createSim + PUSHT_XML + BrowserPushTEnv +
 *      buildObs + loadSamplerPolicy through the fail-closed assertSamplerDrivesPushT
 *      gate. The driver loop mirrors flow.py's eval:
 *        obs = env.obs(); action = await policy.sampleAction(obs); env.step(action)
 *   4. Make the invisible visible: pos_err -> target ticks down as the sampled
 *      strokes home the block; the flow_steps readout ties it to the toy's lesson.
 *   5. ONE control (flow_steps), immediate feedback, default-interesting (boots at
 *      the full step budget, homing). Keyboard + button path to the same aha.
 *   6. Colour discipline: --entity-* for entities, ONE --signal blue for the live
 *      flow_steps control, --alert red for the under-resolved (few-step) readout.
 */
import "./FlowPushtLive.css";
import { useEffect, useRef, useState } from "preact/hooks";

const MODEL_URL = "/models/flow_velocity.onnx";
const CANVAS_PX = 512;
const WORLD_HALF = 0.45; // matches viewport.WORLD_HALF_EXTENT
const MAX_CONTROL_STEPS_PER_FRAME = 2; // sampling is heavier than a v1 forward pass; cap catch-up low
const STEP_CHOICES = [2, 5, 100] as const; // the flow_steps control: few-step -> the default budget
const FEW_STEPS = 2; // at/below this the curved marginal field is under-integrated (the Break-It)

// ---------------------------------------------------------- shared SSR poster
// Pure static JSX (no window/document) — the JS-off experience + pre-boot frame.
// Same ±WORLD_HALF top-down frame as the live canvas, so booting causes no reflow.
const POSTER_V = 500;
const POSTER_S = POSTER_V / (2 * WORLD_HALF);
const w2s = (x: number, y: number): [number, number] => [
  POSTER_V / 2 + x * POSTER_S,
  POSTER_V / 2 - y * POSTER_S,
];

/** A PushT "T" in SVG px at (world) center/yaw. Bar 0.12×0.03, stem 0.03×0.09. */
function PosterTee({ x, y, yawDeg, className }: { x: number; y: number; yawDeg: number; className: string }) {
  const [cx, cy] = w2s(x, y);
  const barW = 0.12 * POSTER_S, barH = 0.03 * POSTER_S;
  const stemW = 0.03 * POSTER_S, stemH = 0.09 * POSTER_S;
  return (
    <g class={className} transform={`translate(${cx.toFixed(1)} ${cy.toFixed(1)}) rotate(${-yawDeg})`}>
      <rect x={-barW / 2} y={-barH / 2} width={barW} height={barH} rx={2} />
      <rect x={-stemW / 2} y={0.06 * POSTER_S - stemH / 2} width={stemW} height={stemH} rx={2} />
    </g>
  );
}

function Poster() {
  const blockX = 0.14, blockY = 0.1;
  const pusherX = 0.21, pusherY = 0.16;
  const [pcx, pcy] = w2s(pusherX, pusherY);
  return (
    <svg
      class="fl-poster-svg"
      viewBox={`0 0 ${POSTER_V} ${POSTER_V}`}
      role="img"
      aria-label="Top-down PushT arena. A magenta T-block sits near the amber pusher, with the dashed green target pose at the center. With JavaScript on, a flow-matching generative policy samples each move by integrating a learned velocity field from noise, and pushes the block onto the target."
    >
      <title>Flow-matching policy — a generative policy drives PushT live</title>
      <desc>
        A flow policy produces each action by starting from noise and integrating a
        learned velocity field forward a few Euler steps. Live, it pushes the T-block
        onto the dashed target; a slider trades sampling steps for speed.
      </desc>

      <rect class="fl-arena" x={2} y={2} width={POSTER_V - 4} height={POSTER_V - 4} rx={6} />
      <g class="fl-grid">
        {Array.from({ length: 9 }, (_, i) => ((i + 1) * POSTER_V) / 10).map((v) => (
          <>
            <line x1={v} y1={2} x2={v} y2={POSTER_V - 2} />
            <line x1={2} y1={v} x2={POSTER_V - 2} y2={v} />
          </>
        ))}
      </g>

      <PosterTee x={0} y={0} yawDeg={0} className="fl-target" />
      <text class="fl-target-label" x={w2s(0.03, 0.02)[0]} y={w2s(0.03, 0.02)[1]}>target</text>
      <PosterTee x={blockX} y={blockY} yawDeg={-20} className="fl-tee" />
      <g transform={`translate(${pcx.toFixed(1)} ${pcy.toFixed(1)})`}>
        <circle class="fl-pusher-ring" r={0.028 * POSTER_S} />
        <circle class="fl-pusher-core" r={0.015 * POSTER_S} />
      </g>
      <text class="fl-poster-hint" x={POSTER_V / 2} y={POSTER_V - 22}>a generative policy samples each move →</text>
    </svg>
  );
}

// ------------------------------------------------------------------- live island
interface Hud {
  posErr: number;
  steps: number; // active flow_steps
  reached: boolean;
  fewStep: boolean;
  fps: number;
  latMs: number;
  error?: string;
}

type Tok =
  | "--entity-pusher" | "--entity-block" | "--entity-target"
  | "--signal" | "--alert" | "--ink-mute" | "--rule-strong";
const TOK_FALLBACK: Record<Tok, string> = {
  "--entity-pusher": "#b0560f",
  "--entity-block": "#a5257d",
  "--entity-target": "#0c7d5f",
  "--signal": "#1f56de",
  "--alert": "#c0362a",
  "--ink-mute": "#6d6252",
  "--rule-strong": "#c8bc9e",
};

function FlowToy() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const figureRef = useRef<HTMLElement>(null);
  const apiRef = useRef<{ setSteps: (n: number) => void; reset: () => void } | null>(null);
  const stepsRef = useRef<number>(STEP_CHOICES[STEP_CHOICES.length - 1]); // start at the full budget
  const [booted, setBooted] = useState(false);
  const [activeSteps, setActiveSteps] = useState<number>(stepsRef.current);
  const [hud, setHud] = useState<Hud>({
    posErr: 0, steps: stepsRef.current, reached: false, fewStep: false, fps: 0, latMs: 0,
  });

  useEffect(() => {
    let disposed = false;
    let sim: { dispose(): void } | null = null;
    let raf = 0;
    const nextFrame = () => new Promise<number>((r) => (raf = requestAnimationFrame(r)));

    (async () => {
      try {
        // --- lazy, hydration-gated: WASM + ONNX pulled only now (scrolled in) ----
        const [simMod, sceneMod, envMod, obsMod, inferMod, contractsMod, vpMod] =
          await Promise.all([
            import("../../../../playground/src/sim/mujoco_sim"),
            import("../../../../playground/src/sim/scene"),
            import("../../../../playground/src/teleop/pusht_env"),
            import("../../../../playground/src/teleop/pusht_obs"),
            import("../../../../playground/src/policy/infer"),
            import("../../../../playground/src/policy/contracts"),
            import("../../../../playground/src/teleop/viewport"),
          ]);
        const { createSim } = simMod;
        const { PUSHT_XML } = sceneMod;
        const { BrowserPushTEnv } = envMod;
        const { buildObs, CONTROL_DT, POS_TOL } = obsMod;
        const { loadSamplerPolicy } = inferMod;
        const { assertSamplerDrivesPushT } = contractsMod;
        const { worldToPx } = vpMod;

        // 1) boot the REAL PushT scene + env
        const realSim: any = await createSim(PUSHT_XML);
        sim = realSim;
        if (disposed) { realSim.dispose(); return; }
        const env: any = new BrowserPushTEnv(realSim);
        let seed = 5;
        env.reset(seed); // default-interesting: a fresh in-distribution start

        const canvas = canvasRef.current!;
        canvas.width = CANVAS_PX;
        canvas.height = CANVAS_PX;
        const ctx = canvas.getContext("2d")!;

        const cs = getComputedStyle(canvas);
        const col = (k: Tok) => (cs.getPropertyValue(k).trim() || TOK_FALLBACK[k]);
        const PUSHER = col("--entity-pusher"), BLOCK = col("--entity-block"),
          TARGET = col("--entity-target"), INK = col("--ink-mute");

        const teeCorners = (cx: number, cy: number, yaw: number, hx: number, hy: number, ox: number, oy: number) => {
          const c = Math.cos(yaw), s = Math.sin(yaw);
          return [[ox - hx, oy - hy], [ox + hx, oy - hy], [ox + hx, oy + hy], [ox - hx, oy + hy]]
            .map(([lx, ly]) => worldToPx(canvas, cx + lx * c - ly * s, cy + lx * s + ly * c));
        };
        const fillPoly = (pts: number[][], style: string) => {
          ctx.beginPath();
          pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
          ctx.closePath();
          ctx.fillStyle = style;
          ctx.fill();
        };
        const strokePoly = (pts: number[][], style: string) => {
          ctx.beginPath();
          pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
          ctx.closePath();
          ctx.strokeStyle = style;
          ctx.stroke();
        };
        const render = () => {
          const w = canvas.width, h = canvas.height;
          ctx.fillStyle = "#fbf9f3";
          ctx.fillRect(0, 0, w, h);
          ctx.strokeStyle = "rgba(200,188,158,0.5)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let i = 1; i < 10; i++) {
            const p = (i * w) / 10;
            ctx.moveTo(p, 0); ctx.lineTo(p, h);
            ctx.moveTo(0, p); ctx.lineTo(w, p);
          }
          ctx.stroke();

          // target (fixed at origin) — dashed emerald outline
          ctx.save();
          ctx.setLineDash([5, 4]);
          ctx.lineWidth = 2;
          strokePoly(teeCorners(0, 0, 0, 0.06, 0.015, 0, 0), TARGET);
          strokePoly(teeCorners(0, 0, 0, 0.015, 0.045, 0, -0.06), TARGET);
          ctx.restore();

          // block (tee) — magenta
          const tx = realSim.jointQpos("tee_x"), ty = realSim.jointQpos("tee_y"), tyaw = realSim.jointQpos("tee_yaw");
          fillPoly(teeCorners(tx, ty, tyaw, 0.06, 0.015, 0, 0), BLOCK);
          fillPoly(teeCorners(tx, ty, tyaw, 0.015, 0.045, 0, -0.06), BLOCK);

          // pusher (agent) — amber
          const [px, py] = worldToPx(canvas, realSim.jointQpos("pusher_x"), realSim.jointQpos("pusher_y"));
          const rPx = 0.015 * (canvas.width / (2 * WORLD_HALF));
          ctx.save();
          ctx.globalAlpha = 0.5; ctx.strokeStyle = PUSHER; ctx.lineWidth = 1.5;
          ctx.beginPath(); ctx.arc(px, py, rPx * 1.9, 0, Math.PI * 2); ctx.stroke();
          ctx.restore();
          ctx.fillStyle = PUSHER;
          ctx.beginPath(); ctx.arc(px, py, rPx, 0, Math.PI * 2); ctx.fill();
          void INK;
        };

        setBooted(true);
        render();

        // 2) load the REAL sampler policy through the fail-closed contract gate
        const policy = await loadSamplerPolicy(MODEL_URL, seed);
        assertSamplerDrivesPushT(policy.contract);
        if (disposed) return;

        // seed the sampler from the episode seed (flow.py rollout does this at reset)
        policy.seedNoise(seed);

        apiRef.current = {
          setSteps: (n: number) => { stepsRef.current = n; setActiveSteps(n); },
          reset: () => { seed += 1; env.reset(seed); policy.seedNoise(seed); },
        };

        // 3) headless-verification hooks (mirror playground's window.__policy) so a
        //    browser driver can PROVE the sampler drives + obs matches training.
        (window as any).__toy = {
          contract: () => ({ ...policy.contract }),
          obsParity: () => {
            const a = env.obs(); const b = buildObs(realSim);
            let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k]));
            return { equal: m === 0, maxErr: m };
          },
          posErr: () => env.errors().posErr,
          flowSteps: () => stepsRef.current,
          setSteps: (n: number) => apiRef.current?.setSteps(n),
          reset: () => apiRef.current?.reset(),
          // Determinism probe: same seed -> same first sampled action.
          async sampleDeterminism(numSteps: number) {
            const obs = env.obs();
            policy.seedNoise(1234);
            const a = await policy.sampleAction(obs, { numSteps });
            policy.seedNoise(1234);
            const b = await policy.sampleAction(obs, { numSteps });
            let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k]));
            return { equal: m === 0, maxErr: m, action: [...a] };
          },
          async drive(n: number, numSteps?: number) {
            const before = env.errors().posErr;
            for (let i = 0; i < n; i++) {
              const o = env.obs();
              const res = env.step(await policy.sampleAction(o, { numSteps: numSteps ?? stepsRef.current }));
              if (res.done) { seed += 1; env.reset(seed); policy.seedNoise(seed); }
            }
            return {
              steps: n, posErrBefore: before, posErrAfter: env.errors().posErr,
              meanLatencyMs: policy.meanLatencyMs(), calls: policy.calls,
            };
          },
          fps: () => lastFps,
        };

        // 4) DRIVE — real-time paced to CONTROL_HZ (mirrors flow.py eval: at every
        //    control step SAMPLE an action by integrating the ODE from noise).
        let lastFps = 0, frames = 0, fpsMark = performance.now(), last = performance.now(), acc = 0, hudMark = 0;

        while (!disposed) {
          await nextFrame();
          const now = performance.now();
          frames++;
          if (now - fpsMark >= 500) { lastFps = (frames * 1000) / (now - fpsMark); frames = 0; fpsMark = now; }

          acc += Math.min(now - last, 100) / 1000;
          last = now;
          let n = 0;
          while (acc >= CONTROL_DT && n < MAX_CONTROL_STEPS_PER_FRAME) {
            const obs = env.obs();                                    // buildObs(sim) — training obs, verbatim
            const action = await policy.sampleAction(obs, { numSteps: stepsRef.current }); // the JS Euler ODE
            const res = env.step(action);                            // mirrors flow.py eval: env.step(sample)
            acc -= CONTROL_DT; n += 1;
            if (res.done) { seed += 1; env.reset(seed); policy.seedNoise(seed); }
            if (disposed) break;
          }
          if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0;

          render();

          if (now - hudMark >= 100) { // throttle HUD to ~10 Hz
            hudMark = now;
            const { posErr } = env.errors();
            setHud({
              posErr,
              steps: stepsRef.current,
              reached: posErr < POS_TOL,
              fewStep: stepsRef.current <= FEW_STEPS,
              fps: lastFps,
              latMs: policy.meanLatencyMs(),
            });
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch1.5 flow toy] failed", err);
        setHud((h) => ({ ...h, error: msg }));
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); if (sim) sim.dispose(); };
  }, []);

  const failed = !!hud.error;

  const onKeyDown = (e: KeyboardEvent) => {
    const api = apiRef.current;
    if (!api) return;
    if (e.key === "r" || e.key === "R") { api.reset(); return; }
    if (e.key === "[" || e.key === "]") {
      e.preventDefault();
      const i = STEP_CHOICES.indexOf(stepsRef.current as typeof STEP_CHOICES[number]);
      const j = e.key === "]" ? Math.min(STEP_CHOICES.length - 1, i + 1) : Math.max(0, i - 1);
      api.setSteps(STEP_CHOICES[j]);
    }
  };

  return (
    <div class="fl">
      <figure
        ref={figureRef}
        class="fl-figure"
        tabIndex={0}
        role="application"
        aria-label="Interactive flow-matching PushT toy. A generative policy samples each move by integrating a learned velocity field from noise, and pushes the T-block toward the target. Use the flow-steps buttons below, or the left/right bracket keys when this figure is focused, to change how many Euler steps each sample takes — a few still land the block, two under-resolves. Press R to reset with a new block spawn."
        onKeyDown={onKeyDown}
      >
        <div class="fl-poster" hidden={booted}><Poster /></div>
        <canvas ref={canvasRef} class="fl-canvas" hidden={!booted} aria-hidden="true" />

        <div class="bk-sr" aria-live="polite">
          {booted && !failed
            ? hud.reached
              ? "The flow policy pushed the block onto the target."
              : hud.fewStep
                ? "With only two Euler steps per sample the policy under-resolves the velocity field, and the pusher wanders instead of homing the block."
                : "The generative policy is sampling each move and pushing the block toward the target."
            : ""}
        </div>

        {booted && !failed && (
          <div class="fl-hud" aria-hidden="true">
            <div class="fl-hud-row">
              <span class="fl-k">pos_err → target</span>
              <span class={`fl-v ${hud.reached ? "fl-ok" : ""}`}>
                {hud.posErr.toFixed(3)} m {hud.reached ? "✓" : ""}
              </span>
            </div>
            <div class="fl-hud-row">
              <span class="fl-k">flow_steps</span>
              <span class={`fl-v ${hud.fewStep ? "fl-bad" : "fl-signal"}`}>
                {hud.steps} {hud.fewStep ? "▲ under-resolved" : "Euler evals/action"}
              </span>
            </div>
          </div>
        )}

        <div class="fl-status" data-failed={failed} aria-hidden="true">
          {failed ? (
            <span>sim failed — the Colab path covers this without WASM</span>
          ) : booted ? (
            <>
              <span>real flow_velocity.onnx · sampled ODE (contract v2)</span>
              <span>{hud.fps.toFixed(0)} fps · {hud.latMs.toFixed(1)} ms/sample</span>
            </>
          ) : (
            <span>booting MuJoCo-WASM + flow sampler…</span>
          )}
        </div>
      </figure>

      <div class="fl-controls">
        <span class="fl-controls-label" id="fl-steps-label">flow_steps</span>
        <div class="fl-seg" role="group" aria-labelledby="fl-steps-label">
          {STEP_CHOICES.map((n) => (
            <button
              type="button"
              class={`fl-btn ${activeSteps === n ? "fl-btn--on" : ""}`}
              aria-pressed={activeSteps === n}
              disabled={!booted || failed}
              onClick={() => apiRef.current?.setSteps(n)}
            >
              {n}
            </button>
          ))}
        </div>
        <button type="button" class="fl-btn fl-btn--reset" onClick={() => apiRef.current?.reset()} disabled={!booted || failed}>
          reset · new spawn
        </button>
        <span class="fl-control-note">
          fewer steps still land it (straight paths) · 2 under-resolves · poster reads with JS off
        </span>
      </div>
    </div>
  );
}

export default function FlowPushtLive() {
  return <FlowToy />;
}
