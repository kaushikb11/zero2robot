/**
 * PlateIsland — the chapter hero's "See it work" island.
 *
 * For ch1.1 (demo === "pusht_bc_recovery") this is the FLAGSHIP covariate-shift
 * concept-toy: a REAL bc_policy.onnx drives a REAL MuJoCo-WASM PushT sim, and the
 * learner DRAGS the T-block out of the region the demonstrations covered to watch
 * a confident policy fail. Every other chapter renders a static poster until its
 * own toy lands.
 *
 * ============================================================================
 * FROZEN CONCEPT-TOY CONTRACT  (P5 replicates this per chapter — read before you
 * build the next four toys; the shape below is the reusable pattern)
 * ----------------------------------------------------------------------------
 * 1. SSR POSTER = JS-OFF FALLBACK.  The component server-renders a complete,
 *    captioned SVG poster of the NOMINAL state (`<Poster/>`). With JS disabled
 *    that poster is the whole experience; no WASM is ever fetched. Keep the
 *    poster and the live canvas in ONE square figure so booting causes no reflow.
 * 2. LAZY, HYDRATION-GATED SIM.  All heavy modules are pulled by *dynamic*
 *    import() INSIDE a post-hydration effect. Mounted `client:visible` (HeroDemo),
 *    the island — and therefore any WASM — is fetched only when scrolled into
 *    view. `optimizeDeps.exclude` in astro.config keeps @mujoco/mujoco +
 *    onnxruntime-web out of the entry graph.
 * 3. REUSE THE PRIMITIVES VERBATIM.  obs / env / policy come from playground/src
 *    unchanged (createSim, PUSHT_XML, BrowserPushTEnv, buildObs, loadPolicy +
 *    the fail-closed assertDrivesPushT gate). The driver loop mirrors main.ts:
 *        const obs = env.obs(); const a = await policy.act(obs); env.step(a);
 *    Only the RENDER is bespoke (display layer, explicitly flexible per the
 *    playground graceful-degradation ladder) so entities wear the page's
 *    --entity-* hues instead of the raw MJCF colours.
 * 4. MAKE THE INVISIBLE VISIBLE.  Overlay the honest training distribution
 *    (src/data/pusht_coverage.json — a radial envelope of block poses the real
 *    scripted-expert demos visited; provenance in that file + scripts/gen_coverage.py).
 *    NEVER an invented shape.
 * 5. ONE CONTROL, IMMEDIATE FEEDBACK, DEFAULT-INTERESTING.  Expose exactly one
 *    variable (drag the block). A live "distance from training data" meter reads
 *    off the same coverage data. Boot in-distribution (meter green), so the first
 *    drag-out is the aha. Provide a keyboard + button path to the same aha.
 * 6. COLOUR DISCIPLINE.  --entity-* for entities, ONE --signal blue for the
 *    live/interactive handle, --alert red for the failing readout, neutral ink
 *    for the coverage map. Nothing static wears signal blue.
 * ============================================================================
 */
import { useEffect, useRef, useState } from "preact/hooks";
import coverage from "../data/pusht_coverage.json";
// P5 per-chapter concept-toys (each is a self-contained island built to the
// FROZEN CONCEPT-TOY CONTRACT below; the toy's own SSR poster is the JS-off
// fallback, its WASM/heavy deps are lazy). Statically imported so each toy's
// poster server-renders; the heavy sim/policy stays dynamic-imported per toy.
import SimLoopPerturb from "./toys/sim-loop-perturb";
import PushTSceneBuild from "./toys/pusht-scene-build";
import FramesDrag from "./toys/frames-drag";
import PushTTeleopToy from "./toys/pusht-teleop";
import CartpolePpo from "./toys/cartpole-ppo";
// Phase-1 chapter concept-toys (2D data panels; each server-renders its own
// SSR poster as the JS-off fallback, data from the chapter's measured vizdata).
import EvalBandsToy from "./toys/EvalBandsToy";
import DiffusionRing from "./toys/diffusion-ring";
import DiffusionPushtLive from "./toys/DiffusionPushtLive";
import FlowRing from "./toys/flow-ring";
import FlowPushtLive from "./toys/FlowPushtLive";
import BridgeCompareToy from "./toys/BridgeCompareToy";
import ActChunkToy from "./toys/ActChunkToy";
import CurateQualityToy from "./toys/CurateQualityToy";
import VlaBrowserToy from "./toys/VlaBrowserToy";
import VlaRolloutToy from "./toys/VlaRolloutToy";

interface Props {
  demo?: string | null;
  task?: string | null;
  title?: string | null;
}

// ---------------------------------------------------------------- coverage data
// The honest training-coverage region + distance metric. Derived offline from
// real scripted-expert demonstration rollouts (the exact distribution
// gen_demos.py builds the BC training set from) — see the JSON's `provenance`.
const WORLD_HALF = coverage.world_half_extent_m; // 0.45 m — the ±extent worldToPx uses
const ENVELOPE: number[] = coverage.envelope_radius_m;
const K_BINS = coverage.k_bins;
const MODEL_URL = "/models/bc_policy.onnx";
const CANVAS_PX = 512;
const MAX_CONTROL_STEPS_PER_FRAME = 4; // mirror main.ts: cap control-step catch-up
const DRAG_BOUND = 0.35; // keep the dragged block inside the walls (±0.41)
const GRAB_RADIUS = 0.1; // m — pointer-to-block distance that starts a drag
const NUDGE_STEP = 0.03; // m — arrow-key block nudge
const METER_FULL = 0.13; // m that fills the meter — ~ the max reachable OOD distance
                         // (DRAG_BOUND 0.35 − min envelope ≈ 0.22)

/** distance-from-training metric, IDENTICAL to gen_coverage.py's Python version:
 *  0 inside the demonstrated envelope, else how far past it (m). Radial because
 *  the demos are star-shaped about the target at the origin (validated in the
 *  JSON: this matches brute nearest-neighbour to the visited cloud to ~0.5 cm). */
function distFromDemos(x: number, y: number): number {
  const r = Math.hypot(x, y);
  const theta = Math.atan2(y, x); // [-pi, pi)
  let b = Math.floor(((theta + Math.PI) / (2 * Math.PI)) * K_BINS);
  b = Math.max(0, Math.min(K_BINS - 1, b));
  return Math.max(0, r - ENVELOPE[b]);
}

/** The coverage envelope as a closed world-space polygon (block-xy meters). */
function envelopeWorld(): Array<[number, number]> {
  return ENVELOPE.map((r, b) => {
    const a = -Math.PI + (2 * Math.PI * (b + 0.5)) / K_BINS;
    return [r * Math.cos(a), r * Math.sin(a)] as [number, number];
  });
}

// ------------------------------------------------------------- shared SVG poster
// Square viewBox; world (x,y) → svg px with world +y up. Same ±WORLD_HALF frame
// as the live canvas, so the poster is a faithful preview (no reflow on boot).
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
      {/* stem hangs toward the block's -y (SVG down), matching the MJCF layout */}
      <rect x={-stemW / 2} y={0.06 * POSTER_S - stemH / 2} width={stemW} height={stemH} rx={2} />
    </g>
  );
}

/** The static, captioned poster — SSR output + the JS-off experience. */
function Poster() {
  const envPts = envelopeWorld().map(([x, y]) => w2s(x, y).map((n) => n.toFixed(1)).join(","));
  // nominal in-distribution demo pose (r≈0.15, inside the envelope)
  const blockX = 0.12, blockY = 0.09;
  const pusherX = 0.185, pusherY = 0.14; // just outside the block, pushing inward
  const [pcx, pcy] = w2s(pusherX, pusherY);
  const [bcx, bcy] = w2s(blockX, blockY);
  return (
    <svg
      class="ct-poster-svg"
      viewBox={`0 0 ${POSTER_V} ${POSTER_V}`}
      role="img"
      aria-label="Top-down PushT arena. A dashed region marks the block poses the demonstrations covered; the magenta T-block sits inside it near the amber pusher, with the green dashed target pose at the center. A blue halo marks the block as draggable out of distribution. When the live sim loads, the policy pushes the block toward the target; dragged outside the dashed region it fails."
    >
      <title>PushT behavior-cloning policy — covariate-shift toy</title>
      <desc>
        The demonstrated-coverage region (dashed) is the set of block positions the
        training demos actually visited, derived from real scripted-expert rollouts.
        Inside it the policy pushes the block onto the target; dragged outside it, the
        policy makes small, confident, wrong movements.
      </desc>

      {/* arena + graph paper */}
      <rect class="ct-arena" x={2} y={2} width={POSTER_V - 4} height={POSTER_V - 4} rx={6} />
      <g class="ct-grid">
        {Array.from({ length: 9 }, (_, i) => ((i + 1) * POSTER_V) / 10).map((v) => (
          <>
            <line x1={v} y1={2} x2={v} y2={POSTER_V - 2} />
            <line x1={2} y1={v} x2={POSTER_V - 2} y2={v} />
          </>
        ))}
      </g>

      {/* the demonstrated-coverage region — the training distribution, made visible */}
      <polygon class="ct-coverage" points={envPts.join(" ")} />
      <text class="ct-coverage-label" x={w2s(-0.23, -0.30)[0]} y={w2s(-0.23, -0.30)[1]}>
        where the demos pushed the block
      </text>

      {/* target pose (fixed at origin) */}
      <PosterTee x={0} y={0} yawDeg={0} className="ct-target" />
      <text class="ct-target-label" x={w2s(0.02, 0.02)[0]} y={w2s(0.02, 0.02)[1]}>target</text>

      {/* the block (in distribution) */}
      <PosterTee x={blockX} y={blockY} yawDeg={-22} className="ct-tee" />

      {/* the pusher */}
      <g transform={`translate(${pcx.toFixed(1)} ${pcy.toFixed(1)})`}>
        <circle class="ct-pusher-ring" r={0.028 * POSTER_S} />
        <circle class="ct-pusher-core" r={0.015 * POSTER_S} />
      </g>

      {/* the drag affordance — the one LIVE handle (signal blue) */}
      <g transform={`translate(${bcx.toFixed(1)} ${bcy.toFixed(1)})`}>
        <circle class="ct-halo" r={0.1 * POSTER_S} />
        <text class="ct-drag-label" x={0.11 * POSTER_S} y={-0.09 * POSTER_S}>drag me out of distribution →</text>
      </g>
    </svg>
  );
}

// ------------------------------------------------------------------- live island
interface Hud {
  phase: string;
  dist: number;
  posErr: number;
  ood: boolean;
  fps: number;
  latMs: number;
  pusherMoved: number;
  error?: string;
}

type PaletteKey =
  | "--entity-pusher" | "--entity-block" | "--entity-target"
  | "--signal" | "--alert" | "--ink-mute" | "--rule-strong";
const PALETTE_FALLBACK: Record<PaletteKey, string> = {
  "--entity-pusher": "#b0560f",
  "--entity-block": "#a5257d",
  "--entity-target": "#0c7d5f",
  "--signal": "#1f56de",
  "--alert": "#c0362a",
  "--ink-mute": "#6d6252",
  "--rule-strong": "#c8bc9e",
};

/** ch1.1 — the interactive covariate-shift toy (SSR poster + live MuJoCo island). */
function PushTConceptToy() {
  const figureRef = useRef<HTMLElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const apiRef = useRef<{
    reset: () => void;
    sendOOD: () => void;
    nudge: (dx: number, dy: number) => void;
  } | null>(null);
  const [booted, setBooted] = useState(false);
  const [hud, setHud] = useState<Hud>({
    phase: "loading…", dist: 0, posErr: 0, ood: false, fps: 0, latMs: 0, pusherMoved: 0,
  });

  useEffect(() => {
    let disposed = false;
    let sim: { dispose(): void } | null = null;
    let raf = 0;
    const nextFrame = () => new Promise<number>((r) => (raf = requestAnimationFrame(r)));

    (async () => {
      try {
        // --- lazy, hydration-gated source-sharing: the WASM modules are only
        //     fetched now (post-hydration, i.e. scrolled into view) --------------
        const [simMod, sceneMod, envMod, obsMod, inferMod, contractsMod, vpMod] =
          await Promise.all([
            import("../../../playground/src/sim/mujoco_sim"),
            import("../../../playground/src/sim/scene"),
            import("../../../playground/src/teleop/pusht_env"),
            import("../../../playground/src/teleop/pusht_obs"),
            import("../../../playground/src/policy/infer"),
            import("../../../playground/src/policy/contracts"),
            import("../../../playground/src/teleop/viewport"),
          ]);
        const { createSim } = simMod;
        const { PUSHT_XML } = sceneMod;
        const { BrowserPushTEnv } = envMod;
        const { buildObs, CONTROL_DT } = obsMod;
        const { loadPolicy } = inferMod;
        const { assertDrivesPushT } = contractsMod;
        const { worldToPx, eventToWorld } = vpMod;

        // 1) boot the REAL PushT scene + env
        const realSim: any = await createSim(PUSHT_XML);
        sim = realSim;
        if (disposed) { realSim.dispose(); return; }
        const env: any = new BrowserPushTEnv(realSim);
        let seed = 7;
        env.reset(seed); // default-interesting: a fresh in-distribution start

        const canvas = canvasRef.current!;
        canvas.width = CANVAS_PX;
        canvas.height = CANVAS_PX;
        const ctx = canvas.getContext("2d")!;

        // resolve the page's design tokens once (canvas can't read CSS vars live)
        const cs = getComputedStyle(canvas);
        const col = (k: PaletteKey) => (cs.getPropertyValue(k).trim() || PALETTE_FALLBACK[k]);
        const PUSHER = col("--entity-pusher"), BLOCK = col("--entity-block"),
          TARGET = col("--entity-target"), SIGNAL = col("--signal"),
          INK = col("--ink-mute"), RULE = col("--rule-strong");

        // --- bespoke top-down renderer: entities in the page's --entity-* hues ---
        const envPts = envelopeWorld();
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
        const drawTee = (cx: number, cy: number, yaw: number, style: string) => {
          fillPoly(teeCorners(cx, cy, yaw, 0.06, 0.015, 0, 0), style);
          fillPoly(teeCorners(cx, cy, yaw, 0.015, 0.045, 0, -0.06), style);
        };
        const render = (dragging: boolean) => {
          const w = canvas.width, h = canvas.height;
          // arena background (warm paper) + faint graph paper
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

          // the demonstrated-coverage region (neutral ink — a map, not an entity)
          const covPx = envPts.map(([x, y]) => worldToPx(canvas, x, y));
          ctx.save();
          ctx.globalAlpha = 0.12;
          fillPoly(covPx, INK);
          ctx.restore();
          ctx.save();
          ctx.setLineDash([6, 4]);
          ctx.globalAlpha = 0.72;
          ctx.strokeStyle = INK;
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          covPx.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
          ctx.closePath();
          ctx.stroke();
          ctx.restore();

          // target (fixed at origin) — dashed emerald outline
          const tgt = [teeCorners(0, 0, 0, 0.06, 0.015, 0, 0), teeCorners(0, 0, 0, 0.015, 0.045, 0, -0.06)];
          ctx.save();
          ctx.setLineDash([5, 4]);
          ctx.strokeStyle = TARGET;
          ctx.lineWidth = 2;
          tgt.forEach((pts) => {
            ctx.beginPath();
            pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
            ctx.closePath();
            ctx.stroke();
          });
          ctx.restore();

          // block (tee) — magenta
          const tx = realSim.jointQpos("tee_x"), ty = realSim.jointQpos("tee_y"),
            tyaw = realSim.jointQpos("tee_yaw");
          drawTee(tx, ty, tyaw, BLOCK);

          // grab halo around the block — the one LIVE handle (signal blue)
          const [bpx, bpy] = worldToPx(canvas, tx, ty);
          if (!dragging) {
            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = SIGNAL;
            ctx.globalAlpha = 0.8;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(bpx, bpy, GRAB_RADIUS * (canvas.width / (2 * WORLD_HALF)), 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
          }

          // pusher (agent) — amber
          const [px, py] = worldToPx(canvas, realSim.jointQpos("pusher_x"), realSim.jointQpos("pusher_y"));
          const rPx = 0.015 * (canvas.width / (2 * WORLD_HALF));
          ctx.save();
          ctx.globalAlpha = 0.5;
          ctx.strokeStyle = PUSHER;
          ctx.lineWidth = 1.5;
          ctx.beginPath();
          ctx.arc(px, py, rPx * 1.9, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
          ctx.fillStyle = PUSHER;
          ctx.beginPath();
          ctx.arc(px, py, rPx, 0, Math.PI * 2);
          ctx.fill();
          void RULE;
        };

        setBooted(true);
        render(false);

        // 2) load the REAL policy through the fail-closed contract gate
        const policy = await loadPolicy(MODEL_URL);
        assertDrivesPushT(policy.contract);
        if (disposed) return;

        // --- interaction state (refs, not React state — the loop must not re-render)
        let dragging = false;
        const clampB = (v: number) => Math.max(-DRAG_BOUND, Math.min(DRAG_BOUND, v));
        const setBlock = (x: number, y: number) => {
          env.perturbBlock(clampB(x), clampB(y), realSim.jointQpos("tee_yaw"));
        };
        const nearBlock = (wx: number, wy: number) =>
          Math.hypot(wx - realSim.jointQpos("tee_x"), wy - realSim.jointQpos("tee_y")) < GRAB_RADIUS;

        const onDown = (e: PointerEvent) => {
          const [wx, wy] = eventToWorld(canvas, e.clientX, e.clientY);
          if (!nearBlock(wx, wy)) return;
          dragging = true;
          canvas.dataset.dragging = "true";
          canvas.setPointerCapture(e.pointerId);
          setBlock(wx, wy);
          e.preventDefault();
        };
        const onMove = (e: PointerEvent) => {
          if (!dragging) return;
          const [wx, wy] = eventToWorld(canvas, e.clientX, e.clientY);
          setBlock(wx, wy);
        };
        const onUp = (e: PointerEvent) => {
          if (!dragging) return;
          dragging = false;
          canvas.dataset.dragging = "false";
          try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
        };
        canvas.addEventListener("pointerdown", onDown);
        canvas.addEventListener("pointermove", onMove);
        canvas.addEventListener("pointerup", onUp);
        canvas.addEventListener("pointercancel", onUp);

        // button/keyboard API (the no-drag path to the same aha)
        apiRef.current = {
          reset: () => env.reset(++seed),
          sendOOD: () => setBlock(0.33 * Math.cos(0.7), 0.33 * Math.sin(0.7)), // r=0.33, well past the ~0.234 envelope, block stays in-frame
          nudge: (dx, dy) => setBlock(realSim.jointQpos("tee_x") + dx, realSim.jointQpos("tee_y") + dy),
        };

        // 3) headless-verification hooks (mirror playground's window.__policy) so a
        //    browser driver can PROVE the policy drives + the metric is live.
        (window as any).__toy = {
          contract: () => ({ ...policy.contract }),
          obsParity: () => {
            const a = env.obs(); const b = buildObs(realSim);
            let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k]));
            return { equal: m === 0, maxErr: m };
          },
          blockXY: () => [realSim.jointQpos("tee_x"), realSim.jointQpos("tee_y")],
          distFromDemos: () => distFromDemos(realSim.jointQpos("tee_x"), realSim.jointQpos("tee_y")),
          posErr: () => env.errors().posErr,
          isDragging: () => dragging,
          sendOOD: () => apiRef.current?.sendOOD(),
          reset: () => apiRef.current?.reset(),
          async drive(n: number) {
            const before = env.obs();
            for (let i = 0; i < n; i++) { const o = env.obs(); env.step(await policy.act(o)); }
            const after = env.obs();
            return {
              steps: n,
              pusherMoved: Math.hypot(after[0] - before[0], after[1] - before[1]),
              meanLatencyMs: policy.meanLatencyMs(), calls: policy.calls,
            };
          },
          fps: () => lastFps,
        };

        // 4) DRIVE — real-time paced to CONTROL_HZ (mirrors runPolicyDriver). While
        //    dragging, the policy pauses so the block tracks the pointer; on release
        //    it resumes and commits to its (now out-of-distribution) strokes.
        const startPusher: [number, number] = [realSim.jointQpos("pusher_x"), realSim.jointQpos("pusher_y")];
        let lastFps = 0, frames = 0, fpsMark = performance.now(), last = performance.now(), acc = 0, hudMark = 0;

        while (!disposed) {
          await nextFrame();
          const now = performance.now();
          frames++;
          if (now - fpsMark >= 500) { lastFps = (frames * 1000) / (now - fpsMark); frames = 0; fpsMark = now; }

          if (dragging) {
            acc = 0; last = now;
          } else {
            acc += Math.min(now - last, 100) / 1000;
            last = now;
            let n = 0;
            while (acc >= CONTROL_DT && n < MAX_CONTROL_STEPS_PER_FRAME) {
              const obs = env.obs();                 // buildObs(sim) — training obs, verbatim
              const action = await policy.act(obs);  // raw action[2]; denorm baked in
              const res = env.step(action);          // mirrors bc.py eval: env.step(policy(obs))
              acc -= CONTROL_DT; n += 1;
              if (res.done) { env.reset(++seed); }    // ordinary starts re-seed in-distribution
              if (disposed) break;
            }
            if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0;
          }

          render(dragging);

          if (now - hudMark >= 100) {                // throttle HUD to ~10 Hz
            hudMark = now;
            const tx = realSim.jointQpos("tee_x"), ty = realSim.jointQpos("tee_y");
            const dist = distFromDemos(tx, ty);
            const { posErr } = env.errors();
            setHud({
              phase: dragging ? "dragging" : "driving",
              dist, posErr, ood: dist > 1e-4, fps: lastFps,
              latMs: policy.meanLatencyMs(),
              pusherMoved: Math.hypot(
                realSim.jointQpos("pusher_x") - startPusher[0],
                realSim.jointQpos("pusher_y") - startPusher[1],
              ),
            });
          }
        }

        canvas.removeEventListener("pointerdown", onDown);
        canvas.removeEventListener("pointermove", onMove);
        canvas.removeEventListener("pointerup", onUp);
        canvas.removeEventListener("pointercancel", onUp);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch1.1 toy] failed", err);
        setHud((h) => ({ ...h, phase: "failed", error: msg }));
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); if (sim) sim.dispose(); };
  }, []);

  const failed = !!hud.error;
  const meterPct = Math.min(100, (hud.dist / METER_FULL) * 100);

  const onKeyDown = (e: KeyboardEvent) => {
    const api = apiRef.current;
    if (!api) return;
    const map: Record<string, [number, number]> = {
      ArrowUp: [0, NUDGE_STEP], ArrowDown: [0, -NUDGE_STEP],
      ArrowLeft: [-NUDGE_STEP, 0], ArrowRight: [NUDGE_STEP, 0],
    };
    if (e.key in map) { e.preventDefault(); api.nudge(...map[e.key]); }
    else if (e.key === "r" || e.key === "R") api.reset();
    else if (e.key === "o" || e.key === "O") api.sendOOD();
  };

  return (
    <div class="ct">
      <figure
        ref={figureRef}
        class="ct-figure"
        tabIndex={0}
        role="application"
        aria-label="Interactive PushT covariate-shift toy. The policy drives the block toward the target. Grab the block with the pointer, or focus here and use the arrow keys, to move it out of the demonstrated region and watch the policy fail. Press R to reset, O to send it out of distribution."
        onKeyDown={onKeyDown}
      >
        {/* SSR poster — the JS-off experience and the pre-boot frame */}
        <div class="ct-poster" hidden={booted}><Poster /></div>

        {/* live MuJoCo-WASM canvas — shown once booted */}
        <canvas ref={canvasRef} class="ct-canvas" hidden={!booted} aria-hidden="true" />

        {/* Non-visual path to the same aha: announce only the qualitative
            in/out-of-distribution transition (not the per-frame distance, which
            would spam the screen reader). The visual HUD stays aria-hidden. */}
        <div class="bk-sr" aria-live="polite">
          {booted && !failed
            ? hud.ood
              ? "Out of distribution: the block is past the region the demonstrations covered, and the policy commits confidently to the wrong move, far from the block."
              : "In distribution: the policy pushes the block toward the target."
            : ""}
        </div>

        {/* live HUD (the poster ships the nominal in-distribution readout below) */}
        {booted && !failed && (
          <div class="ct-hud" aria-hidden="true">
            <div class="ct-hud-row">
              <span class="ct-k">distance from demos</span>
              <span class={`ct-v ${hud.ood ? "ct-bad" : "ct-ok"}`}>
                {hud.dist.toFixed(3)} m {hud.ood ? "▲" : "✓"}
              </span>
            </div>
            <div class="ct-meter">
              <div class="ct-meter-fill" data-ood={hud.ood} style={`width:${meterPct}%`} />
            </div>
            <div class="ct-hud-row">
              <span class="ct-k">policy</span>
              <span class={`ct-v ${hud.ood ? "ct-bad" : "ct-ok"}`}>
                {hud.ood ? "out of distribution" : "in distribution"}
              </span>
            </div>
            <div class="ct-hud-row">
              <span class="ct-k">pos_err → target</span>
              <span class="ct-v">{hud.posErr.toFixed(3)} m</span>
            </div>
          </div>
        )}

        {/* boot / instrument status line */}
        <div class="ct-status" data-failed={failed} aria-hidden="true">
          {failed ? (
            <span>sim failed — the Colab path covers this without WASM</span>
          ) : booted ? (
            <>
              <span>real bc_policy.onnx · pusher moved {hud.pusherMoved.toFixed(2)} m</span>
              <span>{hud.fps.toFixed(0)} fps · {hud.latMs.toFixed(2)} ms/call</span>
            </>
          ) : (
            <span>booting MuJoCo-WASM + policy…</span>
          )}
        </div>
      </figure>

      <div class="ct-controls">
        <button type="button" class="ct-btn ct-btn--primary" onClick={() => apiRef.current?.sendOOD()} disabled={!booted || failed}>
          send it out of distribution →
        </button>
        <button type="button" class="ct-btn" onClick={() => apiRef.current?.reset()} disabled={!booted || failed}>
          reset · put it back
        </button>
        <span class="ct-control-note">
          drag the block (or arrow-keys when focused) · poster reads with JS off
        </span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------- non-BC posters
/** A PushT "T", two rects centred on (0,0); the parent <g> positions it. */
function Tee({ className }: { className: string }) {
  return (
    <g class={className}>
      <rect x={-44} y={-16} width={88} height={23} rx={3} />
      <rect x={-11} y={7} width={22} height={52} rx={3} />
    </g>
  );
}

/** Faint instrument graph paper, shared by the generic posters. */
function GraphPaper() {
  return (
    <g class="bk-grid">
      {Array.from({ length: 10 }, (_, i) => 40 + (i + 1) * 65.4).map((x) => (
        <line x1={x} y1={40} x2={x} y2={460} />
      ))}
      {Array.from({ length: 5 }, (_, i) => 40 + (i + 1) * 70).map((y) => (
        <line x1={40} y1={y} x2={760} y2={y} />
      ))}
    </g>
  );
}

/** Every other chapter — a tasteful static poster keyed off task/demo. */
function GenericPoster({ demo, task }: { demo?: string | null; task?: string | null }) {
  const pusht = task === "pusht";
  const label = pusht ? "pusht · top-down · normalized frame" : `${task ?? "scene"} · top-down`;
  return (
    <div class="bk-stage" data-state="poster">
      <figure class="bk-figure">
        <svg class="bk-svg" viewBox="0 0 800 500" role="img"
          aria-label={
            pusht
              ? "Top-down PushT arena poster: the amber pusher, the magenta T-block, and a dashed green target pose. The interactive sim lands in a later pass."
              : "Top-down arena poster: a free box and the amber slide-actuated pusher. The interactive sim lands in a later pass."
          }
        >
          <title>See it work — poster</title>
          <rect class="bk-arena" x={40} y={40} width={720} height={420} rx={10} />
          <GraphPaper />
          <text class="bk-axis" x={214} y={64}>{label}</text>

          {pusht ? (
            <>
              <g style="transform: translate(470px, 250px)">
                <Tee className="bk-target" />
                <text class="bk-axis bk-target-label" x={-16} y={92}>target</text>
              </g>
              <path class="bk-path" d="M 250 360 Q 320 320 360 300 T 470 250" />
              <g style="transform: translate(360px, 300px) rotate(-14deg)">
                <Tee className="bk-tee" />
              </g>
            </>
          ) : (
            <rect class="bk-tee" x={352} y={276} width={70} height={48} rx={4} style="transform: none" />
          )}

          <g style="transform: translate(250px, 360px)" class="bk-pusher">
            <circle class="bk-pusher-ring" cx={0} cy={0} r={19} />
            <circle class="bk-pusher-core" cx={0} cy={0} r={10.5} />
          </g>
        </svg>
      </figure>

      <div class="bk-poster-cap">
        <span class="bk-poster-demo">demo <b>{demo ?? "—"}</b></span>
        <span class="bk-poster-p2">interactive sim → P2</span>
      </div>
    </div>
  );
}

export default function PlateIsland({ demo, task }: Props) {
  // Each chapter's "See it work" hero dispatches to its concept-tuned toy by
  // demo id; unknown demos fall back to the static poster.
  if (demo === "pusht_bc_recovery") return <PushTConceptToy />;      // ch1.1 covariate shift (flagship)
  if (demo === "sim-loop-perturb") return <SimLoopPerturb />;        // ch0.1 timestep instability
  if (demo === "pusht-scene-build") return <PushTSceneBuild />;      // ch0.2 weld vs two bodies
  if (demo === "frames-drag") return <FramesDrag />;                 // ch0.3 frames / quaternion convention
  if (demo === "pusht-teleop") return <PushTTeleopToy />;            // ch0.4 record → scrub the episode
  if (demo === "pusht_curate_quality") return <CurateQualityToy />;  // ch1.2 curate: better data, not more
  if (demo === "aloha_cube_chunk") return <ActChunkToy />;          // ch1.3 chunking + temporal ensemble
  // ch1.4 diffusion: the 2D diffusion-ring (mode collapse, pure 2D) stacked with
  // the LIVE generative-policy PushT panel it was blocked on (now unblocked by the
  // contract-v2 ddpm sampler runtime). The live panel is also addressable on its
  // own id (diffusion_pusht_live) for embeds/headless drivers.
  if (demo === "diffusion_multimodality")
    return (
      <div class="df-stack">
        <DiffusionRing />
        <DiffusionPushtLive />
      </div>
    );
  if (demo === "diffusion_pusht_live") return <DiffusionPushtLive />; // ch1.4 live DDPM sampler on PushT-WASM
  // ch1.5 flow: the 2D flow-ring (the objective/step-efficiency aha, pure 2D)
  // stacked with the LIVE generative-policy PushT panel it was blocked on (now
  // unblocked by the contract-v2 sampler runtime). The live panel is also
  // addressable on its own id (flow_pusht_live) for embeds/headless drivers.
  if (demo === "flow_multimodality")
    return (
      <div class="fl-stack">
        <FlowRing />
        <FlowPushtLive />
      </div>
    );
  if (demo === "flow_pusht_live") return <FlowPushtLive />;         // ch1.5 live flow sampler on PushT-WASM
  if (demo === "harness_eval_bands") return <EvalBandsToy />;       // ch1.6 single-numbers-lie CI slider
  if (demo === "vla_dataset_browser") return <VlaBrowserToy />;     // ch1.7 VLA data browser + language-leak
  if (demo === "vla_rollout") return <VlaRolloutToy />;            // ch1.8 recorded VLA rollout + fusion attention
  if (demo === "bridge_comparison") return <BridgeCompareToy />;    // ch1.9 bridge: 380-vs-12 + two-ACT gap
  if (demo === "cartpole_ppo_balance") return <CartpolePpo />;      // ch2.1 PPO: shove the cart → it recovers
  return <GenericPoster demo={demo} task={task} />;
}
