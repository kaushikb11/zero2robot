/**
 * ch2.2 "SAC and the Off-Policy Bargain" — the LIVE pusher-reach toy
 * (`demo: sac_reach_live`, also stacked under `pusher_sac_reach` beneath the
 * recorded bargain chart in SacReachToy).
 *
 * The recorded SacReachToy replays ONE held-out reach + the sample-efficiency
 * headline. THIS panel makes it live and interactive: a REAL sac_actor.onnx (the
 * trained deterministic policy — tanh of the mean, sac.py's own eval action)
 * drives a REAL MuJoCo-WASM two-link arm, and the visitor DRAGS the green target
 * around the workspace to watch the arm re-reach. Honest by construction: the
 * policy trained on exactly such fingertip→target vectors sampled across the
 * annulus, so chasing a dragged target is in-distribution — and it is a FREE-TIER
 * policy (eval success ~0.4 at 30k steps), so it sometimes lands just outside the
 * 2 cm ring rather than dead-centre. That honesty is the point, not a polished win.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT at the top of ../PlateIsland.tsx:
 *   1. SSR poster == the JS-off fallback (<Poster/> is pure static JSX).
 *   2. Lazy, hydration-gated sim: MuJoCo-WASM + onnxruntime pulled by *dynamic*
 *      import() inside the post-hydration effect (mounted client:visible).
 *   3. Reuse the primitives verbatim: createSim + PUSHER_REACH_XML +
 *      BrowserPusherReachEnv + buildObs + loadPolicy through a fail-closed
 *      obs[8]/act[2] contract gate. The driver loop mirrors sac.py's eval:
 *        obs = env.obs(); action = await policy.act(obs); env.step(action)
 *   4. Make the invisible visible: the fingertip→target distance ticks down as the
 *      arm homes; the success ring goes green when it lands inside 2 cm.
 *   5. ONE control (drag the target), immediate feedback, default-interesting
 *      (boots reaching a fresh target). Keyboard + button path to the same aha.
 *   6. Colour discipline: entities in stable brand hues, ONE --signal blue for the
 *      draggable target handle, --entity-target green for the reached readout.
 */
import "./SacReachLive.css";
import { useEffect, useRef, useState } from "preact/hooks";

const MODEL_URL = "/models/sac_actor.onnx";
const CANVAS_PX = 440;
const WORLD_HALF = 0.24; // arm reach is 0.2 m; a little margin around the workspace
const MAX_CONTROL_STEPS_PER_FRAME = 4;
const SUCCESS_TOL = 0.02; // m — the success ring (PusherReachEnv.SUCCESS_TOL)
const REACH = 0.2; // m — 2 * LINK_LEN, the reachable-workspace radius
const RANDOM_DIST = 0.176; // m — the random-policy baseline (for the distance meter)
const NUDGE = 0.02; // m — arrow-key target nudge

// entity hues (literal, so the instrument reads identically in light + dark —
// this stage is a hardcoded-light lab instrument like the other live toys).
const INK = "#6d6252";
const ARM = "#1f56de"; // the SAC policy embodied (signal blue)
const ARM_GHOST = "rgba(31,86,222,0.28)";
const TARGET = "#0c7d5f"; // the goal (green)
const BASE = "#473f34";

// ---------------------------------------------------------- shared SSR poster
const POSTER_V = 300;
const POSTER_S = POSTER_V / 2 / WORLD_HALF;
const pw2s = (x: number, y: number): [number, number] => [
  POSTER_V / 2 + x * POSTER_S,
  POSTER_V / 2 - y * POSTER_S,
];

function Poster() {
  const [bx, by] = pw2s(0, 0);
  // a nominal reaching pose (elbow-down) toward a target up-right
  const sh = 0.5, el = 0.9;
  const ex = 0.1 * Math.cos(sh), ey = 0.1 * Math.sin(sh);
  const fx = ex + 0.1 * Math.cos(sh + el), fy = ey + 0.1 * Math.sin(sh + el);
  const [ex_, ey_] = pw2s(ex, ey);
  const [fx_, fy_] = pw2s(fx, fy);
  const [tx, ty] = pw2s(0.14, 0.12);
  return (
    <svg
      class="sl-poster-svg"
      viewBox={`0 0 ${POSTER_V} ${POSTER_V}`}
      role="img"
      aria-label="Top-down pusher-reach arena. A two-link arm anchored at the centre points toward a green target inside its reach ring. With JavaScript on, a SAC-trained policy drives the fingertip onto the target; drag the target and the arm re-reaches."
    >
      <title>SAC pusher-reach — a trained policy reaches a draggable target, live</title>
      <desc>
        A two-link arm driven by a trained SAC policy homes its fingertip onto a green
        target. Live, you can drag the target anywhere in the arm's reach and watch it
        re-reach; the dashed ring is the 2 cm success tolerance.
      </desc>
      <rect class="sl-arena" x={2} y={2} width={POSTER_V - 4} height={POSTER_V - 4} rx={6} />
      <g class="sl-grid">
        {Array.from({ length: 7 }, (_, i) => ((i + 1) * POSTER_V) / 8).map((v) => (
          <>
            <line x1={v} y1={2} x2={v} y2={POSTER_V - 2} />
            <line x1={2} y1={v} x2={POSTER_V - 2} y2={v} />
          </>
        ))}
      </g>
      {/* reachable workspace */}
      <circle class="sl-reach" cx={bx} cy={by} r={REACH * POSTER_S} />
      {/* target: success ring + dot */}
      <circle class="sl-tol" cx={tx} cy={ty} r={SUCCESS_TOL * POSTER_S} />
      <circle class="sl-target-dot" cx={tx} cy={ty} r={4} />
      <text class="sl-target-lbl" x={tx + SUCCESS_TOL * POSTER_S + 5} y={ty + 3}>drag me →</text>
      {/* arm base → elbow → fingertip */}
      <polyline class="sl-arm" points={`${bx},${by} ${ex_},${ey_} ${fx_},${fy_}`} />
      <circle class="sl-base" cx={bx} cy={by} r={4} />
      <circle class="sl-joint" cx={ex_} cy={ey_} r={3.2} />
      <circle class="sl-ftip" cx={fx_} cy={fy_} r={4.2} />
    </svg>
  );
}

// ------------------------------------------------------------------- live island
interface Hud {
  dist: number;
  reached: boolean;
  fps: number;
  latMs: number;
  error?: string;
}

function SacToy() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const figureRef = useRef<HTMLElement>(null);
  const apiRef = useRef<{
    reset: () => void;
    newTarget: () => void;
    nudge: (dx: number, dy: number) => void;
  } | null>(null);
  const [booted, setBooted] = useState(false);
  const [hud, setHud] = useState<Hud>({ dist: RANDOM_DIST, reached: false, fps: 0, latMs: 0 });

  useEffect(() => {
    let disposed = false;
    let sim: { dispose(): void } | null = null;
    let raf = 0;
    const nextFrame = () => new Promise<number>((r) => (raf = requestAnimationFrame(r)));

    (async () => {
      try {
        const prefersReducedMotion =
          typeof window !== "undefined" &&
          typeof window.matchMedia === "function" &&
          window.matchMedia("(prefers-reduced-motion: reduce)").matches;

        const [simMod, sceneMod, envMod, obsMod, inferMod] = await Promise.all([
          import("../../../../playground/src/sim/mujoco_sim"),
          import("../../../../playground/src/sim/scene"),
          import("../../../../playground/src/teleop/pusher_reach_env"),
          import("../../../../playground/src/teleop/pusher_reach_obs"),
          import("../../../../playground/src/policy/infer"),
        ]);
        const { createSim } = simMod;
        const { PUSHER_REACH_XML } = sceneMod;
        const { BrowserPusherReachEnv } = envMod;
        const { buildObs, CONTROL_DT, LINK_LEN } = obsMod;
        const { loadPolicy } = inferMod;

        // 1) boot the REAL pusher-reach scene + env
        const realSim: any = await createSim(PUSHER_REACH_XML);
        sim = realSim;
        if (disposed) { realSim.dispose(); return; }
        const env: any = new BrowserPusherReachEnv(realSim);
        let seed = 3;
        env.reset(seed); // default-interesting: a fresh reach

        const canvas = canvasRef.current!;
        canvas.width = CANVAS_PX;
        canvas.height = CANVAS_PX;
        const ctx = canvas.getContext("2d")!;
        const S = canvas.width / 2 / WORLD_HALF;
        const w2s = (x: number, y: number): [number, number] => [
          canvas.width / 2 + x * S,
          canvas.height / 2 - y * S,
        ];
        const eventToWorld = (clientX: number, clientY: number): [number, number] => {
          const r = canvas.getBoundingClientRect();
          const px = ((clientX - r.left) / r.width) * canvas.width;
          const py = ((clientY - r.top) / r.height) * canvas.height;
          return [(px - canvas.width / 2) / S, -(py - canvas.height / 2) / S];
        };

        const armXY = () => {
          const shoulder = realSim.jointQpos("shoulder");
          const elbow = realSim.jointQpos("elbow");
          const ex = LINK_LEN * Math.cos(shoulder), ey = LINK_LEN * Math.sin(shoulder);
          const fx = ex + LINK_LEN * Math.cos(shoulder + elbow);
          const fy = ey + LINK_LEN * Math.sin(shoulder + elbow);
          return { ex, ey, fx, fy };
        };

        const render = (dragging: boolean) => {
          const w = canvas.width, h = canvas.height;
          ctx.fillStyle = "#fbf9f3";
          ctx.fillRect(0, 0, w, h);
          ctx.strokeStyle = "rgba(200,188,158,0.5)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let i = 1; i < 8; i++) {
            const p = (i * w) / 8;
            ctx.moveTo(p, 0); ctx.lineTo(p, h);
            ctx.moveTo(0, p); ctx.lineTo(w, p);
          }
          ctx.stroke();

          const [bx, by] = w2s(0, 0);
          // reachable workspace ring
          ctx.save();
          ctx.setLineDash([3, 5]);
          ctx.globalAlpha = 0.5;
          ctx.strokeStyle = INK;
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.arc(bx, by, REACH * S, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();

          // target: success ring + dot + draggable halo (signal blue)
          const [txw, tyw] = env.target;
          const [tx, ty] = w2s(txw, tyw);
          ctx.save();
          ctx.setLineDash([4, 3]);
          ctx.strokeStyle = TARGET;
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.arc(tx, ty, SUCCESS_TOL * S, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
          ctx.fillStyle = TARGET;
          ctx.beginPath();
          ctx.arc(tx, ty, 4.5, 0, Math.PI * 2);
          ctx.fill();
          if (!dragging) {
            ctx.save();
            ctx.setLineDash([4, 4]);
            ctx.strokeStyle = ARM;
            ctx.globalAlpha = 0.75;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(tx, ty, 0.045 * S, 0, Math.PI * 2);
            ctx.stroke();
            ctx.restore();
          }

          // the arm: base → elbow → fingertip (signal blue — the SAC policy)
          const { ex, ey, fx, fy } = armXY();
          const [exp, eyp] = w2s(ex, ey);
          const [fxp, fyp] = w2s(fx, fy);
          ctx.strokeStyle = ARM;
          ctx.lineWidth = 4;
          ctx.lineCap = "round";
          ctx.beginPath();
          ctx.moveTo(bx, by); ctx.lineTo(exp, eyp); ctx.lineTo(fxp, fyp);
          ctx.stroke();
          ctx.fillStyle = BASE;
          ctx.beginPath(); ctx.arc(bx, by, 4.5, 0, Math.PI * 2); ctx.fill();
          ctx.fillStyle = ARM;
          ctx.beginPath(); ctx.arc(exp, eyp, 3.5, 0, Math.PI * 2); ctx.fill();
          const dist = Math.hypot(txw - fx, tyw - fy);
          ctx.fillStyle = dist < SUCCESS_TOL ? TARGET : ARM;
          ctx.beginPath(); ctx.arc(fxp, fyp, 5, 0, Math.PI * 2); ctx.fill();
          void ARM_GHOST;
        };

        // 2) load the REAL policy — fail-closed obs[8]/act[2] contract gate
        const policy = await loadPolicy(MODEL_URL);
        if (policy.contract.obsDim !== 8 || policy.contract.actDim !== 2) {
          throw new Error(
            `sac_actor.onnx contract mismatch: expected obs[8]/act[2] (pusher-reach), ` +
            `got obs[${policy.contract.obsDim}]/act[${policy.contract.actDim}].`,
          );
        }
        if (disposed) return;

        // Policy loaded + contract passed: only NOW reveal the canvas and paint the
        // first still frame. A fetch/contract failure throws above with booted still
        // false, so the captioned SSR poster stays visible (fail-closed) instead of a
        // frozen canvas.
        setBooted(true);
        render(false);

        // --- interaction: drag the green target (the one live handle) -----------
        let dragging = false;
        const nearTarget = (wx: number, wy: number) => {
          const [txw, tyw] = env.target;
          return Math.hypot(wx - txw, wy - tyw) < 0.05;
        };
        const onDown = (e: PointerEvent) => {
          const [wx, wy] = eventToWorld(e.clientX, e.clientY);
          if (!nearTarget(wx, wy)) return;
          dragging = true;
          canvas.setPointerCapture(e.pointerId);
          env.setTarget(wx, wy);
          e.preventDefault();
        };
        const onMove = (e: PointerEvent) => {
          if (!dragging) return;
          const [wx, wy] = eventToWorld(e.clientX, e.clientY);
          env.setTarget(wx, wy);
        };
        const onUp = (e: PointerEvent) => {
          if (!dragging) return;
          dragging = false;
          try { canvas.releasePointerCapture(e.pointerId); } catch { /* already released */ }
        };
        canvas.addEventListener("pointerdown", onDown);
        canvas.addEventListener("pointermove", onMove);
        canvas.addEventListener("pointerup", onUp);
        canvas.addEventListener("pointercancel", onUp);

        apiRef.current = {
          reset: () => { seed += 1; env.reset(seed); },
          newTarget: () => {
            const phi = Math.random() * 2 * Math.PI;
            const r = 0.07 + Math.random() * 0.11;
            env.setTarget(r * Math.cos(phi), r * Math.sin(phi));
          },
          nudge: (dx, dy) => {
            const [txw, tyw] = env.target;
            env.setTarget(txw + dx, tyw + dy);
          },
        };

        // 3) headless-verification hooks (mirror playground's window.__policy).
        (window as any).__toy = {
          contract: () => ({ ...policy.contract }),
          obsParity: () => {
            const a = env.obs();
            const [txw, tyw] = env.target;
            const b = buildObs(realSim, txw, tyw);
            let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k]));
            return { equal: m === 0, maxErr: m };
          },
          dist: () => env.dist(),
          target: () => [...env.target],
          setTarget: (x: number, y: number) => env.setTarget(x, y),
          reset: () => apiRef.current?.reset(),
          newTarget: () => apiRef.current?.newTarget(),
          async drive(n: number) {
            const before = env.dist();
            for (let i = 0; i < n; i++) { const o = env.obs(); env.step(await policy.act(o)); }
            return {
              steps: n, distBefore: before, distAfter: env.dist(),
              meanLatencyMs: policy.meanLatencyMs(), calls: policy.calls,
            };
          },
          fps: () => lastFps,
        };

        // 4) DRIVE — real-time paced to CONTROL_HZ (mirrors sac.py eval). No reset on
        //    truncation: the arm holds at the target and re-reaches when it is dragged.
        let lastFps = 0, frames = 0, fpsMark = performance.now(), last = performance.now(), acc = 0, hudMark = 0;

        if (prefersReducedMotion) return; // reduced motion: one still frame already painted; do not spin the auto-driving loop. Interaction/reset handlers + __toy stay live.

        while (!disposed) {
          await nextFrame();
          const now = performance.now();
          frames++;
          if (now - fpsMark >= 500) { lastFps = (frames * 1000) / (now - fpsMark); frames = 0; fpsMark = now; }

          acc += Math.min(now - last, 100) / 1000;
          last = now;
          let n = 0;
          while (acc >= CONTROL_DT && n < MAX_CONTROL_STEPS_PER_FRAME) {
            const obs = env.obs();               // buildObs(sim, target) — training obs, verbatim
            const action = await policy.act(obs); // tanh(mean) deterministic eval action
            env.step(action);                    // mirrors sac.py eval: env.step(policy(obs))
            acc -= CONTROL_DT; n += 1;
            if (disposed) break;
          }
          if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0;

          render(dragging);

          if (now - hudMark >= 100) {
            hudMark = now;
            const dist = env.dist();
            setHud({ dist, reached: dist < SUCCESS_TOL, fps: lastFps, latMs: policy.meanLatencyMs() });
          }
        }

        canvas.removeEventListener("pointerdown", onDown);
        canvas.removeEventListener("pointermove", onMove);
        canvas.removeEventListener("pointerup", onUp);
        canvas.removeEventListener("pointercancel", onUp);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch2.2 sac live toy] failed", err);
        setHud((h) => ({ ...h, error: msg }));
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); if (sim) sim.dispose(); };
  }, []);

  const failed = !!hud.error;
  const meterPct = Math.max(0, Math.min(100, (hud.dist / RANDOM_DIST) * 100));

  const onKeyDown = (e: KeyboardEvent) => {
    const api = apiRef.current;
    if (!api) return;
    const map: Record<string, [number, number]> = {
      ArrowUp: [0, NUDGE], ArrowDown: [0, -NUDGE], ArrowLeft: [-NUDGE, 0], ArrowRight: [NUDGE, 0],
    };
    if (e.key in map) { e.preventDefault(); api.nudge(...map[e.key]); }
    else if (e.key === "r" || e.key === "R") api.reset();
    else if (e.key === "t" || e.key === "T") api.newTarget();
  };

  return (
    <div class="sl">
      <figure
        ref={figureRef}
        class="sl-figure"
        tabIndex={0}
        role="application"
        aria-label="Interactive SAC pusher-reach toy. A trained policy drives a two-link arm's fingertip onto a green target. Drag the target with the pointer, or focus here and use the arrow keys, to move it and watch the arm re-reach. Press T for a new target, R to reset the arm."
        onKeyDown={onKeyDown}
      >
        <div class="sl-poster" hidden={booted}><Poster /></div>
        <canvas ref={canvasRef} class="sl-canvas" hidden={!booted} aria-hidden="true" />

        <div class="bk-sr" aria-live="polite">
          {booted && !failed
            ? hud.reached
              ? "The SAC-trained arm has its fingertip on the target, inside the 2 centimetre success ring."
              : "The SAC-trained arm is driving its fingertip toward the target."
            : ""}
        </div>

        {booted && !failed && (
          <div class="sl-hud" aria-hidden="true">
            <div class="sl-hud-row">
              <span class="sl-k">fingertip → target</span>
              <span class={`sl-v ${hud.reached ? "sl-ok" : ""}`}>
                {hud.dist.toFixed(3)} m {hud.reached ? "✓" : ""}
              </span>
            </div>
            <div class="sl-meter"><div class="sl-meter-fill" data-reached={hud.reached} style={`width:${meterPct}%`} /></div>
          </div>
        )}

        <div class="sl-status" data-failed={failed} aria-hidden="true">
          {failed ? (
            <span>sim failed — the Colab path covers this without WASM</span>
          ) : booted ? (
            <>
              <span>real sac_actor.onnx · deterministic eval (contract v1)</span>
              <span>{hud.fps.toFixed(0)} fps · {hud.latMs.toFixed(2)} ms/call</span>
            </>
          ) : (
            <span>booting MuJoCo-WASM + SAC policy…</span>
          )}
        </div>
      </figure>

      <div class="sl-controls">
        <button type="button" class="sl-btn sl-btn--primary" onClick={() => apiRef.current?.newTarget()} disabled={!booted || failed}>
          new target →
        </button>
        <button type="button" class="sl-btn" onClick={() => apiRef.current?.reset()} disabled={!booted || failed}>
          reset arm
        </button>
        <span class="sl-control-note">
          drag the green target (or arrow-keys when focused) · poster reads with JS off
        </span>
      </div>
    </div>
  );
}

export default function SacReachLive() {
  return <SacToy />;
}
