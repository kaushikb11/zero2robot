/**
 * ch2.5 "Locomotion: The Quadruped Walks" — the LIVE emergent-gait toy
 * (`demo: quadruped_walk`, stacked under the recorded QuadrupedWalkToy).
 *
 * A REAL walk_actor.onnx (the trained deterministic SAC policy — tanh(mean), the
 * hidden-64 free-tier checkpoint) drives a REAL free-floating MuJoCo-WASM quadruped
 * from a standing crouch. The hero is GAIT EMERGENCE: nobody scripted a gait; the
 * legs fall into a repeating stride and the torso travels +x. And the honest
 * ceiling is built in — at the free-tier budget the emergent gait is fast but
 * FRAGILE: it sprints forward and then FALLS before the full 10 s (500-step)
 * horizon (ch2.5 prose: "Emergent does not mean robust"). We don't hide the fall —
 * we mark it. The Scale Lab is where more env steps buy stability.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT (see ../PlateIsland.tsx):
 *   1. SSR poster == the JS-off fallback (<Poster/> is pure static JSX).
 *   2. Lazy, hydration-gated sim: MuJoCo-WASM + onnxruntime via *dynamic* import().
 *   3. Reuse the primitives verbatim: createSim + QUADRUPED_XML +
 *      BrowserQuadrupedEnv + buildObs + loadPolicy through a fail-closed
 *      obs[23]/act[8] contract gate. The driver mirrors walk.py eval:
 *        obs = env.obs(); action = await policy.act(obs); env.step(action)
 *   4. Make the invisible visible: forward-distance meter + the start line on the
 *      ground; the fall is announced, not swept away.
 *   5. ONE control (reset a fresh episode), immediate feedback, default-interesting
 *      (boots mid-walk). Keyboard (R) + button to the same aha.
 *   6. Colour discipline: the robot is the same navy everywhere; ONE --signal blue
 *      accent for the live readout, --alert only on the fall.
 */
import "./QuadrupedWalkLive.css";
import { useEffect, useRef, useState } from "preact/hooks";
import { drawQuadruped, QUAD_COLORS_LIGHT } from "./quadruped_render";

const MODEL_URL = "/models/walk_actor.onnx";
const CANVAS_W = 520;
const CANVAS_H = 340;
const SCALE = 170; // px per metre (side view)
const MAX_CONTROL_STEPS_PER_FRAME = 4;

function Poster() {
  // a stylized side-view crouch on the ground — the pre-boot / JS-off frame
  const groundY = 250;
  const torsoY = groundY - 0.257 * 170;
  return (
    <svg
      class="ql-poster-svg"
      viewBox="0 0 520 340"
      role="img"
      aria-label="Side view of a four-legged robot standing on the ground in a gentle crouch. With JavaScript on, a trained policy makes it walk forward — an emergent gait nobody scripted — and, at the free-tier budget, it sprints and then falls before the full ten-second horizon."
    >
      <title>Quadruped walk — an emergent gait from a trained policy, live</title>
      <desc>
        A from-scratch quadruped viewed side-on. Live, a trained SAC policy drives
        its eight leg joints into a repeating stride that carries the torso forward;
        the gait is fast but not yet robust, so it falls before ten seconds.
      </desc>
      <rect class="ql-arena" x={1} y={1} width={518} height={338} rx={6} />
      <line class="ql-ground" x1={0} y1={groundY} x2={520} y2={groundY} />
      {/* torso */}
      <rect class="ql-torso" x={200} y={torsoY - 12} width={122} height={24} rx={4} />
      <circle class="ql-head" cx={328} cy={torsoY} r={5} />
      {/* four bent legs (near pair solid, far pair faint) */}
      <polyline class="ql-leg" points={`306,${torsoY + 8} 322,${groundY - 30} 314,${groundY}`} />
      <polyline class="ql-leg" points={`214,${torsoY + 8} 198,${groundY - 30} 206,${groundY}`} />
      <polyline class="ql-leg ql-leg--far" points={`300,${torsoY + 8} 316,${groundY - 30} 308,${groundY}`} />
      <polyline class="ql-leg ql-leg--far" points={`220,${torsoY + 8} 204,${groundY - 30} 212,${groundY}`} />
      <circle class="ql-foot" cx={314} cy={groundY} r={5} />
      <circle class="ql-foot" cx={206} cy={groundY} r={5} />
      <text class="ql-poster-lbl" x={16} y={26}>walk_actor.onnx · press play with JS on →</text>
    </svg>
  );
}

interface Hud {
  dist: number;
  height: number;
  upZ: number;
  step: number;
  fell: boolean;
  survived: boolean;
  fps: number;
  latMs: number;
  error?: string;
}

function WalkToy() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const figureRef = useRef<HTMLElement>(null);
  const apiRef = useRef<{ reset: () => void } | null>(null);
  const [booted, setBooted] = useState(false);
  const [hud, setHud] = useState<Hud>({
    dist: 0, height: 0.257, upZ: 1, step: 0, fell: false, survived: false, fps: 0, latMs: 0,
  });

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
          import("../../../../playground/src/teleop/quadruped_env"),
          import("../../../../playground/src/teleop/quadruped_obs"),
          import("../../../../playground/src/policy/infer"),
        ]);
        const { createSim } = simMod;
        const { QUADRUPED_XML } = sceneMod;
        const { BrowserQuadrupedEnv } = envMod;
        const { buildObs, CONTROL_DT } = obsMod;
        const { loadPolicy } = inferMod;

        const realSim: any = await createSim(QUADRUPED_XML);
        sim = realSim;
        if (disposed) { realSim.dispose(); return; }
        const env: any = new BrowserQuadrupedEnv(realSim);
        let seed = 0;
        env.reset(seed);
        let startX = 0;
        let fell = false;
        let survived = false;

        const canvas = canvasRef.current!;
        canvas.width = CANVAS_W;
        canvas.height = CANVAS_H;
        const ctx = canvas.getContext("2d")!;

        const render = () => {
          const camX = realSim.qposAt(realSim.jointQposAdr("root")); // camera follows torso x
          drawQuadruped(ctx, realSim, {
            W: CANVAS_W, H: CANVAS_H, scale: SCALE, camX,
            colors: QUAD_COLORS_LIGHT, startX, fallen: fell,
            label: fell ? "fell — emergent, not yet robust" : "emergent gait",
          });
        };

        const doReset = () => {
          seed += 1;
          env.reset(seed);
          startX = realSim.qposAt(realSim.jointQposAdr("root"));
          fell = false; survived = false;
        };
        startX = realSim.qposAt(realSim.jointQposAdr("root"));

        const policy = await loadPolicy(MODEL_URL);
        if (policy.contract.obsDim !== 23 || policy.contract.actDim !== 8) {
          throw new Error(
            `walk_actor.onnx contract mismatch: expected obs[23]/act[8] (quadruped), ` +
            `got obs[${policy.contract.obsDim}]/act[${policy.contract.actDim}].`,
          );
        }
        if (disposed) return;

        // load-then-boot: reveal the live canvas only after the policy loads and its
        // obs[23]/act[8] contract verifies — a fetch/contract failure keeps booted=false
        // so the captioned SSR poster stays up (fail-closed), never a frozen canvas.
        setBooted(true);
        render();

        apiRef.current = { reset: doReset };

        (window as any).__toy = {
          contract: () => ({ ...policy.contract }),
          obsParity: () => {
            const a = env.obs();
            const b = buildObs(realSim);
            let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k]));
            return { equal: m === 0, maxErr: m };
          },
          forwardDist: () => env.forwardDist(),
          height: () => env.height(),
          upZ: () => env.upZ(),
          fell: () => fell,
          reset: () => apiRef.current?.reset(),
          async drive(n: number) {
            const d0 = env.forwardDist();
            for (let i = 0; i < n; i++) { const o = env.obs(); env.step(await policy.act(o)); }
            return { steps: n, distGained: env.forwardDist() - d0, meanLatencyMs: policy.meanLatencyMs(), calls: policy.calls };
          },
          fps: () => lastFps,
        };

        // DRIVE — real-time paced to CONTROL_HZ (mirrors walk.py eval). The episode
        // runs until the policy falls (terminated) or rides the full horizon
        // (truncated); it then FREEZES on that frame so the visitor sees the honest
        // outcome, and resumes only on reset.
        let lastFps = 0, frames = 0, fpsMark = performance.now(), last = performance.now(), acc = 0, hudMark = 0;

        // Reduced motion: the emergent-gait still frame is painted and the reset control
        // + __toy hooks stay live; don't spin the auto-driving rAF loop.
        if (prefersReducedMotion) return;
        while (!disposed) {
          await nextFrame();
          const now = performance.now();
          frames++;
          if (now - fpsMark >= 500) { lastFps = (frames * 1000) / (now - fpsMark); frames = 0; fpsMark = now; }

          if (!fell && !survived) {
            acc += Math.min(now - last, 100) / 1000;
            let n = 0;
            while (acc >= CONTROL_DT && n < MAX_CONTROL_STEPS_PER_FRAME) {
              const obs = env.obs();
              const action = await policy.act(obs);
              const res = env.step(action);
              acc -= CONTROL_DT; n += 1;
              if (res.done) {
                if (res.terminated) fell = true; else survived = true;
                break;
              }
              if (disposed) break;
            }
            if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0;
          }
          last = now;

          render();

          if (now - hudMark >= 100) {
            hudMark = now;
            setHud({
              dist: env.forwardDist(), height: env.height(), upZ: env.upZ(), step: env.steps,
              fell, survived, fps: lastFps, latMs: policy.meanLatencyMs(),
            });
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch2.5 walk live toy] failed", err);
        setHud((h) => ({ ...h, error: msg }));
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); if (sim) sim.dispose(); };
  }, []);

  const failed = !!hud.error;
  const distPct = Math.max(0, Math.min(100, (hud.dist / 3.0) * 100)); // ~3 m emergent reach

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "r" || e.key === "R") { e.preventDefault(); apiRef.current?.reset(); }
  };

  return (
    <div class="qw">
      <figure
        ref={figureRef}
        class="ql-figure"
        tabIndex={0}
        role="application"
        aria-label="Interactive quadruped-walk toy. A trained policy drives a four-legged robot into an emergent walking gait. Focus here and press R, or use the reset button, to start a fresh episode. The robot sprints forward and then falls before the full horizon."
        onKeyDown={onKeyDown}
      >
        <div class="ql-poster" hidden={booted}><Poster /></div>
        <canvas ref={canvasRef} class="ql-canvas" hidden={!booted} aria-hidden="true" />

        <div class="bk-sr" aria-live="polite">
          {booted && !failed
            ? hud.fell
              ? `The emergent gait carried the robot ${hud.dist.toFixed(2)} metres forward and then fell — fast, but not yet robust.`
              : hud.survived
                ? `The robot rode out the full ten-second horizon, ${hud.dist.toFixed(2)} metres forward.`
                : "The trained policy's legs have fallen into a repeating stride and the torso is travelling forward."
            : ""}
        </div>

        {booted && !failed && (
          <div class="ql-hud" aria-hidden="true">
            <div class="ql-hud-row">
              <span class="ql-k">forward distance</span>
              <span class="ql-v">{hud.dist.toFixed(2)} m</span>
            </div>
            <div class="ql-meter"><div class="ql-meter-fill" data-fell={hud.fell} style={`width:${distPct}%`} /></div>
            <div class="ql-hud-row">
              <span class="ql-k">torso height</span>
              <span class={`ql-v ${hud.fell ? "ql-bad" : ""}`}>{hud.height.toFixed(3)} m</span>
            </div>
            <div class="ql-hud-row">
              <span class="ql-k">step</span>
              <span class="ql-v">{hud.step} / 500 {hud.fell ? "· fell" : hud.survived ? "· survived" : ""}</span>
            </div>
          </div>
        )}

        <div class="ql-status" data-failed={failed} aria-hidden="true">
          {failed ? (
            <span>sim failed — the Colab path covers this without WASM</span>
          ) : booted ? (
            <>
              <span>real walk_actor.onnx · deterministic eval (contract v1)</span>
              <span>{hud.fps.toFixed(0)} fps · {hud.latMs.toFixed(2)} ms/call</span>
            </>
          ) : (
            <span>booting MuJoCo-WASM + walk policy…</span>
          )}
        </div>
      </figure>

      <div class="ql-controls">
        <button type="button" class="ql-btn ql-btn--primary" onClick={() => apiRef.current?.reset()} disabled={!booted || failed}>
          {hud.fell || hud.survived ? "walk again →" : "restart episode →"}
        </button>
        <span class="ql-control-note">
          emergent gait, honest ceiling: it sprints then falls · poster reads with JS off
        </span>
      </div>
    </div>
  );
}

export default function QuadrupedWalkLive() {
  return <WalkToy />;
}
