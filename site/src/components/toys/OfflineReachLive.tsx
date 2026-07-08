/**
 * ch4 offline primer "BC vs AWAC" — the LIVE side-by-side pusher-reach toy
 * (`demo: offline_reach_live`, also stacked under `offline_bc_vs_awac` beneath the
 * recorded OfflineRLToy).
 *
 * The recorded OfflineRLToy shows the measured within-band result (offline RL beats
 * BC on the mixed expert+random dataset, with the naive-diverges Break-It). THIS
 * panel makes it live: TWO real policies trained on the SAME mixed data —
 * offline_bc.onnx (behaviour cloning) and offline_policy.onnx (AWAC, advantage-
 * weighted regression) — drive TWO identical MuJoCo-WASM arms toward the SAME
 * seeded target, in lockstep. The visitor gives them a new target and watches both
 * reach. Honest by construction: AWAC re-weights the mixed data toward the
 * high-advantage (expert-like) actions, so it tends to land tighter than plain BC
 * on the same data — the within-band story, made watchable. Both are FREE-TIER
 * policies (they miss sometimes); the point is the RELATIVE gap on identical data,
 * not a hero reach.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT at the top of ../PlateIsland.tsx (SSR
 * poster == JS-off fallback; lazy hydration-gated WASM+ONNX; primitives reused
 * verbatim — createSim + PUSHER_REACH_XML + BrowserPusherReachEnv + buildObs +
 * loadPolicy through a fail-closed obs[8]/act[2] gate; one control = new target).
 */
import "./OfflineReachLive.css";
import { useEffect, useRef, useState } from "preact/hooks";

const CANVAS_PX = 300;
const WORLD_HALF = 0.24;
const MAX_CONTROL_STEPS_PER_FRAME = 4;
const SUCCESS_TOL = 0.02;
const REACH = 0.2;

interface Lane {
  key: "awac" | "bc";
  label: string;
  url: string;
  color: string;
  blurb: string;
}
const LANES: Lane[] = [
  { key: "awac", label: "AWAC", url: "/models/offline_policy.onnx", color: "#1f56de",
    blurb: "advantage-weighted — leans on the expert-like actions in the mix" },
  { key: "bc", label: "BC", url: "/models/offline_bc.onnx", color: "#b0560f",
    blurb: "behaviour cloning — imitates the whole mix, good and bad alike" },
];

// ---------------------------------------------------------- shared SSR poster
const PV = 300;
const PS = PV / 2 / WORLD_HALF;
const pw2s = (x: number, y: number): [number, number] => [PV / 2 + x * PS, PV / 2 - y * PS];

function PosterArena({ color, label, blurb }: { color: string; label: string; blurb: string }) {
  const [bx, by] = pw2s(0, 0);
  const sh = 0.5, el = 0.9;
  const ex = 0.1 * Math.cos(sh), ey = 0.1 * Math.sin(sh);
  const fx = ex + 0.1 * Math.cos(sh + el), fy = ey + 0.1 * Math.sin(sh + el);
  const [ex_, ey_] = pw2s(ex, ey);
  const [fx_, fy_] = pw2s(fx, fy);
  const [tx, ty] = pw2s(0.14, 0.11);
  return (
    <figure class="or-lane">
      <svg class="or-poster-svg" viewBox={`0 0 ${PV} ${PV}`} role="img"
        aria-label={`Top-down pusher-reach arena for the ${label} policy. A two-link arm points toward a green target inside its reach ring; live, the ${label} policy drives the fingertip onto it.`}>
        <rect class="or-arena" x={2} y={2} width={PV - 4} height={PV - 4} rx={6} />
        <circle class="or-reach" cx={bx} cy={by} r={REACH * PS} />
        <circle class="or-tol" cx={tx} cy={ty} r={SUCCESS_TOL * PS} />
        <circle class="or-target-dot" cx={tx} cy={ty} r={4} />
        <polyline points={`${bx},${by} ${ex_},${ey_} ${fx_},${fy_}`} style={`fill:none;stroke:${color};stroke-width:4;stroke-linecap:round;stroke-linejoin:round`} />
        <circle cx={bx} cy={by} r={4} style="fill:#473f34" />
        <circle cx={fx_} cy={fy_} r={4.2} style={`fill:${color}`} />
      </svg>
      <figcaption class="or-lane-cap">
        <span class="or-lane-name" style={`color:${color}`}>{label}</span>
        <span class="or-lane-blurb">{blurb}</span>
      </figcaption>
    </figure>
  );
}

interface LaneHud { dist: number; reached: boolean; }

export default function OfflineReachLive() {
  const canvasRefs = [useRef<HTMLCanvasElement>(null), useRef<HTMLCanvasElement>(null)];
  const apiRef = useRef<{ newTarget: () => void; reset: () => void } | null>(null);
  const [booted, setBooted] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hud, setHud] = useState<LaneHud[]>([
    { dist: 0.176, reached: false }, { dist: 0.176, reached: false },
  ]);
  const [fps, setFps] = useState(0);

  useEffect(() => {
    let disposed = false;
    let raf = 0;
    const sims: Array<{ dispose(): void }> = [];
    const nextFrame = () => new Promise<number>((r) => (raf = requestAnimationFrame(r)));

    (async () => {
      try {
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

        // one WASM sim + env + policy per lane; both arms share a target sequence.
        const lanes = await Promise.all(LANES.map(async (lane, i) => {
          const s: any = await createSim(PUSHER_REACH_XML);
          sims.push(s);
          const env: any = new BrowserPusherReachEnv(s);
          const policy = await loadPolicy(lane.url);
          if (policy.contract.obsDim !== 8 || policy.contract.actDim !== 2) {
            throw new Error(
              `${lane.label} (${lane.url}) contract mismatch: expected obs[8]/act[2], ` +
              `got obs[${policy.contract.obsDim}]/act[${policy.contract.actDim}].`,
            );
          }
          const canvas = canvasRefs[i].current!;
          canvas.width = CANVAS_PX; canvas.height = CANVAS_PX;
          const ctx = canvas.getContext("2d")!;
          return { ...lane, sim: s, env, policy, ctx, canvas };
        }));
        if (disposed) { sims.forEach((s) => s.dispose()); return; }

        // seed both arms identically + set a shared first target
        let seed = 4;
        lanes.forEach((l) => l.env.reset(seed));
        const setSharedTarget = (x: number, y: number) => lanes.forEach((l) => l.env.setTarget(x, y));
        const newTarget = () => {
          const phi = Math.random() * 2 * Math.PI;
          const r = 0.07 + Math.random() * 0.11;
          setSharedTarget(r * Math.cos(phi), r * Math.sin(phi));
        };

        const S = CANVAS_PX / 2 / WORLD_HALF;
        const w2s = (x: number, y: number): [number, number] => [CANVAS_PX / 2 + x * S, CANVAS_PX / 2 - y * S];
        const renderLane = (l: typeof lanes[number]) => {
          const ctx = l.ctx, w = CANVAS_PX, h = CANVAS_PX;
          ctx.fillStyle = "#fbf9f3"; ctx.fillRect(0, 0, w, h);
          ctx.strokeStyle = "rgba(200,188,158,0.5)"; ctx.lineWidth = 1;
          ctx.beginPath();
          for (let i = 1; i < 8; i++) { const p = (i * w) / 8; ctx.moveTo(p, 0); ctx.lineTo(p, h); ctx.moveTo(0, p); ctx.lineTo(w, p); }
          ctx.stroke();
          const [bx, by] = w2s(0, 0);
          ctx.save(); ctx.setLineDash([3, 5]); ctx.globalAlpha = 0.5; ctx.strokeStyle = "#6d6252"; ctx.lineWidth = 1;
          ctx.beginPath(); ctx.arc(bx, by, REACH * S, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
          const [txw, tyw] = l.env.target; const [tx, ty] = w2s(txw, tyw);
          ctx.save(); ctx.setLineDash([4, 3]); ctx.strokeStyle = "#0c7d5f"; ctx.lineWidth = 1.6;
          ctx.beginPath(); ctx.arc(tx, ty, SUCCESS_TOL * S, 0, Math.PI * 2); ctx.stroke(); ctx.restore();
          ctx.fillStyle = "#0c7d5f"; ctx.beginPath(); ctx.arc(tx, ty, 4.5, 0, Math.PI * 2); ctx.fill();
          const shoulder = l.sim.jointQpos("shoulder"), elbow = l.sim.jointQpos("elbow");
          const ex = LINK_LEN * Math.cos(shoulder), ey = LINK_LEN * Math.sin(shoulder);
          const fx = ex + LINK_LEN * Math.cos(shoulder + elbow), fy = ey + LINK_LEN * Math.sin(shoulder + elbow);
          const [exp, eyp] = w2s(ex, ey), [fxp, fyp] = w2s(fx, fy);
          ctx.strokeStyle = l.color; ctx.lineWidth = 4; ctx.lineCap = "round"; ctx.lineJoin = "round";
          ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(exp, eyp); ctx.lineTo(fxp, fyp); ctx.stroke();
          ctx.fillStyle = "#473f34"; ctx.beginPath(); ctx.arc(bx, by, 4, 0, Math.PI * 2); ctx.fill();
          const dist = Math.hypot(txw - fx, tyw - fy);
          ctx.fillStyle = dist < SUCCESS_TOL ? "#0c7d5f" : l.color;
          ctx.beginPath(); ctx.arc(fxp, fyp, 5, 0, Math.PI * 2); ctx.fill();
        };

        setBooted(true);
        lanes.forEach(renderLane);

        apiRef.current = {
          newTarget,
          reset: () => { seed += 1; lanes.forEach((l) => l.env.reset(seed)); },
        };

        (window as any).__toy = {
          lanes: () => LANES.map((l) => l.key),
          contract: (i: number) => ({ ...lanes[i].policy.contract }),
          obsParity: (i: number) => {
            const l = lanes[i]; const a = l.env.obs();
            const [txw, tyw] = l.env.target; const b = buildObs(l.sim, txw, tyw);
            let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k]));
            return { equal: m === 0, maxErr: m };
          },
          dists: () => lanes.map((l) => l.env.dist()),
          newTarget: () => apiRef.current?.newTarget(),
          async drive(n: number) {
            const before = lanes.map((l) => l.env.dist());
            for (let i = 0; i < n; i++) {
              for (const l of lanes) l.env.step(await l.policy.act(l.env.obs()));
            }
            return { steps: n, distBefore: before, distAfter: lanes.map((l) => l.env.dist()) };
          },
        };

        let frames = 0, fpsMark = performance.now(), last = performance.now(), acc = 0, hudMark = 0, lastFps = 0;
        while (!disposed) {
          await nextFrame();
          const now = performance.now(); frames++;
          if (now - fpsMark >= 500) { lastFps = (frames * 1000) / (now - fpsMark); frames = 0; fpsMark = now; setFps(lastFps); }
          acc += Math.min(now - last, 100) / 1000; last = now;
          let n = 0;
          while (acc >= CONTROL_DT && n < MAX_CONTROL_STEPS_PER_FRAME) {
            for (const l of lanes) l.env.step(await l.policy.act(l.env.obs()));
            acc -= CONTROL_DT; n += 1; if (disposed) break;
          }
          if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0;
          lanes.forEach(renderLane);
          if (now - hudMark >= 100) {
            hudMark = now;
            setHud(lanes.map((l) => { const d = l.env.dist(); return { dist: d, reached: d < SUCCESS_TOL }; }));
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch4 offline live toy] failed", err);
        setError(msg);
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); sims.forEach((s) => s.dispose()); };
  }, []);

  const failed = !!error;

  return (
    <div class="or">
      <div
        class="or-figure"
        role="application"
        aria-label="Interactive offline-RL toy: two arms trained on the same mixed dataset — AWAC and behaviour cloning — reach the same target. Press new target to move both goals and compare where each lands."
      >
        <div class="or-lanes" hidden={booted && !failed}>
          {LANES.map((l) => <PosterArena color={l.color} label={l.label} blurb={l.blurb} />)}
        </div>
        <div class="or-lanes" hidden={!booted || failed}>
          {LANES.map((l, i) => (
            <figure class="or-lane">
              <canvas ref={canvasRefs[i]} class="or-canvas" aria-hidden="true" />
              <figcaption class="or-lane-cap">
                <span class="or-lane-name" style={`color:${l.color}`}>{l.label}</span>
                <span class={`or-lane-dist ${hud[i].reached ? "or-ok" : ""}`}>
                  {hud[i].dist.toFixed(3)} m {hud[i].reached ? "✓" : ""}
                </span>
              </figcaption>
            </figure>
          ))}
        </div>

        {/* Qualitative, STABLE announcement. The fingertip distances update every
            frame, so reading them out here would spam a screen reader (the live
            figures stay in the lane captions for sighted users). Announce the
            comparison the toy exists to make, not per-frame scalars — matching the
            sibling live toys (SacReachLive, DaggerPushtLive). */}
        <div class="bk-sr" aria-live="polite">
          {booted && !failed
            ? "Two arms trained on the same mixed-quality dataset are reaching the same target: AWAC, which up-weights the better demonstrations, and plain behaviour cloning. Watch which fingertip settles inside the target ring — on mixed data AWAC reaches it more reliably than behaviour cloning."
            : ""}
        </div>

        <div class="or-status" data-failed={failed} aria-hidden="true">
          {failed ? (
            <span>sim failed — the Colab path covers this without WASM</span>
          ) : booted ? (
            <span>real offline_policy.onnx (AWAC) vs offline_bc.onnx (BC) · same mixed data · {fps.toFixed(0)} fps</span>
          ) : (
            <span>booting two MuJoCo-WASM arms + both policies…</span>
          )}
        </div>
      </div>

      <div class="or-controls">
        <button type="button" class="or-btn or-btn--primary" onClick={() => apiRef.current?.newTarget()} disabled={!booted || failed}>
          new target (both) →
        </button>
        <button type="button" class="or-btn" onClick={() => apiRef.current?.reset()} disabled={!booted || failed}>
          reset arms
        </button>
        <span class="or-control-note">
          same mixed data, two learners · AWAC re-weights toward the good actions · poster reads with JS off
        </span>
      </div>
    </div>
  );
}
