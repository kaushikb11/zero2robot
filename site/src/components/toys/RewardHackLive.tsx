/**
 * ch2.4 "Reward Design Is Programming" — the LIVE specification-gaming toy
 * (`demo: quadruped_reward_hack`, stacked under the recorded RewardHackToy).
 *
 * SIDE BY SIDE, same robot, same PPO, ONE difference — the reward PROGRAM. Two
 * REAL policies drive two REAL free-floating MuJoCo-WASM quadrupeds from the SAME
 * seeded start:
 *   shaped_walk.onnx  — trained on the env's dense shaped reward: it WALKS FORWARD.
 *   height_hack.onnx  — trained on the naive "be tall" reward (HACK_HEIGHT_W·height,
 *                       no forward term): it REARS UP and stalls. Its own reward
 *                       CLIMBS (the height it optimizes) while it covers ~0 forward.
 * That is reward hacking / specification gaming, made physical: the policy did what
 * you SAID (get tall), not what you MEANT (go forward). Trained results this ships
 * from: shaped +6.27 m forward, hack +0.16 m while its return rose 128 → 841.
 *
 * FROZEN CONCEPT-TOY CONTRACT (see ../PlateIsland.tsx): SSR poster == JS-off
 * fallback; lazy hydration-gated sim; primitives reused verbatim (createSim +
 * QUADRUPED_XML + BrowserQuadrupedEnv + buildObs + loadPolicy through a fail-closed
 * obs[23]/act[8] gate); the invisible made visible (forward distance vs the hack's
 * own climbing reward); ONE control (reset the shared start); colour discipline
 * (the robot is the same navy in both panels — only the BEHAVIOUR differs).
 */
import "./RewardHackLive.css";
import { useEffect, useRef, useState } from "preact/hooks";
import { drawQuadruped, QUAD_COLORS_LIGHT, type QuadColors } from "./quadruped_render";

const SHAPED_URL = "/models/shaped_walk.onnx";
const HACK_URL = "/models/height_hack.onnx";
const HACK_HEIGHT_W = 10.0; // rewards.py: the height-hack program is HACK_HEIGHT_W * torso height
const CANVAS_W = 360;
const CANVAS_H = 300;
const SCALE = 150;
const MAX_CONTROL_STEPS_PER_FRAME = 4;

// per-panel accent (the robot stays navy; the accent tints the readout + start line)
const SHAPED_ACCENT = "#0c7d5f"; // green — it walks
const HACK_ACCENT = "#b0560f"; // amber — it games the reward

function PosterBot({ x, rear, label, sub }: { x: number; rear: boolean; label: string; sub: string }) {
  const groundY = 232;
  const torsoY = groundY - (rear ? 0.34 : 0.24) * 150;
  const tilt = rear ? -18 : 0;
  return (
    <g transform={`translate(${x} 0)`}>
      <line class="rh-ground" x1={0} y1={groundY} x2={180} y2={groundY} />
      <g transform={`rotate(${tilt} 90 ${torsoY})`}>
        <rect class="rh-torso" x={62} y={torsoY - 10} width={80} height={20} rx={4} />
        <circle class="rh-head" cx={146} cy={torsoY} r={4} />
      </g>
      <polyline class="rh-leg" points={`120,${torsoY + 6} 128,${groundY - 22} 122,${groundY}`} />
      <polyline class="rh-leg" points={`72,${torsoY + 6} 64,${groundY - 22} 70,${groundY}`} />
      <text class="rh-poster-lbl" x={90} y={groundY + 22} text-anchor="middle">{label}</text>
      <text class="rh-poster-sub" x={90} y={groundY + 38} text-anchor="middle">{sub}</text>
    </g>
  );
}

function Poster() {
  return (
    <svg
      class="rh-poster-svg"
      viewBox="0 0 400 300"
      role="img"
      aria-label="Two four-legged robots side by side. The left one, trained on a shaped reward, walks forward. The right one, trained to just be tall, rears up and stays put while its reward climbs. With JavaScript on, both are driven live by their trained policies from the same start."
    >
      <title>Reward hacking — shaped walk vs height-hack, live and side by side</title>
      <desc>
        Same robot, same PPO, two reward programs. The shaped policy walks; the
        height-hack rears up and stalls, scoring a large reward while going nowhere —
        specification gaming made physical.
      </desc>
      <rect class="rh-arena" x={1} y={1} width={398} height={298} rx={6} />
      <PosterBot x={10} rear={false} label="shaped reward" sub="walks forward" />
      <PosterBot x={210} rear={true} label="height-hack" sub="rears · reward climbs, goes nowhere" />
    </svg>
  );
}

interface PanelHud {
  dist: number;
  height: number;
  hackReturn: number;
  fell: boolean;
  step: number;
}
interface Hud {
  shaped: PanelHud;
  hack: PanelHud;
  fps: number;
  latMs: number;
  error?: string;
}
const zeroPanel: PanelHud = { dist: 0, height: 0.257, hackReturn: 0, fell: false, step: 0 };

function RewardToy() {
  const shapedCanvas = useRef<HTMLCanvasElement>(null);
  const hackCanvas = useRef<HTMLCanvasElement>(null);
  const figureRef = useRef<HTMLElement>(null);
  const apiRef = useRef<{ reset: () => void } | null>(null);
  const [booted, setBooted] = useState(false);
  const [hud, setHud] = useState<Hud>({ shaped: { ...zeroPanel }, hack: { ...zeroPanel }, fps: 0, latMs: 0 });

  useEffect(() => {
    let disposed = false;
    const sims: { dispose(): void }[] = [];
    let raf = 0;
    const nextFrame = () => new Promise<number>((r) => (raf = requestAnimationFrame(r)));

    (async () => {
      try {
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

        // two independent sims/envs so both policies run on their own physics
        const makePanel = async (
          url: string, canvas: HTMLCanvasElement, accent: string, label: string,
        ) => {
          const s: any = await createSim(QUADRUPED_XML);
          sims.push(s);
          const env: any = new BrowserQuadrupedEnv(s);
          const policy = await loadPolicy(url);
          if (policy.contract.obsDim !== 23 || policy.contract.actDim !== 8) {
            throw new Error(
              `${url} contract mismatch: expected obs[23]/act[8], ` +
              `got obs[${policy.contract.obsDim}]/act[${policy.contract.actDim}].`,
            );
          }
          canvas.width = CANVAS_W; canvas.height = CANVAS_H;
          const ctx = canvas.getContext("2d")!;
          const colors: QuadColors = { ...QUAD_COLORS_LIGHT, accent };
          return {
            s, env, policy, ctx, colors, label,
            startX: 0, fell: false, survived: false, hackReturn: 0,
          };
        };

        const shaped = await makePanel(SHAPED_URL, shapedCanvas.current!, SHAPED_ACCENT, "shaped: walks");
        if (disposed) { sims.forEach((x) => x.dispose()); return; }
        const hack = await makePanel(HACK_URL, hackCanvas.current!, HACK_ACCENT, "height-hack: games it");
        if (disposed) { sims.forEach((x) => x.dispose()); return; }
        const panels = [shaped, hack];

        let seed = 0;
        const resetAll = () => {
          seed += 1;
          for (const p of panels) {
            p.env.reset(seed);
            p.startX = p.s.qposAt(p.s.jointQposAdr("root"));
            p.fell = false; p.survived = false; p.hackReturn = 0;
          }
        };
        // shared start: both panels the SAME seed so only the reward program differs
        for (const p of panels) { p.env.reset(seed); p.startX = p.s.qposAt(p.s.jointQposAdr("root")); }

        const renderPanel = (p: typeof shaped) => {
          const camX = p.s.qposAt(p.s.jointQposAdr("root"));
          drawQuadruped(p.ctx, p.s, {
            W: CANVAS_W, H: CANVAS_H, scale: SCALE, camX, colors: p.colors,
            startX: p.startX, fallen: p.fell, label: p.label,
          });
        };

        setBooted(true);
        panels.forEach(renderPanel);

        apiRef.current = { reset: resetAll };

        (window as any).__toy = {
          contract: () => ({ shaped: { ...shaped.policy.contract }, hack: { ...hack.policy.contract } }),
          obsParity: () => {
            let m = 0;
            for (const p of panels) {
              const a = p.env.obs(); const b = buildObs(p.s);
              for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k]));
            }
            return { equal: m === 0, maxErr: m };
          },
          shapedDist: () => shaped.env.forwardDist(),
          hackDist: () => hack.env.forwardDist(),
          hackHeight: () => hack.env.height(),
          reset: () => apiRef.current?.reset(),
          async drive(n: number) {
            for (let i = 0; i < n; i++) {
              for (const p of panels) { const o = p.env.obs(); p.env.step(await p.policy.act(o)); }
            }
            return {
              steps: n,
              shapedDist: shaped.env.forwardDist(),
              hackDist: hack.env.forwardDist(),
              hackHeight: hack.env.height(),
            };
          },
          fps: () => lastFps,
        };

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
            for (const p of panels) {
              if (p.fell || p.survived) continue;
              const obs = p.env.obs();
              const action = await p.policy.act(obs);
              const res = p.env.step(action);
              // the hack's OWN reward program (rewards.py r_hack): HACK_HEIGHT_W * height
              p.hackReturn += HACK_HEIGHT_W * res.height;
              if (res.done) { if (res.terminated) p.fell = true; else p.survived = true; }
            }
            acc -= CONTROL_DT; n += 1;
            if (disposed) break;
          }
          if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0;

          panels.forEach(renderPanel);

          if (now - hudMark >= 100) {
            hudMark = now;
            const panelHud = (p: typeof shaped): PanelHud => ({
              dist: p.env.forwardDist(), height: p.env.height(), hackReturn: p.hackReturn,
              fell: p.fell, step: p.env.steps,
            });
            setHud({ shaped: panelHud(shaped), hack: panelHud(hack), fps: lastFps, latMs: shaped.policy.meanLatencyMs() });
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch2.4 reward-hack live toy] failed", err);
        setHud((h) => ({ ...h, error: msg }));
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); sims.forEach((s) => s.dispose()); };
  }, []);

  const failed = !!hud.error;
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "r" || e.key === "R") { e.preventDefault(); apiRef.current?.reset(); }
  };

  return (
    <div class="rh">
      <figure
        ref={figureRef}
        class="rh-figure"
        tabIndex={0}
        role="application"
        aria-label="Interactive reward-hacking toy. Two four-legged robots, same algorithm, different reward. The left walks forward; the right rears up and stalls while its own reward climbs. Focus here and press R, or use the reset button, to restart both from the same pose."
        onKeyDown={onKeyDown}
      >
        <div class="rh-poster" hidden={booted}><Poster /></div>
        <div class="rh-panels" hidden={!booted} aria-hidden="true">
          <div class="rh-panel">
            <canvas ref={shapedCanvas} class="rh-canvas" />
            {booted && !failed && (
              <div class="rh-cap rh-cap--shaped">
                <span class="rh-cap-title">shaped reward → walks</span>
                <span class="rh-cap-num">forward {hud.shaped.dist.toFixed(2)} m</span>
              </div>
            )}
          </div>
          <div class="rh-panel">
            <canvas ref={hackCanvas} class="rh-canvas" />
            {booted && !failed && (
              <div class="rh-cap rh-cap--hack">
                <span class="rh-cap-title">“be tall” hack → games it</span>
                <span class="rh-cap-num">forward {hud.hack.dist.toFixed(2)} m · reward {hud.hack.hackReturn.toFixed(0)} ▲</span>
              </div>
            )}
          </div>
        </div>

        {/* Qualitative, STABLE announcement — the numbers churn every ~100 ms as the
            robots move, so surfacing them live here would spam a screen reader (the
            per-frame figures stay in the aria-hidden panel captions for sighted
            users). This mirrors the sibling live toys (QuadrupedWalkLive,
            DomainRandLive), which announce the qualitative story, not per-frame scalars. */}
        <div class="bk-sr" aria-live="polite">
          {booted && !failed
            ? "Same algorithm, two rewards. The shaped policy walks steadily forward; the height-hack barely travels yet its own reward keeps climbing as it rears up and stalls in place — it did what you said, not what you meant."
            : ""}
        </div>

        <div class="rh-status" data-failed={failed} aria-hidden="true">
          {failed ? (
            <span>sim failed — the Colab path covers this without WASM</span>
          ) : booted ? (
            <>
              <span>real shaped_walk.onnx + height_hack.onnx · same PPO, one reward apart (contract v1)</span>
              <span>{hud.fps.toFixed(0)} fps · {hud.latMs.toFixed(2)} ms/call</span>
            </>
          ) : (
            <span>booting MuJoCo-WASM + two reward policies…</span>
          )}
        </div>
      </figure>

      <div class="rh-controls">
        <button type="button" class="rh-btn rh-btn--primary" onClick={() => apiRef.current?.reset()} disabled={!booted || failed}>
          restart both from a fresh pose →
        </button>
        <span class="rh-control-note">
          same robot, same PPO — the only difference is the reward · poster reads with JS off
        </span>
      </div>
    </div>
  );
}

export default function RewardHackLive() {
  return <RewardToy />;
}
