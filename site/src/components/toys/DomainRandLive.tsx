/**
 * ch2.7 "Domain Randomization: Randomize to Generalize" — the LIVE break-the-policy
 * toy (`demo: narrow_vs_randomized_across_the_gap`, stacked under the recorded
 * DomainRandToy).
 *
 * Two REAL policies drive two REAL free-floating MuJoCo-WASM quadrupeds under the
 * SAME shifted dynamics, and the visitor drags a DYNAMICS slider (mass scale):
 *   dr_narrow.onnx      — trained on nominal dynamics only.
 *   dr_randomized.onnx  — trained across a band of mass/friction/gravity each episode.
 * The runtime scales body mass/inertia exactly as dr.py's apply_scales (on top of the
 * pinned contact solver), so no re-export is needed to reshift the world.
 *
 * THE HONEST BEAT (ch2.7 prose): on this free-tier quadruped STAND, DR is the promise
 * you TEST, not a clean win. Across most of the slider the two policies behave ALIKE
 * (standing is a stable equilibrium a nominal policy already handles), and past ~1.2×
 * mass the ±12 N·m servos saturate and BOTH fall. The measured off-nominal survival
 * edge sits INSIDE the seed band (−0.02 on the seed this ships). So the visitor SEES
 * the boundary of what randomization buys: robustness across a RANGE, not a stronger
 * robot — the deliberate counterpoint to a triumphant DR demo.
 *
 * FROZEN CONCEPT-TOY CONTRACT (see ../PlateIsland.tsx): SSR poster == JS-off
 * fallback; lazy hydration-gated sim; primitives reused verbatim (createSim +
 * QUADRUPED_XML + BrowserQuadrupedEnv + buildObs + loadPolicy through a fail-closed
 * obs[23]/act[8] gate); the invisible made visible (each policy's survival at the
 * chosen mass); ONE control (the mass slider); colour discipline (same navy robot).
 */
import "./DomainRandLive.css";
import { useEffect, useRef, useState } from "preact/hooks";
import { drawQuadruped, QUAD_COLORS_LIGHT, type QuadColors } from "./quadruped_render";

const NARROW_URL = "/models/dr_narrow.onnx";
const RAND_URL = "/models/dr_randomized.onnx";
const CANVAS_W = 360;
const CANVAS_H = 300;
const SCALE = 150;
const MAX_CONTROL_STEPS_PER_FRAME = 4;
const MASS_MIN = 0.6, MASS_MAX = 1.8;

const NARROW_ACCENT = "#b0560f"; // amber — trained narrow
const RAND_ACCENT = "#1f56de"; // blue — trained across the band

function PosterBot({ x, label, sub }: { x: number; label: string; sub: string }) {
  const groundY = 226;
  const torsoY = groundY - 0.24 * 150;
  return (
    <g transform={`translate(${x} 0)`}>
      <line class="dl-ground" x1={0} y1={groundY} x2={180} y2={groundY} />
      <rect class="dl-torso" x={62} y={torsoY - 10} width={80} height={20} rx={4} />
      <circle class="dl-head" cx={146} cy={torsoY} r={4} />
      <polyline class="dl-leg" points={`120,${torsoY + 6} 128,${groundY - 22} 122,${groundY}`} />
      <polyline class="dl-leg" points={`72,${torsoY + 6} 64,${groundY - 22} 70,${groundY}`} />
      <text class="dl-poster-lbl" x={90} y={groundY + 22} text-anchor="middle">{label}</text>
      <text class="dl-poster-sub" x={90} y={groundY + 38} text-anchor="middle">{sub}</text>
    </g>
  );
}

function Poster() {
  return (
    <svg
      class="dl-poster-svg"
      viewBox="0 0 400 300"
      role="img"
      aria-label="Two four-legged robots side by side under the same shifted physics. One was trained on nominal dynamics only, the other across a band of mass, friction, and gravity. With JavaScript on, drag a mass slider: across most of the range both stand alike, and past about 1.2 times mass both fall — domain randomization buys robustness across a range, not a stronger robot."
    >
      <title>Domain randomization — narrow vs randomized across the gap, live</title>
      <desc>
        Two policies under the same runtime-scaled dynamics. Drag the mass slider to
        break them: across most of the range they behave alike, and past ~1.2× mass
        both fall — the within-band honesty of what randomization can and cannot buy.
      </desc>
      <rect class="dl-arena" x={1} y={1} width={398} height={298} rx={6} />
      <PosterBot x={10} label="narrow policy" sub="nominal dynamics only" />
      <PosterBot x={210} label="randomized policy" sub="trained across the band" />
    </svg>
  );
}

interface PanelHud { height: number; upZ: number; fell: boolean; survived: boolean; step: number; }
interface Hud { narrow: PanelHud; rand: PanelHud; mass: number; fps: number; latMs: number; error?: string; }
const zeroPanel: PanelHud = { height: 0.257, upZ: 1, fell: false, survived: false, step: 0 };

function DrToy() {
  const narrowCanvas = useRef<HTMLCanvasElement>(null);
  const randCanvas = useRef<HTMLCanvasElement>(null);
  const figureRef = useRef<HTMLElement>(null);
  const apiRef = useRef<{ setMass: (m: number) => void; rerun: () => void } | null>(null);
  const [booted, setBooted] = useState(false);
  const [mass, setMass] = useState(1.0);
  const [hud, setHud] = useState<Hud>({ narrow: { ...zeroPanel }, rand: { ...zeroPanel }, mass: 1.0, fps: 0, latMs: 0 });

  useEffect(() => {
    let disposed = false;
    const sims: { dispose(): void }[] = [];
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

        const makePanel = async (url: string, canvas: HTMLCanvasElement, accent: string, label: string) => {
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
          const model: any = s.model;
          // snapshot the as-built physical params so every scale is from nominal
          // (scaling in place would compound), exactly like dr.py capture_nominal.
          const nbody = model.nbody as number;
          const nominalMass = Array.from({ length: nbody }, (_, i) => model.body_mass[i] as number);
          const nominalInertia = Array.from({ length: nbody * 3 }, (_, i) => model.body_inertia[i] as number);
          return { s, env, policy, ctx, colors, label, model, nbody, nominalMass, nominalInertia,
            startX: 0, fell: false, survived: false };
        };

        const narrow = await makePanel(NARROW_URL, narrowCanvas.current!, NARROW_ACCENT, "narrow policy");
        if (disposed) { sims.forEach((x) => x.dispose()); return; }
        const rand = await makePanel(RAND_URL, randCanvas.current!, RAND_ACCENT, "randomized policy");
        if (disposed) { sims.forEach((x) => x.dispose()); return; }
        const panels = [narrow, rand];

        // Runtime dynamics shift — body 0 is the world; leave it. Mirrors dr.py
        // apply_scales (mass + inertia scaled from the nominal snapshot).
        const applyMass = (p: typeof narrow, scale: number) => {
          for (let i = 1; i < p.nbody; i++) {
            p.model.body_mass[i] = p.nominalMass[i] * scale;
            for (let k = 0; k < 3; k++) p.model.body_inertia[i * 3 + k] = p.nominalInertia[i * 3 + k] * scale;
          }
        };

        let seed = 0;
        let curMass = 1.0;
        const startEpisode = () => {
          seed += 1;
          for (const p of panels) {
            applyMass(p, curMass);           // shift dynamics BEFORE the reset stands it up
            p.env.reset(seed);
            p.s.forward();
            p.startX = p.s.qposAt(p.s.jointQposAdr("root"));
            p.fell = false; p.survived = false;
          }
        };
        // boot: nominal dynamics, a fresh stand for both
        for (const p of panels) { applyMass(p, curMass); p.env.reset(seed); p.startX = p.s.qposAt(p.s.jointQposAdr("root")); }

        const renderPanel = (p: typeof narrow) => {
          const camX = p.s.qposAt(p.s.jointQposAdr("root"));
          drawQuadruped(p.ctx, p.s, {
            W: CANVAS_W, H: CANVAS_H, scale: SCALE, camX, colors: p.colors,
            startX: p.startX, fallen: p.fell, label: p.label,
          });
        };

        setBooted(true);
        panels.forEach(renderPanel);

        apiRef.current = {
          setMass: (m: number) => { curMass = m; startEpisode(); },
          rerun: () => startEpisode(),
        };

        (window as any).__toy = {
          contract: () => ({ narrow: { ...narrow.policy.contract }, rand: { ...rand.policy.contract } }),
          obsParity: () => {
            let m = 0;
            for (const p of panels) {
              const a = p.env.obs(); const b = buildObs(p.s);
              for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k]));
            }
            return { equal: m === 0, maxErr: m };
          },
          setMass: (m: number) => apiRef.current?.setMass(m),
          massOf: () => curMass,
          narrowHeight: () => narrow.env.height(),
          randHeight: () => rand.env.height(),
          async drive(n: number) {
            for (let i = 0; i < n; i++) {
              for (const p of panels) { const o = p.env.obs(); p.env.step(await p.policy.act(o)); }
            }
            return { steps: n, mass: curMass, narrowHeight: narrow.env.height(), randHeight: rand.env.height() };
          },
          fps: () => lastFps,
        };

        let lastFps = 0, frames = 0, fpsMark = performance.now(), last = performance.now(), acc = 0, hudMark = 0;
        // Reduced motion: both panels show their standing still frame and the mass
        // slider + __toy hooks stay live; don't spin the auto-driving rAF loop.
        if (prefersReducedMotion) return;
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
              if (res.done) { if (res.terminated) p.fell = true; else p.survived = true; }
            }
            acc -= CONTROL_DT; n += 1;
            if (disposed) break;
          }
          if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0;

          panels.forEach(renderPanel);

          if (now - hudMark >= 100) {
            hudMark = now;
            const panelHud = (p: typeof narrow): PanelHud => ({
              height: p.env.height(), upZ: p.env.upZ(), fell: p.fell, survived: p.survived, step: p.env.steps,
            });
            setHud({ narrow: panelHud(narrow), rand: panelHud(rand), mass: curMass, fps: lastFps, latMs: narrow.policy.meanLatencyMs() });
          }
        }
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch2.7 dr live toy] failed", err);
        setHud((h) => ({ ...h, error: msg }));
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); sims.forEach((s) => s.dispose()); };
  }, []);

  const failed = !!hud.error;
  const onMass = (e: Event) => {
    const m = parseFloat((e.target as HTMLInputElement).value);
    setMass(m);
    apiRef.current?.setMass(m);
  };
  const onKeyDown = (e: KeyboardEvent) => {
    const api = apiRef.current;
    if (!api) return;
    if (e.key === "ArrowRight" || e.key === "ArrowUp") { e.preventDefault(); const m = Math.min(MASS_MAX, mass + 0.1); setMass(m); api.setMass(m); }
    else if (e.key === "ArrowLeft" || e.key === "ArrowDown") { e.preventDefault(); const m = Math.max(MASS_MIN, mass - 0.1); setMass(m); api.setMass(m); }
    else if (e.key === "r" || e.key === "R") { e.preventDefault(); api.rerun(); }
  };
  const statusText = (p: PanelHud) => (p.fell ? "fell" : p.survived ? "held" : "standing");

  return (
    <div class="dr">
      <figure
        ref={figureRef}
        class="dl-figure"
        tabIndex={0}
        role="application"
        aria-label="Interactive domain-randomization toy. Two four-legged robots under the same shifted physics — one trained narrow, one randomized. Focus here and use the arrow keys, or drag the slider below, to change the mass scale and watch where each holds and where it falls. Press R to re-run at the current mass."
        onKeyDown={onKeyDown}
      >
        <div class="dl-poster" hidden={booted}><Poster /></div>
        <div class="dl-panels" hidden={!booted} aria-hidden="true">
          <div class="dl-panel">
            <canvas ref={narrowCanvas} class="dl-canvas" />
            {booted && !failed && (
              <div class="dl-cap dl-cap--narrow">
                <span class="dl-cap-title">narrow (nominal only)</span>
                <span class="dl-cap-num" data-fell={hud.narrow.fell}>height {hud.narrow.height.toFixed(3)} m · {statusText(hud.narrow)}</span>
              </div>
            )}
          </div>
          <div class="dl-panel">
            <canvas ref={randCanvas} class="dl-canvas" />
            {booted && !failed && (
              <div class="dl-cap dl-cap--rand">
                <span class="dl-cap-title">randomized (across the band)</span>
                <span class="dl-cap-num" data-fell={hud.rand.fell}>height {hud.rand.height.toFixed(3)} m · {statusText(hud.rand)}</span>
              </div>
            )}
          </div>
        </div>

        <div class="bk-sr" aria-live="polite">
          {booted && !failed
            ? `At mass scale ${hud.mass.toFixed(2)}, the narrow policy ${statusText(hud.narrow)} and the randomized policy ${statusText(hud.rand)}. Across most of the range they behave alike; past about 1.2 times mass both fall.`
            : ""}
        </div>

        <div class="dl-status" data-failed={failed} aria-hidden="true">
          {failed ? (
            <span>sim failed — the Colab path covers this without WASM</span>
          ) : booted ? (
            <>
              <span>real dr_narrow.onnx + dr_randomized.onnx · same PPO, different training worlds (contract v1)</span>
              <span>{hud.fps.toFixed(0)} fps · {hud.latMs.toFixed(2)} ms/call</span>
            </>
          ) : (
            <span>booting MuJoCo-WASM + two DR policies…</span>
          )}
        </div>
      </figure>

      <div class="dl-controls">
        <label class="dl-slider-lbl">
          mass scale
          <input
            type="range" class="dl-slider" min={MASS_MIN} max={MASS_MAX} step={0.05} value={mass}
            onInput={onMass} disabled={!booted || failed}
            aria-label="mass scale — the shifted dynamics both policies run under"
          />
          <span class="dl-slider-val">{mass.toFixed(2)}×</span>
        </label>
        <button type="button" class="dl-btn" onClick={() => apiRef.current?.rerun()} disabled={!booted || failed}>
          re-run at this mass
        </button>
        <span class="dl-control-note">
          within-band: DR buys a RANGE, not a stronger robot · poster reads with JS off
        </span>
      </div>
    </div>
  );
}

export default function DomainRandLive() {
  return <DrToy />;
}
