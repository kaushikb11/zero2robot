/**
 * ch2.1 "PPO" — the CARTPOLE recovery concept-toy (`demo: cartpole_ppo_balance`).
 *
 * The chapter's payoff, felt: a REAL ppo_agent.onnx (trained by ppo.py on states
 * IT visited) balances the pole live, and the visitor SHOVES the cart — button,
 * drag, or arrow keys write a velocity impulse to the cart — and watches the
 * policy catch the falling pole and bring it back upright. That recovery is the
 * whole lesson: unlike ch1.1's behavior-cloned PushT policy, which fell apart the
 * moment the visitor dragged the block off the demonstrations' distribution
 * (covariate shift), PPO spent training recovering from exactly these disturbed
 * states, so a shove is still in-distribution. Competence, not fragility.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT documented at the top of
 * ../PlateIsland.tsx (read it before touching this):
 *   1. SSR poster == the JS-off fallback. <Poster/> is pure static JSX (no
 *      window/document at module scope or first render); with JS off it is the
 *      whole experience and no WASM is ever fetched.
 *   2. Lazy, hydration-gated sim. MuJoCo-WASM + onnxruntime are pulled by
 *      *dynamic* import() INSIDE the post-hydration effect (mounted client:visible,
 *      so they load only when scrolled into view). Canvas hidden until booted.
 *   3. Reuse the primitives verbatim: createSim + CARTPOLE_XML + BrowserCartpoleEnv
 *      + buildObs + loadPolicy through the fail-closed assertDrivesCartpole gate.
 *      The driver loop mirrors ppo.py's eval: obs -> policy.act(mean) -> env.step.
 *      Only the RENDER is bespoke (a side-view cart + pole in the page's hues).
 *   4. Make the invisible visible: a live tilt meter (|pole angle| / fall limit)
 *      that stays near zero while balanced and spikes — then falls back — on a shove.
 *   5. ONE interaction (shove the cart), immediate feedback, default-interesting
 *      (boots upright and balancing), plus reset. Keyboard + button path to the aha.
 *   6. Colour discipline: neutral ink for the cart, a warm hue for the pole, ONE
 *      --signal blue for the shove affordance, --alert red only for the leaning/
 *      falling readout, --entity-target green for the balanced return.
 */
import "./cartpole-ppo.css";
import { useEffect, useRef, useState } from "preact/hooks";

// ------------------------------------------------------------------- constants
const MODEL_URL = "/models/ppo_agent.onnx";

// side-view frame: wide (landscape), world +x right, world +z up
const CANVAS_W = 720;
const CANVAS_H = 480;
const VIEW_HALF_W = 2.9; // m — matches the rail half-length; cart limit (2.4) sits inside
const SCALE = CANVAS_W / (2 * VIEW_HALF_W); // px per metre (isotropic)
const GROUND_FRAC = 0.72; // rail height as a fraction down the canvas
const POLE_LEN = 1.0; // m — matches the MJCF capsule length
const CART_HALF_W = 0.16; // m — drawn a touch larger than the 0.1 geom half-width for legibility
const CART_HALF_H = 0.075; // m

const CART_LIMIT = 2.4; // m — cart-off-rail termination (mirror cartpole_obs)
const ANGLE_LIMIT = 0.2095; // rad (~12°) — pole-fall termination
const LEAN_FRAC = 0.35; // fraction of the fall limit (~4.2°) past which we flag "leaning";
                        // above the pole's normal balancing jitter, below a shove's ~7° peak

// A single button/arrow shove the policy reliably RECOVERS from: measured
// in-browser, a 1.0 m/s cart-velocity impulse tips the pole to a clearly-visible
// ~8° peak (well under the 12° fall limit) and the policy catches it every time
// from a settled start. Bigger shoves (or spamming an already-leaning pole) can
// topple it — honest, and reset re-seeds — but the DEFAULT shove is the "watch
// it recover" hero.
const NUDGE_VEL = 1.0; // m/s velocity impulse a button / arrow-key shove adds to the cart
const DRAG_GAIN = 6.0; // m/s of impulse per metre of pointer drag (per move event)
const DRAG_CLAMP = 0.5; // cap on a single drag-move impulse

const MAX_CONTROL_STEPS_PER_FRAME = 6; // cap control-step catch-up per animation frame (mirror main.ts)

// ---------------------------------------------------------- shared SSR poster
// Pure static JSX — the complete experience with JS disabled and the pre-boot
// frame. NOTHING here reads window/document. Same world→px frame as the live
// canvas, so booting causes no reflow.
const POSTER_W = 600;
const POSTER_H = 400;
const PS = POSTER_W / (2 * VIEW_HALF_W);
const PGROUND = POSTER_H * GROUND_FRAC;
const p2s = (wx: number, wz: number): [number, number] => [
  POSTER_W / 2 + wx * PS,
  PGROUND - wz * PS,
];

function Poster() {
  const [railL] = p2s(-CART_LIMIT, 0);
  const [railR] = p2s(CART_LIMIT, 0);
  const [cx, cy] = p2s(0, 0);
  const [tipX, tipY] = p2s(0, POLE_LEN); // upright pole tip
  const cartW = 2 * CART_HALF_W * PS;
  const cartH = 2 * CART_HALF_H * PS;
  return (
    <svg
      class="cp-poster-svg"
      viewBox={`0 0 ${POSTER_W} ${POSTER_H}`}
      role="img"
      aria-label="Side view of a cart on a horizontal rail with a pole hinged upright on top of it. With JavaScript on, a trained reinforcement-learning policy balances the pole live; shove the cart left or right and the policy catches the falling pole and brings it back upright, because it trained on exactly these disturbed states."
    >
      <title>PPO cartpole — the policy recovers from your shove</title>
      <desc>
        A cart slides on a rail and a pole is hinged to it. A policy trained with
        Proximal Policy Optimization keeps the pole upright by pushing the cart. When
        the visitor shoves the cart, the pole tips, and the policy pushes back to
        recover it — the payoff of a policy that learned on the states it itself visits.
      </desc>

      {/* floor + rail */}
      <line class="cp-rail" x1={railL} y1={cy} x2={railR} y2={cy} />
      <line class="cp-rail-end" x1={railL} y1={cy - 8} x2={railL} y2={cy + 8} />
      <line class="cp-rail-end" x1={railR} y1={cy - 8} x2={railR} y2={cy + 8} />

      {/* the pole (upright), hinged at the cart centre */}
      <line class="cp-pole" x1={cx} y1={cy} x2={tipX} y2={tipY} />
      <circle class="cp-pole-bob" cx={tipX} cy={tipY} r={7} />

      {/* the cart */}
      <rect class="cp-cart" x={cx - cartW / 2} y={cy - cartH / 2} width={cartW} height={cartH} rx={3} />
      <circle class="cp-hinge" cx={cx} cy={cy} r={3.4} />

      {/* the one live interaction, as an affordance */}
      <text class="cp-hint" x={cx} y={cy + 44}>← shove the cart →</text>
    </svg>
  );
}

// ------------------------------------------------------------------- live island
interface Hud {
  angleDeg: number;
  ret: number; // episode return (== steps survived, +1/step)
  leaning: boolean;
  recovering: boolean;
  fps: number;
  latMs: number;
  error?: string;
}

type Tok =
  | "--ink" | "--ink-mute" | "--signal" | "--alert"
  | "--entity-pusher" | "--entity-target" | "--rule-strong" | "--paper";
const TOK_FALLBACK: Record<Tok, string> = {
  "--ink": "#23201b",
  "--ink-mute": "#6d6252",
  "--signal": "#1f56de",
  "--alert": "#c0362a",
  "--entity-pusher": "#b0560f",
  "--entity-target": "#0c7d5f",
  "--rule-strong": "#c8bc9e",
  "--paper": "#f7f3ea",
};

function CartpoleToy() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const figureRef = useRef<HTMLElement>(null);
  const apiRef = useRef<{ nudge: (dv: number) => void; reset: () => void } | null>(null);
  const [booted, setBooted] = useState(false);
  const [hud, setHud] = useState<Hud>({
    angleDeg: 0, ret: 0, leaning: false, recovering: false, fps: 0, latMs: 0,
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
        // --- lazy, hydration-gated: the WASM modules are fetched only now ---------
        const [simMod, sceneMod, envMod, obsMod, inferMod, contractsMod] = await Promise.all([
          import("../../../../playground/src/sim/mujoco_sim"),
          import("../../../../playground/src/sim/scene"),
          import("../../../../playground/src/teleop/cartpole_env"),
          import("../../../../playground/src/teleop/cartpole_obs"),
          import("../../../../playground/src/policy/infer"),
          import("../../../../playground/src/policy/contracts"),
        ]);
        const { createSim } = simMod;
        const { CARTPOLE_XML } = sceneMod;
        const { BrowserCartpoleEnv } = envMod;
        const { buildObs, CONTROL_DT, POLE_JOINT, CART_JOINT, ANGLE_LIMIT: OBS_ANGLE_LIMIT } = obsMod;
        const { loadPolicy } = inferMod;
        const { assertDrivesCartpole } = contractsMod;

        // 1) boot the REAL cartpole scene + env
        const realSim: any = await createSim(CARTPOLE_XML);
        sim = realSim;
        if (disposed) { realSim.dispose(); return; }
        const env: any = new BrowserCartpoleEnv(realSim);
        let seed = 3;
        env.reset(seed); // default-interesting: a fresh near-upright start

        const canvas = canvasRef.current!;
        canvas.width = CANVAS_W;
        canvas.height = CANVAS_H;
        const ctx = canvas.getContext("2d")!;

        const groundY = CANVAS_H * GROUND_FRAC;
        const toPx = (wx: number, wz: number): [number, number] => [
          CANVAS_W / 2 + wx * SCALE,
          groundY - wz * SCALE,
        ];

        // a fading shove indicator (signal-blue arrow) after each nudge
        let shoveDir = 0; // -1 | 0 | +1
        let shoveFade = 0; // 0..1

        const render = () => {
          // Resolve the page's design tokens LIVE each frame (a canvas can't read CSS
          // vars natively) so a post-boot theme toggle recolours the instrument rather
          // than freezing the boot-time snapshot. The arena itself stays a warm-light
          // lab surface in BOTH themes — styles.css deliberately pins .cp-figure light.
          const cs = getComputedStyle(canvas);
          const col = (k: Tok) => (cs.getPropertyValue(k).trim() || TOK_FALLBACK[k]);
          const INK = col("--ink"), INK_MUTE = col("--ink-mute"), SIGNAL = col("--signal"),
            ALERT = col("--alert"), POLE_HUE = col("--entity-pusher"), RULE = col("--rule-strong");
          ctx.fillStyle = "#fbf9f3";
          ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

          // faint vertical graph lines (metre marks)
          ctx.save();
          ctx.strokeStyle = RULE;
          ctx.globalAlpha = 0.35;
          ctx.lineWidth = 1;
          ctx.beginPath();
          for (let m = -2; m <= 2; m++) {
            const [gx] = toPx(m, 0);
            ctx.moveTo(gx, 0); ctx.lineTo(gx, CANVAS_H);
          }
          ctx.stroke();
          ctx.restore();

          const cartX = realSim.jointQpos(CART_JOINT);
          const theta = realSim.jointQpos(POLE_JOINT); // raw hinge angle (smooth render)
          const leaning = Math.abs(env.angle()) > OBS_ANGLE_LIMIT * LEAN_FRAC;

          // rail (neutral rule) with end-stops at the cart limit
          const [rlx, ry] = toPx(-CART_LIMIT, 0);
          const [rrx] = toPx(CART_LIMIT, 0);
          ctx.strokeStyle = RULE;
          ctx.lineWidth = 3;
          ctx.beginPath(); ctx.moveTo(rlx, ry); ctx.lineTo(rrx, ry); ctx.stroke();
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(rlx, ry - 10); ctx.lineTo(rlx, ry + 10);
          ctx.moveTo(rrx, ry - 10); ctx.lineTo(rrx, ry + 10); ctx.stroke();

          const [cxp, cyp] = toPx(cartX, 0);
          const [tipX, tipY] = toPx(cartX + POLE_LEN * Math.sin(theta), POLE_LEN * Math.cos(theta));

          // the pole — warm hue, red when leaning past the safe band
          ctx.save();
          ctx.strokeStyle = leaning ? ALERT : POLE_HUE;
          ctx.lineWidth = 7;
          ctx.lineCap = "round";
          ctx.beginPath(); ctx.moveTo(cxp, cyp); ctx.lineTo(tipX, tipY); ctx.stroke();
          ctx.fillStyle = leaning ? ALERT : POLE_HUE;
          ctx.beginPath(); ctx.arc(tipX, tipY, 8, 0, Math.PI * 2); ctx.fill();
          ctx.restore();

          // the cart — neutral ink box straddling the rail
          const cw = 2 * CART_HALF_W * SCALE, ch = 2 * CART_HALF_H * SCALE;
          ctx.fillStyle = INK;
          ctx.beginPath();
          ctx.roundRect(cxp - cw / 2, cyp - ch / 2, cw, ch, 4);
          ctx.fill();
          // hinge dot
          ctx.fillStyle = INK_MUTE;
          ctx.beginPath(); ctx.arc(cxp, cyp, 4, 0, Math.PI * 2); ctx.fill();

          // the shove indicator — the one LIVE handle (signal blue), fading out
          if (shoveFade > 0.01 && shoveDir !== 0) {
            const ax = cxp + shoveDir * (cw / 2 + 10);
            ctx.save();
            ctx.globalAlpha = Math.min(1, shoveFade);
            ctx.strokeStyle = SIGNAL;
            ctx.fillStyle = SIGNAL;
            ctx.lineWidth = 3;
            const len = 26;
            ctx.beginPath(); ctx.moveTo(ax, cyp); ctx.lineTo(ax + shoveDir * len, cyp); ctx.stroke();
            const hx = ax + shoveDir * len;
            ctx.beginPath();
            ctx.moveTo(hx, cyp);
            ctx.lineTo(hx - shoveDir * 8, cyp - 5);
            ctx.lineTo(hx - shoveDir * 8, cyp + 5);
            ctx.closePath(); ctx.fill();
            ctx.restore();
            shoveFade *= 0.9;
          }
        };

        // 2) load the REAL policy through the fail-closed contract gate FIRST, then
        //    reveal the canvas — load-then-boot, so a fetch/contract failure keeps
        //    booted=false and the captioned SSR poster stays up (never a frozen canvas).
        const policy = await loadPolicy(MODEL_URL);
        assertDrivesCartpole(policy.contract);
        if (disposed) return;
        setBooted(true);
        render();

        // --- interaction state (refs, not React state — the loop must not re-render)
        let epReturn = 0;
        let nudgedAt = -1e9;
        const nudge = (dv: number) => {
          env.nudgeCart(dv);
          shoveDir = Math.sign(dv);
          shoveFade = 1;
          nudgedAt = performance.now();
        };
        apiRef.current = {
          nudge,
          reset: () => { env.reset(++seed); epReturn = 0; nudgedAt = -1e9; shoveFade = 0; },
        };

        // drag-to-shove: each horizontal pointer move writes a velocity impulse
        let dragging = false, lastClientX = 0;
        const onDown = (e: PointerEvent) => {
          dragging = true; lastClientX = e.clientX;
          canvas.dataset.dragging = "true";
          canvas.setPointerCapture(e.pointerId);
          e.preventDefault();
        };
        const onMove = (e: PointerEvent) => {
          if (!dragging) return;
          const rect = canvas.getBoundingClientRect();
          const dxWorld = ((e.clientX - lastClientX) / rect.width) * (2 * VIEW_HALF_W);
          lastClientX = e.clientX;
          const dv = Math.max(-DRAG_CLAMP, Math.min(DRAG_CLAMP, dxWorld * DRAG_GAIN));
          if (dv !== 0) nudge(dv);
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

        // 3) headless-verification hooks (mirror playground's window.__policy) so a
        //    browser driver can PROVE the policy drives + the obs matches training.
        (window as any).__toy = {
          contract: () => ({ ...policy.contract }),
          obsParity: () => {
            const a = env.obs(); const b = buildObs(realSim);
            let m = 0; for (let k = 0; k < a.length; k++) m = Math.max(m, Math.abs(a[k] - b[k]));
            return { equal: m === 0, maxErr: m };
          },
          angleDeg: () => (env.angle() * 180) / Math.PI,
          episodeReturn: () => epReturn,
          nudge: (dv: number) => nudge(dv),
          reset: () => apiRef.current?.reset(),
          async drive(n: number) {
            let survived = 0;
            for (let i = 0; i < n; i++) {
              const o = env.obs();
              const res = env.step(await policy.act(o));
              if (!res.done) survived += 1; else env.reset(++seed);
            }
            return { steps: n, survived, meanLatencyMs: policy.meanLatencyMs(), calls: policy.calls };
          },
          fps: () => lastFps,
        };

        // 4) DRIVE — real-time paced to the 50 Hz control rate (mirrors ppo.py eval:
        //    act with the policy MEAN, no sampling; the ONNX wraps actor_mean).
        let lastFps = 0, frames = 0, fpsMark = performance.now(), last = performance.now(),
          acc = 0, hudMark = 0;

        // Reduced motion: the balanced still frame is painted and the shove/reset
        // controls + __toy hooks stay live; don't spin the auto-driving rAF loop.
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
            const obs = env.obs();                 // buildObs(sim) — training obs, verbatim
            const action = await policy.act(obs);  // deterministic mean action[1]
            const res = env.step(action);          // mirrors ppo.py eval: env.step(policy(obs))
            epReturn += res.reward;
            acc -= CONTROL_DT; n += 1;
            if (res.done) { env.reset(++seed); epReturn = 0; nudgedAt = -1e9; }
            if (disposed) break;
          }
          if (n === MAX_CONTROL_STEPS_PER_FRAME) acc = 0;

          render();

          if (now - hudMark >= 100) { // throttle HUD to ~10 Hz
            hudMark = now;
            const angle = env.angle();
            const leaning = Math.abs(angle) > OBS_ANGLE_LIMIT * LEAN_FRAC;
            setHud({
              angleDeg: (angle * 180) / Math.PI,
              ret: epReturn,
              leaning,
              recovering: leaning && now - nudgedAt < 2000,
              fps: lastFps,
              latMs: policy.meanLatencyMs(),
            });
          }
        }

        canvas.removeEventListener("pointerdown", onDown);
        canvas.removeEventListener("pointermove", onMove);
        canvas.removeEventListener("pointerup", onUp);
        canvas.removeEventListener("pointercancel", onUp);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        console.error("[ch2.1 toy] failed", err);
        setHud((h) => ({ ...h, error: msg }));
      }
    })();

    return () => { disposed = true; cancelAnimationFrame(raf); if (sim) sim.dispose(); };
  }, []);

  const failed = !!hud.error;
  const tiltPct = Math.min(100, (Math.abs(hud.angleDeg) / ((ANGLE_LIMIT * 180) / Math.PI)) * 100);

  const onKeyDown = (e: KeyboardEvent) => {
    const api = apiRef.current;
    if (!api) return;
    if (e.key === "ArrowLeft") { e.preventDefault(); api.nudge(-NUDGE_VEL); }
    else if (e.key === "ArrowRight") { e.preventDefault(); api.nudge(NUDGE_VEL); }
    else if (e.key === "r" || e.key === "R") api.reset();
  };

  return (
    <div class="cp">
      <figure
        ref={figureRef}
        class="cp-figure"
        tabIndex={0}
        role="application"
        aria-label="Interactive PPO cartpole toy. A trained policy balances the pole upright. Shove the cart with the buttons below, by dragging it, or — with this figure focused — the left and right arrow keys, and watch the policy catch the pole and recover. Press R to reset."
        onKeyDown={onKeyDown}
      >
        {/* SSR poster — the JS-off experience and the pre-boot frame */}
        <div class="cp-poster" hidden={booted}><Poster /></div>

        {/* live MuJoCo-WASM canvas — shown once booted */}
        <canvas ref={canvasRef} class="cp-canvas" hidden={!booted} aria-hidden="true" />

        {/* Non-visual path to the same aha: announce only the qualitative balance/
            recovery transition (not the per-frame angle). The visual HUD is aria-hidden. */}
        <div class="bk-sr" aria-live="polite">
          {booted && !failed
            ? hud.recovering
              ? "You shoved the cart. The policy is catching the falling pole and pushing the cart to bring it back upright — the states your shove created are ones it trained on."
              : "The policy is balancing the pole upright. Shove the cart to disturb it."
            : ""}
        </div>

        {/* live HUD — pole angle + episode return (invisible made visible) */}
        {booted && !failed && (
          <div class="cp-hud" aria-hidden="true">
            <div class="cp-hud-row">
              <span class="cp-k">pole angle</span>
              <span class={`cp-v ${hud.leaning ? "cp-bad" : "cp-ok"}`}>
                {hud.angleDeg >= 0 ? "+" : ""}{hud.angleDeg.toFixed(1)}° {hud.leaning ? "▲" : "✓"}
              </span>
            </div>
            <div class="cp-meter">
              <div class="cp-meter-fill" data-lean={hud.leaning} style={`width:${tiltPct}%`} />
            </div>
            <div class="cp-hud-row">
              <span class="cp-k">episode_return</span>
              <span class="cp-v cp-ret">{hud.ret.toFixed(0)}</span>
            </div>
            <div class="cp-hud-row">
              <span class="cp-k">status</span>
              <span class={`cp-v ${hud.leaning ? "cp-bad" : "cp-ok"}`}>
                {hud.recovering ? "recovering" : hud.leaning ? "leaning" : "balancing"}
              </span>
            </div>
          </div>
        )}

        {/* boot / instrument status line */}
        <div class="cp-status" data-failed={failed} aria-hidden="true">
          {failed ? (
            <span>sim failed — the Colab path covers this without WASM</span>
          ) : booted ? (
            <>
              <span>real ppo_agent.onnx · trained on states it visited</span>
              <span>{hud.fps.toFixed(0)} fps · {hud.latMs.toFixed(2)} ms/call</span>
            </>
          ) : (
            <span>booting MuJoCo-WASM + policy…</span>
          )}
        </div>
      </figure>

      <div class="cp-controls">
        <button type="button" class="cp-btn cp-btn--primary" onClick={() => apiRef.current?.nudge(-NUDGE_VEL)} disabled={!booted || failed}>
          ← shove
        </button>
        <button type="button" class="cp-btn cp-btn--primary" onClick={() => apiRef.current?.nudge(NUDGE_VEL)} disabled={!booted || failed}>
          shove →
        </button>
        <button type="button" class="cp-btn" onClick={() => apiRef.current?.reset()} disabled={!booted || failed}>
          reset
        </button>
        <span class="cp-control-note">
          shove the cart (or arrow-keys when focused) · it recovers · poster reads with JS off
        </span>
      </div>
    </div>
  );
}

export default function CartpolePpo() {
  return <CartpoleToy />;
}
