/**
 * ch3.9 "Plan Through Your Engine: Sampling-Based MPC (CEM/MPPI)" concept-toy
 * (`demo: mpc_fanout`). A recorded-data sibling of ch1.5's flow-ring and ch3.3's
 * energy-drift toys: sampling-based MPC, made visual. NO WASM, NO ONNX — pure SVG
 * over curves mpc.py itself emitted (seed 0, cpu, mppi). The one thing to FEEL:
 * control WITHOUT learning — swing-up solved by SEARCH, re-solved every step.
 *
 * TWO panels, one control:
 *   1. THE FAN-OUT (hero). At a single planning moment (executed step 6, the pole
 *      hanging DOWN) the 18 sampled action-sequences rolled forward through the
 *      model — their pole-tip paths fanning out, coloured by cost (cheap = good =
 *      bright, expensive = faint) — with the MPPI-chosen plan drawn boldly on top.
 *      That single picture IS sampling-based MPC: sample plans, roll each through
 *      the model, keep the good ones, commit to the first action of the best.
 *   2. THE SWING-UP (timeseries). The realised episode's upright-cos climbing from
 *      -1 (hanging down) to +1 (balanced) and staying there — vs the `--break`
 *      myopic run (H=3) that never comes up. Zero training runs produced either.
 *
 * THE CONTROL — look-ahead horizon (H=25 plan-far ↔ H=3 myopic). Both are REAL
 * measured runs. Dragging to the short horizon makes the fan go BLIND — the plans
 * clip to the first few steps (the planner can no longer see that letting the pole
 * fall further NOW buys the momentum to come up LATER) and the swing-up curve
 * flatlines at the bottom. That is the honest ceiling: MPC needs a good model AND
 * enough look-ahead/compute per step. Here the model IS the world (perfect); with a
 * sim-to-sim gap the plan would optimise an imagined trajectory reality won't follow.
 *
 * Like ch1.5 this needs NO WASM: it renders REAL, pre-computed geometry from mpc.py
 * (--method mppi --horizon 25 --samples 64) via the co-located vizdata.json, so it
 * is pure SVG + design tokens: theme-aware for free, and the server-rendered default
 * (H=25) IS the JS-off experience — only the horizon control goes inert without
 * hydration. Colour by MEANING: blue = the sampled candidates (search); gold =
 * the committed MPPI plan and the swing-up it produces; red = the myopic --break.
 */
import "./MpcPlanToy.css";
import { useState } from "preact/hooks";
// Real geometry from mpc.py, seed 0, cpu, mppi — see the file's `provenance` and
// the generator site/scripts/vizdata/ch3.9_mpc.py. Committed small text, no binary.
import viz from "../../../../curriculum/phase3_advanced/ch3.9_mpc/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
interface Rollout { cost: number; tip: number[][]; }
interface Episode { t: number; cart_x: number; pole_angle: number; cos: number; action: number; }
interface Viz {
  upright_threshold: number;
  planning_moment: {
    step: number; start_tip: number[]; cost_min: number; cost_max: number;
    rollouts: Rollout[]; chosen_plan: number[][];
  };
  episode: Episode[];
  break_horizon_cos: number[];
  summary: {
    mppi_mean_cost: number; mppi_upright_frac: number;
    random_mean_cost: number; random_upright_frac: number;
    break_horizon_upright_frac: number;
  };
}

const V = viz as unknown as Viz;
const PM = V.planning_moment;
const ROLLOUTS = PM.rollouts;
const CHOSEN = PM.chosen_plan;
const START = PM.start_tip;
const COST_MIN = PM.cost_min;
const COST_MAX = PM.cost_max;
const HORIZON = ROLLOUTS[0].tip.length - 1;   // 25 planning steps
const MYOPIC = 3;                             // the --break horizon
const EPISODE = V.episode;
const BREAK = V.break_horizon_cos;
const UP = V.upright_threshold;               // 0.9 — the "upright" cos line
const SUM = V.summary;
const N = EPISODE.length;
const COS = EPISODE.map((e) => e.cos);
// first step the planning run actually reaches upright (cos ≥ threshold)
const CROSS = COS.findIndex((c) => c >= UP);

type Mode = "plan" | "myopic";

// ---------------------------------------------------------- fan-out geometry
// The pole-tip paths live in a (lateral, height) plane where height −1 = hanging
// straight DOWN and +1 = balanced UP. Fit the bounding box of every rollout (full
// horizon, so the viewBox stays put when the horizon control clips the draw) with
// an EQUAL x/y scale — the geometry must not be distorted.
const FW = 380, FH = 300;
const FPAD = { l: 18, r: 18, t: 20, b: 30 };

const bbox = (() => {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const acc = (p: number[]) => {
    if (p[0] < minX) minX = p[0];
    if (p[0] > maxX) maxX = p[0];
    if (p[1] < minY) minY = p[1];
    if (p[1] > maxY) maxY = p[1];
  };
  ROLLOUTS.forEach((r) => r.tip.forEach(acc));
  CHOSEN.forEach(acc);
  acc(START);
  return { minX, maxX, minY, maxY };
})();

const FSCALE = Math.min(
  (FW - FPAD.l - FPAD.r) / (bbox.maxX - bbox.minX),
  (FH - FPAD.t - FPAD.b) / (bbox.maxY - bbox.minY),
);
const FOFFX = FPAD.l + ((FW - FPAD.l - FPAD.r) - (bbox.maxX - bbox.minX) * FSCALE) / 2;
const FOFFY = FPAD.t + ((FH - FPAD.t - FPAD.b) - (bbox.maxY - bbox.minY) * FSCALE) / 2;
// data (x, height) → svg px, with height UP mapping to svg UP (flip y)
const fmap = (x: number, y: number): [number, number] => [
  FOFFX + (x - bbox.minX) * FSCALE,
  FOFFY + (bbox.maxY - y) * FSCALE,
];

const clip = (tip: number[][], mode: Mode) =>
  mode === "myopic" ? tip.slice(0, MYOPIC + 1) : tip;

const poly = (tip: number[][]) =>
  tip.map(([x, y]) => { const [sx, sy] = fmap(x, y); return `${sx.toFixed(1)},${sy.toFixed(1)}`; }).join(" ");

// cost → visual weight: cheap (good) reads bright & thick, expensive fades back
function costStyle(cost: number) {
  const t = Math.max(0, Math.min(1, (cost - COST_MIN) / (COST_MAX - COST_MIN)));
  return { opacity: 0.82 - 0.6 * t, width: 1.9 - 0.75 * t };
}
// draw expensive first (behind, faint), cheapest last (on top, under the chosen plan)
const FAN = [...ROLLOUTS].sort((a, b) => b.cost - a.cost);

// ================================================================ FAN-OUT PANEL
function FanOut({ mode }: { mode: Mode }) {
  const chosen = clip(CHOSEN, mode);
  const [scx, scy] = fmap(START[0], START[1]);
  const upTip = fmap(bbox.minX - 0.01, bbox.maxY); // orientation anchor (top-left)

  return (
    <svg
      class="mp-svg mp-fan-svg"
      viewBox={`0 0 ${FW} ${FH}`}
      role="img"
      aria-label={
        `Sampling-based MPC fan-out at one planning step, with the cart-pole hanging down. ` +
        `${ROLLOUTS.length} sampled action-sequences are rolled through the model; their pole-tip paths fan out from the start, ` +
        `coloured by cost — cheap plans bright, expensive plans faint — and the MPPI-chosen plan is drawn boldly in gold on top. ` +
        (mode === "myopic"
          ? `The look-ahead horizon is clipped to ${MYOPIC} steps: the plans stop short near the bottom, blind to the swing that pays off later.`
          : `The look-ahead horizon is the full ${HORIZON} steps; height rises from −1 (hanging down) toward +1 (balanced).`)
      }
    >
      {/* orientation: which way is "up" (toward balanced) */}
      <line class="mp-fan-guide" x1={upTip[0]} y1={FH - FPAD.b + 6} x2={upTip[0]} y2={FPAD.t - 6} />
      <text class="mp-fan-orient" x={upTip[0] + 4} y={FPAD.t + 2}>↑ upright</text>
      <text class="mp-fan-orient" x={upTip[0] + 4} y={FH - FPAD.b + 2}>↓ hanging down</text>

      {/* the sampled candidate rollouts — the search, coloured by cost */}
      {FAN.map((r) => {
        const s = costStyle(r.cost);
        return (
          <polyline
            class="mp-cand"
            points={poly(clip(r.tip, mode))}
            style={`opacity:${s.opacity.toFixed(2)};stroke-width:${s.width.toFixed(2)}`}
          />
        );
      })}

      {/* the MPPI-chosen plan — bold gold on top, with a soft underlay */}
      <polyline class="mp-chosen-glow" points={poly(chosen)} />
      <polyline class="mp-chosen" points={poly(chosen)} />

      {/* the shared planning state (pole hanging down) + the chosen plan's endpoint */}
      <circle class="mp-start" cx={scx.toFixed(1)} cy={scy.toFixed(1)} r={3.4} />
      {(() => {
        const end = chosen[chosen.length - 1];
        const [ex, ey] = fmap(end[0], end[1]);
        return <circle class="mp-chosen-end" cx={ex.toFixed(1)} cy={ey.toFixed(1)} r={3} />;
      })()}
    </svg>
  );
}

// ================================================================ SWING-UP PANEL
const TW = 470, TH = 210;
const TPAD = { l: 40, r: 14, t: 16, b: 26 };
const tpx = (i: number) => TPAD.l + (i / (N - 1)) * (TW - TPAD.l - TPAD.r);
const tpy = (c: number) => TPAD.t + ((1 - c) / 2) * (TH - TPAD.t - TPAD.b);
const tline = (cos: number[]) => cos.map((c, i) => `${tpx(i).toFixed(1)},${tpy(c).toFixed(1)}`).join(" ");

function SwingUp({ mode }: { mode: Mode }) {
  const yUp = tpy(UP);
  const yZero = tpy(0);
  const xTicks = [0, 30, 60, 90, N - 1];
  const [cx, cy] = [tpx(CROSS), tpy(COS[CROSS])];

  return (
    <svg
      class="mp-svg"
      viewBox={`0 0 ${TW} ${TH}`}
      role="img"
      aria-label={
        `Swing-up outcome: upright-cos versus control step, from −1 (hanging down) to +1 (balanced). ` +
        `The gold planning run (look-ahead ${HORIZON}) climbs past the upright line and holds — it reaches upright at step ${CROSS}. ` +
        `The red myopic run (look-ahead ${MYOPIC}) never leaves the bottom. ` +
        `Currently the ${mode === "plan" ? "planning" : "myopic"} run is highlighted.`
      }
    >
      {/* upright threshold + zero gridlines */}
      <line class="mp-up-line" x1={TPAD.l} y1={yUp} x2={TW - TPAD.r} y2={yUp} />
      <text class="mp-up-lab" x={TW - TPAD.r} y={yUp - 4} text-anchor="end">upright (cos ≥ {UP})</text>
      <line class="mp-grid" x1={TPAD.l} y1={yZero} x2={TW - TPAD.r} y2={yZero} />

      {/* y axis ticks */}
      {[1, 0, -1].map((c) => (
        <text class="mp-tick" x={TPAD.l - 6} y={tpy(c) + 3} text-anchor="end">{c > 0 ? "+1" : c}</text>
      ))}
      <text class="mp-axis-title" x={TPAD.l - 34} y={TPAD.t + 2}>cos</text>

      {/* x axis */}
      <line class="mp-axis" x1={TPAD.l} y1={TH - TPAD.b} x2={TW - TPAD.r} y2={TH - TPAD.b} />
      {xTicks.map((t) => (
        <text class="mp-tick" x={tpx(t)} y={TH - TPAD.b + 13} text-anchor="middle">{t}</text>
      ))}
      <text class="mp-axis-title" x={TW - TPAD.r} y={TH - 3} text-anchor="end">control step →</text>

      {/* the two REAL runs — inactive one faint behind, active one bold */}
      <polyline class={`mp-run mp-run-break ${mode === "myopic" ? "is-active" : ""}`} points={tline(BREAK)} />
      <polyline class={`mp-run mp-run-plan ${mode === "plan" ? "is-active" : ""}`} points={tline(COS)} />

      {/* mark where the planning run first reaches upright */}
      {mode === "plan" && CROSS >= 0 && (
        <>
          <circle class="mp-cross" cx={cx.toFixed(1)} cy={cy.toFixed(1)} r={3.4} />
          <text class="mp-cross-lab" x={cx.toFixed(1)} y={(cy - 8).toFixed(1)} text-anchor="middle">step {CROSS}</text>
        </>
      )}
    </svg>
  );
}

// ==================================================================== THE ISLAND
function MpcPlanToy() {
  const [mode, setMode] = useState<Mode>("plan");
  const isMyopic = mode === "myopic";

  // arrow keys toggle the horizon too (a11y — the whole aha without the pointer)
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "ArrowLeft" || e.key === "Home") { e.preventDefault(); setMode("myopic"); }
    else if (e.key === "ArrowRight" || e.key === "End") { e.preventDefault(); setMode("plan"); }
    else if (e.key === " " || e.key === "Enter") { e.preventDefault(); setMode(isMyopic ? "plan" : "myopic"); }
  };

  const announce = isMyopic
    ? `Look-ahead horizon ${MYOPIC} (the myopic --break): the planner sees only a few steps, blind to the swing that pays off later. ` +
      `The pole never reaches upright — the swing-up curve flatlines at the bottom. Zero of its steps are upright.`
    : `Look-ahead horizon ${HORIZON}: the sampled plans fan out and MPPI commits to the best. ` +
      `The pole swings up, reaching upright at step ${CROSS} and holding there. Mean plan cost ${SUM.mppi_mean_cost}.`;

  return (
    <div class="mp">
      {/* --- the one control: the look-ahead horizon --- */}
      <div
        class="mp-controls"
        role="group"
        aria-label="Look-ahead horizon — drag it down to the myopic break and the plan goes blind"
        tabIndex={0}
        onKeyDown={onKeyDown}
      >
        <span class="mp-seg-label">look-ahead <b>horizon</b></span>
        <div class="mp-seg">
          <button
            type="button"
            class="mp-seg-btn"
            aria-pressed={isMyopic}
            onClick={() => setMode("myopic")}
          >
            H={MYOPIC} · myopic
          </button>
          <button
            type="button"
            class="mp-seg-btn"
            aria-pressed={!isMyopic}
            onClick={() => setMode("plan")}
          >
            H={HORIZON} · plan far
          </button>
        </div>
        <span class="mp-seg-hint">
          {isMyopic ? "the --break: too short to see the swing pay off" : "far enough to see falling now buys the swing later"}
        </span>
      </div>

      {/* --- the two panels: the fan-out + the swing-up outcome --- */}
      <div class="mp-panels">
        <figure class="mp-panel">
          <figcaption class="mp-cap">
            <span>the fan-out · one plan step</span>
            <b>{ROLLOUTS.length} samples · H={isMyopic ? MYOPIC : HORIZON}</b>
          </figcaption>
          <FanOut mode={mode} />
          <div class="mp-legend" aria-hidden="true">
            <span class="mp-lg mp-lg-cand">sampled plans <em>(cheap = bright)</em></span>
            <span class="mp-lg mp-lg-chosen">MPPI-chosen plan</span>
          </div>
        </figure>

        <figure class="mp-panel">
          <figcaption class="mp-cap">
            <span>the swing-up · realised episode</span>
            <b class={isMyopic ? "mp-bad" : "mp-good"}>{isMyopic ? "never upright" : `upright @ step ${CROSS}`}</b>
          </figcaption>
          <SwingUp mode={mode} />
          <div class="mp-readout" aria-hidden="true">
            <div class="mp-row">
              <span class="mp-k mp-k-plan">plan far (H={HORIZON})</span>
              <span class="mp-v mp-good">swings up · cost {SUM.mppi_mean_cost}</span>
            </div>
            <div class="mp-row">
              <span class="mp-k mp-k-break">myopic (H={MYOPIC})</span>
              <span class="mp-v mp-bad">never comes up · {Math.round(SUM.break_horizon_upright_frac * 100)}% upright</span>
            </div>
            <div class="mp-row">
              <span class="mp-k mp-k-rand">random search</span>
              <span class="mp-v mp-muted">cost {SUM.random_mean_cost} · {Math.round(SUM.random_upright_frac * 100)}% upright</span>
            </div>
          </div>
        </figure>
      </div>

      {/* non-visual path to the same aha — the qualitative outcome, not per-frame spam */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      <p class="mp-note">
        This is <b>planning through a model with zero learning</b> — no policy is trained. Every control step re-solves swing-up by
        sampling action sequences, rolling each through the model, and committing to the first action of the best. The honest ceiling:
        it needs a <b>good model and compute every step</b>, and here the model <b>is</b> the world (the best case) — with a sim-to-sim
        gap the plan would optimise an imagined trajectory reality won't follow. Real geometry from mpc.py (seed 0, mppi); poster reads with JS off.
      </p>
    </div>
  );
}

export default function MpcPlan() {
  return <MpcPlanToy />;
}
