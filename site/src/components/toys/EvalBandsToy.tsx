/**
 * EvalBandsToy — ch1.6 "Evaluation Is Hard" FLAGSHIP concept-toy (demo id
 * `harness_eval_bands`). The chapter's whole thesis — "single numbers lie" —
 * made interactive: drag ONE slider (episodes per policy, N) and watch two
 * success-rate confidence intervals go from OVERLAPPING at N≈20 (the ranking is
 * not established) to SEPARATED at N=200 (the gap is real). Same two policies,
 * same rollouts; the only thing that changed is how many episodes you counted.
 *
 * This follows the FROZEN CONCEPT-TOY CONTRACT documented at the top of
 * ../PlateIsland.tsx (SSR figure == JS-off fallback in ONE stable plate so
 * booting causes no reflow; ONE control + live readouts + default-interesting;
 * colour discipline; keyboard + aria-live + reduced-motion). Unlike ch1.1 it
 * needs NO MuJoCo-WASM and NO vizdata.json — the Wilson score interval and the
 * Newcombe difference interval are CLOSED-FORM, so the maths below is
 * harness.py's stats region re-implemented from scratch in TypeScript and the
 * figure recomputes every band client-side as you drag.
 *
 * HONESTY — every number is measured (curriculum/.../ch1.6_harness/meta.yaml,
 * reference_run, seed 0 / cpu, 2026-07-06):
 *   • The two policies' best-estimate (pooled, N=200) success rates are the
 *     FIXED centres: strong k=45/200 = 0.225, weak k=21/200 = 0.105. The slider
 *     varies N; the band around each fixed estimate is the 95% Wilson interval
 *     at that N (the standard "your error bar shrinks as you gather episodes").
 *   • At N=200 the toy reproduces the measured Wilson intervals EXACTLY —
 *     Wilson(0.225,200)=[0.172624,0.287741], Wilson(0.105,200)=[0.069707,
 *     0.155180] — and the measured pooled difference CI [+0.047,+0.192]
 *     (excludes 0 → ranking significant).
 *   • The from-scratch maths is verified against harness.py's own self-check:
 *     Wilson(0,10)=[0,0.2775] and Wilson(5,10)=[0.2366,0.7634] (meta ex3), and
 *     the N=20 suite-0 anchor diff_ci(6/20, 4/20)=[-0.165595,+0.349383].
 *   • The measured swing is the WHY the N=20 band is so wide: across ten
 *     20-episode suites the strong policy's per-suite rate ranged 0.05..0.30
 *     (std 0.087) — one 20-episode suite happened to report 0.30 vs 0.20.
 */
import { useEffect, useState } from "preact/hooks";
import "./eval-bands.css";

// ---------------------------------------------------------------- from-scratch stats
// Mirror of curriculum/phase1_imitation/ch1.6_harness/harness.py's `stats` region,
// numpy/math → TypeScript, SAME closed forms. No table, no library: the point is to
// SEE the formula (harness.py: "no scipy hiding the CI math").
const Z95 = 1.959963985; // the 0.975 standard-normal quantile (95% two-sided)

/** 95% Wilson score interval for a proportion p_hat observed over n trials.
 *  Identical algebra to harness.py's wilson_ci(k, n) with p_hat = k/n; taking
 *  p_hat as a continuous input lets the slider hold each policy's estimate fixed
 *  and vary n. Wilson solves p from an implicit quadratic (not read off like the
 *  naive Wald interval), so it stays inside [0,1] and never collapses to zero
 *  width at the boundary. */
function wilson(pHat: number, n: number, z: number = Z95): [number, number] {
  if (n <= 0) return [0, 1];
  const denom = 1 + (z * z) / n;
  const center = (pHat + (z * z) / (2 * n)) / denom;
  const half = (z / denom) * Math.sqrt((pHat * (1 - pHat)) / n + (z * z) / (4 * n * n));
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

/** Newcombe hybrid-score interval for the difference p_a − p_b (Newcombe 1998,
 *  method 10) — harness.py's diff_ci, built from the two Wilson intervals. This
 *  is the VERDICT on "is A really better than B": if it excludes 0 the ranking is
 *  significant at this n; if it contains 0 the ranking is not established, however
 *  the point estimates look. (Whether the two one-sample bars merely overlap is a
 *  cruder, more conservative eyeball — the difference interval is the one to report.) */
function diffCI(pA: number, nA: number, pB: number, nB: number, z: number = Z95): [number, number] {
  const [loA, hiA] = wilson(pA, nA, z);
  const [loB, hiB] = wilson(pB, nB, z);
  const d = pA - pB;
  const lo = d - Math.sqrt((pA - loA) ** 2 + (hiB - pB) ** 2);
  const hi = d + Math.sqrt((hiA - pA) ** 2 + (pB - loB) ** 2);
  return [lo, hi];
}

// ---------------------------------------------------------------- measured constants
// meta.yaml reference_run (seed 0, cpu): the pooled best estimates of each policy's
// true success rate. These are the FIXED centres; the slider varies N around them.
const STRONG_RATE = 0.225; // k=45 of n=200
const WEAK_RATE = 0.105; //   k=21 of n=200
const TRUE_GAP = STRONG_RATE - WEAK_RATE; // +0.12 — the real difference the CI resolves

const N_MIN = 10;
const N_MAX = 300;
const N_STEP = 5;
const N_DEFAULT = 20; // default-interesting: boots where the ranking is NOT established,
//                       so the first drag toward 200 is the aha (the bands separate).
const N_POOLED = 200; // the meta anchor: pooled N where the gap becomes significant.

// The two teaching thresholds the maths crosses as N grows (both derived live below;
// listed here only to document the pedagogy): the difference CI first excludes 0 at
// N=75, while the cruder "do the raw bars overlap" test doesn't clear until N≈148 —
// the zone where the eyeball still says "too close" but the difference interval has
// already resolved the ranking. That disagreement IS the chapter's point.

// ---------------------------------------------------------------- figure geometry
// One landscape instrument plate. World = (success rate) on a shared x-axis for the
// two policy bars, then a zero-centred (difference) axis below. viewBox px; the CSS
// scales it to width:100%. No window/document here → SSR-safe.
const VW = 720;
const VH = 296;
const PAD_L = 92; // room for the row labels
const PAD_R = 44;
const PLOT_L = PAD_L;
const PLOT_R = VW - PAD_R;
const PLOT_W = PLOT_R - PLOT_L;

const RATE_MAX = 0.6; // rate axis 0..0.6 (strong's widest upper CI ≈0.53 at N=10 fits)
const DIFF_MIN = -0.3; // difference axis −0.3..+0.5 (spans every band across the range)
const DIFF_MAX = 0.5;

const rateX = (r: number): number => PLOT_L + (r / RATE_MAX) * PLOT_W;
const diffX = (d: number): number => PLOT_L + ((d - DIFF_MIN) / (DIFF_MAX - DIFF_MIN)) * PLOT_W;

const Y_STRONG = 54; // centre y of the strong bar
const Y_WEAK = 100; // centre y of the weak bar
const BAR_H = 20;
const Y_RATE_AXIS = 136;
const Y_DIFF_LABEL = 176;
const Y_DIFF = 214; // centre y of the difference bar
const Y_DIFF_AXIS = 250;

const RATE_TICKS = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
const DIFF_TICKS = [-0.2, 0, 0.2, 0.4];

const fmt = (n: number, d = 3): string => (Object.is(n, -0) ? 0 : n).toFixed(d);
const fmtSigned = (n: number, d = 2): string => (n >= 0 ? "+" : "") + (Object.is(n, -0) ? 0 : n).toFixed(d);

// ---------------------------------------------------------------- derived scene
interface Bands {
  strong: [number, number];
  weak: [number, number];
  diff: [number, number];
  significant: boolean; // difference CI excludes 0 (the rigorous verdict)
  barsOverlap: boolean; // the two one-sample CIs still share values (crude eyeball)
  overlapLo: number;
  overlapHi: number;
}

function computeBands(n: number): Bands {
  const strong = wilson(STRONG_RATE, n);
  const weak = wilson(WEAK_RATE, n);
  const diff = diffCI(STRONG_RATE, n, WEAK_RATE, n);
  const significant = diff[0] > 0 || diff[1] < 0;
  const overlapLo = Math.max(strong[0], weak[0]);
  const overlapHi = Math.min(strong[1], weak[1]);
  return { strong, weak, diff, significant, barsOverlap: overlapHi > overlapLo, overlapLo, overlapHi };
}

// ---------------------------------------------------------------- SVG bar primitive
/** A proper error bar: a translucent interval band, a solid centre rule, vertical
 *  end caps at [lo,hi], and the fixed point-estimate dot. `tone` selects the CSS
 *  skin (ink for the two policy estimates, verdict-coloured for the difference). */
function CIBar({
  lo, hi, point, cy, x, tone,
}: { lo: number; hi: number; point: number; cy: number; x: (v: number) => number; tone: string }) {
  const xl = x(lo), xh = x(hi), xp = x(point);
  const top = cy - BAR_H / 2, bot = cy + BAR_H / 2;
  return (
    <g class={`eb-bar eb-bar--${tone}`}>
      <rect class="eb-band" x={xl.toFixed(1)} y={top} width={Math.max(0, xh - xl).toFixed(1)} height={BAR_H} rx={4} />
      <line class="eb-rule" x1={xl.toFixed(1)} y1={cy} x2={xh.toFixed(1)} y2={cy} />
      <line class="eb-cap" x1={xl.toFixed(1)} y1={top + 2} x2={xl.toFixed(1)} y2={bot - 2} />
      <line class="eb-cap" x1={xh.toFixed(1)} y1={top + 2} x2={xh.toFixed(1)} y2={bot - 2} />
      <circle class="eb-point" cx={xp.toFixed(1)} cy={cy} r={4.5} />
    </g>
  );
}

/** The whole plot as SVG children — shared verbatim by the SSR figure (JS-off
 *  fallback + pre-boot frame) and the hydrated island, so booting causes no
 *  reflow; only the slider/buttons below gain interactivity. */
function Plot({ b }: { b: Bands }) {
  const zeroX = diffX(0);
  return (
    <svg
      class="eb-svg"
      viewBox={`0 0 ${VW} ${VH}`}
      role="img"
      aria-hidden="true"
    >
      {/* faint plot frame */}
      <line class="eb-frame" x1={PLOT_L} y1={28} x2={PLOT_L} y2={Y_RATE_AXIS} />

      {/* the overlap zone (made visible): where the two one-sample intervals share
          values — "these two rates could be the same policy". Vanishes as N grows. */}
      {b.barsOverlap && (
        <rect
          class="eb-overlap"
          x={rateX(b.overlapLo).toFixed(1)}
          y={Y_STRONG - BAR_H / 2 - 6}
          width={Math.max(0, rateX(b.overlapHi) - rateX(b.overlapLo)).toFixed(1)}
          height={Y_WEAK - Y_STRONG + BAR_H + 12}
          rx={3}
        />
      )}

      {/* rate axis + ticks */}
      <line class="eb-axis" x1={PLOT_L} y1={Y_RATE_AXIS} x2={PLOT_R} y2={Y_RATE_AXIS} />
      {RATE_TICKS.map((t) => (
        <g class="eb-tick">
          <line x1={rateX(t)} y1={Y_RATE_AXIS} x2={rateX(t)} y2={Y_RATE_AXIS + 5} />
          <text class="eb-tick-lbl" x={rateX(t)} y={Y_RATE_AXIS + 18} text-anchor="middle">{t.toFixed(1)}</text>
        </g>
      ))}
      <text class="eb-axis-title" x={PLOT_R} y={Y_RATE_AXIS + 18} text-anchor="end">success rate</text>

      {/* the two policy CI bars (neutral ink — data, not entities) */}
      <text class="eb-row-lbl eb-row-lbl--strong" x={PLOT_L - 12} y={Y_STRONG + 4} text-anchor="end">strong</text>
      <CIBar lo={b.strong[0]} hi={b.strong[1]} point={STRONG_RATE} cy={Y_STRONG} x={rateX} tone="strong" />
      <text class="eb-row-lbl eb-row-lbl--weak" x={PLOT_L - 12} y={Y_WEAK + 4} text-anchor="end">weak</text>
      <CIBar lo={b.weak[0]} hi={b.weak[1]} point={WEAK_RATE} cy={Y_WEAK} x={rateX} tone="weak" />

      {/* section divider */}
      <line class="eb-divider" x1={PLOT_L} y1={Y_DIFF_LABEL - 22} x2={PLOT_R} y2={Y_DIFF_LABEL - 22} />
      <text class="eb-diff-title" x={PLOT_L} y={Y_DIFF_LABEL} text-anchor="start">
        difference · strong − weak (95% CI)
      </text>

      {/* difference axis with a prominent zero line — the verdict pivot */}
      <line class="eb-axis" x1={PLOT_L} y1={Y_DIFF_AXIS} x2={PLOT_R} y2={Y_DIFF_AXIS} />
      <line class="eb-zero" x1={zeroX.toFixed(1)} y1={Y_DIFF - BAR_H / 2 - 12} x2={zeroX.toFixed(1)} y2={Y_DIFF_AXIS + 4} />
      <text class="eb-zero-lbl" x={zeroX.toFixed(1)} y={Y_DIFF - BAR_H / 2 - 16} text-anchor="middle">0</text>
      {DIFF_TICKS.map((t) => (
        <g class="eb-tick">
          <line x1={diffX(t)} y1={Y_DIFF_AXIS} x2={diffX(t)} y2={Y_DIFF_AXIS + 5} />
          <text class="eb-tick-lbl" x={diffX(t)} y={Y_DIFF_AXIS + 18} text-anchor="middle">{fmtSigned(t, 1)}</text>
        </g>
      ))}

      {/* the difference CI bar — coloured by the live verdict (red spans 0, green excludes 0) */}
      <g data-sig={b.significant ? "true" : "false"}>
        <CIBar lo={b.diff[0]} hi={b.diff[1]} point={TRUE_GAP} cy={Y_DIFF} x={diffX} tone="diff" />
      </g>
    </svg>
  );
}

// ---------------------------------------------------------------- the toy
export default function EvalBandsToy() {
  const [n, setN] = useState(N_DEFAULT);
  const [booted, setBooted] = useState(false);

  // Client-only: reveal the interactive controls (the plot is the JS-off fallback).
  useEffect(() => { setBooted(true); }, []);

  const b = computeBands(n);
  const verdict = b.significant ? "ranking significant" : "ranking NOT established";
  // The chapter's nuance: between N≈75 and N≈148 the difference CI already excludes
  // 0 while the raw bars still touch — trust the difference interval, not the eyeball.
  const eyeballLags = b.significant && b.barsOverlap;

  const reset = () => setN(N_DEFAULT);

  return (
    <div class="eb">
      <p class="eb-caption">
        Two behavior-cloning policies, one trained on more demos than the other. Their
        best-estimate success rates (pooled over 200 episodes, seed 0) are{" "}
        <b>strong 0.225</b> and <b>weak 0.105</b>. Drag <b>N</b> — the episodes you
        evaluate each on — and watch the 95% confidence bands tighten. Do the two
        policies actually differ? It depends entirely on N.
      </p>

      <figure class="eb-figure" data-sig={b.significant ? "true" : "false"}>
        <Plot b={b} />

        {/* live numeric readouts + the verdict (the plot ships the nominal N=20 state) */}
        <div class="eb-hud" aria-hidden="true">
          <div class="eb-hud-row">
            <span class="eb-k">strong 0.225</span>
            <span class="eb-v">[{fmt(b.strong[0])}, {fmt(b.strong[1])}]</span>
          </div>
          <div class="eb-hud-row">
            <span class="eb-k">weak 0.105</span>
            <span class="eb-v">[{fmt(b.weak[0])}, {fmt(b.weak[1])}]</span>
          </div>
          <div class="eb-hud-row">
            <span class="eb-k">difference CI</span>
            <span class={`eb-v ${b.significant ? "eb-ok" : "eb-bad"}`}>
              [{fmtSigned(b.diff[0])}, {fmtSigned(b.diff[1])}]
            </span>
          </div>
        </div>

        <div class="eb-verdict" data-sig={b.significant ? "true" : "false"} aria-hidden="true">
          <span class="eb-verdict-n">N = {n}</span>
          <span class="eb-verdict-txt">
            {b.significant
              ? "difference CI excludes 0 — the gap is real"
              : "difference CI spans 0 — you cannot rank them"}
          </span>
        </div>
      </figure>

      {/* live SR summary — polite, so slider drags coalesce instead of interrupting */}
      <p class="eb-sr" aria-live="polite">
        {`N = ${n} episodes per policy. Strong 95% interval ${fmt(b.strong[0], 2)} to ${fmt(b.strong[1], 2)}; `
          + `weak ${fmt(b.weak[0], 2)} to ${fmt(b.weak[1], 2)}. Difference interval ${fmtSigned(b.diff[0])} to ${fmtSigned(b.diff[1])}: `
          + (b.significant ? "excludes zero, the ranking is significant." : "spans zero, the ranking is not established.")}
      </p>

      {/* controls — JS-only (the plot reads without them). The ONE control is the N
          slider (signal blue); it is natively keyboard-operable (arrow keys). */}
      <div class="eb-controls">
        <div class="eb-slider-row">
          <label class="eb-slider-label" for="eb-n">
            episodes N = <b>{n}</b>
          </label>
          <input
            id="eb-n"
            class="eb-slider"
            type="range"
            min={N_MIN}
            max={N_MAX}
            step={N_STEP}
            value={n}
            disabled={!booted}
            aria-label="Episodes evaluated per policy. Lower N gives wider confidence bands; higher N tightens them until the ranking becomes significant."
            aria-valuetext={`${n} episodes, ${verdict}`}
            onInput={(e) => setN(parseInt((e.currentTarget as HTMLInputElement).value, 10))}
          />
        </div>
        <div class="eb-btn-row">
          <button type="button" class="eb-btn" onClick={() => setN(N_DEFAULT)} disabled={!booted} aria-pressed={n === N_DEFAULT}>
            N = 20 · one suite
          </button>
          <button type="button" class="eb-btn" onClick={() => setN(N_POOLED)} disabled={!booted} aria-pressed={n === N_POOLED}>
            N = 200 · pooled
          </button>
          <button type="button" class="eb-btn" onClick={reset} disabled={!booted}>
            reset
          </button>
          <span class="eb-note">
            {eyeballLags
              ? "the bars still touch, but the difference CI already excludes 0 — trust the difference interval"
              : "drag N or use arrow keys · plot reads with JS off"}
          </span>
        </div>
      </div>

      <p class="eb-foot">
        The point estimates never move — only the bands do. One 20-episode suite
        happened to report 0.30 vs 0.20, but across ten such suites the strong rate
        swung 0.05–0.30 (std 0.087): a single 20-episode number is a coin flip. At
        N=200 these bands are the harness&apos;s measured Wilson intervals exactly.
      </p>
    </div>
  );
}
