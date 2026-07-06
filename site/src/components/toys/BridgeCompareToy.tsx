/**
 * BridgeCompareToy — ch1.9 "Graduation Bridge I" concept-toy (demo id
 * "bridge_comparison").
 *
 * The BRIDGE lesson, made scalar. Three linked, pure-2D panels — no MuJoCo-WASM,
 * no ONNX, no model run (the official ACT is a chunked temporally-ensembled policy
 * with the same contract-v1 mismatch 1.3 documented, so there is nothing to drive
 * live; the teaching artifact is the numbers, exactly as demo/embed.yaml declares):
 *
 *   (1) THE CODE RATIO — the HERO. ch1.3's hand-rolled act.py (~380 lines) beside
 *       bridge.py's `# --- region: official ---` (~12 lines) for the SAME algorithm.
 *       The CVAE (use_vae) and the ResNet image path are one flag away in the
 *       official config but kept OFF here (use_vae=False, env-state input) to match
 *       1.3's cut — so the comparison stays apples-to-apples.
 *
 *   (2) THE TWO ACTs — the honest, CI'd comparison. Official lerobot ACT 0.55 with
 *       its Wilson 95% interval [0.34, 0.74] vs the from-scratch 1.3 ACT 0.95, both
 *       on the SAME demos + SAME held-out seeds at n=20. The from-scratch point sits
 *       ABOVE the official interval: the intervals do not overlap, so the gap is
 *       REAL at this budget, not sampling noise. The Wilson CI is recomputed
 *       CLIENT-SIDE below (closed-form port of ch1.6 harness.py wilson_ci) — the
 *       toggle hides it to show WHY you need it: 0.55-vs-0.95 as bare points is
 *       ambiguous at n=20; the interval is what licenses the verdict.
 *
 *   (3) THE BREAK — `--break train_dist`. The same official ACT scored on the
 *       TRAINING seeds (0.60) sits at/above its held-out score (0.55): the 1.6 sin,
 *       one flag away on the official stack. The inflation is SMALL here (+0.05,
 *       CIs overlap) because our held-out seeds are same-distribution — the
 *       DIRECTION (train >= held-out) is the lesson, not the magnitude.
 *
 * The honest thesis (measured 2026-07-06, seed 0, cpu — meta.yaml reference_run):
 * the code shrank ~32x, yet the task-tuned from-scratch ACT BEATS the general
 * official config. That is not a framework defect — 1.3's four entity tokens are a
 * task-specific inductive bias the general config lacks, and the same bias ports
 * into lerobot. The framework buys the ECOSYSTEM (maintained code, the Hub,
 * real-robot deploy, other policies one import away), not a free accuracy win.
 *
 * Follows the FROZEN CONCEPT-TOY CONTRACT documented in PlateIsland.tsx: the whole
 * figure server-renders (booted=false) as a complete, captioned static view — that
 * SSR output IS the JS-off experience, with the intervals + verdict already shown.
 * Hydration only enables the toggles. No WASM, no dynamic import, no rAF loop, so
 * it is inherently reduced-motion friendly; the sole transitions are gated behind
 * prefers-reduced-motion. Colour comes from the shared flipping design tokens
 * (site/public/styles.css), so it is correct in light AND dark with no special-case
 * (unlike the sim stages, this is a data panel, not a fixed light lab instrument).
 */
import { useEffect, useState } from "preact/hooks";
import "./bridge-compare.css";

// ---------------------------------------------------------------- measured scalars
// All numbers are SCALARS from curriculum/phase1_imitation/ch1.9_bridge/meta.yaml
// (reference_run: seed 0, cpu; num_demos 50, epochs 150, chunk_size 8, model_dim
// 128, eval_episodes 20; official use_vae=False to match 1.3). We store the raw
// success COUNTS (k of n) and recompute every rate + interval below, so the toy
// carries no pre-baked CI — the Wilson formula is the thing on display.
const N_EVAL = 20; // eval_episodes — the ch1.6 point: WIDE intervals at n=20
const OFFICIAL_K = 11; // official_success_rate 0.55  → 11/20  (base seed 20000)
const SCRATCH_K = 19; // scratch_success_rate  0.95  → 19/20  (from-scratch 1.3 ACT)
const BREAK_K = 12; // break_train_dist_success_rate 0.60 → 12/20 (eval on TRAIN seeds)

// Code line counts (the HERO ratio). ch1.3 act.py is 379 lines end-to-end (the
// hand-rolled encoder/decoder/chunking-head/temporal-ensembler); bridge.py's
// `# --- region: official ---` is ~12 lines (11 code) for the SAME ACT. Even
// counting the full official construct+train+eval it is ~20 lines. meta.yaml
// objective: "the code shrinks ~380->~12 lines".
const LINES_SCRATCH = 380;
const LINES_OFFICIAL = 12;
const CODE_RATIO = Math.round(LINES_SCRATCH / LINES_OFFICIAL); // ~32x

// ------------------------------------------------------------------- Wilson score
// Closed-form 95% Wilson score interval — a direct port of ch1.6 harness.py
// wilson_ci (numpy/math only there, math-free here). The Wald interval collapses
// to zero width at k=0 or k=n and can leave [0,1]; Wilson solves the implicit
// quadratic, so it is always inside [0,1] and never degenerate — the interval a
// success rate should ship with. VERIFIED against the harness self-check:
// wilsonCi(0, 10) === [0, 0.2775] (Brown, Cai & DasGupta 2001) and against
// meta.yaml: wilsonCi(11, 20) === [0.342085, 0.741802].
const Z95 = 1.959963985; // 0.975 standard-normal quantile — same constant as ch1.6
function wilsonCi(k: number, n: number, z: number = Z95): [number, number] {
  if (n === 0) return [0, 1];
  const pHat = k / n;
  const denom = 1 + (z * z) / n;
  const center = (pHat + (z * z) / (2 * n)) / denom;
  const half = (z / denom) * Math.sqrt((pHat * (1 - pHat)) / n + (z * z) / (4 * n * n));
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

// Trust-but-verify at dev time only (mirrors harness.py's assert; never throws in
// the shipped bundle).
if (typeof import.meta !== "undefined" && (import.meta as { env?: { DEV?: boolean } }).env?.DEV) {
  const [lo, hi] = wilsonCi(0, 10);
  // eslint-disable-next-line no-console
  console.assert(lo === 0 && Math.abs(hi - 0.2775) < 1e-3, "[bridge toy] Wilson self-check failed", lo, hi);
}

const OFFICIAL_CI = wilsonCi(OFFICIAL_K, N_EVAL); // [0.342085, 0.741802]
const SCRATCH_CI = wilsonCi(SCRATCH_K, N_EVAL); // [0.763869, 0.991119]
const BREAK_CI = wilsonCi(BREAK_K, N_EVAL); // [0.386582, 0.781193]
const OFFICIAL_RATE = OFFICIAL_K / N_EVAL;
const SCRATCH_RATE = SCRATCH_K / N_EVAL;
const BREAK_RATE = BREAK_K / N_EVAL;
// The verdict: the from-scratch interval's LOWER bound clears the official
// interval's UPPER bound → the two 95% intervals do not overlap → a real,
// separable gap at n=20 (the conservative eyeball test; the Newcombe diff CI in
// ch1.6 agrees the ranking is significant here).
const SEPARABLE = SCRATCH_CI[0] > OFFICIAL_CI[1];

const pct = (r: number): string => `${Math.round(r * 100)}%`;
const two = (r: number): string => r.toFixed(2);

// ----------------------------------------------------------- success-rate chart
// One reusable [point + Wilson whisker] chart on a 0..1 success-rate axis, shared
// by panel 2 (the two ACTs) and panel 3 (the break). Pure geometry, no state.
const CW = 480; // svg width (viewBox units)
const GUT_L = 128; // left gutter for the row label
const GUT_R = 52; // right gutter for the rate value
const X0 = GUT_L;
const X1 = CW - GUT_R;
const r2x = (r: number): number => X0 + r * (X1 - X0);

interface RateRow {
  label: string;
  rate: number;
  ci: [number, number];
  tone: "neutral" | "signal" | "alert"; // colour role for the point/whisker
  n: number;
}

function RateChart({
  rows,
  showCI,
  ceilingX,
  ceilingLabel,
  ariaLabel,
  title,
}: {
  rows: RateRow[];
  showCI: boolean;
  ceilingX?: number; // draw a vertical guide at this rate (the "ceiling" to clear)
  ceilingLabel?: string;
  ariaLabel: string;
  title: string;
}) {
  const rowH = 46;
  const top = 20;
  const axisY = top + rows.length * rowH + 8;
  const H = axisY + 34;
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  const cx = ceilingX !== undefined ? r2x(ceilingX) : null;
  return (
    <svg
      class="bc-chart"
      viewBox={`0 0 ${CW} ${H}`}
      role="img"
      aria-label={ariaLabel}
    >
      <title>{title}</title>
      {/* axis baseline + ticks */}
      <line class="bc-axis" x1={X0} y1={axisY} x2={X1} y2={axisY} />
      {ticks.map((t) => (
        <g class="bc-tick">
          <line x1={r2x(t)} y1={axisY} x2={r2x(t)} y2={axisY + 5} />
          <text class="bc-tick-lbl" x={r2x(t)} y={axisY + 18} text-anchor="middle">
            {pct(t)}
          </text>
        </g>
      ))}
      <text class="bc-axis-cap" x={(X0 + X1) / 2} y={H - 2} text-anchor="middle">
        held-out success rate (n={rows[0]?.n})
      </text>

      {/* the "ceiling" guide — the interval the challenger must clear */}
      {showCI && cx !== null && (
        <g class="bc-ceiling">
          <line x1={cx} y1={top - 6} x2={cx} y2={axisY} />
          {ceilingLabel && (
            <text class="bc-ceiling-lbl" x={cx + 6} y={top + 4}>
              {ceilingLabel}
            </text>
          )}
        </g>
      )}

      {rows.map((row, i) => {
        const y = top + i * rowH + rowH / 2;
        const px = r2x(row.rate);
        const lo = r2x(row.ci[0]);
        const hi = r2x(row.ci[1]);
        return (
          <g class={`bc-row bc-row--${row.tone}`}>
            <text class="bc-row-lbl" x={X0 - 12} y={y + 4} text-anchor="end">
              {row.label}
            </text>
            {showCI && (
              <g class="bc-whisker">
                <line x1={lo} y1={y} x2={hi} y2={y} />
                <line x1={lo} y1={y - 7} x2={lo} y2={y + 7} />
                <line x1={hi} y1={y - 7} x2={hi} y2={y + 7} />
              </g>
            )}
            <circle class="bc-point" cx={px} cy={y} r={6} />
            <text class="bc-row-val" x={X1 + 10} y={y + 4}>
              {two(row.rate)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ------------------------------------------------------------------- horizontal bars
// The HERO code-ratio panel: two bars scaled to the larger line count.
function CodeRatio() {
  const BW = 480;
  const barX = 150;
  const barMax = BW - barX - 62;
  const scrW = barMax; // 380 → full
  const offW = Math.max(6, (LINES_OFFICIAL / LINES_SCRATCH) * barMax); // 12 → a sliver
  return (
    <svg
      class="bc-bars"
      viewBox={`0 0 ${BW} 130`}
      role="img"
      aria-label={`Lines of code for the same ACT algorithm. Hand-rolled ch1.3 act.py: ${LINES_SCRATCH} lines. Official lerobot region in bridge.py: ${LINES_OFFICIAL} lines. About ${CODE_RATIO} times less code for the same policy.`}
    >
      <title>380 lines hand-rolled vs 12 lines official — the same ACT</title>
      {/* scratch — the 380-line hand-rolled ACT */}
      <g class="bc-bar bc-bar--scratch">
        <text class="bc-bar-lbl" x={barX - 12} y={34} text-anchor="end">ch1.3 act.py</text>
        <text class="bc-bar-sub" x={barX - 12} y={49} text-anchor="end">hand-rolled ACT</text>
        <rect x={barX} y={22} width={scrW} height={30} rx={3} />
        <text class="bc-bar-val" x={barX + scrW + 8} y={42}>{LINES_SCRATCH} lines</text>
      </g>
      {/* official — the ~12-line region for the SAME algorithm */}
      <g class="bc-bar bc-bar--official">
        <text class="bc-bar-lbl" x={barX - 12} y={90} text-anchor="end">bridge.py</text>
        <text class="bc-bar-sub" x={barX - 12} y={105} text-anchor="end">official region</text>
        <rect x={barX} y={78} width={offW} height={30} rx={3} />
        <text class="bc-bar-val" x={barX + offW + 8} y={98}>{LINES_OFFICIAL} lines</text>
      </g>
    </svg>
  );
}

// -------------------------------------------------------------------------- the toy
export default function BridgeCompareToy() {
  // booted=false is the SSR / JS-off view: intervals + break both shown, verdict
  // resolved. Hydration flips booted=true and enables the toggles.
  const [booted, setBooted] = useState(false);
  const [showCI, setShowCI] = useState(true);
  const [showBreak, setShowBreak] = useState(true);
  // hydrate: enable controls (the poster stays put — no reflow, same DOM). The
  // booted=false SSR view is the JS-off fallback: CI + break already shown.
  useEffect(() => {
    setBooted(true);
  }, []);

  const actRows: RateRow[] = [
    { label: "official lerobot ACT", rate: OFFICIAL_RATE, ci: OFFICIAL_CI, tone: "neutral", n: N_EVAL },
    { label: "from-scratch 1.3 ACT", rate: SCRATCH_RATE, ci: SCRATCH_CI, tone: "signal", n: N_EVAL },
  ];
  const breakRows: RateRow[] = [
    { label: "official · held-out", rate: OFFICIAL_RATE, ci: OFFICIAL_CI, tone: "neutral", n: N_EVAL },
    { label: "official · train seeds", rate: BREAK_RATE, ci: BREAK_CI, tone: "alert", n: N_EVAL },
  ];

  // aria-live summary — announces the verdict as the CI toggle flips.
  const liveSummary = showCI
    ? `From-scratch success rate ${two(SCRATCH_RATE)}, 95% interval ${two(SCRATCH_CI[0])} to ${two(
        SCRATCH_CI[1],
      )}. Official ${two(OFFICIAL_RATE)}, interval ${two(OFFICIAL_CI[0])} to ${two(OFFICIAL_CI[1])}. ${
        SEPARABLE
          ? "The intervals do not overlap: from-scratch is separably better at 20 episodes — a real gap, not noise."
          : "The intervals overlap: the ranking is not established at this episode count."
      }`
    : `Point estimates only: from-scratch ${two(SCRATCH_RATE)} versus official ${two(
        OFFICIAL_RATE,
      )}. Intervals hidden — enable them to test whether the gap is real at 20 episodes.`;

  return (
    <div class="bc">
      {/* PANEL 1 — the hero code ratio */}
      <section class="bc-panel bc-panel--code">
        <header class="bc-head">
          <h3 class="bc-title">The code shrank ~{CODE_RATIO}&times;</h3>
          <p class="bc-sub">Same ACT algorithm, two implementations.</p>
        </header>
        <CodeRatio />
        <p class="bc-note">
          The CVAE (<code>use_vae</code>) and the ResNet image path are one flag away
          in the official config — kept <b>OFF</b> here (<code>use_vae=False</code>,
          env-state input) to match ch1.3&rsquo;s cut, so the comparison is
          apples-to-apples.
        </p>
      </section>

      {/* PANEL 2 — the two ACTs, with Wilson CIs + the separable verdict */}
      <section class="bc-panel bc-panel--acts">
        <header class="bc-head">
          <h3 class="bc-title">&hellip; the accuracy did not follow it down</h3>
          <p class="bc-sub">
            Same demos, same held-out seeds. Success rate &plusmn; Wilson 95% interval
            (recomputed from ch1.6).
          </p>
        </header>
        <RateChart
          rows={actRows}
          showCI={showCI}
          ceilingX={OFFICIAL_CI[1]}
          ceilingLabel="official 95% ceiling"
          title="Official ACT 0.55 versus from-scratch ACT 0.95, with Wilson intervals"
          ariaLabel={`Two success rates on a 0 to 100 percent axis at n equals 20. Official lerobot ACT: 0.55, Wilson 95 percent interval 0.34 to 0.74. From-scratch ch1.3 ACT: 0.95, interval 0.76 to 0.99. The from-scratch point and its interval sit entirely above the official interval.`}
        />
        <div
          class={`bc-verdict ${showCI ? (SEPARABLE ? "bc-verdict--yes" : "bc-verdict--no") : "bc-verdict--pending"}`}
          aria-hidden="true"
        >
          {showCI ? (
            SEPARABLE ? (
              <>
                <span class="bc-verdict-tag">separable &#10003;</span>
                <span>
                  from-scratch <b>0.95</b> clears the official 95% interval [{two(OFFICIAL_CI[0])},{" "}
                  {two(OFFICIAL_CI[1])}] &mdash; a <b>real</b> gap at n=20, not sampling noise.
                </span>
              </>
            ) : (
              <>
                <span class="bc-verdict-tag">overlap</span>
                <span>the intervals overlap &mdash; the ranking is not established at n=20.</span>
              </>
            )
          ) : (
            <>
              <span class="bc-verdict-tag">0.55 vs 0.95?</span>
              <span>
                bare points at n=20 &mdash; real, or a lucky draw? Show the 95% intervals to decide.
              </span>
            </>
          )}
        </div>
        <p class="bc-note">
          The framework did not lose &mdash; ch1.3&rsquo;s four <b>entity tokens</b> (arms /
          cube / target) are a task-specific bias the general config lacks, and the
          same bias ports into lerobot. The framework buys the <b>ecosystem</b> (the
          Hub, real-robot deploy, other policies one import away), not a free number.
        </p>
      </section>

      {/* PANEL 3 — the one-flag break (optional / collapsible) */}
      {showBreak && (
        <section class="bc-panel bc-panel--break">
          <header class="bc-head">
            <h3 class="bc-title">One flag inflates the score</h3>
            <p class="bc-sub">
              The same official ACT under <code>--break train_dist</code> &mdash; scored on
              the <b>training</b> seeds.
            </p>
          </header>
          <RateChart
            rows={breakRows}
            showCI={showCI}
            title="Official ACT: held-out 0.55 versus train-seed 0.60"
            ariaLabel={`The official ACT on a 0 to 100 percent axis at n equals 20. Held-out seeds: 0.55, interval 0.34 to 0.74. Training seeds: 0.60, interval 0.39 to 0.78. Scoring on training seeds sits at or above held-out.`}
          />
          <p class="bc-note">
            train-seed <b>0.60</b> &ge; held-out <b>0.55</b>: the ch1.6 sin, one flag away
            on the official stack. The inflation is small here (+0.05, CIs overlap)
            because our held-out seeds are same-distribution &mdash; the <b>direction</b> is
            the lesson, and it would be far larger on a genuinely out-of-distribution set.
          </p>
        </section>
      )}

      {/* controls — JS-only enhancement (the SSR view ships CI + break already on) */}
      <div class="bc-controls">
        <button
          type="button"
          class={`bc-btn ${showCI ? "bc-btn--primary" : ""}`}
          aria-pressed={showCI}
          onClick={() => setShowCI((v) => !v)}
          disabled={!booted}
        >
          {showCI ? "hide the 95% intervals" : "show the 95% intervals"}
        </button>
        <button
          type="button"
          class="bc-btn"
          aria-pressed={showBreak}
          onClick={() => setShowBreak((v) => !v)}
          disabled={!booted}
        >
          {showBreak ? "hide the train_dist break" : "show the train_dist break"}
        </button>
        <span class="bc-control-note">
          all numbers measured (seed 0, cpu) &middot; poster reads with JS off
        </span>
      </div>

      {/* screen-reader live region — the verdict, spoken as the toggle flips */}
      <p class="bc-sr" aria-live="polite">
        {booted ? liveSummary : ""}
      </p>
    </div>
  );
}
