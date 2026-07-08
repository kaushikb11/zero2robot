/**
 * ch4 "Offline RL Primer: Beat the Data With Its Own Reward" — the BC-vs-AWAC
 * concept-toy (`demo: offline_bc_vs_awac`). A recorded-data sibling of ch3.3's
 * energy-drift toy and ch1.6's eval-bands: the offline-RL headline made visual.
 * No WASM — pure SVG over numbers offline.py itself measured.
 *
 * THREE panels, ONE control (the seed selector — the interactive handle):
 *   1. THE HEADLINE. On the SAME fixed, mixed-quality dataset, behavior cloning
 *      (BC) reaches the target ~5% of the time; AWAC ~22%. Success bars carry
 *      ch1.6 Wilson error bars, and a difference-CI strip shows the AWAC−BC gap
 *      sitting entirely RIGHT of zero on every seed — significant, not noise. But
 *      HONESTLY MODEST: AWAC still stops ~0.09-0.11 m short, far from the scripted
 *      expert's ~0.0001 m. BC is a FAIR clone; the win is the reward-aware
 *      extraction (weight exp(A/beta)), not a bigger network.
 *   2. THE naive-diverges Break-It. On NARROW (expert-only) data, naive
 *      maximize-Q with no data constraint OVERESTIMATES out-of-distribution
 *      actions — its mean|Q| inflates ~7x — while AWAC's stays bounded ~1.1. On
 *      the BROAD expert+random mix the random half covers the action space and
 *      naive survives: the damage is coverage-dependent, which is exactly why
 *      4.3's narrow correction data needs the advantage constraint.
 *   3. THE mechanism. BC minimizes ‖π(s)−a‖² and clones the AVERAGE of the mix
 *      (returns −16.0 … −2.3) — it can't tell a good action from a bad one. AWAC
 *      is the SAME regression, each sample reweighted by exp(A/β): above-average
 *      actions pull, junk is ignored, so it extracts a better-than-average policy.
 *
 * Like ch3.3 this needs NO WASM: it renders REAL measured numbers — regenerated
 * by site/scripts/vizdata/ch4_offline_primer.py into the co-located vizdata.json
 * (success counts + distances + naive |Q| transcribed from meta.yaml's measured
 * reference_run; the Wilson + Newcombe CIs computed by offline.py's OWN error-bar
 * code), verified to match meta. So it is pure SVG + design tokens: theme-aware
 * for free, and the server-rendered default (seed 0) IS the JS-off experience —
 * only the seed selector goes inert without hydration. Colour by MEANING:
 * AWAC = --entity-target green (the reward-aware winner / bounded), BC = neutral
 * ink (the fair baseline clone), naive = --alert red (the diverging Break-It),
 * the seed handle = --signal blue.
 */
import "./OfflineRLToy.css";
import { useState } from "preact/hooks";
// Real measured numbers from offline.py's reference_run — see the file's
// `provenance` and the generator site/scripts/vizdata/ch4_offline_primer.py.
// Committed small text, no binary.
import viz from "../../../../curriculum/phase4_capstone/ch4_offline_primer/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
interface Arm {
  k: number[];
  success: number[];
  ci: number[][];
  dist: number[];
  mean_success: number;
  mean_dist: number;
}
interface Headline { bc: Arm; awac: Arm; diff_ci: number[][]; }
interface Naive {
  expert_frac: number;
  naive_abs_q: number[];
  naive_mean: number;
  awac_abs_q: number;
  inflation: number;
  broad_note: string;
}

const SEEDS = viz.seeds as number[];
const N_POOL = viz.n_pool as number;
const BETA = viz.beta as number;
const HEAD = viz.headline as Headline;
const NAIVE = viz.naive as Naive;
const RET = viz.behavior_return as { expert: number; random: number };
const BASE = viz.baselines as { random_dist: number; expert_dist: number };

// ------------------------------------------------------------ number formatting
const pct = (v: number) => `${Math.round(v * 100)}%`;
const sgn = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;
const meters = (v: number) => `${v.toFixed(3)} m`;

// ============================================================ PANEL 1a: bars
const BW = 440, BH = 232;
const BPAD = { l: 40, r: 14, t: 18, b: 42 };
const BPW = BW - BPAD.l - BPAD.r;
const BPH = BH - BPAD.t - BPAD.b;
const YMAX = 0.4; // covers AWAC's Wilson upper (~0.36) with headroom

function HeadlineBars({ seedIdx }: { seedIdx: number }) {
  const bc = { v: HEAD.bc.success[seedIdx], ci: HEAD.bc.ci[seedIdx], k: HEAD.bc.k[seedIdx] };
  const aw = { v: HEAD.awac.success[seedIdx], ci: HEAD.awac.ci[seedIdx], k: HEAD.awac.k[seedIdx] };
  const py = (v: number) => BPAD.t + (1 - v / YMAX) * BPH;
  const y0 = py(0);
  const yticks = [0, 0.1, 0.2, 0.3, 0.4];
  // two bars, centred in the plot
  const barW = 78;
  const cx = (frac: number) => BPAD.l + frac * BPW;
  const cols = [
    { key: "bc", label: "behavior cloning", cls: "of-bc", frac: 0.31, ...bc },
    { key: "awac", label: "offline RL · AWAC", cls: "of-awac", frac: 0.69, ...aw },
  ];
  return (
    <svg
      class="of-svg"
      viewBox={`0 0 ${BW} ${BH}`}
      role="img"
      aria-label={
        `Grouped bar chart, success rate out of ${N_POOL} held-out rollouts, for seed ${SEEDS[seedIdx]}. ` +
        `Behavior cloning reaches ${pct(bc.v)}; offline RL (AWAC) reaches ${pct(aw.v)}. ` +
        "Each bar carries a Wilson 95% confidence interval; the AWAC bar sits well above the BC bar."
      }
    >
      {/* y grid + ticks */}
      {yticks.map((t) => (
        <g>
          <line class="of-grid" x1={BPAD.l} y1={py(t)} x2={BW - BPAD.r} y2={py(t)} />
          <text class="of-tick" x={BPAD.l - 6} y={py(t) + 3} text-anchor="end">{pct(t)}</text>
        </g>
      ))}
      <line class="of-axis" x1={BPAD.l} y1={y0} x2={BW - BPAD.r} y2={y0} />
      <text class="of-axis-title" x={BPAD.l - 34} y={BPAD.t + 2}>success</text>

      {cols.map((c) => {
        const x = cx(c.frac) - barW / 2;
        const top = py(c.v);
        return (
          <g>
            <rect class={`of-bar ${c.cls}`} x={x} y={top} width={barW} height={y0 - top} rx={2} />
            {/* Wilson CI whisker */}
            <line class="of-whisk" x1={cx(c.frac)} y1={py(c.ci[0])} x2={cx(c.frac)} y2={py(c.ci[1])} />
            <line class="of-whisk" x1={cx(c.frac) - 9} y1={py(c.ci[0])} x2={cx(c.frac) + 9} y2={py(c.ci[0])} />
            <line class="of-whisk" x1={cx(c.frac) - 9} y1={py(c.ci[1])} x2={cx(c.frac) + 9} y2={py(c.ci[1])} />
            {/* value label above the whisker */}
            <text class={`of-barval ${c.cls}`} x={cx(c.frac)} y={py(c.ci[1]) - 6} text-anchor="middle">{pct(c.v)}</text>
            <text class="of-barlab" x={cx(c.frac)} y={BH - 22} text-anchor="middle">{c.label}</text>
            <text class="of-barsub" x={cx(c.frac)} y={BH - 9} text-anchor="middle">{c.k}/{N_POOL}</text>
          </g>
        );
      })}
    </svg>
  );
}

// ============================================================ PANEL 1b: diff strip
const DW = 440, DH = 108;
const DPAD = { l: 40, r: 46, t: 16, b: 22 };
const DPW = DW - DPAD.l - DPAD.r;
const DLO = -0.05, DHI = 0.4; // domain: keep 0 visible, all intervals to its right

function DiffStrip({ seedIdx }: { seedIdx: number }) {
  const dx = (v: number) => DPAD.l + ((v - DLO) / (DHI - DLO)) * DPW;
  const rowY = (i: number) => DPAD.t + 10 + i * ((DH - DPAD.t - DPAD.b) / SEEDS.length) + 6;
  const x0 = dx(0);
  return (
    <svg
      class="of-svg of-diff-svg"
      viewBox={`0 0 ${DW} ${DH}`}
      role="img"
      aria-label={
        "Difference-in-success confidence intervals (AWAC minus behavior cloning), one horizontal interval per seed. " +
        "Every interval lies entirely to the right of the zero line, so the AWAC advantage is statistically significant on every seed."
      }
    >
      {/* the zero (no-difference) reference */}
      <line class="of-zero" x1={x0} y1={DPAD.t - 2} x2={x0} y2={DH - DPAD.b} />
      <text class="of-zero-lab" x={x0} y={DH - 6} text-anchor="middle">0 · no gain</text>
      {/* x ticks */}
      {[0.1, 0.2, 0.3].map((t) => (
        <text class="of-tick" x={dx(t)} y={DH - 6} text-anchor="middle">+{t.toFixed(1)}</text>
      ))}

      {SEEDS.map((s, i) => {
        const ci = HEAD.diff_ci[i];
        const point = HEAD.awac.success[i] - HEAD.bc.success[i];
        const sel = i === seedIdx;
        const y = rowY(i);
        return (
          <g class={sel ? "of-di-sel" : ""}>
            <text class="of-di-seed" x={DPAD.l - 6} y={y + 3} text-anchor="end">s{s}</text>
            <line class="of-di-line" x1={dx(ci[0])} y1={y} x2={dx(ci[1])} y2={y} />
            <line class="of-di-cap" x1={dx(ci[0])} y1={y - 4} x2={dx(ci[0])} y2={y + 4} />
            <line class="of-di-cap" x1={dx(ci[1])} y1={y - 4} x2={dx(ci[1])} y2={y + 4} />
            <circle class="of-di-dot" cx={dx(point)} cy={y} r={3.2} />
            <text class="of-di-val" x={DW - DPAD.r + 6} y={y + 3}>
              {sgn(ci[0])}..{sgn(ci[1])}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ============================================================ PANEL 2: naive Q
const QW = 300, QH = 240;
const QPAD = { l: 40, r: 14, t: 18, b: 44 };
const QPW = QW - QPAD.l - QPAD.r;
const QPH = QH - QPAD.t - QPAD.b;
const QMAX = 8;

function NaiveBars() {
  const py = (v: number) => QPAD.t + (1 - v / QMAX) * QPH;
  const y0 = py(0);
  const barW = 66;
  const cx = (frac: number) => QPAD.l + frac * QPW;
  const naiveHi = Math.max(...NAIVE.naive_abs_q);
  const naiveLo = Math.min(...NAIVE.naive_abs_q);
  const bars = [
    { label: "naive max-Q", cls: "of-naive", frac: 0.3, v: NAIVE.naive_mean, lo: naiveLo, hi: naiveHi },
    { label: "AWAC", cls: "of-awac", frac: 0.72, v: NAIVE.awac_abs_q, lo: NAIVE.awac_abs_q, hi: NAIVE.awac_abs_q },
  ];
  return (
    <svg
      class="of-svg"
      viewBox={`0 0 ${QW} ${QH}`}
      role="img"
      aria-label={
        `Mean absolute Q-value on narrow, expert-only data. Naive maximize-Q inflates to about ${NAIVE.naive_mean}, ` +
        `roughly ${NAIVE.inflation} times AWAC's bounded ${NAIVE.awac_abs_q}. Naive overestimates out-of-distribution actions.`
      }
    >
      {[0, 2, 4, 6, 8].map((t) => (
        <g>
          <line class="of-grid" x1={QPAD.l} y1={py(t)} x2={QW - QPAD.r} y2={py(t)} />
          <text class="of-tick" x={QPAD.l - 6} y={py(t) + 3} text-anchor="end">{t}</text>
        </g>
      ))}
      <line class="of-axis" x1={QPAD.l} y1={y0} x2={QW - QPAD.r} y2={y0} />
      <text class="of-axis-title" x={QPAD.l - 30} y={QPAD.t + 2}>mean|Q|</text>

      {bars.map((b) => {
        const x = cx(b.frac) - barW / 2;
        const top = py(b.v);
        return (
          <g>
            <rect class={`of-bar ${b.cls}`} x={x} y={top} width={barW} height={y0 - top} rx={2} />
            {b.hi > b.lo && (
              <>
                <line class="of-whisk" x1={cx(b.frac)} y1={py(b.lo)} x2={cx(b.frac)} y2={py(b.hi)} />
                <line class="of-whisk" x1={cx(b.frac) - 8} y1={py(b.lo)} x2={cx(b.frac) + 8} y2={py(b.lo)} />
                <line class="of-whisk" x1={cx(b.frac) - 8} y1={py(b.hi)} x2={cx(b.frac) + 8} y2={py(b.hi)} />
              </>
            )}
            <text class={`of-barval ${b.cls}`} x={cx(b.frac)} y={py(b.hi) - 6} text-anchor="middle">{b.v.toFixed(1)}</text>
            <text class="of-barlab" x={cx(b.frac)} y={QH - 24} text-anchor="middle">{b.label}</text>
          </g>
        );
      })}
      <text class="of-barsub of-naive" x={cx(0.3)} y={QH - 11} text-anchor="middle">≈ {NAIVE.inflation}× inflated</text>
      <text class="of-barsub of-awac" x={cx(0.72)} y={QH - 11} text-anchor="middle">bounded</text>
    </svg>
  );
}

// ============================================================ PANEL 3: mechanism
// A schematic (NOT measured data) of WHY AWAC beats BC: on the action-quality axis
// spanned by the mix's behavior returns, BC lands on the average while AWAC's
// exp(A/β) weighting shifts the extracted policy toward the above-average actions.
function Mechanism() {
  const MW = 460, MH = 92;
  const pad = 54;
  const lo = RET.random, hi = RET.expert; // -16 (junk) .. -2.3 (expert)
  const ax = (v: number) => pad + ((v - lo) / (hi - lo)) * (MW - 2 * pad);
  const bcX = ax((lo + hi) / 2);       // BC clones the AVERAGE of the mix
  const awX = ax(lo + 0.78 * (hi - lo)); // AWAC pulled toward the above-average end
  return (
    <div class="of-mech">
      <div class="of-mech-h">the mechanism — the reward is the only difference</div>
      <div class="of-mech-rows">
        <div class="of-mech-row">
          <span class="of-chip of-bc" />
          <span class="of-mech-txt">
            <b>BC</b> minimizes ‖π(s) − a‖² — it clones the <b>average</b> of the mix, so it can't tell a good action from a bad one.
          </span>
        </div>
        <div class="of-mech-row">
          <span class="of-chip of-awac" />
          <span class="of-mech-txt">
            <b>AWAC</b> is the same loss <b>× exp(A/β)</b>, β = {BETA} — each sample weighted by advantage A = Q(s,a) − V(s). Above-average actions pull; junk is ignored.
          </span>
        </div>
      </div>
      <svg class="of-mech-svg" viewBox={`0 0 ${MW} ${MH}`} role="img"
        aria-label={
          "Schematic of the action-quality axis: from the random policy's low return to the expert's high return. " +
          "Behavior cloning lands on the average of the mixed data; AWAC's advantage weighting shifts the extracted policy toward the above-average actions."
        }>
        <line class="of-mech-axis" x1={pad} y1={MH - 30} x2={MW - pad} y2={MH - 30} />
        <text class="of-mech-end" x={pad} y={MH - 12} text-anchor="start">random · return {RET.random}</text>
        <text class="of-mech-end" x={MW - pad} y={MH - 12} text-anchor="end">expert · {RET.expert}</text>
        {/* BC marker (average) */}
        <line class="of-mech-mk of-bc" x1={bcX} y1={MH - 38} x2={bcX} y2={MH - 22} />
        <text class="of-mech-mklab of-bc" x={bcX} y={MH - 44} text-anchor="middle">BC · the average</text>
        {/* AWAC marker (shifted up-quality) + arrow */}
        <line class="of-mech-mk of-awac" x1={awX} y1={MH - 38} x2={awX} y2={MH - 22} />
        <text class="of-mech-mklab of-awac" x={awX} y={MH - 44} text-anchor="middle">AWAC · above average</text>
        <path class="of-mech-arrow" d={`M ${bcX + 6} ${MH - 30} L ${awX - 6} ${MH - 30}`} marker-end="url(#of-arrow)" />
        <defs>
          <marker id="of-arrow" viewBox="0 0 8 8" refX="6" refY="4" markerWidth="6" markerHeight="6" orient="auto">
            <path class="of-mech-arrowhead" d="M0 0 L8 4 L0 8 z" />
          </marker>
        </defs>
      </svg>
      <p class="of-mech-note">Schematic — the extraction, not a measured curve. It is why offline RL exceeds BC's ceiling <em>without</em> any environment interaction.</p>
    </div>
  );
}

// ==================================================================== THE ISLAND
function OfflineRLIsland() {
  const [seedIdx, setSeedIdx] = useState(0);
  const s = SEEDS[seedIdx];
  const bcV = HEAD.bc.success[seedIdx], awV = HEAD.awac.success[seedIdx];
  const dci = HEAD.diff_ci[seedIdx];
  const awDist = HEAD.awac.dist[seedIdx];

  const announce =
    `Seed ${s}: on the same fixed mixed-quality dataset, behavior cloning reaches the target ${pct(bcV)} of the time, ` +
    `offline RL with AWAC ${pct(awV)}. The difference confidence interval is ${sgn(dci[0])} to ${sgn(dci[1])} — it excludes zero, ` +
    `so the win is statistically significant, and it holds on all three seeds. It is honestly modest: AWAC still stops about ` +
    `${meters(awDist)} short, far from the scripted expert's ${meters(BASE.expert_dist)}. The win is the reward-aware extraction, ` +
    `not a bigger network. The Break-It: naive maximize-Q with no data constraint inflates its Q-values about ` +
    `${NAIVE.inflation}-fold on narrow expert-only data, while AWAC stays bounded near ${NAIVE.awac_abs_q}.`;

  return (
    <div class="of">
      {/* --- the one control: the seed selector (the interactive handle) --- */}
      <div class="of-controls">
        <span class="of-seg-label">seed</span>
        <div class="of-seg" role="group" aria-label="Choose which evaluation seed to show">
          {SEEDS.map((sd, i) => (
            <button
              type="button"
              class="of-seg-btn"
              aria-pressed={seedIdx === i}
              onClick={() => setSeedIdx(i)}
            >
              {sd}
            </button>
          ))}
        </div>
        <span class="of-seg-note">the AWAC win holds on every seed — flip through them</span>
      </div>

      {/* --- panels: headline (left) + the naive Break-It (right) --- */}
      <div class="of-panels">
        <figure class="of-panel of-panel-wide">
          <figcaption class="of-cap">
            <span>BC vs AWAC · same fixed dataset</span>
            <b>seed {s} · N = {N_POOL}</b>
          </figcaption>
          <HeadlineBars seedIdx={seedIdx} />
          <div class="of-diff-cap">difference CI (AWAC − BC) · excludes 0 every seed → significant</div>
          <DiffStrip seedIdx={seedIdx} />
          <div class="of-readout" aria-hidden="true">
            <div class="of-row">
              <span class="of-lg of-l-bc"><span class="of-lg-name">BC</span></span>
              <span class="of-v">{pct(bcV)} · {meters(HEAD.bc.dist[seedIdx])}</span>
            </div>
            <div class="of-row">
              <span class="of-lg of-l-awac"><span class="of-lg-name">AWAC</span></span>
              <span class="of-v of-ok">{pct(awV)} · {meters(awDist)}</span>
            </div>
            <div class="of-row">
              <span class="of-lg-name">gap (AWAC − BC)</span>
              <span class="of-v of-ok">{sgn(dci[0])}..{sgn(dci[1])} · significant</span>
            </div>
          </div>
        </figure>

        <figure class="of-panel">
          <figcaption class="of-cap">
            <span>the Break-It · naive vs AWAC</span>
            <b>narrow data · expert-only</b>
          </figcaption>
          <NaiveBars />
          <p class="of-break-note">
            Drop the constraint and maximize Q: on <b>narrow</b> data the critic <b class="of-t-alert">overestimates</b> actions it
            never saw, so <b class="of-t-alert">|Q| inflates ≈{NAIVE.inflation}×</b> while eval collapses. AWAC anchors to the
            data and stays bounded. On the <b>broad</b> expert+random mix, coverage keeps naive honest — the damage is
            coverage-dependent.
          </p>
        </figure>
      </div>

      {/* --- the mechanism, full width --- */}
      <Mechanism />

      {/* non-visual path to the same aha — the qualitative story, never per-frame spam */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      <p class="of-note">
        The gap is <b>real, significant, and seed-robust — but modest</b>: AWAC reaches ~{meters(awDist)} (success {pct(awV)}),
        still short of the scripted expert (~{meters(BASE.expert_dist)}). BC is a <b>fair clone</b>; the win is the reward-aware
        extraction. Real measured numbers from offline.py (seeds {SEEDS.join(", ")}, cpu); poster reads with JS off.
      </p>
    </div>
  );
}

export default function OfflineRLToy() {
  return <OfflineRLIsland />;
}
