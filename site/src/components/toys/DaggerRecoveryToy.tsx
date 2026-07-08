/**
 * ch4.2 "Corrections: Human-in-the-Loop Data (DAgger)" — the COVARIATE-SHIFT
 * RECOVERY concept-toy (`demo: pusht_dagger_recovery`). A recorded-data sibling of
 * ch2.7's DomainRandToy and ch1.6's EvalBandsToy: pure SVG over REAL measured
 * numbers dagger.py produced — no WASM, no invented shapes.
 *
 * THE HONEST STORY (this is the whole point — do not let the UI overclaim):
 *   ch1.1 measured WHY behavior cloning dies: covariate shift. DAgger FIXES it by
 *   correcting the policy on its own mistakes. dagger.py trains a BC seed policy on
 *   a NARROW start region (covariate shift, manufactured honestly), then runs
 *   rollout -> expert-label the visited states -> aggregate -> retrain, evaluating
 *   N=200 pooled held-out episodes each round with ch1.6 Wilson intervals. The BC
 *   floor sits at ~0.06; the BEST DAgger round recovers to ~0.19-0.22 and its
 *   recovery diff CI (best − BC) EXCLUDES 0 on every seed 0-2. But the recovery is
 *   NON-MONOTONIC and the peak round VARIES by seed (3, 4, 4): over-iterating a
 *   reactive clone floods the aggregate with its own failure trajectories and the
 *   gains REGRESS (Ross et al.). So you SELECT the best round, not the last — and,
 *   honestly, the best is selected on the SAME held-out eval as the diff CI (a mild
 *   winner's curse), which is why the caveat says: a non-selected round already
 *   clears BC, and the recovery survives Bonferroni correction.
 *
 * THREE panels, ONE control:
 *   1. THE RECOVERY (headline): success-vs-DAgger-round for the selected seed, with
 *      ch1.6 Wilson error bars. BC (round 0, magenta) sits on the covariate-shift
 *      floor; the green curve climbs to the best round (marked "best"); the diff CI
 *      (best − BC) is shown and EXCLUDES 0. The honest reactive-MLP ceiling (~0.25)
 *      is a faint dashed line — recovered is not omniscient.
 *   2. NON-MONOTONIC + the winner's-curse caveat: all three seeds' round-rate curves
 *      overlaid (the selected seed bold), each seed's PEAK round marked — the peaks
 *      land at 3, 4, 4, and every curve dips/regresses somewhere, so "select the
 *      best round" is visible. Below it: the winner's-curse caveat + the Bonferroni-
 *      surviving evidence (a non-selected round already clears BC).
 *   3. THE MECHANISM: the DAgger loop in plain language — BC only saw EXPERT states
 *      -> the clone DRIFTS -> DAgger labels the states IT VISITS -> aggregate ->
 *      retrain -> recover. Plus the honest ceiling note.
 *   The ONE control is a SEED selector: the recovery is real on every seed (the diff
 *   CI excludes 0 for all), but the PEAK ROUND and the shape change with the seed —
 *   flip it and watch which round wins move.
 *
 * This needs NO WASM: the round rates, Wilson bars and diff CIs come from
 * site/scripts/vizdata/ch4.2_corrections.py, which transcribes dagger.py's MEASURED
 * reference_run and recomputes the CIs with dagger.py's OWN wilson_ci / diff_ci,
 * STOPPING on any drift from meta.yaml. So it is pure SVG + design tokens: theme-
 * aware for free, and the server-rendered default (seed 0's recovery) IS the JS-off
 * experience — only the seed selector goes inert without hydration. Colour by
 * MEANING: BC = --entity-block magenta (the covariate-shift floor), DAgger recovery
 * = --entity-target green (the hero), --alert red for the over-iteration regression,
 * --signal blue for the ONE interactive handle (the seed selector).
 */
import "./DaggerRecoveryToy.css";
import { useMemo, useState } from "preact/hooks";
// Real measured numbers from dagger.py's reference_run — see the file's `provenance`
// and the generator site/scripts/vizdata/ch4.2_corrections.py. Committed small text, no binary.
import viz from "../../../../curriculum/phase4_capstone/ch4.2_corrections/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
interface SeedEntry {
  rates: number[];
  k: number[];
  ci: number[][];
  bc_rate: number;
  best_round: number;
  best_rate: number;
  last_round: number;
  last_rate: number;
  peak_beats_last: number;
  recovery_diff_ci: number[];
  significant: boolean;
}
interface WinnersCurse {
  caveat: string;
  why_not_artifact: string;
  bonferroni_seed0: { round3: number[]; round2: number[] };
}

const N_POOLED = viz.n_pooled as number;
const SEEDS = viz.seeds as number[];
const ROUND_LABELS = viz.round_labels as string[];
const N_ROUNDS = ROUND_LABELS.length;
const DEFAULT_SEED = viz.default_seed as number;
const CEILING = viz.ceiling as number;
const PER_SEED = viz.per_seed as Record<string, SeedEntry>;
const BC_BAND = viz.bc_rate_band as number[];
const BEST_BAND = viz.best_rate_band as number[];
const PEAK_ROUNDS = viz.peak_rounds as number[];
const WINNERS = viz.winners_curse as WinnersCurse;
const MECHANISM = viz.mechanism as string[];
const CEILING_NOTE = viz.ceiling_note as string;

// ------------------------------------------------------------ number formatting
const pct = (v: number): string => `${(v * 100).toFixed(1)}%`;
// difference-CI bound, in points, signed (e.g. "+8.3")
const ptSigned = (v: number): string => `${v >= 0 ? "+" : "−"}${(Math.abs(v) * 100).toFixed(1)}`;

// ================================================================ THE RECOVERY
const CW = 540, CH = 280;
const CPAD = { l: 46, r: 16, t: 18, b: 44 };
const CPW = CW - CPAD.l - CPAD.r;
const CPH = CH - CPAD.t - CPAD.b;
const YMAX = 0.3; // rate axis 0..0.30 — holds the ~0.25 ceiling and the widest Wilson upper (~0.28)
const rx = (round: number) => CPAD.l + (round / (N_ROUNDS - 1)) * CPW;
const ry = (rate: number) => CPAD.t + (1 - rate / YMAX) * CPH;
const Y_TICKS = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3];

/** The recovery chart — shared verbatim by the SSR figure (JS-off fallback +
 *  pre-boot frame; seed = DEFAULT_SEED) and the hydrated island, so a seed flip
 *  causes no reflow. */
function RecoveryChart({ seedId }: { seedId: string }) {
  const s = PER_SEED[seedId];
  const line = s.rates.map((r, i) => `${rx(i).toFixed(1)},${ry(r).toFixed(1)}`).join(" ");

  const ariaLabel =
    `Success rate versus DAgger round for seed ${seedId}, over ${N_POOLED} pooled held-out episodes per round, ` +
    "with 95% Wilson score intervals as error bars. " +
    `Round 0 is behavior cloning at the covariate-shift floor, ${pct(s.bc_rate)}; ` +
    `the curve climbs to the best round, ${ROUND_LABELS[s.best_round]} at ${pct(s.best_rate)}, whose interval clears the BC baseline; ` +
    `the recovery difference interval, best minus BC, is ${ptSigned(s.recovery_diff_ci[0])} to ${ptSigned(s.recovery_diff_ci[1])} points and excludes zero. ` +
    (s.best_round < s.last_round
      ? `After the peak the curve regresses to ${pct(s.last_rate)} by the last round — over-iterating a reactive clone floods the dataset and the gains fall back.`
      : "The curve is still climbing at the last round for this seed.");

  return (
    <svg class="dg-svg" viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label={ariaLabel}>
      {/* rate gridlines + % ticks */}
      {Y_TICKS.map((t) => (
        <g>
          <line class="dg-grid" x1={CPAD.l} y1={ry(t)} x2={CW - CPAD.r} y2={ry(t)} />
          <text class="dg-tick" x={CPAD.l - 6} y={ry(t) + 3} text-anchor="end">{Math.round(t * 100)}%</text>
        </g>
      ))}

      {/* the honest reactive-MLP ceiling — recovered is not omniscient */}
      <line class="dg-ceiling" x1={CPAD.l} y1={ry(CEILING)} x2={CW - CPAD.r} y2={ry(CEILING)} />
      <text class="dg-ceiling-lab" x={CW - CPAD.r} y={ry(CEILING) - 4} text-anchor="end">
        reactive-MLP ceiling ≈ {Math.round(CEILING * 100)}%
      </text>

      {/* the BC covariate-shift floor — a dashed magenta baseline the recovery climbs off */}
      <line class="dg-floor" x1={CPAD.l} y1={ry(s.bc_rate)} x2={CW - CPAD.r} y2={ry(s.bc_rate)} />
      <text class="dg-floor-lab" x={CPAD.l + 2} y={ry(s.bc_rate) - 4}>BC floor · covariate shift</text>

      {/* x axis + round ticks */}
      <line class="dg-axis" x1={CPAD.l} y1={CH - CPAD.b} x2={CW - CPAD.r} y2={CH - CPAD.b} />
      {ROUND_LABELS.map((lab, i) => (
        <text class="dg-tick" x={rx(i)} y={CH - CPAD.b + 15} text-anchor="middle">{lab}</text>
      ))}
      <text class="dg-axis-title" x={CW - CPAD.r} y={CH - 6} text-anchor="end">
        correct → aggregate → retrain →
      </text>

      {/* the recovery polyline (green) */}
      <polyline class="dg-line" points={line} />

      {/* Wilson error bars + dots. Round 0 = BC (magenta floor); the best round wears
          the "best" marker; a regressed last round is flagged red. */}
      {s.rates.map((r, i) => {
        const [lo, hi] = s.ci[i];
        const isBC = i === 0;
        const isBest = i === s.best_round;
        const regressed = i > s.best_round; // over-iteration tail
        const cls = isBC ? "dg-bc" : regressed ? "dg-reg" : "dg-dagger";
        return (
          <g class={cls}>
            <line class="dg-whisker" x1={rx(i)} y1={ry(lo)} x2={rx(i)} y2={ry(hi)} />
            <line class="dg-wcap" x1={rx(i) - 4} y1={ry(hi)} x2={rx(i) + 4} y2={ry(hi)} />
            <line class="dg-wcap" x1={rx(i) - 4} y1={ry(lo)} x2={rx(i) + 4} y2={ry(lo)} />
            {isBest && <circle class="dg-best-halo" cx={rx(i)} cy={ry(r)} r={8} />}
            <circle class="dg-dot" cx={rx(i)} cy={ry(r)} r={isBest ? 4.6 : 3.6} />
          </g>
        );
      })}

      {/* best-round callout */}
      <text class="dg-best-lab" x={rx(s.best_round)} y={ry(s.best_rate) - 13} text-anchor="middle">
        best · selected
      </text>
      {s.best_round < s.last_round && (
        <text class="dg-reg-lab" x={rx(s.last_round)} y={ry(s.last_rate) + 18} text-anchor="middle">
          regresses ↓
        </text>
      )}
    </svg>
  );
}

// ============================================================ NON-MONOTONIC OVERLAY
const OW = 300, OH = 200;
const OPAD = { l: 30, r: 12, t: 14, b: 30 };
const OPW = OW - OPAD.l - OPAD.r;
const OPH = OH - OPAD.t - OPAD.b;
const orx = (round: number) => OPAD.l + (round / (N_ROUNDS - 1)) * OPW;
const ory = (rate: number) => OPAD.t + (1 - rate / YMAX) * OPH;

/** All three seeds' round-rate curves overlaid, the selected seed bold, each seed's
 *  PEAK round marked — makes "the peak VARIES by seed (3,4,4)" and the regression
 *  visible at a glance. */
function NonMonotonicOverlay({ seedId }: { seedId: string }) {
  const ariaLabel =
    "Success-rate versus DAgger-round curves for seeds 0, 1 and 2 overlaid. " +
    `Each seed's best round is marked: they land at rounds ${PEAK_ROUNDS.join(", ")} respectively — the peak is not the same round, and it is not always the last. ` +
    "Every curve dips or falls back somewhere, so the best round must be selected, not assumed to be the last.";
  return (
    <svg class="dg-svg dg-overlay-svg" viewBox={`0 0 ${OW} ${OH}`} role="img" aria-label={ariaLabel}>
      {[0, 0.1, 0.2, 0.3].map((t) => (
        <g>
          <line class="dg-grid" x1={OPAD.l} y1={ory(t)} x2={OW - OPAD.r} y2={ory(t)} />
          <text class="dg-tick" x={OPAD.l - 5} y={ory(t) + 3} text-anchor="end">{Math.round(t * 100)}</text>
        </g>
      ))}
      <line class="dg-axis" x1={OPAD.l} y1={OH - OPAD.b} x2={OW - OPAD.r} y2={OH - OPAD.b} />
      {ROUND_LABELS.map((_, i) => (
        <text class="dg-tick" x={orx(i)} y={OH - OPAD.b + 13} text-anchor="middle">{i === 0 ? "BC" : `D${i}`}</text>
      ))}

      {SEEDS.map((sd) => {
        const s = PER_SEED[String(sd)];
        const sel = String(sd) === seedId;
        const pts = s.rates.map((r, i) => `${orx(i).toFixed(1)},${ory(r).toFixed(1)}`).join(" ");
        return (
          <g class={`dg-ov-seed ${sel ? "dg-ov-sel" : ""}`} data-seed={sd}>
            <polyline class="dg-ov-line" points={pts} />
            {/* the peak dot — where THIS seed's best round lands */}
            <circle class="dg-ov-peak" cx={orx(s.best_round)} cy={ory(s.best_rate)} r={sel ? 4 : 3} />
            <text class="dg-ov-peak-lab" x={orx(s.best_round)} y={ory(s.best_rate) - 6} text-anchor="middle">
              s{sd}·r{s.best_round}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

// ==================================================================== THE ISLAND
function DaggerRecoveryToy() {
  const [seedId, setSeedId] = useState<string>(String(DEFAULT_SEED));
  // The seed selector is JS-only; the seed-0 chart is the JS-off fallback. We do not
  // gate the chart itself on hydration (it renders identically server-side), so no
  // useEffect/booted flag is needed — the buttons simply become live once hydrated.
  const s = PER_SEED[seedId];

  const announce = useMemo(() => {
    const [lo, hi] = s.recovery_diff_ci;
    const tail = s.best_round < s.last_round
      ? `The peak is round ${s.best_round}, not the last round ${s.last_round} (${pct(s.last_rate)}) — over-iterating regresses.`
      : `The curve is still climbing at the last round for this seed (peak round ${s.best_round}).`;
    return `Seed ${seedId}: behavior cloning sits at the covariate-shift floor, ${pct(s.bc_rate)}. ` +
      `The best DAgger round is round ${s.best_round} at ${pct(s.best_rate)}; the recovery difference interval, best minus BC, ` +
      `is ${ptSigned(lo)} to ${ptSigned(hi)} points and excludes zero, so the recovery is significant. ` +
      `${tail} Across seeds 0 to 2 the peak round varies — ${PEAK_ROUNDS.join(", ")} — so you select the best round.`;
  }, [seedId]);

  const b = WINNERS.bonferroni_seed0;

  return (
    <div class="dg">
      {/* --- the ONE control: the seed selector --- */}
      <div class="dg-controls">
        <span class="dg-seg-label">seed</span>
        <div class="dg-seg" role="group" aria-label="Choose which seed's DAgger recovery to show">
          {SEEDS.map((sd) => (
            <button type="button" class="dg-seg-btn" aria-pressed={seedId === String(sd)}
              onClick={() => setSeedId(String(sd))}>seed {sd}</button>
          ))}
        </div>
        <span class="dg-seg-hint">
          recovery is real on every seed — but the winning round moves (peaks: {PEAK_ROUNDS.join(", ")})
        </span>
      </div>

      {/* --- panels 1 + 2 --- */}
      <div class="dg-panels">
        <figure class="dg-panel dg-panel--chart">
          <figcaption class="dg-cap">
            <span>the recovery · success vs round</span>
            <b>seed {seedId}</b>
          </figcaption>
          <RecoveryChart seedId={seedId} />
          <div class="dg-readout" aria-hidden="true">
            <div class="dg-ro-row">
              <span class="dg-ro-k dg-bc-k">BC floor (round 0)</span>
              <span class="dg-ro-v dg-bc-v">{pct(s.bc_rate)}</span>
              <span class="dg-ro-ci">CI [{pct(s.ci[0][0])}, {pct(s.ci[0][1])}]</span>
            </div>
            <div class="dg-ro-row">
              <span class="dg-ro-k dg-best-k">best · {ROUND_LABELS[s.best_round]}</span>
              <span class="dg-ro-v dg-best-v">{pct(s.best_rate)}</span>
              <span class="dg-ro-ci">CI [{pct(s.ci[s.best_round][0])}, {pct(s.ci[s.best_round][1])}]</span>
            </div>
            <div class="dg-ro-edge">
              <span class="dg-ro-k">recovery diff CI (best − BC)</span>
              <span class="dg-ro-edgev dg-sig">
                [{ptSigned(s.recovery_diff_ci[0])}, {ptSigned(s.recovery_diff_ci[1])}] pts · excludes 0
              </span>
            </div>
          </div>
        </figure>

        <figure class="dg-panel dg-panel--overlay">
          <figcaption class="dg-cap">
            <span>non-monotonic · peak varies</span>
            <b>peaks {PEAK_ROUNDS.join(", ")}</b>
          </figcaption>
          <NonMonotonicOverlay seedId={seedId} />
          <div class="dg-caveat" aria-hidden="true">
            <div class="dg-caveat-h">the winner&apos;s-curse caveat</div>
            <p class="dg-caveat-p">
              The best round is selected on the <b>same</b> held-out eval as the diff CI — a mild selection bias.
              It is <b>not</b> an artifact: a <b>non-selected</b> round already clears BC, and the recovery
              survives Bonferroni correction (seed 0: round 2 [{ptSigned(b.round2[0])}, {ptSigned(b.round2[1])}],
              round 3 [{ptSigned(b.round3[0])}, {ptSigned(b.round3[1])}] — both exclude 0).
            </p>
          </div>
        </figure>
      </div>

      {/* --- panel 3: the mechanism --- */}
      <figure class="dg-panel dg-panel--mech">
        <figcaption class="dg-cap">
          <span>the mechanism · the DAgger loop</span>
          <b>correct on its own mistakes</b>
        </figcaption>
        <ol class="dg-mech" aria-label="The DAgger loop, step by step">
          {MECHANISM.map((step, i) => (
            <li class="dg-mech-step">
              <span class="dg-mech-n">{i + 1}</span>
              <span class="dg-mech-t">{step}</span>
              {i < MECHANISM.length - 1 && <span class="dg-mech-arrow" aria-hidden="true">→</span>}
            </li>
          ))}
        </ol>
        <p class="dg-mech-note">{CEILING_NOTE}</p>
      </figure>

      {/* non-visual path to the same aha — the qualitative reading only, never per-frame spam */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      <p class="dg-note">
        <b>DAgger closes ch1.1&apos;s covariate-shift loop.</b> The BC floor ({pct(BC_BAND[0])}–{pct(BC_BAND[1])} across seeds)
        recovers to the best DAgger round ({pct(BEST_BAND[0])}–{pct(BEST_BAND[1])}); the recovery diff CI excludes 0 on every
        seed 0–2. But it is <b>not monotonic</b> — the peak round varies ({PEAK_ROUNDS.join(", ")}) and over-iterating a reactive
        clone <b>regresses</b>, so you <b>select the best round, not the last</b> (Ross et al.). Honest ceiling ≈{Math.round(CEILING * 100)}%
        (reactive MLP). Real measured numbers from dagger.py (N={N_POOLED}/round, seeds 0–2, cpu); the seed-0 recovery reads with JS off.
      </p>
    </div>
  );
}

export default function DaggerRecovery() {
  return <DaggerRecoveryToy />;
}
