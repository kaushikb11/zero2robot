/**
 * ch2.7 "Sim-to-Real Intuition Lab II: Randomize to Generalize" — the DOMAIN
 * RANDOMIZATION concept-toy (`demo: narrow_vs_randomized_across_the_gap`). A
 * recorded-data sibling of ch3.3's EngineDriftToy and ch1.6's EvalBandsToy: pure
 * SVG over REAL measured numbers dr.py produced — no WASM, no invented shapes.
 *
 * THE HONEST STORY (this is the whole point — do not let the UI overclaim):
 *   Domain randomization is the PROMISE you TEST, not a guaranteed win. dr.py
 *   trains a NARROW policy (nominal dynamics only) and a RANDOMIZED policy
 *   (mass/friction/gravity resampled per episode), then sweeps BOTH across a
 *   shifted-mass gap. At this free-tier budget (400k steps/policy, seeds 0-2) the
 *   off-nominal survival edge is {-0.02, +0.22, -0.09} — it SWINGS with the seed
 *   and its mean (+4% ± 13%) sits INSIDE the seed band. The randomized policy does
 *   NOT cleanly beat narrow across the gap. That within-band result IS the lesson
 *   (ch1.6's "single numbers lie" in an RL costume).
 *
 * TWO panels, ONE control:
 *   1. THE GAP SWEEP (headline): survival rate vs shifted test-mass for narrow vs
 *      randomized, with ch1.6-style error bars = the ±std SEED BAND across seeds
 *      0-2. Both stand at nominal (100%); both collapse past ~1.2x (the ±12 Nm
 *      servos saturate — the actuator ceiling DR cannot lift); in between the two
 *      shaded bands OVERLAP everywhere — the edge is inside the band.
 *   2. THE INSURANCE-PREMIUM READOUT (honestly): nominal (both 100%, no premium
 *      paid at this budget) vs off-nominal-mean survival, and the DR edge with its
 *      per-seed spread {-0.02,+0.22,-0.09} shown explicitly.
 *   The ONE control is a SEED selector — the "single numbers lie" reveal: the
 *   default "mean ± band" view shows overlapping bands (honest headline); flip to
 *   seed 1 and randomized tempts you with a clean win (holds to 1.4x where narrow
 *   is gone); flip to seeds 0/2 and the edge vanishes or reverses. Same policies,
 *   same sweep — the seed you pick changes the story.
 *
 * This needs NO WASM: the survival curves + seed band come from
 * site/scripts/vizdata/ch2.7_dr.py, which transcribes dr.py's MEASURED
 * reference_run and STOPS on any drift from meta.yaml. So it is pure SVG + design
 * tokens: theme-aware for free, and the server-rendered default (the mean ± band
 * gap sweep) IS the JS-off experience — only the seed selector goes inert without
 * hydration. Colour by MEANING: narrow = --entity-block magenta (the overfit
 * baseline), randomized = --entity-target green (the policy under test), --alert
 * red for the actuator-ceiling note where both fall.
 */
import "./DomainRandToy.css";
import { useEffect, useMemo, useState } from "preact/hooks";
// Real measured numbers from dr.py's reference_run — see the file's `provenance`
// and the generator site/scripts/vizdata/ch2.7_dr.py. Committed small text, no binary.
import viz from "../../../../curriculum/phase2_reinforcement/ch2.7_dr/demo/vizdata.json";

// ---------------------------------------------------------------- typed vizdata
type Policy = "narrow" | "randomized";
interface AggPoint { scale: number; mean: number; std: number; lo: number; hi: number; seeds: number[]; }
interface SeedEntry {
  narrow: number[]; randomized: number[];
  narrow_offnominal: number; randomized_offnominal: number; edge: number;
}

const GRID = viz.sweep_grid as number[];
const KNOB = viz.sweep_knob as string;
const SEEDS = viz.seeds as number[];
const POLICIES = viz.policies as Policy[];
const AGG = viz.aggregate as Record<Policy, AggPoint[]>;
const PER_SEED = viz.per_seed as Record<string, SeedEntry>;
const EDGE = viz.edge as { per_seed: number[]; per_seed_rounded: number[]; mean: number; std: number; within_band: boolean; meta: number[]; };
const NOMINAL = viz.nominal as Record<Policy, number>;
const OFFNOMINAL = viz.offnominal as Record<Policy, number>;
const RETURN_BAND = viz.nominal_return_band as [number, number];

const CLS: Record<Policy, string> = { narrow: "dr-narrow", randomized: "dr-rand" };
const LABEL: Record<Policy, string> = { narrow: "narrow", randomized: "randomized" };

// ------------------------------------------------------------ number formatting
const pct = (v: number): string => `${Math.round(v * 100)}%`;
const signedPct = (v: number): string => `${v >= 0 ? "+" : "−"}${Math.round(Math.abs(v) * 100)}%`;

type View = "band" | string; // "band" = aggregate mean ± seed band; else a seed id.

// ================================================================ THE GAP SWEEP
const CW = 540, CH = 300;
const CPAD = { l: 46, r: 18, t: 16, b: 42 };
const CPW = CW - CPAD.l - CPAD.r;
const CPH = CH - CPAD.t - CPAD.b;
// x maps mass scale; a little margin past the end stops so end points aren't on the frame.
const XMIN = GRID[0] - 0.1, XMAX = GRID[GRID.length - 1] + 0.1;
const px = (scale: number) => CPAD.l + ((scale - XMIN) / (XMAX - XMIN)) * CPW;
const py = (surv: number) => CPAD.t + (1 - surv) * CPH; // survival 0..1, 1 at top
const Y_TICKS = [0, 0.25, 0.5, 0.75, 1.0];

/** The chart body — shared verbatim by the SSR figure (JS-off fallback + pre-boot
 *  frame; view="band") and the hydrated island, so a seed flip causes no reflow. */
function GapSweep({ view }: { view: View }) {
  const isBand = view === "band";
  const seed = isBand ? null : PER_SEED[view];

  // aggregate error band (lo..hi) as a closed polygon per policy — the SEED BAND.
  const bandPoly = (p: Policy) => {
    const pts = AGG[p];
    const top = pts.map((d) => `${px(d.scale).toFixed(1)},${py(d.hi).toFixed(1)}`);
    const bot = pts.slice().reverse().map((d) => `${px(d.scale).toFixed(1)},${py(d.lo).toFixed(1)}`);
    return [...top, ...bot].join(" ");
  };
  const meanLine = (p: Policy) => AGG[p].map((d) => `${px(d.scale).toFixed(1)},${py(d.mean).toFixed(1)}`).join(" ");
  const seedLine = (p: Policy) => (seed as SeedEntry)[p].map((v, i) => `${px(GRID[i]).toFixed(1)},${py(v).toFixed(1)}`).join(" ");

  const ariaLabel = isBand
    ? `Survival rate versus shifted test-${KNOB} scale for the narrow and randomized policies, mean over seeds 0 to 2 ` +
      "with the seed band as error bars. Both policies survive every episode at nominal and below; both collapse past 1.2 times mass. " +
      "Between, the two shaded seed bands overlap at every shifted mass — the randomized policy does not cleanly beat the narrow policy."
    : `Survival rate versus shifted test-${KNOB} scale for seed ${view}, narrow versus randomized policy, drawn over the faint ` +
      "cross-seed band for context. A single seed can poke above or below the band — which is exactly why one seed cannot settle the ranking.";

  return (
    <svg class="dr-svg" viewBox={`0 0 ${CW} ${CH}`} role="img" aria-label={ariaLabel}>
      {/* survival gridlines + % ticks */}
      {Y_TICKS.map((t) => (
        <g>
          <line class="dr-grid" x1={CPAD.l} y1={py(t)} x2={CW - CPAD.r} y2={py(t)} />
          <text class="dr-tick" x={CPAD.l - 6} y={py(t) + 3} text-anchor="end">{pct(t)}</text>
        </g>
      ))}

      {/* the actuator ceiling: past ~1.2x mass the servos saturate and BOTH fall.
          DR widens the reachable range; it never lifts this physical ceiling. */}
      <rect class="dr-ceiling" x={px(1.3).toFixed(1)} y={CPAD.t}
        width={(CW - CPAD.r - px(1.3)).toFixed(1)} height={CPH} />
      <text class="dr-ceiling-lab" x={((px(1.3) + (CW - CPAD.r)) / 2).toFixed(1)} y={CPAD.t + 13} text-anchor="middle">
        servos saturate · both fall
      </text>

      {/* nominal guide — the point both policies trained around */}
      <line class="dr-nominal" x1={px(1.0)} y1={CPAD.t} x2={px(1.0)} y2={CH - CPAD.b} />
      <text class="dr-nominal-lab" x={px(1.0)} y={CPAD.t - 4} text-anchor="middle">nominal</text>

      {/* x axis + mass-scale ticks */}
      <line class="dr-axis" x1={CPAD.l} y1={CH - CPAD.b} x2={CW - CPAD.r} y2={CH - CPAD.b} />
      {GRID.map((s) => (
        <text class="dr-tick" x={px(s)} y={CH - CPAD.b + 15} text-anchor="middle">{s.toFixed(1)}×</text>
      ))}
      <text class="dr-axis-title" x={CW - CPAD.r} y={CH - 6} text-anchor="end">
        test {KNOB} scale — the gap →
      </text>
      <text class="dr-axis-title dr-ytitle" x={CPAD.l - 40} y={CPAD.t - 4}>survival</text>

      {/* BAND VIEW: two overlapping seed bands + mean lines + whisker caps + dots */}
      {isBand ? (
        <>
          {POLICIES.map((p) => <polygon class={`dr-band ${CLS[p]}`} points={bandPoly(p)} />)}
          {POLICIES.map((p) => (
            <>
              {/* whisker caps at each mass point (the classic ch1.6 error bar) */}
              {AGG[p].map((d) => d.std > 0 && (
                <line class={`dr-whisker ${CLS[p]}`} x1={px(d.scale)} y1={py(d.lo)} x2={px(d.scale)} y2={py(d.hi)} />
              ))}
              <polyline class={`dr-line ${CLS[p]}`} points={meanLine(p)} />
              {AGG[p].map((d) => <circle class={`dr-dot ${CLS[p]}`} cx={px(d.scale)} cy={py(d.mean)} r={3.2} />)}
            </>
          ))}
        </>
      ) : (
        <>
          {/* SEED VIEW: faint cross-seed band for context, bold single-seed lines on top */}
          {POLICIES.map((p) => <polygon class={`dr-band dr-band--ghost ${CLS[p]}`} points={bandPoly(p)} />)}
          {POLICIES.map((p) => (
            <>
              <polyline class={`dr-line dr-line--seed ${CLS[p]}`} points={seedLine(p)} />
              {(seed as SeedEntry)[p].map((v, i) => (
                <circle class={`dr-dot ${CLS[p]}`} cx={px(GRID[i])} cy={py(v)} r={3.4} />
              ))}
            </>
          ))}
        </>
      )}
    </svg>
  );
}

// ============================================================ INSURANCE READOUT
function edgeStory(edge: number): string {
  if (edge >= 0.1) return "randomized holds where narrow falls — but this is one seed";
  if (edge <= -0.05) return "narrow actually survives off-nominal better here";
  return "no measurable randomized edge — they fall together";
}

// ==================================================================== THE ISLAND
function DomainRandToy() {
  const [view, setView] = useState<View>("band");
  const [booted, setBooted] = useState(false);
  useEffect(() => { setBooted(true); }, []); // controls are JS-only; the band chart is the JS-off fallback

  const isBand = view === "band";
  const seed = isBand ? null : PER_SEED[view];

  // the edge shown in the readout: aggregate mean (band) or the selected seed's.
  const edgeVal = isBand ? EDGE.mean : (seed as SeedEntry).edge;
  const offN = isBand ? OFFNOMINAL.narrow : (seed as SeedEntry).narrow_offnominal;
  const offR = isBand ? OFFNOMINAL.randomized : (seed as SeedEntry).randomized_offnominal;

  const announce = useMemo(() => {
    if (isBand) {
      return `Mean over seeds 0 to 2, with the seed band. Both policies survive every episode at nominal ${KNOB}, ` +
        "and both collapse past 1.2 times mass where the servos saturate. Across the gap the randomized policy's " +
        `off-nominal survival edge is ${signedPct(EDGE.mean)} plus or minus ${pct(EDGE.std)} — it sits inside the seed band. ` +
        "The two error bands overlap at every shifted mass, so at this budget domain randomization does not cleanly beat the narrow policy.";
    }
    const e = (seed as SeedEntry).edge;
    const others = SEEDS.filter((s) => `${s}` !== view).join(" and ");
    return `Seed ${view}: the off-nominal survival edge is ${signedPct(e)} — ${edgeStory(e)}. ` +
      `Seeds ${others} tell a different story; a single seed cannot settle whether randomization helps.`;
  }, [view]);

  return (
    <div class="dr">
      {/* --- the ONE control: the seed selector (the "single numbers lie" reveal) --- */}
      <div class="dr-controls">
        <span class="dr-seg-label">view</span>
        <div class="dr-seg" role="group" aria-label="Choose the aggregate seed band or a single seed">
          <button type="button" class="dr-seg-btn" aria-pressed={isBand} disabled={!booted}
            onClick={() => setView("band")}>mean ± band</button>
          {SEEDS.map((s) => (
            <button type="button" class="dr-seg-btn" aria-pressed={view === `${s}`} disabled={!booted}
              onClick={() => setView(`${s}`)}>seed {s}</button>
          ))}
        </div>
        <span class="dr-seg-hint">
          {isBand ? "the honest headline: the bands overlap" : `edge ${signedPct(edgeVal)} — but flip the seed`}
        </span>
      </div>

      {/* --- the two panels: the gap sweep + the insurance-premium readout --- */}
      <div class="dr-panels">
        <figure class="dr-panel dr-panel--chart">
          <figcaption class="dr-cap">
            <span>survival across the gap</span>
            <b>{isBand ? "mean ± seed band" : `seed ${view}`}</b>
          </figcaption>
          <GapSweep view={view} />
          <div class="dr-legend" aria-hidden="true">
            {POLICIES.map((p) => (
              <span class={`dr-lg ${CLS[p]}`}><span class="dr-lg-name">{LABEL[p]}</span></span>
            ))}
            <span class="dr-lg-note">{isBand ? "band = ±std over seeds 0–2" : "faint band = the seed spread"}</span>
          </div>
        </figure>

        <figure class="dr-panel dr-panel--readout">
          <figcaption class="dr-cap">
            <span>the insurance premium</span>
            <b>{isBand ? "off-nominal mean" : `seed ${view}`}</b>
          </figcaption>
          <div class="dr-readout" aria-hidden="true">
            <div class="dr-ro-row dr-ro-head">
              <span class="dr-ro-k" />
              <span class="dr-ro-narrow">narrow</span>
              <span class="dr-ro-rand">randomized</span>
            </div>
            <div class="dr-ro-row">
              <span class="dr-ro-k">at nominal (1.0×)</span>
              <span class="dr-ro-v dr-ro-narrow">{pct(NOMINAL.narrow)}</span>
              <span class="dr-ro-v dr-ro-rand">{pct(NOMINAL.randomized)}</span>
            </div>
            <div class="dr-ro-row">
              <span class="dr-ro-k">across the gap</span>
              <span class="dr-ro-v dr-ro-narrow">{pct(offN)}</span>
              <span class="dr-ro-v dr-ro-rand">{pct(offR)}</span>
            </div>
            <div class="dr-ro-edge">
              <span class="dr-ro-k">DR survival edge</span>
              <span class={`dr-ro-edgev ${edgeVal > 0.1 ? "dr-up" : edgeVal < -0.02 ? "dr-down" : "dr-flat"}`}>
                {signedPct(edgeVal)}{isBand ? ` ± ${pct(EDGE.std)}` : ""}
              </span>
            </div>
          </div>

          {/* the per-seed edge chips: the spread that makes the mean within-band.
              Doubles as a legend for the seed selector — the active seed lights up. */}
          <div class="dr-chips" aria-hidden="true">
            <span class="dr-chips-lab">per-seed edge</span>
            {SEEDS.map((s, i) => (
              <span class={`dr-chip ${EDGE.per_seed_rounded[i] > 0.1 ? "dr-up" : EDGE.per_seed_rounded[i] < -0.02 ? "dr-down" : "dr-flat"}`}
                data-active={view === `${s}`}>
                s{s} {signedPct(EDGE.per_seed_rounded[i])}
              </span>
            ))}
          </div>
          <p class="dr-premium-note">
            {isBand
              ? <>No premium is even paid at nominal here — both stand 100% and return is flat (~{RETURN_BAND[0]}–{RETURN_BAND[1]}). The hoped-for payout off-nominal is <b>{signedPct(EDGE.mean)} ± {pct(EDGE.std)}</b>: inside the seed band.</>
              : <>On this one seed the edge is <b>{signedPct(edgeVal)}</b> — {edgeStory(edgeVal)}.</>}
          </p>
        </figure>
      </div>

      {/* non-visual path to the same aha — the qualitative reading only, never per-frame spam */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      <p class="dr-note">
        <b>DR is the promise you TEST, not a clean win.</b> At this free-tier budget the off-nominal edge sits
        {" "}<b>within the seed band</b> ({signedPct(EDGE.mean)} ± {pct(EDGE.std)}; per-seed {EDGE.per_seed_rounded.map(signedPct).join(", ")}),
        {" "}and past ~1.2× mass the ±12 Nm servos saturate so <b>both</b> policies fall — DR widens the reachable range, it never
        {" "}lifts the actuator ceiling. The <b>Scale Lab</b> spends the compute (and/or a walking gait) to make randomization converge.
        {" "}Real measured numbers from dr.py (seeds 0–2, cpu); the mean ± band view reads with JS off.
      </p>
    </div>
  );
}

export default function DomainRand() {
  return <DomainRandToy />;
}
