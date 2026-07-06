/**
 * ch1.2 "Data Is the Policy" — the DATA-CURATION concept-toy (`demo: pusht_curate_quality`).
 *
 * The chapter's headline, made tactile: at IDENTICAL method and compute (chapter
 * 1.1's behavior cloning, unchanged), the ONLY lever is the dataset. Expose one
 * control — HOW you pick which of your 500 recorded demonstrations to keep — and
 * re-render the measured held-out success. Keeping everything ("more data") scores
 * 8%; keeping only the demos that reached the goal ("better data", 288 of 500 —
 * FEWER episodes) scores 22%; and the seductive trap (keep the demos that AGREE
 * most, low disagreement) keeps 288 too yet scores only 12%. Quality beats
 * quantity, and WHICH demos you keep beats the tidiest-looking metric.
 *
 * This is a SCHEMATIC toy — no WASM, no model run. Every quantitative readout is a
 * REAL measured scalar from the chapter's reference run (see PROVENANCE below);
 * only the spatial ARRANGEMENT of the episode dots is illustrative.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT documented at the top of
 * ../PlateIsland.tsx (read it before touching this):
 *   1. SSR poster == the JS-off fallback. <Poster/> is pure static JSX; with JS
 *      off it is the whole experience (the measured raw-vs-curated comparison,
 *      fully captioned for screen readers). Nothing reads window/document at
 *      module scope or first render.
 *   2. No heavy deps to lazy-load here (no sim), so the live view boots
 *      immediately in a post-hydration effect; the poster is the pre-boot frame.
 *   3. The visual is bespoke SVG in the page's design tokens.
 *   4. Make the invisible visible: the recorded pile as a unit-chart of 500
 *      episode dots (reached-goal vs not), the KEPT subset highlighted per
 *      strategy, and the held-out success as bars against the raw baseline.
 *   5. ONE control (which demos to keep), immediate feedback, default-interesting
 *      (boots showing the curated WIN over raw); a keyboard-native radiogroup.
 *   6. Colour discipline: --entity-target emerald for the reached-goal demos and
 *      the curated win, neutral --ink-mute for the culled/failed demos, ONE
 *      --alert red reserved for the Break-It trap, --signal for the live focus ring.
 *
 * ============================================================================
 * PROVENANCE — every number below is a measured scalar copied verbatim from
 *   curriculum/phase1_imitation/ch1.2_curate/meta.yaml (reference_run):
 *   seed 0, cpu, measured 2026-07-05 (mujoco 3.10.0 / torch 2.10.0 /
 *   lerobot 0.4.4 / numpy 2.4.6); default config (careful 250 @ noise 0.05 +
 *   sloppy 250 @ noise 0.70 = 500 episodes, epochs 300, hidden_dim 256,
 *   eval_episodes 50).
 *     n_episodes 500 · n_reached_goal 288
 *     raw_success_rate 0.08 · curated_success_rate 0.22 · delta 0.14
 *     break_low_disagreement_success 0.12
 *     far starts (meta line 33): raw 2/31 · curated 8/31 · break 5/31
 *     mean_disagreement: raw 0.442585 · curated 0.380268 · break 0.378111
 *   NO number here is invented; a learner who runs curate.py --seed 0 --device
 *   cpu reproduces them, and one who passes --data <their ch0.4 dataset> gets
 *   different absolute digits but the same qualitative story.
 * ============================================================================
 */
import "./CurateQualityToy.css";
import { useEffect, useRef, useState } from "preact/hooks";

// ------------------------------------------------------------------- measured facts
const N_EPISODES = 500; // meta: n_episodes (250 careful + 250 sloppy stand-in)
const N_REACHED = 288; // meta: n_reached_goal — episodes that finished inside PushT's tolerance
// Success is measured over 50 held-out reset seeds (meta provenance: eval_episodes 50),
// 31 of which are "far" starts (start_dist >= 0.15 m) — the split meta line 33 reports.
const FAR_DENOM = 31;

type Kind = "raw" | "good" | "trap";
interface Strategy {
  id: "raw" | "curated" | "break";
  kind: Kind;
  label: string; // the choice
  sub: string; // one-line gloss of the criterion
  kept: number; // # episodes trained on
  success: number; // measured held-out success rate
  farKept: number; // measured successes on the 31 far starts
  meanDisagree: number; // measured mean neighbour-disagreement of the kept set
}

// The three MEASURED operating points — this IS the whole control surface, and
// every scalar is a meta.yaml reference_run value (see PROVENANCE above). We do
// not interpolate "how many" between them, because no measurement exists there.
const STRATEGIES: Strategy[] = [
  {
    id: "raw",
    kind: "raw",
    label: "keep everything",
    sub: "train on all 500 — more data",
    kept: N_EPISODES,
    success: 0.08,
    farKept: 2,
    meanDisagree: 0.442585,
  },
  {
    id: "curated",
    kind: "good",
    label: "keep what reached the goal",
    sub: "outcome filter — better data",
    kept: N_REACHED,
    success: 0.22,
    farKept: 8,
    meanDisagree: 0.380268,
  },
  {
    id: "break",
    kind: "trap",
    label: "keep the demos that agree most",
    sub: "low disagreement — the trap",
    kept: N_REACHED, // meta: the break keeps the SAME COUNT (288), a different WHICH
    success: 0.12,
    farKept: 5,
    meanDisagree: 0.378111,
  },
];
const byId = (id: string) => STRATEGIES.find((s) => s.id === id)!;
const RAW = byId("raw");

const BAR_MAX = 0.28; // full-width success for the bars (headroom above curated's 0.22)
const pct = (r: number) => `${Math.round(r * 100)}%`;
const deltaPts = (r: number) => Math.round((r - RAW.success) * 100); // vs the raw baseline, in points

// --------------------------------------------------------- the recorded pile (unit chart)
// 500 dots on a 25x20 grid, ordered reached-goal first (0..287) then not-reached
// (288..499). Which dots a strategy KEEPS:
//   raw     — all 500.
//   curated — exactly the 288 that reached the goal (a clean emerald block).
//   break   — 288 too, but chosen by neighbour-agreement, not outcome: it drops
//             the hardest reached demos and readmits the easiest failures. The
//             COUNT (288) is measured; this membership is a schematic illustration
//             of "keeps the easy ones, outcome be damned" (BREAK_SWAP is a
//             deliberately modest, illustrative number — the real kept-set
//             disagreement, 0.378, sits a hair below curated's 0.380, so the two
//             sets genuinely differ only at the margin).
const GRID_COLS = 25;
const GRID_ROWS = 20;
const BREAK_SWAP = 40; // illustrative: hard successes dropped == easy failures admitted

function isKept(id: string, i: number): boolean {
  if (id === "raw") return true;
  if (id === "curated") return i < N_REACHED;
  // break (illustrative membership; count is measured):
  if (i < N_REACHED - BREAK_SWAP) return true; // reached, kept
  if (i >= N_REACHED && i < N_REACHED + BREAK_SWAP) return true; // failed but "agreeable", readmitted
  return false;
}

const DOT_CELL = 19;
const DOT_R = 6;
const PILE_W = GRID_COLS * DOT_CELL; // 475
const PILE_H = GRID_ROWS * DOT_CELL; // 380

function Pile({ id }: { id: string }) {
  const dots = [];
  for (let i = 0; i < N_EPISODES; i++) {
    const col = i % GRID_COLS;
    const row = Math.floor(i / GRID_COLS);
    const reached = i < N_REACHED;
    const kept = isKept(id, i);
    dots.push(
      <circle
        class={`cq-dot ${reached ? "cq-dot--reached" : "cq-dot--failed"} ${kept ? "cq-dot--kept" : "cq-dot--culled"}`}
        cx={col * DOT_CELL + DOT_CELL / 2}
        cy={row * DOT_CELL + DOT_CELL / 2}
        r={DOT_R}
      />,
    );
  }
  return (
    <svg class="cq-pile-svg" viewBox={`0 0 ${PILE_W} ${PILE_H}`} aria-hidden="true">
      {dots}
    </svg>
  );
}

// ----------------------------------------------------------------- the result bars
function SuccessBar({ label, rate, kind, muted }: { label: string; rate: number; kind: Kind; muted?: boolean }) {
  const w = Math.min(100, (rate / BAR_MAX) * 100);
  return (
    <div class="cq-bar-row" data-muted={muted ? "true" : "false"}>
      <span class="cq-bar-label">{label}</span>
      <div class="cq-bar-track">
        <div class={`cq-bar-fill cq-bar-fill--${kind}`} style={`width:${w.toFixed(1)}%`} />
      </div>
      <span class="cq-bar-val">{pct(rate)}</span>
    </div>
  );
}

/** The shared visual — pile + result — rendered for a given strategy. Pure &
 *  decorative (aria-hidden), used by both the SSR poster and the live view. */
function Panel({ id }: { id: string }) {
  const s = byId(id);
  const dp = deltaPts(s.success);
  return (
    <div class="cq-panel">
      <div class="cq-pile" aria-hidden="true">
        <div class="cq-pile-head">
          <span class="cq-pile-title">the 500 episodes you recorded</span>
          <span class="cq-legend">
            <span class="cq-legend-item"><i class="cq-swatch cq-swatch--reached" />reached goal · {N_REACHED}</span>
            <span class="cq-legend-item"><i class="cq-swatch cq-swatch--failed" />didn’t · {N_EPISODES - N_REACHED}</span>
          </span>
        </div>
        <Pile id={id} />
        <div class="cq-pile-foot">
          <span>
            training on <b>{s.kept}</b> of {N_EPISODES} demos
          </span>
          <span>{s.id === "raw" ? "the whole pile" : s.id === "curated" ? "the goal-reachers only" : "288 by agreement, not outcome"}</span>
        </div>
      </div>

      <div class="cq-result" aria-hidden="true">
        <div class="cq-result-head">
          <span class="cq-result-title">held-out success</span>
          <span class={`cq-delta cq-delta--${s.kind}`}>
            {dp > 0 ? `+${dp}` : dp} pts vs raw
          </span>
        </div>
        <SuccessBar label="raw · all 500" rate={RAW.success} kind="raw" muted={s.id !== "raw"} />
        <SuccessBar label={s.id === "raw" ? " " : s.label} rate={s.success} kind={s.kind} />
        <div class="cq-result-foot">
          <span class="cq-foot-k">far starts</span>
          <span class="cq-foot-v">
            {s.farKept}/{FAR_DENOM}
            <span class="cq-foot-sub"> · raw {RAW.farKept}/{FAR_DENOM}</span>
          </span>
          <span class="cq-foot-k">kept-set disagreement</span>
          <span class="cq-foot-v">{s.meanDisagree.toFixed(3)}</span>
        </div>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------- SSR poster
// Boots showing the curated WIN over raw — the default-interesting aha. Pure
// static JSX; the role=img wrapper carries the full description for JS-off /
// screen-reader users (the inner SVG + text are aria-hidden).
function Poster() {
  return (
    <div
      class="cq-poster-fig"
      role="img"
      aria-label={
        `Data-curation comparison, top: the 500 PushT demonstrations you recorded as a grid of dots — ` +
        `${N_REACHED} reached the goal, ${N_EPISODES - N_REACHED} did not. Bottom: held-out success rate. ` +
        `Training chapter 1.1's behavior cloning on all 500 episodes scores 8 percent. Curating down to just the ` +
        `${N_REACHED} episodes that reached the goal — fewer demonstrations — scores 22 percent, a 14-point gain ` +
        `concentrated on the far starts (8 of 31 versus raw's 2 of 31). Same method, same compute: better data, not ` +
        `more. With JavaScript on, one control lets you switch how the demos are picked, including the trap that keeps ` +
        `the demos that agree most and scores only 12 percent.`
      }
    >
      <Panel id="curated" />
    </div>
  );
}

// ------------------------------------------------------------------- live island
function CurateToy() {
  const [booted, setBooted] = useState(false);
  const [selected, setSelected] = useState<string>("curated"); // default-interesting: the win
  const optionRefs = useRef<Record<string, HTMLButtonElement | null>>({});

  useEffect(() => {
    // No WASM / heavy deps: the live view is ready as soon as we hydrate.
    setBooted(true);
  }, []);

  const select = (id: string, focus = false) => {
    setSelected(id);
    if (focus) optionRefs.current[id]?.focus();
  };

  // Keyboard-native radiogroup (roving tabindex): arrows move + select, Home/End jump.
  const onKeyDown = (e: KeyboardEvent) => {
    const ids: string[] = STRATEGIES.map((s) => s.id);
    const i = ids.indexOf(selected);
    let next = -1;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") next = (i + 1) % ids.length;
    else if (e.key === "ArrowLeft" || e.key === "ArrowUp") next = (i - 1 + ids.length) % ids.length;
    else if (e.key === "Home") next = 0;
    else if (e.key === "End") next = ids.length - 1;
    if (next >= 0) {
      e.preventDefault();
      select(ids[next], true);
    }
  };

  const s = byId(selected);
  const dp = deltaPts(s.success);

  // Non-visual path to the same aha: announce the selected strategy's qualitative
  // result (the visual Panel is aria-hidden).
  const announce =
    s.id === "raw"
      ? `Keeping everything: all ${N_EPISODES} episodes, held-out success ${pct(s.success)}. This is the "more data" baseline.`
      : s.id === "curated"
        ? `Curated by outcome: trained on ${s.kept} of ${N_EPISODES} episodes — fewer demonstrations — yet held-out success rises from ${pct(RAW.success)} raw to ${pct(s.success)}, up ${dp} points. The whole gain is on the far starts, ${s.farKept} of ${FAR_DENOM} against raw's ${RAW.farKept} of ${FAR_DENOM}. Better data, not more.`
        : `Low-disagreement filter: it keeps ${s.kept} episodes too, but chosen by which demos agree most, not by outcome. Success is ${pct(s.success)} — above raw's ${pct(RAW.success)} but well short of curated's ${pct(byId("curated").success)}, and it gives up on the far starts, ${s.farKept} of ${FAR_DENOM}. Its kept set even has slightly lower disagreement than honest curation, yet a worse policy: a quality signal is not a quality objective.`;

  return (
    <div class="cq">
      <figure class="cq-figure">
        {/* SSR poster — the JS-off experience and the pre-boot frame */}
        <div class="cq-poster" hidden={booted}><Poster /></div>

        {/* live view — same layout, so booting causes no reflow */}
        <div class="cq-live" hidden={!booted}>
          <Panel id={selected} />
        </div>

        <div class="bk-sr" aria-live="polite">{booted ? announce : ""}</div>
      </figure>

      {/* THE one control — how do you pick which demos to keep? (radiogroup) */}
      <div class="cq-controls">
        <div class="cq-control-lead" id="cq-q">which demos do you keep?</div>
        <div
          class="cq-radiogroup"
          role="radiogroup"
          aria-labelledby="cq-q"
          onKeyDown={onKeyDown}
        >
          {STRATEGIES.map((opt) => {
            const on = opt.id === selected;
            return (
              <button
                type="button"
                ref={(el) => { optionRefs.current[opt.id] = el as HTMLButtonElement | null; }}
                class="cq-option"
                role="radio"
                aria-checked={on ? "true" : "false"}
                data-kind={opt.kind}
                tabIndex={on ? 0 : -1}
                disabled={!booted}
                onClick={() => select(opt.id)}
              >
                <span class="cq-option-label">{opt.label}</span>
                <span class="cq-option-sub">{opt.sub}</span>
                <span class="cq-option-stat">
                  <b>{pct(opt.success)}</b> · {opt.kept} demos
                </span>
              </button>
            );
          })}
        </div>
        <p class="cq-note">
          same behavior cloning, same compute — the dataset is the only lever · curated beats raw on
          fewer demos · poster reads with JS off
        </p>
      </div>
    </div>
  );
}

export default function CurateQualityToy() {
  return <CurateToy />;
}
