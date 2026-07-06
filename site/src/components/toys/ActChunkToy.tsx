/**
 * ActChunkToy — ch1.3 "ACT: Commit to the Chunk" concept-toy (demo id
 * `aloha_cube_chunk`).
 *
 * A pure 2-D SCHEMATIC of action chunking + temporal ensembling — NO WASM, NO
 * model run, no binary. Everything drawn here is either the exact structure of
 * ACT's eval loop (which timesteps each predicted chunk covers) or a closed-form
 * computed CLIENT-SIDE (the exp(-m·i) ensemble weights). It teaches two ideas:
 *
 *   1. ACTION CHUNKING — at every timestep the policy emits a CHUNK of the next
 *      K actions in one forward pass, so successive chunks OVERLAP. Drawn as a
 *      staggered staircase of bars: the chunk predicted at query-time q covers
 *      timesteps q … q+K-1.
 *   2. TEMPORAL ENSEMBLING — a single timestep t* is covered by up to K chunks
 *      predicted at earlier steps. ACT combines those overlapping predictions
 *      with a weighted vote, weight = exp(-m·i) where i is the chunk's AGE and
 *      i=0 is the OLDEST prediction — so the OLDEST vote is weighted MOST. The
 *      weight bars re-render live as you drag the ensemble coefficient m.
 *
 * WEIGHTING DIRECTION — faithful to act.py's eval region (do NOT invert):
 *     votes   = all_time[:t+1, t][populated[:t+1, t]]     # oldest first
 *     weights = np.exp(-args.ensemble_m * np.arange(len(votes)))
 *     action  = (votes * (weights / weights.sum())[:, None]).sum(0)
 *   `np.arange` starts at 0, `votes` are oldest-first, so i=0 (oldest) gets
 *   exp(0)=1, the LARGEST weight; larger m concentrates weight on that oldest
 *   prediction (act.py --ensemble_m help: "larger = concentrate weight on the
 *   OLDEST overlapping prediction"). `ensembleWeights` below mirrors this exactly.
 *
 * Built to the FROZEN CONCEPT-TOY CONTRACT (top of ../PlateIsland.tsx), the pure-
 * SVG variant like ch0.3's frames-drag: the SSR poster IS the JS-off experience
 * (the same <svg>, default params, hydrated in place with no reflow); ONE primary
 * control (the m slider) with immediate feedback and a default-interesting start;
 * colour discipline (ONE --signal blue for the live/voting elements, --alert only
 * for the no-chunk/off degenerate readout, neutral ink for the timeline map);
 * keyboard + aria-live parity; reduced-motion friendly (no rAF, transitions gated).
 *
 * -------------------------------------------------------------------------
 * WIRE ME IN (PlateIsland.tsx — orchestrator does this; do NOT edit it here):
 *   import ActChunkToy from "./toys/ActChunkToy";
 *   // …in PlateIsland()'s dispatch, beside the other demo ids:
 *   if (demo === "aloha_cube_chunk") return <ActChunkToy />;   // ch1.3 chunking + temporal ensembling
 * -------------------------------------------------------------------------
 */
import { useEffect, useRef, useState } from "preact/hooks";
import "./ActChunkToy.css";

// ---------------------------------------------------------------- measured constants
// Provenance: curriculum/phase1_imitation/ch1.3_act/meta.yaml → reference_run
// (seed 0, cpu, default config chunk_size 8 / eval_episodes 25, measured 2026-07-06).
// These are documented scalars, not run here — this toy never runs the model.
const SUCCESS_ENSEMBLE = 0.88; // trained ACT, chunk + temporal ensembling
const SUCCESS_NO_CHUNK = 0.6;  // break_no_chunk (K=1): single-step BC through the transformer
// act.py defaults (also curriculum/…/ch1.3_act/demo/embed.yaml runtime.*):
const M_DEFAULT = 0.1; // --ensemble_m default
const K_DEFAULT = 8;   // --chunk_size default

// ---------------------------------------------------------------- toy parameters
const N = 12;          // timesteps shown on the timeline (aloha episodes are ~27 steps)
const K_MIN = 1, K_MAX = 12;
const M_MIN = 0, M_MAX = 1, M_STEP = 0.02;
const TSTAR_DEFAULT = 7; // a full-overlap interior column at K=8 (votes = K)

// SVG frame + panel geometry (one landscape figure; poster == live, no reflow)
const W = 680, H = 560;
const GX0 = 150, GX1 = 656;          // timeline plot x-extent (room for row labels)
const CW = (GX1 - GX0) / N;          // column width (one timestep)
const A_TOP = 66, A_BOT = 300;       // staircase panel y-extent
const RH = (A_BOT - A_TOP) / N;      // one chunk row
const B_BASE = 492, B_TOP = 372;     // weight-bar baseline + tallest-bar top
const B_MAXH = B_BASE - B_TOP;

const colLeft = (t: number) => GX0 + t * CW;
const colMid = (t: number) => GX0 + (t + 0.5) * CW;
const fmt = (n: number, d = 2) => (Object.is(n, -0) ? 0 : n).toFixed(d);
const pct = (x: number) => `${Math.round(x * 100)}%`;

// ---------------------------------------------------------------- the ensemble maths
/** Temporal-ensembling weights, IDENTICAL in direction to act.py:
 *    weights = np.exp(-m * np.arange(V));  weights /= weights.sum()
 *  votes are OLDEST-FIRST, so i=0 is the oldest overlapping prediction and gets
 *  the LARGEST weight (exp(0)=1). Returns the normalized shares (sum to 1). */
function ensembleWeights(m: number, votes: number): number[] {
  const raw = Array.from({ length: votes }, (_, i) => Math.exp(-m * i)); // i=0 oldest = max
  const s = raw.reduce((a, b) => a + b, 0) || 1;
  return raw.map((w) => w / s);
}

interface Vote { q: number; i: number; share: number; }
interface Model { votes: Vote[]; qLo: number; oldestShare: number; newestShare: number; }

/** The V chunks that overlap timestep t*, oldest → newest, with their vote shares.
 *  Ensembling OFF mirrors act.py's `no_ensemble` branch (action = chunk[0]): only
 *  the FRESHLY-predicted chunk — the NEWEST vote, q = t* — is executed. */
function buildModel(m: number, K: number, tstar: number, ensembleOn: boolean): Model {
  const qLo = Math.max(0, tstar - K + 1);        // oldest chunk still covering t*
  const V = tstar - qLo + 1;                     // number of overlapping predictions
  const shares = ensembleOn
    ? ensembleWeights(m, V)
    : Array.from({ length: V }, (_, i) => (i === V - 1 ? 1 : 0)); // only the newest (q = t*)
  const votes = shares.map((share, i) => ({ q: qLo + i, i, share }));
  return { votes, qLo, oldestShare: shares[0], newestShare: shares[V - 1] };
}

// ---------------------------------------------------------------- the scene (shared)
// One <svg> body drawn from the current params — server-rendered as the poster
// (defaults) and re-rendered in place after hydration, so booting never reflows.
function Scene({ K, tstar, ensembleOn, model }: {
  K: number; tstar: number; ensembleOn: boolean; model: Model;
}) {
  const { votes, qLo } = model;
  const V = votes.length;
  const voteSet = new Set(votes.map((v) => v.q));
  const fx = colMid(tstar);
  const maxShare = Math.max(...votes.map((v) => v.share), 1e-9);
  const slot = (GX1 - GX0) / Math.max(V, 1);
  const barW = Math.min(slot * 0.56, 46);
  const barCenter = (i: number) => GX0 + (i + 0.5) * slot;
  const barTop = (share: number) => B_BASE - (share / maxShare) * B_MAXH;

  return (
    <>
      {/* ---------- panel titles ---------- */}
      <text class="ac-h" x={GX0} y={30}>action chunking — every step predicts the next K actions</text>
      <text class="ac-h" x={GX0} y={A_BOT + 52}>
        temporal ensembling at t={tstar} — {ensembleOn ? `weight = exp(−m·i), oldest (i=0) weighted most` : "ensembling OFF — execute only the newest chunk"}
      </text>

      {/* ================= PANEL A — the chunk staircase ================= */}
      {/* faint per-timestep gridlines + column headers */}
      <g class="ac-grid">
        {Array.from({ length: N + 1 }, (_, t) => (
          <line x1={colLeft(t)} y1={A_TOP - 6} x2={colLeft(t)} y2={A_BOT + 4} />
        ))}
      </g>
      <g class="ac-tick">
        {Array.from({ length: N }, (_, t) => (
          <text x={colMid(t)} y={A_TOP - 12} text-anchor="middle" data-focus={t === tstar}>{t}</text>
        ))}
        <text class="ac-axis-cap" x={colMid(N - 1)} y={A_TOP - 30} text-anchor="end">timestep →</text>
      </g>

      {/* one staggered bar per predicted chunk: chunk q covers columns q … q+K-1
          (clipped to the visible window). Chunks that cover t* are voters. */}
      <g class="ac-chunks">
        {Array.from({ length: N }, (_, q) => {
          const x = colLeft(q);
          const xEnd = colLeft(Math.min(q + K, N));
          const y = A_TOP + q * RH + (RH - RH * 0.62) / 2;
          const voter = voteSet.has(q);
          const oldest = voter && q === qLo;
          return (
            <g class="ac-chunk" data-voter={voter} data-oldest={oldest}>
              <rect x={x + 1.5} y={y} width={Math.max(xEnd - x - 3, 2)} height={RH * 0.62} rx={3} />
              <text class="ac-chunk-lbl" x={GX0 - 12} y={y + RH * 0.44} text-anchor="end">
                chunk @ t={q}
              </text>
            </g>
          );
        })}
      </g>

      {/* the focus timestep t* — a vertical read-line down the overlapping chunks */}
      <line class="ac-focus" x1={fx} y1={A_TOP - 6} x2={fx} y2={A_BOT + 4} />
      <text class="ac-focus-cap" x={fx} y={A_BOT + 20} text-anchor="middle">
        t={tstar}: {V} vote{V === 1 ? "" : "s"}
      </text>

      {/* ================= PANEL B — the ensemble weights ================= */}
      <line class="ac-baseline" x1={GX0} y1={B_BASE} x2={GX1} y2={B_BASE} />
      {/* the exp(−m·i) envelope over the bar tops (skipped when a single vote) */}
      {ensembleOn && V > 1 && (
        <polyline
          class="ac-curve"
          points={votes.map((v) => `${barCenter(v.i).toFixed(1)},${barTop(v.share).toFixed(1)}`).join(" ")}
        />
      )}
      <g class="ac-bars">
        {votes.map((v) => {
          const cx = barCenter(v.i);
          const top = barTop(v.share);
          const oldest = v.i === 0;
          const executed = !ensembleOn && v.i === V - 1; // the one action no_ensemble runs
          return (
            <g class="ac-bar" data-oldest={oldest} data-executed={executed} data-zero={v.share < 1e-9}>
              <rect x={cx - barW / 2} y={top} width={barW} height={Math.max(B_BASE - top, 0)} rx={2} />
              <text class="ac-bar-share" x={cx} y={top - 7} text-anchor="middle">{pct(v.share)}</text>
              <text class="ac-bar-i" x={cx} y={B_BASE + 18} text-anchor="middle">
                i={v.i}
              </text>
              <text class="ac-bar-q" x={cx} y={B_BASE + 33} text-anchor="middle">t={v.q}</text>
            </g>
          );
        })}
      </g>
      {V > 1 && (
        <>
          <text class="ac-end-lbl" x={barCenter(0)} y={B_TOP - 8} text-anchor="middle">oldest — most weight</text>
          <text class="ac-end-lbl ac-end-lbl--mute" x={barCenter(V - 1)} y={B_TOP - 8} text-anchor="middle">newest</text>
        </>
      )}
    </>
  );
}

// ---------------------------------------------------------------- the toy
export default function ActChunkToy() {
  const figureRef = useRef<HTMLElement>(null);
  const [booted, setBooted] = useState(false);
  const [m, setM] = useState(M_DEFAULT);
  const [K, setK] = useState(K_DEFAULT);
  const [tstar, setTstar] = useState(TSTAR_DEFAULT);
  const [ensembleOn, setEnsembleOn] = useState(true);

  // Client-only: reveal the interactive island (the SSR poster is the JS-off path).
  useEffect(() => { setBooted(true); }, []);

  // keep the focus column in range as K shrinks/grows
  const clampT = (t: number) => Math.max(0, Math.min(N - 1, t));
  const model = buildModel(m, K, clampT(tstar), ensembleOn);
  const V = model.votes.length;

  const reset = () => { setM(M_DEFAULT); setK(K_DEFAULT); setTstar(TSTAR_DEFAULT); setEnsembleOn(true); };

  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === "ArrowLeft") { e.preventDefault(); setTstar((t) => clampT(t - 1)); }
    else if (e.key === "ArrowRight") { e.preventDefault(); setTstar((t) => clampT(t + 1)); }
    else if (e.key === "ArrowUp") { e.preventDefault(); setM((v) => Math.min(M_MAX, +(v + M_STEP).toFixed(2))); }
    else if (e.key === "ArrowDown") { e.preventDefault(); setM((v) => Math.max(M_MIN, +(v - M_STEP).toFixed(2))); }
    else if (e.key === "[" || e.key === "-") { e.preventDefault(); setK((v) => Math.max(K_MIN, v - 1)); }
    else if (e.key === "]" || e.key === "+" || e.key === "=") { e.preventDefault(); setK((v) => Math.min(K_MAX, v + 1)); }
    else if (e.key === "e" || e.key === "E") { setEnsembleOn((v) => !v); }
    else if (e.key === "r" || e.key === "R") reset();
  };

  const onM = (e: Event) => setM(parseFloat((e.currentTarget as HTMLInputElement).value));
  const onT = (e: Event) => setTstar(clampT(parseInt((e.currentTarget as HTMLInputElement).value, 10)));

  // aria-live summary — the non-visual path to the same reading of the weights
  const live = K === 1
    ? `Chunk size 1: single-step policy, no chunking and nothing to ensemble. Measured no-chunk success is ${SUCCESS_NO_CHUNK.toFixed(2)}, versus ${SUCCESS_ENSEMBLE.toFixed(2)} with chunking.`
    : !ensembleOn
      ? `Temporal ensembling off: at timestep ${clampT(tstar)}, only the newest of ${V} overlapping chunk predictions is executed, so the trajectory jerks at chunk boundaries.`
      : `Ensemble coefficient m ${fmt(m)}, chunk size K ${K}. At timestep ${clampT(tstar)}, ${V} overlapping chunk predictions are averaged; the oldest prediction gets ${pct(model.oldestShare)} of the weight — the most of any vote — and the weight decays for newer predictions.`;

  const degenerate = K === 1 || !ensembleOn;

  return (
    <div class="ac">
      <figure
        ref={figureRef}
        class="ac-figure"
        data-degenerate={degenerate}
        tabIndex={booted ? 0 : -1}
        role={booted ? "application" : "img"}
        aria-label={
          booted
            ? "Interactive action-chunking and temporal-ensembling schematic. The top panel is a staircase of overlapping action chunks: the chunk predicted at each timestep covers the next K timesteps. A read-line marks the focus timestep; the chunks crossing it are the overlapping predictions. The bottom panel shows how those predictions are combined by exponential weights exp(minus m times i), where i is the chunk age and i=0 is the oldest, weighted most. Drag the ensemble-coefficient slider, or focus here and use arrow keys, to re-render the weights; left and right move the focus timestep, up and down change m; bracket keys change the chunk size K; press E to toggle temporal ensembling and R to reset."
            : "A two-panel schematic of ACT action chunking and temporal ensembling. Top: a staircase of overlapping action chunks, each covering the next K timesteps. Bottom: the exponential vote weights exp(minus m times i) that combine the chunks overlapping one timestep, with the oldest prediction weighted most. Enable JavaScript to drag the ensemble coefficient and watch the weights re-render."
        }
        onKeyDown={booted ? onKeyDown : undefined}
      >
        <svg class="ac-svg" viewBox={`0 0 ${W} ${H}`} role="img" aria-hidden={booted ? "true" : undefined}>
          {!booted && <title>ACT action chunking + temporal ensembling — schematic</title>}
          {!booted && (
            <desc>
              Each timestep predicts a chunk of the next K actions, so successive chunks overlap.
              A given timestep is covered by several chunks predicted at earlier steps; ACT combines
              them with exponential weights exp(minus m times the chunk age), weighting the oldest
              prediction most, for a smooth committed trajectory.
            </desc>
          )}
          <Scene K={K} tstar={clampT(tstar)} ensembleOn={ensembleOn} model={model} />
        </svg>

        {/* live HUD — the linked numbers (the poster ships the default readout) */}
        <div class="ac-hud" aria-hidden="true">
          <div class="ac-hud-row">
            <span class="ac-k">ensemble m</span>
            <span class="ac-v">{fmt(m)}</span>
          </div>
          <div class="ac-hud-row">
            <span class="ac-k">chunk size K</span>
            <span class={`ac-v ${K === 1 ? "ac-bad" : ""}`}>{K}{K === 1 ? " · no chunking" : ""}</span>
          </div>
          <div class="ac-hud-row">
            <span class="ac-k">overlapping votes @ t={clampT(tstar)}</span>
            <span class="ac-v">{V}</span>
          </div>
          <div class="ac-hud-row">
            <span class="ac-k">oldest-vote weight</span>
            <span class={`ac-v ${ensembleOn && K > 1 ? "ac-hot" : "ac-bad"}`}>
              {ensembleOn && K > 1 ? `${pct(model.oldestShare)} ✓` : "n/a"}
            </span>
          </div>
        </div>

        <div class="ac-status" data-degenerate={degenerate} aria-hidden="true">
          {K === 1 ? (
            <span>K=1 · single-step BC through the transformer — the no-chunk ablation</span>
          ) : !ensembleOn ? (
            <span>no_ensemble · action = chunk[0], the newest vote only — jerky at chunk seams</span>
          ) : (
            <span>exp(−m·i) · i=0 oldest = most weight · larger m commits harder to the oldest plan</span>
          )}
        </div>
      </figure>

      {/* controls — JS-only affordances (the poster reads without them) */}
      <div class="ac-controls">
        <div class="ac-slider-row">
          <label class="ac-slider-label" for="ac-m">ensemble <b>m</b></label>
          <input
            id="ac-m" class="ac-slider" type="range"
            min={M_MIN} max={M_MAX} step={M_STEP} value={m}
            onInput={onM} disabled={!booted || !ensembleOn}
            aria-label="Temporal-ensembling coefficient m. Larger m concentrates the vote weight on the oldest overlapping chunk prediction."
            aria-valuetext={`m equals ${fmt(m)}${!ensembleOn ? ", ensembling off" : ""}`}
          />
          <span class="ac-slider-val">{fmt(m)}</span>
        </div>
        <div class="ac-slider-row">
          <label class="ac-slider-label" for="ac-t">focus <b>t*</b></label>
          <input
            id="ac-t" class="ac-slider" type="range"
            min={0} max={N - 1} step={1} value={clampT(tstar)}
            onInput={onT} disabled={!booted}
            aria-label="Focus timestep on the timeline whose overlapping chunk predictions are combined."
            aria-valuetext={`timestep ${clampT(tstar)}, ${V} overlapping votes`}
          />
          <span class="ac-slider-val">t={clampT(tstar)}</span>
        </div>
        <div class="ac-btn-row">
          <span class="ac-stepper" role="group" aria-label="Chunk size K">
            <button type="button" class="ac-btn ac-btn--icon" onClick={() => setK((v) => Math.max(K_MIN, v - 1))} disabled={!booted || K <= K_MIN} aria-label="Decrease chunk size K">−</button>
            <span class="ac-stepper-val">K = {K}</span>
            <button type="button" class="ac-btn ac-btn--icon" onClick={() => setK((v) => Math.min(K_MAX, v + 1))} disabled={!booted || K >= K_MAX} aria-label="Increase chunk size K">+</button>
          </span>
          <button
            type="button"
            class={`ac-btn ${ensembleOn ? "ac-btn--primary" : "ac-btn--alert"}`}
            aria-pressed={ensembleOn}
            onClick={() => setEnsembleOn((v) => !v)}
            disabled={!booted}
          >
            {ensembleOn ? "temporal ensembling: ON" : "temporal ensembling: OFF ✗"}
          </button>
          <button type="button" class="ac-btn" onClick={reset} disabled={!booted}>reset</button>
          <span class="ac-note">drag m · t* moves the read-line · K steps the chunk · poster reads with JS off</span>
        </div>
      </div>

      {/* keyboard / screen-reader live summary — discrete, no motion needed */}
      <p class="ac-sr" aria-live="polite">{booted ? live : ""}</p>

      {/* measured caption — the honest tie to the reference run (meta.yaml) */}
      <p class="ac-cap" aria-hidden="false">
        <b>Measured</b> (meta.yaml, seed 0 · cpu · 25 eval episodes): chunking clears a bar single-step
        never reaches — <b class="ac-cap-hi">{SUCCESS_ENSEMBLE.toFixed(2)}</b> success with chunk +
        temporal ensembling vs <b class="ac-cap-lo">{SUCCESS_NO_CHUNK.toFixed(2)}</b> at K=1 (no chunk).
        That gap is the robust <em>chunking</em> lever. Temporal ensembling’s own reliable payoff is a
        smooth, committed trajectory — the weighted blend above instead of the newest vote alone.
      </p>
    </div>
  );
}
