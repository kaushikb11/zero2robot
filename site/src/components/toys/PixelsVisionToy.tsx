/**
 * PixelsVisionToy — ch5.3 "Control From Pixels: Visuomotor BC" concept-toy
 * (demo id `pixels_load_bearing_vision`). LOAD-BEARING VISION, made honest.
 *
 * THE HONEST HEADLINE IS A DIRECTION, NOT A ROLLOUT WIN. From pixels with no state,
 * the encoder IS the policy — so the question is whether an *aligned* encoder puts the
 * geometry a controller needs INTO its features. The REPRODUCIBLE, SEED-ROBUST, GATED
 * answer is the CONTROL-USEFULNESS PROBE: an action-regression probe on the FROZEN
 * features recovers the expert action with LOWER held-out val MSE on the aligned encoder
 * than the random one, on EVERY seed (0/1/2). That probe-gap is the hero of this toy —
 * the measured payoff of ch1.8's `--break blind` cliffhanger.
 *
 * THE ROLLOUT IS THE SCALE LAB — WE DO NOT SELL IT. Closed-loop pixel control FLOORS at
 * free-tier: 0/12 for BOTH encoders (our tiny from-scratch ViT is not SigLIP-quality).
 * The side-by-side rollout REPLAY below is recorded geometry shown to make the
 * pixels-only setup concrete; the visible difference between the two recorded traces is
 * NOT a gated claim, and neither policy moves the T. Closing the rollout gap needs a
 * pretrained aligned backbone (OpenVLA's SigLIP) — the read-the-real-thing / GPU Scale
 * Lab. The copy here says so plainly.
 *
 * WHERE THE NUMBERS COME FROM (two honest sources, deliberately):
 *   · THE PROBE HEADLINE — the GATED, reproducible seeds-0/1/2 numbers — comes from
 *     curriculum/…/ch5.3_pixels/meta.yaml `reference_run` (MEASURED 2026-07-08, cpu, at
 *     exercise_config). They are typed in as PROBE below with that provenance. We do NOT
 *     read the probe numbers from vizdata.json: that file's `meta.probe_mse_gap` is a
 *     single-config, single-seed artifact — not the seed-robust headline. Surfacing that
 *     one number as "the headline" would MISREPRESENT the chapter's reproducible claim,
 *     so the vizdata probe meta is ignored (the gated seed-0/1/2 values are used instead).
 *   · THE ROLLOUT GEOMETRY + SALIENCY — the recorded replay (aligned vs random pusher
 *     paths, the fixed T, the target) and the aligned encoder's 8×8 CLS-attention grid —
 *     come from the co-located vizdata.json (real recorded geometry, seed 0, cpu).
 *
 * NO WASM, NO ONNX, NO HEAVY IMPORT. The replay is pure inline-SVG vector geometry over a
 * static JSON recording — a LIVE in-browser pixel rollout is BLOCKED on offscreen RGB
 * rendering to featurize each 64×64×3 frame, which MuJoCo-WASM does not expose (strictly
 * more than ch1.1's state policy; documented in demo/embed.yaml). JS-off: the whole toy
 * server-renders — the probe-gap ledger (pure DOM/SVG), both recorded pusher paths, the
 * static T + target, and the saliency grid are all in the SSR poster; hydration only adds
 * the shared frame scrubber + play. Theme-aware via the flipping design tokens (light
 * fallbacks throughout); reduced-motion boots PAUSED.
 *
 * COLOUR by MEANING: aligned encoder → --signal (blue, the control-useful hero); random
 * encoder → --alert (red, the baseline that carries no control signal); the T-block →
 * --entity-block; the pusher → --entity-pusher; the target → --entity-target.
 *
 * Follows the FROZEN CONCEPT-TOY CONTRACT documented at the top of ../PlateIsland.tsx.
 */
import "./PixelsVisionToy.css";
import { useEffect, useRef, useState } from "preact/hooks";
// Real recorded rollout geometry + saliency from pixels.py (seed 0, cpu; recorded replay,
// geometry only) — see the file's `provenance`. Committed small text, no binary. NOTE: the
// probe numbers in this file's `meta` are a single-seed default-config point (probe_mse_gap
// +0.028, POSITIVE — aligned wins), NOT the seed-robust seeds-0/1/2 claim, so they are
// deliberately NOT used here — the reproducible probe headline is the reference_run below.
import vizRaw from "../../../../curriculum/phase5_practitioner/ch5.3_pixels/demo/vizdata.json";

// ------------------------------------------------------------------- data shape
type Frame = [number, number, number, number, number]; // [pusher_x, pusher_y, tee_x, tee_y, tee_yaw]
interface PvData {
  provenance: string;
  seed: number;
  world_half_extent_m: number;
  target: { x: number; y: number; yaw: number };
  tee: { bar_half: [number, number]; stem_half: [number, number]; stem_offset_y: number };
  labels: string[];
  aligned: { success: boolean; mean_return: number; frames: Frame[] };
  random: { success: boolean; mean_return: number; frames: Frame[] };
  saliency: { grid: number; note: string; weights: number[][] };
}
const DATA = vizRaw as unknown as PvData;

const WORLD_HALF = DATA.world_half_extent_m; // 0.45 m
const ALIGNED = DATA.aligned.frames as Frame[];
const RANDOM = DATA.random.frames as Frame[];
const N = Math.min(ALIGNED.length, RANDOM.length);
const BAR_HALF = DATA.tee.bar_half;
const STEM_HALF = DATA.tee.stem_half;
const STEM_OY = DATA.tee.stem_offset_y;
const PLAY_FPS = 30; // recorded frames advanced per second during playback

// =========================================================================
// THE REPRODUCIBLE PROBE HEADLINE — the GATED, seed-robust numbers.
// Source: curriculum/phase5_practitioner/ch5.3_pixels/meta.yaml `reference_run`
// (MEASURED 2026-07-08, cpu, seeds 0/1/2, at exercise_config: episodes 70, dim 96,
// depth 3, heads 3, align 30, probe 150, bc 250, eval 12). The GATED claim is the
// DIRECTION: aligned held-out val MSE < random, on EVERY seed. checks.py ex1 asserts
// probe_mse_gap ≥ MIN_ADVANTAGE. (Absolutes are platform-sensitive — MuJoCo raster is
// not bitwise across CPU arches — so we present the direction + the 3-seed band, not a %.)
// =========================================================================
const PROBE = {
  seeds: [0, 1, 2],
  // held-out action-regression val MSE (LOWER = the probe recovers the expert action better)
  aligned_mse: [0.1297, 0.1205, 0.1100],
  random_mse: [0.1578, 0.1604, 0.1635],
  gap: [0.0281, 0.0399, 0.0535], // random − aligned; > 0 == aligned features are more control-useful
  min_advantage: 0.015, // the gate ex1 must clear (measured min +0.028, so 0.015 is a safe floor)
};
// The higher, harder bar — the closed-loop pixel ROLLOUT — floors at free-tier and is
// UNGATED. Source: same reference_run. Both encoders: 0/12, every seed (a Scale Lab).
const ROLLOUT = {
  eval_episodes: 12,
  aligned_success: 0, // /12
  random_success: 0, // /12
};

// ------------------------------------------------------------- number formatting
const mse = (v: number) => v.toFixed(3);
const sgn = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(3)}`;
const allSeedsHold = PROBE.gap.every((g) => g >= PROBE.min_advantage);

// --------------------------------------------------------------- world → svg px
// Square viewBox; world (x,y) → svg px with world +y up. Same ±WORLD_HALF frame for
// every panel so the two rollouts read on one shared scale.
const PV = 300;
const PS = PV / (2 * WORLD_HALF);
const w2s = (x: number, y: number): [number, number] => [PV / 2 + x * PS, PV / 2 - y * PS];

/** A PushT "T" in SVG px at (world) center/yaw, from the recorded tee geometry. */
function Tee({ x, y, yaw, className }: { x: number; y: number; yaw: number; className: string }) {
  const [cx, cy] = w2s(x, y);
  const barW = 2 * BAR_HALF[0] * PS, barH = 2 * BAR_HALF[1] * PS;
  const stemW = 2 * STEM_HALF[0] * PS, stemH = 2 * STEM_HALF[1] * PS;
  // world +y is up (svg down), so a +yaw (CCW in world) is a −rotate in svg
  return (
    <g class={className} transform={`translate(${cx.toFixed(1)} ${cy.toFixed(1)}) rotate(${(-yaw * 180 / Math.PI).toFixed(1)})`}>
      <rect x={-barW / 2} y={-barH / 2} width={barW} height={barH} rx={2} />
      <rect x={-stemW / 2} y={-STEM_OY * PS - stemH / 2} width={stemW} height={stemH} rx={2} />
    </g>
  );
}

// ============================================================ THE PROBE HERO
/** The reproducible headline: per-seed aligned-vs-random probe val MSE (lower = more
 *  control-useful), with the gap. Pure DOM/SVG — the SSR view IS the JS-off experience. */
function ProbeLedger() {
  // shared scale for the bars: the largest val MSE across both arms, all seeds
  const maxMse = Math.max(...PROBE.aligned_mse, ...PROBE.random_mse) * 1.08;
  const barPct = (v: number) => `${Math.max(2, (v / maxMse) * 100)}%`;
  return (
    <figure class="pv-probe" role="group" aria-label="Control-usefulness probe: held-out action-regression validation MSE, aligned encoder versus random encoder, on three seeds. Lower is better.">
      <figcaption class="pv-probe-cap">
        <span class="pv-probe-title">the reproducible headline — the control-usefulness probe</span>
        <span class="pv-probe-sub">
          a frozen-feature action-regression probe recovers the expert action with{" "}
          <b>lower held-out val MSE</b> on the <b class="pv-c-align">aligned</b> encoder than the{" "}
          <b class="pv-c-rand">random</b> one — on <b>every</b> seed. Lower is better.
        </span>
      </figcaption>

      <div class="pv-probe-grid" aria-hidden="true">
        <span class="pv-pg-h" />
        <span class="pv-pg-h">aligned</span>
        <span class="pv-pg-h">random</span>
        <span class="pv-pg-h pv-pg-h--gap">gap (aligned better by)</span>
        {PROBE.seeds.map((s, i) => (
          <>
            <span class="pv-pg-seed">seed {s}</span>
            <span class="pv-pg-bar-cell">
              <span class="pv-pg-bar pv-pg-bar--align" style={`width:${barPct(PROBE.aligned_mse[i])}`} />
              <span class="pv-pg-num pv-c-align">{mse(PROBE.aligned_mse[i])}</span>
            </span>
            <span class="pv-pg-bar-cell">
              <span class="pv-pg-bar pv-pg-bar--rand" style={`width:${barPct(PROBE.random_mse[i])}`} />
              <span class="pv-pg-num pv-c-rand">{mse(PROBE.random_mse[i])}</span>
            </span>
            <span class="pv-pg-gap">{sgn(PROBE.gap[i])}</span>
          </>
        ))}
      </div>

      <p class="pv-probe-note">
        {allSeedsHold ? (
          <>
            aligned <b>&lt;</b> random val MSE on all {PROBE.seeds.length} seeds (gap{" "}
            {sgn(PROBE.gap[0])} / {sgn(PROBE.gap[1])} / {sgn(PROBE.gap[2])}), so the DIRECTION clears the{" "}
            <b>≥ {PROBE.min_advantage.toFixed(3)}</b> gate. From pixels with no state, that is the whole
            payoff of ch1.8's <code>--break blind</code>: the aligned backbone puts the controller's
            geometry <b>into</b> the features.
          </>
        ) : null}
      </p>
    </figure>
  );
}

// ========================================================= THE ROLLOUT REPLAY
/** One recorded pusher rollout as inline SVG: arena, target, the fixed T, the full
 *  pusher path (faint) + the trail up to `idx` + the moving pusher marker. The full path
 *  is drawn regardless of `idx`, so the SSR view already tells the rollout's story. */
function RolloutPanel({
  frames, idx, label, encClass, encName,
}: { frames: Frame[]; idx: number; label: string; encClass: string; encName: string }) {
  const fullPath = frames.map(([px, py]) => w2s(px, py).map((n) => n.toFixed(1)).join(",")).join(" ");
  const trail = frames.slice(0, idx + 1).map(([px, py]) => w2s(px, py).map((n) => n.toFixed(1)).join(",")).join(" ");
  const cur = frames[Math.min(idx, frames.length - 1)];
  const [pmx, pmy] = w2s(cur[0], cur[1]);
  // the T never moves in either recording — draw its (fixed) recorded pose once
  const t0 = frames[0];
  return (
    <figure class={`pv-roll ${encClass}`}>
      <figcaption class="pv-roll-cap">
        <span class="pv-roll-name">{encName}</span>
        <span class="pv-roll-tag">{label}</span>
      </figcaption>
      <svg
        class="pv-roll-svg"
        viewBox={`0 0 ${PV} ${PV}`}
        role="img"
        aria-label={
          `Top-down PushT arena — recorded pusher path for the ${encName} encoder policy, driving from pixels alone. ` +
          `The magenta T stays put and the ${encName} pusher's full recorded path is drawn; neither policy moves the T. This is recorded geometry, not live inference.`
        }
      >
        <rect class="pv-arena" x={1.5} y={1.5} width={PV - 3} height={PV - 3} rx={5} />
        <g class="pv-grid">
          {Array.from({ length: 5 }, (_, i) => ((i + 1) * PV) / 6).map((v) => (
            <>
              <line x1={v} y1={1.5} x2={v} y2={PV - 1.5} />
              <line x1={1.5} y1={v} x2={PV - 1.5} y2={v} />
            </>
          ))}
        </g>

        {/* target pose (fixed at origin) — dashed */}
        <Tee x={DATA.target.x} y={DATA.target.y} yaw={DATA.target.yaw} className="pv-target" />

        {/* the fixed T-block (its recorded pose — unchanged across the episode) */}
        <Tee x={t0[2]} y={t0[3]} yaw={t0[4]} className="pv-tee" />

        {/* the pusher's full recorded route (faint) + the trail so far (bright) */}
        <polyline class="pv-path-full" points={fullPath} />
        {idx > 0 && <polyline class={`pv-path-trail ${encClass}`} points={trail} />}

        {/* start + current pusher marker */}
        <circle class="pv-pusher-start" cx={w2s(frames[0][0], frames[0][1])[0]} cy={w2s(frames[0][0], frames[0][1])[1]} r={3} />
        <circle class={`pv-pusher-ring ${encClass}`} cx={pmx} cy={pmy} r={7} />
        <circle class={`pv-pusher-core ${encClass}`} cx={pmx} cy={pmy} r={3.5} />
      </svg>
    </figure>
  );
}

// =============================================================== THE SALIENCY
/** The aligned encoder's CLS attention over the 8×8 patch grid (last recorded frame).
 *  Pure SVG rects, opacity ∝ attention. SSR-rendered; honest, illustrative caption. */
function SaliencyGrid() {
  const w = DATA.saliency.weights;
  const g = DATA.saliency.grid;
  const flat = w.flat();
  const lo = Math.min(...flat), hi = Math.max(...flat);
  const norm = (v: number) => (hi > lo ? (v - lo) / (hi - lo) : 0);
  const cell = PV / g;
  return (
    <figure class="pv-sal">
      <figcaption class="pv-sal-cap">
        <span class="pv-sal-title">what the aligned encoder looks at</span>
        <span class="pv-sal-sub">CLS attention over the 8×8 patch grid (last recorded frame)</span>
      </figcaption>
      <svg class="pv-sal-svg" viewBox={`0 0 ${PV} ${PV}`} role="img"
        aria-label="An 8 by 8 heatmap of the aligned encoder's CLS-token attention over the image patches for the last recorded frame; brighter patches receive more attention.">
        <rect class="pv-arena" x={1.5} y={1.5} width={PV - 3} height={PV - 3} rx={5} />
        {w.map((row, r) =>
          row.map((v, c) => (
            <rect
              class="pv-sal-cell"
              x={c * cell} y={r * cell} width={cell} height={cell}
              style={`opacity:${(0.06 + 0.9 * norm(v)).toFixed(3)}`}
            />
          )),
        )}
        <g class="pv-sal-lines">
          {Array.from({ length: g - 1 }, (_, i) => (i + 1) * cell).map((p) => (
            <>
              <line x1={p} y1={0} x2={p} y2={PV} />
              <line x1={0} y1={p} x2={PV} y2={p} />
            </>
          ))}
        </g>
      </svg>
      <p class="pv-sal-note">
        The aligned encoder's attention concentrates on the block/pusher region — the geometry the
        alignment taught it to encode. Illustrative single-frame attention, not a gated metric.
      </p>
    </figure>
  );
}

// ================================================================ THE ISLAND
export default function PixelsVisionToy() {
  const [idx, setIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [booted, setBooted] = useState(false);
  const [announce, setAnnounce] = useState("");
  const reducedRef = useRef(false);

  // boot: respect reduced-motion, autoplay the shared replay otherwise
  useEffect(() => {
    reducedRef.current =
      typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
    setBooted(true);
    setPlaying(!reducedRef.current);
  }, []);

  // shared playback loop — advances the ONE frame index both panels read
  useEffect(() => {
    if (!playing || !booted) return;
    let raf = 0, last = performance.now(), acc = 0;
    const dt = 1 / PLAY_FPS;
    const tick = (now: number) => {
      acc += Math.min(now - last, 200) / 1000;
      last = now;
      while (acc >= dt) { acc -= dt; setIdx((i) => Math.min(N - 1, i + 1)); }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, booted]);

  useEffect(() => { if (idx >= N - 1 && playing) setPlaying(false); }, [idx, playing]);

  useEffect(() => {
    if (!booted) return;
    if (idx >= N - 1) {
      setAnnounce(
        `Recorded replay complete. Neither encoder moved the T — closed-loop pixel control floors at ` +
        `free-tier, ${ROLLOUT.aligned_success} of ${ROLLOUT.eval_episodes} for both. The reproducible ` +
        `result is the control-usefulness probe, where aligned features beat random on every seed.`,
      );
    } else if (playing) {
      setAnnounce(`Playing recorded replay, frame ${idx + 1} of ${N}.`);
    } else {
      setAnnounce(`Recorded replay paused at frame ${idx + 1} of ${N}.`);
    }
  }, [idx, playing, booted]);

  const togglePlay = () => {
    if (idx >= N - 1) { setIdx(0); setPlaying(true); return; }
    setPlaying((p) => !p);
  };
  const onScrub = (e: Event) => {
    setPlaying(false);
    setIdx(parseInt((e.currentTarget as HTMLInputElement).value, 10));
  };
  const onKeyDown = (e: KeyboardEvent) => {
    if (e.key === " " || e.key === "Spacebar") { e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); setPlaying(false); setIdx((i) => Math.min(N - 1, i + 1)); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); setPlaying(false); setIdx((i) => Math.max(0, i - 1)); }
    else if (e.key === "Home") { e.preventDefault(); setPlaying(false); setIdx(0); }
    else if (e.key === "End") { e.preventDefault(); setPlaying(false); setIdx(N - 1); }
  };

  return (
    <div class="pv">
      <header class="pv-head">
        <h3 class="pv-title">Control from pixels — the encoder IS the policy</h3>
        <p class="pv-sub">
          With the state removed, the same tiny ViT — <b class="pv-c-align">aligned</b> vs{" "}
          <b class="pv-c-rand">random</b> — is the whole policy. The <b>reproducible</b> signal is the{" "}
          <b>control-usefulness probe</b> below; the closed-loop rollout is the <b>Scale Lab</b>.
        </p>
      </header>

      {/* THE HERO — the reproducible, gated probe-gap */}
      <ProbeLedger />

      {/* THE ROLLOUT REPLAY — recorded geometry, honestly framed (both floor) */}
      <section class="pv-rollouts" aria-label="Recorded side-by-side pixel rollout, aligned versus random encoder.">
        <div class="pv-rollouts-head">
          <span class="pv-rollouts-title">the pixels-only rollout — recorded replay</span>
          <span class="pv-badge">recorded geometry · not live inference</span>
        </div>

        <figure
          class="pv-stage"
          tabIndex={0}
          role="application"
          aria-label="Recorded side-by-side PushT rollout from pixels alone: aligned encoder on the left, random on the right. Recorded geometry, not live inference. Press space to play or pause, arrow keys to scrub, Home and End to jump."
          onKeyDown={onKeyDown}
        >
          <div class="pv-roll-pair">
            <RolloutPanel frames={ALIGNED} idx={idx} encName="aligned" encClass="pv-e-align"
              label={`floored ${ROLLOUT.aligned_success}/${ROLLOUT.eval_episodes}`} />
            <RolloutPanel frames={RANDOM} idx={idx} encName="random" encClass="pv-e-rand"
              label={`floored ${ROLLOUT.random_success}/${ROLLOUT.eval_episodes}`} />
          </div>
          <div class="pv-legend" aria-hidden="true">
            <span class="pv-lg pv-lg-tee">T-block (fixed)</span>
            <span class="pv-lg pv-lg-align">aligned pusher</span>
            <span class="pv-lg pv-lg-rand">random pusher</span>
            <span class="pv-lg pv-lg-target">target</span>
          </div>
        </figure>

        {/* the ONE control — shared play/pause + frame scrubber (keyboard-native) */}
        <div class="pv-controls">
          <button type="button" class="pv-btn pv-btn--primary" onClick={togglePlay} disabled={!booted}>
            {idx >= N - 1 ? "↻ replay" : playing ? "❚❚ pause" : "► play"}
          </button>
          <input
            class="pv-scrub"
            type="range"
            min={0}
            max={N - 1}
            step={1}
            value={idx}
            onInput={onScrub}
            disabled={!booted}
            aria-label="Scrub both recorded rollouts frame by frame."
            aria-valuetext={`Frame ${idx + 1} of ${N}`}
          />
          <span class="pv-frame-out" aria-hidden="true">{idx + 1}/{N}</span>
        </div>

        <p class="pv-roll-note" aria-hidden="true">
          <b>Both policies fail:</b> at free-tier the closed-loop pixel rollout floors —{" "}
          <b>{ROLLOUT.aligned_success}/{ROLLOUT.eval_episodes}</b> for aligned and{" "}
          <b>{ROLLOUT.random_success}/{ROLLOUT.eval_episodes}</b> for random — neither moves the T. Our
          tiny from-scratch ViT is not SigLIP-quality, so this rollout is <b>the Scale Lab</b>, not a win.
          The visible difference between the two recorded traces is one seed's geometry, <b>not</b> a
          reproducible claim — the gated result is the probe above. A <b>live</b> in-browser pixel rollout
          is blocked: it needs offscreen RGB rendering to featurize each 64×64×3 frame, which MuJoCo-WASM
          does not expose.
        </p>
      </section>

      {/* OPTIONAL — the aligned encoder's saliency */}
      <SaliencyGrid />

      {/* non-visual path to the same aha */}
      <div class="bk-sr" aria-live="polite">{announce}</div>

      <p class="pv-foot" aria-hidden="true">
        probe headline: MEASURED cpu, seeds {PROBE.seeds.join("/")}, exercise_config (from
        meta.yaml <code>reference_run</code>) — aligned &lt; random val MSE every seed, gated at{" "}
        <b>≥ {PROBE.min_advantage.toFixed(3)}</b> · rollout geometry + saliency: recorded from pixels.py
        (seed {DATA.seed}, cpu) · rollout floors 0/{ROLLOUT.eval_episodes} both = Scale Lab · poster reads
        with JS off
      </p>
    </div>
  );
}

// ============================================================================
// WIRE ME IN — PlateIsland.tsx (the shared dispatch; do NOT edit here):
//   import PixelsVisionToy from "./toys/PixelsVisionToy";              // with the other toy imports
//   if (demo === "pixels_load_bearing_vision") return <PixelsVisionToy />;  // ch5.3 load-bearing vision: probe-gap + recorded pixel rollout
// (No lazy/heavy deps: this toy replays a static-JSON recording — no WASM, no onnx — so it
//  is safe to import eagerly like the other data toys and fully server-renders the probe
//  ledger + both recorded pusher paths + the saliency grid for the JS-off path.)
// ============================================================================
