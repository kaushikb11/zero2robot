/**
 * ProgressOverview — the landing-page progress dashboard island.
 *
 * Renders an overall completion %, a per-phase completion bar for each phase,
 * and a "continue where you left off →" resume link. Everything is derived from
 * localStorage via ../../lib/progress; the chapter LIST arrives as a prop because
 * curriculum discovery is build-time (Node) and this island is client-side.
 *
 * SSR / hydration contract (see PlateIsland.tsx "FROZEN CONCEPT-TOY CONTRACT"):
 *   · No window/localStorage at module scope or in the INITIAL render. The first
 *     render (server + client-first paint) uses computeView(chapters, /*live*​/ false),
 *     a pure 0%/start-here shell — so the hydrated DOM matches the SSR'd DOM.
 *   · Real values are read in an effect (post-hydration) and pushed via setState,
 *     then kept live through subscribe().
 *   · aria-live on the overall %; the phase bars carry text + an aria-label.
 *   · Focus-visible + reduced-motion are handled in progress.css.
 */
import { useEffect, useState } from "preact/hooks";
import {
  chapterCompletion,
  overallCompletion,
  getProgress,
  lastVisited,
  subscribe,
} from "../../lib/progress";
import "./progress.css";

export interface ProgressChapter {
  slug: string;
  title: string;
  number: string;
  phaseKey: string;
  phaseLabel: string;
  exerciseIds?: string[];
}

interface Props {
  chapters: ProgressChapter[];
}

interface PhaseView {
  phaseKey: string;
  phaseLabel: string;
  pct: number; // 0..1
  done: number; // chapters at 100%
  total: number;
}

interface View {
  overall: number; // 0..1
  phases: PhaseView[];
  anyProgress: boolean;
  continueSlug: string | null;
  continueTitle: string | null;
  continueIsStart: boolean; // nothing done yet → "start the first chapter"
}

/** Pure view model. `live` gates every localStorage read so the initial
 *  (SSR + first-paint) render is a deterministic 0%/start-here shell. */
function computeView(chapters: ProgressChapter[], live: boolean): View {
  const prog = live ? getProgress() : {};

  const perChapter = new Map<string, number>();
  for (const ch of chapters) {
    perChapter.set(ch.slug, live ? chapterCompletion(ch.slug, ch.exerciseIds ?? []) : 0);
  }

  // phases in first-seen (curriculum) order
  const phases: PhaseView[] = [];
  for (const ch of chapters) {
    if (!phases.some((p) => p.phaseKey === ch.phaseKey)) {
      phases.push({ phaseKey: ch.phaseKey, phaseLabel: ch.phaseLabel, pct: 0, done: 0, total: 0 });
    }
  }
  for (const p of phases) {
    const inPhase = chapters.filter((c) => c.phaseKey === p.phaseKey);
    const sum = inPhase.reduce((acc, c) => acc + (perChapter.get(c.slug) ?? 0), 0);
    p.total = inPhase.length;
    p.pct = inPhase.length ? sum / inPhase.length : 0;
    p.done = inPhase.filter((c) => (perChapter.get(c.slug) ?? 0) >= 1).length;
  }

  const overall = live ? overallCompletion(chapters) : 0;
  const anyProgress = live && [...perChapter.values()].some((v) => v > 0);

  // resume target: last visited (if still a real chapter) → first unread → first.
  let continueSlug: string | null = null;
  if (live) {
    const last = lastVisited();
    if (last && chapters.some((c) => c.slug === last)) continueSlug = last;
  }
  let continueIsStart = false;
  if (!continueSlug) {
    const firstUnread = chapters.find((c) => !prog[c.slug]?.read);
    continueSlug = (firstUnread ?? chapters[0])?.slug ?? null;
    continueIsStart = !anyProgress;
  }
  const continueTitle = chapters.find((c) => c.slug === continueSlug)?.title ?? null;

  return { overall, phases, anyProgress, continueSlug, continueTitle, continueIsStart };
}

export default function ProgressOverview({ chapters }: Props) {
  // Initial render (SSR + first client paint) = the static 0% shell.
  const [view, setView] = useState<View>(() => computeView(chapters, false));

  useEffect(() => {
    const update = () => setView(computeView(chapters, true));
    update(); // read the real localStorage values now that we're mounted
    return subscribe(update); // stay live across reads / other tabs
    // chapters is SSR'd once per page load and stable for the island's life.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const pct = Math.round(view.overall * 100);

  // Self-hide until there is REAL progress. The first render (SSR + first client
  // paint) is the live=false shell, so anyProgress is false and this returns null
  // on both server and client-first-paint (no hydration mismatch). A newcomer
  // therefore never sees an empty 0% widget on the landing; the moment they read
  // a chapter or commit a prediction, the effect re-renders this with content.
  if (!view.anyProgress) return null;

  return (
    <section class="pg-overview pg-overview--landing" aria-label="Your progress">
      <div class="pg-top">
        <div class="pg-overall">
          <span class="pg-overall-label">overall</span>
          {/* announce the headline number as it changes */}
          <span class="pg-overall-num" aria-live="polite">
            {pct}%
          </span>
        </div>

        {view.continueSlug && (
          <a class="pg-continue" href={`/${view.continueSlug}/`}>
            {view.continueIsStart ? "start the first chapter" : "continue where you left off"}
            {view.continueTitle && <span class="pg-continue-title"> · {view.continueTitle}</span>}
            <span aria-hidden="true"> →</span>
          </a>
        )}
      </div>

      <div
        class="pg-overall-bar"
        role="img"
        aria-label={`Overall: ${pct} percent complete`}
      >
        <div class="pg-bar-fill" style={`width:${pct}%`} />
      </div>

      <ul class="pg-phases">
        {view.phases.map((p) => {
          const ppct = Math.round(p.pct * 100);
          return (
            <li class="pg-phase" key={p.phaseKey}>
              <div class="pg-phase-head">
                <span class="pg-phase-label">{p.phaseLabel}</span>
                <span class="pg-phase-pct">
                  {p.done}/{p.total} · {ppct}%
                </span>
              </div>
              <div
                class="pg-bar"
                role="img"
                aria-label={`${p.phaseLabel}: ${ppct} percent complete, ${p.done} of ${p.total} chapters done`}
              >
                <div class="pg-bar-fill" style={`width:${ppct}%`} />
              </div>
            </li>
          );
        })}
      </ul>
    </section>
  );
}
