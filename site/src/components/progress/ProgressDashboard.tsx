/**
 * ProgressDashboard — the /progress page's full breakdown island.
 *
 * A superset of the landing's ProgressOverview: overall completion, a resume
 * link, AND a per-phase → per-chapter tree where each chapter carries its live
 * CompletionBadge (read ✓ · N-of-M predicted). Everything is derived from
 * localStorage via ../../lib/progress; the chapter LIST arrives as a prop because
 * curriculum discovery is build-time (Node) and this island is client-side.
 *
 * SSR / hydration contract (see PlateIsland.tsx "FROZEN CONCEPT-TOY CONTRACT",
 * and mirrored from ProgressOverview):
 *   · No window/localStorage at module scope or in the INITIAL render. The first
 *     render (server + client-first paint) uses computeView(chapters, false) — a
 *     pure 0% / start-here shell — so the hydrated DOM matches the SSR'd DOM.
 *   · Real values are read in an effect (post-hydration) and pushed via setState,
 *     then kept live through subscribe().
 *   · CompletionBadge (reused verbatim) keeps the same contract per chapter row.
 *   · aria-live on the overall %; every bar carries text + an aria-label.
 *
 * HONESTY: completion counts ONLY real signals — a chapter actually read, and
 * gated (predict-then-run) predictions actually committed. exerciseIds are the
 * GATED ids only (the sole thing that can leave a localStorage trace), passed in
 * from progress.astro exactly as the landing does — so a ring reaches 100%
 * legitimately (read a chapter with no gated exercises, or read + answer all of
 * them) and never by merely opening a page.
 */
import { useEffect, useState } from "preact/hooks";
import {
  chapterCompletion,
  overallCompletion,
  getProgress,
  isRead,
  predictionsFor,
  lastVisited,
  subscribe,
} from "../../lib/progress";
import CompletionBadge from "./CompletionBadge";
import "./progress.css";

export interface DashboardChapter {
  slug: string;
  title: string;
  number: string;
  phaseKey: string;
  phaseLabel: string;
  exerciseIds?: string[];
}

interface Props {
  chapters: DashboardChapter[];
}

interface ChapterView {
  slug: string;
  title: string;
  number: string;
  exerciseIds: string[];
  read: boolean;
  predicted: number;
  total: number;
  pct: number; // 0..1
}

interface PhaseView {
  phaseKey: string;
  phaseLabel: string;
  pct: number; // 0..1
  done: number; // chapters at 100%
  total: number;
  chapters: ChapterView[];
}

interface View {
  overall: number; // 0..1
  phases: PhaseView[];
  anyProgress: boolean;
  continueSlug: string | null;
  continueTitle: string | null;
  continueIsStart: boolean;
  readCount: number;
  totalCount: number;
}

/** Pure view model. `live` gates every localStorage read so the initial
 *  (SSR + first-paint) render is a deterministic 0% / start-here shell. */
function computeView(chapters: DashboardChapter[], live: boolean): View {
  const prog = live ? getProgress() : {};

  // per-chapter, from REAL signals only
  const chViews = new Map<string, ChapterView>();
  for (const ch of chapters) {
    const ids = ch.exerciseIds ?? [];
    const committed = live ? predictionsFor(ch.slug) : [];
    const predicted = ids.filter((id) => committed.includes(id)).length;
    chViews.set(ch.slug, {
      slug: ch.slug,
      title: ch.title,
      number: ch.number,
      exerciseIds: ids,
      read: live ? isRead(ch.slug) : false,
      predicted,
      total: ids.length,
      pct: live ? chapterCompletion(ch.slug, ids) : 0,
    });
  }

  // phases in first-seen (curriculum) order, each carrying its chapter rows
  const phases: PhaseView[] = [];
  for (const ch of chapters) {
    let p = phases.find((x) => x.phaseKey === ch.phaseKey);
    if (!p) {
      p = {
        phaseKey: ch.phaseKey,
        phaseLabel: ch.phaseLabel,
        pct: 0,
        done: 0,
        total: 0,
        chapters: [],
      };
      phases.push(p);
    }
    p.chapters.push(chViews.get(ch.slug)!);
  }
  for (const p of phases) {
    const sum = p.chapters.reduce((acc, c) => acc + c.pct, 0);
    p.total = p.chapters.length;
    p.pct = p.total ? sum / p.total : 0;
    p.done = p.chapters.filter((c) => c.pct >= 1).length;
  }

  const overall = live ? overallCompletion(chapters) : 0;
  const anyProgress = live && [...chViews.values()].some((v) => v.pct > 0);
  const readCount = [...chViews.values()].filter((v) => v.read).length;

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

  return {
    overall,
    phases,
    anyProgress,
    continueSlug,
    continueTitle,
    continueIsStart,
    readCount,
    totalCount: chapters.length,
  };
}

export default function ProgressDashboard({ chapters }: Props) {
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

  return (
    <section class="pg-dash" aria-label="Your progress dashboard">
      <div class="pg-top">
        <div class="pg-overall">
          <span class="pg-overall-label">overall</span>
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

      <div class="pg-overall-bar" role="img" aria-label={`Overall: ${pct} percent complete`}>
        <div class="pg-bar-fill" style={`width:${pct}%`} />
      </div>

      <p class="pg-dash-sub">
        {view.readCount} of {view.totalCount} chapters read. Completion counts only
        what you actually did — a chapter you read, and predictions you committed.
      </p>

      {!view.anyProgress && (
        <p class="pg-empty">
          Nothing completed yet — read a chapter, or commit a prediction in an exercise,
          and it shows up here. No account needed; this lives only in your browser.
        </p>
      )}

      <div class="pg-dash-phases">
        {view.phases.map((p) => {
          const ppct = Math.round(p.pct * 100);
          return (
            <section class="pg-dash-phase" key={p.phaseKey} aria-label={p.phaseLabel}>
              <div class="pg-phase-head">
                <h2 class="pg-dash-phase-label">{p.phaseLabel}</h2>
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

              <ul class="pg-chapters">
                {p.chapters.map((c) => (
                  <li class="pg-chapter" key={c.slug} data-complete={c.pct >= 1}>
                    <a class="pg-chapter-link" href={`/${c.slug}/`}>
                      <span class="pg-chapter-num">{c.number}</span>
                      <span class="pg-chapter-title">{c.title}</span>
                    </a>
                    {/* reused verbatim — its own SSR-safe effect reads live state */}
                    <CompletionBadge slug={c.slug} exerciseIds={c.exerciseIds} />
                  </li>
                ))}
              </ul>
            </section>
          );
        })}
      </div>
    </section>
  );
}
