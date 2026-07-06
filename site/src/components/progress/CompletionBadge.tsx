/**
 * CompletionBadge — a tiny per-chapter completion indicator: a progress ring
 * plus a terse readout ("read ✓ · 1 of 2 predicted"). Built for nav rows and
 * chapter cards; it's an island so it can reflect live localStorage state, but
 * it's cheap (no heavy deps) and reactive via subscribe().
 *
 * SSR / hydration contract (same as ProgressOverview): the initial render is a
 * deterministic 0% / "not started" shell with NO localStorage access; real
 * values are read in an effect after mount. Mount it `client:visible`.
 *
 * The ring fraction is chapterCompletion(); it fills solid at 100%. The label
 * text stays ink-dark for WCAG AA (the emerald is a graphic element, and only
 * the fully-complete label switches to the darker --entity-target-ink token).
 */
import { useEffect, useState } from "preact/hooks";
import { chapterCompletion, predictionsFor, isRead, subscribe } from "../../lib/progress";
import "./progress.css";

interface Props {
  slug: string;
  exerciseIds?: string[];
}

interface BadgeView {
  pct: number; // 0..1
  read: boolean;
  predicted: number;
  total: number;
}

function computeBadge(slug: string, exerciseIds: string[], live: boolean): BadgeView {
  if (!live) return { pct: 0, read: false, predicted: 0, total: exerciseIds.length };
  const committed = predictionsFor(slug);
  const predicted = exerciseIds.filter((id) => committed.includes(id)).length;
  return {
    pct: chapterCompletion(slug, exerciseIds),
    read: isRead(slug),
    predicted,
    total: exerciseIds.length,
  };
}

function labelFor(v: BadgeView): string {
  if (v.read && v.total > 0) return `read ✓ · ${v.predicted} of ${v.total} predicted`;
  if (v.read) return "read ✓";
  if (v.total > 0 && v.predicted > 0) return `${v.predicted} of ${v.total} predicted`;
  return "not started";
}

// Ring geometry: r = 7 in an 18-unit box → circumference for the dash math.
const RING_R = 7;
const RING_C = 2 * Math.PI * RING_R;

export default function CompletionBadge({ slug, exerciseIds = [] }: Props) {
  const [view, setView] = useState<BadgeView>(() => computeBadge(slug, exerciseIds, false));

  useEffect(() => {
    const update = () => setView(computeBadge(slug, exerciseIds, true));
    update();
    return subscribe(update);
    // slug/exerciseIds are stable per SSR'd instance.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [slug]);

  const pct = Math.round(view.pct * 100);
  const complete = view.pct >= 1;
  const dash = (view.pct * RING_C).toFixed(2);
  const label = labelFor(view);

  return (
    <span
      class="pg-badge"
      data-complete={complete}
      role="img"
      aria-label={`Chapter progress: ${pct}% — ${label}`}
    >
      <svg class="pg-ring" viewBox="0 0 18 18" aria-hidden="true">
        <circle class="pg-ring-track" cx="9" cy="9" r={RING_R} />
        <circle
          class="pg-ring-fill"
          cx="9"
          cy="9"
          r={RING_R}
          stroke-dasharray={`${dash} ${RING_C.toFixed(2)}`}
        />
      </svg>
      <span class="pg-badge-text" aria-hidden="true">
        {label}
      </span>
    </span>
  );
}
