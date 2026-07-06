/**
 * SearchOverlay — the site's client-side, keyboard-first command palette.
 *
 * WHY IT EXISTS. The site is a static Astro build with no server, so there is no
 * search backend. This island lazy-fetches a small static index
 * (/search-index.json, emitted at build time by pages/search-index.json.ts from
 * the SAME curriculum source the pages render from) and matches it with a tiny
 * CUSTOM ranker — NO new npm dependency (no pagefind / lunr / fuse).
 *
 * ── SSR-SAFE ISLAND CONTRACT (mirrors PlateIsland.tsx's frozen contract) ──────
 * 1. No window/document at module scope or in the initial render. Every browser
 *    API touch lives inside an effect or an event handler.
 * 2. Renders NOTHING until opened (`open === false` → returns null). With JS off
 *    there is simply no search; reading the chapter is unaffected (acceptable
 *    degradation — the whole book is server-rendered).
 * 3. The index is fetched ONLY on first open — never on page load.
 *
 * ── HOW IT OPENS (the public API the orchestrator wires) ──────────────────────
 * The island self-binds the global "/" (and ⌘K / Ctrl-K) shortcut once hydrated,
 * so it works with nothing more than mounting it. It ALSO exposes an imperative
 * handle for a visible affordance / the orchestrator:
 *    • window.z2rSearch.open() / .close()          — call directly
 *    • window.dispatchEvent(new Event("z2r:search-open"))  — fire an event
 * Both routes converge on the same controlled `open` state. Esc / backdrop close.
 */
import { useEffect, useMemo, useRef, useState } from "preact/hooks";
import "./search.css";

// --- the shape of /search-index.json (must match search-index.json.ts) -------
interface IndexHeading {
  text: string;
  anchor: string;
}
interface IndexSnippet {
  anchor: string;
  text: string;
}
interface IndexChapter {
  slug: string;
  title: string;
  number: string;
  phaseLabel: string;
  objectives: string[];
  headings: IndexHeading[];
  snippets: IndexSnippet[];
  regions: string[];
}

// --- one rendered result row -------------------------------------------------
interface Result {
  slug: string;
  title: string;
  number: string;
  phaseLabel: string;
  anchor: string; // "" → chapter top; else a heading slug (best-effort deep link)
  context: string; // the line shown under the title (the best sub-match)
  score: number;
}

const INDEX_URL = "/search-index.json";
const MAX_RESULTS = 12;

// Field weights — title beats heading beats objective/region beats snippet, per
// the ranking spec. Region names are precise terms, so they rank near objectives.
const W = {
  title: 12,
  region: 7,
  heading: 6,
  objective: 5,
  phase: 2.5,
  snippet: 2,
} as const;

// --- tiny custom matcher (no dependency) -------------------------------------
/** Is `q` a subsequence of `text` (loose fuzzy fallback for typos/gaps)? */
function isSubsequence(q: string, text: string): boolean {
  let i = 0;
  for (let j = 0; j < text.length && i < q.length; j++) {
    if (text[j] === q[i]) i++;
  }
  return i === q.length;
}

/**
 * Score one field's plain text against the query. Rewards exact substrings
 * (with a word-boundary bonus), the whole-query phrase, and — softly — a fuzzy
 * subsequence, so "trian" still finds "train". Returns 0 for no match.
 */
function scoreField(fullQuery: string, tokens: string[], text: string): number {
  if (!text) return 0;
  const t = text.toLowerCase();
  let s = 0;

  const phraseAt = t.indexOf(fullQuery);
  if (phraseAt >= 0) s += 2 + (phraseAt === 0 ? 1 : 0);

  for (const tok of tokens) {
    const at = t.indexOf(tok);
    if (at >= 0) {
      const boundary = at === 0 || /\W|_/.test(t[at - 1]);
      s += 1 + (boundary ? 0.6 : 0);
    } else if (tok.length >= 3 && isSubsequence(tok, t)) {
      s += 0.3;
    }
  }
  return s;
}

/** Rank the whole index for a query. Empty query → browse-all (by chapter no.). */
function runSearch(index: IndexChapter[], query: string): Result[] {
  const q = query.trim().toLowerCase();
  if (!q) {
    return index.map((ch) => ({
      slug: ch.slug,
      title: ch.title,
      number: ch.number,
      phaseLabel: ch.phaseLabel,
      anchor: "",
      context: ch.phaseLabel,
      score: 0,
    }));
  }

  const tokens = q.split(/\s+/).filter(Boolean);
  const out: Result[] = [];

  for (const ch of index) {
    let score = 0;
    // Track the single best-scoring sub-hit so the row can deep-link + show context.
    let best = { weighted: 0, anchor: "", text: "" };
    const add = (raw: number, weight: number, anchor: string, text: string) => {
      if (raw <= 0) return;
      const weighted = raw * weight;
      score += weighted;
      if (weighted > best.weighted) best = { weighted, anchor, text };
    };

    add(scoreField(q, tokens, `${ch.title} ${ch.number}`), W.title, "", ch.title);
    add(scoreField(q, tokens, ch.phaseLabel), W.phase, "", ch.phaseLabel);
    for (const h of ch.headings) add(scoreField(q, tokens, h.text), W.heading, h.anchor, h.text);
    for (const o of ch.objectives) add(scoreField(q, tokens, o), W.objective, "", o);
    for (const r of ch.regions) add(scoreField(q, tokens, r), W.region, "", `#${r} — code region`);
    for (const sn of ch.snippets) add(scoreField(q, tokens, sn.text), W.snippet, sn.anchor, sn.text);

    if (score > 0) {
      out.push({
        slug: ch.slug,
        title: ch.title,
        number: ch.number,
        phaseLabel: ch.phaseLabel,
        anchor: best.anchor,
        context: best.text || ch.headings[0]?.text || ch.phaseLabel,
        score,
      });
    }
  }

  out.sort((a, b) => b.score - a.score || a.number.localeCompare(b.number));
  return out.slice(0, MAX_RESULTS);
}

// --- match highlighting ------------------------------------------------------
function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

/** Wrap query-token substrings of `text` in <mark> nodes (case-insensitive). */
function highlight(text: string, tokens: string[]) {
  if (!tokens.length) return text;
  const re = new RegExp(`(${tokens.map(escapeRegExp).sort((a, b) => b.length - a.length).join("|")})`, "ig");
  const parts = text.split(re);
  const lowered = new Set(tokens);
  return parts.map((p, i) =>
    p && lowered.has(p.toLowerCase()) ? (
      <mark class="se-mark" key={i}>
        {p}
      </mark>
    ) : (
      p
    ),
  );
}

// -----------------------------------------------------------------------------
export default function SearchOverlay() {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const [index, setIndex] = useState<IndexChapter[] | null>(null);
  const [status, setStatus] = useState<"idle" | "loading" | "ready" | "error">("idle");

  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLUListElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const restoreFocusRef = useRef<Element | null>(null);
  const loadingRef = useRef(false);

  const tokens = useMemo(() => query.trim().toLowerCase().split(/\s+/).filter(Boolean), [query]);
  const results = useMemo(() => (index ? runSearch(index, query) : []), [index, query]);

  // Lazy-fetch the static index on FIRST open only (never on page load).
  const ensureIndex = () => {
    if (loadingRef.current || index) return;
    loadingRef.current = true;
    setStatus("loading");
    fetch(INDEX_URL)
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then((data: IndexChapter[]) => {
        setIndex(data);
        setStatus("ready");
      })
      .catch(() => setStatus("error"))
      .finally(() => {
        loadingRef.current = false;
      });
  };

  const openOverlay = () => {
    if (open) return;
    restoreFocusRef.current = document.activeElement;
    ensureIndex();
    setOpen(true);
  };
  const closeOverlay = () => setOpen(false);

  // --- global keybind + imperative API (client-only; registered post-hydration)
  useEffect(() => {
    const isEditable = (el: EventTarget | null): boolean => {
      const n = el as HTMLElement | null;
      if (!n) return false;
      const tag = n.tagName;
      return (
        tag === "INPUT" ||
        tag === "TEXTAREA" ||
        tag === "SELECT" ||
        n.isContentEditable === true
      );
    };

    const onKey = (e: KeyboardEvent) => {
      // ⌘K / Ctrl-K opens from anywhere; "/" opens unless the user is typing.
      const cmdK = (e.metaKey || e.ctrlKey) && (e.key === "k" || e.key === "K");
      const slash = e.key === "/" && !e.metaKey && !e.ctrlKey && !e.altKey && !isEditable(e.target);
      if (cmdK || slash) {
        e.preventDefault();
        openOverlay();
      }
    };
    const onOpenEvent = () => openOverlay();

    document.addEventListener("keydown", onKey);
    window.addEventListener("z2r:search-open", onOpenEvent);
    (window as any).z2rSearch = { open: openOverlay, close: closeOverlay };

    return () => {
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("z2r:search-open", onOpenEvent);
      if ((window as any).z2rSearch) delete (window as any).z2rSearch;
    };
    // openOverlay/closeOverlay are stable enough for this island's lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, index]);

  // --- open/close side effects: scroll lock, focus, focus restore --------------
  useEffect(() => {
    if (open) {
      const prevOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      // focus the input after paint so the caret lands and screen readers announce
      requestAnimationFrame(() => inputRef.current?.focus());
      return () => {
        document.body.style.overflow = prevOverflow;
      };
    }
    // On close, hand focus back to whatever had it (the affordance, usually).
    const el = restoreFocusRef.current as HTMLElement | null;
    if (el && typeof el.focus === "function") el.focus();
    return undefined;
  }, [open]);

  // Reset selection to the top whenever the result set changes.
  useEffect(() => {
    setActive(0);
  }, [query, index]);

  // Keep the active option scrolled into view.
  useEffect(() => {
    if (!open) return;
    const el = listRef.current?.querySelector<HTMLElement>(`#se-opt-${active}`);
    el?.scrollIntoView({ block: "nearest" });
  }, [active, open, results.length]);

  const goTo = (r: Result | undefined) => {
    if (!r) return;
    const href = `/${r.slug}/${r.anchor ? `#${r.anchor}` : ""}`;
    closeOverlay();
    window.location.href = href;
  };

  const onDialogKeyDown = (e: KeyboardEvent) => {
    switch (e.key) {
      case "Escape":
        e.preventDefault();
        closeOverlay();
        break;
      case "ArrowDown":
        e.preventDefault();
        if (results.length) setActive((a) => (a + 1) % results.length);
        break;
      case "ArrowUp":
        e.preventDefault();
        if (results.length) setActive((a) => (a - 1 + results.length) % results.length);
        break;
      case "Home":
        if (results.length) {
          e.preventDefault();
          setActive(0);
        }
        break;
      case "End":
        if (results.length) {
          e.preventDefault();
          setActive(results.length - 1);
        }
        break;
      case "Enter":
        e.preventDefault();
        goTo(results[active]);
        break;
      case "Tab": {
        // Minimal focus trap: cycle between the input and the close button only.
        e.preventDefault();
        const onInput = document.activeElement === inputRef.current;
        (onInput ? closeRef.current : inputRef.current)?.focus();
        break;
      }
      default:
        break;
    }
  };

  // Contract rule 2: render NOTHING until opened (SSR + JS-off safe).
  if (!open) return null;

  const activeId = results.length ? `se-opt-${active}` : undefined;
  const hasQuery = query.trim().length > 0;

  return (
    <div class="se-root" onKeyDown={onDialogKeyDown}>
      <div class="se-backdrop" onClick={closeOverlay} aria-hidden="true" />

      <div class="se-dialog" role="dialog" aria-modal="true" aria-label="Search the textbook">
        <div class="se-inputrow">
          <svg class="se-glyph" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <circle cx="11" cy="11" r="7" fill="none" stroke="currentColor" stroke-width="2" />
            <line x1="16.5" y1="16.5" x2="21" y2="21" stroke="currentColor" stroke-width="2" stroke-linecap="round" />
          </svg>
          <input
            ref={inputRef}
            class="se-input"
            type="text"
            value={query}
            placeholder="Search chapters, headings, objectives, code regions…"
            autocomplete="off"
            autocapitalize="off"
            autocorrect="off"
            spellcheck={false}
            role="combobox"
            aria-expanded={results.length > 0}
            aria-controls="se-listbox"
            aria-activedescendant={activeId}
            aria-label="Search query"
            onInput={(e) => setQuery((e.currentTarget as HTMLInputElement).value)}
          />
          <button ref={closeRef} type="button" class="se-close" onClick={closeOverlay} aria-label="Close search">
            Esc
          </button>
        </div>

        {/* Screen-reader-only live count (uses the site's existing .bk-sr helper). */}
        <div class="bk-sr" role="status" aria-live="polite">
          {status === "loading"
            ? "Loading search index…"
            : status === "error"
              ? "Search index failed to load."
              : hasQuery
                ? `${results.length} result${results.length === 1 ? "" : "s"} for ${query.trim()}`
                : `${results.length} chapter${results.length === 1 ? "" : "s"}`}
        </div>

        <ul ref={listRef} id="se-listbox" class="se-list" role="listbox" aria-label="Search results">
          {status === "loading" && <li class="se-status">Loading…</li>}
          {status === "error" && (
            <li class="se-status se-status--error">Could not load the search index. Try reloading the page.</li>
          )}
          {status !== "loading" && status !== "error" && results.length === 0 && (
            <li class="se-status">{hasQuery ? "No matches." : "Type to search."}</li>
          )}

          {results.map((r, i) => (
            <li
              id={`se-opt-${i}`}
              key={`${r.slug}#${r.anchor}`}
              class={`se-opt${i === active ? " se-opt--active" : ""}`}
              role="option"
              aria-selected={i === active}
              onMouseMove={() => setActive(i)}
              onClick={() => goTo(r)}
            >
              <span class="se-opt-num">{r.number}</span>
              <span class="se-opt-body">
                <span class="se-opt-title">{highlight(r.title, tokens)}</span>
                <span class="se-opt-context">{highlight(r.context, tokens)}</span>
              </span>
              <span class="se-opt-phase">{r.phaseLabel}</span>
            </li>
          ))}
        </ul>

        <div class="se-footer" aria-hidden="true">
          <span><kbd class="se-kbd">↑</kbd><kbd class="se-kbd">↓</kbd> navigate</span>
          <span><kbd class="se-kbd">↵</kbd> open</span>
          <span><kbd class="se-kbd">esc</kbd> close</span>
        </div>
      </div>
    </div>
  );
}
