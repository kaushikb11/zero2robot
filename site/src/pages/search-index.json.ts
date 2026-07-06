// Build-time SEARCH INDEX endpoint. Astro runs this GET at SSG time (Node, never
// the browser) and writes the returned JSON to dist/search-index.json — a static
// asset served at `/search-index.json`. The client search overlay
// (components/search/SearchOverlay.tsx) lazy-fetches it on first open.
//
// SINGLE SOURCE OF TRUTH. The index is built from the SAME curriculum bridge the
// chapter pages render from (lib/curriculum.ts): discoverChapters() for the map,
// readProse() for the exact draft markdown, extractRegions() for the real region
// names. So the index can never drift from the pages — rebuild the site and the
// index is regenerated from identical bytes. No hand-maintained list, ever.
//
// KEEP IT LEAN. Snippets are short (~1 sentence, plain text). This is fetched
// once per session over the wire, so every field earns its place.
import type { APIRoute } from "astro";
import { discoverChapters, readProse, extractRegions } from "../lib/curriculum.ts";

// Prerendered like every other page (output: 'static'); stated for the record so
// it is unmistakably a build-time asset, not a runtime handler.
export const prerender = true;

// --- one search record per chapter ------------------------------------------
interface IndexHeading {
  text: string; // the heading text, plain
  anchor: string; // github-style slug — a best-effort deep-link fragment
}
interface IndexSnippet {
  anchor: string; // the heading the excerpt sits under (deep-link target)
  text: string; // short plain-text excerpt (markdown/code stripped)
}
interface IndexChapter {
  slug: string;
  title: string;
  number: string;
  phaseLabel: string;
  objectives: string[];
  headings: IndexHeading[];
  snippets: IndexSnippet[];
  regions: string[]; // region names from the artifact (e.g. "model", "train")
}

// --- markdown → plain text (mirrors the intent of prose.ts's stripping) ------
// prose.ts renders the draft to HTML for display; here we only need clean, short
// text to search over, so we strip structure rather than render it.
const LEADING_COMMENT_RE = /^\s*<!--[\s\S]*?-->\s*/; // top-of-file build note
const HTML_COMMENT_RE = /<!--[\s\S]*?-->/g; // author notes anywhere
const FENCE_RE = /```[\s\S]*?```/g; // fenced code + include directives
const LEADING_H1_RE = /^\s*#\s+[^\n]*\n/; // the "# N.M: Title" the header owns

/** One prose fragment → single-line plain text: no code, no markup, collapsed. */
function toPlainText(md: string): string {
  return md
    .replace(FENCE_RE, " ")
    .replace(/`([^`]+)`/g, "$1") // inline code → its text
    .replace(/!\[[^\]]*\]\([^)]*\)/g, " ") // images → gone
    .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1") // links → link text
    .replace(/^[>\-*+]\s+/gm, "") // list / quote markers
    .replace(/[*_]{1,3}([^*_]+)[*_]{1,3}/g, "$1") // emphasis markers
    .replace(/\s+/g, " ")
    .trim();
}

/** GitHub-style heading slug (a best-effort deep-link fragment). */
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}

/** Trim to ~1 sentence / N chars at a word boundary, so snippets stay small. */
function clip(text: string, max = 160): string {
  if (text.length <= max) return text;
  const cut = text.slice(0, max);
  const lastSpace = cut.lastIndexOf(" ");
  return (lastSpace > max * 0.6 ? cut.slice(0, lastSpace) : cut).trimEnd() + "…";
}

/** Read a chapter's draft prose, tolerating a missing draft (fall back to .md). */
function readChapterProse(chapterDir: string): string {
  for (const rel of [`${chapterDir}/prose/chapter.draft.md`, `${chapterDir}/prose/chapter.md`]) {
    try {
      return readProse(rel);
    } catch {
      /* try the next candidate */
    }
  }
  return "";
}

/** Parse ## / ### headings + a short excerpt under each, from draft markdown. */
function extractProse(markdown: string): { headings: IndexHeading[]; snippets: IndexSnippet[] } {
  // Strip the build-note block, the title H1, then all comments — same order as
  // prose.ts so headings/snippets never include internal author notes.
  let src = markdown.replace(LEADING_COMMENT_RE, "");
  src = src.replace(LEADING_H1_RE, "");
  src = src.replace(HTML_COMMENT_RE, "");

  const headings: IndexHeading[] = [];
  const snippets: IndexSnippet[] = [];

  // Walk the markdown by lines, tracking the current ## / ### section so the body
  // between headings becomes that heading's excerpt.
  const lines = src.split("\n");
  let current: IndexHeading | null = null;
  let buffer: string[] = [];

  const flush = () => {
    if (!current) return;
    const text = clip(toPlainText(buffer.join("\n")));
    if (text) snippets.push({ anchor: current.anchor, text });
    buffer = [];
  };

  // De-dup anchors in document order, identically to prose.ts addHeadingIds, so
  // a #anchor deep-link resolves to the right <h2>/<h3>.
  const seen = new Map<string, number>();
  const uniqAnchor = (base: string): string => {
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    return n === 0 ? base : `${base}-${n + 1}`;
  };

  for (const line of lines) {
    const m = line.match(/^(#{2,3})\s+(.*\S)\s*$/);
    if (m) {
      flush();
      const text = toPlainText(m[2]);
      const base = slugify(text);
      // "See it work" is stripped from the rendered lede (prose.ts) and has no
      // anchor on the page — don't index it as a dead deep-link target.
      if (base === "see-it-work") {
        current = null;
        continue;
      }
      current = { text, anchor: uniqAnchor(base) };
      headings.push(current);
    } else if (current) {
      buffer.push(line);
    }
  }
  flush();

  return { headings, snippets };
}

export const GET: APIRoute = () => {
  const chapters = discoverChapters();

  const index: IndexChapter[] = chapters.map((ch) => {
    const { headings, snippets } = extractProse(readChapterProse(ch.chapterDir));

    // Region names come from the SAME extractor the code panels use, so the
    // searchable region list is byte-honest with what the page shows.
    let regions: string[] = [];
    try {
      regions = extractRegions(ch.artifactPath).map((r) => r.name);
    } catch {
      /* a chapter without a parseable artifact simply has no region terms */
    }

    return {
      slug: ch.slug,
      title: ch.title,
      number: ch.number,
      phaseLabel: ch.phaseLabel,
      objectives: ch.objectives ?? [],
      headings,
      snippets,
      regions,
    };
  });

  // Compact (no pretty-print) — this ships to every searcher.
  return new Response(JSON.stringify(index), {
    headers: { "content-type": "application/json; charset=utf-8" },
  });
};
