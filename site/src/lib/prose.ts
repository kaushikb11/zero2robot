// Prose → HTML for a chapter page. This is the CONTENT-EXTRACTION layer: it
// turns the curriculum draft markdown into readable HTML and injects the EXACT
// code regions (real bytes, via extractRegions/the bridge) at each
// `[include-by-region: <artifact>#<region>]` directive, plus the CSV-backed
// wall-clock ledger at its placeholder. The Variant C design skin ("read like a
// book, code like a lab") restyles the OUTPUT via CSS classes + custom
// properties in public/styles.css and never calls in here.
//
// Everything renders server-side (SSG) into static HTML, so a chapter's prose,
// every code panel, and the wall-clock ledger survive with JavaScript disabled.
//
// PIPELINE NOTE (why placeholder tokens): code panels are injected via
// PLACEHOLDER TOKENS, so `marked` only ever parses prose — never the code.
// Injecting code HTML into the markdown would let a blank line *inside* a region
// close marked's HTML block and turn the code's `#` comments into headings. We
// render marked first, then substitute the real dark panels into the HTML.
//
// ENTITY TINTING IS DISPLAY-ONLY. Region bytes are hashed upstream
// (region.sha256, from the same subprocess the drift gate uses); tinting only
// wraps <span>s around already-escaped display text, so the drift contract is
// never touched. It is applied only for PushT chapters (see `entities` flag);
// where a chapter's entities don't map cleanly, the tokens stay defined in CSS
// but NO tinting is applied — never wrong tinting.

import { marked } from "marked";
import katex from "katex";
import { Buffer } from "node:buffer";
import type { Region, WallclockRow } from "./curriculum.ts";

/** Escape a raw code region / prose fragment for safe embedding. */
function escapeHtml(s: string): string {
  return s
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

// --- heading anchors --------------------------------------------------------
// Give every rendered <h2>/<h3> a stable id so in-page + search deep-links
// (#anchor) land on the section. The slug MUST match search-index.json.ts's
// slugify() (built from the same raw heading text) so search fragments resolve.
function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^\w\s-]/g, "")
    .trim()
    .replace(/\s+/g, "-");
}
function decodeBasicEntities(s: string): string {
  return s
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}
/** Add id="slug-of-text" to each <h2>/<h3> that lacks one. De-dups collisions
 *  (append -2, -3, …) in document order so ids stay unique (valid HTML) — the
 *  same order search-index.json.ts sees, keeping deep-link anchors in parity.
 *  `seen` is shared across the lede + body passes so an id can't collide between
 *  them (they render into the same page). */
function addHeadingIds(html: string, seen: Map<string, number>): string {
  return html.replace(/<(h[23])>([\s\S]*?)<\/\1>/g, (m, tag: string, inner: string) => {
    const text = decodeBasicEntities(inner.replace(/<[^>]+>/g, ""));
    const base = slugify(text);
    if (!base) return m;
    const n = seen.get(base) ?? 0;
    seen.set(base, n + 1);
    const id = n === 0 ? base : `${base}-${n + 1}`;
    return `<${tag} id="${id}">${inner}</${tag}>`;
  });
}

// --- entity linking (display-only; PushT chapters) --------------------------
// Identifiers inside CODE panels, tinted the same three hues as the plate and
// prose. Longest tokens first so `train_actions` wins over `action`.
const CODE_TOKENS = new Map<string, "block" | "pusher" | "target">();
(
  [
    ["block", ["block_start_distance", "tee_x", "tee_y"]],
    ["pusher", ["train_actions", "val_actions", "act_range", "act_min", "actions", "action"]],
    ["target", ["success_rate", "successes", "success", "pos_err"]],
  ] as const
).forEach(([cls, toks]) => toks.forEach((t) => CODE_TOKENS.set(t, cls)));
const CODE_RE = new RegExp(
  "\\b(" + [...CODE_TOKENS.keys()].sort((a, b) => b.length - a.length).join("|") + ")\\b",
  "g",
);

/** One code line → escaped HTML, comments dimmed, identifiers tinted (if on). */
function renderCodeLine(line: string, entities: boolean): string {
  // Whole comment lines stay dim; never tint identifiers inside a comment.
  if (/^\s*#/.test(line)) return `<span class="bk-cmt">${escapeHtml(line)}</span>`;
  const esc = escapeHtml(line);
  if (!entities) return esc;
  return esc.replace(CODE_RE, (m) => `<span class="bk-e-${CODE_TOKENS.get(m)}">${m}</span>`);
}

/**
 * The canonical code-panel markup: a crisp DARK lab instrument. One place
 * defines the panel DOM so prose-injected panels and the standalone component
 * stay identical and skinnable.
 *
 *   figure.bk-panel[data-region][data-code]  the whole panel (data-code: base64
 *                                             of the exact region bytes, for the
 *                                             JS-only copy button)
 *     .bk-panel-bar   .bk-panel-file          artifact basename (e.g. "bc.py")
 *                     .bk-panel-region        "#model"
 *                     .bk-panel-hash          "sha256:abc123…" (drift provenance)
 *                     .bk-copy                copy button (JS-only; hidden JS-off)
 *     pre > code > .bk-cl > .bk-cx            one span per line (numbers via CSS)
 */
export function codePanelHtml(
  artifactBasename: string,
  region: Region,
  entities = false,
): string {
  const lines = region.text.replace(/\n$/, "").split("\n");
  const body = lines
    .map(
      (l) =>
        `<span class="bk-cl"><span class="bk-cx">${renderCodeLine(l, entities) || "&nbsp;"}</span></span>`,
    )
    .join("");
  const b64 = Buffer.from(region.text, "utf-8").toString("base64");
  return (
    `<figure class="bk-panel" data-region="${escapeHtml(region.name)}" data-code="${b64}">` +
    `<div class="bk-panel-bar">` +
    `<span class="bk-panel-file">${escapeHtml(artifactBasename)}</span>` +
    `<span class="bk-panel-region"><span class="bk-hash-sym">#</span>${escapeHtml(region.name)}</span>` +
    `<span class="bk-panel-hash" title="sha256 of the included region — re-checked by check_prose_code_drift on every PR">` +
    `sha256:${region.sha256.slice(0, 10)}…</span>` +
    `<button class="bk-copy" type="button" aria-label="Copy the ${escapeHtml(region.name)} region">copy</button>` +
    `</div><pre><code>${body}</code></pre></figure>`
  );
}

function missingPanelHtml(name: string): string {
  return (
    `<figure class="bk-panel bk-panel--missing" data-region="${escapeHtml(name)}">` +
    `<div class="bk-panel-bar">missing region: ${escapeHtml(name)}</div></figure>`
  );
}

/**
 * The canonical wall-clock ledger markup — an honest instrument. Only MEASURED
 * tiers are shown (the rendered line + a "measured" pill); a tier with no
 * measured number for this chapter is simply omitted, not shown as "pending".
 * NEVER an estimate. Shared by WallclockTable.astro and the prose injection
 * below, so there is one source of markup.
 */
// Display-only tier labels: the ledger key stays "cpu-laptop" (so the wall-clock
// gate and provenance are untouched), but the table shows the shorter "cpu".
const TIER_LABEL: Record<string, string> = { "cpu-laptop": "cpu" };

export function wallclockTableHtml(rows: WallclockRow[]): string {
  const body = rows
    .filter((w) => w.minutes !== null)
    .map((w) => {
      const measured = w.minutes !== null;
      return (
        `<tr data-tier="${escapeHtml(w.tier)}" data-measured="${measured}">` +
        `<td class="bk-tier">${escapeHtml(TIER_LABEL[w.tier] ?? w.tier)}</td>` +
        `<td class="bk-line">${escapeHtml(w.line)}</td>` +
        `<td><span class="bk-pill" data-measured="${measured}">${measured ? "measured" : "pending"}</span></td>` +
        `</tr>`
      );
    })
    .join("");
  return (
    `<div class="bk-wall" role="table" aria-label="Measured wall-clock by tier">` +
    `<div class="bk-wall-head">` +
    `<span class="bk-wall-eyebrow">wall-clock · rendered from wallclock.csv</span>` +
    `<span class="bk-wall-eyebrow">one source · measured tiers</span>` +
    `</div><table><tbody>${body}</tbody></table></div>`
  );
}

// --- prose-word entity linking (display-only; PushT chapters) ---------------
// Same three hues, a quiet colored underline. Tag-aware: never touches text
// inside <pre>, <figure>, <code>, links, or headings, so code panels and inline
// code keep their own treatment. Runs on the prose HTML BEFORE panels/wall are
// substituted in, so it only ever sees plain prose text nodes.
const SKIP = /^<\s*\/?\s*(pre|figure|code|a|h1|h2|h3|h4)\b/i;
function linkEntities(html: string): string {
  const parts = html.split(/(<[^>]+>)/g);
  let skip = 0;
  return parts
    .map((part) => {
      if (part.startsWith("<")) {
        if (SKIP.test(part)) {
          if (/^<\s*\//.test(part)) skip = Math.max(0, skip - 1);
          else if (!/\/>\s*$/.test(part)) skip += 1;
        }
        return part;
      }
      if (skip > 0 || part.trim() === "") return part;
      return part.replace(/\b(pusher|T-block|blocks?|tee|target)\b/gi, (w) => {
        const lw = w.toLowerCase();
        const kind = lw === "pusher" ? "pusher" : lw === "target" ? "target" : "block";
        return `<span class="bk-ent bk-ent-${kind}">${w}</span>`;
      });
    })
    .join("");
}

// A fenced include directive, e.g.
//   ```
//   [include-by-region: bc.py#model]
//   ```
// The language tag after the opening fence is optional; the artifact basename is
// matched generically (any chapter's artifact works) and the region name captured.
const INCLUDE_RE =
  /```[a-zA-Z0-9]*\n\[include-by-region:\s*[\w.]+#(\w+)\]\n```/g;

// The wall-clock placeholder: the one HTML comment that mentions "wall-clock
// table" (the top-of-file banner says "wall-clock values"). Swapped for a token
// before comments are stripped.
const WALLCLOCK_PLACEHOLDER_RE = /<!--[^>]*wall-clock table[^>]*-->/i;

// --- display math ($$…$$) ---------------------------------------------------
// The ~8 load-bearing equations (Rec 4 of the structure review) are rendered
// SERVER-SIDE with KaTeX into static HTML+MathML, so they render with JS
// disabled (grep the built page for class="katex" / <math>). The KaTeX CSS +
// fonts are self-hosted (bundled by Astro/Vite from node_modules — no CDN, no
// external fetch), imported once in ChapterLayout.astro.
//
// Like code panels, display math is extracted to a PLACEHOLDER TOKEN *before*
// `marked` runs, so marked never sees the raw LaTeX (its `_`, `\`, `*`, `&`
// would otherwise be mangled into emphasis/entities). The rendered KaTeX HTML is
// substituted back after marked, so the token protects the math end-to-end.
const DISPLAY_MATH_RE = /\$\$([\s\S]+?)\$\$/g;
const MATH_TOKEN_RE = /<p>\s*@@BK_MATH_(\d+)@@\s*<\/p>|@@BK_MATH_(\d+)@@/g;

/** Render one display equation to a static HTML+MathML block. `throwOnError:
 *  false` keeps a stray macro from ever failing the build; the wrapper carries
 *  overflow-x so a wide equation scrolls inside its own box (the page body never
 *  scrolls sideways). */
function renderDisplayMath(tex: string): string {
  const html = katex.renderToString(tex.trim(), {
    displayMode: true,
    throwOnError: false,
    output: "htmlAndMathml",
    strict: "ignore",
  });
  return `<div class="bk-math">${html}</div>`;
}

/** Pull every `$$…$$` block out of the raw markdown into a token + a parallel
 *  array of pre-rendered KaTeX HTML. Token indices are global across the whole
 *  source, so a token resolves the same in the lede or the body half. */
function extractMath(src: string): { src: string; blocks: string[] } {
  const blocks: string[] = [];
  const out = src.replace(DISPLAY_MATH_RE, (_m, tex: string) => {
    const i = blocks.length;
    blocks.push(renderDisplayMath(tex));
    return `\n\n@@BK_MATH_${i}@@\n\n`;
  });
  return { src: out, blocks };
}

/** Substitute the pre-rendered KaTeX blocks back in after marked has run. */
function injectMath(html: string, blocks: string[]): string {
  return html.replace(MATH_TOKEN_RE, (_m, a, b) => blocks[Number(a ?? b)] ?? "");
}

// Strip the leading top-of-file build-note comment block, then any remaining
// stray comments (author notes) so internal notes never reach the reader.
const LEADING_COMMENT_RE = /^\s*<!--[\s\S]*?-->\s*/;
const HTML_COMMENT_RE = /<!--[\s\S]*?-->/g;

// The draft opens with `# N.M: Title`, but the layout header already renders the
// title. Drop the first top-level heading to avoid a duplicate <h1>.
const LEADING_H1_RE = /^\s*#\s+[^\n]*\n/;

export interface RenderedChapter {
  ledeHtml: string; // the "See it work" lede — set beside the hero plate
  bodyHtml: string; // "The problem" onward — the book column (panels + wall inside)
  wallclockInjected: boolean; // true if the Run-it placeholder was found + filled
}

/**
 * Render a prose-ONLY module (a Phase-5 reading-track / graduation appendix) to
 * HTML. Unlike renderChapterProse these files carry no include-by-region code
 * panels, no wall-clock placeholder and no "See it work" split — they are plain
 * markdown essays. We drop the leading H1 (the page header renders the title),
 * strip author-note HTML comments, run marked, and add stable heading ids (same
 * slugify the chapter pages use), so the module reads fully server-side / JS-off.
 */
export function renderModuleProse(markdown: string): string {
  let src = markdown;
  src = src.replace(LEADING_COMMENT_RE, "");
  src = src.replace(LEADING_H1_RE, "");
  const math = extractMath(src); // display math → tokens before marked
  src = math.src;
  src = src.replace(HTML_COMMENT_RE, "");
  const html = addHeadingIds(marked.parse(src, { async: false }) as string, new Map<string, number>());
  return injectMath(html, math.blocks);
}

export interface RenderOptions {
  /** Rendered wall-clock ledger HTML to drop at the Run-it placeholder. */
  wallclockHtml?: string;
  /** Apply PushT entity tinting (prose words + code identifiers). Default off. */
  entities?: boolean;
}

/**
 * Render a chapter's draft markdown, splitting the "See it work" lede (rendered
 * beside the hero plate) from the body ("The problem" onward). Each
 * include-by-region directive becomes the exact region's dark code panel, and
 * the Run-it placeholder becomes the CSV-backed wall-clock ledger. `regions`
 * come from extractRegions() so the bytes are byte-identical to the drift gate.
 */
export function renderChapterProse(
  markdown: string,
  regions: Region[],
  artifactBasename: string,
  opts: RenderOptions = {},
): RenderedChapter {
  const byName = new Map<string, Region>(regions.map((r) => [r.name, r]));
  const entities = opts.entities ?? false;

  let src = markdown;

  // Swap the wall-clock placeholder for a token BEFORE stripping comments (the
  // placeholder itself is a comment).
  let wallclockInjected = false;
  if (opts.wallclockHtml && WALLCLOCK_PLACEHOLDER_RE.test(src)) {
    src = src.replace(WALLCLOCK_PLACEHOLDER_RE, "\n\n@@BK_WALL@@\n\n");
    wallclockInjected = true;
  }

  // Drop the top build-note comment and the draft's title H1 (the header owns it).
  src = src.replace(LEADING_COMMENT_RE, "");
  src = src.replace(LEADING_H1_RE, "");

  // Pull display math out to tokens BEFORE marked (raw LaTeX must not reach it),
  // rendering each equation to static KaTeX HTML now.
  const math = extractMath(src);
  src = math.src;

  // Swap every include fence for a PANEL token, then strip any remaining comments.
  src = src.replace(INCLUDE_RE, (_m, name: string) => `\n\n@@BK_PANEL_${name}@@\n\n`);
  src = src.replace(HTML_COMMENT_RE, "");

  // Split the "See it work" lede (goes with the plate) from the body, which
  // opens at "The problem" (where the drop cap lives). All chapters share the arc.
  const seeIdx = src.indexOf("## See it work");
  const problemIdx = src.indexOf("## The problem");
  const ledeMd =
    seeIdx >= 0 && problemIdx > seeIdx
      ? src.slice(seeIdx, problemIdx).replace(/^##\s+See it work\s*$/m, "")
      : "";
  const bodyMd = problemIdx >= 0 ? src.slice(problemIdx) : src;

  const PANEL_RE = /<p>\s*@@BK_PANEL_(\w+)@@\s*<\/p>|@@BK_PANEL_(\w+)@@/g;
  const WALL_RE = /<p>\s*@@BK_WALL@@\s*<\/p>|@@BK_WALL@@/;
  function codePanel(name: string): string {
    const r = byName.get(name);
    return r ? codePanelHtml(artifactBasename, r, entities) : missingPanelHtml(name);
  }
  function inject(html: string): string {
    return html
      .replace(PANEL_RE, (_m, a, b) => codePanel(a ?? b))
      .replace(WALL_RE, () => (opts.wallclockHtml ?? ""));
  }
  // Shared across the lede + body passes so heading ids are globally unique.
  const headingSeen = new Map<string, number>();
  function pipe(md: string): string {
    // marked (prose only) → optional entity-link → substitute the real panels +
    // wall + the pre-rendered KaTeX equations.
    let html = marked.parse(md, { async: false }) as string;
    if (entities) html = linkEntities(html);
    html = addHeadingIds(html, headingSeen);
    return injectMath(inject(html), math.blocks);
  }

  return {
    ledeHtml: pipe(ledeMd),
    bodyHtml: pipe(bodyMd),
    wallclockInjected,
  };
}
