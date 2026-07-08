// Build-time dependency-graph layer for the learning-path map. Runs at SSG time
// (Node), never in the browser.
//
// HONESTY CONTRACT (mirrors lib/curriculum.ts): every edge is DERIVED from a
// real meta.yaml field — never hand-typed here. The four fields that encode a
// cross-chapter dependency in the curriculum are:
//
//   requires:        [chX.Y, ...]   hard map prerequisite (e.g. ch3.1 requires ch1.4)
//   builds_on:       chX.Y-slug     direct sequel — reuses the prior chapter's code
//   phase_requires:  [chX.Y, ...]   Phase-4 prerequisite (Post-Training's assumed chapters)
//   policy_reused:   chX.Y-slug     optionally runs an EARLIER chapter's trained policy
//
// Each field may be written inline (`key: val` or `key: [a, b]`) OR as a YAML
// block list on the following indented `- token` lines (ch5.2 declares two
// builds_on sources that way). We read both forms; a field can therefore carry
// more than one token. We still never invent an edge: only tokens actually
// present in the meta are read, and any that resolve to no chapter are dropped.
//
// The canonical meta loader (the python bridge, lib/curriculum.ts -> chapterMeta)
// does NOT surface these fields, and this package may not edit it. So we read the
// same meta.yaml files the bridge reads — read-only, the site's whole design is a
// lens over curriculum/ — and extract only these four scalar/list fields with a
// line-oriented parser. We never execute anything and never invent an edge: a
// token that resolves to no discovered chapter is dropped, not faked.

import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { REPO_ROOT, type ChapterInfo } from "./curriculum.ts";

export type EdgeRel = "requires" | "builds_on" | "phase_requires" | "policy_reused";

/** Human labels for the legend + the static (JS-off) thread list. */
export const REL_LABEL: Record<EdgeRel, string> = {
  builds_on: "builds on",
  requires: "requires",
  phase_requires: "phase gate",
  policy_reused: "reuses policy",
};

export interface RawDeps {
  requires: string[];
  buildsOn: string[];
  phaseRequires: string[];
  policyReused: string[];
}

export interface GraphEdge {
  from: string; // prerequisite chapter id (the source you learn first)
  to: string; // dependent chapter id (the chapter it unblocks)
  rel: EdgeRel;
}

/** Strip a trailing YAML `# comment` — chapter tokens never contain `#`. */
function stripComment(v: string): string {
  const i = v.indexOf("#");
  return (i >= 0 ? v.slice(0, i) : v).trim();
}

/** Parse an inline YAML flow list `[a, b, c]` into trimmed, unquoted tokens. */
function parseList(raw: string): string[] {
  const m = stripComment(raw).match(/\[([^\]]*)\]/);
  if (!m) return [];
  return m[1]
    .split(",")
    .map((s) => s.trim().replace(/^['"]|['"]$/g, ""))
    .filter(Boolean);
}

/** Parse a single scalar token (unquoted), or null if empty. */
function parseScalar(raw: string): string | null {
  const v = stripComment(raw).replace(/^['"]|['"]$/g, "");
  return v || null;
}

/** meta.yaml key -> the RawDeps field it fills. */
const DEP_FIELD: Record<string, keyof RawDeps> = {
  requires: "requires",
  builds_on: "buildsOn",
  phase_requires: "phaseRequires",
  policy_reused: "policyReused",
};

/** Read the four dependency-bearing fields from a chapter's meta.yaml.
 *  Only TOP-LEVEL keys count (no leading whitespace), so an indented mention
 *  inside another block can never be mistaken for a real edge. Each field may be
 *  inline (`key: val` / `key: [a, b]`) or a YAML block list on the following
 *  `  - token` lines; both forms are collected into the field's token array. */
export function chapterDeps(chapterDirRel: string): RawDeps {
  const deps: RawDeps = { requires: [], buildsOn: [], phaseRequires: [], policyReused: [] };
  let text: string;
  try {
    text = readFileSync(resolve(REPO_ROOT, chapterDirRel, "meta.yaml"), "utf-8");
  } catch {
    return deps; // no meta — no edges (never throws, never invents)
  }
  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length; i++) {
    const m = lines[i].match(/^(requires|builds_on|phase_requires|policy_reused):\s*(.*)$/);
    if (!m) continue;
    const field = DEP_FIELD[m[1]];
    const rest = m[2];
    const inline = stripComment(rest);
    if (inline.startsWith("[")) {
      deps[field].push(...parseList(rest)); // inline flow list
    } else if (inline) {
      const v = parseScalar(rest); // inline scalar
      if (v) deps[field].push(v);
    } else {
      // block list: consume the following indented `  - token` lines. The first
      // line that is not a list item (blank, comment, next key) ends the block.
      for (let j = i + 1; j < lines.length; j++) {
        const item = lines[j].match(/^\s+-\s+(.*)$/);
        if (!item) break;
        const v = parseScalar(item[1]);
        if (v) deps[field].push(v);
      }
    }
  }
  return deps;
}

/** Resolve a meta token to a discovered chapter id, or null.
 *  Tokens come in two shapes: a full id/slug (`ch3.1-world-models`,
 *  `ch2.1-ppo`) or the short number form (`ch1.4`). We never invent a chapter:
 *  an unresolvable token returns null and its edge is dropped. */
export function resolveToken(token: string, chapters: ChapterInfo[]): string | null {
  const t = token.trim();
  if (!t) return null;
  // 1. exact id
  const exact = chapters.find((c) => c.id === t);
  if (exact) return exact.id;
  // 2. short number form `chX.Y` -> chapter whose derived number is X.Y
  const num = t.match(/^ch([\d.]+)$/);
  if (num) {
    const byNum = chapters.find((c) => c.number === num[1]);
    if (byNum) return byNum.id;
  }
  // 3. id prefix (`ch3.1` -> `ch3.1-world-models`)
  const byPrefix = chapters.find((c) => c.id === t || c.id.startsWith(`${t}-`));
  return byPrefix ? byPrefix.id : null;
}

/** The full, de-duplicated edge set derived from every chapter's meta.yaml.
 *  Order: requires, builds_on, phase_requires, policy_reused, in chapter order. */
export function buildEdges(chapters: ChapterInfo[]): GraphEdge[] {
  const edges: GraphEdge[] = [];
  const seen = new Set<string>();
  const add = (fromTok: string, toId: string, rel: EdgeRel) => {
    const from = resolveToken(fromTok, chapters);
    if (!from || from === toId) return; // unresolvable or self-edge → drop
    const key = `${from}->${toId}`;
    if (seen.has(key)) return; // one thread per pair; first (strongest-declared) wins
    seen.add(key);
    edges.push({ from, to: toId, rel });
  };
  for (const ch of chapters) {
    const d = chapterDeps(ch.chapterDir);
    d.requires.forEach((t) => add(t, ch.id, "requires"));
    d.buildsOn.forEach((t) => add(t, ch.id, "builds_on"));
    d.phaseRequires.forEach((t) => add(t, ch.id, "phase_requires"));
    d.policyReused.forEach((t) => add(t, ch.id, "policy_reused"));
  }
  return edges;
}

export interface ChapterGraph {
  edges: GraphEdge[];
  /** id -> ids this chapter depends on (its prerequisites / threads-in). */
  prereqOf: Map<string, string[]>;
  /** id -> ids this chapter unblocks (its threads-out). */
  unblockOf: Map<string, string[]>;
}

/** Adjacency in both directions, for highlight + the per-node text summary. */
export function buildGraph(chapters: ChapterInfo[]): ChapterGraph {
  const edges = buildEdges(chapters);
  const prereqOf = new Map<string, string[]>();
  const unblockOf = new Map<string, string[]>();
  for (const e of edges) {
    (prereqOf.get(e.to) ?? prereqOf.set(e.to, []).get(e.to)!).push(e.from);
    (unblockOf.get(e.from) ?? unblockOf.set(e.from, []).get(e.from)!).push(e.to);
  }
  return { edges, prereqOf, unblockOf };
}
