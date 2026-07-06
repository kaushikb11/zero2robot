/**
 * HelpOverlay — the site's keyboard-shortcuts + "how to use this site" overlay.
 *
 * WHY IT EXISTS. A keyboard-first reader needs one place that lists every real
 * shortcut and the keyboard path through the interactive pieces (search, the
 * concept toys, the predict-then-run gate). This island IS that place, opened
 * from anywhere with "?".
 *
 * ── SSR-SAFE ISLAND CONTRACT (mirrors SearchOverlay.tsx's frozen contract) ────
 * 1. No window/document at module scope or in the initial render. Every browser
 *    API touch lives inside an effect or an event handler.
 * 2. Renders NOTHING until opened (`open === false` → returns null). With JS off
 *    there is simply no overlay; the /about page documents the same shortcuts as
 *    static HTML, so reading is entirely unaffected (acceptable degradation).
 * 3. No data is fetched; the content is static. There is nothing to load.
 *
 * ── HOW IT OPENS (the public API the orchestrator wires) ──────────────────────
 * The island self-binds the global "?" shortcut once hydrated, so it works with
 * nothing more than mounting it. It ALSO exposes an imperative handle + an event
 * for a visible affordance / the orchestrator:
 *    • window.z2rHelp.open() / .close()                 — call directly
 *    • window.dispatchEvent(new Event("z2r:help-open")) — fire an event
 * Both routes converge on the same controlled `open` state. Esc / backdrop close.
 * A full focus trap keeps Tab inside the dialog; focus is restored on close.
 */
import { useEffect, useRef, useState } from "preact/hooks";
import "./help.css";

interface Shortcut {
  keys: string[]; // rendered as <kbd> chips joined by "then"/"or" via `sep`
  sep?: "or" | "then";
  label: string;
}
interface Group {
  heading: string;
  note?: string;
  rows: Shortcut[];
}

// The REAL shortcuts, mirroring what the components actually bind:
//  · SearchOverlay: "/" and ⌘/Ctrl-K open; ↑↓/Home/End/↵/Esc inside.
//  · This overlay: "?" opens, Esc closes.
//  · PlateIsland / concept toys: focus the figure, arrow keys nudge, R reset,
//    O out-of-distribution (ch1.1); other toys expose their own one control.
//  · PredictGate: Tab to a choice, Space/↵ to select, Commit to reveal.
const GROUPS: Group[] = [
  {
    heading: "Anywhere on the site",
    rows: [
      { keys: ["/", "⌘ K"], sep: "or", label: "Open search — jump to a chapter, heading, objective, or code region" },
      { keys: ["?"], label: "Open this shortcuts & how-to overlay" },
      { keys: ["Esc"], label: "Close the open overlay" },
      { keys: ["Tab"], label: "Move through links and controls; the skip-link jumps you to the reading" },
    ],
  },
  {
    heading: "In search",
    rows: [
      { keys: ["↑", "↓"], sep: "or", label: "Move between results" },
      { keys: ["Home", "End"], sep: "or", label: "Jump to the first or last result" },
      { keys: ["↵"], label: "Open the selected result" },
      { keys: ["Esc"], label: "Close search" },
    ],
  },
  {
    heading: "In a concept toy",
    note: "Focus the “See it work” toy first (Tab to it, or click it).",
    rows: [
      { keys: ["↑", "↓", "←", "→"], label: "Nudge the one exposed control (e.g. move the block)" },
      { keys: ["R"], label: "Reset the toy to its starting state" },
      { keys: ["O"], label: "Send it out of distribution (flagship covariate-shift toy)" },
    ],
  },
  {
    heading: "In a predict-then-run exercise",
    rows: [
      { keys: ["Tab"], label: "Move to the prediction choices" },
      { keys: ["Space", "↵"], sep: "or", label: "Select the focused choice" },
      { keys: ["Tab", "↵"], sep: "then", label: "Reach the Commit button and reveal the measured answer" },
    ],
  },
];

export default function HelpOverlay() {
  const [open, setOpen] = useState(false);

  const dialogRef = useRef<HTMLDivElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);
  const restoreFocusRef = useRef<Element | null>(null);

  const openOverlay = () => {
    if (open) return;
    restoreFocusRef.current = document.activeElement;
    setOpen(true);
  };
  const closeOverlay = () => setOpen(false);

  const isEditable = (el: EventTarget | null): boolean => {
    const n = el as HTMLElement | null;
    if (!n) return false;
    const tag = n.tagName;
    return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT" || n.isContentEditable === true;
  };

  // --- global keybind + imperative API (client-only; registered post-hydration)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      // "?" (Shift+/ on most layouts) opens, unless the user is typing or holding
      // a command/control/alt modifier.
      const help = e.key === "?" && !e.metaKey && !e.ctrlKey && !e.altKey && !isEditable(e.target);
      if (help) {
        e.preventDefault();
        // toggle: pressing "?" again closes, so it never feels stuck open.
        setOpen((o) => {
          if (!o) restoreFocusRef.current = document.activeElement;
          return !o;
        });
      }
    };
    const onOpenEvent = () => openOverlay();

    document.addEventListener("keydown", onKey);
    window.addEventListener("z2r:help-open", onOpenEvent);
    (window as any).z2rHelp = { open: openOverlay, close: closeOverlay };

    return () => {
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("z2r:help-open", onOpenEvent);
      if ((window as any).z2rHelp) delete (window as any).z2rHelp;
    };
    // openOverlay/closeOverlay are stable enough for this island's lifetime.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  // --- open/close side effects: scroll lock, focus, focus restore --------------
  useEffect(() => {
    if (open) {
      const prevOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
      // focus the close button after paint so screen readers announce the dialog
      requestAnimationFrame(() => closeRef.current?.focus());
      return () => {
        document.body.style.overflow = prevOverflow;
      };
    }
    const el = restoreFocusRef.current as HTMLElement | null;
    if (el && typeof el.focus === "function") el.focus();
    return undefined;
  }, [open]);

  const focusables = (): HTMLElement[] => {
    const root = dialogRef.current;
    if (!root) return [];
    return Array.from(
      root.querySelectorAll<HTMLElement>(
        'a[href], button:not([disabled]), [tabindex]:not([tabindex="-1"])',
      ),
    ).filter((el) => el.offsetParent !== null || el === document.activeElement);
  };

  const onDialogKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Escape") {
      e.preventDefault();
      closeOverlay();
      return;
    }
    if (e.key !== "Tab") return;
    // Full focus trap: keep Tab / Shift-Tab cycling inside the dialog.
    const items = focusables();
    if (items.length === 0) return;
    const first = items[0];
    const last = items[items.length - 1];
    const activeEl = document.activeElement as HTMLElement | null;
    if (e.shiftKey) {
      if (activeEl === first || !dialogRef.current?.contains(activeEl)) {
        e.preventDefault();
        last.focus();
      }
    } else if (activeEl === last) {
      e.preventDefault();
      first.focus();
    }
  };

  // Contract rule 2: render NOTHING until opened (SSR + JS-off safe).
  if (!open) return null;

  return (
    <div class="hlp-root" onKeyDown={onDialogKeyDown}>
      <div class="hlp-backdrop" onClick={closeOverlay} aria-hidden="true" />

      <div
        ref={dialogRef}
        class="hlp-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="hlp-title"
      >
        <div class="hlp-head">
          <div>
            <p class="hlp-eyebrow">Keyboard &amp; how-to</p>
            <h2 class="hlp-title" id="hlp-title">Shortcuts and how to use this site</h2>
          </div>
          <button ref={closeRef} type="button" class="hlp-close" onClick={closeOverlay} aria-label="Close help">
            Esc
          </button>
        </div>

        <div class="hlp-body">
          {GROUPS.map((g) => (
            <section class="hlp-group" key={g.heading} aria-label={g.heading}>
              <h3 class="hlp-group-title">{g.heading}</h3>
              {g.note && <p class="hlp-group-note">{g.note}</p>}
              <dl class="hlp-list">
                {g.rows.map((row) => (
                  <div class="hlp-row" key={row.label}>
                    <dt class="hlp-keys">
                      {row.keys.map((k, i) => (
                        <span class="hlp-keychip" key={k}>
                          {i > 0 && <span class="hlp-sep">{row.sep === "then" ? "then" : "or"}</span>}
                          <kbd class="hlp-kbd">{k}</kbd>
                        </span>
                      ))}
                    </dt>
                    <dd class="hlp-desc">{row.label}</dd>
                  </div>
                ))}
              </dl>
            </section>
          ))}
        </div>

        <div class="hlp-footer">
          <span>
            The full walkthrough lives on the <a class="hlp-link" href="/about/">Start here</a> page.
          </span>
          <span class="hlp-footer-hint" aria-hidden="true">
            <kbd class="hlp-kbd">?</kbd> toggles this · <kbd class="hlp-kbd">esc</kbd> closes
          </span>
        </div>
      </div>
    </div>
  );
}
