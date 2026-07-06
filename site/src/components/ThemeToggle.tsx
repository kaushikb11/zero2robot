/**
 * ThemeToggle — the light↔dark switch for the interactive-textbook.
 *
 * The whole palette is token-based (see the DARK THEME block in public/styles.css):
 * dark = redefine the tokens under `:root[data-theme="dark"]`, everything adapts.
 * This button's ONLY job is to stamp `data-theme` on <html>, persist the choice
 * in localStorage ("z2r:theme"), and reflect the current state accessibly.
 *
 * SSR / hydration contract (mirrors ProgressOverview + the FROZEN concept-toy
 * contract in PlateIsland.tsx):
 *   · NO window/localStorage at module scope. `readTheme()` is only ever called
 *     inside a lazy state initialiser / effect / event handler, each guarded so
 *     it is a no-op on the server.
 *   · The no-flash <head> snippet (see wire-in note) has already stamped
 *     data-theme before hydration, so the lazy initialiser reads the true value
 *     on the first client render — no flash, no aria mismatch.
 *   · Mount it `client:only="preact"` (like SearchOverlay): the toggle is a
 *     JS-only enhancement. With JS off there is no button and readers get the
 *     OS-media default from the stylesheet — exactly the intended fallback.
 *   · Accessible: a native <button> (Enter/Space free) with aria-pressed
 *     reflecting "dark active" + an aria-label; focus-visible ring in styles.css.
 *   · While the reader has NOT made an explicit choice, it follows OS changes
 *     live (matchMedia listener, only when no stored preference exists).
 */
import { useEffect, useState } from "preact/hooks";

type Theme = "light" | "dark";
const KEY = "z2r:theme";

/** Resolve the OS preference. Guarded — safe to call only in the browser. */
function systemTheme(): Theme {
  return typeof window !== "undefined" &&
    window.matchMedia("(prefers-color-scheme: dark)").matches
    ? "dark"
    : "light";
}

/** The theme in effect right now: the <html data-theme> stamp the no-flash
 *  script set → else a stored preference → else the OS default. Browser-only. */
function readTheme(): Theme {
  if (typeof document === "undefined") return "light";
  const stamped = document.documentElement.dataset.theme;
  if (stamped === "light" || stamped === "dark") return stamped;
  try {
    const stored = localStorage.getItem(KEY);
    if (stored === "light" || stored === "dark") return stored;
  } catch {
    /* storage blocked — fall through to OS */
  }
  return systemTheme();
}

export default function ThemeToggle() {
  // Lazy init reads the already-stamped <html data-theme> on the first client
  // render (the head snippet ran before hydration). `null` only on the server.
  const [theme, setTheme] = useState<Theme | null>(() =>
    typeof document === "undefined" ? null : readTheme(),
  );

  useEffect(() => {
    // reconcile once mounted (covers client:only, where the first render is
    // already client-side but we still want a single source of truth)
    setTheme(readTheme());

    // follow the OS while the reader has made no explicit choice
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onOS = () => {
      let explicit = false;
      try {
        const s = localStorage.getItem(KEY);
        explicit = s === "light" || s === "dark";
      } catch {
        /* ignore */
      }
      if (!explicit) {
        // leave data-theme unstamped so the media query keeps driving it;
        // just re-render the glyph to match the new OS state
        setTheme(systemTheme());
      }
    };
    mq.addEventListener("change", onOS);
    return () => mq.removeEventListener("change", onOS);
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next; // wins over the OS media query
    try {
      localStorage.setItem(KEY, next);
    } catch {
      /* storage blocked — the in-page toggle still works for this session */
    }
    setTheme(next);
  }

  const active: Theme = theme ?? "light";
  const isDark = active === "dark";

  return (
    <button
      type="button"
      class="theme-toggle"
      data-active={active}
      aria-pressed={isDark}
      aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
      title={isDark ? "Light theme" : "Dark theme"}
      onClick={toggle}
    >
      {/* sun — shown in light (click → go dark) */}
      <svg class="tt-sun" viewBox="0 0 24 24" fill="none" stroke="currentColor"
        stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="4.2" />
        <path d="M12 2.4v2.4M12 19.2v2.4M4.22 4.22l1.7 1.7M18.08 18.08l1.7 1.7M2.4 12h2.4M19.2 12h2.4M4.22 19.78l1.7-1.7M18.08 5.92l1.7-1.7" />
      </svg>
      {/* moon — shown in dark (click → go light) */}
      <svg class="tt-moon" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
        <path d="M20 14.6A8.2 8.2 0 0 1 9.4 4 8.2 8.2 0 1 0 20 14.6Z" />
      </svg>
    </button>
  );
}
