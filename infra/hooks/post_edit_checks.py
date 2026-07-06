#!/usr/bin/env python3
"""PostToolUse hook: advisory checks that must actually reach the agent.

Per the Claude Code hooks API, a PostToolUse hook that just prints to stdout
and exits 0 is only visible in transcript mode — the agent never sees it. So:

- wallclock.csv manual edit (doctrine: "must not happen" — numbers come only
  from the wallclock-bench skill): use the STRONG channel — print the reason to
  stderr and exit 2, which is surfaced back to the agent.
- prose/chapter.md drift NOTE (advisory): emit structured JSON on stdout with
  `hookSpecificOutput.additionalContext` (exit 0) so the agent actually reads it.

Stdlib only.
"""
import json
import sys


def emit_context(message: str) -> None:
    """Feed an advisory back to the agent via PostToolUse additionalContext."""
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": message,
        }
    }))


def main():
    data = json.load(sys.stdin)
    ti = data.get("tool_input", {}) or {}
    path = ti.get("file_path", "") or ti.get("notebook_path", "") or ""

    # Strong channel: a manual wallclock.csv edit must not happen silently.
    if "curriculum/common/wallclock.csv" in path:
        print(
            "BLOCKED-ADVISORY: wallclock.csv was edited directly. Numbers must "
            "come from the wallclock-bench skill run, not hand-edits — revert "
            "this unless it promotes measured data from a bench report.",
            file=sys.stderr,
        )
        sys.exit(2)

    # Advisory context: prose edits need voice-check / drift labeling.
    if "/prose/" in path and path.endswith("chapter.md"):
        emit_context(
            "NOTE: prose edit detected. The voice-check skill must pass before "
            "author review. If this is a code-drift sync, label the PR "
            "'prose-sync'."
        )
        sys.exit(0)

    sys.exit(0)


if __name__ == "__main__":
    main()
