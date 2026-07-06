"""Exercise auto-checker — the local self-check, formalized.

This is job (1) of the grader (grader/CLAUDE.md): discover a chapter's
`exercises/suggested/checks.py` and run it via pytest, reporting pass/fail/skip
per test. It is exactly the "run it locally" the site links to (exercise-spec:
"Local: pytest exercises/checks.py — instant, offline, free-tier. This is the
primary experience. Exercises never phone home"). No network, no scoring
server: this path only runs pytest on human-owned check files.

Usage:
    python -m grader.check 0.1          # or ch0.1, or a full chapter path
    python -m grader.check 0.1 --json   # machine-readable report

Skips are first-class and expected: prediction gates SKIP until the learner
records a choice, and bug-hunt checks SKIP while the injected bug is present
(exercise-spec / ch0.1 checks.py). A chapter "passes" when nothing FAILED.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = REPO_ROOT / "curriculum"


@dataclass
class TestOutcome:
    nodeid: str
    outcome: str  # "passed" | "failed" | "skipped"
    message: str = ""


@dataclass
class CheckReport:
    chapter: str
    checks_path: str
    outcomes: list[TestOutcome] = field(default_factory=list)
    collection_error: str = ""

    @property
    def passed(self) -> int:
        return sum(o.outcome == "passed" for o in self.outcomes)

    @property
    def failed(self) -> int:
        return sum(o.outcome == "failed" for o in self.outcomes)

    @property
    def skipped(self) -> int:
        return sum(o.outcome == "skipped" for o in self.outcomes)

    @property
    def errors(self) -> int:
        return sum(o.outcome == "error" for o in self.outcomes)

    @property
    def ok(self) -> bool:
        """Green only when nothing failed OR errored and collection succeeded.
        A fixture-setup error (e.g. a checks.py that can't build its dataset
        offline) MUST fail the check, not pass silently."""
        return not self.collection_error and self.failed == 0 and self.errors == 0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        d["summary"] = {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "ok": self.ok,
        }
        return d


def find_checks(chapter: str) -> Path:
    """Resolve a chapter token to its exercises/suggested/checks.py.

    Accepts a number ("0.1"), a "ch" prefix ("ch0.1"), a chapter dir name
    ("ch0.1_sim_loop"), or a direct path to a checks.py / chapter dir.
    """
    p = Path(chapter)
    if p.is_file() and p.name == "checks.py":
        return p.resolve()
    if p.is_dir():
        found = list(p.glob("exercises/suggested/checks.py"))
        if found:
            return found[0].resolve()

    token = chapter.strip()
    if token.startswith("ch"):
        token = token[2:]
    # match a directory named ch{token}_...  (e.g. ch0.1_sim_loop)
    matches = sorted(CURRICULUM.glob(f"**/ch{token}_*/exercises/suggested/checks.py"))
    if not matches:
        # also allow an exact dir name like ch0.1_sim_loop
        matches = sorted(
            CURRICULUM.glob(f"**/{chapter}/exercises/suggested/checks.py")
        )
    if not matches:
        raise FileNotFoundError(
            f"no exercises/suggested/checks.py for chapter {chapter!r} under "
            f"{CURRICULUM} (looked for ch{token}_*/exercises/suggested/checks.py)"
        )
    if len(matches) > 1:
        raise ValueError(
            f"chapter {chapter!r} is ambiguous: {[str(m) for m in matches]}"
        )
    return matches[0].resolve()


class _Collector:
    """pytest plugin: record one outcome per test (call phase)."""

    def __init__(self) -> None:
        self.outcomes: list[TestOutcome] = []
        self.collection_error = ""

    def pytest_runtest_logreport(self, report) -> None:  # pytest hook
        # Skips surface at setup; pass/fail at call. A fixture error surfaces as
        # a FAILED setup (or teardown) phase and the call never runs — record it
        # as an "error" so a broken fixture can NEVER report a false green.
        if report.when == "setup":
            if report.skipped:
                self.outcomes.append(
                    TestOutcome(report.nodeid, "skipped", _skip_reason(report))
                )
            elif report.failed:
                msg = str(report.longrepr).splitlines()[-1] if report.longrepr else ""
                self.outcomes.append(TestOutcome(report.nodeid, "error", msg))
        elif report.when == "call":
            msg = ""
            if report.failed:
                msg = str(report.longrepr).splitlines()[-1] if report.longrepr else ""
            elif report.skipped:
                msg = _skip_reason(report)
            self.outcomes.append(TestOutcome(report.nodeid, report.outcome, msg))
        elif report.when == "teardown" and report.failed:
            msg = str(report.longrepr).splitlines()[-1] if report.longrepr else ""
            self.outcomes.append(TestOutcome(report.nodeid, "error", msg))

    def pytest_collectreport(self, report) -> None:  # pytest hook
        if report.failed:
            self.collection_error = str(report.longrepr)


def _skip_reason(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr) if longrepr else ""


def run_chapter_checks(chapter: str) -> CheckReport:
    """Run a chapter's checks.py via pytest, offline, and collect outcomes."""
    checks_path = find_checks(chapter)
    collector = _Collector()
    # -p no:cacheprovider keeps runs hermetic; rootdir at repo root so the
    # checks' `from curriculum.common...` and cwd-relative paths resolve.
    pytest.main(
        [str(checks_path), "-q", "-p", "no:cacheprovider", "--no-header"],
        plugins=[collector],
    )
    return CheckReport(
        chapter=chapter,
        checks_path=str(checks_path),
        outcomes=collector.outcomes,
        collection_error=collector.collection_error,
    )


def _print_human(report: CheckReport) -> None:
    print(f"chapter {report.chapter}")
    print(f"  checks: {report.checks_path}")
    if report.collection_error:
        print("  COLLECTION ERROR:")
        print("    " + report.collection_error.replace("\n", "\n    "))
        return
    glyph = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP", "error": "ERROR"}
    for o in report.outcomes:
        name = o.nodeid.split("::", 1)[-1]
        line = f"  [{glyph.get(o.outcome, o.outcome.upper())}] {name}"
        if o.message and o.outcome in ("failed", "skipped", "error"):
            line += f"  — {o.message}"
        print(line)
    err = f", {report.errors} error" if report.errors else ""
    print(
        f"  => {report.passed} passed, {report.failed} failed, "
        f"{report.skipped} skipped{err} — {'OK' if report.ok else 'FAILED'}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m grader.check",
        description="Run a chapter's suggested exercise checks offline.",
    )
    parser.add_argument("chapter", help="chapter token (e.g. 0.1, ch0.1) or path")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args(argv)

    try:
        report = run_chapter_checks(args.chapter)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        _print_human(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
