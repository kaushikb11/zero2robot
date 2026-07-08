"""Exercise auto-checker — the local self-check, formalized.

This is job (1) of the grader (grader/CLAUDE.md): discover a chapter's
`exercises/suggested/checks.py` and run its PUBLIC checks via pytest, reporting
pass/fail/skip per exercise with the seeded band + provenance each check verified
against. It is exactly the "run it locally" the site links to (exercise-spec:
"Local: pytest exercises/checks.py — instant, offline, free-tier. This is the
primary experience. Exercises never phone home"). No network, no scoring server,
no hidden seeds: this path only runs pytest on human-owned check files and reads
the public `exercise_checks` bands from each chapter's meta.yaml.

Usage:
    python -m grader.check 1.1          # or ch1.1, a dir name, or a path
    python -m grader.check 1.1 --json   # machine-readable report
    python -m grader.check 1.1 --slow   # also run @slow checks (the fast lane
                                        # is the default; slow/gpu deselected)

By default only the FAST lane runs (`-m "not slow and not gpu"`, matching
`make check`): the checks that a learner can run instantly on a CPU laptop.
The @slow checks (short training runs) are deselected and reported as a count.

Skips are first-class and expected: predict-then-run gates SKIP until the learner
records a choice, and bug-hunt checks SKIP while the injected bug is still present
(finding it is the exercise). A chapter "passes" when nothing FAILED or errored.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import io
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

try:  # pyyaml is a dev/CI dep; degrade gracefully if a bare env lacks it.
    import yaml
except ModuleNotFoundError:  # pragma: no cover - exercised only on bare envs
    yaml = None

REPO_ROOT = Path(__file__).resolve().parents[1]
CURRICULUM = REPO_ROOT / "curriculum"

# nodeid -> exercise slug. checks are named test_ex<N>_...; that <N> ties a check
# back to its exercise file and the meta exercise_checks[ex<N>] band.
_EX_RE = re.compile(r"\btest_(ex\d+)")
# docstring fallback for the ~5 exercise files with no METADATA dict:
#   """SUGGESTED exercise candidate (humans promote) — bug-hunt, ch1.1.
_DOC_TYPE_RE = re.compile(r"[—-]\s*([a-z][a-z-]+),")


@dataclass
class TestOutcome:
    nodeid: str
    outcome: str  # "passed" | "failed" | "skipped" | "error"
    message: str = ""
    exercise: str = ""  # "ex1", "ex2", ... or "" when a check maps to no exercise


@dataclass
class CheckReport:
    chapter: str
    checks_path: str
    meta_path: str = ""
    outcomes: list[TestOutcome] = field(default_factory=list)
    # exercise slug -> its public meta exercise_checks block (bands + provenance).
    bands: dict[str, dict] = field(default_factory=dict)
    types: dict[str, str] = field(default_factory=dict)  # exercise slug -> type
    deselected: int = 0  # @slow / @gpu checks not run in the fast lane
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
            "deselected": self.deselected,
            "ok": self.ok,
        }
        return d


def find_checks(chapter: str) -> Path:
    """Resolve a chapter token to its exercises/suggested/checks.py.

    Accepts a number ("1.1"), a "ch" prefix ("ch1.1"), a chapter dir name
    ("ch1.1_bc"), or a direct path to a checks.py / chapter dir.
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
    # match a directory named ch{token}_...  (e.g. ch1.1_bc)
    matches = sorted(CURRICULUM.glob(f"**/ch{token}_*/exercises/suggested/checks.py"))
    if not matches:
        # also allow an exact dir name like ch1.1_bc
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


def _exercise_id(nodeid: str) -> str:
    m = _EX_RE.search(nodeid.split("::", 1)[-1])
    return m.group(1) if m else ""


def _exercise_type(suggested_dir: Path, exid: str) -> str:
    """Read an exercise's declared type WITHOUT executing the file.

    Prefers the METADATA = {"type": ...} dict (via AST); falls back to the
    docstring header ("— bug-hunt, ch1.1"). Never imports the module (some
    exercise files run heavy work on import)."""
    matches = sorted(suggested_dir.glob(f"{exid}_*.py"))
    if not matches:
        return ""
    src = matches[0].read_text()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in tree.body:
            targets = getattr(node, "targets", [])
            if (
                isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "METADATA" for t in targets)
                and isinstance(node.value, ast.Dict)
            ):
                for key, val in zip(node.value.keys, node.value.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "type"
                        and isinstance(val, ast.Constant)
                    ):
                        return str(val.value)
    doc = ast.get_docstring(tree) if tree is not None else None
    m = _DOC_TYPE_RE.search(doc or src[:400])
    return m.group(1) if m else ""


def _load_meta_checks(checks_path: Path) -> tuple[Path | None, dict]:
    """Return (meta_path, exercise_checks) for the chapter owning checks_path."""
    # .../<chapter>/exercises/suggested/checks.py -> parents[2] is <chapter>.
    meta_path = checks_path.parents[2] / "meta.yaml"
    if yaml is None or not meta_path.is_file():
        return (meta_path if meta_path.is_file() else None), {}
    try:
        meta = yaml.safe_load(meta_path.read_text()) or {}
    except yaml.YAMLError:
        return meta_path, {}
    return meta_path, meta.get("exercise_checks", {}) or {}


class _Collector:
    """pytest plugin: record one outcome per test (call phase)."""

    def __init__(self) -> None:
        self.outcomes: list[TestOutcome] = []
        self.collection_error = ""
        self.deselected = 0

    def _add(self, nodeid: str, outcome: str, message: str = "") -> None:
        self.outcomes.append(
            TestOutcome(nodeid, outcome, message, exercise=_exercise_id(nodeid))
        )

    def pytest_deselected(self, items) -> None:  # pytest hook
        self.deselected += len(items)

    def pytest_runtest_logreport(self, report) -> None:  # pytest hook
        # Skips surface at setup; pass/fail at call. A fixture error surfaces as
        # a FAILED setup (or teardown) phase and the call never runs — record it
        # as an "error" so a broken fixture can NEVER report a false green.
        if report.when == "setup":
            if report.skipped:
                self._add(report.nodeid, "skipped", _skip_reason(report))
            elif report.failed:
                msg = str(report.longrepr).splitlines()[-1] if report.longrepr else ""
                self._add(report.nodeid, "error", msg)
        elif report.when == "call":
            msg = ""
            if report.failed:
                msg = str(report.longrepr).splitlines()[-1] if report.longrepr else ""
            elif report.skipped:
                msg = _skip_reason(report)
            self._add(report.nodeid, report.outcome, msg)
        elif report.when == "teardown" and report.failed:
            msg = str(report.longrepr).splitlines()[-1] if report.longrepr else ""
            self._add(report.nodeid, "error", msg)

    def pytest_collectreport(self, report) -> None:  # pytest hook
        if report.failed:
            self.collection_error = str(report.longrepr)


def _skip_reason(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr) if longrepr else ""


def run_chapter_checks(chapter: str, include_slow: bool = False) -> CheckReport:
    """Run a chapter's checks.py via pytest, offline, and collect outcomes.

    By default only the fast lane runs (@slow / @gpu checks deselected), matching
    `make check`. Pass include_slow=True to run everything."""
    checks_path = find_checks(chapter)
    meta_path, exercise_checks = _load_meta_checks(checks_path)
    collector = _Collector()
    # -p no:cacheprovider keeps runs hermetic; rootdir at repo root so the
    # checks' `from curriculum.common...` and cwd-relative paths resolve.
    # --import-mode=importlib so every chapter's identically-named `checks.py`
    # imports under a unique name (no "import file mismatch" when the grader's
    # own test suite checks several chapters in one process). We redirect
    # pytest's terminal chatter into a buffer and print a clean report ourselves
    # (the raw output is kept for the collection-error case).
    args = [
        str(checks_path), "-q", "-p", "no:cacheprovider", "--no-header",
        "--import-mode=importlib",
    ]
    if not include_slow:
        args += ["-m", "not slow and not gpu"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        pytest.main(args, plugins=[collector])

    # attach the public meta band + declared type for each exercise touched.
    suggested_dir = checks_path.parent
    seen = {o.exercise for o in collector.outcomes if o.exercise}
    bands: dict[str, dict] = {}
    types: dict[str, str] = {}
    top_provenance = exercise_checks.get("provenance")
    for exid in sorted(seen):
        block = exercise_checks.get(exid)
        if isinstance(block, dict):
            block = dict(block)
            if top_provenance and "provenance" not in block:
                block["provenance"] = top_provenance
            bands[exid] = block
        types[exid] = _exercise_type(suggested_dir, exid)

    report = CheckReport(
        chapter=chapter,
        checks_path=str(checks_path),
        meta_path=str(meta_path) if meta_path else "",
        outcomes=collector.outcomes,
        bands=bands,
        types=types,
        deselected=collector.deselected,
        collection_error=collector.collection_error,
    )
    if report.collection_error:
        # keep pytest's own traceback available when collection blew up.
        tail = buf.getvalue().strip()
        if tail and tail not in report.collection_error:
            report.collection_error += "\n" + tail
    return report


_GLYPH = {"passed": "PASS", "failed": "FAIL", "skipped": "SKIP", "error": "ERROR"}


def _format_band(block: dict) -> tuple[str, str]:
    """Split an exercise_checks block into (band summary, provenance)."""
    provenance = str(block.get("provenance", "")).strip()
    parts = []
    for key, val in block.items():
        if key == "provenance":
            continue
        if isinstance(val, list):
            shown = val if len(val) <= 4 else [*val[:4], "..."]
            parts.append(f"{key}={shown}")
        else:
            parts.append(f"{key}={val}")
    return ", ".join(parts), provenance


def _print_human(report: CheckReport) -> None:
    print(f"chapter {report.chapter}")
    print(f"  checks: {report.checks_path}")
    if report.meta_path:
        print(f"  meta:   {report.meta_path}")
    if report.collection_error:
        print("  COLLECTION ERROR:")
        print("    " + report.collection_error.replace("\n", "\n    "))
        return

    # group outcomes by exercise, preserving first-seen order.
    order: list[str] = []
    grouped: dict[str, list[TestOutcome]] = {}
    for o in report.outcomes:
        key = o.exercise or "(other)"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(o)

    for key in order:
        etype = report.types.get(key, "")
        header = f"  {key}" + (f"  [{etype}]" if etype else "")
        print(header)
        for o in grouped[key]:
            name = o.nodeid.split("::", 1)[-1]
            line = f"    [{_GLYPH.get(o.outcome, o.outcome.upper())}] {name}"
            if o.message and o.outcome in ("failed", "skipped", "error"):
                line += f"  — {o.message}"
            print(line)
        block = report.bands.get(key)
        if block:
            band, provenance = _format_band(block)
            if band:
                print(f"      band: {band}")
            if provenance:
                print(f"      provenance: {provenance}")

    err = f", {report.errors} error" if report.errors else ""
    desel = f", {report.deselected} slow/gpu deselected" if report.deselected else ""
    print(
        f"  => {report.passed} passed, {report.failed} failed, "
        f"{report.skipped} skipped{err}{desel} — {'OK' if report.ok else 'FAILED'}"
    )
    if report.deselected and not report.errors:
        print("     (fast lane; re-run with --slow to include training checks)")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m grader.check",
        description="Run a chapter's public suggested exercise checks offline.",
    )
    parser.add_argument("chapter", help="chapter token (e.g. 1.1, ch1.1) or path")
    parser.add_argument(
        "--slow", action="store_true",
        help="also run @slow / @gpu checks (default: fast lane only)",
    )
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args(argv)

    try:
        report = run_chapter_checks(args.chapter, include_slow=args.slow)
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
