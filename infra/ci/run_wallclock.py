#!/usr/bin/env python3
"""run_wallclock — full chapter runs with wall-clock capture.

Contract (ci-lanes.md lane 3, .github/workflows/ci-gpu.yml): run each selected
chapter artifact FULL (no --smoke) as `{artifact} --seed {seed} --no-rerun
--out {tmpdir}`, wall-time it, and append a MEASURED row to
infra/ci/reports/wallclock_latest.csv using the shared 7-column schema
(chapter,tier,wallclock_min,config_hash,commit,date,status).

- This script NEVER touches curriculum/common/wallclock.csv — promotion of
  measured rows into the canonical ledger is a separate reviewed PR
  (opened by the headless-Claude step in ci-gpu.yml).
- config_hash = first 12 hex of sha256(artifact bytes + "|tier=..|seed=..").
- commit = `git rev-parse --short HEAD` ("unknown" outside a git repo).
- --tier accepts every CI tier (cpu-laptop, t4-parity, t4, 4090, mps) even on
  machines where only some are meaningful; the tier is provenance, not a
  device selector.
- --chapters: all | changed-since-last | scale-labs | <chapter dir>.
  scale-labs selects chapters whose meta.yaml carries a truthy scale_lab or
  scale_lab_ref key (curriculum/CLAUDE.md meta.yaml "scale-lab ref").
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from lib.chapters import (
    Chapter,
    ChapterError,
    discover_chapters,
    load_chapter,
    select_changed,
)

# l40s: the honest Scale-Lab GPU tier measured on Modal (Ada, the 4090 analog) —
# proposed in infra/decisions/014; promotion into the ledger/registry is human-ratified.
TIERS = ["cpu-laptop", "t4-parity", "t4", "4090", "l40s", "mps"]
CSV_HEADER = [
    "chapter",
    "tier",
    "wallclock_min",
    "config_hash",
    "commit",
    "date",
    "status",
]


def resolve_python(root: Path, override: str | None) -> str:
    if override:
        return override
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def git_short_head(root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def config_hash(artifact: Path, tier: str, seed: int) -> str:
    digest = hashlib.sha256()
    digest.update(artifact.read_bytes())
    digest.update(f"|tier={tier}|seed={seed}".encode())
    return digest.hexdigest()[:12]


def select_chapters(
    spec: str, chapters: list[Chapter], root: Path
) -> tuple[list[Chapter], str]:
    if spec == "all":
        return chapters, f"selected ALL {len(chapters)} chapter(s)"
    if spec == "changed-since-last":
        return select_changed(chapters, root)
    if spec == "scale-labs":
        selected = [
            chapter
            for chapter in chapters
            if chapter.meta.get("scale_lab") or chapter.meta.get("scale_lab_ref")
        ]
        return selected, f"selected {len(selected)} scale-lab chapter(s)"
    # A single chapter directory, absolute or repo-root/curriculum relative.
    for candidate in (Path(spec), root / spec, root / "curriculum" / spec):
        if (candidate / "meta.yaml").is_file():
            return [load_chapter(candidate.resolve())], f"selected {spec}"
    raise ChapterError(
        f"--chapters '{spec}' is not all|changed-since-last|scale-labs and no "
        "chapter directory with a meta.yaml was found at that path"
    )


def run_full(
    python: str, chapter: Chapter, root: Path, seed: int, timeout: int | None
) -> tuple[float | None, str]:
    """Full (non-smoke) run. Returns (seconds, detail); seconds None on failure."""
    artifact = chapter.artifact_path
    if artifact is None or not artifact.is_file():
        return None, (
            f"artifact '{chapter.meta.get('artifact')}' missing "
            "(chapter layout, curriculum/CLAUDE.md)"
        )
    with tempfile.TemporaryDirectory(prefix="z2r-wallclock-") as tmp:
        cmd = [
            python,
            str(artifact),
            "--seed",
            str(seed),
            "--no-rerun",
            "--out",
            tmp,
        ]
        start = time.monotonic()
        try:
            result = subprocess.run(
                cmd,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout}s"
        except OSError as exc:
            return None, f"could not launch: {exc}"
        seconds = time.monotonic() - start
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        detail = f"exit {result.returncode}"
        if tail:
            detail += ": " + " | ".join(tail)
        return None, detail
    return seconds, "ok"


def append_row(csv_path: Path, row: list[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not csv_path.is_file()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Full chapter runs with wall-clock capture into "
        "infra/ci/reports/wallclock_latest.csv (ci-lanes.md lane 3)."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root (default: auto-detected from this script's location)",
    )
    parser.add_argument("--tier", required=True, choices=TIERS)
    parser.add_argument(
        "--chapters",
        required=True,
        help="all | changed-since-last | scale-labs | <chapter dir>",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed (default 0)")
    parser.add_argument(
        "--timeout",
        type=int,
        default=0,
        help="per-chapter timeout in seconds; 0 = unlimited (default: full "
        "runs are long)",
    )
    parser.add_argument(
        "--out-csv",
        default=None,
        help="output CSV (default: {root}/infra/ci/reports/wallclock_latest.csv)",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="interpreter for artifacts (default: {root}/.venv/bin/python, "
        "else this interpreter)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    python = resolve_python(root, args.python)
    out_csv = (
        Path(args.out_csv)
        if args.out_csv
        else root / "infra" / "ci" / "reports" / "wallclock_latest.csv"
    )
    timeout = args.timeout if args.timeout > 0 else None

    try:
        chapters = discover_chapters(root)
        selected, note = select_chapters(args.chapters, chapters, root)
    except ChapterError as exc:
        print(f"FAIL run_wallclock: {exc}", file=sys.stderr)
        return 1
    print(f"run_wallclock: {note} (tier={args.tier}, seed={args.seed})")

    if not selected:
        print("run_wallclock: no chapters selected — OK (nothing to bench)")
        return 0

    commit = git_short_head(root)
    today = datetime.date.today().isoformat()
    failures: list[str] = []
    for chapter in selected:
        seconds, detail = run_full(python, chapter, root, args.seed, timeout)
        if seconds is None:
            failures.append(f"{chapter.id}: {detail}")
            print(f"FAIL {chapter.id}: {detail}", file=sys.stderr)
            continue
        minutes = round(seconds / 60.0, 2)
        row = [
            chapter.id,
            args.tier,
            f"{minutes}",
            config_hash(chapter.artifact_path, args.tier, args.seed),
            commit,
            today,
            "MEASURED",
        ]
        append_row(out_csv, row)
        print(f"measured {chapter.id}: {minutes} min -> {out_csv}")

    if failures:
        print(
            f"FAIL run_wallclock: {len(failures)}/{len(selected)} chapter(s) "
            "failed; no rows were recorded for failures (ci-lanes.md lane 3):",
            file=sys.stderr,
        )
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print(
        f"run_wallclock: {len(selected)} chapter(s) measured; promote via a "
        "reviewed PR — this script never edits curriculum/common/wallclock.csv"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
