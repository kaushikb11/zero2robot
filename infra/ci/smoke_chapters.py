#!/usr/bin/env python3
"""smoke_chapters — CPU smoke runner with determinism verification.

Contract (ci-lanes.md lane 1, Makefile `make smoke`, root CLAUDE.md invariant
#2): every chapter artifact must complete a tiny deterministic run.

- Selection: --changed-only picks chapters with files changed vs
  merge-base(HEAD, origin/main); young-repo fallback: no origin/main -> HEAD~1;
  no usable history -> ALL chapters. --all overrides and selects everything.
- Per chapter: `{python} {artifact} --smoke --seed {seed} --no-rerun --out
  {tmpdir}` (subprocess, cwd=repo root, 600s timeout). The artifact must exit 0
  and write {out}/metrics.json (json.dump sort_keys=True, floats rounded to 6
  decimals — the shared artifact CLI contract).
- --verify-determinism: run twice into two tmpdirs and byte-compare
  metrics.json; any difference fails (determinism invariant, root CLAUDE.md #2).
- Clear pass/fail table; exit 1 on any failure. No chapters selected = success.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from lib.chapters import Chapter, ChapterError, discover_chapters, select_changed


def resolve_python(root: Path, override: str | None) -> str:
    if override:
        return override
    venv_python = root / ".venv" / "bin" / "python"
    if venv_python.is_file():
        return str(venv_python)
    return sys.executable


def run_smoke(
    python: str,
    chapter: Chapter,
    root: Path,
    seed: int,
    out_dir: Path,
    timeout: int,
) -> tuple[bool, str]:
    """One smoke run. Returns (ok, detail)."""
    artifact = chapter.artifact_path
    if artifact is None or not artifact.is_file():
        return False, (
            f"artifact '{chapter.meta.get('artifact')}' missing "
            "(chapter layout, curriculum/CLAUDE.md)"
        )
    cmd = [
        python,
        str(artifact),
        "--smoke",
        "--seed",
        str(seed),
        "--no-rerun",
        "--out",
        str(out_dir),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout}s"
    except OSError as exc:
        return False, f"could not launch: {exc}"
    if result.returncode != 0:
        tail = (result.stderr or result.stdout).strip().splitlines()[-3:]
        return False, (
            f"exit {result.returncode}: " + " | ".join(tail) if tail
            else f"exit {result.returncode}"
        )
    if not (out_dir / "metrics.json").is_file():
        return False, (
            "exit 0 but no metrics.json written — artifacts must write "
            "{out}/metrics.json under --smoke (shared artifact CLI contract)"
        )
    return True, "ok"


def smoke_chapter(
    python: str,
    chapter: Chapter,
    root: Path,
    seed: int,
    verify_determinism: bool,
    timeout: int,
) -> tuple[bool, str]:
    """Smoke one chapter (twice if verifying determinism)."""
    with tempfile.TemporaryDirectory(prefix="z2r-smoke-") as tmp:
        out_a = Path(tmp) / "run_a"
        ok, detail = run_smoke(python, chapter, root, seed, out_a, timeout)
        if not ok or not verify_determinism:
            return ok, detail
        out_b = Path(tmp) / "run_b"
        ok, detail = run_smoke(python, chapter, root, seed, out_b, timeout)
        if not ok:
            return False, f"second run: {detail}"
        bytes_a = (out_a / "metrics.json").read_bytes()
        bytes_b = (out_b / "metrics.json").read_bytes()
        if bytes_a != bytes_b:
            return False, (
                f"metrics.json differs across two runs at --seed {seed} — "
                "determinism invariant violated (root CLAUDE.md #2)"
            )
        return True, "ok (deterministic x2)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke-run chapter artifacts with optional 2x determinism "
        "verification (ci-lanes.md lane 1)."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root (default: auto-detected from this script's location)",
    )
    parser.add_argument(
        "--changed-only",
        action="store_true",
        help="only chapters changed vs the git base (merge-base with "
        "origin/main; falls back to HEAD~1, then to all)",
    )
    parser.add_argument(
        "--all",
        dest="run_all",
        action="store_true",
        help="smoke every chapter (overrides --changed-only)",
    )
    parser.add_argument("--seed", type=int, default=0, help="seed (default 0)")
    parser.add_argument(
        "--verify-determinism",
        action="store_true",
        help="run each chapter twice and byte-compare metrics.json",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="per-run timeout in seconds (default 600)",
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

    try:
        chapters = discover_chapters(root)
    except ChapterError as exc:
        print(f"FAIL smoke_chapters: {exc}", file=sys.stderr)
        return 1

    if args.run_all or not args.changed_only:
        selected = chapters
        note = f"selected ALL {len(chapters)} chapter(s)"
    else:
        selected, note = select_changed(chapters, root)
    print(f"smoke_chapters: {note}")

    if not selected:
        print("smoke_chapters: no chapters selected — OK (nothing to smoke)")
        return 0

    results: list[tuple[str, bool, str]] = []
    for chapter in selected:
        ok, detail = smoke_chapter(
            python, chapter, root, args.seed, args.verify_determinism,
            args.timeout,
        )
        results.append((chapter.id, ok, detail))

    width = max(len(chapter_id) for chapter_id, _, _ in results)
    print(f"\n{'chapter'.ljust(width)}  result  detail")
    for chapter_id, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"{chapter_id.ljust(width)}  {status}    {detail}")

    failed = [chapter_id for chapter_id, ok, _ in results if not ok]
    if failed:
        print(
            f"\nFAIL smoke_chapters: {len(failed)}/{len(results)} chapter(s) "
            f"failed: {', '.join(failed)} (ci-lanes.md lane 1 blocks merge)",
            file=sys.stderr,
        )
        return 1

    print(f"\nsmoke_chapters: {len(results)} chapter(s) passed — OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
