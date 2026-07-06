#!/usr/bin/env python3
"""check_forbidden_imports — forbidden-import gate for chapter code.

Contract (curriculum/CLAUDE.md doctrine #3): chapter code may not import
framework/abstraction layers that hide the lesson. Blocked at ANY indentation
(top-level or function-level `import X` / `from X import ...`):

- always: hydra, omegaconf, pytorch_lightning, stable_baselines3, gym, gymnasium
- transformers: blocked UNLESS the chapter's meta.yaml grants
  `allow_transformers: true` (tiny-VLA chapters only)

Scans every lowercase .py file directly inside each chapter directory (the
artifact plus any sibling chapter code), matching the session-time twin
infra/hooks/pedagogy_gate.py: regex `^\\s*(import|from)\\s+<mod>\\b` per line.
Exit 1 listing `path:line: module`.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from lib.chapters import Chapter, ChapterError, discover_chapters

FORBIDDEN_ALWAYS = [
    "hydra",
    "omegaconf",
    "pytorch_lightning",
    "stable_baselines3",
    "gymnasium",
    "gym",
]
FORBIDDEN_UNLESS_GRANTED = ["transformers"]

CHAPTER_CODE_RE = re.compile(r"^[a-z0-9_]+\.py$")


def _import_re(module: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*(import|from)\s+{module}\b")


def scan_file(path: Path, allow_transformers: bool) -> list[tuple[int, str]]:
    """Return (lineno, module) for every forbidden import in `path`."""
    blocked = list(FORBIDDEN_ALWAYS)
    if not allow_transformers:
        blocked += FORBIDDEN_UNLESS_GRANTED
    patterns = [(module, _import_re(module)) for module in blocked]
    hits: list[tuple[int, str]] = []
    text = path.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        for module, pattern in patterns:
            if pattern.match(line):
                hits.append((lineno, module))
    return hits


def chapter_code_files(chapter: Chapter) -> list[Path]:
    """Chapter code .py files, matching the pedagogy_gate scope: the artifact
    plus everything under the chapter dir (exercises/**, demo/**) EXCEPT
    human-owned tests/. Filenames may contain digits (rl_v2.py, ex1_gym.py)."""
    return sorted(
        p for p in chapter.directory.rglob("*.py")
        if p.is_file()
        and CHAPTER_CODE_RE.match(p.name)
        and "tests" not in p.relative_to(chapter.directory).parts
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Block forbidden framework imports in chapter code "
        "(curriculum/CLAUDE.md doctrine #3)."
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[2]),
        help="repo root (default: auto-detected from this script's location)",
    )
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()

    try:
        chapters = discover_chapters(root)
    except ChapterError as exc:
        print(f"FAIL check_forbidden_imports: {exc}", file=sys.stderr)
        return 1

    if not chapters:
        print("check_forbidden_imports: no chapters discovered — OK")
        return 0

    violations: list[str] = []
    scanned = 0
    for chapter in chapters:
        allow_transformers = bool(chapter.meta.get("allow_transformers"))
        for code_file in chapter_code_files(chapter):
            scanned += 1
            rel = code_file.relative_to(root).as_posix()
            for lineno, module in scan_file(code_file, allow_transformers):
                if module in FORBIDDEN_UNLESS_GRANTED:
                    violations.append(
                        f"{rel}:{lineno}: '{module}' requires "
                        "`allow_transformers: true` in this chapter's "
                        "meta.yaml (tiny-VLA chapters only)"
                    )
                else:
                    violations.append(f"{rel}:{lineno}: '{module}'")

    if violations:
        print(
            f"FAIL check_forbidden_imports: {len(violations)} forbidden "
            "import(s) in chapter code (pedagogical doctrine, "
            "curriculum/CLAUDE.md #3):",
            file=sys.stderr,
        )
        for violation in violations:
            print(f"  {violation}", file=sys.stderr)
        return 1

    print(
        f"check_forbidden_imports: {scanned} chapter code file(s) clean — OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
