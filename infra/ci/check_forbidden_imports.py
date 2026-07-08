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
infra/hooks/pedagogy_gate.py. Imports are extracted with `ast` (not a line
regex) so compound (`import numpy, gym`), aliased (`import gym as g`), and
semicolon-joined (`import os; import hydra`) forms cannot slip past.
Exit 1 listing `path:line: module`.
"""

from __future__ import annotations

import argparse
import ast
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
# jax/flax/optax are the ONE excursion (decision 015): allowed only in a chapter that
# declares `allow_mjx: true` (ch2.3 MJX). Everywhere else they are forbidden — this is
# what actually enforces the doctrine's "MJX chapters additionally" carve-out.
FORBIDDEN_UNLESS_MJX = ["jax", "flax", "optax"]

CHAPTER_CODE_RE = re.compile(r"^[a-z0-9_]+\.py$")


def _imported_modules(text: str) -> list[tuple[int, str]]:
    """(lineno, top-level module) for every import, via ast when the file parses.

    ast handles `import a, b`, `import a as x`, `import a.b`, `from a import b`,
    and `import a; import b` uniformly — the top-level name is what the doctrine
    blocks. A file that does not parse falls back to a per-statement scan (split
    on ';' and ',') so a syntax error never silently passes the gate.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        pairs: list[tuple[int, str]] = []
        for lineno, raw in enumerate(text.splitlines(), start=1):
            for stmt in raw.split(";"):
                s = stmt.strip()
                if s.startswith("from "):
                    m = re.match(r"from\s+([\w.]+)", s)
                    if m:
                        pairs.append((lineno, m.group(1).split(".")[0]))
                elif s.startswith("import "):
                    for part in s[len("import "):].split(","):
                        tok = part.strip().split(" as ")[0].strip()
                        if tok:
                            pairs.append((lineno, tok.split(".")[0]))
        return pairs
    pairs = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                pairs.append((node.lineno, alias.name.split(".")[0]))
        elif isinstance(node, ast.ImportFrom) and node.module:
            pairs.append((node.lineno, node.module.split(".")[0]))
    return pairs


def scan_file(path: Path, allow_transformers: bool, allow_mjx: bool = False) -> list[tuple[int, str]]:
    """Return (lineno, module) for every forbidden import in `path`."""
    blocked = set(FORBIDDEN_ALWAYS)
    if not allow_transformers:
        blocked |= set(FORBIDDEN_UNLESS_GRANTED)
    if not allow_mjx:
        blocked |= set(FORBIDDEN_UNLESS_MJX)
    text = path.read_text(encoding="utf-8")
    hits = {(lineno, mod) for lineno, mod in _imported_modules(text) if mod in blocked}
    return sorted(hits)


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
        allow_mjx = bool(chapter.meta.get("allow_mjx"))
        for code_file in chapter_code_files(chapter):
            scanned += 1
            rel = code_file.relative_to(root).as_posix()
            for lineno, module in scan_file(code_file, allow_transformers, allow_mjx):
                if module in FORBIDDEN_UNLESS_GRANTED:
                    violations.append(
                        f"{rel}:{lineno}: '{module}' requires "
                        "`allow_transformers: true` in this chapter's "
                        "meta.yaml (tiny-VLA chapters only)"
                    )
                elif module in FORBIDDEN_UNLESS_MJX:
                    violations.append(
                        f"{rel}:{lineno}: '{module}' requires "
                        "`allow_mjx: true` in this chapter's "
                        "meta.yaml (MJX chapters only — decision 015)"
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
