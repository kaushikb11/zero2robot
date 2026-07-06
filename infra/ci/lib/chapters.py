"""Chapter discovery and meta.yaml parsing shared by the infra/ci gates.

Contract (curriculum/CLAUDE.md, infra/decisions/003-repo-layout.md):
- A chapter lives at curriculum/phase*/ch*/ and is identified by its meta.yaml.
- meta.yaml keys used by the gates: id, artifact, allow_transformers,
  notebook_hash, region_hashes, scale_lab / scale_lab_ref.
- The artifact is the single runnable .py file named by meta.yaml `artifact:`,
  located in the chapter directory.

Also provides git-based changed-chapter selection (read-only `git diff` /
`git ls-files`; never mutates git state).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml


class ChapterError(Exception):
    """A chapter tree violates the layout contract (curriculum/CLAUDE.md)."""


@dataclass
class Chapter:
    """One discovered chapter: its directory and parsed meta.yaml."""

    directory: Path  # absolute path to the chapter directory
    meta_path: Path
    meta: dict

    @property
    def id(self) -> str:
        """Chapter id from meta.yaml, falling back to the directory name."""
        return str(self.meta.get("id") or self.directory.name)

    @property
    def artifact_path(self) -> Path | None:
        """Absolute path to the artifact named by meta.yaml, or None if unset."""
        artifact = self.meta.get("artifact")
        if not artifact:
            return None
        return self.directory / str(artifact)

    def rel(self, root: Path) -> str:
        """Chapter directory relative to the repo root (posix), for messages."""
        try:
            return self.directory.relative_to(root).as_posix()
        except ValueError:
            return self.directory.as_posix()


def load_chapter(chapter_dir: Path) -> Chapter:
    """Load one chapter from its directory. Raises ChapterError on bad meta.yaml."""
    meta_path = chapter_dir / "meta.yaml"
    try:
        raw = meta_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ChapterError(f"{meta_path}: cannot read meta.yaml: {exc}") from exc
    try:
        meta = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise ChapterError(f"{meta_path}: invalid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise ChapterError(
            f"{meta_path}: meta.yaml must be a mapping (got {type(meta).__name__}); "
            "see curriculum/CLAUDE.md chapter layout"
        )
    return Chapter(directory=chapter_dir, meta_path=meta_path, meta=meta)


def discover_chapters(root: Path) -> list[Chapter]:
    """Find every chapter under {root}/curriculum/phase*/ch*/ with a meta.yaml.

    Raises ChapterError if any meta.yaml is unreadable or malformed — a broken
    meta.yaml must fail the gates loudly, never be skipped.
    """
    curriculum = root / "curriculum"
    if not curriculum.is_dir():
        return []
    chapters = []
    for meta_path in sorted(curriculum.glob("phase*/ch*/meta.yaml")):
        chapters.append(load_chapter(meta_path.parent))
    return chapters


def _git(args: list[str], cwd: Path) -> str | None:
    """Run a read-only git command; return stdout, or None on any failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout


def changed_paths(root: Path) -> list[str] | None:
    """Repo-relative paths changed vs merge-base(HEAD, origin/main).

    Fallback chain for a young repo: no origin/main -> diff against HEAD~1;
    no usable git history at all -> None (caller should select everything).
    Untracked files are included (a brand-new chapter must be selected).
    """
    base = _git(["merge-base", "HEAD", "origin/main"], root)
    ref = base.strip() if base and base.strip() else "HEAD~1"
    diff = _git(["diff", "--name-only", ref], root)
    if diff is None:
        return None
    paths = [line.strip() for line in diff.splitlines() if line.strip()]
    untracked = _git(["ls-files", "--others", "--exclude-standard"], root) or ""
    paths += [line.strip() for line in untracked.splitlines() if line.strip()]
    return paths


def select_changed(chapters: list[Chapter], root: Path) -> tuple[list[Chapter], str]:
    """Filter chapters to those with files changed vs the git base.

    Returns (selected, note). If git history is unusable, returns ALL chapters
    (the safe direction for a gate) with a note saying so.
    """
    paths = changed_paths(root)
    if paths is None:
        return list(chapters), (
            "note: git history unusable (no origin/main and no HEAD~1); "
            "falling back to ALL chapters"
        )
    selected = []
    for chapter in chapters:
        prefix = chapter.rel(root) + "/"
        if any(p == prefix.rstrip("/") or p.startswith(prefix) for p in paths):
            selected.append(chapter)
    return selected, (
        f"selected {len(selected)}/{len(chapters)} chapters changed vs git base"
    )
