"""SUGGESTED (human-owned tests/ — promote if wanted): common/ fan-out gate.

Covers the fix for the deep-review "gate gap": a change under
curriculum/common/ (shared envs/seeding/device) touches no chapter directory,
so per-directory selection would run NO chapters and let a determinism-breaking
shared change bypass the smoke + determinism check (root CLAUDE.md #2).
`select_changed` must instead fan out to ALL chapters on any common/ change.

These tests monkeypatch `changed_paths` so they exercise the selection logic
directly, without standing up a git repo. Mirrors the conftest `make_chapter`
fixture already used by the infra/ci gate tests.
"""

from __future__ import annotations

from lib import chapters as chapters_mod
from lib.chapters import discover_chapters, select_changed


def _discover(make_chapter):
    make_chapter(dirname="ch1.1_bc", chapter_id="ch1.1-bc", artifact_name="bc.py")
    make_chapter(
        dirname="ch1.2_act",
        chapter_id="ch1.2-act",
        artifact_name="act.py",
    )
    root = make_chapter(
        dirname="ch3.6_compare",
        chapter_id="ch3.6-compare",
        phase="phase3_advanced",
        artifact_name="compare.py",
    ).parents[2]
    return root, discover_chapters(root)


def test_common_change_selects_all_chapters(make_chapter, monkeypatch):
    root, discovered = _discover(make_chapter)
    monkeypatch.setattr(
        chapters_mod,
        "changed_paths",
        lambda _root: ["curriculum/common/seeding.py"],
    )
    selected, note = select_changed(discovered, root)
    assert len(selected) == len(discovered) == 3
    assert "curriculum/common/" in note and "ALL" in note


def test_common_dir_itself_selects_all(make_chapter, monkeypatch):
    root, discovered = _discover(make_chapter)
    monkeypatch.setattr(
        chapters_mod, "changed_paths", lambda _root: ["curriculum/common"]
    )
    selected, _ = select_changed(discovered, root)
    assert len(selected) == len(discovered)


def test_single_chapter_change_selects_only_that_chapter(
    make_chapter, monkeypatch
):
    root, discovered = _discover(make_chapter)
    monkeypatch.setattr(
        chapters_mod,
        "changed_paths",
        lambda _root: ["curriculum/phase1_imitation/ch1.1_bc/bc.py"],
    )
    selected, _ = select_changed(discovered, root)
    assert [c.id for c in selected] == ["ch1.1-bc"]


def test_unrelated_change_selects_nothing(make_chapter, monkeypatch):
    # A change outside any chapter and outside common/ (e.g. site/, infra/)
    # selects no chapters — the common/ fan-out must not over-select.
    root, discovered = _discover(make_chapter)
    monkeypatch.setattr(
        chapters_mod, "changed_paths", lambda _root: ["site/pages/index.astro"]
    )
    selected, _ = select_changed(discovered, root)
    assert selected == []
