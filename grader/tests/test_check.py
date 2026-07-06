"""The exercise auto-checker runs a real chapter's checks green."""

from __future__ import annotations

import pytest

from grader.check import find_checks, run_chapter_checks


def test_finds_ch01_by_token():
    path = find_checks("0.1")
    assert path.name == "checks.py"
    assert "ch0.1_sim_loop" in str(path)
    # "ch0.1" and "0.1" resolve to the same file.
    assert find_checks("ch0.1") == path


def test_unknown_chapter_raises():
    with pytest.raises(FileNotFoundError):
        find_checks("9.9")


def test_ch01_checks_report_green():
    report = run_chapter_checks("0.1")
    # ch0.1 ships 2 passing checks + 3 that SKIP by design (prediction gates,
    # bug-hunt-still-buggy). Green == nothing failed.
    assert report.collection_error == "", report.collection_error
    assert report.failed == 0
    assert report.passed >= 1
    assert report.skipped >= 1
    assert report.ok is True


def test_report_serializes():
    report = run_chapter_checks("0.1")
    d = report.to_dict()
    assert d["summary"]["ok"] is True
    assert d["summary"]["passed"] == report.passed
    assert {o["outcome"] for o in d["outcomes"]} <= {"passed", "failed", "skipped"}
