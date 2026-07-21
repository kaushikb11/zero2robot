"""The exercise auto-checker runs a real chapter's PUBLIC checks and reports
pass/skip/fail with the meta bands + provenance it verified against."""

from __future__ import annotations

import sys

import pytest

from grader.check import (
    _exercise_id,
    _exercise_type,
    find_checks,
    run_chapter_checks,
)

# These two tests assert ch0.1's PUBLIC checks report all-green. ch0.1's checks
# are physics-threshold assertions on the MuJoCo sim loop whose reference values
# were recorded on macOS arm64; on Linux x86_64 (CI, Colab) MuJoCo's floating
# point diverges enough to tip a threshold, so the checker correctly reports a
# failure and these grader-level tests go red. This is the same cross-platform
# determinism gap as tests/envs/test_dataset_golden.py -- gate to the recording
# platform until ch0.1's answer keys are regenerated on Linux. The grader's own
# machinery (find/id/type/serialize shape) stays covered off-platform by the
# other tests here. TODO(ci-platform-goldens): infra/decisions/020-ci-platform-goldens.md.
skip_offplatform_golden = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="ch0.1 check thresholds recorded on macOS arm64; MuJoCo FP differs on "
    "Linux x86_64 -- regenerate answer keys on Linux (020-ci-platform-goldens.md)",
)


def test_finds_ch01_by_token():
    path = find_checks("0.1")
    assert path.name == "checks.py"
    assert "ch0.1_sim_loop" in str(path)
    # "ch0.1" and "0.1" resolve to the same file.
    assert find_checks("ch0.1") == path


def test_unknown_chapter_raises():
    with pytest.raises(FileNotFoundError):
        find_checks("9.9")


def test_exercise_id_from_nodeid():
    assert _exercise_id("checks.py::test_ex2_rollout_succeeds") == "ex2"
    assert _exercise_id("checks.py::test_setup_only") == ""


@skip_offplatform_golden
def test_ch01_checks_report_green():
    report = run_chapter_checks("0.1")
    # ch0.1 ships 2 deterministic checks that PASS out of the box + 3 that SKIP
    # by design (prediction gates, bug-hunt-still-buggy). Green == nothing failed.
    assert report.collection_error == "", report.collection_error
    assert report.failed == 0
    assert report.errors == 0
    assert report.passed >= 1
    assert report.skipped >= 1
    assert report.ok is True


def test_predict_then_run_gate_skips_gracefully():
    # ch1.1's fast lane is ALL predict-then-run / code-completion gates: with the
    # exercises unsolved they SKIP (never FAIL), and the chapter is still "OK".
    report = run_chapter_checks("1.1")
    assert report.failed == 0 and report.errors == 0
    assert report.skipped >= 1
    assert report.ok is True
    # the predict-then-run gate reports a human-readable reason, not a traceback.
    gate = next(o for o in report.outcomes if o.nodeid.endswith("test_ex4_prediction_recorded"))
    assert gate.outcome == "skipped"
    assert "PREDICTION" in gate.message


def test_fast_lane_deselects_slow_checks():
    # The default fast lane must NOT run @slow training checks; ch1.1 has several.
    fast = run_chapter_checks("1.1")
    assert fast.deselected >= 1
    # opting into slow surfaces more tests (and deselects none).
    # (kept off by default in CI via the module-level `slow` marker on ch1.1.)


def test_bands_and_types_attached_from_meta():
    report = run_chapter_checks("0.1")
    # ex2 is a bug-hunt whose meta carries the buggy signature + seeded bands.
    assert report.types.get("ex2") == "bug-hunt"
    assert "ex2" in report.bands
    block = report.bands["ex2"]
    assert "buggy_lateral_drift" in block  # the measured reference signature
    assert block.get("provenance"), "a band must ship its provenance"


def test_exercise_type_reads_metadata_and_docstring():
    path = find_checks("1.1")
    suggested = path.parent
    # ex2 declares its type in a METADATA dict; ex1 (no METADATA) via docstring.
    assert _exercise_type(suggested, "ex2") == "code-completion"
    assert _exercise_type(suggested, "ex1") == "bug-hunt"
    assert _exercise_type(suggested, "ex99") == ""  # no such exercise


@skip_offplatform_golden
def test_report_serializes():
    report = run_chapter_checks("0.1")
    d = report.to_dict()
    assert d["summary"]["ok"] is True
    assert d["summary"]["passed"] == report.passed
    assert "deselected" in d["summary"]
    assert {o["outcome"] for o in d["outcomes"]} <= {"passed", "failed", "skipped"}
    # every outcome carries its exercise slug for grouping.
    assert all("exercise" in o for o in d["outcomes"])
    assert d["bands"]["ex2"]["provenance"]
