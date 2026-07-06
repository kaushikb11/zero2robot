"""Submission schema: divisions, free-tier plausibility, percentile bands."""

from __future__ import annotations

from pathlib import Path

from grader.submission import (
    Division,
    Submission,
    free_tier_declaration_is_plausible,
    percentile_band,
)


def test_division_is_first_class():
    assert Division("free") is Division.FREE
    assert Division("open") is Division.OPEN
    assert {d.value for d in Division} == {"free", "open"}


def test_free_tier_plausibility():
    ok = Submission(Path("x.onnx"), "c", Division.FREE, declared_runtime_min=30.0)
    too_long = Submission(Path("x.onnx"), "c", Division.FREE, declared_runtime_min=10_000.0)
    open_div = Submission(Path("x.onnx"), "c", Division.OPEN, declared_runtime_min=10_000.0)
    unknown = Submission(Path("x.onnx"), "c", Division.FREE)
    assert free_tier_declaration_is_plausible(ok) is True
    assert free_tier_declaration_is_plausible(too_long) is False
    assert free_tier_declaration_is_plausible(open_div) is None  # open: unknowable
    assert free_tier_declaration_is_plausible(unknown) is None   # not declared


def test_percentile_bands():
    assert percentile_band(50.0, []) == "unranked (no cohort yet)"
    cohort = [0.0, 25.0, 50.0, 75.0, 100.0]
    assert percentile_band(100.0, cohort) == "top 10%"
    assert percentile_band(0.0, cohort) == "bottom 25%"
