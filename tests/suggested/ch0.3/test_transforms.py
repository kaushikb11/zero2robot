"""SUGGESTED tests for ch0.3 — transforms.py (humans promote into tests/).

Covers the chapter's CI contract:
- smoke determinism: two --smoke --seed 0 runs produce byte-identical metrics.json
- doctrine mechanics: LOC hard cap, region markers present and balanced
- core correctness: the from-scratch ops match MuJoCo's mju_* to machine precision
- Break It: --break injects a measurably larger error than the correct run
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CHAPTER = REPO / "curriculum/phase0_foundations/ch0.3_transforms"
ARTIFACT = CHAPTER / "transforms.py"

LOC_HARD_CAP = 450
REQUIRED_REGIONS = {"setup", "quaternions", "rotations", "frames", "demo"}
_META = yaml.safe_load((CHAPTER / "meta.yaml").read_text())
CHECKS = _META["exercise_checks"]
SIGS = _META["break_signatures"]
MJU_AGREEMENT_MAX = CHECKS["mju_agreement_max"]
ERROR_KEYS = [
    "quat_multiply_max_err",
    "quat_to_matrix_max_err",
    "rotate_vector_max_err",
    "frame_roundtrip_max_err",
]


def run_smoke(out: Path, *extra: str) -> Path:
    cmd = [sys.executable, str(ARTIFACT), "--smoke", "--seed", "0", "--no-rerun", "--out", str(out), *extra]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, f"artifact failed:\n{result.stderr}"
    metrics = out / "metrics.json"
    assert metrics.is_file(), "smoke run must write metrics.json"
    return metrics


def test_smoke_metrics_byte_identical_across_runs(tmp_path):
    first = run_smoke(tmp_path / "run1").read_bytes()
    second = run_smoke(tmp_path / "run2").read_bytes()
    assert first == second, "same seed, same smoke config -> metrics.json must match byte-for-byte"


def test_metrics_are_sorted(tmp_path):
    metrics = json.loads(run_smoke(tmp_path / "run").read_text())
    assert list(metrics) == sorted(metrics), "metrics.json must be written with sort_keys=True"


def test_loc_cap_and_region_markers():
    lines = ARTIFACT.read_text().splitlines()
    assert len(lines) <= LOC_HARD_CAP, f"artifact is {len(lines)} lines; hard cap is {LOC_HARD_CAP}"

    open_regions, seen = [], set()
    for line in lines:
        start = re.match(r"# --- region: (\w+) ---$", line.strip())
        if start:
            assert not open_regions, f"nested region '{start.group(1)}' inside '{open_regions[-1]}'"
            open_regions.append(start.group(1))
            seen.add(start.group(1))
        elif line.strip() == "# --- endregion ---":
            assert open_regions, "endregion without a matching region marker"
            open_regions.pop()
    assert not open_regions, f"unclosed region(s): {open_regions}"
    assert REQUIRED_REGIONS <= seen, f"missing regions: {REQUIRED_REGIONS - seen}"


def test_from_scratch_math_matches_mujoco(tmp_path):
    metrics = json.loads(run_smoke(tmp_path / "correct").read_text())
    assert metrics["break_mode"] == "none"
    # The core promise: every from-scratch op agrees with mju_* to ~machine
    # epsilon, well under the generous ceiling in meta.yaml.
    for key in ERROR_KEYS:
        assert metrics[key] < MJU_AGREEMENT_MAX, f"{key} = {metrics[key]} exceeds {MJU_AGREEMENT_MAX}"
    assert metrics["break_max_err"] < MJU_AGREEMENT_MAX, "correct run's convention check should agree with MuJoCo"


def test_break_introduces_measurably_larger_error(tmp_path):
    correct = json.loads(run_smoke(tmp_path / "correct").read_text())
    # All three bugs the chapter promises must inject a measurable signature.
    for mode, band in (("quat-convention", SIGS["quat_convention"]),
                       ("compose-order", SIGS["compose_order"]),
                       ("point-vs-frame", SIGS["point_vs_frame"])):
        broken = json.loads(run_smoke(tmp_path / mode, "--break", mode).read_text())
        assert broken["break_mode"] == mode
        # The bug's error dwarfs the correct run's (~machine epsilon) by orders
        # of magnitude, and lands near the reference signature in meta.yaml.
        assert broken["break_max_err"] > 1e-3, f"{mode} should diverge visibly, got {broken['break_max_err']}"
        assert broken["break_max_err"] > correct["break_max_err"] + 1e-3
        assert abs(broken["break_max_err"] - band) < SIGS["band"], f"{mode} signature {broken['break_max_err']} != {band}"
