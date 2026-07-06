"""SUGGESTED tests for ch0.1 — sim_loop.py (humans promote into tests/).

Covers the chapter's CI contract:
- smoke determinism: two --smoke --seed 0 runs produce byte-identical metrics.json
- doctrine mechanics: LOC hard cap, region markers present and balanced
- physics contract: the seeded shove measurably moves the box vs a --no-perturb baseline
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARTIFACT = REPO / "curriculum/phase0_foundations/ch0.1_sim_loop/sim_loop.py"

LOC_HARD_CAP = 450
REQUIRED_REGIONS = {"setup", "scene", "perturb", "loop", "inspect"}


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


def test_metrics_are_sorted_and_rounded(tmp_path):
    metrics = json.loads(run_smoke(tmp_path / "run").read_text())
    assert list(metrics) == sorted(metrics), "metrics.json must be written with sort_keys=True"
    floats = [metrics["box_final_speed"], metrics["lateral_drift"], *metrics["box_final_pos"]]
    for value in floats:
        assert value == round(value, 6), f"float {value} not rounded to 6 decimals"


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


def test_perturbation_moves_the_box(tmp_path):
    shoved = json.loads(run_smoke(tmp_path / "shoved").read_text())
    baseline = json.loads(run_smoke(tmp_path / "baseline", "--no-perturb").read_text())

    assert shoved["perturb"] is True and baseline["perturb"] is False
    assert shoved["shove_moved_box"] is True, "seeded shove should displace the box"
    assert baseline["shove_moved_box"] is False, "baseline run should show no lateral drift"
    lateral_gap = abs(shoved["box_final_pos"][1] - baseline["box_final_pos"][1])
    assert lateral_gap > 0.01, f"shove changed final y by only {lateral_gap} m vs baseline"
