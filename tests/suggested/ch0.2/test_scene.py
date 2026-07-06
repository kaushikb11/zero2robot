"""SUGGESTED tests for ch0.2 — scene.py (humans promote into tests/).

Covers the chapter's CI contract:
- smoke determinism: two --smoke --seed 0 runs produce byte-identical metrics.json
- doctrine mechanics: LOC hard cap, region markers present and balanced
- scene structure: the MJCF compiles to the expected body/joint/DOF counts
- Break It: --break split-tee measurably changes both structure and behaviour
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARTIFACT = REPO / "curriculum/phase0_foundations/ch0.2_scene/scene.py"

LOC_HARD_CAP = 450
REQUIRED_REGIONS = {"setup", "ground", "tee", "pusher", "target", "build"}


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
    floats = [metrics["tee_gap_max_dev"], metrics["tee_gap_settled"],
              *metrics["tee_final_pose"], *metrics["tee_settled_pose"], *metrics["pusher_final"]]
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


def test_scene_compiles_to_expected_kinematic_tree(tmp_path):
    metrics = json.loads(run_smoke(tmp_path / "run").read_text())
    # The PushT scene: table + walls + target + welded T + pusher.
    assert metrics["n_bodies"] == 3, "target, tee, pusher hang off worldbody"
    assert metrics["n_joints"] == 5, "3 planar block joints + 2 pusher slides"
    assert metrics["nq"] == 5 and metrics["nv"] == 5, "5 DOF, no quaternion in the tree"
    assert metrics["nu"] == 2, "two velocity actuators on the pusher"
    assert metrics["joint_types"] == ["slide", "slide", "hinge", "slide", "slide"]


def test_welded_tee_moves_as_one_rigid_body(tmp_path):
    metrics = json.loads(run_smoke(tmp_path / "run").read_text())
    # The weld invariant: bar-stem gap is a constant 0.06 m no matter the push, so
    # it never deviates from 0.06 anywhere in the run — that whole-run peak is the
    # rigidity verdict, not the fragile final gap.
    assert metrics["tee_stayed_rigid"] is True
    assert metrics["tee_gap_settled"] == 0.06, "welded T must hold a fixed 0.06 m bar-stem gap"
    assert metrics["tee_gap_max_dev"] == 0.0, "welded T's gap must never leave 0.06 across the whole run"
    # The pusher is actuated: it drove the block a long way north.
    assert metrics["tee_final_pose"][1] > metrics["tee_settled_pose"][1] + 0.2


def test_break_split_tee_changes_structure_and_behaviour(tmp_path):
    welded = json.loads(run_smoke(tmp_path / "welded").read_text())
    split = json.loads(run_smoke(tmp_path / "split", "--break", "split-tee").read_text())
    # Structure: the split T is a second body with its own 3 joints.
    assert split["n_bodies"] == welded["n_bodies"] + 1
    assert split["n_joints"] == welded["n_joints"] + 3
    assert split["nq"] == 8 and split["nv"] == 8
    # Behaviour: the two halves are never rigidly attached. The robust tell is the
    # SETTLED gap and the whole-run PEAK deviation from 0.06 — both far off 0.06
    # with a large margin (the final gap is NOT reliable: the loose halves can
    # drift back near 0.06 under the push, which is why the verdict ignores it).
    assert split["tee_stayed_rigid"] is False
    assert abs(split["tee_gap_settled"] - 0.06) > 0.02, "split T cannot even sit still as one piece"
    assert split["tee_gap_max_dev"] > 0.02, "split T's gap deviates far from the 0.06 weld at its peak"
    # The margin over the welded verdict (max_dev 0.0) is ~300x the 1e-4 tolerance.
    assert split["tee_gap_max_dev"] > welded["tee_gap_max_dev"] + 0.02
