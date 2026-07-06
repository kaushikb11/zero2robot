"""SUGGESTED tests for ch1.1 — bc.py (humans promote into tests/).

Covers the chapter's CI contract:
- smoke determinism: two --smoke --seed 0 runs produce byte-identical metrics.json
- doctrine mechanics: LOC hard cap, region markers present and balanced
- artifact pipeline: ONNX exists after smoke, carries tensor contract v1
  metadata (obs 10 / act 2), and the in-run parity gate passed
- learning sanity: a small non-smoke run measurably reduces train loss
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARTIFACT = REPO / "curriculum/phase1_imitation/ch1.1_bc/bc.py"

LOC_HARD_CAP = 450
REQUIRED_REGIONS = {"setup", "data", "model", "train", "eval"}


def run_smoke(out: Path, *extra: str) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(ARTIFACT), "--smoke", "--seed", "0", "--no-rerun",
           "--out", str(out), *extra]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, f"artifact failed:\n{result.stderr}"
    assert (out / "metrics.json").is_file(), "smoke run must write metrics.json"
    return result


def test_smoke_metrics_byte_identical_across_runs(tmp_path):
    run_smoke(tmp_path / "run1")
    run_smoke(tmp_path / "run2")
    first = (tmp_path / "run1" / "metrics.json").read_bytes()
    second = (tmp_path / "run2" / "metrics.json").read_bytes()
    assert first == second, "same seed, same smoke config -> metrics.json must match byte-for-byte"


def test_metrics_are_sorted_rounded_and_sane(tmp_path):
    run_smoke(tmp_path / "run")
    metrics = json.loads((tmp_path / "run" / "metrics.json").read_text())
    assert list(metrics) == sorted(metrics), "metrics.json must be written with sort_keys=True"
    for key in ("final_train_loss", "final_val_loss", "mean_episode_return",
                "parity_delta", "success_rate"):
        assert metrics[key] == round(metrics[key], 6), f"{key} not rounded to 6 decimals"
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert metrics["final_train_loss"] > 0.0
    # assert_parity gates at 1e-4 inside the run; the recorded delta must agree
    assert metrics["parity_delta"] < 1e-4


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


def test_onnx_export_carries_tensor_contract_v1(tmp_path):
    run_smoke(tmp_path / "run")
    onnx_path = tmp_path / "run" / "bc_policy.onnx"
    assert onnx_path.is_file(), "smoke run must export bc_policy.onnx"
    assert (tmp_path / "run" / "bc_policy.pt").is_file(), "smoke run must save the torch checkpoint"

    import onnx

    proto = onnx.load(str(onnx_path))
    meta = {p.key: p.value for p in proto.metadata_props}
    assert meta.get("z2r_contract_version") == "v1"
    assert meta.get("z2r_obs_dim") == "10"
    assert meta.get("z2r_act_dim") == "2"
    graph_input = proto.graph.input[0]
    dims = [d.dim_value for d in graph_input.type.tensor_type.shape.dim]
    assert graph_input.name == "observation" and dims == [1, 10]


def test_training_reduces_loss(tmp_path):
    # Reuse the hermetic smoke dataset, then train a slightly longer non-smoke
    # run on it: loss at the last epoch must sit well below epoch 0's.
    # Measured 2026-07-03 (cpu, seed 0): 0.103 -> 0.076 (ratio 0.74); the tiny
    # 6-episode dataset floors quickly, hence the soft 0.85 threshold.
    run_smoke(tmp_path / "seed_run")
    data = tmp_path / "seed_run" / "smoke-demos"
    cmd = [sys.executable, str(ARTIFACT), "--data", str(data), "--epochs", "60",
           "--hidden_dim", "256", "--eval_episodes", "2", "--seed", "0",
           "--no-rerun", "--out", str(tmp_path / "train_run")]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert result.returncode == 0, f"artifact failed:\n{result.stderr}"
    losses = [float(m) for m in re.findall(r"train_loss (\d+\.\d+)", result.stdout)]
    assert len(losses) >= 2, f"expected epoch progress lines in stdout:\n{result.stdout}"
    assert min(losses) < 0.85 * losses[0], (
        f"60 epochs should measurably cut train loss: first {losses[0]}, best {min(losses)}"
    )
