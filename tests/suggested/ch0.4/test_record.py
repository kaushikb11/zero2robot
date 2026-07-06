"""SUGGESTED tests for ch0.4 — record.py (humans promote into tests/).

Covers the chapter's CI contract:
- smoke determinism: two --smoke --seed 0 runs produce byte-identical metrics.json
- doctrine mechanics: LOC hard cap, region markers present and balanced
- GOLDEN PARITY (the "pin the format on both sides" test): record.py's --smoke
  dataset is schema-identical to a gen_demos.py dataset — same features, dtypes,
  shapes, meta structure, parquet schemas, and CODEBASE_VERSION. Because both
  write through the SAME pinned lerobot path, this MUST hold; the test is the
  tripwire that catches any drift. Episode/frame COUNTS legitimately differ and
  are normalized out (per decision 008); only the FORMAT is compared.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq
import pytest

os.environ.setdefault("HF_HUB_OFFLINE", "1")  # lerobot writes fully offline

REPO = Path(__file__).resolve().parents[3]
CHAPTER = REPO / "curriculum/phase0_foundations/ch0.4_record"
ARTIFACT = CHAPTER / "record.py"

LOC_HARD_CAP = 450
REQUIRED_REGIONS = {"setup", "features", "teleop", "ingest", "write", "run"}


def run_smoke(out: Path, *extra: str) -> Path:
    """Record a tiny deterministic dataset; return the run dir (out)."""
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    cmd = [sys.executable, str(ARTIFACT), "--smoke", "--seed", "0", "--no-rerun", "--out", str(out), *extra]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, env=env)
    assert result.returncode == 0, f"artifact failed:\n{result.stderr}"
    assert (out / "metrics.json").is_file(), "smoke run must write metrics.json"
    assert (out / "dataset" / "meta" / "info.json").is_file(), "smoke run must write a v3 dataset under {out}/dataset"
    return out


# ------------------------------------------------------- smoke determinism
def test_smoke_metrics_byte_identical_across_runs(tmp_path):
    first = (run_smoke(tmp_path / "run1") / "metrics.json").read_bytes()
    second = (run_smoke(tmp_path / "run2") / "metrics.json").read_bytes()
    assert first == second, "same seed, same smoke config -> metrics.json must match byte-for-byte"


def test_metrics_are_sorted_and_rounded(tmp_path):
    metrics = json.loads((run_smoke(tmp_path / "run") / "metrics.json").read_text())
    assert list(metrics) == sorted(metrics), "metrics.json must be written with sort_keys=True"
    floats = [*metrics["first_obs"], *metrics["last_obs"], *metrics["first_action"], *metrics["last_action"]]
    for value in floats:
        assert value == round(value, 6), f"float {value} not rounded to 6 decimals"


def test_metrics_report_the_episode_layout(tmp_path):
    metrics = json.loads((run_smoke(tmp_path / "run") / "metrics.json").read_text())
    assert metrics["source"] == "local-teleop"
    assert metrics["obs_dim"] == 10 and metrics["act_dim"] == 2
    assert metrics["feature_keys"] == ["action", "observation.state"]
    assert metrics["n_frames"] == metrics["n_episodes"] * 40  # fixed smoke episode length
    assert len(metrics["first_obs"]) == 10 and len(metrics["first_action"]) == 2


# ------------------------------------------------------- doctrine mechanics
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


# --------------------------------------------------------- golden parity
def _parquet_schema(path: Path) -> list:
    """Ordered (name, arrow-type-str) pairs — the on-disk column contract."""
    return [(f.name, str(f.type)) for f in pq.read_schema(path)]


def _fingerprint(root: Path) -> dict:
    """FORMAT-only fingerprint (schemas, dtypes, shapes, structural keys) with
    chunk/file indices normalized, so a different episode count never trips it.
    The format-diff approach was proven by the Phase-0 spike (decision 008)."""
    info = json.loads((root / "meta/info.json").read_text())
    features = {name: {"dtype": f["dtype"], "shape": list(f["shape"]), "names": f["names"]}
                for name, f in info["features"].items()}
    stems = sorted({
        re.sub(r"(chunk|file)-\d+", r"\1-*", str(p.relative_to(root)))
        for p in root.rglob("*") if p.is_file() and p.suffix in {".json", ".parquet"}
    })
    return {
        "codebase_version": info["codebase_version"],
        "robot_type": info["robot_type"],
        "fps": info["fps"],
        "features": features,
        "data_path": info["data_path"],
        "video_path": info["video_path"],
        "info_keys": sorted(info.keys()),
        "stats_keys": sorted(json.loads((root / "meta/stats.json").read_text()).keys()),
        "meta_file_stems": stems,
        "data_parquet_schema": _parquet_schema(root / "data/chunk-000/file-000.parquet"),
        "episodes_parquet_schema": _parquet_schema(next((root / "meta/episodes").rglob("*.parquet"))),
        "tasks_parquet_schema": _parquet_schema(root / "meta/tasks.parquet"),
    }


@pytest.fixture(scope="module")
def gen_demos_ref(tmp_path_factory) -> Path:
    """A reference dataset from the scripted expert — the format record.py must match."""
    from curriculum.common.envs.pusht import gen_demos
    out = tmp_path_factory.mktemp("ref") / "pusht_ref"
    gen_demos.main(["--episodes", "2", "--seed", "0", "--out", str(out), "--no-video"])
    return out


def test_record_local_is_schema_identical_to_gen_demos(tmp_path, gen_demos_ref):
    dataset = run_smoke(tmp_path / "rec") / "dataset"
    # (a) it must LOAD via the pinned lerobot loader.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    loaded = LeRobotDataset(repo_id="zero2robot/pusht_teleop", root=dataset)
    assert loaded.num_frames > 0
    sample = loaded[0]
    assert tuple(sample["observation.state"].shape) == (10,)
    assert tuple(sample["action"].shape) == (2,)

    # (b) format fingerprints must be identical (counts normalized out).
    ref, cand = _fingerprint(gen_demos_ref), _fingerprint(dataset)
    assert cand["codebase_version"] == "v3.0"
    assert cand == ref, (
        "record.py output drifted from the gen_demos v3 schema:\n"
        + "\n".join(f"  {k}: ref={ref[k]!r} cand={cand[k]!r}" for k in sorted(set(ref) | set(cand)) if ref.get(k) != cand.get(k))
    )


# ------------------------------------------ interchange video ingest (tripwire)
def _write_gradient_png(path: Path, hw: int = 96):
    """A real, non-solid HxWx3 frame — the case that must survive ingest."""
    import numpy as np
    from PIL import Image

    yy, xx = np.mgrid[0:hw, 0:hw].astype(np.uint8)
    arr = np.stack([xx * 2, yy * 2, np.full((hw, hw), 128, np.uint8)], axis=-1)
    Image.fromarray(arr, "RGB").save(path)


def test_interchange_video_ingest_preserves_image_content(tmp_path):
    """--from-interchange must pass a real browser frame through, not flatten it.

    Regression tripwire: record.py once broadcast the top-left pixel over the
    whole frame (correct only for the 1x1 reference stand-in), silently
    collapsing every real 96x96 recording to a solid color."""
    pytest.importorskip("PIL")
    pytest.importorskip("av")  # lerobot video encode/decode
    import numpy as np

    bundle = tmp_path / "bundle"
    (bundle / "frames" / "ep0").mkdir(parents=True)
    _write_gradient_png(bundle / "frames" / "ep0" / "f0.png")
    _write_gradient_png(bundle / "frames" / "ep0" / "f1.png")
    manifest = {
        "interchange_version": "z2r-teleop-1",
        "repo_id": "zero2robot/pusht_teleop_test",
        "robot_type": "pusher_2d",
        "fps": 10,
        "task": "Push the T-shaped block to the target pose.",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [10], "names": [f"s{i}" for i in range(10)]},
            "action": {"dtype": "float32", "shape": [2], "names": ["pusher_vx", "pusher_vy"]},
            "observation.image": {"dtype": "video", "shape": [96, 96, 3], "names": ["height", "width", "channel"]},
        },
        "episodes": [{
            "length": 2,
            "observation.state": [[0.0] * 10, [0.1] * 10],
            "action": [[0.0, 0.0], [0.1, -0.1]],
            "timestamp": [0.0, 0.1],
            "observation.image": ["frames/ep0/f0.png", "frames/ep0/f1.png"],
        }],
    }
    (bundle / "interchange.json").write_text(json.dumps(manifest))

    out = tmp_path / "out"
    env = {**os.environ, "HF_HUB_OFFLINE": "1"}
    result = subprocess.run(
        [sys.executable, str(ARTIFACT), "--from-interchange", str(bundle), "--out", str(out), "--no-rerun"],
        capture_output=True, text=True, cwd=REPO, env=env)
    assert result.returncode == 0, f"interchange ingest failed:\n{result.stderr}"

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(repo_id=manifest["repo_id"], root=out / "dataset")
    image = np.asarray(ds[0]["observation.image"])
    assert np.unique(image).size > 10, (
        f"observation.image ingested as {np.unique(image).size} unique value(s) — a real "
        "frame was flattened to a solid color (record.py broadcast bug)")
