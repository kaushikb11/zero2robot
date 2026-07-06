"""Shared fixtures for the infra/ci gate tests.

Everything runs against tiny synthetic chapter trees built in tmp_path — never
against the real curriculum/ tree (which is being built in parallel).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

CI_DIR = Path(__file__).resolve().parents[1]
if str(CI_DIR) not in sys.path:
    sys.path.insert(0, str(CI_DIR))


# A minimal artifact honoring the shared chapter CLI contract:
# --smoke --seed INT --out DIR --no-rerun; writes {out}/metrics.json with
# sorted keys and 6-decimal floats; byte-identical across runs at a seed.
DETERMINISTIC_ARTIFACT = """\
import argparse, json, os

# --- region: model ---
def loss_for_seed(seed):
    return round(1.0 / (seed + 2), 6)
# --- endregion ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-rerun", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    metrics = {"loss": loss_for_seed(args.seed), "steps": 3}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, sort_keys=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""

NONDETERMINISTIC_ARTIFACT = """\
import argparse, json, os, time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-rerun", action="store_true")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    metrics = {"loss": 0.5, "nonce": time.monotonic_ns()}
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, sort_keys=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
"""

CRASHING_ARTIFACT = """\
import sys
sys.exit("synthetic training crash")
"""


@pytest.fixture
def make_chapter(tmp_path):
    """Factory building a synthetic chapter under tmp_path/curriculum/...

    Returns the chapter directory. `meta` overrides/extends the default
    meta.yaml mapping; `artifact_text=None` skips writing the artifact file.
    """

    def _make(
        dirname: str = "ch1.1_bc",
        chapter_id: str = "ch1.1-bc",
        phase: str = "phase1_imitation",
        artifact_name: str = "bc.py",
        artifact_text: str | None = DETERMINISTIC_ARTIFACT,
        meta: dict | None = None,
    ) -> Path:
        chapter_dir = tmp_path / "curriculum" / phase / dirname
        chapter_dir.mkdir(parents=True)
        merged = {
            "id": chapter_id,
            "artifact": artifact_name,
            "objectives": ["learn a thing"],
            "wallclock": [],
        }
        merged.update(meta or {})
        (chapter_dir / "meta.yaml").write_text(
            yaml.safe_dump(merged, sort_keys=True), encoding="utf-8"
        )
        if artifact_text is not None:
            (chapter_dir / artifact_name).write_text(
                artifact_text, encoding="utf-8"
            )
        return chapter_dir

    return _make


@pytest.fixture
def write_wallclock(tmp_path):
    """Write tmp_path/curriculum/common/wallclock.csv with the given text."""

    def _write(text: str) -> Path:
        csv_path = tmp_path / "curriculum" / "common" / "wallclock.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path.write_text(text, encoding="utf-8")
        return csv_path

    return _write
