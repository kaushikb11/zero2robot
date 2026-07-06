"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch2.8.

Objective tested: the chapter's central claim that the control RATE is not a
detail — it is the thing keeping the robot alive. The sensor and the plant keep
running at 50 Hz; you slow only the POLICY node (--control_hz), so every control
step in between re-applies the last action (zero-order hold). The trained ch2.1
policy is a good balancer, so it tolerates a surprising amount of slowdown — and
then falls off a cliff.

PREDICT before you run: as you drop --control_hz from 50 down through 25, 20, 15,
10, 5, at roughly what rate does the pole first FAIL to survive the full 10-second
run? Write your threshold (a Hz number) and one sentence of why in PREDICTION.

Then run this file. It runs the graph at each rate on seeds 0,1,2 under the
deterministic virtual clock and prints, per rate, whether the pole balanced and
the mean sense->act latency. Notice that the balanced/fell outcome is identical
across seeds while it is surviving (a deterministic graph), and that latency
climbs steadily long before the pole actually falls — the warning sign is visible
before the failure.

Estimated learner time: 20 minutes (mostly reading the latency trend).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- e.g. "8 because below ~10 Hz the 100 ms staleness exceeds the pole's reaction window"

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch2.8-runtime",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.8_runtime/runtime.py"
SEEDS = (0, 1, 2)
RATES = (50.0, 25.0, 20.0, 15.0, 10.0, 5.0)
# Deterministic virtual clock, forced CPU: same (rate, seed) -> same metrics.json.
COMMON = ["--device", "cpu", "--clock", "virtual", "--no-rerun"]


def run_one(control_hz: float, seed: int, workdir: Path) -> dict:
    out = workdir / f"hz{control_hz:g}_seed{seed}"
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", str(seed),
                    "--control_hz", str(control_hz), "--out", str(out), *COMMON],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[float, dict]:
    """Return {control_hz: {"balanced": [...per seed], "latency_ms": [...],
    "steps": [...]}}. Deterministic per seed, so this never flakes run-to-run."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-rt-ex1-"))
    results: dict[float, dict] = {}
    for hz in RATES:
        runs = [run_one(hz, seed, workdir) for seed in SEEDS]
        results[hz] = {
            "balanced": [bool(r["balanced"]) for r in runs],
            "latency_ms": [float(r["mean_latency_ms"]) for r in runs],
            "steps": [int(r["steps"]) for r in runs],
        }
    return results


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    print(f"{'control_hz':>10}  {'balanced (seeds 0-2)':>22}  {'mean latency ms':>15}")
    for hz, r in results.items():
        up = "".join("Y" if b else "n" for b in r["balanced"])
        lat = sum(r["latency_ms"]) / len(r["latency_ms"])
        print(f"{hz:>10g}  {up:>22}  {lat:>15.1f}")
    print("\nReconcile: at what rate did 'Y' turn to 'n'? Note how far the latency "
          "had already climbed at the last rate that still balanced — the runtime "
          "was telling you it was in trouble before the pole fell.")
