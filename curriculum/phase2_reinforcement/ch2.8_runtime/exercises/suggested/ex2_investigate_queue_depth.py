"""SUGGESTED exercise candidate (humans promote) — hyperparameter-investigation,
ch2.8.

Objective tested: the difference between a message-delivery problem and a control
problem — and why fixing the first does nothing for the second. When the policy
node is too slow (--control_hz 5) the 50 Hz sensor overruns the depth-1 /obs
topic, so most sensor messages are DROPPED before the policy ever reads them. The
tempting fix is "make the queue deeper so we stop dropping messages." Try it.

THE INVESTIGATION. Hold --control_hz at 5 Hz (too slow to balance). Compare a
shallow queue (--queue_depth 1) against a deep one (--queue_depth 100) across
seeds. Watch two numbers: obs_dropped, and whether the pole balanced.

(This is a hyperparameter-investigation, not a bug-hunt: there is no injected bug
to find. The lesson is what the knob does and does not do — seed-robust, because
the outcome is deterministic per seed under the virtual clock.)

Estimated learner time: 20 minutes.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

# Before you run — predict both outcomes for the deep queue: what happens to
# obs_dropped, and does the pole stay up? Write one sentence committing to both.
PREDICTION = None

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch2.8-runtime",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
ARTIFACT = REPO / "curriculum/phase2_reinforcement/ch2.8_runtime/runtime.py"
SEEDS = (0, 1, 2)
SLOW_CONTROL_HZ = 5.0
DEPTHS = (1, 100)  # shallow vs deep /obs and /action queues
# Virtual clock + CPU + the checkpoint-free scripted brain, all pinned so the
# investigation is deterministic per seed on any machine.
COMMON = ["--device", "cpu", "--clock", "virtual", "--no-rerun", "--policy", "scripted"]


def run_one(queue_depth: int, seed: int, workdir: Path) -> dict:
    out = workdir / f"qd{queue_depth}_seed{seed}"
    subprocess.run([sys.executable, str(ARTIFACT), "--seed", str(seed),
                    "--control_hz", str(SLOW_CONTROL_HZ), "--queue_depth", str(queue_depth),
                    "--out", str(out), *COMMON],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[int, dict]:
    """Return {queue_depth: {"balanced": [...per seed], "obs_dropped": [...],
    "steps": [...], "latency_ms": [...]}}. Deterministic per seed."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-rt-ex2-"))
    results: dict[int, dict] = {}
    for depth in DEPTHS:
        runs = [run_one(depth, seed, workdir) for seed in SEEDS]
        results[depth] = {
            "balanced": [bool(r["balanced"]) for r in runs],
            "obs_dropped": [int(r["obs_dropped"]) for r in runs],
            "steps": [int(r["steps"]) for r in runs],
            "latency_ms": [float(r["mean_latency_ms"]) for r in runs],
        }
    return results


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    results = measure()
    print(f"{'queue_depth':>11}  {'obs_dropped':>18}  {'balanced':>18}  {'mean latency ms':>15}")
    for depth, r in results.items():
        drops = "/".join(str(d) for d in r["obs_dropped"])
        up = "".join("Y" if b else "n" for b in r["balanced"])
        lat = sum(r["latency_ms"]) / len(r["latency_ms"])
        print(f"{depth:>11}  {drops:>18}  {up:>18}  {lat:>15.1f}")
    print("\nReconcile: the deep queue took obs_dropped to zero. Did the pole "
          "balance? Did the latency change? A deeper buffer removes the dropped-message "
          "SYMPTOM but not the CAUSE — the policy still decides only 5 times a second, "
          "so it is still acting on ~100 ms-stale state. Buffering trades drops for "
          "latency; it never buys you a faster controller. The rate is the killer.")
