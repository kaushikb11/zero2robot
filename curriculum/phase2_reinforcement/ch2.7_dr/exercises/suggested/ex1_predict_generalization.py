"""SUGGESTED exercise candidate (humans promote) — predict-then-run, ch2.7.

Objective tested: the chapter's central, honest claim — domain randomization CAN
extend robustness across the reality gap, but at a free-tier budget it does so
UNRELIABLY, and whether it helped is a question you answer with error bars, not
one lucky seed. This is the ch1.6 "single numbers lie" lesson wearing an RL
costume (ch2.1 spike, H1/H2): you predict a STRUCTURAL fact and check it holds
across seeds.

THE QUESTION. dr.py trains a NARROW policy (nominal dynamics only) and a
RANDOMIZED policy (mass/friction/gravity resampled each episode) and sweeps both
across a mass-scale gap. Across that gap, does the randomized policy reliably HOLD
its survival where the narrow one FALLS?

PREDICT before you run: (a) yes — randomized reliably survives deeper into the gap
than narrow, on every seed; (b) it depends on the seed — sometimes DR extends the
range dramatically, sometimes not at all, so the average edge is within the seed
band; (c) no — randomized is reliably WORSE, having paid a premium for nothing.
Write your choice and one sentence of mechanism in PREDICTION.

Then run this file. It trains both policies on seeds 0, 1, 2 and reads their
survival at nominal (mass 1.0) and at the DEEPEST gap point (heaviest mass). Two
facts hold on every seed — both policies stand at nominal, and both collapse in
the DEEPEST gap. What happens BETWEEN those brackets (does randomized hold at 1.2x
or 1.4x mass?) swings hard with the seed: that swing, not any single run, is the
answer to whether DR "worked" here.

Estimated learner time: ~15 min (three short two-policy trains + sweeps).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause, e.g. "b because ..."

METADATA = {
    "type": "predict-then-run",
    "chapter": "ch2.7-dr",
    "gate_before_run": True,
}

REPO = Path(__file__).resolve().parents[5]
DR = REPO / "curriculum/phase2_reinforcement/ch2.7_dr/dr.py"
SEEDS = (0, 1, 2)
EXERCISE_STEPS = 400_000  # the chapter default: enough to train the stand reliably (below it, DR often fails to converge)


def run_seed(seed: int, workdir: Path) -> dict:
    """Train narrow + randomized and sweep the mass gap; return metrics.json.
    Deterministic per seed (the whole pipeline — torch, env resets, DR draws — is
    seeded, so this does not flake run-to-run)."""
    out = workdir / f"seed{seed}"
    subprocess.run([sys.executable, str(DR), "--seed", str(seed), "--device", "cpu",
                    "--sweep_knob", "mass", "--total_steps", str(EXERCISE_STEPS),
                    "--no-rerun", "--out", str(out)], check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[str, list]:
    """Per-seed survival at nominal and in the deep gap, for both policies. The
    'deep gap' is the last (heaviest) point on the mass grid."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-dr-ex1-"))
    narrow_nom, rand_nom, narrow_deep, rand_deep = [], [], [], []
    for seed in SEEDS:
        m = run_seed(seed, workdir)
        narrow_nom.append(m["narrow_nominal_survival"])
        rand_nom.append(m["randomized_nominal_survival"])
        narrow_deep.append(m["narrow_curve"][-1]["survival"])
        rand_deep.append(m["randomized_curve"][-1]["survival"])
    return {"narrow_nominal": narrow_nom, "randomized_nominal": rand_nom,
            "narrow_deepgap": narrow_deep, "randomized_deepgap": rand_deep}


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    r = measure()
    for seed, nn, rn, nd, rd in zip(SEEDS, r["narrow_nominal"], r["randomized_nominal"],
                                    r["narrow_deepgap"], r["randomized_deepgap"]):
        print(f"seed {seed}: nominal survival  narrow {nn:.2f} / randomized {rn:.2f}   "
              f"deep-gap survival  narrow {nd:.2f} / randomized {rd:.2f}")
    print("\nBoth stand at nominal and both collapse in the DEEPEST gap on every seed — "
          "those brackets are solid. But look at the spread BETWEEN them across seeds: if "
          "the randomized edge flips sign seed to seed, you have not shown DR 'worked' — "
          "you have shown the honest answer lives inside the seed band. Measure it.")
