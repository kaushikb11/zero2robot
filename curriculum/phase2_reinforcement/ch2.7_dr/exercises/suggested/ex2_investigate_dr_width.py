"""SUGGESTED exercise candidate (humans promote) — hyperparameter investigation, ch2.7.

Objective tested: the MECHANISM behind domain randomization — the width of the
randomization band, and where it stops helping. The folklore is "randomize more,
generalize more." This exercise makes you test that against a seed-robust sweep
(ch2.1 spike, H1): form a directional hypothesis about band width, then read it
against the numbers.

THE KNOB. `--dr_width` scales the randomization band. 0 = no randomization (the
narrow policy). 1 = the chapter's nominal band (mass +-40%, friction +-50%,
gravity +-25%). 2 = double-wide, reaching into dynamics so heavy that the deepest
episodes are near-unlearnable on this hardware.

PREDICT before you run: as `--dr_width` grows from 0 to 2, the randomized policy's
survival at the DEEPEST gap point (heaviest mass) will (a) rise steadily — more
randomization, more robustness even there; (b) stay pinned near zero — that point
is a physical ceiling no amount of sampling can cross; (c) rise then fall. And what
happens to NOMINAL return as the band widens? Write your choice and a one-sentence
reason in PREDICTION.

Then run this file. It trains the randomized policy at each width on seeds 0, 1 and
reports, per width, the nominal return and the deepest-gap survival. Read the
MEANS: a real hyperparameter effect has to show in the average, not one seed. Watch
nominal hold steady while the deepest-gap survival refuses to leave the floor —
"randomize harder" widens the range you can reach, but it cannot buy a load the
motors cannot lift.

Estimated learner time: ~20 min (six short two-policy trains).
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PREDICTION = None  # <- "a" | "b" | "c", plus a because-clause

METADATA = {
    "type": "hyperparameter-investigation",
    "chapter": "ch2.7-dr",
    "knob": "--dr_width (randomization band width)",
}

REPO = Path(__file__).resolve().parents[5]
DR = REPO / "curriculum/phase2_reinforcement/ch2.7_dr/dr.py"
SEEDS = (0, 1)
WIDTHS = (0.0, 1.0, 2.0)
EXERCISE_STEPS = 400_000  # the chapter default; below it the randomized policy often fails to converge


def train_width(width: float, seed: int, workdir: Path) -> dict:
    """Train at one --dr_width and return the RANDOMIZED policy's sweep metrics.
    (width 0 makes 'randomized' identical to 'narrow' — the no-DR anchor.)"""
    out = workdir / f"w{width}_s{seed}"
    subprocess.run([sys.executable, str(DR), "--seed", str(seed), "--device", "cpu",
                    "--sweep_knob", "mass", "--dr_width", str(width),
                    "--total_steps", str(EXERCISE_STEPS), "--no-rerun", "--out", str(out)],
                   check=True, capture_output=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


def measure(workdir: Path | None = None) -> dict[float, dict]:
    """Per width: {nominal_return: [per seed], deepgap_survival: [per seed]} for
    the randomized policy. Deterministic per seed."""
    workdir = workdir or Path(tempfile.mkdtemp(prefix="z2r-dr-ex2-"))
    result: dict[float, dict] = {}
    for width in WIDTHS:
        nom_ret, deep_surv = [], []
        for seed in SEEDS:
            m = train_width(width, seed, workdir)
            nominal_idx = m["sweep_grid"].index(1.0)
            nom_ret.append(m["randomized_curve"][nominal_idx]["mean"])
            deep_surv.append(m["randomized_curve"][-1]["survival"])
        result[width] = {"nominal_return": nom_ret, "deepgap_survival": deep_surv}
    return result


if __name__ == "__main__":
    if PREDICTION is None:
        raise SystemExit("write your PREDICTION first — that's the whole exercise")
    print(f"your prediction: {PREDICTION}\n")
    r = measure()
    print(f"{'dr_width':>9s}  {'nominal_return (mean)':>22s}  {'deep-gap survival (mean)':>26s}")
    for width in WIDTHS:
        nr = sum(r[width]["nominal_return"]) / len(SEEDS)
        ds = sum(r[width]["deepgap_survival"]) / len(SEEDS)
        print(f"{width:>9g}  {nr:>22.0f}  {ds:>26.2f}")
    print("\nReconcile: nominal return should hold roughly steady across widths, and the "
          "DEEPEST-gap survival should stay near the floor no matter how wide you go. A "
          "wider band does buy more robustness through the MID gap (on average — it is "
          "noisy) and widens the range you can reach; it never lifts a load the +-12 Nm "
          "motors cannot — 'randomize harder' is not a substitute for a stronger robot.")
