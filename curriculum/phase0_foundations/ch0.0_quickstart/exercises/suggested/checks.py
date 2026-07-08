"""SUGGESTED local pytest checks for the ch0.0 exercise candidate.

Run from anywhere:
    pytest curriculum/phase0_foundations/ch0.0_quickstart/exercises/suggested/checks.py

Conventions (match ch1.1 / ch1.3):
- The prediction gate (PREDICTION unset) SKIPS rather than fails: the site
  enforces predict-before-run; locally we only verify the recorded choice.
- The reproduce check trains a policy (starved 5-demo config) and is therefore
  @pytest.mark.slow — excluded from `make check`, which runs only the fast gates.
- Reference bands live in the chapter meta.yaml with provenance (exercise-spec:
  no bare magic numbers) — read them, don't inline.
"""

import sys
from pathlib import Path

import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
QUICKSTART = REPO / "curriculum/phase0_foundations/ch0.0_quickstart/quickstart.py"
sys.path.insert(0, str(HERE))

import ex1_predict_data_hunger as ex1  # noqa: E402

EX1 = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]["ex1"]


# ----------------------------------------------------------------- prediction

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == EX1["answer"], "run ex1 and watch the 5-demo policy sit at the random floor"


# ------------------------------------------------------------- reproduce (slow)

@pytest.mark.slow
def test_ex1_five_demos_do_not_clear_the_floor(tmp_path):
    """Five succeeding demos still fail to clear the random floor — the claim the
    exercise makes the learner predict. Provenance + band in meta.yaml."""
    m = ex1.run_quickstart(tmp_path / "starved", EX1["demos"])
    assert m["expert_successes"] == EX1["demos"], "all 5 scripted demos should succeed"
    assert m["random_rate"] == 0.0, "the random-action floor on these starts is zero"
    assert m["success_rate"] <= EX1["starved_success_rate_max"], (
        f"5 demos cleared the floor ({m['success_rate']:.2f}) — expected <= "
        f"{EX1['starved_success_rate_max']}; the whole lesson is that it does not"
    )
