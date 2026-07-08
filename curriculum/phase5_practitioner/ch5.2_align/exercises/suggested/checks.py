"""SUGGESTED local pytest checks for the ch5.2 exercise candidates.

Run from anywhere:
    pytest curriculum/phase5_practitioner/ch5.2_align/exercises/suggested/checks.py

Conventions (match ch1.7 / ch1.8):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- The two self-contained completion gates (ex3 InfoNCE loss, ex4 retrieval@1) are FAST
  and deterministic — they run in `make check`, and SKIP while the stub raises
  NotImplementedError (implementing it is the exercise).
- The two reproduce checks TRAIN the contrastive encoder at a reduced config, so they are
  @pytest.mark.slow (excluded from `make check`) and assert the DIRECTION, not an exact %:
  ex1 (full InfoNCE > --break noneg) and ex2 (aligned > supervised > random, fine).
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic
  numbers) — read them, don't inline.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
ALIGN = REPO / "curriculum/phase5_practitioner/ch5.2_align/align.py"
sys.path.insert(0, str(HERE))

import ex1_predict_negatives as ex1  # noqa: E402
import ex2_predict_aligned_vs_supervised as ex2  # noqa: E402
import ex3_completion_infonce as ex3  # noqa: E402
import ex4_completion_retrieval as ex4  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_align(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(ALIGN), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--episodes", str(RC["episodes"]), "--epochs", str(RC["epochs"]),
           "--sup_epochs", str(RC["sup_epochs"]), "--batch_size", str(RC["batch_size"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1 and compare correct InfoNCE vs --break noneg"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2 and order aligned / supervised / random"


# --------------------------------------------------- ex3 InfoNCE completion (fast)

def _ref_info_nce(img_e, txt_e, logit_scale):
    logits = logit_scale.exp() * img_e @ txt_e.t()
    labels = torch.arange(len(logits))
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def test_ex3_matches_reference():
    torch.manual_seed(0)
    img_e = F.normalize(torch.randn(8, 16), dim=-1)
    txt_e = F.normalize(torch.randn(8, 16), dim=-1)
    scale = torch.tensor(2.0)
    try:
        got = ex3.symmetric_info_nce(img_e, txt_e, scale)
    except NotImplementedError:
        pytest.skip("symmetric_info_nce not implemented yet — that's the exercise")
    want = _ref_info_nce(img_e, txt_e, scale)
    assert abs(float(got) - float(want)) <= CHECKS["ex3"]["abs_tol"], f"loss differs: got {got}, want {want}"


def test_ex3_is_symmetric():
    # A correct symmetric loss is invariant to swapping the two towers.
    torch.manual_seed(1)
    a = F.normalize(torch.randn(6, 12), dim=-1)
    b = F.normalize(torch.randn(6, 12), dim=-1)
    scale = torch.tensor(1.5)
    try:
        lab, lba = ex3.symmetric_info_nce(a, b, scale), ex3.symmetric_info_nce(b, a, scale)
    except NotImplementedError:
        pytest.skip("symmetric_info_nce not implemented yet")
    assert abs(float(lab) - float(lba)) <= CHECKS["ex3"]["abs_tol"], "a symmetric loss must not depend on tower order"


# ------------------------------------------------- ex4 retrieval@1 completion (fast)

def _ref_retrieval(text_emb, image_emb, gal_cls, qry_cls):
    top1 = (text_emb @ image_emb.T).argmax(axis=1)
    return float((gal_cls[top1] == qry_cls).mean())


def test_ex4_matches_reference():
    rng = np.random.default_rng(0)
    image_emb = rng.standard_normal((10, 8))
    image_emb /= np.linalg.norm(image_emb, axis=1, keepdims=True)
    text_emb = image_emb[[2, 5, 9]] + 0.01 * rng.standard_normal((3, 8))  # queries near gallery 2,5,9
    text_emb /= np.linalg.norm(text_emb, axis=1, keepdims=True)
    gal_cls = np.arange(10) % 4
    qry_cls = gal_cls[[2, 5, 9]]
    try:
        got = ex4.retrieval_at1(text_emb, image_emb, gal_cls, qry_cls)
    except NotImplementedError:
        pytest.skip("retrieval_at1 not implemented yet — that's the exercise")
    want = _ref_retrieval(text_emb, image_emb, gal_cls, qry_cls)
    assert abs(got - want) <= CHECKS["ex4"]["abs_tol"], f"retrieval@1 differs: got {got}, want {want}"


# ----------------------------------------------------- reproduce (SLOW: trains)

@pytest.mark.slow
def test_ex1_noneg_is_worse(tmp_path):
    correct = run_align(tmp_path / "correct")
    noneg = run_align(tmp_path / "noneg", ["--break", "noneg"])
    band = RC["min_negatives_over_noneg"]
    assert correct["retrieval_at1_aligned"] - noneg["retrieval_at1_aligned"] >= band, \
        f"full InfoNCE should beat --break noneg by >= {band}: correct={correct}, noneg={noneg}"
    assert noneg["retrieval_at1_aligned"] >= correct["retrieval_at1_random"], \
        "noneg collapses but should still beat a random encoder (it kept the positive pull)"


@pytest.mark.slow
def test_ex2_aligned_beats_supervised_and_random(tmp_path):
    m = run_align(tmp_path / "mix")
    assert m["retrieval_at1_aligned"] - m["retrieval_at1_random"] >= RC["min_aligned_over_random"], \
        f"aligned must beat random (fine): {m}"
    assert m["retrieval_at1_aligned"] - m["retrieval_at1_supervised"] >= RC["min_aligned_over_supervised"], \
        f"aligned must beat the supervised probe on the FINE metric: {m}"
