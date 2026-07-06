"""SUGGESTED local pytest checks for the ch1.7 exercise candidates.

Run from anywhere:
    pytest curriculum/phase1_imitation/ch1.7_vla_data/exercises/suggested/checks.py

Conventions (match ch1.1 / ch1.3 / ch1.4 / ch1.5):
- Prediction gates (PREDICTION unset) SKIP rather than fail: the site enforces
  predict-before-run; locally we only verify the recorded choice.
- The two self-contained gates (ex3 tokenizer bug-hunt, ex4 bag-of-words completion)
  are FAST and deterministic — they run in `make check`.
- ex3 SKIPS while the missing-BOS bug is still present (finding it is the learner's
  job) and asserts the fix puts BOS first and EOS after the last real word.
- ex4 SKIPS while `bag_of_words` raises NotImplementedError.
- This chapter trains NO policy, so the "reproduce" checks just rebuild the dataset at
  a reduced config (seconds on CPU) and assert dataset/leakage properties — none are
  @pytest.mark.slow.
- Reference bands live in meta.yaml with provenance (exercise-spec: no bare magic
  numbers) — read them, don't inline.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
VLA = REPO / "curriculum/phase1_imitation/ch1.7_vla_data/vla_data.py"
sys.path.insert(0, str(HERE))

import ex1_predict_leakage as ex1  # noqa: E402
import ex2_predict_multitask_mix as ex2  # noqa: E402
import ex3_bughunt_tokenizer as ex3  # noqa: E402
import ex4_completion_bagofwords as ex4  # noqa: E402

CHECKS = yaml.safe_load((HERE.parents[1] / "meta.yaml").read_text())["exercise_checks"]
RC = CHECKS["exercise_config"]
ANSWER_KEY = {"ex1": CHECKS["ex1"]["answer"], "ex2": CHECKS["ex2"]["answer"]}


def run_vla(out: Path, extra: list[str] | None = None) -> dict:
    cmd = [sys.executable, str(VLA), "--seed", "0", "--device", "cpu", "--no-rerun",
           "--out", str(out), "--episodes_per_task", str(RC["episodes_per_task"]),
           "--frame_stride", str(RC["frame_stride"]), "--feature_dim", str(RC["feature_dim"]),
           "--conv_width", str(RC["conv_width"]), *(extra or [])]
    subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=REPO)
    return json.loads((out / "metrics.json").read_text())


# ----------------------------------------------------------------- predictions

def test_ex1_prediction_recorded():
    if ex1.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex1 first")
    assert ex1.PREDICTION == ANSWER_KEY["ex1"], "run ex1 and compare the clean vs leak action_from_language_r2"


def test_ex2_prediction_recorded():
    if ex2.PREDICTION is None:
        pytest.skip("PREDICTION not set — record your choice in ex2 first")
    assert ex2.PREDICTION == ANSWER_KEY["ex2"], "run ex2: both tasks appear, with different frame counts"


# ------------------------------------------ ex3 tokenizer bug-hunt (fast)

def test_ex3_bos_first_and_eos_after_words():
    ids = ex3.encode("push the cube to the target")
    if ids[0] != ex3.STOI["<bos>"]:
        pytest.skip("ex3 encode still drops the leading BOS — find it")
    words = "push the cube to the target".split()
    # BOS, then one id per word, then EOS at index len(words)+1.
    assert list(ids[1:1 + len(words)]) == [ex3.STOI[w] for w in words], "words must follow BOS in order"
    assert ids[1 + len(words)] == ex3.STOI["<eos>"], "EOS must sit right after the last real word"
    assert len(ids) == ex3.MAX_TOKENS, "encode must always return MAX_TOKENS ids"
    assert (ids[2 + len(words):] == ex3.STOI["<pad>"]).all(), "the tail must be PAD"


def test_ex3_oov_maps_to_unk():
    ids = ex3.encode("push the banana")   # 'banana' is out of vocab
    if ids[0] != ex3.STOI["<bos>"]:
        pytest.skip("ex3 not fixed yet — BOS still missing")
    assert ids[3] == ex3.STOI["<unk>"], "an out-of-vocab word must tokenize to <unk>"


# ------------------------------------------- ex4 bag-of-words completion (fast)

def _ref_bag(tokens, vocab_size, pad_id=0):
    bag = np.zeros(vocab_size, np.float64)
    for t in tokens:
        if t != pad_id:
            bag[t] += 1.0
    return bag


def test_ex4_counts_match_reference():
    tokens = np.array([2, 5, 5, 7, 3, 0, 0, 0], dtype=np.int64)  # ids with repeats + pad
    try:
        got = np.asarray(ex4.bag_of_words(tokens, vocab_size=8), dtype=np.float64)
    except NotImplementedError:
        pytest.skip("bag_of_words not implemented yet — that's the exercise")
    want = _ref_bag(tokens, 8)
    assert got.shape == want.shape, f"expected shape {want.shape}, got {got.shape}"
    assert np.allclose(got, want, atol=CHECKS["ex4"]["abs_tol"]), f"counts differ: got {got}, want {want}"


def test_ex4_pad_is_ignored():
    try:
        got = ex4.bag_of_words(np.array([0, 0, 0, 0], dtype=np.int64), vocab_size=8)
    except NotImplementedError:
        pytest.skip("bag_of_words not implemented yet — that's the exercise")
    assert float(np.sum(got)) == 0.0, "an all-PAD instruction must produce an all-zero bag"


# ---------------------------------------------------------- reproduce (fast: no training)

def test_ex1_leak_makes_action_decodable(tmp_path):
    # The chapter headline at the reduced config: clean templates keep the action
    # essentially UN-decodable from language, and --break leak sends the probe R^2 up
    # by a wide, measured margin. Measured reduced: clean ~0.016, leak ~0.795.
    clean = run_vla(tmp_path / "clean")
    leak = run_vla(tmp_path / "leak", ["--break", "leak"])
    assert clean["action_from_language_r2"] <= RC["max_clean_r2"], \
        f"clean templates should keep the probe near 0: {clean['action_from_language_r2']}"
    assert leak["action_from_language_r2"] - clean["action_from_language_r2"] >= RC["min_leak_gap"], \
        f"leak should make the action decodable from language: clean={clean}, leak={leak}"


def test_ex2_both_tasks_present(tmp_path):
    m = run_vla(tmp_path / "mix")
    if RC["both_tasks_present"]:
        assert m["num_examples_pusht"] > 0 and m["num_examples_aloha"] > 0, \
            f"a multi-task dataset must contain BOTH tasks: {m}"
    assert m["num_examples"] == m["num_examples_pusht"] + m["num_examples_aloha"], \
        f"total examples must be the sum of the per-task counts: {m}"
