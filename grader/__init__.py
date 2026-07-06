"""grader/ — zero2robot's grading harness (INFRA, not pedagogy).

Two jobs (grader/CLAUDE.md):
  1. exercise auto-checking  — formalize the "run it locally" self-check
     (grader.check): discover a chapter's exercises/suggested/checks.py and
     run them offline via pytest, reporting pass/fail/skip per test.
  2. leaderboard submission grading — validate an ONNX policy against tensor
     contract v1 (grader.contract, reusing curriculum/common's keys) and score
     it deterministically on PushT public seeds (grader.scoring).

Determinism (grader/CLAUDE.md): same submission + same seed set -> same score;
every score is reproducible from the submission hash (grader.submission).

Sandbox: the scoring path is designed to run INSIDE the gVisor sandbox
described by grader/sandbox/policy.yaml (network none, resource/time caps,
onnxruntime-only ONNX loading). The Python harness honors the parts it can
express in-process (onnxruntime-only load, file-size cap, no network calls,
wallclock guard); the container enforces isolation/network/CPU/memory. See
grader.sandbox for the policy loader and grader/README-not: this is infra.

HIDDEN SEEDS ARE HUMAN-OWNED. grader/hidden_seeds/ does not exist here and must
never be created by an agent (hook-denied). grader.seeds ships only the PUBLIC
seed source plus a documented seam where the hidden implementation plugs in.
"""

__version__ = "0.0.1"
