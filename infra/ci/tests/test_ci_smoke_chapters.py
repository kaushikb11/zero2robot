"""smoke_chapters: smoke-run + 2x determinism verification logic.

Uses fake artifact scripts honoring the shared chapter CLI contract
(--smoke --seed --out --no-rerun -> {out}/metrics.json).
"""

import sys

import smoke_chapters

NONDETERMINISTIC_ARTIFACT = """\
import argparse, json, os, time

parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--no-rerun", action="store_true")
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "metrics.json"), "w") as f:
    json.dump({"nonce": time.monotonic_ns()}, f, sort_keys=True)
"""

CRASHING_ARTIFACT = 'import sys\nsys.exit("synthetic training crash")\n'

NO_METRICS_ARTIFACT = """\
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--smoke", action="store_true")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--no-rerun", action="store_true")
parser.parse_args()
"""


def run(tmp_path, *extra):
    # Synthetic roots have no .venv, so pin the interpreter explicitly.
    return smoke_chapters.main(
        ["--root", str(tmp_path), "--python", sys.executable, "--all", *extra]
    )


def test_deterministic_artifact_passes_with_verification(
    make_chapter, tmp_path, capsys
):
    make_chapter()
    assert run(tmp_path, "--verify-determinism") == 0
    out = capsys.readouterr().out
    assert "deterministic x2" in out
    assert "PASS" in out


def test_single_run_without_verification_passes(make_chapter, tmp_path):
    make_chapter()
    assert run(tmp_path) == 0


def test_nondeterministic_artifact_fails_verification(
    make_chapter, tmp_path, capsys
):
    make_chapter(artifact_text=NONDETERMINISTIC_ARTIFACT)
    assert run(tmp_path, "--verify-determinism") == 1
    err_and_out = capsys.readouterr()
    assert "differs across two runs" in err_and_out.out
    assert "FAIL" in err_and_out.out


def test_nondeterministic_artifact_passes_without_verification(
    make_chapter, tmp_path
):
    # Determinism is only checked when asked; a single run still passes.
    make_chapter(artifact_text=NONDETERMINISTIC_ARTIFACT)
    assert run(tmp_path) == 0


def test_crashing_artifact_fails(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text=CRASHING_ARTIFACT)
    assert run(tmp_path) == 1
    assert "exit 1" in capsys.readouterr().out


def test_artifact_without_metrics_json_fails(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text=NO_METRICS_ARTIFACT)
    assert run(tmp_path) == 1
    assert "no metrics.json" in capsys.readouterr().out


def test_missing_artifact_fails(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text=None)
    assert run(tmp_path) == 1
    assert "missing" in capsys.readouterr().out


def test_no_chapters_is_success(tmp_path, capsys):
    assert run(tmp_path) == 0
    assert "nothing to smoke" in capsys.readouterr().out


def test_changed_only_outside_git_falls_back_to_all(
    make_chapter, tmp_path, capsys
):
    # tmp_path is not a git repo: --changed-only must degrade to ALL chapters
    # (the safe direction for a gate), not crash or skip silently.
    make_chapter()
    assert (
        smoke_chapters.main(
            [
                "--root",
                str(tmp_path),
                "--python",
                sys.executable,
                "--changed-only",
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "ALL" in out
    assert "PASS" in out


def test_seed_is_forwarded_to_artifact(make_chapter, tmp_path):
    # The default artifact writes seed-dependent metrics; both runs at the
    # same nonzero seed must still byte-match.
    make_chapter()
    assert run(tmp_path, "--seed", "7", "--verify-determinism") == 0
