"""Subprocess tests for the live PreToolUse/PostToolUse hooks.

The review found NO tests exercised infra/hooks/*. These pipe realistic
tool_input shapes (Write / Edit / NotebookEdit) through the hooks as
subprocesses and assert exit codes + which channel (stdout/stderr) the message
uses — the exact contract Claude Code enforces.

Regression coverage:
- #0  Edit sends `new_string` (not `new_str`) — forbidden imports on Edit block.
- #1/#2  exercise/demo code + digit-named files are gated (not just the artifact).
- #3/L0  `allow_transformers` in a YAML COMMENT does not grant (line-anchored).
- #3/#6  NotebookEdit into notebooks/ is blocked (hook reads notebook_path).
- #4  PostToolUse advisories reach the agent (stderr+exit2 / JSON additionalContext).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HOOKS = Path(__file__).resolve().parents[2] / "hooks"
PEDAGOGY_GATE = HOOKS / "pedagogy_gate.py"
POST_EDIT_CHECKS = HOOKS / "post_edit_checks.py"

# A realistic curriculum path — the hook keys off the path STRING; the file
# need not exist for the import/path-classification logic (only meta_grants
# reads a sibling file, exercised separately with a real tmp tree).
CH = "curriculum/phase1_imitation/ch1.4_rl"


def run_hook(hook: Path, payload: dict) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(hook)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
    )


# --- pedagogy_gate: forbidden imports -------------------------------------


def test_edit_with_forbidden_import_blocks():
    """#0: Edit sends new_string; the forbidden-import check must fire."""
    payload = {
        "tool_name": "Edit",
        "tool_input": {
            "file_path": f"{CH.replace('ch1.4_rl', 'ch1.1_bc')}/bc.py",
            "old_string": "import torch",
            "new_string": "import torch\nimport hydra\nfrom stable_baselines3 import PPO",
        },
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 2, result.stderr
    assert "BLOCKED" in result.stderr
    assert "hydra" in result.stderr


def test_write_with_forbidden_import_blocks():
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": f"{CH}/rl.py", "content": "import gymnasium\n"},
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 2
    assert "gymnasium" in result.stderr


def test_exercise_file_with_forbidden_import_blocks():
    """#1/#2: exercise code (digit-named, under exercises/) is now gated."""
    payload = {
        "tool_name": "Write",
        "tool_input": {
            "file_path": f"{CH}/exercises/ex1_gym.py",
            "content": "import gymnasium\n",
        },
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "gymnasium" in result.stderr


def test_edit_exercise_file_blocks():
    """#1/#2 via Edit tool shape on an exercise file."""
    payload = {
        "tool_name": "Edit",
        "tool_input": {
            "file_path": f"{CH}/exercises/ex2_solution.py",
            "old_string": "pass",
            "new_string": "from stable_baselines3 import PPO",
        },
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 2
    assert "stable_baselines3" in result.stderr


def test_tests_dir_not_gated():
    """Human-owned tests/ is off-limits — hook must not gate it."""
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": f"{CH}/tests/test_rl.py", "content": "import gym\n"},
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 0, result.stderr


def test_clean_edit_passes():
    payload = {
        "tool_name": "Edit",
        "tool_input": {
            "file_path": f"{CH}/rl.py",
            "old_string": "x = 1",
            "new_string": "import torch\nimport numpy as np",
        },
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 0, result.stderr


def test_word_boundary_not_tripped_by_prefix():
    """`gymnastics` must not trip the `gym` rule."""
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": f"{CH}/rl.py", "content": "import gymnastics\n"},
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 0


def test_non_chapter_path_not_gated():
    """Forbidden imports outside curriculum chapter code aren't the hook's job."""
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": "infra/ci/helper.py", "content": "import gym\n"},
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 0


# --- pedagogy_gate: transformers grant (line-anchored, real meta.yaml) ------


def _tinyvla_chapter(tmp_path: Path, meta_text: str) -> Path:
    chapter_dir = tmp_path / "curriculum" / "phase2_vla" / "ch2.1_tinyvla"
    chapter_dir.mkdir(parents=True)
    (chapter_dir / "meta.yaml").write_text(meta_text, encoding="utf-8")
    return chapter_dir / "vla.py"


def test_transformers_in_meta_comment_does_not_grant(tmp_path):
    """#3/L0: allow_transformers in a YAML COMMENT must not grant the exception."""
    py = _tinyvla_chapter(
        tmp_path,
        "# allow_transformers: true  <- a comment, must NOT grant\nid: ch2.1\n",
    )
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(py), "content": "from transformers import AutoModel\n"},
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "allow_transformers" in result.stderr


def test_transformers_line_anchored_grant_allows(tmp_path):
    """#3/L0: a real `allow_transformers: true` KEY line grants the exception."""
    py = _tinyvla_chapter(tmp_path, "id: ch2.1\nallow_transformers: true\n")
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": str(py), "content": "from transformers import AutoModel\n"},
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 0, result.stderr


# --- pedagogy_gate: protected paths (notebooks) -----------------------------


def test_notebookedit_into_notebooks_blocks():
    """#3/#6: NotebookEdit sends notebook_path — protected block must fire."""
    payload = {
        "tool_name": "NotebookEdit",
        "tool_input": {"notebook_path": "notebooks/ch1.1-bc.ipynb", "new_source": "print(1)"},
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 2, result.stdout + result.stderr
    assert "protected" in result.stderr


def test_write_into_notebooks_blocks():
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": "notebooks/ch1.1-bc.ipynb", "content": "{}"},
    }
    result = run_hook(PEDAGOGY_GATE, payload)
    assert result.returncode == 2
    assert "protected" in result.stderr


# --- post_edit_checks: advisories must reach the agent ----------------------


def test_wallclock_manual_edit_uses_stderr_and_exit2():
    """#4: wallclock.csv hand-edit is 'must not happen' -> stderr + exit 2."""
    payload = {
        "tool_name": "Edit",
        "tool_input": {
            "file_path": "curriculum/common/wallclock.csv",
            "old_string": "1.0",
            "new_string": "0.5",
        },
    }
    result = run_hook(POST_EDIT_CHECKS, payload)
    assert result.returncode == 2, result.stdout
    assert "wallclock.csv" in result.stderr
    assert result.stdout.strip() == ""  # not on the ignored stdout channel


def test_prose_edit_emits_additional_context_json():
    """#4: prose drift NOTE -> JSON hookSpecificOutput.additionalContext, exit 0."""
    payload = {
        "tool_name": "Edit",
        "tool_input": {
            "file_path": "curriculum/phase1_imitation/ch1.1_bc/prose/chapter.md",
            "old_string": "a",
            "new_string": "b",
        },
    }
    result = run_hook(POST_EDIT_CHECKS, payload)
    assert result.returncode == 0, result.stderr
    payload_out = json.loads(result.stdout)
    hso = payload_out["hookSpecificOutput"]
    assert hso["hookEventName"] == "PostToolUse"
    assert "voice-check" in hso["additionalContext"]


def test_post_edit_unrelated_file_silent():
    payload = {
        "tool_name": "Write",
        "tool_input": {"file_path": "curriculum/phase1_imitation/ch1.1_bc/bc.py", "content": "x"},
    }
    result = run_hook(POST_EDIT_CHECKS, payload)
    assert result.returncode == 0
    assert result.stdout.strip() == ""
