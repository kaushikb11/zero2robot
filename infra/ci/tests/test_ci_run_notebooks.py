"""run_notebooks: discovery, reporting, fail-fast, and exit-code logic.

Kernel-free: `execute_notebook` is monkeypatched (ipykernel is not part of the
dev extra; real execution is exercised by the ci-notebook lane itself).
"""

import json
import os
from pathlib import Path

import pytest
import run_notebooks


def put_notebook(tmp_path, name: str) -> Path:
    notebooks = tmp_path / "notebooks"
    notebooks.mkdir(exist_ok=True)
    path = notebooks / name
    path.write_text('{"cells": []}', encoding="utf-8")
    return path


@pytest.fixture
def fake_executor(monkeypatch):
    """Replace kernel execution; notebooks named fail_*.ipynb raise."""
    executed = []

    def _execute(path, timeout):
        executed.append(path.name)
        if path.name.startswith("fail_"):
            raise RuntimeError("synthetic cell failure")

    monkeypatch.setattr(run_notebooks, "execute_notebook", _execute)
    return executed


def run(tmp_path, *extra):
    report = tmp_path / "report.json"
    code = run_notebooks.main(
        ["--root", str(tmp_path), "--report", str(report), *extra]
    )
    data = json.loads(report.read_text()) if report.exists() else None
    return code, data


def test_zero_notebooks_is_success_with_empty_report(tmp_path, capsys):
    code, data = run(tmp_path)
    assert code == 0
    assert data == {}
    assert "no notebooks" in capsys.readouterr().out


def test_all_passing(tmp_path, fake_executor):
    put_notebook(tmp_path, "ch1.1-bc.ipynb")
    put_notebook(tmp_path, "ch1.2-act.ipynb")
    code, data = run(tmp_path)
    assert code == 0
    assert {v["status"] for v in data.values()} == {"pass"}
    assert sorted(fake_executor) == ["ch1.1-bc.ipynb", "ch1.2-act.ipynb"]


def test_failure_reported_and_exit_1_after_report(tmp_path, fake_executor):
    put_notebook(tmp_path, "ch1.1-bc.ipynb")
    put_notebook(tmp_path, "fail_ch9.ipynb")
    code, data = run(tmp_path)
    assert code == 1
    assert data["notebooks/ch1.1-bc.ipynb"]["status"] == "pass"
    failed = data["notebooks/fail_ch9.ipynb"]
    assert failed["status"] == "fail"
    assert "synthetic cell failure" in failed["error"]
    assert failed["seconds"] >= 0


def test_fail_fast_skips_remaining(tmp_path, fake_executor):
    put_notebook(tmp_path, "fail_aaa.ipynb")  # sorts first
    put_notebook(tmp_path, "zzz.ipynb")
    code, data = run(tmp_path, "--fail-fast", "true")
    assert code == 1
    assert data["notebooks/fail_aaa.ipynb"]["status"] == "fail"
    assert data["notebooks/zzz.ipynb"]["status"] == "skipped"
    assert fake_executor == ["fail_aaa.ipynb"]


def test_fail_fast_false_runs_everything(tmp_path, fake_executor):
    put_notebook(tmp_path, "fail_aaa.ipynb")
    put_notebook(tmp_path, "zzz.ipynb")
    code, data = run(tmp_path, "--fail-fast", "false")
    assert code == 1
    assert data["notebooks/zzz.ipynb"]["status"] == "pass"


def test_profile_sets_kernel_env(tmp_path, fake_executor, monkeypatch):
    monkeypatch.delenv("Z2R_PROFILE", raising=False)
    put_notebook(tmp_path, "ch1.1-bc.ipynb")
    code, _ = run(tmp_path, "--profile", "cpu-smoke")
    assert code == 0
    assert os.environ["Z2R_PROFILE"] == "cpu-smoke"


def test_checkpoints_dir_ignored(tmp_path, fake_executor):
    put_notebook(tmp_path, "ch1.1-bc.ipynb")
    checkpoints = tmp_path / "notebooks" / ".ipynb_checkpoints"
    checkpoints.mkdir()
    (checkpoints / "fail_ghost.ipynb").write_text("{}", encoding="utf-8")
    code, data = run(tmp_path)
    assert code == 0
    assert list(data) == ["notebooks/ch1.1-bc.ipynb"]


def test_str2bool_rejects_garbage():
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        run_notebooks.str2bool("maybe")
