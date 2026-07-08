"""check_export_parity: discovery, --smoke detection, and pass/skip/fail
classification via a real subprocess (fake export scripts in tmp_path)."""

import sys

import check_export_parity as cep


def _mk(root, rel, body):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_discovery_globs_export_scripts(tmp_path):
    _mk(tmp_path, "curriculum/phase1/ch1.1_bc/export_bc_onnx.py", "x")
    _mk(tmp_path, "curriculum/phase2/ch2.1_ppo/export_ppo_onnx.py", "x")
    _mk(tmp_path, "curriculum/phase1/ch1.1_bc/bc.py", "x")  # not an export script
    found = [p.name for p in cep.discover_export_scripts(tmp_path)]
    assert found == ["export_bc_onnx.py", "export_ppo_onnx.py"]


def test_no_curriculum_dir_is_empty(tmp_path):
    assert cep.discover_export_scripts(tmp_path) == []


def test_has_smoke_flag(tmp_path):
    with_flag = _mk(tmp_path, "a.py", 'p.add_argument("--smoke", action="store_true")')
    without = _mk(tmp_path, "b.py", 'p.add_argument("--run")')
    assert cep.has_smoke_flag(with_flag) is True
    assert cep.has_smoke_flag(without) is False


PASS_SCRIPT = '''\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--smoke", action="store_true")
p.parse_args()
print("exported foo.onnx — torch/onnx parity delta 1e-9")
'''

NO_PARITY_SCRIPT = '''\
import argparse
p = argparse.ArgumentParser()
p.add_argument("--smoke", action="store_true")
p.parse_args()
print("exported foo.onnx")   # exits 0 but never asserts parity
'''

CRASH_SCRIPT = '''\
import argparse, sys
p = argparse.ArgumentParser()
p.add_argument("--smoke", action="store_true")
p.parse_args()
sys.stderr.write("AttributeError: 'tuple' object has no attribute 'eval'\\n")
sys.exit(1)
'''


def test_run_smoke_pass(tmp_path):
    script = _mk(tmp_path, "export_ok_onnx.py", PASS_SCRIPT)
    status, detail = cep.run_export_smoke(sys.executable, script, timeout=60)
    assert status == "PASS"
    assert "parity" in detail.lower()


def test_run_smoke_skip_when_no_flag(tmp_path):
    script = _mk(tmp_path, "export_noskip_onnx.py", 'print("needs a run dir")')
    status, detail = cep.run_export_smoke(sys.executable, script, timeout=60)
    assert status == "SKIP"
    assert "no --smoke path" in detail


def test_run_smoke_fail_when_no_parity_line(tmp_path):
    script = _mk(tmp_path, "export_np_onnx.py", NO_PARITY_SCRIPT)
    status, detail = cep.run_export_smoke(sys.executable, script, timeout=60)
    assert status == "FAIL"
    assert "parity" in detail.lower()


def test_run_smoke_fail_on_crash(tmp_path):
    script = _mk(tmp_path, "export_crash_onnx.py", CRASH_SCRIPT)
    status, detail = cep.run_export_smoke(sys.executable, script, timeout=60)
    assert status == "FAIL"
    assert "exit 1" in detail


def test_main_exit_1_on_failure(tmp_path, capsys):
    _mk(tmp_path, "curriculum/phase1/ch1_x/export_ok_onnx.py", PASS_SCRIPT)
    _mk(tmp_path, "curriculum/phase1/ch2_y/export_crash_onnx.py", CRASH_SCRIPT)
    code = cep.main(["--root", str(tmp_path), "--python", sys.executable])
    assert code == 1
    out = capsys.readouterr().out
    assert "PASS" in out and "FAIL" in out


def test_main_ok_with_skip(tmp_path, capsys):
    _mk(tmp_path, "curriculum/phase1/ch1_x/export_ok_onnx.py", PASS_SCRIPT)
    _mk(tmp_path, "curriculum/phase1/ch2_y/export_noskip_onnx.py", "print('x')")
    code = cep.main(["--root", str(tmp_path), "--python", sys.executable])
    assert code == 0
    assert "skipped" in capsys.readouterr().out


def test_main_no_scripts_is_ok(tmp_path, capsys):
    code = cep.main(["--root", str(tmp_path)])
    assert code == 0
    assert "no export_" in capsys.readouterr().out
