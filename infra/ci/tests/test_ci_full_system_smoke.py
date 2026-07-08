"""full_system_smoke: stage registry consistency + stage selection logic.

Stages shell out to real tools; here we exercise the pure orchestration (which
stages run for which flags) and the registry invariants, not the subprocesses."""

import argparse

import full_system_smoke as fss


def _args(**kw):
    base = {"only": None, "skip": None, "with_notebooks": False}
    base.update(kw)
    return argparse.Namespace(**base)


def test_registry_and_fns_agree():
    # Every advertised stage has an implementation and vice versa.
    assert set(fss.STAGES) == set(fss.STAGE_FNS)
    assert all(s in fss.STAGES for s in fss.DEFAULT_STAGES)


def test_default_stage_set_excludes_notebooks():
    # The heavy notebook lane is opt-in — never in the default whole-product run.
    stages = fss.select_stages(_args())
    assert stages == fss.DEFAULT_STAGES
    assert "notebooks" not in stages


def test_with_notebooks_appends():
    stages = fss.select_stages(_args(with_notebooks=True))
    assert stages[-1] == "notebooks"
    assert stages[:-1] == fss.DEFAULT_STAGES


def test_only_overrides():
    stages = fss.select_stages(_args(only="export-parity, gates"))
    assert stages == ["export-parity", "gates"]


def test_skip_filters():
    stages = fss.select_stages(_args(skip="site-build"))
    assert "site-build" not in stages
    assert "gates" in stages


def test_list_flag(capsys):
    code = fss.main(["--list"])
    assert code == 0
    out = capsys.readouterr().out
    assert "notebooks" in out and "(default)" in out


def test_unknown_stage_fails(capsys):
    code = fss.main(["--only", "bogus"])
    assert code == 1
    assert "unknown stage" in capsys.readouterr().err


def test_no_stages_selected_is_ok(capsys):
    code = fss.main(["--only", "gates", "--skip", "gates"])
    assert code == 0
    assert "no stages selected" in capsys.readouterr().out
