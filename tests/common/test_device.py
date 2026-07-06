from curriculum.common import wallclock
from curriculum.common.device import banner, detect_device

HEADER = "chapter,tier,wallclock_min,config_hash,commit,date,status\n"


def test_detect_device_returns_valid_value():
    assert detect_device() in {"cpu", "mps", "cuda"}


def test_banner_prints_measured_wallclock(monkeypatch, capsys, tmp_path):
    # Measured rows for every tier, so this passes on cpu, mps, and cuda hosts.
    fake_csv = tmp_path / "wallclock.csv"
    fake_csv.write_text(
        HEADER
        + "ch1.1-bc,cpu-laptop,12.5,abc123,deadbee,2026-07-01,MEASURED\n"
        + "ch1.1-bc,mps,12.5,abc123,deadbee,2026-07-01,MEASURED\n"
        + "ch1.1-bc,t4,12.5,abc123,deadbee,2026-07-01,MEASURED\n"
    )
    monkeypatch.setattr(wallclock, "WALLCLOCK_CSV", fake_csv)

    banner("ch1.1-bc")
    out = capsys.readouterr().out

    assert "ch1.1-bc" in out
    assert "device: " in out
    assert "tier: " in out
    assert "~12.5 min (measured)" in out


def test_banner_prints_unmeasured_without_crashing(monkeypatch, capsys, tmp_path):
    fake_csv = tmp_path / "wallclock.csv"
    fake_csv.write_text(HEADER)  # no rows at all
    monkeypatch.setattr(wallclock, "WALLCLOCK_CSV", fake_csv)

    banner("ch2.3-mjx")
    out = capsys.readouterr().out

    assert "ch2.3-mjx" in out
    assert "not yet measured" in out
    assert "~" not in out  # never a guessed number
