from curriculum.common import wallclock

HEADER = "chapter,tier,wallclock_min,config_hash,commit,date,status\n"


def _write_csv(tmp_path, rows):
    path = tmp_path / "wallclock.csv"
    path.write_text(HEADER + "".join(row + "\n" for row in rows))
    return path


def test_lookup_measured(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            "ch1.1-bc,t4,12.5,abc123,deadbee,2026-07-01,MEASURED",
            "ch1.1-bc,cpu-laptop,,,,,PENDING",
        ],
    )
    assert wallclock.lookup("ch1.1-bc", "t4", path) == 12.5


def test_lookup_pending_is_none(tmp_path):
    path = _write_csv(tmp_path, ["ch1.1-bc,cpu-laptop,,,,,PENDING"])
    assert wallclock.lookup("ch1.1-bc", "cpu-laptop", path) is None


def test_lookup_absent_is_none(tmp_path):
    path = _write_csv(tmp_path, ["ch1.1-bc,t4,12.5,abc123,deadbee,2026-07-01,MEASURED"])
    assert wallclock.lookup("ch1.1-bc", "4090", path) is None
    assert wallclock.lookup("ch9.9-nope", "t4", path) is None


def test_lookup_missing_file_is_none(tmp_path):
    assert wallclock.lookup("ch1.1-bc", "t4", tmp_path / "nope.csv") is None


def test_render_line_measured_exact(tmp_path):
    path = _write_csv(
        tmp_path,
        [
            "ch1.1-bc,t4,12.5,abc123,deadbee,2026-07-01,MEASURED",
            "ch1.1-bc,4090,25.0,abc123,deadbee,2026-07-01,MEASURED",
        ],
    )
    assert (
        wallclock.render_line("ch1.1-bc", "t4", path)
        == "expected wall-clock on t4: ~12.5 min (measured)"
    )
    # integral minutes render without a trailing .0
    assert (
        wallclock.render_line("ch1.1-bc", "4090", path)
        == "expected wall-clock on 4090: ~25 min (measured)"
    )


def test_render_line_unmeasured_exact(tmp_path):
    path = _write_csv(tmp_path, ["ch1.1-bc,cpu-laptop,,,,,PENDING"])
    assert (
        wallclock.render_line("ch1.1-bc", "cpu-laptop", path)
        == "wall-clock on cpu-laptop: not yet measured"
    )
    # absent row renders the same honest line — never a guessed number
    assert (
        wallclock.render_line("ch1.1-bc", "t4", path)
        == "wall-clock on t4: not yet measured"
    )


def test_default_csv_ships_with_repo():
    # The real wallclock.csv exists and parses; unmeasured rows read as None.
    assert wallclock.WALLCLOCK_CSV.exists()
    assert wallclock.lookup("ch2.3-mjx", "t4") is None   # ch2.3 not yet benched
