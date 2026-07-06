"""run_wallclock: full-run timing + wallclock_latest.csv appending."""

import csv
import datetime
import sys

import run_wallclock

SLOW_OK_ARTIFACT = """\
import argparse, json, os

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", required=True)
parser.add_argument("--no-rerun", action="store_true")
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
assert not args.smoke  # wallclock runs are FULL runs, never --smoke
os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "metrics.json"), "w") as f:
    json.dump({"loss": 0.1}, f, sort_keys=True)
"""

CRASHING_ARTIFACT = 'import sys\nsys.exit("boom")\n'


def run(tmp_path, chapters="all", tier="cpu-laptop", *extra):
    return run_wallclock.main(
        [
            "--root",
            str(tmp_path),
            "--python",
            sys.executable,
            "--tier",
            tier,
            "--chapters",
            chapters,
            *extra,
        ]
    )


def read_csv(tmp_path):
    path = tmp_path / "infra" / "ci" / "reports" / "wallclock_latest.csv"
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_measures_and_appends_measured_row(make_chapter, tmp_path):
    make_chapter(artifact_text=SLOW_OK_ARTIFACT)
    assert run(tmp_path) == 0
    rows = read_csv(tmp_path)
    assert rows[0] == run_wallclock.CSV_HEADER
    assert len(rows) == 2
    row = dict(zip(run_wallclock.CSV_HEADER, rows[1]))
    assert row["chapter"] == "ch1.1-bc"
    assert row["tier"] == "cpu-laptop"
    assert float(row["wallclock_min"]) >= 0
    assert len(row["config_hash"]) == 12
    assert row["commit"]  # "unknown" outside a git repo, never empty
    assert row["date"] == datetime.date.today().isoformat()
    assert row["status"] == "MEASURED"


def test_appends_across_invocations_without_duplicate_header(
    make_chapter, tmp_path
):
    make_chapter(artifact_text=SLOW_OK_ARTIFACT)
    assert run(tmp_path) == 0
    assert run(tmp_path, "all", "mps") == 0
    rows = read_csv(tmp_path)
    assert len(rows) == 3
    assert rows[1][1] == "cpu-laptop"
    assert rows[2][1] == "mps"


def test_failed_run_records_nothing_and_exits_1(
    make_chapter, tmp_path, capsys
):
    make_chapter(artifact_text=CRASHING_ARTIFACT)
    assert run(tmp_path) == 1
    assert not (
        tmp_path / "infra" / "ci" / "reports" / "wallclock_latest.csv"
    ).exists()
    assert "exit 1" in capsys.readouterr().err


def test_never_touches_canonical_wallclock_csv(
    make_chapter, write_wallclock, tmp_path
):
    make_chapter(artifact_text=SLOW_OK_ARTIFACT)
    canonical = write_wallclock(
        "chapter,tier,wallclock_min,config_hash,commit,date,status\n"
        "ch1.1-bc,cpu-laptop,,,,,PENDING\n"
    )
    before = canonical.read_bytes()
    assert run(tmp_path) == 0
    assert canonical.read_bytes() == before


def test_chapter_dir_selection(make_chapter, tmp_path):
    make_chapter(artifact_text=SLOW_OK_ARTIFACT)
    make_chapter(
        dirname="ch1.2_act",
        chapter_id="ch1.2-act",
        artifact_name="act.py",
        artifact_text=SLOW_OK_ARTIFACT,
    )
    assert run(tmp_path, "curriculum/phase1_imitation/ch1.2_act") == 0
    rows = read_csv(tmp_path)
    assert len(rows) == 2
    assert rows[1][0] == "ch1.2-act"


def test_scale_labs_selection(make_chapter, tmp_path):
    make_chapter(artifact_text=SLOW_OK_ARTIFACT)  # not a scale lab
    make_chapter(
        dirname="ch2.9_lab",
        chapter_id="ch2.9-lab",
        artifact_name="lab.py",
        artifact_text=SLOW_OK_ARTIFACT,
        meta={"scale_lab": True},
    )
    assert run(tmp_path, "scale-labs") == 0
    rows = read_csv(tmp_path)
    assert len(rows) == 2
    assert rows[1][0] == "ch2.9-lab"


def test_bogus_chapters_spec_fails(make_chapter, tmp_path, capsys):
    make_chapter(artifact_text=SLOW_OK_ARTIFACT)
    assert run(tmp_path, "no/such/dir") == 1
    assert "no chapter directory" in capsys.readouterr().err


def test_all_tiers_accepted(make_chapter, tmp_path):
    make_chapter(artifact_text=SLOW_OK_ARTIFACT)
    for tier in run_wallclock.TIERS:
        assert run(tmp_path, "all", tier) == 0


def test_no_chapters_selected_is_success(tmp_path, capsys):
    assert run(tmp_path, "scale-labs") == 0
    assert "nothing to bench" in capsys.readouterr().out
