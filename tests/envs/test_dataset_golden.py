"""Golden-file test: gen_demos output must match the LeRobot-v3 fixture.

The fixture (tests/envs/fixtures/pusht_mini) was produced by
`gen_demos.py --episodes 2 --seed 0 --no-video`. Because the env and expert
are bitwise deterministic, regenerating with the same arguments must
reproduce the fixture's data values exactly -- this pins both the dataset
schema AND the physics in one test.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from curriculum.common.envs.pusht import PushTEnv
from curriculum.common.envs.pusht import gen_demos

FIXTURE = Path(__file__).parent / "fixtures" / "pusht_mini"
DATA_REL = "data/chunk-000/file-000.parquet"
EPISODES_REL = "meta/episodes/chunk-000/file-000.parquet"

# The pusht_mini fixture was recorded on macOS arm64. MuJoCo's CPU solver is
# bitwise-reproducible WITHIN a platform but not ACROSS architectures, so on
# Linux x86_64 (CI, Colab) the expert's trajectory terminates at a different
# frame count (169 vs the fixture's 180) and the exact-value compare diverges.
# Gate the two fixture-comparing tests to the recording platform until the
# goldens are regenerated on the Linux learner tier. The physics/schema is still
# covered off-platform by test_row_counts_and_bounds (self-consistent, no golden).
# TODO(ci-platform-goldens): regenerate goldens on Linux x86_64 and drop this
# guard -- tracked in infra/decisions/020-ci-platform-goldens.md.
skip_offplatform_golden = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="golden recorded on macOS arm64; MuJoCo FP differs on Linux x86_64 "
    "(169 vs 180 frames) -- regenerate goldens on Linux (020-ci-platform-goldens.md)",
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("demos") / "pusht_mini"
    gen_demos.main(["--episodes", "2", "--seed", "0", "--out", str(out), "--no-video"])
    return out


@skip_offplatform_golden
def test_info_schema(generated: Path):
    info = json.loads((generated / "meta" / "info.json").read_text())
    golden = json.loads((FIXTURE / "meta" / "info.json").read_text())

    assert info["codebase_version"] == "v3.0"
    assert info["fps"] == PushTEnv.CONTROL_HZ
    assert set(info["features"]) == set(golden["features"]) == {
        "observation.state", "action", "timestamp",
        "frame_index", "episode_index", "index", "task_index",
    }
    for key in ("observation.state", "action"):
        assert info["features"][key]["dtype"] == "float32"
        assert info["features"][key]["shape"] == golden["features"][key]["shape"]
    assert info["features"]["observation.state"]["shape"] == [PushTEnv.OBS_DIM]
    assert info["features"]["action"]["shape"] == [PushTEnv.ACT_DIM]
    assert info["total_episodes"] == 2
    assert info["total_frames"] == golden["total_frames"]


@skip_offplatform_golden
def test_data_matches_fixture(generated: Path):
    df = pd.read_parquet(generated / DATA_REL)
    golden = pd.read_parquet(FIXTURE / DATA_REL)

    assert list(df.columns) == list(golden.columns)
    assert [str(t) for t in df.dtypes] == [str(t) for t in golden.dtypes]
    assert len(df) == len(golden)

    # bitwise-deterministic env + expert => identical values on regeneration
    for col in ("observation.state", "action"):
        got = np.stack(df[col].to_numpy())
        want = np.stack(golden[col].to_numpy())
        assert got.tobytes() == want.tobytes(), f"{col} diverged from fixture"
    for col in ("frame_index", "episode_index", "index", "task_index"):
        np.testing.assert_array_equal(df[col].to_numpy(), golden[col].to_numpy())


def test_row_counts_and_bounds(generated: Path):
    df = pd.read_parquet(generated / DATA_REL)
    episodes = pd.read_parquet(generated / EPISODES_REL)

    assert len(episodes) == 2
    assert episodes["length"].sum() == len(df)
    per_episode = df.groupby("episode_index").size()
    np.testing.assert_array_equal(
        per_episode.sort_index().to_numpy(), episodes["length"].to_numpy()
    )

    actions = np.stack(df["action"].to_numpy())
    assert actions.dtype == np.float32
    assert actions.shape[1] == PushTEnv.ACT_DIM
    assert np.all(np.abs(actions) <= 1.0), "actions must lie in [-1, 1]"

    states = np.stack(df["observation.state"].to_numpy())
    assert states.shape[1] == PushTEnv.OBS_DIM
    # timestamps advance at 1/fps within each episode
    for _, ep in df.groupby("episode_index"):
        dt = np.diff(ep["timestamp"].to_numpy())
        assert np.allclose(dt, 1.0 / PushTEnv.CONTROL_HZ, atol=1e-6)
