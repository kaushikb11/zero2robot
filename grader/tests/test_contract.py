"""Fail-closed contract validation: good passes, everything else is rejected."""

from __future__ import annotations

import dataclasses

import pytest

from curriculum.common.envs.pusht import PushTEnv
from grader.contract import ContractError, validate_submission
from grader.sandbox import load_policy


def _validate(path):
    return validate_submission(
        path, expected_obs_dim=PushTEnv.OBS_DIM, expected_act_dim=PushTEnv.ACT_DIM
    )


def test_valid_policy_passes(sample_onnx):
    v = _validate(sample_onnx)
    assert v.obs_dim == PushTEnv.OBS_DIM
    assert v.act_dim == PushTEnv.ACT_DIM
    assert v.contract_version == "v1"
    assert v.opset > 0


def test_missing_file_rejected(tmp_path):
    with pytest.raises(ContractError, match="no such ONNX"):
        _validate(tmp_path / "nope.onnx")


def test_wrong_dims_rejected(wrong_dims_onnx):
    # metadata is internally consistent, but the dims don't match PushT.
    with pytest.raises(ContractError, match="do not match the scoring env"):
        _validate(wrong_dims_onnx)


def test_missing_metadata_rejected(no_metadata_onnx):
    with pytest.raises(ContractError):
        _validate(no_metadata_onnx)


def test_non_onnx_rejected(tmp_path):
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"not an onnx model at all")
    with pytest.raises(ContractError, match="not a valid ONNX"):
        _validate(junk)


def test_empty_file_rejected(tmp_path):
    empty = tmp_path / "empty.onnx"
    empty.write_bytes(b"")
    with pytest.raises(ContractError, match="empty"):
        _validate(empty)


def test_oversize_rejected(sample_onnx):
    # Shrink the sandbox cap so the sample trips it — proves the cap is wired
    # from the policy, not hardcoded.
    policy = load_policy()
    tiny = dataclasses.replace(policy, onnx_max_file_mb=0)
    with pytest.raises(ContractError, match="exceeds sandbox cap"):
        validate_submission(
            sample_onnx,
            expected_obs_dim=PushTEnv.OBS_DIM,
            expected_act_dim=PushTEnv.ACT_DIM,
            policy=tiny,
        )
