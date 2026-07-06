"""API shell tests — SKIP entirely when fastapi (optional dep) is absent.

The scorer's real behavior is covered by test_scoring/test_contract without the
API; these only check the transport wiring when fastapi is installed."""

from __future__ import annotations

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

from grader.api import app  # noqa: E402  (after importorskip)

client = fastapi_testclient.TestClient(app)


def test_health_reports_sandbox_caps():
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["sandbox"]["network"] == "none"
    assert body["sandbox"]["load_via"].startswith("onnxruntime")


def test_score_endpoint_scores_a_submission(sample_onnx):
    with open(sample_onnx, "rb") as fh:
        resp = client.post(
            "/score",
            files={"policy": ("sample.onnx", fh, "application/octet-stream")},
            data={"config_hash": "cfg123", "division": "free", "n_seeds": "4"},
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["division"] == "free"
    assert 0.0 <= body["score"] <= 100.0
    assert len(body["submission_hash"]) == 64
    assert body["contract_version"] == "v1"


def test_score_endpoint_rejects_bad_onnx(tmp_path):
    junk = tmp_path / "junk.onnx"
    junk.write_bytes(b"not onnx")
    with open(junk, "rb") as fh:
        resp = client.post(
            "/score",
            files={"policy": ("junk.onnx", fh, "application/octet-stream")},
            data={"config_hash": "c", "division": "free"},
        )
    assert resp.status_code == 422
    assert "contract" in resp.json()["detail"]


def test_score_endpoint_unknown_division(sample_onnx):
    with open(sample_onnx, "rb") as fh:
        resp = client.post(
            "/score",
            files={"policy": ("sample.onnx", fh, "application/octet-stream")},
            data={"config_hash": "c", "division": "platinum"},
        )
    assert resp.status_code == 400
