"""Thin FastAPI shell over the scorer (leaderboard-submission API).

This is a thin transport layer: it accepts an upload, delegates to
grader.score.grade (the same validate-then-score path the CLI uses), and
returns the report. All the real logic — the fail-closed contract gate, the
deterministic PushT scoring, the reproducible submission hash — lives in the
core modules and is tested WITHOUT this shell.

fastapi/uvicorn are OPTIONAL deps (see infra/decisions/013-grader-deps.md).
Nothing in grader's core imports this module, so the checker and scorer run
under `make check` with only onnx/onnxruntime installed. Import this only to
serve the API:

    pip install -e '.[grader]'          # fastapi + uvicorn
    uvicorn grader.api:app --port 8000

DEPLOYMENT SEAM: in production this app runs as the entrypoint of the gVisor
sandbox described by grader/sandbox/policy.yaml — network none, CPU/time/memory
caps, read-only FS except /tmp. This process trusts that isolation; it does not
re-implement it. The uploaded ONNX is written under /tmp and scored there.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from . import __version__
from .contract import ContractError
from .sandbox import load_policy
from .score import grade
from .submission import Division

app = FastAPI(
    title="zero2robot grader",
    version=__version__,
    summary="Leaderboard submission grading (PushT, tensor contract v1).",
)


@app.get("/health")
def health() -> dict:
    """Liveness + the sandbox caps this process is meant to run under."""
    policy = load_policy()
    return {
        "status": "ok",
        "version": __version__,
        "sandbox": {
            "isolation": policy.isolation,
            "network": policy.network,
            "wallclock_limit_s": policy.wallclock_limit_s,
            "onnx_max_file_mb": policy.onnx_max_file_mb,
            "load_via": policy.onnx_load_via,
        },
    }


@app.post("/score")
async def score_submission_endpoint(
    policy: UploadFile = File(..., description="contract-v1 ONNX policy"),
    config_hash: str = Form(...),
    division: str = Form("free"),
    n_seeds: int = Form(50),
    declared_runtime_min: float | None = Form(None),
) -> dict:
    """Validate + score an uploaded ONNX policy on the public seeds.

    413 if over the sandbox file cap, 422 on a contract violation, 400 on a bad
    division. A passing submission returns the score, band, and reproducible
    submission hash.
    """
    try:
        div = Division(division)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"unknown division {division!r}")

    sandbox = load_policy()
    # Stream to a temp file under /tmp (the one writable mount in the sandbox),
    # enforcing the size cap as we read so a hostile upload can't fill memory.
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as tmp:
        written = 0
        while chunk := await policy.read(1 << 20):
            written += len(chunk)
            if written > sandbox.onnx_max_file_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"ONNX exceeds sandbox cap {sandbox.onnx_max_file_mb} MB",
                )
            tmp.write(chunk)
        tmp.flush()
        try:
            return grade(
                Path(tmp.name),
                config_hash,
                div,
                n_seeds=n_seeds,
                declared_runtime_min=declared_runtime_min,
            )
        except ContractError as exc:
            raise HTTPException(status_code=422, detail=f"contract: {exc}")
