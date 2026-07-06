"""Loader for the gVisor sandbox contract (grader/sandbox/policy.yaml).

The sandbox itself (gVisor container, network namespace, cgroup caps) is
HUMAN/INFRA-owned and enforced OUTSIDE this process. This module exists so the
Python harness reads its caps from the ONE policy file rather than hardcoding
magic numbers: contract validation uses `max_file_mb`, the scoring guard uses
`wallclock_limit_s`, and both refuse anything the policy forbids
(`onnx.load_via` must be "onnxruntime only").

Integration seam: a real deployment runs `grader.scoring` as the entrypoint of
a gVisor sandbox stood up per policy.yaml. Nothing here starts that sandbox —
it only lets the in-process code stay faithful to the same contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

POLICY_PATH = Path(__file__).parent / "sandbox" / "policy.yaml"


@dataclass(frozen=True)
class SandboxPolicy:
    """Parsed grader/sandbox/policy.yaml. Caps the harness must honor."""

    isolation: str
    network: str
    cpu_limit: str
    memory_limit: str
    wallclock_limit_s: int
    onnx_max_file_mb: int
    onnx_opset_allowlist: bool
    onnx_load_via: str

    @property
    def onnx_max_file_bytes(self) -> int:
        return self.onnx_max_file_mb * 1024 * 1024

    @property
    def loads_onnxruntime_only(self) -> bool:
        # policy.yaml value is "onnxruntime only" — never torch.load / pickle.
        return "onnxruntime" in self.onnx_load_via and "only" in self.onnx_load_via


def load_policy(path: str | Path = POLICY_PATH) -> SandboxPolicy:
    """Parse policy.yaml into a SandboxPolicy. Raises on a malformed file."""
    raw = yaml.safe_load(Path(path).read_text())
    onnx = raw["onnx"]
    return SandboxPolicy(
        isolation=str(raw["isolation"]),
        network=str(raw["network"]),
        cpu_limit=str(raw["cpu_limit"]),
        memory_limit=str(raw["memory_limit"]),
        wallclock_limit_s=int(raw["wallclock_limit_s"]),
        onnx_max_file_mb=int(onnx["max_file_mb"]),
        onnx_opset_allowlist=bool(onnx["opset_allowlist"]),
        onnx_load_via=str(onnx["load_via"]),
    )
