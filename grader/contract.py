"""Fail-closed validation of a submitted ONNX policy against tensor contract v1.

This REUSES curriculum/common/export_onnx.py — the same contract the exporter
stamps and the playground enforces. We import its key/name/version constants so
the grader can never drift from the exporter (grader/CLAUDE.md: "import, never
copy").

The gate is fail-CLOSED: a submission is rejected unless every check passes.
Checks, in order:
  1. file exists and is within the sandbox size cap (policy.yaml max_file_mb)
  2. loads via onnxruntime ONLY (never torch.load / pickle — policy.yaml)
  3. the graph declares input "observation" [1, obs_dim] and output "action"
     [1, act_dim] (static shapes per contract v1)
  4. metadata_props carry z2r_contract_version="v1" and decimal z2r_obs_dim /
     z2r_act_dim that AGREE with the graph shapes
  5. the declared dims match the scoring env (PushT: obs=10, act=2)
  6. opset within the allowlist IF the allowlist has been frozen (seam: the
     allowlist is populated from export_onnx.py's emitted opset at Drop 4 —
     until then policy.yaml's opset_allowlist flag gates whether we enforce)

Loading is onnxruntime-only by construction: we never call torch.load or
onnx-to-torch here; the only executor is onnxruntime.InferenceSession.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import onnx
import onnxruntime as ort

# Import the contract constants from the exporter so we cannot drift from it.
from curriculum.common.export_onnx import (
    CONTRACT_VERSION,
    INPUT_NAME,
    META_ACT_DIM_KEY,
    META_OBS_DIM_KEY,
    META_VERSION_KEY,
    OUTPUT_NAME,
)

from .sandbox import SandboxPolicy, load_policy

# --- opset allowlist seam ---------------------------------------------------
# HUMAN-OWNED at Drop 4 freeze: populate from export_onnx.py's emitted opset
# (policy.yaml: "populated from export_onnx.py's emitted opset at Drop 4
# freeze"). While None, the allowlist is NOT yet frozen and opset is not
# gated even if policy.yaml flips opset_allowlist on — we refuse to invent an
# allowlist the exporter has not committed to. Set to e.g. frozenset({17, 18}).
FROZEN_OPSET_ALLOWLIST: frozenset[int] | None = None


class ContractError(ValueError):
    """A submission violated tensor contract v1 / the sandbox policy."""


@dataclass(frozen=True)
class ValidatedPolicy:
    """Result of a passing validation — the dims the scorer will run with."""

    onnx_path: Path
    obs_dim: int
    act_dim: int
    contract_version: str
    opset: int


def validate_submission(
    onnx_path: str | Path,
    *,
    expected_obs_dim: int,
    expected_act_dim: int,
    policy: SandboxPolicy | None = None,
) -> ValidatedPolicy:
    """Validate an ONNX file against contract v1. Raise ContractError on any
    violation; return a ValidatedPolicy on success (fail-closed)."""
    policy = policy or load_policy()
    onnx_path = Path(onnx_path)

    if not policy.loads_onnxruntime_only:
        raise ContractError(
            f"sandbox policy load_via={policy.onnx_load_via!r} is not "
            "onnxruntime-only; refusing to score under a relaxed loader"
        )

    # 1. existence + size cap ------------------------------------------------
    if not onnx_path.is_file():
        raise ContractError(f"no such ONNX file: {onnx_path}")
    size = onnx_path.stat().st_size
    if size > policy.onnx_max_file_bytes:
        raise ContractError(
            f"ONNX file {size / 1e6:.1f} MB exceeds sandbox cap "
            f"{policy.onnx_max_file_mb} MB"
        )
    if size == 0:
        raise ContractError(f"empty ONNX file: {onnx_path}")

    # 2. parse (structure) + confirm it loads under onnxruntime only ---------
    try:
        proto = onnx.load(str(onnx_path))
        onnx.checker.check_model(proto)
    except Exception as exc:  # malformed protobuf, bad graph, etc.
        raise ContractError(f"not a valid ONNX model: {exc}") from exc

    opset = max((imp.version for imp in proto.opset_import), default=0)

    # 3. graph I/O names + static [1, dim] shapes ----------------------------
    graph_obs_dim = _static_vector_dim(proto, INPUT_NAME, kind="input")
    graph_act_dim = _static_vector_dim(proto, OUTPUT_NAME, kind="output")

    # 4. metadata_props present + agree with the graph -----------------------
    meta = {p.key: p.value for p in proto.metadata_props}
    version = meta.get(META_VERSION_KEY)
    if version != CONTRACT_VERSION:
        raise ContractError(
            f"{META_VERSION_KEY}={version!r}, expected {CONTRACT_VERSION!r} "
            "— re-export with curriculum/common/export_onnx.py"
        )
    meta_obs = _decimal_meta(meta, META_OBS_DIM_KEY)
    meta_act = _decimal_meta(meta, META_ACT_DIM_KEY)
    if meta_obs != graph_obs_dim:
        raise ContractError(
            f"{META_OBS_DIM_KEY}={meta_obs} disagrees with graph input dim "
            f"{graph_obs_dim}"
        )
    if meta_act != graph_act_dim:
        raise ContractError(
            f"{META_ACT_DIM_KEY}={meta_act} disagrees with graph output dim "
            f"{graph_act_dim}"
        )

    # 5. dims match the scoring env ------------------------------------------
    if graph_obs_dim != expected_obs_dim or graph_act_dim != expected_act_dim:
        raise ContractError(
            f"policy dims obs={graph_obs_dim} act={graph_act_dim} do not match "
            f"the scoring env obs={expected_obs_dim} act={expected_act_dim}"
        )

    # 6. opset allowlist (only once the allowlist is frozen) -----------------
    if policy.onnx_opset_allowlist and FROZEN_OPSET_ALLOWLIST is not None:
        if opset not in FROZEN_OPSET_ALLOWLIST:
            raise ContractError(
                f"opset {opset} not in the frozen allowlist "
                f"{sorted(FROZEN_OPSET_ALLOWLIST)}"
            )

    # 7. final belt-and-suspenders: it must instantiate under onnxruntime ----
    try:
        ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    except Exception as exc:
        raise ContractError(f"onnxruntime refused the model: {exc}") from exc

    return ValidatedPolicy(
        onnx_path=onnx_path,
        obs_dim=graph_obs_dim,
        act_dim=graph_act_dim,
        contract_version=version,
        opset=opset,
    )


def _static_vector_dim(proto: onnx.ModelProto, name: str, *, kind: str) -> int:
    """Return dim d of the [1, d] tensor named `name`; raise if missing/dynamic."""
    seq = proto.graph.input if kind == "input" else proto.graph.output
    value_info = next((v for v in seq if v.name == name), None)
    if value_info is None:
        have = [v.name for v in seq]
        raise ContractError(
            f"contract v1 requires an {kind} named {name!r}; found {have}"
        )
    dims = value_info.type.tensor_type.shape.dim
    if len(dims) != 2:
        raise ContractError(
            f"{kind} {name!r} must be rank-2 [1, dim]; got rank {len(dims)}"
        )
    batch, feat = dims[0], dims[1]
    # contract v1 is static [1, dim]; a dynamic/symbolic dim (dim_param) fails.
    if batch.dim_param or batch.dim_value != 1:
        raise ContractError(
            f"{kind} {name!r} batch dim must be static 1 (contract v1)"
        )
    if feat.dim_param or feat.dim_value <= 0:
        raise ContractError(
            f"{kind} {name!r} feature dim must be a positive static int "
            "(contract v1)"
        )
    return int(feat.dim_value)


def _decimal_meta(meta: dict[str, str], key: str) -> int:
    value = meta.get(key)
    if value is None or not value.isdigit():
        raise ContractError(
            f"metadata {key}={value!r} missing or not a decimal integer"
        )
    return int(value)
