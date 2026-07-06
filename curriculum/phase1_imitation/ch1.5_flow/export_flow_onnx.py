"""Export the trained ch1.5 flow velocity net to ONNX under tensor contract v2.

flow.py trains a flow-matching PushT policy and writes the velocity-net CORE to
outputs/ch1.5-flow/flow_velocity.onnx (3 inputs: point, flow_time, observation ->
velocity) — but with NO contract metadata, so the browser cannot drive it. That
raw ONNX is the source of truth for the weights; this is a FOLLOW-UP script (kept
out of flow.py's readable loop, per the embed.yaml contract note) that turns it
into a contract-v2 artifact the sampler-aware runtime can load:

  - rebuild flow.py's VelocityNet and load the trained weights straight from the
    raw ONNX initializers (net.*, obs_min, obs_range) — the obs normalization is
    baked inside the net, exactly as flow.py exports it;
  - recompute the per-dim action mean/std from the SAME deterministic demos
    flow.py trained on (the un-standardization stats the runtime needs — flow.py
    applies them OUTSIDE the net, so they must ship as metadata);
  - export with export_sampler_policy (stamps z2r_contract_version=v2 + the io
    spec + sampler metadata) and prove torch==onnx velocity parity on random
    (point, flow_time, obs) BEFORE the file ships;
  - copy the .onnx to site/public/models/ (git-ignored, provisioned like
    bc_policy.onnx / ppo_agent.onnx; never committed).

Run with the repo venv, from the repo root (same env flow.py trained under):
  HF_HUB_OFFLINE=1 HF_TOKEN= .venv/bin/python \
    curriculum/phase1_imitation/ch1.5_flow/export_flow_onnx.py

Prereq: flow.py has been trained so outputs/ch1.5-flow/{flow_velocity.onnx,demos}
exist:
  HF_HUB_OFFLINE=1 HF_TOKEN= .venv/bin/python \
    curriculum/phase1_imitation/ch1.5_flow/flow.py --seed 0 --device cpu
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

# Same offline posture flow.py's Colab/CI runs under; harmless if already set.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from curriculum.common.envs.pusht import PushTEnv  # noqa: E402
from curriculum.common.export_onnx import (  # noqa: E402
    assert_sampler_parity,
    export_sampler_policy,
)

OBS_DIM, ACT_DIM = PushTEnv.OBS_DIM, PushTEnv.ACT_DIM  # 10, 2
TIME_DIM = 32       # must match flow.py.TIME_DIM
TIME_SCALE = 1000.0  # must match flow.py.TIME_SCALE

# The site fetches this path (git-ignored, provisioned like bc_policy.onnx).
SITE_MODEL = REPO_ROOT / "site" / "public" / "models" / "flow_velocity.onnx"


def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Byte-for-byte flow.py.sinusoidal_embed — the velocity net's time features."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class VelocityNet(nn.Module):
    """Byte-for-byte flow.py.VelocityNet (conditioned variant). Redefined here
    because flow.py runs argparse/training at import time and cannot be imported;
    only the parameter/buffer layout matters — we load the trained weights from
    flow.py's raw ONNX, then re-export the CORE under contract v2."""

    def __init__(self, x_dim: int, cond_dim: int, hidden: int):
        super().__init__()
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + TIME_DIM + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )
        # Registered so load_state_dict restores the baked-in obs normalization.
        self.register_buffer("obs_min", torch.zeros(cond_dim))
        self.register_buffer("obs_range", torch.ones(cond_dim))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        parts = [x_t, sinusoidal_embed(t * TIME_SCALE, TIME_DIM)]
        parts.append((2.0 * (cond - self.obs_min) / self.obs_range - 1.0).clamp(-1.0, 1.0))
        return self.net(torch.cat(parts, dim=1))


def load_weights_from_onnx(net: VelocityNet, raw_onnx: Path) -> None:
    """Load flow.py's trained weights straight from its raw ONNX initializers.
    torch.onnx.export (dynamo=False) names initializers by their state_dict keys
    (net.0.weight, ..., obs_min, obs_range), so this is a direct, strict load."""
    proto = onnx.load(str(raw_onnx))
    state = {init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
             for init in proto.graph.initializer}
    missing, unexpected = net.load_state_dict(state, strict=False)
    # obs_min/obs_range + net.{0,2,4}.{weight,bias} — 8 tensors, all present.
    if missing:
        raise SystemExit(f"weights missing from {raw_onnx}: {missing}")
    # Non-parameter graph initializers (constants) may appear; ignore them, but
    # every net/buffer param must have been filled (missing == [] above).
    return None


def action_stats(demos: Path) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the per-dim action mean/std EXACTLY as flow.py's data region:
    load the same deterministic demos, standardize stats over all frames."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # heavy import

    frames = LeRobotDataset("local/pusht-demos", root=demos).hf_dataset.with_format("numpy")
    actions = np.stack(frames["action"]).astype(np.float32)  # (N, 2)
    act_mean = actions.mean(0)
    act_std = np.where(actions.std(0) < 1e-4, np.float32(1.0), actions.std(0))
    return act_mean, act_std


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/ch1.5-flow"),
                        help="flow.py output dir (holds flow_velocity.onnx + demos/)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output .onnx (default: <run>/flow_velocity_v2.onnx)")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="default Euler steps stamped into the contract (flow.py default)")
    parser.add_argument("--no-copy", dest="copy_site", action="store_false", default=True,
                        help="skip copying the .onnx into site/public/models/")
    args = parser.parse_args()

    raw_onnx = args.run / "flow_velocity.onnx"
    demos = args.run / "demos"
    out = args.out or (args.run / "flow_velocity_v2.onnx")
    if not raw_onnx.is_file():
        sys.exit(f"no raw velocity ONNX at {raw_onnx} — train it first:\n"
                 f"  HF_HUB_OFFLINE=1 HF_TOKEN= .venv/bin/python "
                 f"curriculum/phase1_imitation/ch1.5_flow/flow.py --seed 0 --device cpu")
    if not (demos / "meta" / "info.json").is_file():
        sys.exit(f"no demos at {demos} — flow.py writes them next to flow_velocity.onnx.")

    # Rebuild the net at flow.py's hidden width (read off the first Linear) and
    # load the trained weights from the raw ONNX.
    proto = onnx.load(str(raw_onnx))
    inits = {i.name: numpy_helper.to_array(i) for i in proto.graph.initializer}
    hidden = int(inits["net.0.weight"].shape[0])
    net = VelocityNet(ACT_DIM, OBS_DIM, hidden)
    load_weights_from_onnx(net, raw_onnx)

    act_mean, act_std = action_stats(demos)
    print(f"action stats (from {demos.name}): mean={act_mean.tolist()} std={act_std.tolist()}")

    onnx_path = export_sampler_policy(
        net, OBS_DIM, ACT_DIM, args.num_steps, act_mean, act_std, out,
    )
    # Prove torch and onnxruntime agree on the velocity net BEFORE it ships.
    delta = assert_sampler_parity(net, onnx_path, OBS_DIM, ACT_DIM)
    print(f"exported {onnx_path} (contract v2, hidden={hidden}, "
          f"num_steps={args.num_steps}); torch/onnx velocity parity {delta:.2e}")

    if args.copy_site:
        SITE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx_path, SITE_MODEL)
        print(f"provisioned {SITE_MODEL} (git-ignored — not committed)")


if __name__ == "__main__":
    main()
