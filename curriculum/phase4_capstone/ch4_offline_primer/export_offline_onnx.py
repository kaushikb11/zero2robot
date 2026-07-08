"""Export the trained ch4 offline policies to ONNX (tensor contract v1).

offline.py trains BOTH arms of the demo on the mixed expert+random PusherReach
dataset and `torch.save`s each state_dict to outputs/ch4-offline-primer/:

  offline_policy  the AWAC arm — advantage-weighted regression that beats the mix.
  bc_policy       the behavior-cloning baseline the AWAC arm is compared against.

Both are the SAME Policy network (identical architecture), so one export path
covers both. Those checkpoints are the source of truth; this is a FOLLOW-UP script
(kept out of offline.py's readable loop, per the embed.yaml note that the ONNX
export is wired when the chapter graduates from spike to drop).

The Policy is already a stateless deterministic map: obs -> tanh(net(obs)), actions
in [-1, 1]. That IS tensor contract v1 (obs[1,8] -> action[1,2]) with no wrapper —
we load the weights and export directly (see curriculum/common/export_onnx.py).

Run with the repo venv, from the repo root:
  .venv/bin/python curriculum/phase4_capstone/ch4_offline_primer/export_offline_onnx.py
Smoke (no checkpoints; random-init policies, proves the export+parity path):
  .venv/bin/python curriculum/phase4_capstone/ch4_offline_primer/export_offline_onnx.py --smoke
"""

import argparse
import shutil
import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from curriculum.common.assert_parity import assert_parity  # noqa: E402
from curriculum.common.envs.pusher_reach import PusherReachEnv  # noqa: E402
from curriculum.common.export_onnx import export_policy  # noqa: E402

OBS_DIM, ACT_DIM = PusherReachEnv.OBS_DIM, PusherReachEnv.ACT_DIM  # 8, 2

# The two demo panels: checkpoint stem -> exported filename (also the site path).
ARMS = {"offline_policy": "offline_policy.onnx", "bc_policy": "offline_bc.onnx"}  # offline_bc (NOT bc_policy) — ch1.1 owns bc_policy.onnx in the shared /models/ namespace
SITE_MODELS = REPO_ROOT / "site" / "public" / "models"


class Policy(nn.Module):
    """Byte-for-byte the offline.py Policy (obs -> action MLP, tanh-bounded),
    redefined here because offline.py runs argparse/training at import time and
    cannot be imported. Already a contract-v1 stateless map — no deterministic
    wrapper needed (forward IS obs[1,8] -> action[1,2] in [-1, 1])."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(observation))  # actions live in [-1, 1]


def load_policy(ckpt: Path, hidden_dim: int) -> Policy:
    policy = Policy(OBS_DIM, ACT_DIM, hidden_dim)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return policy


def smoke_policy(seed: int = 0, hidden_dim: int = 64) -> tuple[Policy, int, int]:
    """A random-init policy — no checkpoint, no env rollout needed. Exercises the
    exact Policy architecture + export path in CI (hermetic)."""
    torch.manual_seed(seed)
    return Policy(OBS_DIM, ACT_DIM, hidden_dim).eval(), OBS_DIM, ACT_DIM


def export_arm(policy: Policy, out: Path, onnx_name: str, copy_site: bool) -> None:
    onnx_path = export_policy(policy, OBS_DIM, ACT_DIM, out)
    parity_delta = assert_parity(policy, onnx_path, OBS_DIM)  # torch vs onnxruntime before it ships
    print(f"exported {onnx_path} — torch/onnx parity delta {parity_delta:.2e}")
    if copy_site:
        SITE_MODELS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx_path, SITE_MODELS / onnx_name)
        print(f"provisioned {SITE_MODELS / onnx_name} (git-ignored — not committed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_dir", type=Path, default=Path("outputs/ch4-offline-primer"),
                        help="dir holding offline_policy.pt / bc_policy.pt saved by offline.py")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ch4-offline-primer"))
    parser.add_argument("--hidden_dim", type=int, default=256)  # offline.py default
    parser.add_argument("--smoke", action="store_true",
                        help="skip the checkpoints; export random-init policies (CI parity smoke)")
    parser.add_argument("--no-copy", dest="copy_site", action="store_false", default=True,
                        help="skip copying the .onnx files into site/public/models/")
    args = parser.parse_args()

    for stem, onnx_name in ARMS.items():
        if args.smoke:
            policy, _, _ = smoke_policy(seed=0, hidden_dim=args.hidden_dim)
        else:
            ckpt = args.ckpt_dir / f"{stem}.pt"
            if not ckpt.is_file():
                sys.exit(f"no checkpoint at {ckpt} — train it first:\n"
                         f"  .venv/bin/python curriculum/phase4_capstone/ch4_offline_primer/offline.py --seed 0 --device cpu")
            policy = load_policy(ckpt, args.hidden_dim)
        export_arm(policy, args.out_dir / onnx_name, onnx_name, args.copy_site and not args.smoke)


if __name__ == "__main__":
    main()
