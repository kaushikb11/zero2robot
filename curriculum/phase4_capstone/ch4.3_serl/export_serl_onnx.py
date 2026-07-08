"""Export the trained ch4.3 HIL-SERL actors to ONNX (tensor contract v1).

serl.py runs HIL-SERL on PusherReach and `torch.save`s both demo panels'
Actor state_dicts to outputs/ch4.3-serl/:

  serl_actor     the HIL-SERL best-checkpoint prior (the "hil" panel).
  scratch_actor  the from-scratch SAC baseline (the "scratch" panel).

Both are the SAME Actor network (identical architecture), so one export path
covers both. Those checkpoints are the source of truth; this is a FOLLOW-UP script
(kept out of serl.py's readable loop, per the embed.yaml note that the ONNX export
is wired when the chapter graduates from spike to drop).

The Actor is the ch2.2 squashed-Gaussian policy (trunk + mean + log_std). Eval and
the browser act with the DETERMINISTIC mean — tanh(mean), no sampling — exactly the
deterministic policy serl.py's held-out eval uses. Each exported policy is that
tanh(mean) path as a stateless obs[1,8] -> action[1,2] nn.Module (tensor contract
v1; see curriculum/common/export_onnx.py).

Run with the repo venv, from the repo root:
  .venv/bin/python curriculum/phase4_capstone/ch4.3_serl/export_serl_onnx.py
Smoke (no checkpoints; random-init actors, proves the export+parity path):
  .venv/bin/python curriculum/phase4_capstone/ch4.3_serl/export_serl_onnx.py --smoke
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
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # must match serl.py

# The two demo panels: checkpoint stem -> exported filename (also the site path).
ARMS = {"serl_actor": "serl_actor.onnx", "scratch_actor": "scratch_actor.onnx"}
SITE_MODELS = REPO_ROOT / "site" / "public" / "models"


class Actor(nn.Module):
    """Byte-for-byte the serl.py Actor (trunk + mean + log_std heads), redefined
    here because serl.py runs argparse/training at import time and cannot be
    imported as a module. Only the parameter layout matters — we load the trained
    weights, then export the tanh(mean) deterministic path alone."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.mean(h), torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)


class DeterministicPolicy(nn.Module):
    """Stateless obs -> action wrapper: tanh(mean), no sampling. Exactly the
    deterministic policy serl.py's held-out eval uses, so the browser and the
    Python eval share one function. Contract v1: [1,8]->[1,2]."""

    def __init__(self, actor: Actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.actor(observation)
        return torch.tanh(mean)


def load_policy(ckpt: Path, hidden_dim: int) -> DeterministicPolicy:
    actor = Actor(OBS_DIM, ACT_DIM, hidden_dim)
    actor.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return DeterministicPolicy(actor)


def smoke_policy(seed: int = 0, hidden_dim: int = 64) -> tuple[DeterministicPolicy, int, int]:
    """A random-init deterministic policy — no checkpoint, no env rollout needed.
    Exercises the exact Actor architecture + export path in CI (hermetic)."""
    torch.manual_seed(seed)
    return DeterministicPolicy(Actor(OBS_DIM, ACT_DIM, hidden_dim).eval()), OBS_DIM, ACT_DIM


def export_arm(policy: DeterministicPolicy, out: Path, onnx_name: str, copy_site: bool) -> None:
    onnx_path = export_policy(policy, OBS_DIM, ACT_DIM, out)
    parity_delta = assert_parity(policy, onnx_path, OBS_DIM)  # torch vs onnxruntime before it ships
    print(f"exported {onnx_path} — torch/onnx parity delta {parity_delta:.2e}")
    if copy_site:
        SITE_MODELS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx_path, SITE_MODELS / onnx_name)
        print(f"provisioned {SITE_MODELS / onnx_name} (git-ignored — not committed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_dir", type=Path, default=Path("outputs/ch4.3-serl"),
                        help="dir holding serl_actor.pt / scratch_actor.pt saved by serl.py")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ch4.3-serl"))
    parser.add_argument("--hidden_dim", type=int, default=256)  # serl.py default
    parser.add_argument("--smoke", action="store_true",
                        help="skip the checkpoints; export random-init actors (CI parity smoke)")
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
                         f"  .venv/bin/python curriculum/phase4_capstone/ch4.3_serl/serl.py --seed 0 --device cpu")
            policy = load_policy(ckpt, args.hidden_dim)
        export_arm(policy, args.out_dir / onnx_name, onnx_name, args.copy_site and not args.smoke)


if __name__ == "__main__":
    main()
