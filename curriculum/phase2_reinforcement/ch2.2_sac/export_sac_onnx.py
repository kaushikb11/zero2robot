"""Export the trained ch2.2 SAC actor to ONNX (tensor contract v1) for the site.

sac.py trains PusherReach and `torch.save`s the Actor state_dict to
outputs/ch2.2-sac/sac_actor.pt. That checkpoint is the source of truth; this is a
FOLLOW-UP script (kept out of sac.py's readable loop, per the embed.yaml note that
the ONNX export is wired when the chapter graduates from spike to drop).

The Actor is a squashed-Gaussian policy: trunk(obs) feeds a state-dependent mean
and log_std; SAC samples then squashes through tanh. Eval (and the browser) act
with the DETERMINISTIC mean — tanh(mean), no sampling — exactly how sac.py's
sample() returns its deterministic_action. So the exported policy is just that
tanh(mean) path wrapped as a stateless obs[1,8] -> action[1,2] nn.Module, the
interface tensor contract v1 wants (see curriculum/common/export_onnx.py).

Pipeline (mirrors export_ppo_onnx.py):
  train sac.py -> export_policy(deterministic wrapper, 8, 2) -> assert_parity ->
  copy the .onnx to site/public/models/ (git-ignored artifact, provisioned like
  bc_policy.onnx; never committed).

Run with the repo venv, from the repo root:
  .venv/bin/python curriculum/phase2_reinforcement/ch2.2_sac/export_sac_onnx.py
Smoke (no checkpoint; random-init actor, proves the export+parity path):
  .venv/bin/python curriculum/phase2_reinforcement/ch2.2_sac/export_sac_onnx.py --smoke
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
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0  # must match sac.py

# The site fetches this path (git-ignored, provisioned like bc_policy.onnx).
SITE_MODEL = REPO_ROOT / "site" / "public" / "models" / "sac_actor.onnx"


class Actor(nn.Module):
    """Byte-for-byte the sac.py Actor (trunk + mean + log_std heads), redefined
    here because sac.py runs argparse/training at import time and cannot be
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
    """Stateless obs -> action wrapper: tanh(mean), no sampling. This is exactly
    how sac.py's eval acts (the deterministic_action sample() returns), so the
    browser and the Python eval share one function. Contract v1: [1,8]->[1,2]."""

    def __init__(self, actor: Actor):
        super().__init__()
        self.actor = actor

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        mean, _ = self.actor(observation)
        return torch.tanh(mean)


def smoke_policy(seed: int = 0, hidden_dim: int = 64) -> tuple[nn.Module, int, int]:
    """A random-init deterministic policy — no checkpoint, no env rollout needed.
    Exercises the exact Actor architecture + export path in CI (hermetic)."""
    torch.manual_seed(seed)
    return DeterministicPolicy(Actor(OBS_DIM, ACT_DIM, hidden_dim).eval()), OBS_DIM, ACT_DIM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("outputs/ch2.2-sac/sac_actor.pt"),
                        help="the Actor state_dict sac.py saved")
    parser.add_argument("--out", type=Path, default=Path("outputs/ch2.2-sac/sac_actor.onnx"))
    parser.add_argument("--hidden_dim", type=int, default=256)  # sac.py default
    parser.add_argument("--smoke", action="store_true",
                        help="skip the checkpoint; export a random-init actor (CI parity smoke)")
    parser.add_argument("--no-copy", dest="copy_site", action="store_false", default=True,
                        help="skip copying the .onnx into site/public/models/")
    args = parser.parse_args()

    if args.smoke:
        policy, obs_dim, act_dim = smoke_policy(hidden_dim=args.hidden_dim)
    else:
        if not args.ckpt.is_file():
            sys.exit(f"no checkpoint at {args.ckpt} — train it first:\n"
                     f"  .venv/bin/python curriculum/phase2_reinforcement/ch2.2_sac/sac.py --seed 0 --device cpu")
        actor = Actor(OBS_DIM, ACT_DIM, args.hidden_dim)
        actor.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        policy, obs_dim, act_dim = DeterministicPolicy(actor), OBS_DIM, ACT_DIM

    onnx_path = export_policy(policy, obs_dim, act_dim, args.out)
    parity_delta = assert_parity(policy, onnx_path, obs_dim)
    print(f"exported {onnx_path} — torch/onnx parity delta {parity_delta:.2e}")

    if args.copy_site and not args.smoke:
        SITE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx_path, SITE_MODEL)
        print(f"provisioned {SITE_MODEL} (git-ignored — not committed)")


if __name__ == "__main__":
    main()
