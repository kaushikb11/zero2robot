"""Export the trained ch2.7 domain-randomization policies to ONNX (tensor contract v1).

dr.py trains TWO PPO policies on the quadruped, changing ONE thing between them, and
`torch.save`s each Agent state_dict to outputs/ch2.7-dr/{dr_narrow,dr_randomized}.pt:

  dr_narrow      trained on the nominal dynamics only (mass/friction/gravity fixed).
  dr_randomized  resamples the dynamics each episode within a band around nominal.

Those checkpoints are the source of truth; this is a FOLLOW-UP script (kept out of
dr.py's readable loop, per the embed.yaml note that the ONNX export is wired when the
chapter graduates from spike to drop).

The Agent is the ch2.1 continuous-control agent: actor_mean(obs) is the Gaussian MEAN
and actor_logstd is the learned spread. Eval (and the browser) act with the
DETERMINISTIC MEAN — no sampling — so each exported policy is just actor_mean wrapped
as a stateless obs[1,23] -> action[1,8] nn.Module, exactly the interface tensor
contract v1 wants (see curriculum/common/export_onnx.py). action[1,8] is the residual
target the QuadrupedEnv adds to its trot prior.

Pipeline (mirrors export_walk_onnx.py, one arm per demo panel):
  train dr.py -> export_policy(actor_mean wrapper, 23, 8) -> assert_parity ->
  copy each .onnx to site/public/models/ (git-ignored artifact; never committed).

Run with the repo venv, from the repo root:
  .venv/bin/python curriculum/phase2_reinforcement/ch2.7_dr/export_dr_onnx.py
Smoke (no checkpoints; random-init actors, proves the export+parity path):
  .venv/bin/python curriculum/phase2_reinforcement/ch2.7_dr/export_dr_onnx.py --smoke
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from curriculum.common.assert_parity import assert_parity  # noqa: E402
from curriculum.common.envs.quadruped import QuadrupedEnv  # noqa: E402
from curriculum.common.export_onnx import export_policy  # noqa: E402

OBS_DIM, ACT_DIM = QuadrupedEnv.OBS_DIM, QuadrupedEnv.ACT_DIM  # 23, 8

# The two demo panels: checkpoint stem -> exported filename (also the site path).
ARMS = {"dr_narrow": "dr_narrow.onnx", "dr_randomized": "dr_randomized.onnx"}
SITE_MODELS = REPO_ROOT / "site" / "public" / "models"


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0)) -> nn.Linear:
    """Orthogonal init — copied verbatim from dr.py so the rebuilt Agent's
    parameter shapes/names match the saved state_dict exactly."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Agent(nn.Module):
    """Byte-for-byte the dr.py Agent architecture (critic + actor_mean +
    actor_logstd), redefined here because dr.py runs argparse/training at import
    time and cannot be imported as a module. Only the parameter layout matters —
    we load the trained weights, then export actor_mean alone."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))


class DeterministicPolicy(nn.Module):
    """Stateless obs -> action wrapper: the policy MEAN, no sampling. Exactly how
    dr.py's eval acts (agent.actor_mean(obs)), so the browser and the Python eval
    share one deterministic function. Contract v1: [1,23]->[1,8]."""

    def __init__(self, actor_mean: nn.Module):
        super().__init__()
        self.actor_mean = actor_mean

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(observation)


def load_policy(ckpt: Path, hidden_dim: int) -> DeterministicPolicy:
    agent = Agent(OBS_DIM, ACT_DIM, hidden_dim)
    agent.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return DeterministicPolicy(agent.actor_mean)


def smoke_policy(seed: int, hidden_dim: int) -> DeterministicPolicy:
    """A random-init deterministic policy — no checkpoint, no env rollout needed.
    Exercises the exact Agent architecture + export path in CI (hermetic)."""
    torch.manual_seed(seed)
    return DeterministicPolicy(Agent(OBS_DIM, ACT_DIM, hidden_dim).actor_mean).eval()


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
    parser.add_argument("--ckpt_dir", type=Path, default=Path("outputs/ch2.7-dr"),
                        help="dir holding dr_narrow.pt / dr_randomized.pt saved by dr.py")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ch2.7-dr"))
    parser.add_argument("--hidden_dim", type=int, default=64)  # dr.py default
    parser.add_argument("--smoke", action="store_true",
                        help="skip the checkpoints; export random-init actors (CI parity smoke)")
    parser.add_argument("--no-copy", dest="copy_site", action="store_false", default=True,
                        help="skip copying the .onnx files into site/public/models/")
    args = parser.parse_args()

    for stem, onnx_name in ARMS.items():
        if args.smoke:
            policy = smoke_policy(seed=0, hidden_dim=args.hidden_dim)
        else:
            ckpt = args.ckpt_dir / f"{stem}.pt"
            if not ckpt.is_file():
                sys.exit(f"no checkpoint at {ckpt} — train it first:\n"
                         f"  .venv/bin/python curriculum/phase2_reinforcement/ch2.7_dr/dr.py --seed 0 --device cpu")
            policy = load_policy(ckpt, args.hidden_dim)
        export_arm(policy, args.out_dir / onnx_name, onnx_name, args.copy_site and not args.smoke)


if __name__ == "__main__":
    main()
