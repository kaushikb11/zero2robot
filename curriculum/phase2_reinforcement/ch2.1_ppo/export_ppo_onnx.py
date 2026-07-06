"""Export the trained ch2.1 PPO policy to ONNX (tensor contract v1) for the site.

ppo.py trains cartpole and `torch.save`s the Agent state_dict to
outputs/ch2.1-ppo/ppo_agent.pt. That checkpoint is the source of truth; this is
a FOLLOW-UP script (kept out of ppo.py's readable 349-LOC loop, per the embed.yaml
note that the ONNX export is wired when the chapter graduates from spike to drop).

The Agent is a Gaussian policy: actor_mean(obs) is the distribution MEAN and
actor_logstd is the learned spread. Eval (and the browser) act with the
DETERMINISTIC MEAN — no sampling — so the exported policy is just actor_mean
wrapped as a stateless obs[1,5] -> action[1,1] nn.Module, exactly the interface
tensor contract v1 wants (see curriculum/common/export_onnx.py).

Pipeline (mirrors playground/scripts/make_toy_onnx.py + bc.py's export tail):
  train ppo.py -> export_policy(actor_mean wrapper, 5, 1) -> assert_parity ->
  copy the .onnx to site/public/models/ (git-ignored artifact, provisioned like
  bc_policy.onnx; never committed).

Run with the repo venv, from the repo root:
  .venv/bin/python curriculum/phase2_reinforcement/ch2.1_ppo/export_ppo_onnx.py
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
from curriculum.common.envs.cartpole import CartpoleEnv  # noqa: E402
from curriculum.common.export_onnx import export_policy  # noqa: E402

# The site fetches this path (git-ignored, provisioned like bc_policy.onnx).
SITE_MODEL = REPO_ROOT / "site" / "public" / "models" / "ppo_agent.onnx"


def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0)) -> nn.Linear:
    """Orthogonal init — copied verbatim from ppo.py so the rebuilt Agent's
    parameter shapes/names match the saved state_dict exactly."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Agent(nn.Module):
    """Byte-for-byte the ppo.py Agent architecture (critic + actor_mean +
    actor_logstd), redefined here because ppo.py runs argparse/training at import
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
    """Stateless obs -> action wrapper: the policy MEAN, no sampling. This is
    exactly how ppo.py's eval acts (agent.actor_mean(obs)), so the browser and
    the Python eval share one deterministic function. Contract v1: [1,5]->[1,1]."""

    def __init__(self, actor_mean: nn.Module):
        super().__init__()
        self.actor_mean = actor_mean

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(observation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, default=Path("outputs/ch2.1-ppo/ppo_agent.pt"),
                        help="the state_dict ppo.py saved")
    parser.add_argument("--out", type=Path, default=Path("outputs/ch2.1-ppo/ppo_agent.onnx"))
    parser.add_argument("--hidden_dim", type=int, default=64)  # ppo.py default
    parser.add_argument("--no-copy", dest="copy_site", action="store_false", default=True,
                        help="skip copying the .onnx into site/public/models/")
    args = parser.parse_args()

    if not args.ckpt.is_file():
        sys.exit(f"no checkpoint at {args.ckpt} — train it first:\n"
                 f"  .venv/bin/python curriculum/phase2_reinforcement/ch2.1_ppo/ppo.py --seed 0 --device cpu")

    obs_dim, act_dim = CartpoleEnv.OBS_DIM, CartpoleEnv.ACT_DIM  # 5, 1
    agent = Agent(obs_dim, act_dim, args.hidden_dim)
    agent.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    policy = DeterministicPolicy(agent.actor_mean)

    onnx_path = export_policy(policy, obs_dim, act_dim, args.out)
    # Prove torch and onnxruntime agree on random obs[5] BEFORE the file ships.
    parity_delta = assert_parity(policy, onnx_path, obs_dim)
    print(f"exported {onnx_path} — torch/onnx parity delta {parity_delta:.2e}")

    if args.copy_site:
        SITE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx_path, SITE_MODEL)
        print(f"provisioned {SITE_MODEL} (git-ignored — not committed)")


if __name__ == "__main__":
    main()
