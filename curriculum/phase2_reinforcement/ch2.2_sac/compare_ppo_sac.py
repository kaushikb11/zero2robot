"""Off-policy vs on-policy, MEASURED — the ch2.2 headline, as companion tooling.

The chapter's claim is "off-policy beats on-policy on a dense-reward task, when
measured." This file measures it head-to-head on the SAME env (pusher-reach):

  - SAC (off-policy): run sac.py as a subprocess; read its env-steps-to-solve
    and its return-vs-env-steps curve out of metrics.json.
  - PPO (on-policy): a COMPACT from-scratch PPO reference, inlined below. It is
    the ch2.1 family — a Gaussian policy + value net, GAE with the truncation
    bootstrap, a clipped surrogate — retargeted from cartpole to pusher-reach's
    8-dim obs / 2-dim torque action. On-policy means it DISCARDS each rollout
    after its update; that discard is exactly the cost the comparison exposes.

Why a companion file and not inside sac.py: a faithful second RL algorithm does
not fit under sac.py's 450-LOC cap (the ch2.1 spike's H3 finding, applied). The
teaching artifact stays SAC; this is the measurement rig. It imports no RL
framework (doctrine #3) — plain torch/numpy, self-contained.

Both learners are judged by ONE metric: environment steps until the held-out
eval mean final distance falls below SOLVE_DIST. Fewer steps = more
sample-efficient. Run:

    python curriculum/phase2_reinforcement/ch2.2_sac/compare_ppo_sac.py --seed 0
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.envs.pusher_reach import PusherReachEnv  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

SOLVE_DIST = 0.05  # m — identical threshold to sac.py, so the two numbers compare
REPO = Path(__file__).resolve().parents[3]
SAC = Path(__file__).resolve().parent / "sac.py"


# --- on-policy PPO reference (the ch2.1 family, retargeted to pusher-reach) ----
class PPOAgent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=64):
        super().__init__()
        self.critic = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                    nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.actor_mean = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                        nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act_dim))
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get(self, obs, action=None):
        mean = self.actor_mean(obs)
        dist = torch.distributions.Normal(mean, self.actor_logstd.exp().expand_as(mean))
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), self.critic(obs).flatten()


def eval_ppo(agent, device, seed, episodes=10):
    """Deterministic eval: act with the tanh-clipped policy mean; mean final dist."""
    finals = []
    for ep in range(episodes):
        e = PusherReachEnv()
        obs, done, info = e.reset(seed=500_000 + seed + ep), False, {"dist": 0.0}
        while not done:
            with torch.no_grad():
                a = agent.actor_mean(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            obs, _, done, info = e.step(np.clip(a[0].cpu().numpy(), -1, 1))
        finals.append(info["dist"])
    return float(np.mean(finals))


def train_ppo(seed, total_steps, device):
    """Compact PPO on pusher-reach. Returns (steps_to_solve or None, curve)."""
    set_seed(seed)
    dev = torch.device(device)
    num_envs, num_steps = 8, 100  # one 100-step episode per env per rollout
    batch = num_envs * num_steps
    envs = [PusherReachEnv() for _ in range(num_envs)]
    ep_count = np.zeros(num_envs, dtype=np.int64)

    def reset(i):
        o = envs[i].reset(seed=seed + i * 1000 + int(ep_count[i]))
        ep_count[i] += 1
        return o

    agent = PPOAgent(PusherReachEnv.OBS_DIM, PusherReachEnv.ACT_DIM).to(dev)
    opt = torch.optim.Adam(agent.parameters(), lr=3e-4, eps=1e-5)
    gamma, lam, clip = 0.99, 0.95, 0.2
    obs_b = torch.zeros((num_steps, num_envs, PusherReachEnv.OBS_DIM), device=dev)
    act_b = torch.zeros((num_steps, num_envs, PusherReachEnv.ACT_DIM), device=dev)
    lp_b = torch.zeros((num_steps, num_envs), device=dev)
    rew_b = torch.zeros((num_steps, num_envs), device=dev)
    val_b = torch.zeros((num_steps, num_envs), device=dev)
    term_b = torch.zeros((num_steps, num_envs), device=dev)  # terminated mask (always 0 here)
    done_b = torch.zeros((num_steps, num_envs), device=dev)
    boot_b = torch.zeros((num_steps, num_envs), device=dev)

    next_obs = np.stack([reset(i) for i in range(num_envs)])
    steps_to_solve, curve, global_step = None, [], 0
    num_iters = total_steps // batch
    for _ in range(num_iters):
        for t in range(num_steps):
            ot = torch.as_tensor(next_obs, dtype=torch.float32, device=dev)
            obs_b[t] = ot
            with torch.no_grad():
                a, lp, v = agent.get(ot)
            act_b[t], lp_b[t], val_b[t] = a, lp, v
            a_np = a.cpu().numpy()
            nxt = np.empty_like(next_obs)
            for i in range(num_envs):
                oi, r, done, info = envs[i].step(a_np[i])
                rew_b[t, i], term_b[t, i], done_b[t, i] = r, float(info["terminated"]), float(done)
                if done:
                    with torch.no_grad():
                        boot_b[t, i] = agent.critic(torch.as_tensor(oi, dtype=torch.float32, device=dev).unsqueeze(0)).item()
                    oi = reset(i)
                nxt[i] = oi
            next_obs = nxt
        global_step += batch
        # GAE with the truncation bootstrap (ch2.1's compute_advantages, one env-batch)
        with torch.no_grad():
            next_v = agent.critic(torch.as_tensor(next_obs, dtype=torch.float32, device=dev)).flatten()
        adv = torch.zeros_like(rew_b)
        last = torch.zeros(num_envs, device=dev)
        for t in reversed(range(num_steps)):
            nv = next_v if t == num_steps - 1 else val_b[t + 1]
            nv = torch.where(done_b[t].bool(), boot_b[t], nv)
            delta = rew_b[t] + gamma * nv * (1 - term_b[t]) - val_b[t]
            last = delta + gamma * lam * (1 - done_b[t]) * last
            adv[t] = last
        returns = adv + val_b
        bo, ba = obs_b.reshape(-1, PusherReachEnv.OBS_DIM), act_b.reshape(-1, PusherReachEnv.ACT_DIM)
        blp, badv, bret = lp_b.reshape(-1), adv.reshape(-1), returns.reshape(-1)
        for _ in range(10):  # update_epochs
            idx = torch.randperm(batch, device=dev)
            for s in range(0, batch, batch // 4):
                mb = idx[s:s + batch // 4]
                _, nlp, nv = agent.get(bo[mb], ba[mb])
                ratio = (nlp - blp[mb]).exp()
                mba = (badv[mb] - badv[mb].mean()) / (badv[mb].std() + 1e-8)
                pg = torch.max(-mba * ratio, -mba * torch.clamp(ratio, 1 - clip, 1 + clip)).mean()
                vl = 0.5 * ((nv - bret[mb]) ** 2).mean()
                opt.zero_grad()
                (pg + 0.5 * vl).backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                opt.step()
        d = eval_ppo(agent, dev, seed)
        curve.append((global_step, round(d, 5)))
        if steps_to_solve is None and d < SOLVE_DIST:
            steps_to_solve = global_step
    return steps_to_solve, curve


# --- SAC side: run the real artifact, read its measured sample efficiency ------
def run_sac(seed, total_steps, device, workdir):
    out = workdir / f"sac-seed{seed}"
    subprocess.run([sys.executable, str(SAC), "--seed", str(seed), "--device", device,
                    "--total_steps", str(total_steps), "--no-rerun", "--out", str(out)],
                   check=True, capture_output=True, cwd=REPO)
    m = json.loads((out / "metrics.json").read_text())
    return m["env_steps_to_solve"], [(s, d) for s, _, d in m["curve"]]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--sac_steps", type=int, default=30_000)
    p.add_argument("--ppo_steps", type=int, default=200_000, help="on-policy needs a bigger budget; that IS the finding")
    p.add_argument("--out", type=Path, default=Path("outputs/ch2.2-compare"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"SAC (off-policy, replay reuse) on pusher-reach, seed {args.seed}...")
    sac_solve, sac_curve = run_sac(args.seed, args.sac_steps, args.device, args.out)
    print(f"PPO (on-policy, discards each rollout) on pusher-reach, seed {args.seed}...")
    ppo_solve, ppo_curve = train_ppo(args.seed, args.ppo_steps, args.device)

    def fmt(x):
        return f"{x:,} env steps" if x else "NOT solved in budget"
    print("\n=== sample efficiency: env steps to eval mean final dist < "
          f"{SOLVE_DIST} m (seed {args.seed}) ===")
    print(f"  SAC (off-policy):  {fmt(sac_solve)}   [budget {args.sac_steps:,}]")
    print(f"  PPO (on-policy):   {fmt(ppo_solve)}   [budget {args.ppo_steps:,}]")
    if sac_solve and ppo_solve:
        print(f"  -> SAC solves it in {ppo_solve / sac_solve:.1f}x fewer env steps.")
    elif sac_solve and not ppo_solve:
        print(f"  -> SAC solved; PPO did not reach the bar within {args.ppo_steps:,} steps.")
    (args.out / "comparison.json").write_text(json.dumps(
        {"seed": args.seed, "sac_steps_to_solve": sac_solve, "ppo_steps_to_solve": ppo_solve,
         "sac_curve": sac_curve, "ppo_curve": ppo_curve}, indent=2) + "\n")
    print(f"\nwrote {args.out / 'comparison.json'}")


if __name__ == "__main__":
    main()
