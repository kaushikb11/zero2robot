"""zero2robot 2.2 — SAC: The Off-Policy Bargain.

PPO (ch2.1) throws each rollout away. It collects a batch by acting, takes a
handful of gradient steps on exactly that batch, and then discards every
transition — on-policy means the data must come from the CURRENT policy, so
yesterday's experience is worthless. That is safe and stable, and it is
wasteful: every environment step is used once and deleted.

Soft Actor-Critic strikes the opposite bargain. Keep EVERYTHING. Every
transition the policy ever generated goes into a replay buffer and gets reused
across many gradient steps — off-policy. The catch is instability: learning a
Q-function from your own bootstrapped estimates, on stale data, diverges if you
let it. SAC tames that with three braces you build here from scratch: TWIN Q
critics with target networks (clipped double-Q kills the overestimation
feedback loop), a squashed-Gaussian policy trained by the reparameterization
trick, and a maximum-ENTROPY objective with an auto-tuned temperature alpha that
keeps the policy exploring instead of collapsing onto one Q-function artifact.

The lesson is measured, not asserted: on the DENSE-reward pusher-reach env
(common/envs/pusher_reach, reward = -distance every step), SAC's replay reuse
makes it far more SAMPLE-EFFICIENT than on-policy PPO — it drives the fingertip
to the target in far fewer environment steps. See compare_ppo_sac.py for the
head-to-head env-steps-to-solve number (it did not fit inside this file's LOC
budget — SAC itself is the artifact; the comparison is companion tooling).

Run it:      python curriculum/phase2_reinforcement/ch2.2_sac/sac.py --seed 0
Investigate: python .../sac.py --seed 0 --buffer_size 5000   (shrink the replay)
Break it:    python .../sac.py --seed 0 --break               (drop target nets -> Q diverges)
CI smoke:    python .../sac.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch2.1 / tests/).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusher_reach import PusherReachEnv, reach_action  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

SOLVE_DIST = 0.05  # m: eval mean final distance below this counts as "solved" (random ~0.176, scripted ~0.0001)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch2.2-sac"))
parser.add_argument("--total_steps", type=int, default=30_000)  # cpu-laptop: minutes | smoke: 1500
parser.add_argument("--learning_starts", type=int, default=1000, help="random-action warmup that seeds the replay buffer before any gradient step")
parser.add_argument("--buffer_size", type=int, default=100_000, help="replay capacity; shrink it to feel the off-policy bargain weaken")
parser.add_argument("--batch_size", type=int, default=256, help="transitions sampled from replay per gradient step")
parser.add_argument("--hidden_dim", type=int, default=256)  # T4: 256 | 4090: 256 (pusher is tiny; width is not the bottleneck)
parser.add_argument("--lr", type=float, default=3e-4, help="shared Adam lr for actor, critics, and alpha")
parser.add_argument("--gamma", type=float, default=0.99, help="reward discount")
parser.add_argument("--tau", type=float, default=0.005, help="soft target-update rate: target <- (1-tau)*target + tau*online")
parser.add_argument("--alpha", type=float, default=0.2, help="entropy temperature (used as a FIXED value only under --no-autotune)")
parser.add_argument("--autotune", dest="autotune", action="store_true", default=True,
                    help="auto-tune alpha toward target entropy -act_dim (on by default)")
parser.add_argument("--no-autotune", dest="autotune", action="store_false")
parser.add_argument("--eval_episodes", type=int, default=10)   # T4: 10 | smoke: 3
parser.add_argument("--eval_interval", type=int, default=2000, help="env steps between held-out evals (the sample-efficiency curve)")
parser.add_argument("--seed", type=int, default=0, help="seeds torch, numpy, AND every env reset")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--break", dest="break_bug", action="store_true",
                    help="Break It: bootstrap the Q-target off the ONLINE critics (no target networks) — watch Q diverge")
parser.add_argument("--smoke", action="store_true", help="tiny CPU run for CI; two runs must match byte-for-byte")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; env resets are seeded explicitly below
if args.smoke:  # pin everything the CI byte-compare depends on
    args.total_steps, args.learning_starts, args.buffer_size = 1500, 200, 5000
    args.batch_size, args.eval_episodes, args.eval_interval = 64, 3, 500
    args.hidden_dim, args.device = 64, "cpu"
banner("ch2.2-sac", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
target_entropy = -float(PusherReachEnv.ACT_DIM)  # SAC's standard heuristic: -|A|
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch2.2-sac", spawn=False)
    rr.save(str(args.out / "sac.rrd"))
# --- endregion ---

# --- region: env ---
# SAC is a single-env algorithm here (CleanRL's sac_continuous_action shape): one
# env step, then one gradient step on a batch sampled from replay. Because there
# is exactly one env, global_step IS the environment-step count — which makes the
# sample-efficiency curve (return vs env steps) read directly off the loop.
# No gym, no vector wrapper: the autoreset stays in view.
env = PusherReachEnv()
episode_count = 0


def env_reset() -> np.ndarray:
    """Reset with a fresh deterministic seed per episode so --seed is reproducible."""
    global episode_count
    obs = env.reset(seed=args.seed + episode_count)
    episode_count += 1
    return obs
# --- endregion ---

# --- region: model ---
def layer_init(layer: nn.Linear) -> nn.Linear:
    """Plain fan-in init; SAC is far less init-sensitive than PPO (no on-policy
    trust region to protect), so we skip the orthogonal-gain ceremony."""
    return layer


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Actor(nn.Module):
    """Squashed-Gaussian policy. The net outputs a STATE-DEPENDENT mean and
    log-std (unlike ch2.1's PPO, where log-std was a bare parameter); we sample a
    Gaussian, then squash through tanh so actions land in [-1, 1]. The squash
    needs a log-prob CORRECTION — tanh compresses probability mass, and the
    change-of-variables term log(1 - tanh(x)^2) accounts for it. Omit it and the
    entropy term is wrong and SAC's exploration falls apart."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
        )
        self.mean = layer_init(nn.Linear(hidden_dim, act_dim))
        self.log_std = layer_init(nn.Linear(hidden_dim, act_dim))

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        return self.mean(h), log_std

    def sample(self, obs: torch.Tensor):
        """Returns (action, log_prob, deterministic_action). `rsample` is the
        REPARAMETERIZATION trick: it keeps the sample differentiable w.r.t. the
        net so the actor loss can backprop through the sampled action."""
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()
        action = torch.tanh(x)
        # tanh change-of-variables correction, summed over action dims
        log_prob = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    """A single Q(obs, action) network. We build TWO of these (twin critics) and
    always bootstrap from the MIN of the pair — clipped double-Q, the brace that
    stops the Q-function from chasing its own overestimates into divergence."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + act_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
            layer_init(nn.Linear(hidden_dim, 1)),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=1))


obs_dim, act_dim = PusherReachEnv.OBS_DIM, PusherReachEnv.ACT_DIM
actor = Actor(obs_dim, act_dim, args.hidden_dim).to(device)
q1 = Critic(obs_dim, act_dim, args.hidden_dim).to(device)
q2 = Critic(obs_dim, act_dim, args.hidden_dim).to(device)
# Target critics: slow-moving copies the bootstrap targets are read from. They
# are what --break removes. deepcopy + requires_grad_(False): never trained by
# gradient descent, only nudged by the soft update (region: update).
q1_targ = copy.deepcopy(q1).requires_grad_(False)
q2_targ = copy.deepcopy(q2).requires_grad_(False)
actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
# alpha (entropy temperature). Optimize log_alpha for a positivity-free
# parameterization; alpha = exp(log_alpha). Fixed at args.alpha under --no-autotune.
log_alpha = torch.tensor(float(np.log(args.alpha)), dtype=torch.float32, device=device, requires_grad=args.autotune)
alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr) if args.autotune else None
# --- endregion ---

# --- region: replay ---
class ReplayBuffer:
    """A fixed-size circular buffer of transitions — the off-policy bargain in
    one data structure. Every (obs, action, reward, next_obs, terminated) tuple
    the policy generates is stored and REUSED across many gradient steps, exactly
    what on-policy PPO throws away. We store `terminated` (NOT `done`): a
    time-limit truncation must still bootstrap from next_obs, so only a TRUE
    terminal masks the future value. Pusher-reach never terminates early
    (terminate_on_success=False), so terminated is always 0 and every target
    bootstraps — the ch2.1 truncation lesson, carried over unchanged."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i, 0] = obs, action, reward
        self.next_obs[i], self.terminated[i, 0] = next_obs, float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        # torch.randint draws from the seeded global torch RNG -> reproducible
        idx = torch.randint(0, self.size, (batch_size,)).numpy()
        t = lambda a: torch.as_tensor(a[idx], device=device)  # noqa: E731
        return t(self.obs), t(self.actions), t(self.rewards), t(self.next_obs), t(self.terminated)


buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
# --- endregion ---

# --- region: update ---
def sac_update() -> dict:
    """One SAC gradient step on a replay batch: critic update, then actor +
    temperature, then the soft target nudge. This is the whole algorithm."""
    obs, action, reward, next_obs, terminated = buffer.sample(args.batch_size)
    alpha = log_alpha.exp().detach()

    # --- critic update: regress both Q's toward the entropy-augmented target ---
    with torch.no_grad():
        next_action, next_logp, _ = actor.sample(next_obs)
        # Break It: bootstrapping off the ONLINE critics removes the slow target
        # and the Q-estimate chases itself -> divergence (the whole point of targets).
        tq1 = (q1 if args.break_bug else q1_targ)(next_obs, next_action)
        tq2 = (q2 if args.break_bug else q2_targ)(next_obs, next_action)
        # clipped double-Q: the MIN of the pair, minus the entropy bonus (soft value)
        min_next_q = torch.min(tq1, tq2) - alpha * next_logp
        target_q = reward + args.gamma * (1.0 - terminated) * min_next_q
    q1_loss = F.mse_loss(q1(obs, action), target_q)
    q2_loss = F.mse_loss(q2(obs, action), target_q)
    critic_loss = q1_loss + q2_loss
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()

    # --- actor update: maximize (min-Q - alpha*logprob) via the reparam sample ---
    new_action, logp, _ = actor.sample(obs)
    min_q = torch.min(q1(obs, new_action), q2(obs, new_action))
    actor_loss = (alpha * logp - min_q).mean()  # ascend min_q, keep entropy high
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    # --- temperature update: push alpha until entropy sits at target_entropy ---
    if args.autotune:
        alpha_loss = -(log_alpha.exp() * (logp.detach() + target_entropy)).mean()
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()

    # --- soft target update: Polyak-average the targets a hair toward online ---
    if not args.break_bug:
        with torch.no_grad():
            for online, targ in ((q1, q1_targ), (q2, q2_targ)):
                for p, tp in zip(online.parameters(), targ.parameters()):
                    tp.mul_(1.0 - args.tau).add_(args.tau * p)
    return {"q_loss": critic_loss.item() / 2.0, "actor_loss": actor_loss.item(),
            "alpha": float(log_alpha.exp().item()), "q_value": min_q.mean().item()}
# --- endregion ---

# --- region: eval ---
def evaluate(episodes: int) -> tuple[float, float, float]:
    """Held-out eval with the DETERMINISTIC policy (tanh of the mean, no
    sampling), on seeds disjoint from training. Returns (mean_return,
    mean_final_dist, success_rate) — mean_final_dist is the sample-efficiency
    signal that must fall toward 0 (scripted reach_action gets ~0.0001 m)."""
    eval_env = PusherReachEnv()
    returns, finals, successes = [], [], []
    for ep in range(episodes):
        obs = eval_env.reset(seed=500_000 + args.seed + ep)  # held out from training seeds
        done, ret = False, 0.0
        info = {"dist": eval_env._dist(), "success": False}
        while not done:
            with torch.no_grad():
                _, _, mean_action = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            obs, reward, done, info = eval_env.step(mean_action[0].cpu().numpy())
            ret += reward
        returns.append(ret)
        finals.append(info["dist"])
        successes.append(float(info["success"]))
    return float(np.mean(returns)), float(np.mean(finals)), float(np.mean(successes))
# --- endregion ---

# --- region: train ---
# The loop: act (random during warmup, then policy), store the transition, and
# once the buffer has enough, take one gradient step PER env step. `curve` records
# (env_step, mean_return, mean_dist) at each eval — that IS the return-vs-env-steps
# sample-efficiency curve the chapter measures SAC against on-policy PPO.
next_obs = env_reset()
ep_return, recent_returns = 0.0, []
curve, steps_to_solve = [], None
stats = {"q_loss": float("nan"), "actor_loss": float("nan"), "alpha": args.alpha, "q_value": float("nan")}
for global_step in range(1, args.total_steps + 1):
    if global_step <= args.learning_starts:  # warmup: uniform random actions fill the buffer
        action = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
    else:
        with torch.no_grad():
            action, _, _ = actor.sample(torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0))
        action = action[0].cpu().numpy()

    obs_after, reward, done, info = env.step(action)
    # store `terminated` (not `done`): a truncation must still bootstrap (see replay)
    buffer.add(next_obs, action, reward, obs_after, info["terminated"])
    next_obs = obs_after
    ep_return += reward
    if done:
        recent_returns.append(ep_return)
        ep_return, next_obs = 0.0, env_reset()  # autoreset

    if global_step > args.learning_starts:  # one gradient step per env step
        stats = sac_update()

    if global_step % args.eval_interval == 0 or global_step == args.total_steps:
        eval_return, eval_dist, eval_success = evaluate(args.eval_episodes)
        curve.append((global_step, round(eval_return, 3), round(eval_dist, 5)))
        if steps_to_solve is None and eval_dist < SOLVE_DIST:
            steps_to_solve = global_step  # first env step the policy holds the target
        mean_train = float(np.mean(recent_returns[-20:])) if recent_returns else float("nan")
        if args.rerun:
            rr.set_time("global_step", sequence=global_step)
            rr.log("charts/eval_return", rr.Scalars([eval_return]))
            rr.log("charts/eval_dist", rr.Scalars([eval_dist]))
            rr.log("charts/eval_success", rr.Scalars([eval_success]))
            rr.log("charts/train_return", rr.Scalars([mean_train]))
            rr.log("replay/size", rr.Scalars([float(buffer.size)]))
            for name, value in stats.items():
                rr.log(f"losses/{name}", rr.Scalars([value]))
        print(f"step {global_step:6d}/{args.total_steps}  eval_return {eval_return:8.1f}  "
              f"eval_dist {eval_dist:.4f}m  success {eval_success:.2f}  "
              f"alpha {stats['alpha']:.3f}  q_loss {stats['q_loss']:.3f}")
# --- endregion ---

# --- region: report ---
final_return, final_dist, final_success = evaluate(args.eval_episodes)
# Scripted IK baseline (the ceiling SAC chases) on the SAME held-out seeds.
s_env = PusherReachEnv()
s_finals = []
for ep in range(args.eval_episodes):
    obs = s_env.reset(seed=500_000 + args.seed + ep)
    done, info = False, {"dist": s_env._dist()}
    while not done:
        obs, _, done, info = s_env.step(reach_action(s_env))
    s_finals.append(info["dist"])
scripted_dist = float(np.mean(s_finals))
print(f"\neval: mean final dist {final_dist:.4f}m  success {final_success:.2f}  return {final_return:.1f}")
print(f"      (random ~0.176m, scripted {scripted_dist:.4f}m, solve<{SOLVE_DIST}m)")
print(f"sample efficiency: solved (eval_dist<{SOLVE_DIST}m) at {steps_to_solve} env steps"
      if steps_to_solve else f"sample efficiency: did NOT reach eval_dist<{SOLVE_DIST}m in {args.total_steps} steps")

torch.save(actor.state_dict(), args.out / "sac_actor.pt")
metrics = {
    "break_bug": bool(args.break_bug),
    "autotune": bool(args.autotune),
    "buffer_size": args.buffer_size,
    "tau": args.tau,
    "mean_eval_final_dist": round(final_dist, 5),
    "mean_eval_return": round(final_return, 4),
    "mean_eval_success": round(final_success, 4),
    "mean_scripted_final_dist": round(scripted_dist, 5),
    "env_steps_to_solve": steps_to_solve,
    "curve": curve,
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "total_steps": args.total_steps,
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'sac.rrd'} — open it with: rerun {args.out / 'sac.rrd'}")
# --- endregion ---
