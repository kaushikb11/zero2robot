"""zero2robot 2.5 — Locomotion: The Quadruped Walks.

Nobody programs the gait. You write down a reward — go forward, stay upright,
hold your ride height, don't fall, don't thrash — hand it to SAC, and a walking
motion EMERGES from trial and error. The trot in ch2.4's baseline was
hand-scripted (a sinusoid you tuned); here the policy discovers its own gait by
maximizing return on the exact same quadruped. That emergence is the whole
chapter.

This file trains SAC (the ch2.2 algorithm, reused nearly unchanged) on the
common/envs/quadruped locomotion env, and MEASURES the gait appearing: forward
distance and forward velocity rising over training, the return climbing off the
"just stand" floor toward the scripted-trot ceiling. It is honest about the
free-tier budget — a full trot wants far more environment steps than a CPU
laptop affords, so the lesson is the SHAPE of the emergence, not a finished walk.

Two design ideas the env's obs[23]/action[8] encode — locomotion-RL's vocabulary:

  OBSERVATION (float32[23]) — what the policy must SEE to walk. Joint angles +
  velocities (0..15) are proprioception; torso height (16), up-vector (17..19),
  and linear velocity (20..22) are the body state the reward reads. There is no
  clock and no contact flag — the policy infers gait phase from the velocities.
  (An exercise blinds the velocity and watches the walk degrade: how you learn
  which coordinates matter.)

  ACTION (float32[8]) — RESIDUAL position targets, not torques. The env commands
  DEFAULT_POSE + ACTION_SCALE * action to PD servos, so action 0 already stands.
  The policy only learns the small offsets that turn a stand into a stride — a
  far easier search than raw torque, and the anchor the gait emerges from.

A LIGHT DOMAIN-RANDOMIZATION intro (a preview of ch2.7): we jitter the torso mass
a few percent per episode (--domain_rand, on by default), so the policy can't
memorize one body's exact dynamics. Minimal — one randomized number — where
ch2.7 makes randomization (friction, mass, latency, perturbations) the story.

Run it:      python curriculum/phase2_reinforcement/ch2.5_walk/walk.py --seed 0
Investigate: python .../walk.py --seed 0 --no-domain-rand   (drop the DR preview)
Ablate obs:  python .../walk.py --seed 0 --blind_velocity   (hide torso vx from the policy)
CI smoke:    python .../walk.py --smoke --seed 0 --no-rerun
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
# sys.path so `curriculum.common` resolves (same pattern as ch2.1 / ch2.2).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.quadruped import QuadrupedEnv, trot_action  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

# The bar (env baselines): scripted trot +2.14 m / return ~306; stand ~-0.01 m /
# ~199; random ~-0.30 m / ~143. "Walking" = clearly forward, well past the stand.
WALK_DIST = 0.5  # m of forward progress that counts as "a gait has emerged"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch2.5-walk"))
parser.add_argument("--total_steps", type=int, default=60_000)  # cpu-laptop: minutes | smoke: 2000 | T4/4090: 500k+ for a full trot
parser.add_argument("--learning_starts", type=int, default=1000, help="random-action warmup that seeds the replay buffer before any gradient step")
parser.add_argument("--buffer_size", type=int, default=200_000, help="replay capacity (locomotion is long-horizon; keep a big history)")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=256)  # T4: 256 | 4090: 512
parser.add_argument("--lr", type=float, default=3e-4, help="shared Adam lr for actor, critics, and alpha")
parser.add_argument("--gamma", type=float, default=0.99, help="reward discount")
parser.add_argument("--tau", type=float, default=0.005, help="soft target-update rate")
parser.add_argument("--alpha", type=float, default=0.2, help="entropy temperature (FIXED only under --no-autotune)")
parser.add_argument("--autotune", dest="autotune", action="store_true", default=True,
                    help="auto-tune alpha toward target entropy -act_dim (on by default)")
parser.add_argument("--no-autotune", dest="autotune", action="store_false")
parser.add_argument("--domain_rand", dest="domain_rand", action="store_true", default=True,
                    help="DR preview (ch2.7): jitter torso mass per episode so the gait can't overfit one body")
parser.add_argument("--no-domain-rand", dest="domain_rand", action="store_false")
parser.add_argument("--dr_mass_frac", type=float, default=0.15, help="torso mass jitter: uniform[1-f, 1+f] per training episode")
parser.add_argument("--blind_velocity", action="store_true", help="obs-design ablation: zero out torso linear velocity (obs 20..22) the policy sees")
parser.add_argument("--eval_episodes", type=int, default=10)  # T4: 10 | smoke: 3
parser.add_argument("--eval_interval", type=int, default=5000, help="env steps between held-out evals (the gait-emergence curve)")
parser.add_argument("--seed", type=int, default=0, help="seeds torch, numpy, AND every env reset")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true", help="tiny CPU run for CI; two runs must match byte-for-byte")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; env resets are seeded explicitly below
if args.smoke:  # pin everything the CI byte-compare depends on
    args.total_steps, args.learning_starts, args.buffer_size = 2000, 200, 5000
    args.batch_size, args.eval_episodes, args.eval_interval = 64, 3, 500
    args.hidden_dim, args.device = 64, "cpu"
banner("ch2.5-walk", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
dr_rng = np.random.Generator(np.random.PCG64(args.seed))  # drives domain randomization only
target_entropy = -float(QuadrupedEnv.ACT_DIM)  # SAC's standard heuristic: -|A|
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch2.5-walk", spawn=False)
    rr.save(str(args.out / "walk.rrd"))
# --- endregion ---

# --- region: env ---
# One env, one gradient step per env step (ch2.2's SAC shape). The quadruped is
# the ch2.4 reward env verbatim — we change NOTHING about it; the gait must emerge
# from the reward as shipped. DOMAIN RANDOMIZATION lives here: each training reset
# scales the torso mass by a small seeded factor so the policy meets a slightly
# different body every episode. Eval always uses the NOMINAL body (comparable).
env = QuadrupedEnv()
torso_id = env.model.body("torso").id
default_torso_mass = float(env.model.body_mass[torso_id])
episode_count = 0


def _blind(obs: np.ndarray) -> np.ndarray:
    """Obs-design ablation: optionally hide torso linear velocity (obs 20..22) —
    the policy loses its direct forward-speed sense; watch the walk suffer."""
    if args.blind_velocity:
        obs = obs.copy()
        obs[20:23] = 0.0
    return obs


def env_reset(randomize: bool) -> np.ndarray:
    """Reset with a fresh deterministic seed per episode. When `randomize`, jitter
    the torso mass (the DR preview); otherwise restore the nominal body."""
    global episode_count
    if randomize and args.domain_rand:
        factor = float(dr_rng.uniform(1.0 - args.dr_mass_frac, 1.0 + args.dr_mass_frac))
        env.model.body_mass[torso_id] = default_torso_mass * factor
    else:
        env.model.body_mass[torso_id] = default_torso_mass
    obs = env.reset(seed=args.seed + episode_count)
    episode_count += 1
    return _blind(obs)
# --- endregion ---

# --- region: model ---
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Actor(nn.Module):
    """Squashed-Gaussian policy (ch2.2, unchanged): a state-dependent mean +
    log-std, sampled and squashed through tanh into [-1, 1] — exactly the
    residual-target range the env expects. The tanh log-prob correction keeps the
    entropy term honest; drop it and SAC's exploration collapses."""

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

    def sample(self, obs: torch.Tensor):
        # (action, log_prob, deterministic_action); rsample = reparameterization
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = normal.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True), torch.tanh(mean)


class Critic(nn.Module):
    """A single Q(obs, action). We build TWO (twin critics) and bootstrap from
    the MIN — clipped double-Q, the brace that stops Q chasing its own overshoot."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=1))


obs_dim, act_dim = QuadrupedEnv.OBS_DIM, QuadrupedEnv.ACT_DIM
actor = Actor(obs_dim, act_dim, args.hidden_dim).to(device)
q1 = Critic(obs_dim, act_dim, args.hidden_dim).to(device)
q2 = Critic(obs_dim, act_dim, args.hidden_dim).to(device)
q1_targ = copy.deepcopy(q1).requires_grad_(False)
q2_targ = copy.deepcopy(q2).requires_grad_(False)
actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
log_alpha = torch.tensor(float(np.log(args.alpha)), dtype=torch.float32, device=device, requires_grad=args.autotune)
alpha_opt = torch.optim.Adam([log_alpha], lr=args.lr) if args.autotune else None
# --- endregion ---

# --- region: replay ---
class ReplayBuffer:
    """Fixed-size circular buffer of transitions — the off-policy bargain in one
    data structure. We store `terminated` (a real fall), NOT `done`: a time-limit
    truncation must still bootstrap the future value from next_obs, so only a true
    fall masks it — the one thing that zeroes the locomotion return's future."""

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
        idx = torch.randint(0, self.size, (batch_size,)).numpy()  # seeded torch RNG
        t = lambda a: torch.as_tensor(a[idx], device=device)  # noqa: E731
        return t(self.obs), t(self.actions), t(self.rewards), t(self.next_obs), t(self.terminated)


buffer = ReplayBuffer(args.buffer_size, obs_dim, act_dim)
# --- endregion ---

# --- region: update ---
def sac_update() -> dict:
    """One SAC gradient step (ch2.2, verbatim): critic update, then actor +
    temperature, then the soft target nudge. Locomotion changes the data, not the
    algorithm — that reuse is the point."""
    obs, action, reward, next_obs, terminated = buffer.sample(args.batch_size)
    alpha = log_alpha.exp().detach()

    with torch.no_grad():
        next_action, next_logp, _ = actor.sample(next_obs)
        min_next_q = torch.min(q1_targ(next_obs, next_action), q2_targ(next_obs, next_action)) - alpha * next_logp
        target_q = reward + args.gamma * (1.0 - terminated) * min_next_q
    q1_loss = F.mse_loss(q1(obs, action), target_q)
    q2_loss = F.mse_loss(q2(obs, action), target_q)
    critic_opt.zero_grad()
    (q1_loss + q2_loss).backward()
    critic_opt.step()

    new_action, logp, _ = actor.sample(obs)
    min_q = torch.min(q1(obs, new_action), q2(obs, new_action))
    actor_loss = (alpha * logp - min_q).mean()  # ascend min_q, keep entropy high
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()

    if args.autotune:
        alpha_loss = -(log_alpha.exp() * (logp.detach() + target_entropy)).mean()
        alpha_opt.zero_grad()
        alpha_loss.backward()
        alpha_opt.step()

    with torch.no_grad():  # Polyak-average the targets a hair toward online
        for online, targ in ((q1, q1_targ), (q2, q2_targ)):
            for p, tp in zip(online.parameters(), targ.parameters()):
                tp.mul_(1.0 - args.tau).add_(args.tau * p)
    return {"q_loss": (q1_loss + q2_loss).item() / 2.0, "actor_loss": actor_loss.item(),
            "alpha": float(log_alpha.exp().item()), "q_value": min_q.mean().item()}
# --- endregion ---

# --- region: eval ---
def rollout(policy_env: QuadrupedEnv, seed: int, use_actor: bool) -> tuple[float, float, float, float]:
    """One episode on the NOMINAL body. Returns (return, forward_distance_m,
    mean_forward_vel, length). use_actor=False rolls the scripted trot (the bar)."""
    policy_env.model.body_mass[policy_env.model.body("torso").id] = default_torso_mass
    obs = _blind(policy_env.reset(seed=seed))
    x0 = float(policy_env.data.qpos[policy_env._root_qadr])
    done, ret, vxs, steps = False, 0.0, [], 0
    while not done:
        if use_actor:
            with torch.no_grad():
                _, _, mean_action = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            action = mean_action[0].cpu().numpy()
        else:
            action = trot_action(policy_env)
        obs, reward, done, info = policy_env.step(action)
        obs = _blind(obs)
        ret += reward
        vxs.append(info["forward_vel"])
        steps += 1
    forward = float(policy_env.data.qpos[policy_env._root_qadr]) - x0
    return ret, forward, float(np.mean(vxs)), float(steps)


def evaluate(episodes: int) -> tuple[float, float, float, float]:
    """Held-out eval, DETERMINISTIC policy (tanh of mean), seeds disjoint from
    training. Mean (return, forward_dist, forward_vel, len); forward_dist rising
    is the gait emerging."""
    eval_env = QuadrupedEnv()
    rows = [rollout(eval_env, 500_000 + args.seed + ep, use_actor=True) for ep in range(episodes)]
    return tuple(float(np.mean([r[k] for r in rows])) for k in range(4))
# --- endregion ---

# --- region: train ---
# The loop: act (random during warmup, then policy), store the transition, and
# once warm take one gradient step per env step. `curve` records (env_step,
# eval_return, eval_forward_dist) at each eval — that IS the gait-emergence curve:
# forward distance climbing off the "just stand" floor.
next_obs = env_reset(randomize=True)
ep_return, recent_returns = 0.0, []
curve, walks_at = [], None
stats = {"q_loss": float("nan"), "actor_loss": float("nan"), "alpha": args.alpha, "q_value": float("nan")}
for global_step in range(1, args.total_steps + 1):
    if global_step <= args.learning_starts:  # warmup: uniform random actions fill the buffer
        action = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
    else:
        with torch.no_grad():
            action, _, _ = actor.sample(torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0))
        action = action[0].cpu().numpy()

    obs_after, reward, done, info = env.step(action)
    obs_after = _blind(obs_after)
    buffer.add(next_obs, action, reward, obs_after, info["terminated"])  # `terminated`, not `done`
    next_obs = obs_after
    ep_return += reward
    if done:
        recent_returns.append(ep_return)
        ep_return, next_obs = 0.0, env_reset(randomize=True)  # autoreset (new randomized body)

    if global_step > args.learning_starts:
        stats = sac_update()

    if global_step % args.eval_interval == 0 or global_step == args.total_steps:
        eval_return, eval_dist, eval_vx, eval_len = evaluate(args.eval_episodes)
        curve.append((global_step, round(eval_return, 3), round(eval_dist, 4)))
        if walks_at is None and eval_dist > WALK_DIST:
            walks_at = global_step  # first env step the emergent gait clears WALK_DIST
        mean_train = float(np.mean(recent_returns[-20:])) if recent_returns else float("nan")
        if args.rerun:
            rr.set_time("global_step", sequence=global_step)
            rr.log("charts/eval_return", rr.Scalars([eval_return]))
            rr.log("charts/eval_forward_dist", rr.Scalars([eval_dist]))
            rr.log("charts/eval_forward_vel", rr.Scalars([eval_vx]))
            rr.log("charts/train_return", rr.Scalars([mean_train]))
            for name, value in stats.items():
                rr.log(f"losses/{name}", rr.Scalars([value]))
        print(f"step {global_step:6d}/{args.total_steps}  eval_return {eval_return:7.1f}  "
              f"fwd_dist {eval_dist:+.3f}m  fwd_vel {eval_vx:+.3f}m/s  len {eval_len:.0f}  "
              f"alpha {stats['alpha']:.3f}")
# --- endregion ---

# --- region: report ---
final_return, final_dist, final_vx, final_len = evaluate(args.eval_episodes)
# The bar: scripted open-loop trot on the SAME held-out seeds. Honest nuance
# (measured): the emergent gait beats the trot on DISTANCE but falls before the
# horizon, so its RETURN stays below the trot's. Report both, don't cherry-pick.
bar_env = QuadrupedEnv()
trot_rows = [rollout(bar_env, 500_000 + args.seed + ep, use_actor=False) for ep in range(args.eval_episodes)]
trot_dist, trot_return = float(np.mean([r[1] for r in trot_rows])), float(np.mean([r[0] for r in trot_rows]))
print(f"\neval: forward {final_dist:+.3f} m  vel {final_vx:+.3f} m/s  return {final_return:.1f}  len {final_len:.0f}")
print(f"      bar: scripted trot {trot_dist:+.3f} m / return {trot_return:.1f}  (random ~-0.30 m, stand ~-0.01 m)")
print(f"gait emergence: {'walks (fwd>' + str(WALK_DIST) + 'm) at ' + str(walks_at) + ' env steps' if walks_at else 'did NOT clear ' + str(WALK_DIST) + 'm forward in ' + str(args.total_steps) + ' steps — partial gait, see curve'}")

# Record one deterministic rollout to rerun WITH the env's own logger, so the
# emergent gait + foot-contact pattern is inspectable (world/robot/feet, torso,
# forward_vel). This is the "watch the gait" artifact for the chapter page.
if args.rerun:
    gait_env = QuadrupedEnv()
    gait_env.enable_rerun(path=str(args.out / "walk_gait.rrd"))
    rollout(gait_env, 500_000 + args.seed, use_actor=True)

torch.save(actor.state_dict(), args.out / "walk_actor.pt")
metrics = {
    "domain_rand": bool(args.domain_rand),
    "blind_velocity": bool(args.blind_velocity),
    "mean_eval_forward_dist": round(final_dist, 4),
    "mean_eval_forward_vel": round(final_vx, 4),
    "mean_eval_return": round(final_return, 4),
    "mean_eval_length": round(final_len, 2),
    "scripted_trot_forward_dist": round(trot_dist, 4),
    "scripted_trot_return": round(trot_return, 4),
    "walks_at_env_steps": walks_at,
    "curve": curve,
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "total_steps": args.total_steps,
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'walk.rrd'} + gait {args.out / 'walk_gait.rrd'}")
# --- endregion ---
