"""zero2robot 4.3 — RL Post-Training: HIL-SERL in Sim (the capstone algorithm).

Where the whole arc converges. You cloned demos into a base policy (BC, ch1.1),
watched it die of covariate shift and FIXED it with CORRECTIONS on the states it
visits (DAgger, ch4.2 — the offline stand-in for a human teleoperating the robot
out of its own failures), and learned to squeeze a policy from a FIXED log using
the reward BC ignores (AWAC, the offline primer). HIL-SERL (Human-in-the-Loop
Sample-Efficient RL, Luo et al.) composes those three moves:

  1. take the CORRECTIONS (here: scripted-expert transitions on the task, ch4.2),
  2. turn them into an OFFLINE PRIOR with AWAC (twin-Q + advantage-weighted
     regression — the primer, reusing ch2.2's clipped double-Q),
  3. fine-tune ONLINE with SAC (ch2.2) while the corrections STAY in the replay
     (RLPD-style: every gradient step still sees them),

so the policy reaches a good return in FAR FEWER online samples than RL-from-
scratch. SAMPLE-EFFICIENCY is the whole point, and we MEASURE it: samples-to-
threshold, HIL-SERL (primed) vs SAC-from-scratch (cold), with ch1.6 error bars.
The measured lesson on this free-tier task is honest and blunt: the CORRECTIONS-
AS-PRIOR buy essentially all of the efficiency — the prior clears the threshold
before a single online step, while from-scratch SAC needs thousands of online
samples to catch up. Online fine-tuning on top of a prior already near this small
dense-reward task's short-horizon ceiling mostly HOLDS rather than improves it (we
return the best checkpoint, ch4.2's idiom); its gains show on the harder tasks of
the real HIL-SERL paper — and on the GATED capstone suite.

PUBLIC TASK, GATED LEADERBOARD: the map's capstone suite is HIDDEN-SEED graded
(grader/hidden_seeds, off-limits). This file runs the SAME algorithm on the PUBLIC
pusher_reach env (ch2.2's) so the mechanism is readable; the graded leaderboard
number is the gated capstone (ch4.4).

Run:   python curriculum/phase4_capstone/ch4.3_serl/serl.py --seed 0 --device cpu
Smoke: python .../serl.py --smoke --seed 0 --no-rerun   (fast CI config; pin --device cpu on Apple Silicon — mps diverges)
"""

# --- region: setup ---
import argparse
import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # so curriculum.common resolves (ch2.2 pattern)

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusher_reach import PusherReachEnv, reach_action  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

RANDOM_DIST = 0.176  # m: mean final distance a random policy leaves (env baseline)
obs_dim, act_dim = PusherReachEnv.OBS_DIM, PusherReachEnv.ACT_DIM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch4.3-serl"))
parser.add_argument("--corr_episodes", type=int, default=60, help="scripted-expert correction episodes (the prior's whole data); smoke: 8")
parser.add_argument("--prior_steps", type=int, default=8000, help="AWAC gradient steps to build the offline prior; smoke: 300")
parser.add_argument("--hil_steps", type=int, default=6000, help="online SAC fine-tune steps for HIL-SERL (primed); smoke: 600")
parser.add_argument("--scratch_steps", type=int, default=12000, help="online SAC steps for the from-scratch baseline; smoke: 600")
parser.add_argument("--threshold", type=float, default=0.10, help="samples-to-threshold bar: eval mean final dist below this (m). random ~0.176; env success is <0.02. 0.10 is a clear NEAR-SOLVE bar sitting above the prior's ~0.06m — the sample-efficiency gap is robust anywhere in 0.08-0.176m, not chosen-to-fit (ex1 lets you sweep it)")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=256)  # pusher is tiny; width is not the bottleneck
parser.add_argument("--lr", type=float, default=3e-4, help="shared Adam lr for actor and critics")
parser.add_argument("--gamma", type=float, default=0.99, help="reward discount")
parser.add_argument("--tau", type=float, default=0.005, help="soft target-update rate (ch2.2 idiom)")
parser.add_argument("--beta", type=float, default=0.3, help="AWAC temperature: small -> sharp advantage weighting, large -> toward BC (primer)")
parser.add_argument("--weight_clip", type=float, default=20.0, help="cap on exp(advantage/beta) so one correction can't dominate (primer)")
parser.add_argument("--alpha", type=float, default=0.1, help="SAC entropy temperature, fixed small (gentle online fine-tune; ch2.2 autotunes)")
parser.add_argument("--learning_starts", type=int, default=1000, help="random-action warmup for the FROM-SCRATCH arm only (HIL-SERL is primed)")
parser.add_argument("--buffer_size", type=int, default=200_000, help="replay capacity > all transitions, so corrections are NEVER evicted")
parser.add_argument("--eval_interval", type=int, default=1000, help="online steps between held-out evals (the sample-efficiency curve)")
parser.add_argument("--eval_episodes", type=int, default=10, help="episodes per eval suite; cpu: 10 | smoke: 4")
parser.add_argument("--n_seeds", type=int, default=3, help="eval suites for the final headline CI; pooled N = n_seeds * eval_episodes")
parser.add_argument("--seed", type=int, default=0, help="seeds torch, numpy, corrections, every training phase, and every eval reset")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())  # cpu: bitwise-deterministic
parser.add_argument("--smoke", action="store_true", help="tiny CPU run for CI; two runs must match byte-for-byte")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # returns the numpy Generator; also seeds torch/numpy globals
if args.smoke:  # pin everything the CI byte-compare depends on
    args.corr_episodes, args.prior_steps, args.hil_steps, args.scratch_steps = 8, 300, 600, 600
    args.batch_size, args.hidden_dim, args.eval_episodes, args.n_seeds = 64, 64, 4, 2
    args.learning_starts, args.eval_interval, args.device = 200, 300, "cpu"
banner("ch4.3-serl", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch4.3-serl", spawn=False)
    rr.save(str(args.out / "serl.rrd"))
# --- endregion ---

# --- region: stats ---
# ch1.6's error bars, inlined (numpy + math, no scipy — same code as the eval
# chapter and the primer). Wilson turns k/n into an honest [lo, hi]; the Newcombe
# difference CI decides whether HIL-SERL vs from-scratch is a REAL gap or noise.
Z95 = 1.959963985  # 0.975 standard-normal quantile (95% two-sided)


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def diff_ci(k_a: int, n_a: int, k_b: int, n_b: int, z: float = Z95) -> tuple[float, float]:
    """Newcombe hybrid-score CI for p_a - p_b. Excludes 0 => the ranking is real."""
    p_a, p_b = k_a / n_a, k_b / n_b
    lo_a, hi_a = wilson_ci(k_a, n_a, z)
    lo_b, hi_b = wilson_ci(k_b, n_b, z)
    d = p_a - p_b
    lo = d - math.sqrt((p_a - lo_a) ** 2 + (hi_b - p_b) ** 2)
    hi = d + math.sqrt((hi_a - p_a) ** 2 + (p_b - lo_b) ** 2)
    return (lo, hi)
# --- endregion ---

# --- region: model ---
# ONE actor + ONE critic, shared by the offline prior AND both online arms — the
# point is that HIL-SERL is the SAME networks, warm-started from corrections
# rather than from scratch. Actor: ch2.2's squashed-Gaussian policy, but we init
# log_std SMALL so a freshly-primed policy is near-deterministic — a wide-open
# policy would explore off the prior's good manifold and unlearn it immediately.
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class Actor(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -2.0)  # start near-deterministic (std ~0.13)

    def forward(self, obs: torch.Tensor):
        h = self.trunk(obs)
        return self.mean(h), self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def sample(self, obs: torch.Tensor):
        """Returns (action, log_prob, deterministic_action). rsample is the reparam
        trick (ch2.2); log(1 - tanh(x)^2) corrects the squashed log-prob."""
        mean, log_std = self(obs)
        normal = torch.distributions.Normal(mean, log_std.exp())
        x = normal.rsample()
        action = torch.tanh(x)
        log_prob = (normal.log_prob(x) - torch.log(1.0 - action.pow(2) + 1e-6)).sum(1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class Critic(nn.Module):
    """Q(obs, action). We build TWO (twin critics) and always bootstrap from the
    MIN — ch2.2's clipped double-Q, the brace the primer's AWAC critic reused."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=1))
# --- endregion ---

# --- region: data ---
# THE CORRECTIONS. ch4.2's mechanism: the scripted expert (reach_action, the
# offline stand-in for a human teleoperator) takes over on the task and drives it
# to the goal; we record its full transitions — a human-intervention log, exactly
# HIL-SERL's input. This is the ONLY data the offline prior sees; the same
# transitions are later pre-loaded into the online replay and STAY there (RLPD).
def collect_corrections(num_episodes: int) -> dict[str, torch.Tensor]:
    env = PusherReachEnv()
    cols: dict[str, list] = {k: [] for k in ("obs", "action", "reward", "next_obs", "terminated")}
    returns = []
    for ep in range(num_episodes):
        obs = env.reset(seed=args.seed + ep)  # episode ep uses a distinct, reproducible seed
        ep_return, done = 0.0, False
        while not done:
            action = reach_action(env)  # the correction: what the expert/teleop would do HERE
            next_obs, reward, done, info = env.step(action)
            for key, val in zip(cols, (obs, action, reward, next_obs, float(info["terminated"]))):
                cols[key].append(val)
            obs = next_obs
            ep_return += reward
        returns.append(ep_return)
    data = {k: torch.tensor(np.array(v), dtype=torch.float32, device=device) for k, v in cols.items()}
    data["reward"], data["terminated"] = data["reward"].unsqueeze(1), data["terminated"].unsqueeze(1)
    print(f"corrections: {num_episodes} episodes / {len(data['obs'])} transitions  "
          f"expert return {np.mean(returns):.2f} (the prior's whole data)")
    return data
# --- endregion ---

# --- region: replay ---
class ReplayBuffer:
    """ch2.2's circular replay. HIL-SERL pre-loads it with the corrections before
    the first online step and NEVER evicts them (buffer_size > all transitions),
    so every online SAC update samples corrections mixed with fresh online data —
    the RLPD anchor that keeps fine-tuning from wandering off the prior. The
    from-scratch arm starts it empty."""

    def __init__(self, capacity: int):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity, 1), dtype=np.float32)
        self.capacity, self.ptr, self.size = capacity, 0, 0

    def add(self, obs, action, reward, next_obs, terminated):
        i = self.ptr
        self.obs[i], self.actions[i], self.rewards[i, 0] = obs, action, reward
        self.next_obs[i], self.terminated[i, 0] = next_obs, float(terminated)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def preload(self, corr: dict[str, torch.Tensor]):
        c = {k: v.cpu().numpy() for k, v in corr.items()}
        for i in range(len(c["obs"])):
            self.add(c["obs"][i], c["action"][i], c["reward"][i, 0], c["next_obs"][i], c["terminated"][i, 0])

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.size, (batch_size,)).numpy()  # seeded torch RNG -> reproducible on CPU
        t = lambda a: torch.as_tensor(a[idx], device=device)  # noqa: E731
        return t(self.obs), t(self.actions), t(self.rewards), t(self.next_obs), t(self.terminated)
# --- endregion ---

# --- region: prior ---
# THE OFFLINE PRIOR (the primer's AWAC, on the corrections). A twin-Q critic learns
# the reward-to-go on the fixed corrections; the policy is extracted by advantage-
# weighted regression — ch1.1's BC loss on the correction action, each sample scaled
# by exp(A/beta), A = Q(s, a_corr) - Q(s, pi(s)). This anchors the policy to actions
# the corrections contain (no OOD extrapolation) and warm-starts BOTH the actor AND
# the critics online SAC continues from — the "corrections as prior."
def train_prior(corr: dict[str, torch.Tensor]):
    torch.manual_seed(args.seed)  # net init + sampling reproducible regardless of upstream RNG use
    actor = Actor(args.hidden_dim).to(device)
    q1, q2 = Critic(args.hidden_dim).to(device), Critic(args.hidden_dim).to(device)
    q1_targ = copy.deepcopy(q1).requires_grad_(False)
    q2_targ = copy.deepcopy(q2).requires_grad_(False)
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    o, a, r, no, t = (corr[k] for k in ("obs", "action", "reward", "next_obs", "terminated"))
    n = len(o)
    for step in range(args.prior_steps):
        idx = torch.randint(0, n, (args.batch_size,))
        obs, action, reward, next_obs, term = o[idx], a[idx], r[idx], no[idx], t[idx]
        with torch.no_grad():  # critic: TD toward r + gamma * min_targ Q(s', pi(s'))
            next_action, _, _ = actor.sample(next_obs)
            min_next_q = torch.min(q1_targ(next_obs, next_action), q2_targ(next_obs, next_action))
            target_q = reward + args.gamma * (1.0 - term) * min_next_q
        critic_loss = F.mse_loss(q1(obs, action), target_q) + F.mse_loss(q2(obs, action), target_q)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()
        _, _, pred = actor.sample(obs)  # deterministic mean action
        with torch.no_grad():  # AWAC advantage weight
            q_data = torch.min(q1(obs, action), q2(obs, action))
            value = torch.min(q1(obs, pred), q2(obs, pred))
            weight = torch.exp((q_data - value) / args.beta).clamp(max=args.weight_clip)
        actor_loss = (weight * (pred - action).pow(2).mean(1, keepdim=True)).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()
        soft_update(((q1, q1_targ), (q2, q2_targ)))
        if args.rerun and step % 100 == 0:
            rr.set_time("prior_step", sequence=step)
            rr.log("prior/critic_loss", rr.Scalars([critic_loss.item() / 2]))
    return actor, q1, q2, q1_targ, q2_targ


def soft_update(pairs):
    """Polyak-average targets a hair toward online (ch2.2)."""
    with torch.no_grad():
        for online, targ in pairs:
            for p, tp in zip(online.parameters(), targ.parameters()):
                tp.mul_(1.0 - args.tau).add_(args.tau * p)
# --- endregion ---

# --- region: online ---
# ONE online SAC loop, called TWICE: warm (HIL-SERL — primed nets + corrections
# pre-loaded in replay) and cold (from-scratch — fresh nets, empty replay, random
# warmup). Identical algorithm; the ONLY differences are the starting weights and
# the replay's initial contents. It returns the sample-efficiency curve, the FIRST
# online step it clears the threshold, and the BEST checkpoint over the run (ch4.2's
# return-the-best: online can dip before recovering, and the prior may be best).
def run_online(actor, q1, q2, q1_targ, q2_targ, buffer, total_steps, primed: bool, tag: str):
    actor_opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    env = PusherReachEnv()
    episode = 0
    obs = env.reset(seed=100_000 + args.seed + episode)
    episode += 1
    curve, sts = [], None
    best_dist, best_state = float("inf"), copy.deepcopy(actor.state_dict())
    if primed:  # evaluate the prior at ZERO online samples — the head start corrections buy
        d0, _, _ = evaluate(actor, args.eval_episodes)
        curve.append((0, round(d0, 5)))
        best_dist = d0
        sts = 0 if d0 < args.threshold else None
    for step in range(1, total_steps + 1):
        if not primed and step <= args.learning_starts:  # cold start: random warmup fills the buffer
            action = rng.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
        else:
            with torch.no_grad():
                action, _, _ = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            action = action[0].cpu().numpy()
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, info["terminated"])
        obs = env.reset(seed=100_000 + args.seed + episode) if done else next_obs
        episode += 1 if done else 0
        if primed or step > args.learning_starts:  # one SAC gradient step per env step
            sac_update(actor, q1, q2, q1_targ, q2_targ, actor_opt, critic_opt, buffer)
        if step % args.eval_interval == 0 or step == total_steps:
            eval_dist, _, _ = evaluate(actor, args.eval_episodes)
            curve.append((step, round(eval_dist, 5)))
            if eval_dist < best_dist:
                best_dist, best_state = eval_dist, copy.deepcopy(actor.state_dict())
            if sts is None and eval_dist < args.threshold:
                sts = step
            if args.rerun:
                rr.set_time("online_step", sequence=step)
                rr.log(f"eval/{tag}/dist", rr.Scalars([eval_dist]))
            print(f"  [{tag}] step {step:6d}/{total_steps}  eval_dist {eval_dist:.4f}m  best {best_dist:.4f}m")
    actor.load_state_dict(best_state)  # return the BEST policy over the run (prior included)
    return curve, sts, best_dist


def sac_update(actor, q1, q2, q1_targ, q2_targ, actor_opt, critic_opt, buffer):
    """One SAC gradient step on a replay batch (ch2.2), fixed entropy temperature.
    For HIL-SERL the batch mixes fresh online transitions with the pre-loaded
    corrections, which keep pulling the critic toward known-good actions."""
    obs, action, reward, next_obs, terminated = buffer.sample(args.batch_size)
    with torch.no_grad():
        next_action, next_logp, _ = actor.sample(next_obs)
        min_next_q = torch.min(q1_targ(next_obs, next_action), q2_targ(next_obs, next_action)) - args.alpha * next_logp
        target_q = reward + args.gamma * (1.0 - terminated) * min_next_q
    critic_loss = F.mse_loss(q1(obs, action), target_q) + F.mse_loss(q2(obs, action), target_q)
    critic_opt.zero_grad()
    critic_loss.backward()
    critic_opt.step()
    new_action, logp, _ = actor.sample(obs)
    actor_loss = (args.alpha * logp - torch.min(q1(obs, new_action), q2(obs, new_action))).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    soft_update(((q1, q1_targ), (q2, q2_targ)))
# --- endregion ---

# --- region: eval ---
# Held-out eval with the DETERMINISTIC policy (tanh of the mean), on reset seeds
# disjoint from corrections (seed+ep) and online (100_000+). n_suites x
# eval_episodes pooled -> the success count the Wilson CI reads.
def evaluate(actor, eval_episodes: int, n_suites: int = 1) -> tuple[float, int, int]:
    eval_env = PusherReachEnv()
    finals, successes = [], []
    actor.eval()
    for s in range(n_suites):
        for e in range(eval_episodes):
            obs = eval_env.reset(seed=500_000 + args.seed + s * eval_episodes + e)
            done, info = False, {"dist": eval_env._dist(), "success": False}
            while not done:
                with torch.no_grad():
                    _, _, mean_action = actor.sample(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                obs, _, done, info = eval_env.step(mean_action[0].cpu().numpy())
            finals.append(info["dist"])
            successes.append(float(info["success"]))
    actor.train()
    return float(np.mean(finals)), int(np.sum(successes)), len(successes)
# --- endregion ---

# --- region: report ---
# Build the prior from corrections, run BOTH arms, grade with error bars. The
# headline is sample-efficiency: samples-to-threshold, HIL-SERL vs scratch.
corr = collect_corrections(args.corr_episodes)
prior_actor, q1, q2, q1_targ, q2_targ = train_prior(corr)
prior_dist, prior_k, _ = evaluate(prior_actor, args.eval_episodes, args.n_seeds)
print(f"\noffline prior (AWAC on corrections): eval_dist {prior_dist:.4f}m  (0 online samples)")

hil_buffer = ReplayBuffer(args.buffer_size)
hil_buffer.preload(corr)  # the corrections STAY in the replay (RLPD)
print(f"\nHIL-SERL: primed + {hil_buffer.size} corrections in replay, fine-tune {args.hil_steps} online steps")
hil_curve, hil_sts, hil_best = run_online(prior_actor, q1, q2, q1_targ, q2_targ, hil_buffer, args.hil_steps, True, "hil")

torch.manual_seed(args.seed)  # from-scratch SAC: fresh nets, empty replay, random warmup
scr_actor = Actor(args.hidden_dim).to(device)
s1, s2 = Critic(args.hidden_dim).to(device), Critic(args.hidden_dim).to(device)
s1_targ, s2_targ = copy.deepcopy(s1).requires_grad_(False), copy.deepcopy(s2).requires_grad_(False)
print(f"\nfrom-scratch SAC: cold, empty replay, {args.scratch_steps} online steps")
scr_curve, scr_sts, scr_best = run_online(scr_actor, s1, s2, s1_targ, s2_targ, ReplayBuffer(args.buffer_size), args.scratch_steps, False, "scratch")

n_pool = args.n_seeds * args.eval_episodes
_, hil_pk, _ = evaluate(prior_actor, args.eval_episodes, args.n_seeds)
_, scr_pk, _ = evaluate(scr_actor, args.eval_episodes, args.n_seeds)
gap = diff_ci(hil_pk, n_pool, scr_pk, n_pool)  # HIL - scratch
gap_real = gap[0] > 0 or gap[1] < 0

print(f"\n[headline: sample-efficiency] samples-to-threshold (eval_dist < {args.threshold}m):")
print(f"  HIL-SERL (primed) : {hil_sts if hil_sts is not None else '>' + str(args.hil_steps)} online samples  ({'prior clears it at 0; ' if hil_sts == 0 else ''}best {hil_best:.4f}m)")
print(f"  SAC from scratch  : {scr_sts if scr_sts is not None else '>' + str(args.scratch_steps)} online samples  (best {scr_best:.4f}m)")
if hil_sts == 0 and scr_sts:
    print(f"  -> corrections-as-prior clear the bar with ZERO online samples; scratch needs ~{scr_sts}. That gap IS the sample efficiency.")

print(f"\n[ablation: what each piece buys] pooled N={n_pool} held-out rollouts each:")
for name, k, dist in (("prior alone (offline, 0 online)", prior_k, prior_dist),
                      ("scratch (online, no prior/corr)", scr_pk, scr_best),
                      ("HIL-SERL (prior+corr+online)", hil_pk, hil_best)):
    print(f"  {name:<34s} success {k}/{n_pool}={k/n_pool:.2f} {tuple(round(x,2) for x in wilson_ci(k,n_pool))}  best_dist {dist:.4f}m")
print(f"  diff CI (HIL-SERL - scratch): {gap[0]:+.2f}..{gap[1]:+.2f}  ({'SIGNIFICANT' if gap_real else 'not significant at this N'})")
print(f"  (random ~{RANDOM_DIST}m; prior & HIL-SERL sit near this task's short-horizon SAC ceiling, so online fine-tuning")
print("   HOLDS the prior rather than beating it — the corrections buy the efficiency. Online post-training on harder")
print("   tasks is scored on the GATED capstone suite [ch4.4, hidden-seed].)")

if args.rerun:
    for name, k in (("prior", prior_k), ("scratch", scr_pk), ("hil", hil_pk)):
        rr.log(f"final/{name}/success_rate", rr.Scalars([k / n_pool]))

torch.save(prior_actor.state_dict(), args.out / "serl_actor.pt")  # HIL-SERL best checkpoint
torch.save(scr_actor.state_dict(), args.out / "scratch_actor.pt")  # from-scratch SAC baseline (site ONNX demo)
metrics = {
    "threshold": args.threshold, "corr_episodes": args.corr_episodes,
    "prior_eval_dist": round(prior_dist, 5), "prior_success_rate": round(prior_k / n_pool, 6),
    "hil_steps_to_threshold": hil_sts, "scratch_steps_to_threshold": scr_sts,
    "hil_best_dist": round(hil_best, 5), "scratch_best_dist": round(scr_best, 5),
    "hil_success_rate": round(hil_pk / n_pool, 6), "scratch_success_rate": round(scr_pk / n_pool, 6),
    "diff_ci_lo": round(gap[0], 6), "diff_ci_hi": round(gap[1], 6), "gap_significant": bool(gap_real),
    "hil_curve": hil_curve, "scratch_curve": scr_curve, "n_pooled": n_pool,
    "seed": args.seed, "smoke": bool(args.smoke),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"\nmetrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'serl.rrd'} — open it with: rerun {args.out / 'serl.rrd'}")
# --- endregion ---
