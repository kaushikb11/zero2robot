"""zero2robot 4.x — Offline RL Primer: Beat the Data With Its Own Reward.

MAP PLACEMENT IS UNASSIGNED (author decides the final number). This chapter
resolves the map's OPEN Phase-4 dependency: 4.3 (HIL-SERL) assumes an
"offline-primed" policy — RL fine-tuning that starts from a policy already
squeezed out of logged correction data — but no offline-RL chapter existed. This
primer supplies exactly that prior.

The setting: you have a FIXED dataset of logged transitions — some good demos,
some clumsy/noisy attempts — and you CANNOT collect more (no env interaction
while learning). Behavior cloning (ch1.1) can only CLONE that data: it regresses
the average action per state, so its ceiling is the dataset's average quality —
copy a mediocre mix and you get a mediocre policy. Offline RL does better by
using the REWARD. It fits a Q-function on the fixed data, then extracts a policy
that prefers the actions the reward says were ABOVE average — advantage-weighted
regression (AWAC/AWR family). The exact same regression BC does, but each sample
is weighted by how much better than the current policy that logged action was.

The catch, and why offline RL is its own algorithm: naive off-policy RL (learn Q,
then just MAXIMIZE it) breaks offline. With no fresh data to correct it, the
policy drifts to out-of-distribution actions where the Q-function is a fantasy —
it OVERESTIMATES actions it never saw, and the policy happily walks into them.
Run --naive to see it. The damage scales with how NARROW the data is: on the
broad expert+random mix the random half covers the action space, so the critic
stays honest and even naive-maxQ is fine; but on narrow, expert-only data
(--naive --expert_frac 1.0 — the shape of real demo/correction data) the naive
critic inflates ~7x while its policy collapses to near-random. AWAC's advantage
weighting is the fix that does not depend on coverage: it anchors the policy to
actions the data actually contains, so the critic is only ever asked about
actions it has evidence for. That coverage-independence is the offline prior 4.3
builds on — correction data is narrow.

We reuse ch2.2's twin-Q idiom (clipped double-Q + target nets) and ch1.1's BC
regression idiom, on ch2.2's dense-reward pusher_reach env, and grade BC vs
offline RL with ch1.6's error bars (Wilson CI + a difference CI).

Run it:      python curriculum/phase4_capstone/ch4_offline_primer/offline.py --seed 0
Break it:    python .../offline.py --seed 0 --naive --expert_frac 1.0  (NARROW data -> Q overestimates, eval collapses)
Investigate: python .../offline.py --seed 0 --expert_frac 0.6         (cleaner dataset)
CI smoke:    python .../offline.py --smoke --seed 0 --no-rerun
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

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch1.1 / ch2.2).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusher_reach import PusherReachEnv, reach_action  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

RANDOM_DIST = 0.176  # m: mean final distance a random policy leaves (env baseline)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch4-offline-primer"))
parser.add_argument("--episodes", type=int, default=200, help="fixed-dataset size (episodes); T4/cpu: 200 | smoke: 12")
parser.add_argument("--expert_frac", type=float, default=0.3,
                    help="fraction of dataset episodes from the clean scripted expert; the rest are RANDOM (mixed quality)")
parser.add_argument("--steps", type=int, default=8000, help="gradient steps for BOTH BC and the offline learner; cpu: minutes | smoke: 200")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--hidden_dim", type=int, default=256)  # pusher is tiny; width is not the bottleneck
parser.add_argument("--lr", type=float, default=3e-4, help="shared Adam lr for policy and critics")
parser.add_argument("--gamma", type=float, default=0.99, help="reward discount for the critic's TD target")
parser.add_argument("--tau", type=float, default=0.005, help="soft target-update rate (ch2.2 idiom)")
parser.add_argument("--beta", type=float, default=0.3, help="AWAC temperature: small -> sharp advantage weighting, large -> back toward BC")
parser.add_argument("--weight_clip", type=float, default=20.0, help="cap on exp(advantage/beta) so one transition can't dominate")
parser.add_argument("--eval_episodes", type=int, default=20)  # per suite; T4: 20 | smoke: 4
parser.add_argument("--n_seeds", type=int, default=5, help="independent eval suites; pooled N = n_seeds * eval_episodes")
parser.add_argument("--seed", type=int, default=0, help="seeds torch, numpy, the dataset, and every eval reset")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--naive", action="store_true",
                    help="Break It: extract the policy by MAXIMIZING Q with no data constraint (DDPG-style). Clearest with --expert_frac 1.0 (narrow data): Q overestimates, eval collapses")
parser.add_argument("--smoke", action="store_true", help="tiny CPU run for CI; two runs must match byte-for-byte")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # returns the numpy Generator the dataset + bootstrap draw from
if args.smoke:  # pin everything the CI byte-compare depends on
    args.episodes, args.steps, args.batch_size = 20, 400, 64
    args.eval_episodes, args.n_seeds, args.hidden_dim, args.device = 4, 2, 64, "cpu"
banner("ch4-offline-primer", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
obs_dim, act_dim = PusherReachEnv.OBS_DIM, PusherReachEnv.ACT_DIM
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch4-offline-primer", spawn=False)
    rr.save(str(args.out / "offline.rrd"))
# --- endregion ---

# --- region: data ---
# THE FIXED DATASET. This is the whole premise: we build it ONCE, then never
# touch the env again during learning. It is MIXED QUALITY on purpose — a
# `--expert_frac` slice is the clean scripted IK reacher (near-optimal), and the
# rest is a RANDOM policy (uniform torques — flails, rarely reaches). This is the
# canonical offline mix (the D4RL "expert + random" recipe): some good
# demonstrations buried in a lot of junk. BC averages over ALL of it and gets
# dragged toward the junk; offline RL uses the reward to tell the two apart. The
# random half also does the offline learner a favor — it COVERS the state-action
# space, so the critic sees what bad actions cost. We store full transitions
# (obs, action, reward, next_obs, terminated): BC needs only (obs, action), the
# critic needs the rest.
def build_dataset(num_episodes: int) -> dict[str, torch.Tensor]:
    env = PusherReachEnv()
    n_expert = int(round(args.expert_frac * num_episodes))
    cols: dict[str, list] = {k: [] for k in ("obs", "action", "reward", "next_obs", "terminated")}
    ret_expert, ret_noisy = [], []
    for ep in range(num_episodes):
        obs = env.reset(seed=args.seed + ep)  # episode ep uses a distinct, reproducible seed
        is_expert = ep < n_expert
        ep_return, done = 0.0, False
        while not done:
            action = (reach_action(env) if is_expert  # scripted IK reach — the expert
                      else rng.uniform(-1, 1, size=act_dim).astype(np.float32))  # the junk half
            next_obs, reward, done, info = env.step(action)
            cols["obs"].append(obs)
            cols["action"].append(action)
            cols["reward"].append(reward)
            cols["next_obs"].append(next_obs)
            cols["terminated"].append(float(info["terminated"]))
            obs = next_obs
            ep_return += reward
        (ret_expert if is_expert else ret_noisy).append(ep_return)
    data = {k: torch.tensor(np.array(v), dtype=torch.float32, device=device) for k, v in cols.items()}
    data["reward"] = data["reward"].unsqueeze(1)
    data["terminated"] = data["terminated"].unsqueeze(1)
    print(f"dataset: {num_episodes} episodes / {len(data['obs'])} transitions "
          f"({n_expert} expert, {num_episodes - n_expert} random)")
    print(f"  behavior return: expert {np.mean(ret_expert):7.2f}  random {np.mean(ret_noisy):7.2f}  "
          f"(BC clones the mix of these; offline RL should beat it)")
    return data


data = build_dataset(args.episodes)
N = len(data["obs"])


def sample_batch() -> tuple[torch.Tensor, ...]:
    idx = torch.randint(0, N, (args.batch_size,))  # seeded global torch RNG -> reproducible on CPU
    return (data["obs"][idx], data["action"][idx], data["reward"][idx],
            data["next_obs"][idx], data["terminated"][idx])
# --- endregion ---

# --- region: model ---
# Policy: obs -> action MLP, tanh-bounded to the env's [-1, 1] torque range. This
# is deliberately the SAME network BC and offline RL both train — the algorithms
# differ ONLY in the loss (see region: train), which is the whole lesson. Critic:
# a Q(obs, action) net; we build TWO (twin critics, ch2.2's clipped double-Q) and
# always bootstrap from the MIN, the brace that fights Q overestimation.
class Policy(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))  # actions live in [-1, 1]


class Critic(nn.Module):
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

# --- region: stats ---
# ch1.6's error bars, inlined (numpy + math only — no scipy). A success rate is a
# binomial proportion; the Wilson interval turns k/n into an honest [lo, hi], and
# the Newcombe difference CI decides whether BC vs offline RL is a REAL gap or
# just noise at this N. Same code as the eval chapter, on purpose.
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

# --- region: eval ---
# Loss on the dataset says how well we imitate; rollouts say what we care about —
# does the fingertip reach the target when the POLICY picks the states? We run
# n_seeds independent suites on HELD-OUT reset seeds (disjoint from the dataset's
# seeds by construction) and return per-episode success + final distance.
def evaluate(policy: Policy) -> tuple[np.ndarray, np.ndarray]:
    eval_env = PusherReachEnv()
    successes, finals = [], []
    policy.eval()
    for s in range(args.n_seeds):
        for e in range(args.eval_episodes):
            obs = eval_env.reset(seed=500_000 + args.seed + s * args.eval_episodes + e)
            done, info = False, {"success": False, "dist": eval_env._dist()}
            while not done:
                with torch.no_grad():
                    action = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
                obs, _, done, info = eval_env.step(action[0].cpu().numpy())
            successes.append(float(info["success"]))
            finals.append(info["dist"])
    policy.train()
    return np.array(successes), np.array(finals)
# --- endregion ---

# --- region: train ---
# The two learners, side by side. BC and offline RL share the policy network and
# the SAME regression — predict the dataset action — and differ in ONE line: the
# per-sample weight. That is the entire point of the chapter, so read the losses
# together.
def train_bc() -> Policy:
    """Behavior cloning: uniform-weighted MSE to the dataset action (ch1.1). The
    ceiling is the data's average quality — it cannot tell a good action from a
    bad one, because it never looks at the reward."""
    policy = Policy(args.hidden_dim).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    for step in range(args.steps):
        obs, action, *_ = sample_batch()
        loss = F.mse_loss(policy(obs), action)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if args.rerun and step % 50 == 0:
            rr.set_time("step", sequence=step)
            rr.log("bc/loss", rr.Scalars([loss.item()]))
    return policy


def train_offline() -> tuple[Policy, float]:
    """Offline RL. First a twin-Q critic learns the reward-to-go on the FIXED
    data (bootstrapping from the policy's own next action, ch2.2 idiom). Then the
    policy is extracted:
      AWAC  advantage-weighted regression — the BC loss, but each sample scaled by
            exp(A/beta) where A = Q(s,a_data) - Q(s,pi(s)). Actions the reward
            says beat the current policy get pulled toward; bad ones get ignored.
      naive (--naive) NO constraint: maximize Q(s, pi(s)) directly. Offline, the
            policy walks to out-of-distribution actions where Q is a fantasy — the
            overestimation the constraint exists to prevent. Worst on narrow data
            (--expert_frac 1.0). This is the Break It.
    """
    policy = Policy(args.hidden_dim).to(device)
    q1, q2 = Critic(args.hidden_dim).to(device), Critic(args.hidden_dim).to(device)
    q1_targ = copy.deepcopy(q1).requires_grad_(False)
    q2_targ = copy.deepcopy(q2).requires_grad_(False)
    policy_opt = torch.optim.Adam(policy.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=args.lr)
    mean_q = float("nan")
    for step in range(args.steps):
        obs, action, reward, next_obs, terminated = sample_batch()
        # --- critic: TD-regress both Q's toward r + gamma * min_targ Q(s', pi(s')) ---
        with torch.no_grad():
            next_action = policy(next_obs)
            min_next_q = torch.min(q1_targ(next_obs, next_action), q2_targ(next_obs, next_action))
            target_q = reward + args.gamma * (1.0 - terminated) * min_next_q
        q1_loss, q2_loss = F.mse_loss(q1(obs, action), target_q), F.mse_loss(q2(obs, action), target_q)
        critic_opt.zero_grad()
        (q1_loss + q2_loss).backward()
        critic_opt.step()
        # --- policy extraction: the ONE line that separates the algorithms ---
        pred = policy(obs)
        if args.naive:  # maximize Q with no anchor to the data -> divergence offline
            policy_loss = -torch.min(q1(obs, pred), q2(obs, pred)).mean()
        else:  # AWAC: advantage-weighted regression toward the dataset action
            with torch.no_grad():
                q_data = torch.min(q1(obs, action), q2(obs, action))          # Q of the logged action
                value = torch.min(q1(obs, pred), q2(obs, pred))               # V(s) = Q(s, pi(s)) baseline
                weight = torch.exp((q_data - value) / args.beta).clamp(max=args.weight_clip)
            policy_loss = (weight * (pred - action).pow(2).mean(1, keepdim=True)).mean()
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()
        # --- soft target update (Polyak), ch2.2 idiom ---
        with torch.no_grad():
            for online, targ in ((q1, q1_targ), (q2, q2_targ)):
                for p, tp in zip(online.parameters(), targ.parameters()):
                    tp.mul_(1.0 - args.tau).add_(args.tau * p)
        mean_q = 0.5 * (q1(obs, action) + q2(obs, action)).mean().item()
        if args.rerun and step % 50 == 0:
            rr.set_time("step", sequence=step)
            rr.log("offline/critic_loss", rr.Scalars([(q1_loss + q2_loss).item() / 2]))
            rr.log("offline/policy_loss", rr.Scalars([policy_loss.item()]))
            rr.log("offline/mean_q", rr.Scalars([mean_q]))  # watch this EXPLODE under --naive
    return policy, mean_q
# --- endregion ---

# --- region: report ---
# Train both, then grade them with error bars. The headline is the difference CI:
# does offline RL BEAT BC on the SAME fixed dataset, and is the gap real at this N?
bc_policy = train_bc()
offline_policy, offline_mean_q = train_offline()
mode = "naive-maxQ" if args.naive else "AWAC"
print(f"\ntrained BC and offline ({mode}); offline final mean|Q| over data = {abs(offline_mean_q):.1f}")

bc_succ, bc_dist = evaluate(bc_policy)
off_succ, off_dist = evaluate(offline_policy)
n_pool = args.n_seeds * args.eval_episodes
bc_k, off_k = int(bc_succ.sum()), int(off_succ.sum())
gap = diff_ci(off_k, n_pool, bc_k, n_pool)  # offline - BC; excludes 0 => real
gap_real = gap[0] > 0 or gap[1] < 0
verdict = ("offline RL BEATS BC" if gap[0] > 0 else
           "BC beats offline RL" if gap[1] < 0 else "tie (diff CI spans 0)")

print(f"\n[headline] BC vs offline RL on the same fixed dataset (N={n_pool} rollouts each):")
print(f"  BC       success {bc_k}/{n_pool} = {bc_k/n_pool:.2f}  {tuple(round(x,2) for x in wilson_ci(bc_k,n_pool))}"
      f"   mean final dist {bc_dist.mean():.4f} m")
print(f"  offline  success {off_k}/{n_pool} = {off_k/n_pool:.2f}  {tuple(round(x,2) for x in wilson_ci(off_k,n_pool))}"
      f"   mean final dist {off_dist.mean():.4f} m")
print(f"  diff CI (offline - BC): {gap[0]:+.2f}..{gap[1]:+.2f}  ->  {verdict}"
      f"  ({'SIGNIFICANT' if gap_real else 'not significant at this N'})")
print(f"  (random baseline ~{RANDOM_DIST} m; lower dist is better)")
if args.naive:
    print(f"\n[Break It] naive offline RL (maximize Q, no data constraint): mean|Q| over data = {abs(offline_mean_q):.1f}.")
    print("  On narrow data (--expert_frac 1.0) this inflates ~7x vs AWAC's ~1 while eval collapses toward random:")
    print("  the policy chases OOD actions the critic OVERESTIMATES. On the broad expert+random mix, coverage")
    print("  keeps the critic honest and naive survives — which is exactly why narrow correction data needs the constraint.")

if args.rerun:
    for name, k, dist in (("bc", bc_k, bc_dist), ("offline", off_k, off_dist)):
        lo, hi = wilson_ci(k, n_pool)
        rr.log(f"eval/{name}/success_rate", rr.Scalars([k / n_pool]))
        rr.log(f"eval/{name}/success_ci", rr.Scalars([lo, hi]))
        rr.log(f"eval/{name}/mean_dist", rr.Scalars([float(dist.mean())]))

torch.save(offline_policy.state_dict(), args.out / "offline_policy.pt")
torch.save(bc_policy.state_dict(), args.out / "bc_policy.pt")  # the BC baseline arm, for the site ONNX demo
metrics = {
    "mode": mode,
    "episodes": args.episodes,
    "expert_frac": args.expert_frac,
    "steps": args.steps,
    "beta": args.beta,
    "n_pooled": n_pool,
    "bc_success_rate": round(bc_k / n_pool, 6),
    "offline_success_rate": round(off_k / n_pool, 6),
    "bc_mean_final_dist": round(float(bc_dist.mean()), 6),
    "offline_mean_final_dist": round(float(off_dist.mean()), 6),
    "diff_ci_lo": round(gap[0], 6),
    "diff_ci_hi": round(gap[1], 6),
    "gap_significant": bool(gap_real),
    "offline_mean_abs_q": round(abs(offline_mean_q), 4),
    "naive": bool(args.naive),
    "seed": args.seed,
    "smoke": bool(args.smoke),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"\nmetrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'offline.rrd'} — open it with: rerun {args.out / 'offline.rrd'}")
# --- endregion ---
