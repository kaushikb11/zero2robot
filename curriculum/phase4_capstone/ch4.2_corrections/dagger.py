"""zero2robot 4.2 — Corrections: Human-in-the-Loop Data (DAgger).

Chapter 1.1 taught you WHY behavior cloning dies: covariate shift. BC only ever
sees the states the demonstrator visited. At deploy time the policy drifts a
little, lands in a state no demo covered, guesses worse, drifts further — and
the block slides past the goal. You measured that death. This file FIXES it.

DAgger (Dataset Aggregation, Ross et al. 2011) is the fix, and it is almost
embarrassingly direct: if the problem is that BC never saw the states the policy
visits, then GO GET THOSE STATES. Roll out the CURRENT policy, collect the
states it actually visits (including the drifted ones), ask the expert what to do
on THOSE states, aggregate the new (state, expert-action) pairs into the dataset,
and retrain. Iterate. The policy learns to recover from its own mistakes.

Here the "expert" is the scripted PushT controller from `common/envs/pusht` — the
offline stand-in for a HUMAN teleoperator correcting the robot through the browser
playground. The mechanism is identical: label the states the policy visits with
the action a competent controller would take. The browser-teleop version (the
demo follow-up) swaps the scripted labeler for your hand on the mouse; nothing
else changes.

To make covariate shift MEASURABLE on the free tier we manufacture it honestly
(the ch1.6 held-out trick): the BC demos come from a NARROW region — the block
only ever starts CLOSE to the goal, the demonstrator's limited practice set — but
we DEPLOY on the full task, where the block starts anywhere. BC never saw the far
starts; it covariate-shifts and fails. DAgger's on-policy corrections cover
exactly those far starts, which more narrow demos never could.

Run it:      python curriculum/phase4_capstone/ch4.2_corrections/dagger.py --seed 0 --device cpu
CI smoke:    python curriculum/phase4_capstone/ch4.2_corrections/dagger.py --smoke --seed 0 --no-rerun

The success rates in this file are measured on CPU at --seed 0. On Apple Silicon
pin --device cpu to reproduce them: mps diverges (a Phase-1 finding).

Everything is torch + numpy + mujoco — the Wilson intervals (ch1.6) are hand-rolled,
no scipy.
"""

# --- region: setup ---
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch1.1 / ch1.6).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv, ScriptedExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

OBS_DIM, ACT_DIM = PushTEnv.OBS_DIM, PushTEnv.ACT_DIM
EVAL_BASE = 10_000  # eval start seeds live here; demos use [--seed, ...) — held out by construction

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch4.2-corrections"))
parser.add_argument("--seed", type=int, default=0, help="seeds demos, every retrain, the beta-mix, and the eval order")
parser.add_argument("--bc_demos", type=int, default=50, help="narrow-region expert demos for the BC seed policy; smoke: 6")
parser.add_argument("--r_max", type=float, default=0.13,
                    help="demos only from starts with block-to-goal distance <= this (the covariate-shift knob; env spawns 0.10..0.24)")
parser.add_argument("--dagger_iters", type=int, default=4, help="rounds of correct-aggregate-retrain; smoke: 1")
parser.add_argument("--rollouts", type=int, default=40, help="policy rollout episodes collected per DAgger round; smoke: 3")
parser.add_argument("--beta_decay", type=float, default=0.7,
                    help="beta_i = beta_decay**(i-1): the DAgger mixture. beta=1 executes the expert (round 1), decaying to the policy")
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--epochs", type=int, default=300, help="BC + each retrain; cpu: ~seconds each | smoke: 3")
parser.add_argument("--eval_episodes", type=int, default=10, help="episodes PER eval suite; smoke: 3")
parser.add_argument("--n_seeds", type=int, default=20, help="independent eval suites; pooled N = n_seeds * eval_episodes")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())  # cpu: deterministic (statistical repro on gpu/mps)
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # returns the numpy Generator the beta-mix draws from
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.bc_demos, args.dagger_iters, args.rollouts = 6, 1, 3
    args.epochs, args.eval_episodes, args.n_seeds, args.device = 3, 3, 2, "cpu"
banner("ch4.2-corrections", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch4.2-corrections", spawn=False)
    rr.save(str(args.out / "dagger.rrd"))
# --- endregion ---

# --- region: stats ---
# A success rate k/n is a binomial proportion; every rate this chapter prints
# ships as a Wilson score interval (ch1.6, hand-rolled — no scipy). The DAgger
# recovery is only believable BECAUSE the BC and DAgger intervals separate.
Z95 = 1.959963985  # 0.975 standard-normal quantile (95% two-sided)


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials (ch1.6). Always
    inside [0, 1], never degenerate at k=0 or k=n — the interval a rate ships with."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def diff_ci(ka: int, na: int, kb: int, nb: int, z: float = Z95) -> tuple[float, float]:
    """Newcombe hybrid-score interval for p_a - p_b (ch1.6). The verdict on
    'did DAgger really beat BC': if the interval EXCLUDES 0 the recovery is
    significant at this N; if it contains 0 you have not established it."""
    pa, pb = ka / na, kb / nb
    la, ha = wilson_ci(ka, na, z)
    lb, hb = wilson_ci(kb, nb, z)
    d = pa - pb
    lo = d - math.sqrt((pa - la) ** 2 + (hb - pb) ** 2)
    hi = d + math.sqrt((ha - pa) ** 2 + (pb - lb) ** 2)
    return (lo, hi)
# --- endregion ---

# --- region: policy ---
# The ch1.1 behavior-cloning policy, unchanged: a reactive 3-layer MLP,
# obs float32[10] -> action float32[2], with normalization living inside the
# module as buffers. DAgger changes the DATA the policy trains on, never the
# policy. That is the whole point — same clone, honest states.
class BCPolicy(nn.Module):
    def __init__(self, hidden_dim: int, stats: dict[str, np.ndarray]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, ACT_DIM),
        )
        for name, value in stats.items():
            self.register_buffer(name, torch.from_numpy(value))  # saved with the weights, never trained

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (2.0 * (obs - self.obs_min) / self.obs_range - 1.0).clamp(-1.0, 1.0)
        return (self.net(normalized) + 1.0) / 2.0 * self.act_range + self.act_min


def norm_stats(obs: np.ndarray, act: np.ndarray) -> dict[str, np.ndarray]:
    """Per-dim min/max -> [-1, 1] (ch1.1). Constant dims (the fixed target) carry
    range 0; give them range 1 so they map to a constant, not a divide-by-zero."""
    omin, amin = obs.min(0), act.min(0)
    orange = np.where(obs.max(0) - omin < 1e-4, np.float32(1.0), obs.max(0) - omin)
    arange = np.where(act.max(0) - amin < 1e-4, np.float32(1.0), act.max(0) - amin)
    return {"obs_min": omin, "obs_range": orange, "act_min": amin, "act_range": arange}


def train_bc(obs: np.ndarray, act: np.ndarray, stats: dict, hidden: int, epochs: int, seed: int) -> BCPolicy:
    """Plain MSE behavior cloning (ch1.1), seeded per call so training is
    bit-reproducible on CPU. `stats` is FROZEN to the BC demos and passed in: if
    we recomputed normalization on the DAgger aggregate, far-field flailing
    frames would blow up the ranges and wreck the on-task fit."""
    torch.manual_seed(seed)
    policy = BCPolicy(hidden, stats).to(device)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ot, at = torch.from_numpy(obs).to(device), torch.from_numpy(act).to(device)
    shuffle = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for batch in torch.randperm(len(ot), generator=shuffle).split(256):
            loss = F.mse_loss(policy(ot[batch]), at[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return policy.eval()


def act_fn(policy: BCPolicy):
    def act(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return policy(torch.from_numpy(obs).to(device).unsqueeze(0))[0].cpu().numpy()
    return act
# --- endregion ---

# --- region: data ---
# The scripted expert is the corrector. In the browser follow-up this is YOUR
# teleop; the mechanism — "label the state the policy is in with a good action" —
# is identical, which is exactly why the offline stand-in is honest.
def collect_bc_demos(n: int, seed: int, r_max: float):
    """Expert demos from the NARROW practice region only: keep an episode iff the
    block STARTS within r_max of the goal. This manufactures the covariate shift —
    deployment (full annulus 0.10..0.24) has far starts these demos never cover."""
    env = PushTEnv()
    obs_buf, act_buf = [], []
    accepted, s = 0, seed
    while accepted < n:
        obs = env.reset(s)
        s += 1
        if float(np.hypot(*env.tee_pose[:2])) > r_max:
            continue  # start too far from the goal — outside the demonstrator's practice set
        expert = ScriptedExpert(noise=0.0, seed=s)
        done = False
        while not done:
            a = expert.action(env)
            obs_buf.append(obs)
            act_buf.append(a)
            obs, _, done, _ = env.step(a)
        accepted += 1
    return np.asarray(obs_buf, np.float32), np.asarray(act_buf, np.float32)


def dagger_rollout(policy: BCPolicy, n: int, seed: int, beta: float):
    """One DAgger round of data collection on the FULL deployment distribution.

    At every state the trajectory visits, record the EXPERT's action (the label —
    what a good controller, or your teleop hand, would do HERE). The action
    EXECUTED is a beta-mixture: with prob beta the expert's, else the policy's.
    High beta early keeps the visited states near the expert manifold instead of
    flooding the set with far-off-manifold flailing; as beta decays the policy
    drives and we collect corrections on the states IT causes (Ross et al.)."""
    env = PushTEnv()
    act = act_fn(policy)
    obs_buf, act_buf = [], []
    for i in range(n):
        obs = env.reset(seed + i)  # full annulus — the deployment distribution
        expert = ScriptedExpert(noise=0.0, seed=seed + i)
        done = False
        while not done:
            expert_a = expert.action(env)  # the LABEL for this visited state
            obs_buf.append(obs)
            act_buf.append(expert_a)
            drive = expert_a if rng.random() < beta else act(obs)  # beta-mix executes
            obs, _, done, _ = env.step(drive)
    return np.asarray(obs_buf, np.float32), np.asarray(act_buf, np.float32)


def eval_suites(policy: BCPolicy, n_seeds: int, ep: int) -> np.ndarray:
    """n_seeds independent suites of `ep` rollouts on held-out full-distribution
    starts. Returns (n_seeds, ep) bool successes; pooled they feed the Wilson CI."""
    env = PushTEnv()
    act = act_fn(policy)
    out = np.zeros((n_seeds, ep), dtype=bool)
    for s in range(n_seeds):
        for e in range(ep):
            obs = env.reset(EVAL_BASE + s * ep + e)  # never a start we trained on
            done, info = False, {}
            while not done:
                obs, _, done, info = env.step(act(obs))
            out[s, e] = bool(info["success"])
    return out
# --- endregion ---

# --- region: loop ---
# Freeze normalization to the BC demos, then run the loop: BC seed policy, then
# correct -> aggregate -> retrain, evaluating every round with its Wilson CI.
obs, act = collect_bc_demos(args.bc_demos, args.seed, args.r_max)
stats = norm_stats(obs, act)
print(f"BC demos: {args.bc_demos} narrow-region episodes / {len(obs)} frames (r_max={args.r_max})")


def evaluate(policy, label: str, it: int):
    o = eval_suites(policy, args.n_seeds, args.eval_episodes)
    k, n = int(o.sum()), o.size
    lo, hi = wilson_ci(k, n)
    print(f"round {it} ({label:<6s}) success {k}/{n} = {k/n:.3f}  CI [{lo:.3f}, {hi:.3f}]  dataset {len(obs)}")
    if args.rerun:
        rr.set_time("dagger_round", sequence=it)
        rr.log("eval/success_rate", rr.Scalars([k / n]))
        rr.log("eval/ci_low", rr.Scalars([lo]))
        rr.log("eval/ci_high", rr.Scalars([hi]))
        rr.log("data/frames", rr.Scalars([float(len(obs))]))
    return k, n


policy = train_bc(obs, act, stats, args.hidden_dim, args.epochs, args.seed)
kbc0, nbc0 = evaluate(policy, "BC", 0)
rounds = [("BC", kbc0, nbc0)]
best_policy, best_rate = policy, kbc0 / nbc0  # best over ALL rounds incl. BC (Ross et al.), selected on the held-out eval — a mild winner's curse; the recovery survives Bonferroni multiplicity correction (see meta HONESTY)
for it in range(1, args.dagger_iters + 1):
    beta = args.beta_decay ** (it - 1)  # 1.0, then decaying — the DAgger mixture
    ro, ra = dagger_rollout(policy, args.rollouts, args.seed + 1000 * it, beta)
    obs = np.concatenate([obs, ro])  # AGGREGATE (the second D in DAgger)
    act = np.concatenate([act, ra])
    policy = train_bc(obs, act, stats, args.hidden_dim, args.epochs, args.seed)
    k, n = evaluate(policy, "DAgger", it)
    if k / n > best_rate:
        best_policy, best_rate = policy, k / n
    rounds.append((f"DAgger{it}", k, n))
# --- endregion ---

# --- region: verdict ---
# Ross et al.: RETURN THE BEST policy over rounds, not the last — over-iterating a
# weak reactive clone floods the aggregate with its own failure trajectories and
# the gains can regress (measured; the "how many rounds?" exercise makes you see
# it). The recovery is the diff CI between the best round and the BC baseline.
kbc, nbc = rounds[0][1], rounds[0][2]
best = max(rounds, key=lambda r: r[1] / r[2])  # argmax over ALL rounds incl. BC -> matches the saved best_policy (consistent even if no DAgger round beats BC: then best IS BC and the recovery is honestly ~0)
kb, nb = best[1], best[2]
d = diff_ci(kb, nb, kbc, nbc)
significant = d[0] > 0 or d[1] < 0
print(f"\nBC {kbc/nbc:.3f} {wilson_ci(kbc, nbc)} -> best {best[0]} {kb/nb:.3f} {wilson_ci(kb, nb)}")
print(f"recovery diff CI [{d[0]:+.3f}, {d[1]:+.3f}]  -> "
      f"{'RECOVERY SIGNIFICANT (CI excludes 0)' if significant else 'not significant at this N'}")

torch.save(best_policy, args.out / "dagger_policy.pt")  # best round; reload where THIS file's BCPolicy is importable
# TorchScript export for the browser-teleop follow-up. trace, not script: a
# fixed-architecture inference MLP traces exactly (no data-dependent control
# flow), and tracing sidesteps jit.script inspecting BCPolicy's numpy-typed
# __init__ annotation — a resolution that can trip under CI's parallel fan-out.
torch.jit.trace(best_policy.eval(), torch.zeros(1, OBS_DIM, device=device)).save(str(args.out / "dagger_policy.ts.pt"))

metrics = {
    "bc_demos": args.bc_demos,
    "bc_rate": round(kbc / nbc, 6),
    "bc_ci_lo": round(wilson_ci(kbc, nbc)[0], 6),
    "bc_ci_hi": round(wilson_ci(kbc, nbc)[1], 6),
    "beta_decay": args.beta_decay,
    "best_round": best[0],
    "best_rate": round(kb / nb, 6),
    "best_ci_lo": round(wilson_ci(kb, nb)[0], 6),
    "best_ci_hi": round(wilson_ci(kb, nb)[1], 6),
    "dagger_iters": args.dagger_iters,
    "n_pooled": nbc,
    "r_max": args.r_max,
    "recovery_diff_ci_lo": round(d[0], 6),
    "recovery_diff_ci_hi": round(d[1], 6),
    "recovery_significant": bool(significant),
    "round_rates": [round(k / n, 6) for _, k, n in rounds],
    "seed": args.seed,
    "smoke": bool(args.smoke),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'dagger.rrd'} — open it with: rerun {args.out / 'dagger.rrd'}")
# --- endregion ---
