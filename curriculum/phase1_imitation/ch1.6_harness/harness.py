"""zero2robot 1.6 — Evaluation Is Hard.

Chapters 1.3, 1.4, and 1.5 kept reporting success rates over TWENTY-odd rollouts.
Some of those gaps are decisively real ("ACT 0.88 vs an untrained 0.0"); one is
genuinely within the noise ("flow 0.40 vs diffusion 0.25", whose difference CI
spans zero) — and you cannot tell which by staring at the point estimates. A
success rate is a coin you flipped ~20 times; 0.40 is not a number, it is the
center of a band. Report the band; it tells you which rankings hold. This file
is the harness the earlier chapters should have graded
themselves with:

  1. SEEDED SUITES   run a policy over many independent eval suites and watch
                     the single-suite number swing across seeds.
  2. CONFIDENCE      a Wilson score interval on the success rate (from scratch,
     INTERVALS       numpy only) + a percentile bootstrap that agrees with it,
                     so every "0.40" ships as a [lo, hi].
  3. SINGLE NUMBERS  two real policies whose 20-episode point estimates differ
     LIE             but whose CIs OVERLAP — the ranking is not significant at
                     N=20; grow N and the gap becomes real (or doesn't).
  4. HELD-OUT        a LIBERO-style eval on start states the demos never
     VARIANTS        covered: train-distribution success != held-out success.

The two policies under test are tiny behavior-cloning MLPs (the ch1.1 pattern),
one trained on many demos and one on few — the ch1.2 "data is the policy"
result, now with the honest error bars ch1.2 could not yet draw. The policies
are cheap on purpose: the STATISTICS are the subject, not the robot.

Run it:      python curriculum/phase1_imitation/ch1.6_harness/harness.py --seed 0 --device cpu
Break it:    python curriculum/phase1_imitation/ch1.6_harness/harness.py --seed 0 --device cpu --break too_few
CI smoke:    python curriculum/phase1_imitation/ch1.6_harness/harness.py --smoke --seed 0 --device cpu --no-rerun

The reference figures are seed 0 on CPU; on Apple Silicon pass --device cpu to
reproduce them bitwise (mps is fast but not bit-reproducible). Everything here is
torch + numpy + mujoco — no scipy hiding the CI math.
"""

# --- region: setup ---
import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch1.1's bc.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv, ScriptedExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

OBS_DIM, ACT_DIM = PushTEnv.OBS_DIM, PushTEnv.ACT_DIM
EVAL_BASE_SEED = 10_000  # eval start seeds live here; demos use [--seed, --seed+num_demos)
BREAK_EPISODES = 5       # --break too_few shrinks each suite to this — a deliberately thin eval

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch1.6-harness"))
parser.add_argument("--seed", type=int, default=0, help="seeds demos, both trainings, and the bootstrap")
parser.add_argument("--num_demos", type=int, default=120)       # strong policy's demos; T4/cpu: 120 | smoke: 6
parser.add_argument("--num_demos_weak", type=int, default=40)   # weak policy trains on the first this-many demos
parser.add_argument("--hidden_dim", type=int, default=192)
parser.add_argument("--epochs", type=int, default=300)          # cpu: ~0.6 min whole run | smoke: 3
parser.add_argument("--eval_episodes", type=int, default=20,    # per suite; 20 = the noisy number the arc reported
                    help="episodes PER suite; the small-N estimate uses one suite of this size")
parser.add_argument("--n_seeds", type=int, default=10,          # independent suites; pooled N = n_seeds * eval_episodes
                    help="independent eval suites; their spread is the swing, their pool is the large-N estimate")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--break", dest="break_", choices=("too_few",), default=None,
                    help="too_few: shrink every suite to 5 episodes and watch the CI go useless / the ranking flip")
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # returns the numpy Generator the bootstrap resamples from
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.num_demos, args.num_demos_weak = 6, 3
    args.epochs, args.eval_episodes, args.n_seeds, args.device = 3, 4, 2, "cpu"
if args.break_ == "too_few":
    args.eval_episodes = BREAK_EPISODES  # the misconception: "5 episodes is a number"
banner("ch1.6-harness", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch1.6-harness", spawn=False)
    rr.save(str(args.out / "harness.rrd"))
# --- endregion ---

# --- region: stats ---
# The core of the chapter. A success rate k/n is a binomial proportion, and
# these three functions turn it into an honest interval. numpy + math only —
# no scipy, because the point is to SEE the formula, not to trust a black box.
Z95 = 1.959963985  # the 0.975 standard-normal quantile (95% two-sided); hardcoded so we import no stats table


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n Bernoulli trials.

    The naive textbook interval is Wald: p_hat +- z*sqrt(p_hat(1-p_hat)/n). It
    is taught first and wrong at the edges — at k=0 or k=n it collapses to zero
    width (claiming certainty from a handful of trials), and it can poke outside
    [0, 1]. The Wilson interval solves p (an implicit quadratic) instead of
    reading it off, so it is always inside [0, 1] and never degenerate. It is
    the interval a success rate should ship with.
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1.0 + z * z / n
    center = (p_hat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p_hat * (1.0 - p_hat) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def bootstrap_ci(outcomes: np.ndarray, boot_rng: np.random.Generator,
                 reps: int = 2000, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI for the success rate of a 0/1 outcome array.

    Resample the n episode outcomes WITH replacement `reps` times, take each
    resample's mean, and read the 2.5th / 97.5th percentiles of those means.
    No formula, no normal approximation — just "what would this eval have said
    on a slightly different draw of the same n episodes?". It should land on top
    of the Wilson interval at these n; disagreement means one of them is wrong.
    """
    n = len(outcomes)
    if n == 0:
        return (0.0, 1.0)
    resamples = outcomes[boot_rng.integers(0, n, size=(reps, n))].mean(axis=1)
    lo, hi = np.percentile(resamples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (float(lo), float(hi))


def diff_ci(k_a: int, n_a: int, k_b: int, n_b: int, z: float = Z95) -> tuple[float, float]:
    """Newcombe hybrid-score interval for the difference p_a - p_b.

    Built from the two Wilson intervals (Newcombe 1998, method 10). This is the
    verdict on "is A really better than B": if the interval EXCLUDES 0 the
    ranking is significant at this n; if it CONTAINS 0 you have not established
    the ranking, no matter how the point estimates look. Comparing whether the
    two one-sample CIs overlap is a cruder, more conservative eyeball test — the
    difference interval is the one to report.
    """
    p_a, p_b = k_a / n_a, k_b / n_b
    lo_a, hi_a = wilson_ci(k_a, n_a, z)
    lo_b, hi_b = wilson_ci(k_b, n_b, z)
    d = p_a - p_b
    lo = d - math.sqrt((p_a - lo_a) ** 2 + (hi_b - p_b) ** 2)
    hi = d + math.sqrt((hi_a - p_a) ** 2 + (p_b - lo_b) ** 2)
    return (lo, hi)


# Trust-but-verify: the Wilson CI for 0 successes in 10 trials is the textbook
# [0, 0.278] (Brown, Cai & DasGupta 2001, "Interval Estimation for a Binomial
# Proportion"). A wrong CI taught as canonical is the worst defect this file
# could ship, so we assert the known value every run.
_lo, _hi = wilson_ci(0, 10)
assert _lo == 0.0 and abs(_hi - 0.2775) < 1e-3, f"Wilson CI self-check failed: got [{_lo}, {_hi}]"
# --- endregion ---

# --- region: policy ---
# A tiny behavior-cloning MLP — the ch1.1 policy, trimmed. Normalization lives
# inside the module as buffers (ch1.1 pattern), so `policy(obs_raw)` just works.
class BCPolicy(nn.Module):
    def __init__(self, hidden_dim: int, stats: dict[str, np.ndarray]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, ACT_DIM),
        )
        for name, value in stats.items():
            self.register_buffer(name, torch.from_numpy(value))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (2.0 * (obs - self.obs_min) / self.obs_range - 1.0).clamp(-1.0, 1.0)
        return (self.net(normalized) + 1.0) / 2.0 * self.act_range + self.act_min


def norm_stats(obs: np.ndarray, actions: np.ndarray) -> dict[str, np.ndarray]:
    obs_min, act_min = obs.min(0), actions.min(0)
    # constant dims (the fixed target block) carry range 0; give them range 1 so
    # they map to a constant instead of dividing by zero (ch1.1's guard).
    obs_range = np.where(obs.max(0) - obs_min < 1e-4, np.float32(1.0), obs.max(0) - obs_min)
    act_range = np.where(actions.max(0) - act_min < 1e-4, np.float32(1.0), actions.max(0) - act_min)
    return {"obs_min": obs_min, "obs_range": obs_range, "act_min": act_min, "act_range": act_range}


def train_bc(obs: np.ndarray, actions: np.ndarray, hidden_dim: int, epochs: int, seed: int) -> BCPolicy:
    """Plain MSE behavior cloning. Seeded per call so training order can't leak
    between the strong and weak policies (CPU torch is bitwise-deterministic)."""
    torch.manual_seed(seed)
    policy = BCPolicy(hidden_dim, norm_stats(obs, actions)).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(actions).to(device)
    shuffle = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for batch in torch.randperm(len(obs_t), generator=shuffle).split(256):
            loss = F.mse_loss(policy(obs_t[batch]), act_t[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return policy.eval()


def bc_action_fn(policy: BCPolicy):
    def act(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            return policy(torch.from_numpy(obs).to(device).unsqueeze(0))[0].cpu().numpy()
    return act
# --- endregion ---

# --- region: eval ---
def collect_demos(num_demos: int, seed: int):
    """Roll out the scripted expert for num_demos episodes; return per-frame
    (obs, action, episode_id). Episode i uses seed+i (bit-reproducible)."""
    env = PushTEnv()
    obs_all, act_all, ep_all = [], [], []
    for i in range(num_demos):
        obs = env.reset(seed + i)
        expert = ScriptedExpert(noise=0.0, seed=seed + i)
        done = False
        while not done:
            action = expert.action(env)
            obs_all.append(obs)
            act_all.append(action)
            ep_all.append(i)
            obs, _, done, _ = env.step(action)
    return (np.asarray(obs_all, np.float32), np.asarray(act_all, np.float32),
            np.asarray(ep_all, np.int64))


def reset_held_out(env: PushTEnv, seed: int) -> np.ndarray:
    """A LIBERO-style held-out reset: the block starts FARTHER from the target
    (annulus 0.24-0.30 m) than any training demo, whose blocks spawn in
    0.10-0.24 m. Same task, a start distribution the policy never saw. We reuse
    env.reset for its MuJoCo bookkeeping, then overwrite the block+pusher pose
    (env exposes no OOD reset, so we set qpos directly and re-forward)."""
    env.reset(seed)
    ood = np.random.Generator(np.random.PCG64(seed))
    r, phi = ood.uniform(0.24, 0.30), ood.uniform(0.0, 2.0 * np.pi)
    tee_xy = np.array([r * np.cos(phi), r * np.sin(phi)])
    while True:  # keep the pusher clear of the moved block (env.reset's rule)
        pusher_xy = ood.uniform(-0.32, 0.32, size=2)
        if np.linalg.norm(pusher_xy - tee_xy) > 0.13:
            break
    q = env.data.qpos
    adr = {n: env.model.joint(n).qposadr[0] for n in ("tee_x", "tee_y", "tee_yaw", "pusher_x", "pusher_y")}
    q[adr["tee_x"]], q[adr["tee_y"]] = tee_xy
    q[adr["tee_yaw"]] = ood.uniform(-np.pi, np.pi)
    q[adr["pusher_x"]], q[adr["pusher_y"]] = pusher_xy
    mujoco.mj_forward(env.model, env.data)
    return env._obs()


def eval_suite(act_fn, reset_fn, n_seeds: int, eval_episodes: int) -> np.ndarray:
    """Run n_seeds independent suites of eval_episodes rollouts each. Returns a
    (n_seeds, eval_episodes) bool array of successes. Suite s, episode e uses a
    distinct held-out start seed, so no rollout repeats and none was trained on."""
    env = PushTEnv()
    outcomes = np.zeros((n_seeds, eval_episodes), dtype=bool)
    for s in range(n_seeds):
        for e in range(eval_episodes):
            obs = reset_fn(env, EVAL_BASE_SEED + s * eval_episodes + e)
            done, info = False, {}
            while not done:
                obs, _, done, info = env.step(act_fn(obs))
            outcomes[s, e] = bool(info["success"])
    return outcomes
# --- endregion ---

# --- region: report ---
# Train two real policies: the ch1.2 experiment (more data vs less), now graded
# with error bars. The whole chapter is which conclusions survive them.
obs, actions, episode_ids = collect_demos(args.num_demos, args.seed)
strong = train_bc(obs, actions, args.hidden_dim, args.epochs, args.seed)
weak_mask = episode_ids < args.num_demos_weak
weak = train_bc(obs[weak_mask], actions[weak_mask], args.hidden_dim, args.epochs, args.seed)
print(f"trained strong ({args.num_demos} demos) and weak ({args.num_demos_weak} demos) BC policies")

# Evaluate both on the same held-out start seeds; evaluate the strong one again
# on the OOD held-out variant. Booleans in, statistics out.
strong_out = eval_suite(bc_action_fn(strong), lambda env, s: env.reset(s), args.n_seeds, args.eval_episodes)
weak_out = eval_suite(bc_action_fn(weak), lambda env, s: env.reset(s), args.n_seeds, args.eval_episodes)
heldout_out = eval_suite(bc_action_fn(strong), reset_held_out, args.n_seeds, args.eval_episodes)

# (1) THE SWING. A single 20-episode suite is one row of this matrix; look at
# how far the rows disagree before trusting any one of them as "the" number.
suite_rates = strong_out.mean(axis=1)
n_small, n_pooled = args.eval_episodes, args.n_seeds * args.eval_episodes
print(f"\n[1] strong policy, {args.n_seeds} suites of {n_small}: per-suite success "
      f"ranged {suite_rates.min():.2f}..{suite_rates.max():.2f} "
      f"(std {suite_rates.std():.3f}) — that spread is what one number hides")

# (2) CONFIDENCE INTERVALS. The pooled strong estimate two independent ways: the
# analytic Wilson interval and a seeded percentile bootstrap of the same outcomes.
# They should land on top of each other — agreement is the proof the band is not an
# artifact of either derivation, not a coincidence.
ks_pool, n_pool = int(strong_out.sum()), n_pooled
boot = bootstrap_ci(strong_out.reshape(-1).astype(np.float64), rng)
wl, wh = wilson_ci(ks_pool, n_pool)
print(f"\n[2] strong pooled {ks_pool}/{n_pool} = {ks_pool / n_pool:.3f}: "
      f"Wilson [{wl:.3f}, {wh:.3f}]  bootstrap [{boot[0]:.3f}, {boot[1]:.3f}] "
      f"— two roads, one band")

# (3) SINGLE NUMBERS LIE. The small-N estimate is suite 0 (N=eval_episodes); the
# large-N estimate pools every suite (N=n_pooled). Same rollouts, nested.
ks_small, n_small = int(strong_out[0].sum()), args.eval_episodes
kw_small = int(weak_out[0].sum())
kw_pool = int(weak_out.sum())
small_diff = diff_ci(ks_small, n_small, kw_small, n_small)
pool_diff = diff_ci(ks_pool, n_pool, kw_pool, n_pool)
small_sig = small_diff[0] > 0 or small_diff[1] < 0
pool_sig = pool_diff[0] > 0 or pool_diff[1] < 0
print("\n[3] strong vs weak, single-numbers-lie:")
print(f"    N={n_small:<4d} strong {ks_small/n_small:.2f} {wilson_ci(ks_small,n_small)}  "
      f"weak {kw_small/n_small:.2f} {wilson_ci(kw_small,n_small)}")
print(f"           diff CI {small_diff[0]:+.2f}..{small_diff[1]:+.2f}  "
      f"-> {'ranking SIGNIFICANT' if small_sig else 'NOT significant (CIs overlap 0)'}")
print(f"    N={n_pool:<4d} strong {ks_pool/n_pool:.2f} {wilson_ci(ks_pool,n_pool)}  "
      f"weak {kw_pool/n_pool:.2f} {wilson_ci(kw_pool,n_pool)}")
print(f"           diff CI {pool_diff[0]:+.2f}..{pool_diff[1]:+.2f}  "
      f"-> {'ranking SIGNIFICANT' if pool_sig else 'NOT significant (CIs overlap 0)'}")

# (4) HELD-OUT. Same policy, a start region it never trained on. The gap is the
# generalization story — with its own CI, because it is a success rate too.
hs_pool = int(heldout_out.sum())
heldout_diff = diff_ci(ks_pool, n_pool, hs_pool, n_pool)
heldout_sig = heldout_diff[0] > 0 or heldout_diff[1] < 0
print(f"\n[4] strong policy, train-distribution {ks_pool/n_pool:.2f} vs held-out "
      f"{hs_pool/n_pool:.2f} (N={n_pool} each)")
print(f"    gap CI {heldout_diff[0]:+.2f}..{heldout_diff[1]:+.2f}  "
      f"-> {'generalization gap REAL' if heldout_sig else 'gap NOT significant at this N'}")

if args.break_ == "too_few":
    print(f"\n[break too_few] every suite is {BREAK_EPISODES} episodes. The strong-vs-weak "
          f"diff CI is {small_diff[1]-small_diff[0]:.2f} wide and "
          f"{'FLIPPED the ranking' if ks_small < kw_small else 'still spans 0'} — a number, "
          f"but not evidence.")

if args.rerun:
    for s, rate in enumerate(suite_rates):
        rr.set_time("suite", sequence=s)
        rr.log("eval/suite_success_rate", rr.Scalars([float(rate)]))
    for name, (lo, hi), rate in [("strong_pooled", wilson_ci(ks_pool, n_pool), ks_pool / n_pool),
                                 ("weak_pooled", wilson_ci(kw_pool, n_pool), kw_pool / n_pool),
                                 ("heldout_pooled", wilson_ci(hs_pool, n_pool), hs_pool / n_pool)]:
        rr.log(f"eval/{name}/rate", rr.Scalars([rate]))
        rr.log(f"eval/{name}/ci", rr.Scalars([lo, hi]))

metrics = {
    "break": args.break_ or "none",
    "eval_episodes": args.eval_episodes,
    "heldout_gap_significant": bool(heldout_sig),
    "heldout_pooled_ci_hi": round(wilson_ci(hs_pool, n_pool)[1], 6),
    "heldout_pooled_ci_lo": round(wilson_ci(hs_pool, n_pool)[0], 6),
    "heldout_pooled_rate": round(hs_pool / n_pool, 6),
    "n_pooled": n_pool,
    "n_seeds": args.n_seeds,
    "n_small": n_small,
    "num_demos": args.num_demos,
    "num_demos_weak": args.num_demos_weak,
    "pooled_diff_ci_hi": round(pool_diff[1], 6),
    "pooled_diff_ci_lo": round(pool_diff[0], 6),
    "pooled_significant": bool(pool_sig),
    "seed": args.seed,
    "small_diff_ci_hi": round(small_diff[1], 6),
    "small_diff_ci_lo": round(small_diff[0], 6),
    "small_significant": bool(small_sig),
    "smoke": bool(args.smoke),
    "strong_pooled_bootstrap_hi": round(boot[1], 6),
    "strong_pooled_bootstrap_lo": round(boot[0], 6),
    "strong_pooled_ci_hi": round(wilson_ci(ks_pool, n_pool)[1], 6),
    "strong_pooled_ci_lo": round(wilson_ci(ks_pool, n_pool)[0], 6),
    "strong_pooled_rate": round(ks_pool / n_pool, 6),
    "strong_small_rate": round(ks_small / n_small, 6),
    "suite_rate_max": round(float(suite_rates.max()), 6),
    "suite_rate_min": round(float(suite_rates.min()), 6),
    "suite_rate_std": round(float(suite_rates.std()), 6),
    "weak_pooled_rate": round(kw_pool / n_pool, 6),
    "weak_small_rate": round(kw_small / n_small, 6),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"\nmetrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'harness.rrd'} — open it with: rerun {args.out / 'harness.rrd'}")
# --- endregion ---
