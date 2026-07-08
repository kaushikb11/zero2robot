"""zero2robot 2.7 — Domain Randomization: Randomize to Generalize.

Ch2.6 measured the reality gap: a policy trained in ONE fixed world falls apart
when that world shifts (its masses change, its floor grips differently, gravity
is a touch off). You cannot fix the real world to match your sim. So do the
opposite — make the SIM stop being one fixed world. Randomize its dynamics EVERY
episode at train time, and the policy is forced to learn a behavior that survives
the whole RANGE instead of overfitting to one point. That is domain
randomization, the workhorse behind sim-to-real transfer (Tobin 2017, OpenAI's
in-hand cube).

This file trains TWO PPO policies on the quadruped, from scratch, changing ONE
thing between them:

  NARROW      trains on the nominal dynamics only (mass/friction/gravity fixed).
  RANDOMIZED  resamples mass, foot friction, and gravity each episode, uniformly
              within a band around nominal (--dr_width scales the band).

Then it does the honest part: EVALUATE BOTH across a sweep of SHIFTED test
dynamics — the "gap" — with ch1.6 error bars (mean return +- std over held-out
episodes). This is the promise DR makes, and the one you TEST rather than assume:
the narrow policy peaks at nominal and should degrade as the test dynamics move
away; the randomized policy gives up a little at nominal (the insurance premium)
to hold up across the gap. Whether it actually does at THIS free-tier budget is
the measurement — and the honest answer is nuanced. The off-nominal survival
edge sits INSIDE the seed band (-0.02, +0.22, -0.09 across seeds 0-2): DR helps on
one seed and not the others, so the mean benefit is not yet a demonstrated win. That
within-band result is the lesson (the ch1.6 "single numbers lie" trap in an RL
costume), and the eval sweep is the "break-the-policy" demo that surfaces it.

Determinism: we randomize ON TOP of the quadruped's pinned contact solver (we
only scale physical params — body mass/inertia, foot friction, gravity — never
the solver iterations/cone the env README pins), and the DR draws come from a
seeded generator, so a fixed --seed reproduces byte-for-byte on CPU.

Run it:      python curriculum/phase2_reinforcement/ch2.7_dr/dr.py --seed 0
Wider band:  python .../dr.py --seed 0 --dr_width 2.0   (randomize harder)
Sweep knob:  python .../dr.py --seed 0 --sweep_knob gravity
CI smoke:    python .../dr.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch2.1's ppo.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.quadruped.quadruped_env import QuadrupedEnv  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

OBS_DIM, ACT_DIM = QuadrupedEnv.OBS_DIM, QuadrupedEnv.ACT_DIM
EVAL_SEED0 = 500_000  # held-out eval seeds, disjoint from any training seed

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch2.7-dr"))
parser.add_argument("--total_steps", type=int, default=400_000)  # PER policy; cpu-laptop: minutes | smoke: 1024
parser.add_argument("--num_envs", type=int, default=16)      # parallel rollouts; T4: 16 | 4090: 64
parser.add_argument("--num_steps", type=int, default=256)    # steps per env per rollout -> batch = envs*steps
parser.add_argument("--update_epochs", type=int, default=5, help="passes over each rollout batch")
parser.add_argument("--num_minibatches", type=int, default=8)
parser.add_argument("--lr", type=float, default=3e-4, help="peak Adam lr; annealed to 0 over training")
parser.add_argument("--gamma", type=float, default=0.99, help="reward discount")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE bias/variance knob")
parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO trust region: clip the prob ratio to 1 +- this")
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--dr_width", type=float, default=1.0,
                    help="randomization band width for the RANDOMIZED policy (0 = no DR = narrow; 2 = double-wide)")
parser.add_argument("--sweep_knob", choices=("mass", "friction", "gravity"), default="mass",
                    help="which test-dynamics axis the generalization sweep moves across the gap")
parser.add_argument("--eval_episodes", type=int, default=16, help="held-out episodes per sweep point; T4: 16 | smoke: 3")
parser.add_argument("--seed", type=int, default=0, help="seeds torch, numpy, the DR draws, AND every env reset")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true", help="tiny CPU run for CI; two runs must match byte-for-byte")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; env resets + the DR RNG are seeded explicitly per policy below
# The test-dynamics grid the sweep walks across, per knob. Extends BEYOND the
# randomized band (max width ~1.4 scale) into the gap, where DR must extrapolate.
SWEEP_GRIDS = {"mass": [0.8, 1.0, 1.2, 1.4, 1.6], "friction": [0.3, 0.6, 1.0, 1.5, 2.0],
               "gravity": [0.7, 0.85, 1.0, 1.15, 1.3]}
if args.smoke:  # pin everything the CI byte-compare depends on
    args.total_steps, args.num_envs, args.num_steps = 1024, 4, 128
    args.update_epochs, args.eval_episodes, args.device = 1, 3, "cpu"
    SWEEP_GRIDS = {k: [0.75, 1.0, 1.5] for k in SWEEP_GRIDS}
banner("ch2.7-dr", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
batch_size = args.num_envs * args.num_steps
minibatch_size = batch_size // args.num_minibatches
num_iterations = args.total_steps // batch_size
sweep_grid = SWEEP_GRIDS[args.sweep_knob]
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch2.7-dr", spawn=False)
    rr.save(str(args.out / "dr.rrd"))
# --- endregion ---

# --- region: randomize ---
# Domain randomization = scaling the env's PHYSICAL parameters, in place, on top
# of the pinned contact solver. We touch only mass/inertia (heavier or lighter
# robot), foot friction (a slick vs grippy floor), and gravity (a heavier or
# lighter world) — never the solver iterations/cone the quadruped README pins, so
# the bitwise-CPU determinism guarantee still holds per fixed seed. Half-widths
# below are the nominal band at --dr_width 1.0; dr_width scales them (0 => narrow).
MASS_HW, FRIC_HW, GRAV_HW = 0.4, 0.5, 0.25   # mass +-40%, friction +-50%, gravity +-25%
SCALE_FLOOR = 0.2                            # keep every scale physical (no zero-mass / frictionless / zero-g)


def capture_nominal(env: QuadrupedEnv) -> dict:
    """Snapshot the env's as-built physical params so every episode scales from
    the SAME nominal baseline (scaling in place would otherwise compound)."""
    return {"mass": env.model.body_mass.copy(), "inertia": env.model.body_inertia.copy(),
            "friction": env.model.geom_friction.copy(), "gravity": float(env.model.opt.gravity[2])}


def sample_scales(rng: np.random.Generator, width: float) -> dict:
    """Draw one (mass, friction, gravity) scale triple for an episode, uniform in
    the band around 1.0. width 0 collapses the band to a point => all scales 1.0
    => the NARROW policy trains on nominal dynamics every episode."""
    def draw(half_width: float) -> float:
        return max(SCALE_FLOOR, float(rng.uniform(1.0 - width * half_width, 1.0 + width * half_width)))
    return {"mass": draw(MASS_HW), "friction": draw(FRIC_HW), "gravity": draw(GRAV_HW)}


def apply_scales(env: QuadrupedEnv, nominal: dict, scales: dict) -> None:
    """Write scaled params back into the compiled MjModel (body 0 is the world —
    leave its zero mass alone; friction column 0 is the tangential coefficient)."""
    env.model.body_mass[1:] = nominal["mass"][1:] * scales["mass"]
    env.model.body_inertia[1:] = nominal["inertia"][1:] * scales["mass"]
    env.model.geom_friction[:, 0] = nominal["friction"][:, 0] * scales["friction"]
    env.model.opt.gravity[2] = nominal["gravity"] * scales["gravity"]
# --- endregion ---

# --- region: model ---
def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0)) -> nn.Linear:
    """Orthogonal init with a tuned gain — the quiet PPO trick (see ch2.1). The
    tiny final-layer policy gain starts the robot near 'do nothing = stand'."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Agent(nn.Module):
    """Separate value and policy MLPs + a state-independent log-std, exactly the
    ch2.1 continuous-control agent — the algorithm is unchanged; only the WORLD it
    trains in differs between the two policies. The policy outputs the Gaussian
    MEAN over the 8 joint residuals; exploration is the Normal's spread."""

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
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),  # tiny gain: start almost still
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).flatten()

    def get_action_and_value(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """(action, log_prob, entropy, value). Pass `action` to SCORE it (update);
        omit it to SAMPLE a fresh one (rollout collection)."""
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(obs).flatten()
# --- endregion ---

# --- region: ppo ---
def train_policy(width: float, label: str) -> Agent:
    """Train ONE PPO policy end to end and return it. Called twice — once with
    width 0 (NARROW) and once with args.dr_width (RANDOMIZED). Re-seeds torch and
    the env/DR streams from args.seed at entry, so the two policies start from the
    IDENTICAL init and see the same reset noise; the ONLY difference between them
    is whether the dynamics get randomized. That is the controlled experiment."""
    torch.manual_seed(args.seed)
    envs = [QuadrupedEnv() for _ in range(args.num_envs)]
    nominal = [capture_nominal(e) for e in envs]
    dr_rng = np.random.Generator(np.random.PCG64(args.seed + 1234))  # the DR draw stream, seeded
    episode_count = np.zeros(args.num_envs, dtype=np.int64)

    def env_reset(i: int) -> np.ndarray:
        # each env/episode gets its own reset seed, then fresh randomized dynamics
        obs = envs[i].reset(seed=args.seed + i * 1000 + int(episode_count[i]))
        apply_scales(envs[i], nominal[i], sample_scales(dr_rng, width))
        episode_count[i] += 1
        return obs

    agent = Agent(OBS_DIM, ACT_DIM, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    obs_buf = torch.zeros((args.num_steps, args.num_envs, OBS_DIM), device=device)
    act_buf = torch.zeros((args.num_steps, args.num_envs, ACT_DIM), device=device)
    logp_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rew_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    val_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    term_buf = torch.zeros((args.num_steps, args.num_envs), device=device)   # fell (terminal)
    done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)   # fell OR time-limit
    boot_buf = torch.zeros((args.num_steps, args.num_envs), device=device)   # V(real next state) at episode ends

    next_obs = np.stack([env_reset(i) for i in range(args.num_envs)])
    ep_return = np.zeros(args.num_envs, dtype=np.float64)
    recent: list[float] = []
    for iteration in range(1, num_iterations + 1):
        optimizer.param_groups[0]["lr"] = args.lr * (1.0 - (iteration - 1.0) / num_iterations)  # anneal
        for step in range(args.num_steps):
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            obs_buf[step] = obs_t
            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(obs_t)
            act_buf[step], logp_buf[step], val_buf[step] = action, logp, value
            action_np = action.cpu().numpy()
            next_obs = np.empty_like(next_obs)
            for i in range(args.num_envs):
                obs_i, reward, done, info = envs[i].step(action_np[i])
                rew_buf[step, i] = reward
                term_buf[step, i] = float(info["terminated"])  # a FALL is terminal; a time-limit is not
                done_buf[step, i] = float(done)
                ep_return[i] += reward
                if done:
                    with torch.no_grad():  # value of the state we're leaving — bootstrap on truncation, masked on a fall
                        boot_buf[step, i] = agent.get_value(
                            torch.as_tensor(obs_i, dtype=torch.float32, device=device).unsqueeze(0)).item()
                    recent.append(ep_return[i])
                    ep_return[i] = 0.0
                    obs_i = env_reset(i)  # autoreset with fresh randomized dynamics
                next_obs[i] = obs_i

        with torch.no_grad():
            next_value = agent.get_value(torch.as_tensor(next_obs, dtype=torch.float32, device=device))
        advantages = torch.zeros_like(rew_buf)
        last_gae = torch.zeros(args.num_envs, device=device)
        for t in reversed(range(args.num_steps)):  # GAE; 1-terminated masks the bootstrap so a FALL earns no future value
            next_v = next_value if t == args.num_steps - 1 else val_buf[t + 1]
            next_v = torch.where(done_buf[t].bool(), boot_buf[t], next_v)
            delta = rew_buf[t] + args.gamma * next_v * (1.0 - term_buf[t]) - val_buf[t]
            last_gae = delta + args.gamma * args.gae_lambda * (1.0 - done_buf[t]) * last_gae
            advantages[t] = last_gae
        returns = advantages + val_buf

        b_obs, b_act = obs_buf.reshape(-1, OBS_DIM), act_buf.reshape(-1, ACT_DIM)
        b_logp, b_adv = logp_buf.reshape(-1), advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        for _ in range(args.update_epochs):
            for start in range(0, batch_size, minibatch_size):
                mb = torch.randperm(batch_size, device=device)[start:start + minibatch_size]
                _, new_logp, entropy, new_val = agent.get_action_and_value(b_obs[mb], b_act[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                adv = (b_adv[mb] - b_adv[mb].mean()) / (b_adv[mb].std() + 1e-8)  # normalize advantages
                pg_loss = torch.max(-adv * ratio,
                                    -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
                v_loss = 0.5 * ((new_val.flatten() - b_ret[mb]) ** 2).mean()
                loss = pg_loss + args.vf_coef * v_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
        mean_return = float(np.mean(recent[-50:])) if recent else float("nan")
        if args.rerun:
            rr.set_time("global_step", sequence=iteration * batch_size)
            rr.log(f"train/{label}/episodic_return", rr.Scalars([mean_return]))
        if iteration % 10 == 0 or iteration == num_iterations:
            print(f"  [{label}] iter {iteration:3d}/{num_iterations}  mean_return {mean_return:6.1f}")
    return agent
# --- endregion ---

# --- region: eval ---
def rollout(agent: Agent, env: QuadrupedEnv, seed: int) -> tuple[float, bool]:
    """One deterministic (policy MEAN, no sampling) episode; return (total_reward,
    survived). survived = reached the time limit without falling — the honest
    binary signal a brittle policy misses (return folds in walking too, but staying
    up is what the reality gap threatens first)."""
    obs, done, total = env.reset(seed=seed), False, 0.0
    info = {"truncated": False}
    while not done:
        with torch.no_grad():
            action = agent.actor_mean(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
        obs, reward, done, info = env.step(action[0].cpu().numpy())
        total += reward
    return total, bool(info["truncated"])  # truncated == survived the full horizon (never fell)


def evaluate_at(agent: Agent, knob: str, scale: float) -> dict:
    """Return mean+-std return AND survival rate over held-out episodes at ONE
    fixed test-dynamics point. The std / rate are the ch1.6 honesty: a single point
    estimate hides the spread, and survival is the binomial the reality gap moves."""
    env = QuadrupedEnv()
    nominal = capture_nominal(env)
    scales = {"mass": 1.0, "friction": 1.0, "gravity": 1.0}
    scales[knob] = scale  # move ONE axis; hold the others nominal so the curve is attributable
    returns, survived = [], []
    for ep in range(args.eval_episodes):
        # Pin the test dynamics on the MODEL (mj_resetData in rollout resets the
        # DATA — qpos/qvel — but never these model params, so the scale persists).
        apply_scales(env, nominal, scales)
        ret, ok = rollout(agent, env, EVAL_SEED0 + args.seed + ep)
        returns.append(ret)
        survived.append(ok)
    return {"scale": scale, "mean": float(np.mean(returns)), "std": float(np.std(returns)),
            "survival": float(np.mean(survived))}


def sweep(agent: Agent) -> list[dict]:
    """Evaluate one policy across the whole test-dynamics grid (the 'gap')."""
    return [evaluate_at(agent, args.sweep_knob, s) for s in sweep_grid]
# --- endregion ---

# --- region: report ---
print(f"[ch2.7-dr] training NARROW (dr_width 0) then RANDOMIZED (dr_width {args.dr_width}) "
      f"on the quadruped; sweep knob = {args.sweep_knob}")
narrow = train_policy(0.0, "narrow")
randomized = train_policy(args.dr_width, "randomized")
torch.save(narrow.state_dict(), args.out / "dr_narrow.pt")            # for the site ONNX demo (export_dr_onnx.py)
torch.save(randomized.state_dict(), args.out / "dr_randomized.pt")    # for the site ONNX demo (export_dr_onnx.py)

narrow_curve, rand_curve = sweep(narrow), sweep(randomized)
nominal_idx = sweep_grid.index(1.0)  # the point both policies trained AROUND

print(f"\ngeneralization across the gap ({args.sweep_knob}-scale, over "
      f"{args.eval_episodes} held-out episodes; return +- std | survival rate):")
print(f"  {args.sweep_knob + '_scale':>12s}  " + "  ".join(f"{s:>15g}" for s in sweep_grid))
for name, curve in [("narrow", narrow_curve), ("randomized", rand_curve)]:
    print(f"  {name:>12s}  " + "  ".join(f"{p['mean']:5.0f}+-{p['std']:<3.0f}|{p['survival']:.2f}" for p in curve))
    if args.rerun:
        for p in curve:
            rr.set_time(f"{args.sweep_knob}_scale_milli", sequence=int(round(p["scale"] * 1000)))
            rr.log(f"gap/{name}/mean_return", rr.Scalars([p["mean"]]))
            rr.log(f"gap/{name}/survival", rr.Scalars([p["survival"]]))

# The lesson as numbers: compare on-nominal (both trained AROUND here) to the
# off-nominal test points (the gap). A robust policy holds its survival across the
# gap; a brittle one falls. Honest caveat for this env: on the quadruped STAND at
# free-tier the gap DR can reach is narrow — standing is a stable equilibrium a
# nominal policy already handles across the controllable range, and beyond it the
# +-12 Nm servos saturate and BOTH policies fall. So measure the edge; do not
# assume it. The reported edge sits inside the seed band (seeds 0-2), which is the
# finding, not a bug.
off = [i for i in range(len(sweep_grid)) if i != nominal_idx]
def summarize(curve: list[dict]) -> dict:
    return {"nominal_survival": curve[nominal_idx]["survival"],
            "offnominal_survival": float(np.mean([curve[i]["survival"] for i in off])),
            "offnominal_return": float(np.mean([curve[i]["mean"] for i in off]))}
n_sum, r_sum = summarize(narrow_curve), summarize(rand_curve)
print(f"\nat nominal ({args.sweep_knob} 1.0):     narrow survival {n_sum['nominal_survival']:.2f}  "
      f"randomized survival {r_sum['nominal_survival']:.2f}")
print(f"across the gap (off-nominal mean):  narrow survival {n_sum['offnominal_survival']:.2f}  "
      f"randomized survival {r_sum['offnominal_survival']:.2f}  "
      f"-> DR survival edge {r_sum['offnominal_survival'] - n_sum['offnominal_survival']:+.2f}")

metrics = {
    "dr_width": args.dr_width,
    "sweep_knob": args.sweep_knob,
    "sweep_grid": sweep_grid,
    "narrow_curve": [{k: round(v, 4) for k, v in p.items()} for p in narrow_curve],
    "randomized_curve": [{k: round(v, 4) for k, v in p.items()} for p in rand_curve],
    "narrow_nominal_survival": round(n_sum["nominal_survival"], 4),
    "randomized_nominal_survival": round(r_sum["nominal_survival"], 4),
    "narrow_offnominal_survival": round(n_sum["offnominal_survival"], 4),
    "randomized_offnominal_survival": round(r_sum["offnominal_survival"], 4),
    "narrow_offnominal_return": round(n_sum["offnominal_return"], 4),
    "randomized_offnominal_return": round(r_sum["offnominal_return"], 4),
    "dr_survival_edge": round(r_sum["offnominal_survival"] - n_sum["offnominal_survival"], 4),
    "eval_episodes": args.eval_episodes,
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "total_steps": args.total_steps,
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"\nmetrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'dr.rrd'} — open it with: rerun {args.out / 'dr.rrd'}")
# --- endregion ---
