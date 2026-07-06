"""zero2robot 2.6 — Sim-to-Real Intuition Lab I: Latency & Noise.

Every policy you have trained so far — PPO on cartpole (ch2.1), SAC on
pusher-reach (ch2.2) — grew up in a PERFECT world. The observation was exact,
it arrived the instant you asked, and the simulator's masses and gravity were
the true ones because they were the ONLY ones. A real robot lives in none of
that. Its sensors are noisy, its observations arrive a few control steps late
(cameras, USB, filtering, the network), and its physical parameters never quite
match the model you trained in. That mismatch is the "reality gap," and it is
why a policy that scores 500/500 in sim can fall over on hardware.

This file is the hardware-intuition lab WITHOUT hardware. It loads a policy that
was trained in the clean sim and re-evaluates it while injecting the three
perturbations that dominate the gap, one family at a time:

  (1) SENSOR NOISE   — gaussian noise added to the observation (--obs_noise).
  (2) LATENCY        — the policy acts on a STALE observation, delayed through a
                       ring buffer by --latency_steps control steps.
  (3) MODEL MISMATCH — the eval env's mass / damping / gravity are scaled away
                       from the values training saw (--mass_scale, etc.).

We SWEEP each family from clean to broken and MEASURE the degradation curve —
success rate falling as the perturbation grows. The lesson is comparative and
honest: WHICH perturbation this policy is most brittle to, and that a policy
trained clean has no defense it was never asked to learn. That is the exact
motivation for ch2.7 (domain randomization): train ACROSS the gap so the policy
meets noise and mismatch before the robot does.

No policy binary ships in this repo (.pt is gitignored). Point --ckpt at a
policy you trained with ch2.1 / ch2.2; with no checkpoint (and always under
--smoke, for hermetic CI) the artifact falls back to the chapter's scripted
baseline controller — which was ALSO tuned for the clean sim, and degrades the
same way, so the lesson still lands.

Run it:      python .../perturb.py --seed 0 --task cartpole --ckpt outputs/ch2.1-ppo/ppo_agent.pt
Pusher:      python .../perturb.py --seed 0 --task pusher_reach --ckpt outputs/ch2.2-sac/sac_actor.pt
Scripted:    python .../perturb.py --seed 0 --task cartpole            (no ckpt -> scripted baseline)
Break it:    python .../perturb.py --seed 0 --break                   (extreme latency -> total failure)
CI smoke:    python .../perturb.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch2.1 / ch2.2).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.cartpole import CartpoleEnv  # noqa: E402
from curriculum.common.envs.pusher_reach import PusherReachEnv  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

EVAL_SEED0 = 500_000  # held-out eval seeds, disjoint from any training seed (matches ch2.1/2.2)

# Per-task config: the env, the observation/action dims, the "success" test a
# perturbed episode must still pass, and the natural secondary metric. Kept as a
# flat table (no class hierarchy) so both tasks stay readable side by side.
TASKS = {
    "cartpole": {
        "env": CartpoleEnv, "obs_dim": 5, "act_dim": 1,
        "metric_name": "mean_return",              # steps balanced (== return); higher is better
        "mismatch": ("gravity_scale", [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]),  # cartpole's sharp knob
        "ckpt": "outputs/ch2.1-ppo/ppo_agent.pt",
    },
    "pusher_reach": {
        "env": PusherReachEnv, "obs_dim": 8, "act_dim": 2,
        "metric_name": "mean_final_dist",          # fingertip->target at episode end; lower is better
        "mismatch": ("mass_scale", [0.5, 1.0, 1.5, 2.0, 3.0]),  # heavier links, same learned torques
        "ckpt": "outputs/ch2.2-sac/sac_actor.pt",
    },
}
# Sweep grids, shared across tasks. Free-tier friendly; --smoke shrinks them.
NOISE_GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2]   # gaussian obs-noise std
LATENCY_GRID = [0, 1, 2, 4, 8]                    # control steps of staleness (20 ms each @ 50 Hz)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch2.6-perturb"))
parser.add_argument("--task", choices=tuple(TASKS), default="cartpole")
parser.add_argument("--ckpt", type=Path, default=None, help="trained policy .pt (ch2.1/2.2 output); omit -> scripted baseline")
parser.add_argument("--eval_episodes", type=int, default=20)  # T4: 20 | smoke: 4
# Single-point perturbation knobs (used under --no-sweep and --break; the sweep overrides them one family at a time)
parser.add_argument("--obs_noise", type=float, default=0.0, help="gaussian obs-noise std")
parser.add_argument("--latency_steps", type=int, default=0, help="delay the observation by this many control steps")
parser.add_argument("--mass_scale", type=float, default=1.0, help="scale every body mass+inertia at eval")
parser.add_argument("--damping_scale", type=float, default=1.0, help="scale joint damping at eval")
parser.add_argument("--gravity_scale", type=float, default=1.0, help="scale gravity at eval")
parser.add_argument("--seed", type=int, default=0, help="seeds torch, numpy, the noise RNG, AND every env reset")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--sweep", dest="sweep", action="store_true", default=True,
                    help="measure the full degradation curves (on by default; the chapter's headline run)")
parser.add_argument("--no-sweep", dest="sweep", action="store_false", help="eval a SINGLE config from the knob flags")
parser.add_argument("--break", dest="break_bug", action="store_true",
                    help="Break It: pin latency to an extreme (16 steps) and watch the policy fail outright")
parser.add_argument("--smoke", action="store_true", help="tiny CPU run for CI; two runs must match byte-for-byte")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; env resets + the noise RNG are seeded explicitly below
if args.smoke:  # pin everything the CI byte-compare depends on
    args.eval_episodes, args.device, args.task = 4, "cpu", "cartpole"
    NOISE_GRID, LATENCY_GRID = [0.0, 0.05, 0.2], [0, 2, 8]
    TASKS["cartpole"]["mismatch"] = ("gravity_scale", [1.0, 1.5, 2.0])
banner("ch2.6-perturb", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
torch.set_grad_enabled(False)  # eval only: nothing here ever backprops
# The noise RNG is its OWN seeded stream, independent of env resets, so obs-noise
# is reproducible without perturbing the (separately seeded) env dynamics.
noise_rng = np.random.Generator(np.random.PCG64(args.seed + 777))
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch2.6-perturb", spawn=False)
    rr.save(str(args.out / "perturb.rrd"))
TASK = TASKS[args.task]
# --- endregion ---

# --- region: policy ---
# The policy is ALWAYS a plain obs->action function. A perturbation acts on the
# observation the policy consumes (or on the env's physics) — so a learned net
# and the scripted baseline are perturbed identically, through the exact same
# obs the real robot's autonomy stack would see. That is why the scripted
# fallback is a faithful stand-in: it, too, only ever reads observations.


def load_learned(task: str, ckpt: Path):
    """Rebuild the eval (deterministic) forward pass of a ch2.1/2.2 policy and
    load its weights. We infer hidden width from the checkpoint tensors, so the
    same loader reads PPO's width-64 actor and SAC's width-256 actor unchanged.
    Only the action-producing path is reconstructed — the critic, log-std, and
    sampling branches are training machinery and eval never touches them."""
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    if task == "cartpole":  # PPO: action = actor_mean(obs), a 3-layer Tanh MLP
        hidden = sd["actor_mean.0.weight"].shape[0]
        net = nn.Sequential(
            nn.Linear(TASK["obs_dim"], hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, TASK["act_dim"]),
        ).to(device)
        net.load_state_dict({k.removeprefix("actor_mean."): v for k, v in sd.items()
                             if k.startswith("actor_mean.")})

        def act(obs: np.ndarray) -> np.ndarray:
            o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            return net(o)[0].cpu().numpy()
        return act
    # SAC: deterministic action = tanh(mean(trunk(obs))), a squashed-Gaussian mean
    hidden = sd["trunk.0.weight"].shape[0]  # Linear(obs, hidden) weight is (hidden, obs)
    trunk = nn.Sequential(
        nn.Linear(TASK["obs_dim"], hidden), nn.ReLU(),
        nn.Linear(hidden, hidden), nn.ReLU(),
    ).to(device)
    mean = nn.Linear(hidden, TASK["act_dim"]).to(device)
    trunk.load_state_dict({k.removeprefix("trunk."): v for k, v in sd.items() if k.startswith("trunk.")})
    mean.load_state_dict({k.removeprefix("mean."): v for k, v in sd.items() if k.startswith("mean.")})

    def act(obs: np.ndarray) -> np.ndarray:
        o = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        return torch.tanh(mean(trunk(o)))[0].cpu().numpy()
    return act


def scripted_cartpole(obs: np.ndarray) -> np.ndarray:
    """The ch2.1 linear balancer, rebuilt to read the OBSERVATION (so latency and
    noise reach it). obs = [cart_pos, cart_vel, cos(theta), sin(theta), angvel];
    same gains as common/envs/cartpole.balance_action."""
    cart_pos, cart_vel, cos_t, sin_t, angvel = obs
    theta = math.atan2(sin_t, cos_t)
    u = 10.0 * theta + 2.0 * angvel + 0.4 * cart_pos + 0.8 * cart_vel
    return np.array([np.clip(u, -1.0, 1.0)], dtype=np.float32)


def scripted_pusher(obs: np.ndarray) -> np.ndarray:
    """The ch2.2 analytic-IK + PD reacher, rebuilt to read the OBSERVATION. We
    recover the joint angles and (via forward kinematics) the target world
    position from the obs, then run the same closed-form IK + PD as
    common/envs/pusher_reach.reach_action."""
    cos_sh, sin_sh, cos_el, sin_el, sh_vel, el_vel, dx, dy = obs
    sh, el, L = math.atan2(sin_sh, cos_sh), math.atan2(sin_el, cos_el), PusherReachEnv.LINK_LEN
    ftip_x = L * cos_sh + L * math.cos(sh + el)  # forward kinematics of the fingertip
    ftip_y = L * sin_sh + L * math.sin(sh + el)
    tx, ty = ftip_x + dx, ftip_y + dy            # target world position = fingertip + (dx, dy)
    cos_e = np.clip((tx * tx + ty * ty - 2.0 * L**2) / (2.0 * L**2), -1.0, 1.0)
    el_des = math.acos(cos_e)                    # elbow-down IK branch
    sh_des = math.atan2(ty, tx) - math.atan2(L * math.sin(el_des), L * (1.0 + cos_e))
    err = np.array([_wrap(sh_des - sh), _wrap(el_des - el)])
    return np.clip(25.0 * err - 2.0 * np.array([sh_vel, el_vel]), -1.0, 1.0).astype(np.float32)


def _wrap(a: float) -> float:
    """Wrap an angle error to [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def build_policy(task: str, ckpt: Path | None) -> tuple:
    """Return (policy_fn, label). A real checkpoint -> the trained policy; no
    checkpoint (or --smoke) -> the scripted baseline (hermetic, needs no .pt)."""
    if ckpt is not None and ckpt.is_file():
        return load_learned(task, ckpt), f"learned({ckpt.name})"
    scripted = scripted_cartpole if task == "cartpole" else scripted_pusher
    return scripted, "scripted-baseline"
# --- endregion ---

# --- region: perturb ---
# Three injectable perturbations. Sensor noise and latency wrap the OBSERVATION
# stream (what the policy reads); model mismatch scales the ENV's physics (what
# the policy acts on). Nothing here touches the policy weights — the reality gap
# is entirely outside the network.


class ObsDelay:
    """A ring buffer that hands the policy an observation from `steps` control
    steps ago — the whole latency model in one data structure. steps=0 is a
    pass-through. Primed with the first observation so the first `steps` actions
    see the start state rather than an empty buffer (a real robot boots the same
    way: the estimator has no history yet)."""

    def __init__(self, steps: int, first_obs: np.ndarray):
        self.buf: deque = deque([first_obs.copy()] * (steps + 1), maxlen=steps + 1)

    def push_pop(self, obs: np.ndarray) -> np.ndarray:
        self.buf.append(obs.copy())
        return self.buf[0]  # oldest in the window == delayed by `steps`


def apply_mismatch(env, mass_scale: float, damping_scale: float, gravity_scale: float) -> None:
    """Scale the eval env's physical parameters away from what training saw — an
    in-place edit of the compiled MjModel. mj_step reads these every step, so the
    dynamics change immediately; the policy is unaware its world moved. (On the
    planar pusher, gravity is perpendicular to the arm's motion, so gravity_scale
    is deliberately a NO-OP there — an honest reminder that which knob bites
    depends on the task's geometry.)"""
    env.model.body_mass[:] *= mass_scale
    env.model.body_inertia[:] *= mass_scale       # keep mass and inertia consistent
    env.model.dof_damping[:] *= damping_scale
    env.model.opt.gravity[:] *= gravity_scale


def perceive(raw_obs: np.ndarray, delay: ObsDelay, obs_noise: float) -> np.ndarray:
    """The full sensor pipeline for one step: delay first (the reading is stale),
    then add gaussian noise (the reading is also imprecise)."""
    obs = delay.push_pop(raw_obs)
    if obs_noise > 0.0:
        obs = obs + noise_rng.normal(0.0, obs_noise, size=obs.shape).astype(np.float32)
    return obs
# --- endregion ---

# --- region: eval ---
def evaluate(policy, episodes: int, obs_noise: float = 0.0, latency_steps: int = 0,
             mass_scale: float = 1.0, damping_scale: float = 1.0, gravity_scale: float = 1.0) -> dict:
    """Roll out `episodes` held-out episodes under one perturbation config and
    return {success_rate, metric}. success_rate is the task-agnostic degradation
    axis (fraction of episodes that still succeed); metric is the task's natural
    secondary number (cartpole return, pusher final distance)."""
    env = TASK["env"]()
    apply_mismatch(env, mass_scale, damping_scale, gravity_scale)
    successes, metrics = [], []
    for ep in range(episodes):
        raw = env.reset(seed=EVAL_SEED0 + args.seed + ep)
        delay = ObsDelay(latency_steps, raw)
        done, ret = False, 0.0
        info = {"truncated": False, "dist": float("nan")}
        while not done:
            action = policy(perceive(raw, delay, obs_noise))
            raw, reward, done, info = env.step(action)
            ret += reward
        if args.task == "cartpole":
            successes.append(float(info["truncated"]))  # survived to the horizon (never fell)
            metrics.append(ret)                          # return == steps balanced
        else:
            successes.append(float(info["dist"] < 0.05))  # within the ch2.2 solve bar at episode end
            metrics.append(info["dist"])
    return {"success_rate": float(np.mean(successes)), "metric": float(np.mean(metrics))}
# --- endregion ---

# --- region: sweep ---
def run_sweep(policy) -> dict:
    """Measure the clean baseline, then sweep each perturbation FAMILY one at a
    time (the others held clean), so every curve is attributable to a single
    cause. Returns the baseline + the three degradation curves."""
    baseline = evaluate(policy, args.eval_episodes)
    mismatch_knob, mismatch_grid = TASK["mismatch"]
    sweeps = {
        "sensor_noise": {"knob": "obs_noise", "points": [
            [m, *_score(evaluate(policy, args.eval_episodes, obs_noise=m))] for m in NOISE_GRID]},
        "latency": {"knob": "latency_steps", "points": [
            [s, *_score(evaluate(policy, args.eval_episodes, latency_steps=s))] for s in LATENCY_GRID]},
        "model_mismatch": {"knob": mismatch_knob, "points": [
            [g, *_score(evaluate(policy, args.eval_episodes, **{mismatch_knob: g}))] for g in mismatch_grid]},
    }
    return {"baseline": _named(baseline), "sweeps": sweeps}


def _score(result: dict) -> tuple[float, float]:
    return round(result["success_rate"], 6), round(result["metric"], 6)


def _named(result: dict) -> dict:
    return {"success_rate": round(result["success_rate"], 6),
            TASK["metric_name"]: round(result["metric"], 6)}


def worst_family(baseline: dict, sweeps: dict) -> tuple[str, float]:
    """Which perturbation hurts most: the family with the largest drop in success
    rate from the clean baseline to its most-perturbed grid point."""
    base = baseline["success_rate"]
    drops = {name: base - fam["points"][-1][1] for name, fam in sweeps.items()}
    worst = max(drops, key=drops.get)
    return worst, drops[worst]


def log_rerun(sweeps: dict) -> None:
    """Log each degradation curve to rerun: success rate vs perturbation
    magnitude, one timeline per family (the chapter's headline figure)."""
    for name, fam in sweeps.items():
        for magnitude, success_rate, metric in fam["points"]:
            rr.set_time(f"{name}_milli", sequence=int(round(magnitude * 1000)))
            rr.log(f"degradation/{name}/success_rate", rr.Scalars([success_rate]))
            rr.log(f"degradation/{name}/{TASK['metric_name']}", rr.Scalars([metric]))
# --- endregion ---

# --- region: report ---
# Resolve the checkpoint: explicit --ckpt wins; otherwise probe the task's
# default ch2.1/2.2 output path; under --smoke, never load a binary (hermetic CI).
ckpt = args.ckpt
if ckpt is None and not args.smoke:
    default_ckpt = Path(TASK["ckpt"])
    ckpt = default_ckpt if default_ckpt.is_file() else None
policy, policy_label = build_policy(args.task, ckpt)
print(f"[ch2.6-perturb] task={args.task}  policy={policy_label}  eval_episodes={args.eval_episodes}")

metrics = {"task": args.task, "policy": policy_label, "seed": args.seed,
           "smoke": bool(args.smoke), "eval_episodes": args.eval_episodes}

if args.break_bug:  # Break It: one extreme-latency eval, no sweep — the failure is the point
    result = evaluate(policy, args.eval_episodes, latency_steps=16)
    metrics.update({"mode": "break", "latency_steps": 16, "single": _named(result)})
    print(f"BREAK (latency 16 steps == 320 ms stale): success {result['success_rate']:.2f}  "
          f"{TASK['metric_name']} {result['metric']:.4f} — the clean-trained policy has no memory to fill the gap")
elif not args.sweep:  # single targeted eval from the knob flags (exercises / manual probing)
    result = evaluate(policy, args.eval_episodes, args.obs_noise, args.latency_steps,
                      args.mass_scale, args.damping_scale, args.gravity_scale)
    metrics.update({"mode": "single", "single": _named(result),
                    "config": {"obs_noise": args.obs_noise, "latency_steps": args.latency_steps,
                               "mass_scale": args.mass_scale, "damping_scale": args.damping_scale,
                               "gravity_scale": args.gravity_scale}})
    print(f"single: success {result['success_rate']:.2f}  {TASK['metric_name']} {result['metric']:.4f}")
else:  # the headline run: three degradation sweeps + which perturbation hurts most
    swept = run_sweep(policy)
    worst, drop = worst_family(swept["baseline"], swept["sweeps"])
    metrics.update({"mode": "sweep", **swept, "worst_perturbation": worst})
    base_succ = swept["baseline"]["success_rate"]
    print(f"clean baseline: success {base_succ:.2f}  {TASK['metric_name']} "
          f"{swept['baseline'][TASK['metric_name']]:.4f}")
    for name, fam in swept["sweeps"].items():
        curve = "  ".join(f"{knob:g}:{sr:.2f}" for knob, sr, _ in fam["points"])
        print(f"  {name:15s} ({fam['knob']:>14s})  success@ {curve}")
    print(f"worst perturbation for this policy: {worst}  (success drop {drop:+.2f} from clean baseline)")
    if args.rerun:
        log_rerun(swept["sweeps"])

(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'perturb.rrd'} — open it with: rerun {args.out / 'perturb.rrd'}")
# --- endregion ---
