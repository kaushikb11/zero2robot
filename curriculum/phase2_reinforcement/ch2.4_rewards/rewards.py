"""zero2robot 2.4 — Reward Design Is Programming.

The reward function is the program you write for the robot. You never tell a
quadruped "walk"; you write down a number for every state, and the policy finds
whatever behaviour maximizes that number. Get the number wrong and the policy
does *exactly what you said* — which is often nothing like what you meant. This
file makes that concrete on the `common/envs/quadruped` locomotion env, three
ways, training the SAME from-scratch PPO under different reward *programs* and
measuring what each one actually produces.

The env hands us its reward already split into five named, shapeable terms in
`info["reward_terms"]` (forward, upright, height, alive, ctrl) plus the raw
signals (`height`, `up_z`, `forward_vel`). So a "reward design" here is just a
python function `reward_fn(info, action, frac) -> float`: we IGNORE the env's
summed reward and score each step with our own program. Everything else — the
env, the PPO, the seeds — is held fixed, so the reward is the only variable.
That is the whole lesson in one line of architecture.

Three designs, three lessons:

  1. SHAPING. A SPARSE reward ("+1 only when you're already moving fast forward")
     gives the policy no gradient out of a standing crouch — it barely trains. The
     env's DENSE shaped reward (forward + upright + height + alive) guides the
     policy up from standing into a walk. We train both and compare forward
     distance: shaping is what makes the walk learnable at free-tier scale.

  2. REWARD HACKING. A NAIVE reward — "make the torso tall", `height` alone, no
     forward term — is optimized happily: the policy rears and straightens its
     legs, its reward CLIMBS, and it walks nowhere. We MEASURE the mismatch: the
     hacked design's own return rises while forward distance stays ~0. The policy
     did what you SAID (get tall), not what you MEANT (go forward). This is
     specification gaming — the boat-race hack, in miniature (see the prose).

  3. CURRICULUM VIA REWARD STAGES. Splitting the reward in time — stand first
     (upright + height + alive, no forward), then add the forward term once
     standing is solid — is the staged-reward idea. We train it and report the
     comparison honestly.

HONEST SCOPE. The point is the reward-vs-behaviour MISMATCH, not a strong gait.
At the free-tier CPU config the walks are wobbly and short; what reproduces
cleanly is the *ordering* (shaped beats sparse on forward distance; the hack's
reward rises while its forward distance does not). This env is a cartoon 2-DOF/leg
quadruped (see its README) — locomotion RL *structure*, not a transferable gait.

Run it:      python curriculum/phase2_reinforcement/ch2.4_rewards/rewards.py --seed 0 --device cpu
One design:  python .../rewards.py --seed 0 --design hack        (train just the hack)
CI smoke:    python .../rewards.py --smoke --seed 0 --no-rerun
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
# sys.path so `curriculum.common` resolves (same pattern as ch2.1 / tests/).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.quadruped import QuadrupedEnv  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch2.4-rewards"))
parser.add_argument("--design", choices=("all", "sparse", "shaped", "hack", "curriculum"),
                    default="all", help="train one reward design, or 'all' for the full comparison")
parser.add_argument("--total_steps", type=int, default=400_000)  # PER design; cpu-laptop: minutes | smoke: 1024
parser.add_argument("--num_envs", type=int, default=16)     # parallel rollouts; T4: 16 | 4090: 128
parser.add_argument("--num_steps", type=int, default=64)    # steps per env per rollout -> batch = envs*steps
parser.add_argument("--update_epochs", type=int, default=4)
parser.add_argument("--num_minibatches", type=int, default=4)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--gae_lambda", type=float, default=0.95)
parser.add_argument("--clip_coef", type=float, default=0.2, help="PPO trust region: clip the prob ratio to 1 +- this")
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--hidden_dim", type=int, default=64)
parser.add_argument("--eval_episodes", type=int, default=10)  # T4: 10 | smoke: 2
parser.add_argument("--seed", type=int, default=0, help="seeds torch, numpy, AND every env reset")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true", help="tiny CPU run for CI; two runs must match byte-for-byte")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; env resets are seeded explicitly below
if args.smoke:  # pin everything the CI byte-compare depends on
    args.total_steps, args.num_envs, args.num_steps = 1024, 4, 64
    args.update_epochs, args.eval_episodes, args.device = 1, 2, "cpu"
banner("ch2.4-rewards", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
batch_size = args.num_envs * args.num_steps
minibatch_size = batch_size // args.num_minibatches
num_iterations = args.total_steps // batch_size
rr = None
if args.rerun:
    import rerun as rr  # noqa: F811
    rr.init("zero2robot/ch2.4-rewards", spawn=False)
    rr.save(str(args.out / "rewards.rrd"))
# --- endregion ---

# --- region: rewards ---
# THE PAYLOAD. A reward design is a program over the env's exposed signals. Each
# takes (info, action, frac) — frac is training progress in [0,1], used only by
# the curriculum. We score every step with one of these INSTEAD of the env's own
# summed reward, so swapping the design is the only thing that changes.
SPARSE_VX = 0.4        # m/s; the sparse reward fires only above this forward speed
HACK_HEIGHT_W = 10.0   # weight on raw torso height for the reward-hacking design
CURRICULUM_SWITCH = 0.5  # fraction of training after which the forward term turns on


def r_sparse(info, action, frac):
    """SPARSE 'did it move forward': +1 only when already fast, else 0. No shaping
    gradient out of the crouch, so PPO has almost nothing to climb — the point."""
    return 1.0 if info["forward_vel"] >= SPARSE_VX else 0.0


def r_shaped(info, action, frac):
    """DENSE shaped reward: the env's five designed terms, summed. forward drives
    progress; upright/height/alive keep it standing; ctrl discourages thrashing.
    The graded signal that makes a walk emerge from a standing start."""
    return sum(info["reward_terms"].values())


def r_hack(info, action, frac):
    """NAIVE 'make the torso tall': raw height, no forward term. Optimized happily
    (rear up, straighten the legs) — reward CLIMBS while the robot walks nowhere.
    The specification-gaming demo: what you SAID (be tall) != what you MEANT (go)."""
    return HACK_HEIGHT_W * info["height"]


def r_curriculum(info, action, frac):
    """STAGED reward: stand first (upright+height+alive+ctrl, NO forward), then add
    the forward term once past CURRICULUM_SWITCH. Shape the reward in TIME."""
    t = info["reward_terms"]
    stand = t["upright"] + t["height"] + t["alive"] + t["ctrl"]
    return stand if frac < CURRICULUM_SWITCH else stand + t["forward"]


REWARD_DESIGNS = {"sparse": r_sparse, "shaped": r_shaped,
                  "hack": r_hack, "curriculum": r_curriculum}
# --- endregion ---

# --- region: envs ---
# A minimal synchronous vector env: a plain list of QuadrupedEnv stepped in a
# python loop (no gym wrapper hiding the autoreset). Each env gets its own
# deterministic seed stream so --seed is reproducible and no two envs share a
# start. env_step returns the info dict too, because the reward PROGRAM reads it.
def make_envs(n):
    return [QuadrupedEnv() for _ in range(n)]


def env_reset(envs, i, episode_count):
    obs = envs[i].reset(seed=args.seed + i * 1000 + int(episode_count[i]))
    episode_count[i] += 1
    return obs


def env_step(envs, i, action):
    """Step env i; return (obs, terminated, truncated, done, info)."""
    obs, _reward, done, info = envs[i].step(action)  # env's reward ignored; we score info
    return obs, bool(info["terminated"]), bool(info["truncated"]), done, info
# --- endregion ---

# --- region: model ---
def layer_init(layer, std=np.sqrt(2.0)):
    """Orthogonal init with a tuned gain — the quiet PPO trick that just works."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Agent(nn.Module):
    """Separate value and policy MLPs (no shared trunk) + a state-independent
    log-std. The policy outputs the MEAN of a Gaussian over the 8 joint offsets;
    exploration is the Normal's spread. Same shape as ch2.1/2.2 — the RL is held
    fixed across every reward design so the reward is the only variable."""

    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),  # tiny gain: start near the crouch
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, obs):
        return self.critic(obs).flatten()

    def get_action_and_value(self, obs, action=None):
        """Returns (action, log_prob, entropy, value). Pass `action` to SCORE a
        stored action (the update); omit it to SAMPLE a fresh one (rollout)."""
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(obs).flatten()
# --- endregion ---

# --- region: ppo ---
# One compact PPO update: rollout collection with the truncation-vs-termination
# bootstrap (the ch2.1 lesson, reused), GAE, and the clipped surrogate. Nothing
# here changes between reward designs — only the reward_fn passed to train() does.
OBS_DIM, ACT_DIM = QuadrupedEnv.OBS_DIM, QuadrupedEnv.ACT_DIM


def compute_gae(rewards, values, terminated, done, bootstrap, next_value):
    """GAE walking backward. (1 - terminated) masks the bootstrap after a fall; a
    truncated step keeps its stored bootstrap value (learned in ch2.1)."""
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros(args.num_envs, device=device)
    for t in reversed(range(args.num_steps)):
        next_v = next_value if t == args.num_steps - 1 else values[t + 1]
        next_v = torch.where(done[t].bool(), bootstrap[t], next_v)
        delta = rewards[t] + args.gamma * next_v * (1.0 - terminated[t]) - values[t]
        last_gae = delta + args.gamma * args.gae_lambda * (1.0 - done[t]) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values


def ppo_update(agent, optimizer, buf, advantages, returns):
    """`update_epochs` passes of minibatch SGD on the clipped surrogate."""
    b_obs = buf["obs"].reshape(-1, OBS_DIM)
    b_actions = buf["actions"].reshape(-1, ACT_DIM)
    b_logprobs, b_adv = buf["logprobs"].reshape(-1), advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    for _ in range(args.update_epochs):
        order = torch.randperm(batch_size, device=device)  # torch RNG is seeded -> reproducible
        for start in range(0, batch_size, minibatch_size):
            mb = order[start:start + minibatch_size]
            _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb], b_actions[mb])
            ratio = (new_logprob - b_logprobs[mb]).exp()  # pi_new / pi_old
            adv = b_adv[mb]
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)  # normalize advantages (a PPO trick)
            pg_loss = torch.max(-adv * ratio,
                                -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()
            v_loss = 0.5 * ((new_value.flatten() - b_returns[mb]) ** 2).mean()
            loss = pg_loss + args.vf_coef * v_loss - 0.0 * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()
# --- endregion ---

# --- region: train ---
def train(reward_fn):
    """Train PPO on the quadruped, scoring every step with `reward_fn`. Returns
    (agent, history) where history["design_return"] is the per-iteration mean
    episode return UNDER reward_fn — for the hack, watch this climb while the
    robot goes nowhere. This one function is reused for every design: the reward
    program is the only thing that varies (the chapter's whole thesis)."""
    envs = make_envs(args.num_envs)
    episode_count = np.zeros(args.num_envs, dtype=np.int64)
    agent = Agent(OBS_DIM, ACT_DIM, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    buf = {k: torch.zeros((args.num_steps, args.num_envs, d), device=device)
           for k, d in (("obs", OBS_DIM), ("actions", ACT_DIM))}
    for k in ("logprobs", "rewards", "values", "terminated", "done", "bootstrap"):
        buf[k] = torch.zeros((args.num_steps, args.num_envs), device=device)

    next_obs = np.stack([env_reset(envs, i, episode_count) for i in range(args.num_envs)])
    ep_return = np.zeros(args.num_envs, dtype=np.float64)
    recent, history = [], {"design_return": []}
    for iteration in range(1, num_iterations + 1):
        frac = (iteration - 1) / max(1, num_iterations)  # training progress for the curriculum
        for step in range(args.num_steps):
            obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
            buf["obs"][step] = obs_t
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(obs_t)
            buf["actions"][step], buf["logprobs"][step], buf["values"][step] = action, logprob, value
            action_np = action.cpu().numpy()
            next_obs = np.empty_like(next_obs)
            for i in range(args.num_envs):
                obs_i, terminated, truncated, done, info = env_step(envs, i, action_np[i])
                reward = reward_fn(info, action_np[i], frac)  # <- the reward PROGRAM
                buf["rewards"][step, i] = reward
                buf["terminated"][step, i], buf["done"][step, i] = float(terminated), float(done)
                ep_return[i] += reward
                if done:
                    with torch.no_grad():
                        boot = agent.get_value(torch.as_tensor(obs_i, dtype=torch.float32, device=device).unsqueeze(0))
                    buf["bootstrap"][step, i] = boot.item()
                    recent.append(ep_return[i])
                    ep_return[i] = 0.0
                    obs_i = env_reset(envs, i, episode_count)
                next_obs[i] = obs_i
        with torch.no_grad():
            next_value = agent.get_value(torch.as_tensor(next_obs, dtype=torch.float32, device=device))
        advantages, returns = compute_gae(buf["rewards"], buf["values"], buf["terminated"],
                                          buf["done"], buf["bootstrap"], next_value)
        ppo_update(agent, optimizer, buf, advantages, returns)
        history["design_return"].append(float(np.mean(recent[-50:])) if recent else float("nan"))
    return agent, history
# --- endregion ---

# --- region: eval ---
def evaluate(agent, reward_fn, design):
    """Roll out the policy MEAN (no sampling) on held-out seeds and measure what
    the reward ACTUALLY produced: forward distance (the intended behaviour),
    return under the design's own reward, plus height/up_z/length. Logs the
    per-term reward contributions and behaviour to rerun for the first episode."""
    env = QuadrupedEnv()
    forwards, design_returns, heights, up_zs, lengths = [], [], [], [], []
    for episode in range(args.eval_episodes):
        obs = env.reset(seed=500_000 + args.seed + episode)  # held out from training seeds
        x0 = float(env.data.qpos[env._root_qadr])
        done, dret, hs, us, n = False, 0.0, [], [], 0
        while not done:
            with torch.no_grad():
                mean = agent.actor_mean(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            action = mean[0].cpu().numpy()
            obs, _r, done, info = env.step(action)
            dret += reward_fn(info, action, 1.0)
            hs.append(info["height"])
            us.append(info["up_z"])
            n += 1
            if rr is not None and episode == 0:
                rr.set_time("eval_step", sequence=n)
                for name, val in info["reward_terms"].items():
                    rr.log(f"{design}/reward_terms/{name}", rr.Scalars([val]))
                rr.log(f"{design}/behavior/forward_vel", rr.Scalars([info["forward_vel"]]))
                rr.log(f"{design}/behavior/height", rr.Scalars([info["height"]]))
        forwards.append(float(env.data.qpos[env._root_qadr]) - x0)
        design_returns.append(dret)
        heights.append(float(np.mean(hs)))
        up_zs.append(float(np.mean(us)))
        lengths.append(n)
    return {"forward_m": float(np.mean(forwards)), "design_return": float(np.mean(design_returns)),
            "height_m": float(np.mean(heights)), "up_z_m": float(np.mean(up_zs)),
            "ep_len": float(np.mean(lengths))}
# --- endregion ---

# --- region: demos ---
def run_design(name):
    """Train + evaluate one reward design; returns its eval metrics + the training
    return curve endpoints (first vs last) — the hack's rising self-reward."""
    print(f"\n=== design '{name}': training PPO under r_{name} "
          f"({num_iterations} iters x {batch_size} steps) ===")
    agent, history = train(REWARD_DESIGNS[name])
    metrics = evaluate(agent, REWARD_DESIGNS[name], name)
    curve = [v for v in history["design_return"] if v == v]  # drop nan (no episode finished yet)
    metrics["train_return_first"] = round(curve[0], 4) if curve else float("nan")
    metrics["train_return_last"] = round(curve[-1], 4) if curve else float("nan")
    print(f"  forward_m {metrics['forward_m']:+.3f}  design_return {metrics['design_return']:8.2f}  "
          f"height_m {metrics['height_m']:.3f}  up_z {metrics['up_z_m']:.3f}  ep_len {metrics['ep_len']:.0f}")
    if rr is not None:
        rr.log(f"summary/{name}/forward_m", rr.Scalars([metrics["forward_m"]]))
        rr.log(f"summary/{name}/design_return", rr.Scalars([metrics["design_return"]]))
    return {k: round(v, 6) for k, v in metrics.items()}


names = list(REWARD_DESIGNS) if args.design == "all" else [args.design]
results = {name: run_design(name) for name in names}

if {"sparse", "shaped"} <= results.keys():
    print(f"\nSHAPING: shaped walks forward {results['shaped']['forward_m']:+.3f} m vs "
          f"sparse {results['sparse']['forward_m']:+.3f} m — the dense signal is what makes the walk learnable.")
if "hack" in results:
    h = results["hack"]
    print(f"REWARD HACK: the hack's own return rose {h['train_return_first']:.1f} -> {h['train_return_last']:.1f} "
          f"(it optimized 'be tall': height_m {h['height_m']:.3f}) while forward_m stayed {h['forward_m']:+.3f} "
          "— the policy did what you SAID, not what you MEANT.")

metrics = {"designs": results, "seed": args.seed, "smoke": bool(args.smoke),
           "total_steps_per_design": args.total_steps, "num_iterations": num_iterations}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"\nmetrics: {args.out / 'metrics.json'}")
if rr is not None:
    print(f"recording: {args.out / 'rewards.rrd'} — open it with: rerun {args.out / 'rewards.rrd'}")
# --- endregion ---
