"""zero2robot 2.1 — PPO: The Policy That Acts and Sees the Consequences.

Behavior cloning (ch1.1) died off-distribution: it never acts, so it never
sees the states its own mistakes create, and the covariate shift you measured
in the rerun trace is the whole problem. Reinforcement learning removes the
demonstrator. The policy acts, the environment answers with reward, and the
states it visits are the states IT causes — the exact distribution BC could
never see. This file is Proximal Policy Optimization from scratch on cartpole,
CleanRL-style: a Gaussian policy + value net, rollout collection, GAE
advantages, and a clipped surrogate objective, in one readable loop.

The one subtlety cartpole forces you to get right: termination vs truncation.
The pole FALLING is a terminal state (value 0 ahead). The 500-step budget
RUNNING OUT is a time limit — the pole is still up, and the value function must
BOOTSTRAP from where it would have continued. Conflate them and PPO learns that
balancing to the horizon is worthless. The env hands you both flags; we use them.

Run it:      python curriculum/phase2_reinforcement/ch2.1_ppo/ppo.py --seed 0
Ablate:      python .../ppo.py --seed 0 --no-norm-adv        (kill a PPO trick)
Break it:    python .../ppo.py --seed 0 --break              (drop the time-limit bootstrap)
CI smoke:    python .../ppo.py --smoke --seed 0 --no-rerun
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
# sys.path so `curriculum.common` resolves (same pattern as ch1.1 / tests/).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.cartpole import CartpoleEnv, balance_action  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch2.1-ppo"))
parser.add_argument("--total_steps", type=int, default=200_000)  # cpu-laptop: minutes | smoke: 512
parser.add_argument("--num_envs", type=int, default=8)      # parallel rollouts; T4: 8 | 4090: 64
parser.add_argument("--num_steps", type=int, default=256)   # steps per env per rollout -> batch = envs*steps
parser.add_argument("--update_epochs", type=int, default=10, help="passes over each rollout batch")
parser.add_argument("--num_minibatches", type=int, default=8)
parser.add_argument("--lr", type=float, default=3e-4, help="peak Adam lr; --anneal-lr decays it to 0")
parser.add_argument("--gamma", type=float, default=0.99, help="reward discount")
parser.add_argument("--gae_lambda", type=float, default=0.95, help="GAE bias/variance knob; 1.0 = plain Monte-Carlo")
parser.add_argument("--clip_coef", type=float, default=0.2, help="the PPO trust region: clip the prob ratio to 1 +- this")
parser.add_argument("--ent_coef", type=float, default=0.0, help="entropy bonus; raise it to keep exploring")
parser.add_argument("--vf_coef", type=float, default=0.5)
parser.add_argument("--max_grad_norm", type=float, default=0.5)
parser.add_argument("--hidden_dim", type=int, default=64)  # cartpole is tiny; width is not the bottleneck
parser.add_argument("--eval_episodes", type=int, default=20)  # T4: 20 | smoke: 3
parser.add_argument("--seed", type=int, default=0, help="seeds torch, numpy, AND every env reset")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
# --- the five PPO tricks, each a toggle so you can measure what it buys (ablation) ---
parser.add_argument("--norm-adv", dest="norm_adv", action="store_true", default=True,
                    help="normalize advantages per minibatch (on by default)")
parser.add_argument("--no-norm-adv", dest="norm_adv", action="store_false")
parser.add_argument("--clip-vloss", dest="clip_vloss", action="store_true", default=True,
                    help="clip the value loss like the policy (on by default)")
parser.add_argument("--no-clip-vloss", dest="clip_vloss", action="store_false")
parser.add_argument("--anneal-lr", dest="anneal_lr", action="store_true", default=True)
parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
parser.add_argument("--break", dest="break_bug", action="store_true",
                    help="Break It: treat a truncation like a fall — drop the time-limit bootstrap (conflate time running out with the pole falling)")
parser.add_argument("--smoke", action="store_true", help="tiny CPU run for CI; two runs must match byte-for-byte")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; env resets are seeded explicitly below
if args.smoke:  # pin everything the CI byte-compare depends on
    args.total_steps, args.num_envs, args.num_steps = 512, 4, 128
    args.update_epochs, args.eval_episodes, args.device = 1, 3, "cpu"
banner("ch2.1-ppo", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
batch_size = args.num_envs * args.num_steps
minibatch_size = batch_size // args.num_minibatches
num_iterations = args.total_steps // batch_size
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch2.1-ppo", spawn=False)
    rr.save(str(args.out / "ppo.rrd"))
# --- endregion ---

# --- region: envs ---
# A minimal synchronous vector env: a plain list of CartpoleEnv, stepped in a
# python loop. No gym, no SyncVectorEnv wrapper hiding the reset logic — the
# autoreset and the truncation bootstrap are the lesson, so they stay in view.
# Each env gets its own deterministic seed stream: env i's k-th episode resets
# with seed (seed + i*1000 + episode_count[i]), so --seed 0 is reproducible and
# no two envs ever share a start.
envs = [CartpoleEnv() for _ in range(args.num_envs)]
episode_count = np.zeros(args.num_envs, dtype=np.int64)


def env_reset(i: int) -> np.ndarray:
    obs = envs[i].reset(seed=args.seed + i * 1000 + int(episode_count[i]))
    episode_count[i] += 1
    return obs


def env_step(i: int, action: np.ndarray):
    """Step env i; return (obs, reward, terminated, truncated, done)."""
    obs, reward, done, info = envs[i].step(action)
    return obs, reward, bool(info["terminated"]), bool(info["truncated"]), done
# --- endregion ---

# --- region: model ---
def layer_init(layer: nn.Linear, std: float = np.sqrt(2.0)) -> nn.Linear:
    """Orthogonal init with a tuned gain — the quiet PPO trick that just works.
    Small final-layer gains (below) start the policy near-deterministic and the
    value head near-zero, which keeps early updates from exploding."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class Agent(nn.Module):
    """Separate value and policy MLPs (no shared trunk — one less thing to
    reason about), plus a state-INDEPENDENT log-std. The policy outputs the
    MEAN of a Gaussian over the force; exploration is the Normal's spread, and
    log_std is a learned parameter, not a network output — the standard
    continuous-control choice and the simplest thing that works on cartpole."""

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
        """Returns (action, log_prob, entropy, value). Pass `action` to SCORE a
        stored action under the current policy (the update); omit it to SAMPLE a
        fresh one (rollout collection)."""
        mean = self.actor_mean(obs)
        std = torch.exp(self.actor_logstd.expand_as(mean))
        dist = torch.distributions.Normal(mean, std)
        if action is None:
            action = dist.sample()
        # sum over action dims: independent Gaussians -> joint log-prob is the sum
        return action, dist.log_prob(action).sum(1), dist.entropy().sum(1), self.critic(obs).flatten()


agent = Agent(CartpoleEnv.OBS_DIM, CartpoleEnv.ACT_DIM, args.hidden_dim).to(device)
optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
# --- endregion ---

# --- region: rollout ---
# Rollout storage. Transition-indexed: everything at step t describes the ONE
# transition (obs[t], action[t]) -> reward[t], and the terminated/truncated/done
# flags for THAT transition. bootstrap_value[t] holds V(the real next state) for
# steps that ended an episode — the value we cut off, needed for truncation.
obs_buf = torch.zeros((args.num_steps, args.num_envs, CartpoleEnv.OBS_DIM), device=device)
actions_buf = torch.zeros((args.num_steps, args.num_envs, CartpoleEnv.ACT_DIM), device=device)
logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
terminated_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
done_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
bootstrap_buf = torch.zeros((args.num_steps, args.num_envs), device=device)


def collect_rollout(next_obs: np.ndarray, ep_return: np.ndarray, recent: list) -> np.ndarray:
    """Fill the buffers with one rollout; return the observation to start the
    next one from. `ep_return` accumulates per-env return; finished episodes are
    appended to `recent` (a sliding window the training loop averages for logging)."""
    for step in range(args.num_steps):
        obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        obs_buf[step] = obs_t
        with torch.no_grad():  # collection never backprops — just run the policy
            action, logprob, _, value = agent.get_action_and_value(obs_t)
        actions_buf[step], logprobs_buf[step], values_buf[step] = action, logprob, value
        action_np = action.cpu().numpy()

        next_obs = np.empty_like(next_obs)
        for i in range(args.num_envs):
            obs_i, reward, terminated, truncated, done = env_step(i, action_np[i])
            rewards_buf[step, i] = reward
            terminated_buf[step, i], done_buf[step, i] = float(terminated), float(done)
            ep_return[i] += reward
            if done:
                # Value of the state we're LEAVING behind. On truncation this is
                # a real state PPO must bootstrap from; on termination it is a
                # dead state whose value gets masked to 0 in the GAE below.
                with torch.no_grad():
                    boot = agent.get_value(torch.as_tensor(obs_i, dtype=torch.float32, device=device).unsqueeze(0))
                bootstrap_buf[step, i] = boot.item()
                recent.append(ep_return[i])
                ep_return[i] = 0.0
                obs_i = env_reset(i)  # autoreset: the next step sees a fresh episode
            next_obs[i] = obs_i
    return next_obs


def compute_advantages(next_obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over the collected rollout, walking
    backward. The one line that matters: `1 - terminated` masks the bootstrap so
    a FALLEN pole contributes no future value, while a TRUNCATED episode keeps
    its bootstrap_value. `--break` swaps in `done` and destroys that distinction."""
    with torch.no_grad():
        next_value = agent.get_value(torch.as_tensor(next_obs, dtype=torch.float32, device=device))
    advantages = torch.zeros_like(rewards_buf)
    last_gae = torch.zeros(args.num_envs, device=device)
    mask = done_buf if args.break_bug else terminated_buf  # Break It: masking on `done` treats a time-limit truncation like a fall — drops its bootstrap
    for t in reversed(range(args.num_steps)):
        next_v = next_value if t == args.num_steps - 1 else values_buf[t + 1]
        # steps that ended an episode bootstrap from the stored real-next-state value
        next_v = torch.where(done_buf[t].bool(), bootstrap_buf[t], next_v)
        delta = rewards_buf[t] + args.gamma * next_v * (1.0 - mask[t]) - values_buf[t]
        last_gae = delta + args.gamma * args.gae_lambda * (1.0 - done_buf[t]) * last_gae
        advantages[t] = last_gae
    return advantages, advantages + values_buf  # (advantages, returns)
# --- endregion ---

# --- region: update ---
def ppo_update(advantages: torch.Tensor, returns: torch.Tensor) -> dict:
    """The PPO update: flatten the rollout, then take `update_epochs` passes of
    minibatch SGD on the clipped surrogate. Everything the four flags toggle
    lives here."""
    b_obs = obs_buf.reshape(-1, CartpoleEnv.OBS_DIM)
    b_actions = actions_buf.reshape(-1, CartpoleEnv.ACT_DIM)
    b_logprobs, b_advantages = logprobs_buf.reshape(-1), advantages.reshape(-1)
    b_returns, b_values = returns.reshape(-1), values_buf.reshape(-1)
    stats = {}
    for _ in range(args.update_epochs):
        order = torch.randperm(batch_size, device=device)  # torch RNG is seeded -> reproducible
        for start in range(0, batch_size, minibatch_size):
            mb = order[start:start + minibatch_size]
            _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[mb], b_actions[mb])
            log_ratio = new_logprob - b_logprobs[mb]
            ratio = log_ratio.exp()  # pi_new / pi_old for these actions

            mb_adv = b_advantages[mb]
            if args.norm_adv:  # trick 1: standardize advantages within the minibatch
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            # clipped surrogate: take the PESSIMISTIC (max) of the two losses, so
            # an update that helps too much gets clipped exactly like one that hurts
            pg_loss = torch.max(-mb_adv * ratio,
                                -mb_adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)).mean()

            new_value = new_value.flatten()
            if args.clip_vloss:  # trick 2: keep the value from moving too far in one step
                v_clipped = b_values[mb] + torch.clamp(new_value - b_values[mb], -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max((new_value - b_returns[mb]) ** 2,
                                         (v_clipped - b_returns[mb]) ** 2).mean()
            else:
                v_loss = 0.5 * ((new_value - b_returns[mb]) ** 2).mean()

            entropy_loss = entropy.mean()  # trick 3: reward being uncertain, to keep exploring
            loss = pg_loss - args.ent_coef * entropy_loss + args.vf_coef * v_loss
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)  # trick 4: cap the step size
            optimizer.step()
        stats = {"pg_loss": pg_loss.item(), "v_loss": v_loss.item(),
                 "entropy": entropy_loss.item(), "approx_kl": (-log_ratio).mean().item()}
    return stats
# --- endregion ---

# --- region: train ---
next_obs = np.stack([env_reset(i) for i in range(args.num_envs)])
ep_return = np.zeros(args.num_envs, dtype=np.float64)
recent_returns: list[float] = []  # sliding window of finished-episode returns
global_step = 0
for iteration in range(1, num_iterations + 1):
    if args.anneal_lr:  # trick 5: linearly decay lr to 0 over training (stabilizes late updates)
        optimizer.param_groups[0]["lr"] = args.lr * (1.0 - (iteration - 1.0) / num_iterations)
    next_obs = collect_rollout(next_obs, ep_return, recent_returns)
    global_step += batch_size
    advantages, returns = compute_advantages(next_obs)
    stats = ppo_update(advantages, returns)

    mean_return = float(np.mean(recent_returns[-50:])) if recent_returns else float("nan")
    if args.rerun:
        rr.set_time("global_step", sequence=global_step)
        rr.log("charts/episodic_return", rr.Scalars([mean_return]))  # the learning curve
        rr.log("charts/lr", rr.Scalars([optimizer.param_groups[0]["lr"]]))
        for name, value in stats.items():
            rr.log(f"losses/{name}", rr.Scalars([value]))
    if iteration % 10 == 0 or iteration == num_iterations:
        print(f"iter {iteration:4d}/{num_iterations}  step {global_step:7d}  "
              f"mean_return {mean_return:6.1f}  pg {stats['pg_loss']:+.3f}  v {stats['v_loss']:.2f}")
# --- endregion ---

# --- region: eval ---
# Training return is noisy (it includes exploration). Eval strips the noise:
# act with the policy MEAN (no sampling), on seeds held out from every rollout,
# and report the mean length == mean return. This is the number that must climb
# from the random baseline (~34) toward the cap (500) — and beside it we print
# the scripted balancer (500) as the ceiling PPO is chasing.
eval_env = CartpoleEnv()
returns, scripted = [], []
for episode in range(args.eval_episodes):
    obs_now = eval_env.reset(seed=500_000 + args.seed + episode)  # held out from training seeds
    done, ep = False, 0.0
    while not done:
        with torch.no_grad():
            mean_action = agent.actor_mean(torch.as_tensor(obs_now, dtype=torch.float32, device=device).unsqueeze(0))
        obs_now, reward, done, _ = eval_env.step(mean_action[0].cpu().numpy())
        ep += reward
    returns.append(ep)
    s_env = CartpoleEnv()
    s_env.reset(seed=500_000 + args.seed + episode)
    s_done, s_ep = False, 0.0
    while not s_done:
        _, s_r, s_done, _ = s_env.step(balance_action(s_env))
        s_ep += s_r
    scripted.append(s_ep)
mean_eval = float(np.mean(returns))
print(f"eval: mean return {mean_eval:.1f} over {args.eval_episodes} episodes "
      f"(random ~34, scripted {np.mean(scripted):.0f}, cap {CartpoleEnv.MAX_STEPS})")

torch.save(agent.state_dict(), args.out / "ppo_agent.pt")
metrics = {
    "break_bug": bool(args.break_bug),
    "gae_lambda": args.gae_lambda,
    "mean_eval_return": round(mean_eval, 4),
    "mean_scripted_return": round(float(np.mean(scripted)), 4),
    "mean_train_return_last50": round(float(np.mean(recent_returns[-50:])) if recent_returns else float("nan"), 4),
    "norm_adv": bool(args.norm_adv),
    "clip_vloss": bool(args.clip_vloss),
    "num_iterations": num_iterations,
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "total_steps": global_step,
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'ppo.rrd'} — open it with: rerun {args.out / 'ppo.rrd'}")
# --- endregion ---
