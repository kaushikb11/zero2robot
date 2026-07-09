"""zero2robot 0.0 — Your First Robot Policy in a Few Minutes.

This is the whole loop, on one page, running in a couple of minutes on a laptop:
generate a handful of demonstrations, train a tiny neural network to copy them,
and watch it push a T-shaped block onto a target it has never seen. That's a
robot policy. You trained it. It works.

We skip the *why* on purpose. Every piece here — the sim, the dataset format,
the network, the covariate-shift disease behavior cloning dies of — you build
properly, one idea at a time, starting in ch0.1. This chapter exists only to
put a working policy on your screen first, so the rest of Phase 0 reads as
"now let's understand what you just did" instead of "trust me, this pays off."

Two honest simplifications, both undone later:
  * Demos come from a SCRIPTED expert (ch0.4 teaches you to record your own by
    hand); we hold them in memory instead of writing a real LeRobot dataset.
  * The network is a plain MLP trained with mean-squared error (ch1.1 is this
    same idea, done right, with the failure modes named).

Run it:      python curriculum/phase0_foundations/ch0.0_quickstart/quickstart.py --seed 0
CI smoke:    python curriculum/phase0_foundations/ch0.0_quickstart/quickstart.py --smoke --seed 0 --no-rerun
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
# sys.path so `curriculum.common` resolves (same pattern as every chapter).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv, ScriptedExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="seeds the demos, the init, the shuffle, and the held-out eval starts")
parser.add_argument("--out", type=Path, default=Path("outputs/ch0.0-quickstart"))
parser.add_argument("--demos", type=int, default=300, help="scripted-expert episodes to learn from (cpu-laptop: 300 | smoke: 6)")
parser.add_argument("--epochs", type=int, default=600)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--eval_episodes", type=int, default=25, help="held-out starts to grade on (never seen in training)")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device(),
                    help="reference numbers are --device cpu, the only bitwise-deterministic tier")
parser.add_argument("--smoke", action="store_true", help="tiny fixed run for CI; two --smoke runs produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # seeds python/numpy/torch; returns the numpy Generator the random baseline draws from
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.demos, args.epochs, args.eval_episodes, args.device = 6, 3, 3, "cpu"
banner("ch0.0-quickstart", device=args.device)  # tier + measured wall-clock to stdout
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch0.0-quickstart", spawn=False)
    rr.save(str(args.out / "quickstart.rrd"))
# --- endregion ---

# --- region: demos ---
# A demonstration is a list of (observation, action) pairs — what the world
# looked like, and what a competent driver did about it. In ch0.4 that driver
# is YOU, with a mouse. Here it's the scripted expert from the shared PushT env
# (a hand-tuned controller that took real engineering to write — that effort is
# exactly what behavior cloning lets us skip). We drive it a few hundred times
# and just remember what it saw and did. No dataset file, no disk: two arrays.
def collect_demos(n_episodes: int, seed: int):
    env = PushTEnv()
    observations, actions, expert_successes = [], [], 0
    for i in range(n_episodes):
        obs = env.reset(seed + i)  # seed+i per episode: each demo is a different, reproducible start
        expert = ScriptedExpert(noise=0.0, seed=seed + i)
        done, info = False, {}
        while not done:
            action = expert.action(env)
            observations.append(obs.copy())  # the obs we acted ON, never the terminal one
            actions.append(action.copy())
            obs, _, done, info = env.step(action)
        expert_successes += bool(info["success"])
    return np.asarray(observations, np.float32), np.asarray(actions, np.float32), expert_successes


demo_obs, demo_actions, expert_successes = collect_demos(args.demos, args.seed)
print(f"demos: {args.demos} expert episodes ({expert_successes} solved) -> {len(demo_obs)} (obs, action) pairs")
# --- endregion ---

# --- region: model ---
# Rescale every input and output dimension to [-1, 1] using its min/max over the
# demos — meters, sin/cos and velocities all live on different scales, and a
# network learns far faster when they don't. The stats live INSIDE the model so
# the policy is self-contained. (ch1.1 shows how a lie in these stats sinks a
# policy that every loss curve swears is healthy.)
def minmax(x):  # -> (min, range) with zero-width dims (the constant target pose) held safe
    lo = x.min(0)
    span = np.where(x.max(0) - lo < 1e-4, np.float32(1.0), x.max(0) - lo)
    return torch.from_numpy(lo), torch.from_numpy(span.astype(np.float32))


class BCPolicy(nn.Module):
    """obs float32[10] -> action float32[2]. A 3-layer MLP: no planner, no
    search, no idea what a "T" is. It maps "when the world looks like this" to
    "do that", and that is enough to push a block home most of the time."""

    def __init__(self, hidden_dim, obs_min, obs_range, act_min, act_range):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PushTEnv.OBS_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, PushTEnv.ACT_DIM),
        )
        for name, stat in [("obs_min", obs_min), ("obs_range", obs_range),
                           ("act_min", act_min), ("act_range", act_range)]:
            self.register_buffer(name, stat)  # travels with the weights, never trained

    def forward(self, obs):
        normalized = (2.0 * (obs - self.obs_min) / self.obs_range - 1.0).clamp(-1.0, 1.0)
        action = self.net(normalized)
        return (action + 1.0) / 2.0 * self.act_range + self.act_min


obs_min, obs_range = minmax(demo_obs)
act_min, act_range = minmax(demo_actions)
policy = BCPolicy(args.hidden_dim, obs_min, obs_range, act_min, act_range).to(device)
# --- endregion ---

# --- region: train ---
# The entire training method: predict the action the expert took, penalize the
# squared error, repeat. No DataLoader, no early stopping — the demos are two
# tensors in memory and a "batch" is a slice of a shuffled index permutation.
train_obs = torch.from_numpy(demo_obs).to(device)
train_actions = torch.from_numpy(demo_actions).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)  # decay to 0 so the last epochs settle
shuffle = torch.Generator().manual_seed(args.seed)
for epoch in range(args.epochs):
    epoch_loss, num_batches = 0.0, 0
    for batch in torch.randperm(len(train_obs), generator=shuffle).split(256):
        loss = nn.functional.mse_loss(policy(train_obs[batch]), train_actions[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss, num_batches = epoch_loss + loss.item(), num_batches + 1
    scheduler.step()
    if args.rerun:
        rr.set_time("epoch", sequence=epoch)
        rr.log("policy/loss", rr.Scalars([epoch_loss / num_batches]))
    if epoch % 100 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:3d}  loss {epoch_loss / num_batches:.5f}")
# --- endregion ---

# --- region: eval ---
# The only test that counts: turn the policy loose on starts it never trained
# on and see whether the block reaches the target. Eval seeds are offset by
# 10_000 so they can't collide with a demo start. We also roll out a RANDOM
# policy on the same starts — the floor the trained policy has to clear for
# "it works" to mean anything. And we save the first success for the browser.
def rollout(act_fn, seed, record=False):
    env = PushTEnv()
    obs, done, info = env.reset(seed), False, {}
    frames = []
    while not done:
        action = act_fn(obs)
        if record:
            px, py = env.pusher_pos
            tx, ty, tyaw = env.tee_pose
            pos_err, _ = env._errors()
            frames.append([round(float(v), 5) for v in (px, py, tx, ty, tyaw, pos_err)])
        obs, _, done, info = env.step(action)
    return bool(info["success"]), frames


policy.eval()


def policy_action(obs):
    with torch.no_grad():
        return policy(torch.from_numpy(obs).to(device).unsqueeze(0))[0].cpu().numpy()


def random_action(_obs):
    return rng.uniform(-1.0, 1.0, size=PushTEnv.ACT_DIM).astype(np.float32)


successes, saved_rollout = 0, None
for episode in range(args.eval_episodes):
    solved, frames = rollout(policy_action, 10_000 + args.seed + episode, record=True)
    successes += solved
    if solved and saved_rollout is None:  # keep the first honest success for replay
        saved_rollout = {"seed": 10_000 + args.seed + episode, "frames": frames}
    if args.rerun:
        rr.set_time("eval_episode", sequence=episode)
        rr.log("eval/success", rr.Scalars([float(solved)]))
random_successes = sum(rollout(random_action, 10_000 + args.seed + e)[0] for e in range(args.eval_episodes))

success_rate = successes / args.eval_episodes
random_rate = random_successes / args.eval_episodes
print(f"\n  trained policy: {successes}/{args.eval_episodes} solved  ({success_rate:.0%})")
print(f"  random policy:  {random_successes}/{args.eval_episodes} solved  ({random_rate:.0%})   <- the floor you cleared")
if success_rate > random_rate:  # the real run: a visible, honest win over the random floor
    print("\nYou trained a robot policy and it works. Now go build every piece of it, starting in ch0.1.")
else:  # only the tiny --smoke budget lands here; it exists to check the plumbing, not to win
    print("\n(smoke budget — too small to solve the task; the full run is where it works.)")

metrics = {
    "seed": args.seed,
    "demos": args.demos,
    "epochs": args.epochs,
    "eval_episodes": args.eval_episodes,
    "expert_successes": expert_successes,
    "successes": successes,
    "success_rate": round(success_rate, 6),
    "random_successes": random_successes,
    "random_rate": round(random_rate, 6),
    "smoke": bool(args.smoke),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
# The rollout the site replays: one real, held-out success (never cherry-picked
# beyond "the first one"). None only if the tiny --smoke budget solved nothing.
rollout_path = args.out / "rollout.json"
rollout_path.write_text(json.dumps({
    "success_rate": round(success_rate, 6),
    "random_rate": round(random_rate, 6),
    "target_pose": [round(float(v), 5) for v in PushTEnv.TARGET_POSE],
    "rollout": saved_rollout,
}, indent=2) + "\n")
print(f"\nmetrics: {args.out / 'metrics.json'}   rollout: {rollout_path}")
if args.rerun:
    print(f"recording: {args.out / 'quickstart.rrd'} — open it with: rerun {args.out / 'quickstart.rrd'}")
# --- endregion ---
