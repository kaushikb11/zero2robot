"""zero2robot 1.1 — Behavior Cloning: The Dumbest Thing That Works.

You drove the pusher yourself in chapter 0.4. Behavior cloning skips the
part where you explain HOW you drove: collect (observation, action) pairs,
fit a network to map one to the other with plain MSE, and hope the state
distribution at rollout time looks like the one in your dataset. This file
is the whole method — load demos, split by episode, normalize, fit a small
MLP, roll it out in the real env, export to ONNX for the browser playground.

Run it:      python curriculum/phase1_imitation/ch1.1_bc/bc.py --seed 0
Break it:    python curriculum/phase1_imitation/ch1.1_bc/bc.py --seed 0 --normalize narrow
CI smoke:    python curriculum/phase1_imitation/ch1.1_bc/bc.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as tests/).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.assert_parity import assert_parity  # noqa: E402
from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv, gen_demos  # noqa: E402
from curriculum.common.export_onnx import export_policy  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path("outputs/pusht-demos"),
                    help="LeRobot-format demo dataset (your ch0.4 teleop session, or gen_demos.py output)")
parser.add_argument("--out", type=Path, default=Path("outputs/ch1.1-bc"))
parser.add_argument("--epochs", type=int, default=600)  # cpu-laptop: minutes | smoke: 3
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3, help="peak Adam lr; cosine-decays to 0 over --epochs")
parser.add_argument("--hidden_dim", type=int, default=512)  # width is NOT the bottleneck — see the model region
parser.add_argument("--seed", type=int, default=0, help="seeds the split, the init, and the shuffle")
parser.add_argument("--eval_episodes", type=int, default=50)  # T4: 50 | smoke: 5
parser.add_argument("--normalize", choices=("full", "narrow"), default="full",
                    help="narrow = stats from a lopsided slice of the demos; the Break It flag")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())  # T4: cuda | Mac: mps | cpu: deterministic (statistical repro on GPU/mps)
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # seeds python/numpy/torch; returns the numpy Generator the split draws from
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.epochs, args.eval_episodes, args.device = 3, 5, "cpu"
banner("ch1.1-bc", device=args.device)  # report the device the run ACTUALLY uses (after --smoke pins cpu)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch1.1-bc", spawn=False)
    rr.save(str(args.out / "bc.rrd"))
# --- endregion ---

# --- region: data ---
# Two dataset paths, same format: the reference is `lerobot/pusht` from the
# HF Hub (human demos); the path this chapter assumes is YOURS — the teleop
# session you recorded in ch0.4, or the scripted-expert set from gen_demos.py.
if args.smoke:
    # Smoke runs are hermetic: CI regenerates its own tiny deterministic set.
    # REGENERATE it every run (never reuse a leftover dir): a cache from a
    # different --seed would train on seed-0 demos while metrics.json records
    # seed 1 — silent wrong data. gen_demos is deterministic, so same seed ->
    # bit-identical dataset whether it was just built or rebuilt.
    args.data = args.out / "smoke-demos"
    if args.data.exists():
        shutil.rmtree(args.data)
    gen_demos.main(["--episodes", "6", "--seed", str(args.seed),
                    "--out", str(args.data), "--no-video"])
if not (args.data / "meta" / "info.json").is_file():
    sys.exit(f"no dataset at {args.data} — record one in ch0.4, or generate demos first:\n"
             f"  python curriculum/common/envs/pusht/gen_demos.py "
             f"--episodes 500 --seed 0 --out {args.data} --no-video")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402  (heavy import — after the cheap failures)

frames = LeRobotDataset("local/pusht-demos", root=args.data).hf_dataset.with_format("numpy")
obs = np.stack(frames["observation.state"])  # (N, 10) — layout documented in pusht_env.py
actions = np.stack(frames["action"])         # (N, 2)  — pusher velocity in [-1, 1]
episode_ids = np.asarray(frames["episode_index"])  # (N,) which demo each frame came from

# Split by EPISODE, never by frame. Frames 0.1 s apart are near-duplicates;
# a frame-level split puts one twin in train and one in val, and val loss
# becomes a memorization test you can only pass. Episode-level keeps val
# honest: whole trajectories the network has never seen.
episode_order = rng.permutation(np.unique(episode_ids))
val_episodes = episode_order[: max(1, len(episode_order) // 10)]
train_episodes = episode_order[len(val_episodes):]
in_val = np.isin(episode_ids, val_episodes)

# Normalization stats: per-dim min/max, mapping each dim to [-1, 1] — the
# same scheme the real PushT policies use (diffusion_policy, LeRobot).
# full: stats over every training frame, so by construction no training
# input ever leaves [-1, 1]. narrow: stats over only the ~20% of training
# episodes whose block STARTS closest to the target — the demos you'd record
# first while testing your teleop rig, block placed gently near the goal.
# Both splits share the same stats either way, which is exactly why no loss
# curve will flag the difference (Break It).
if args.normalize == "narrow":
    def block_start_distance(episode: int) -> float:
        tee_x, tee_y = obs[episode_ids == episode][0][2:4]  # episode's first frame
        return float(np.hypot(tee_x, tee_y))
    easiest = sorted(train_episodes, key=block_start_distance)[: max(1, len(train_episodes) // 5)]
    stats_frames = np.isin(episode_ids, easiest)
else:
    stats_frames = ~in_val
obs_min, act_min = obs[stats_frames].min(0), actions[stats_frames].min(0)
obs_range = obs[stats_frames].max(0) - obs_min
act_range = actions[stats_frames].max(0) - act_min
# The 4 target dims are constant in this phase -> range 0. A constant carries
# no information; give it range 1 so it maps to a constant instead of a
# division by zero.
obs_range = np.where(obs_range < 1e-4, np.float32(1.0), obs_range)
act_range = np.where(act_range < 1e-4, np.float32(1.0), act_range)

train_obs = torch.from_numpy(obs[~in_val]).to(device)
train_actions = torch.from_numpy(actions[~in_val]).to(device)
val_obs = torch.from_numpy(obs[in_val]).to(device)
val_actions = torch.from_numpy(actions[in_val]).to(device)
print(f"dataset: {len(episode_order)} episodes / {len(obs)} frames "
      f"({len(train_episodes)} train / {len(val_episodes)} val episodes), normalize={args.normalize}")
# --- endregion ---

# --- region: model ---
class BCPolicy(nn.Module):
    """3-layer MLP, obs float32[10] -> action float32[2]. Deliberately boring.

    The ceiling of behavior cloning is the data, not the network: this policy
    can never act better than the demonstrations it averages over, and past a
    point extra width just buys a sharper copy of the same mistakes (exercise
    3 makes you measure where that point is). Normalization lives INSIDE the
    model, as buffers: the checkpoint and the ONNX export carry their own
    stats, so the playground (tensor contract v1) feeds raw observations.
    """

    def __init__(self, hidden_dim: int, obs_min, obs_range, act_min, act_range):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PushTEnv.OBS_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, PushTEnv.ACT_DIM),
        )
        for name, stat in [("obs_min", obs_min), ("obs_range", obs_range),
                           ("act_min", act_min), ("act_range", act_range)]:
            self.register_buffer(name, torch.from_numpy(stat))  # saved with the weights, but never trained

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # (B, 10) raw -> (B, 10) in [-1, 1] -> (B, 2) in [-1, 1] -> (B, 2) raw
        normalized_obs = 2.0 * (obs - self.obs_min) / self.obs_range - 1.0
        # The clamp guards the net against inputs outside the range it trained
        # on. When the stats cover the demos (they define [-1, 1]) it never
        # moves a training value — remember it exists. Break It wakes it up.
        normalized_action = self.net(normalized_obs.clamp(-1.0, 1.0))
        return (normalized_action + 1.0) / 2.0 * self.act_range + self.act_min


policy = BCPolicy(args.hidden_dim, obs_min, obs_range, act_min, act_range).to(device)
# --- endregion ---

# --- region: train ---
# A plain loop — no DataLoader, no scheduler, no early stopping. The whole
# dataset sits in memory as two tensors; "batching" is indexing a shuffled
# permutation. MSE says: predict the average action the demonstrator took in
# this state. Defensible (it's maximum likelihood under a Gaussian), flawed
# (when demos disagree, the average of two good actions can be a bad one —
# that flaw is chapter 1.2's opening problem).
optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
# One concession to optimization reality: decay the lr to 0 over the run, or
# the last epochs bounce around the minimum instead of settling into it
# (measured: +6 points of success rate at the default config).
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
shuffle = torch.Generator().manual_seed(args.seed)  # torch-side RNG: same seed -> same batch order
train_loss, global_step = float("nan"), 0
for epoch in range(args.epochs):
    epoch_loss, num_batches = 0.0, 0
    for batch in torch.randperm(len(train_obs), generator=shuffle).split(args.batch_size):
        loss = nn.functional.mse_loss(policy(train_obs[batch]), train_actions[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss, num_batches = epoch_loss + loss.item(), num_batches + 1
        if args.rerun:
            rr.set_time("step", sequence=global_step)
            rr.log("policy/loss/train", rr.Scalars([loss.item()]))
        global_step += 1
    scheduler.step()
    train_loss = epoch_loss / num_batches
    with torch.no_grad():
        val_loss = nn.functional.mse_loss(policy(val_obs), val_actions).item()
    if args.rerun:
        rr.log("policy/loss/val", rr.Scalars([val_loss]))
    if epoch % 50 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:3d}  train_loss {train_loss:.5f}  val_loss {val_loss:.5f}")

torch.save(policy, args.out / "bc_policy.pt")  # whole module — reloadable only where THIS file's BCPolicy is importable
# TorchScript carries its own code, so assert_parity's CLI reloads it in a
# fresh interpreter with no BCPolicy on the path (the cross-process gate):
#   python curriculum/common/assert_parity.py <out>/bc_policy.onnx <out>/bc_policy.ts.pt
torch.jit.script(policy.eval()).save(str(args.out / "bc_policy.ts.pt"))
# --- endregion ---

# --- region: eval ---
# Loss measured how well we imitate on the dataset's states. Rollouts measure
# what we actually care about: does the block reach the target when the POLICY
# picks the states it visits? Those are different questions — Break It shows
# just how different.
env = PushTEnv()
policy.eval()
successes, episode_returns = 0, []
for episode in range(args.eval_episodes):
    # 10_000 + offset: demo episode i used reset seed (seed + i), so eval
    # seeds are held out by construction — never graded on a start we trained on.
    obs_now = env.reset(seed=10_000 + args.seed + episode)
    episode_return, done, info = 0.0, False, {}
    while not done:
        with torch.no_grad():  # the rollout loop: obs -> action -> step (exercise 2 blanks this)
            obs_batch = torch.from_numpy(obs_now).to(device).unsqueeze(0)  # (10,) -> (1, 10)
            action = policy(obs_batch)[0].cpu().numpy()
        obs_now, reward, done, info = env.step(action)
        episode_return += reward
        if args.rerun:
            # offset each episode by the max horizon so traces don't overlap on one timeline
            rr.set_time("sim_time", duration=episode * (PushTEnv.MAX_STEPS / PushTEnv.CONTROL_HZ) + env.data.time)
            rr.log("policy/action", rr.Scalars(action.astype(np.float64)))
            rr.log("eval/pos_err", rr.Scalars([info["pos_err"]]))
    successes += bool(info["success"])
    episode_returns.append(episode_return)
    if args.rerun:
        rr.log("eval/success", rr.Scalars([float(info["success"])]))
        rr.log("eval/episode_return", rr.Scalars([episode_return]))
success_rate = successes / args.eval_episodes
print(f"eval: {successes}/{args.eval_episodes} episodes succeeded "
      f"(success rate {success_rate:.2f}), mean return {np.mean(episode_returns):.3f}")

# The full loop ends in the browser: export to ONNX (tensor contract v1),
# then prove torch and onnxruntime agree before the file goes anywhere.
onnx_path = export_policy(policy, PushTEnv.OBS_DIM, PushTEnv.ACT_DIM, args.out / "bc_policy.onnx")
parity_delta = assert_parity(policy, onnx_path, PushTEnv.OBS_DIM)
print(f"exported {onnx_path} — torch/onnx parity delta {parity_delta:.2e}")

metrics = {
    "epochs": args.epochs,
    "eval_episodes": args.eval_episodes,
    "final_train_loss": round(train_loss, 6),
    "final_val_loss": round(val_loss, 6),
    "mean_episode_return": round(float(np.mean(episode_returns)), 6),
    "normalize": args.normalize,
    "parity_delta": round(parity_delta, 6),  # rounds to 0.0 in practice; the gate already asserted < 1e-4
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "success_rate": round(success_rate, 6),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'bc.rrd'} — open it with: rerun {args.out / 'bc.rrd'}")
# --- endregion ---
