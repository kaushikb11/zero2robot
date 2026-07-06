"""zero2robot 1.2 — Data Is the Policy.

Chapter 1.1 ended on a confession: behavior cloning can only be as good as the
demonstrations it averages over, and when demonstrations disagree, the average
of two good actions can be a bad one. This chapter takes that seriously. Your
ch0.4 recordings are a MIXTURE — some careful runs that reached the goal, some
sloppy ones that wandered and never rotated the block home. Train 1.1 on the
whole pile and the sloppy episodes poison exactly the hard states. So we SCORE
each episode on honest, dataset-only quality signals (did it reach the goal?
how long? how much do its states DISAGREE with the rest of the data?), FILTER
the bad ones out, and re-train — on FEWER episodes — to a MEASURABLY higher
success rate. Quality beats quantity, and here you measure by how much.

The twist (Break It): the obvious quality knob is the wrong one. `--break
low_disagreement` keeps the episodes that agree most with their neighbours —
which sounds like exactly the 1.1 lesson — and it makes the policy WORSE than
honest curation, because disagreement was never measuring noise. It was
measuring difficulty.

Run it:      python curriculum/phase1_imitation/ch1.2_curate/curate.py --seed 0
Break it:    python curriculum/phase1_imitation/ch1.2_curate/curate.py --seed 0 --break low_disagreement
CI smoke:    python curriculum/phase1_imitation/ch1.2_curate/curate.py --smoke --seed 0 --no-rerun
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
# sys.path so `curriculum.common` resolves (same pattern as ch1.1's bc.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv, gen_demos, wrap_angle  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

# The two halves of your simulated ch0.4 session. `--data` overrides this with
# your REAL recordings; without it we build a reproducible stand-in so the
# chapter's numbers reproduce on your machine (same honesty as bc.py's smoke).
CAREFUL_NOISE = 0.05  # a steady hand: the expert barely wobbles, reaches the goal
SLOPPY_NOISE = 0.70   # a shaky hand: large action noise, most runs wander and fail

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=None,
                    help="an existing LeRobot dataset to curate (your ch0.4 session). Omitted => build the reproducible careful+sloppy stand-in")
parser.add_argument("--out", type=Path, default=Path("outputs/ch1.2-curate"))
parser.add_argument("--careful", type=int, default=250, help="stand-in only: # careful (mostly-good) episodes")  # smoke: 3
parser.add_argument("--sloppy", type=int, default=250, help="stand-in only: # sloppy (mostly-bad) episodes")   # smoke: 3
parser.add_argument("--epochs", type=int, default=300)  # cpu-laptop: minutes (trains TWICE) | smoke: 3
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--eval_episodes", type=int, default=50)  # held-out reset seeds | smoke: 3
parser.add_argument("--knn", type=int, default=8, help="neighbours per frame for the disagreement estimate")
parser.add_argument("--seed", type=int, default=0, help="seeds data generation, the split, the init, and the shuffle")
parser.add_argument("--break", dest="break_mode", choices=("low_disagreement", "shortest"), default=None,
                    help="Break It: replace the honest outcome filter with a plausible-but-wrong heuristic")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())  # cpu: deterministic (statistical repro on GPU/mps)
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.careful, args.sloppy, args.epochs, args.eval_episodes, args.device = 3, 3, 3, 3, "cpu"
banner("ch1.2-curate", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch1.2-curate", spawn=False)
    rr.save(str(args.out / "curate.rrd"))
# --- endregion ---

# --- region: data ---
# A curation module needs a dataset with BOTH good and bad episodes in it, or
# there is nothing to curate. Real teleop is exactly that mixture; the scripted
# expert with a large noise std is the reproducible stand-in (careful hand vs
# shaky hand). Both halves are written by the SAME gen_demos as every other
# PushT dataset, so `observation.state`/`action` are byte-identical in layout to
# what bc.py trains on — the curated set drops straight back into chapter 1.1.
def load_lerobot(root: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # heavy import — after cheap failures

    frames = LeRobotDataset("local/curate", root=root).hf_dataset.with_format("numpy")
    return (np.stack(frames["observation.state"]), np.stack(frames["action"]),
            np.asarray(frames["episode_index"]))


def build_stand_in(out: Path, careful: int, sloppy: int, seed: int):
    """Careful + sloppy halves via gen_demos, concatenated into one array set.
    Regenerated every run (never reuse a leftover dir): a cache from a different
    --seed would silently train on the wrong data. gen_demos is deterministic,
    so same seed -> bit-identical episodes whether just built or rebuilt."""
    obs_parts, act_parts, eid_parts, next_eid = [], [], [], 0
    for name, count, noise, seed0 in [("careful", careful, CAREFUL_NOISE, seed),
                                      ("sloppy", sloppy, SLOPPY_NOISE, seed + 5000)]:
        root = out / f"raw-{name}"
        if root.exists():
            shutil.rmtree(root)
        gen_demos.main(["--episodes", str(count), "--seed", str(seed0), "--noise", str(noise),
                        "--out", str(root), "--no-video", "--repo-id", f"zero2robot/pusht_{name}"])
        o, a, e = load_lerobot(root)
        obs_parts.append(o)
        act_parts.append(a)
        eid_parts.append(e + next_eid)
        next_eid += int(e.max()) + 1
    return np.concatenate(obs_parts), np.concatenate(act_parts), np.concatenate(eid_parts)


if args.data is not None:
    if not (args.data / "meta" / "info.json").is_file():
        sys.exit(f"no dataset at {args.data} — record one in ch0.4 first, or omit --data to build the stand-in")
    obs, actions, episode_ids = load_lerobot(args.data)
else:
    obs, actions, episode_ids = build_stand_in(args.out, args.careful, args.sloppy, args.seed)

episodes = np.unique(episode_ids)  # one row per demonstration; the unit we score and filter
print(f"raw dataset: {len(episodes)} episodes / {len(obs)} frames")
# --- endregion ---

# --- region: quality ---
# Three honest, dataset-ONLY quality signals — no environment rollouts, nothing
# the learner could not compute from the recording sitting on their disk.
def episode_reached_goal(ep_obs: np.ndarray) -> bool:
    """Did the block finish inside the task tolerance? The last recorded frame
    carries tee_xy and sin/cos(yaw); decode and compare to PushT's own limits.
    A demonstration that never reached the goal is a bad label source."""
    last = ep_obs[-1]
    pos_err = float(np.hypot(last[2], last[3]))
    ang_err = float(abs(wrap_angle(np.arctan2(last[4], last[5]))))
    return pos_err < PushTEnv.POS_TOL and ang_err < PushTEnv.ANG_TOL


def frame_disagreement(obs: np.ndarray, actions: np.ndarray, episode_ids: np.ndarray, k: int) -> np.ndarray:
    """For each frame, how much do the demonstrations DISAGREE near its state?
    Find the k nearest frames from OTHER episodes (same-episode neighbours are
    temporal near-duplicates and trivially agree) and take the spread of their
    actions. High spread = a state where good demonstrators chose differently —
    the multimodality chapter 1.1 said MSE would average into mush."""
    span = obs.max(0) - obs.min(0)
    obs_n = torch.from_numpy(((obs - obs.min(0)) / np.where(span < 1e-4, 1.0, span)).astype(np.float32))
    act_t = torch.from_numpy(actions.astype(np.float32))
    eid_t = torch.from_numpy(episode_ids)
    out = np.zeros(len(obs_n), dtype=np.float64)
    for start in range(0, len(obs_n), 1000):  # chunk the distance matrix so it fits in memory
        query, query_eid = obs_n[start:start + 1000], eid_t[start:start + 1000]
        dist = torch.cdist(query, obs_n)
        dist[query_eid[:, None] == eid_t[None, :]] = float("inf")  # mask same-episode neighbours
        neighbours = act_t[dist.topk(k, largest=False).indices]    # (chunk, k, 2)
        # population std (correction=0) — matches numpy's default so the exercise
        # completion can use either library; the RANKING is what curation uses.
        out[start:start + 1000] = neighbours.std(dim=1, correction=0).mean(dim=1).numpy()
    return out


def coverage(ep_starts: np.ndarray, bins: int = 6) -> float:
    """Fraction of a bins x bins grid over the arena that some episode STARTS in.
    A dataset can be large and still blind to whole regions of the state space."""
    cells = np.clip(((ep_starts + 0.3) / 0.6 * bins).astype(int), 0, bins - 1)
    return len(set(map(tuple, cells))) / (bins * bins)


disagree = frame_disagreement(obs, actions, episode_ids, args.knn)
# Roll the per-frame signals up to one row per episode.
reached = np.array([episode_reached_goal(obs[episode_ids == e]) for e in episodes])
lengths = np.array([int((episode_ids == e).sum()) for e in episodes])
ep_disagree = np.array([disagree[episode_ids == e].mean() for e in episodes])
ep_starts = np.array([obs[episode_ids == e][0][2:4] for e in episodes])
print(f"quality: {int(reached.sum())}/{len(episodes)} episodes reached the goal | "
      f"mean disagreement {ep_disagree.mean():.4f} | coverage {coverage(ep_starts):.2f}")
if args.rerun:
    for e in range(len(episodes)):  # scrub the per-episode quality signals in rerun
        rr.set_time("episode", sequence=e)
        rr.log("quality/reached_goal", rr.Scalars([float(reached[e])]))
        rr.log("quality/disagreement", rr.Scalars([ep_disagree[e]]))
        rr.log("quality/length", rr.Scalars([float(lengths[e])]))
# --- endregion ---

# --- region: curate ---
# The honest filter keeps the episodes that reached the goal. Break It swaps in
# a plausible-but-wrong ranking, keeping the SAME NUMBER of episodes so the only
# thing that changes is WHICH ones — isolating the heuristic from dataset size.
kept_by_outcome = episodes[reached]
if args.break_mode is None:
    kept = kept_by_outcome
    selection = "outcome (reached the goal)"
else:
    budget = len(kept_by_outcome)  # match honest curation's episode count exactly
    if args.break_mode == "low_disagreement":
        order = episodes[np.argsort(ep_disagree)]          # "keep the demos that agree" — the trap
    else:  # shortest
        order = episodes[np.argsort(lengths)]              # "keep the efficient demos" — also a trap
    kept = order[:budget]
    selection = f"BREAK:{args.break_mode}"

keep_mask = np.isin(episode_ids, kept)
kept_starts = np.array([obs[episode_ids == e][0][2:4] for e in kept])
print(f"curated ({selection}): {len(kept)}/{len(episodes)} episodes / {int(keep_mask.sum())} frames | "
      f"coverage {coverage(kept_starts):.2f} | mean disagreement {ep_disagree[np.isin(episodes, kept)].mean():.4f}")
# --- endregion ---

# --- region: train ---
# A compact copy of chapter 1.1's behavior cloning — deliberately duplicated
# (single-file doctrine), so "re-train 1.1 on the curated data" is something you
# can read here, not a black box you import. Same 3-layer MLP, same in-model
# min-max normalization, same cosine-decayed Adam. We call it TWICE (raw, then
# curated) and compare the only number that matters: held-out success rate.
class BCPolicy(nn.Module):
    def __init__(self, hidden_dim, obs_min, obs_range, act_min, act_range):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PushTEnv.OBS_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, PushTEnv.ACT_DIM),
        )
        for name, stat in [("obs_min", obs_min), ("obs_range", obs_range),
                           ("act_min", act_min), ("act_range", act_range)]:
            self.register_buffer(name, torch.from_numpy(stat))

    def forward(self, obs):  # (B,10) raw -> normalize -> net -> denormalize -> (B,2) raw
        normalized_obs = 2.0 * (obs - self.obs_min) / self.obs_range - 1.0
        normalized_action = self.net(normalized_obs.clamp(-1.0, 1.0))
        return (normalized_action + 1.0) / 2.0 * self.act_range + self.act_min


def train_and_eval(frame_mask: np.ndarray, tag: str) -> dict:
    """Fit BC on the frames selected by `frame_mask`, roll out on held-out seeds."""
    set_seed(args.seed)  # each policy starts from the same seeded init, so the comparison is fair
    ds_obs, ds_act = obs[frame_mask], actions[frame_mask]
    obs_min, act_min = ds_obs.min(0), ds_act.min(0)
    obs_range = np.where(ds_obs.max(0) - obs_min < 1e-4, np.float32(1.0), ds_obs.max(0) - obs_min)
    act_range = np.where(ds_act.max(0) - act_min < 1e-4, np.float32(1.0), ds_act.max(0) - act_min)
    policy = BCPolicy(args.hidden_dim, obs_min, obs_range, act_min, act_range).to(device)
    train_obs = torch.from_numpy(ds_obs).to(device)
    train_actions = torch.from_numpy(ds_act).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    shuffle = torch.Generator().manual_seed(args.seed)
    train_loss = float("nan")
    for epoch in range(args.epochs):
        epoch_loss, num_batches = 0.0, 0
        for batch in torch.randperm(len(train_obs), generator=shuffle).split(128):
            loss = nn.functional.mse_loss(policy(train_obs[batch]), train_actions[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss, num_batches = epoch_loss + loss.item(), num_batches + 1
            if args.rerun:
                rr.log(f"policy/loss/train/{tag}", rr.Scalars([loss.item()]))
        scheduler.step()
        train_loss = epoch_loss / num_batches

    env = PushTEnv()
    policy.eval()
    successes, returns, near, far = 0, [], [0, 0], [0, 0]
    for episode in range(args.eval_episodes):
        obs_now = env.reset(seed=10_000 + args.seed + episode)  # held-out starts, never trained on
        start_dist = float(np.hypot(obs_now[2], obs_now[3]))
        episode_return, done, info = 0.0, False, {}
        while not done:
            with torch.no_grad():
                action = policy(torch.from_numpy(obs_now).to(device).unsqueeze(0))[0].cpu().numpy()
            obs_now, reward, done, info = env.step(action)
            episode_return += reward
        successes += bool(info["success"])
        returns.append(episode_return)
        bucket = near if start_dist < 0.15 else far  # split by difficulty: near vs far starts
        bucket[0] += bool(info["success"])
        bucket[1] += 1
    return {"tag": tag, "n_frames": int(frame_mask.sum()),
            "success_rate": successes / args.eval_episodes,
            "mean_return": float(np.mean(returns)), "final_train_loss": train_loss,
            "near_success": near, "far_success": far}


raw_result = train_and_eval(np.ones(len(obs), dtype=bool), "raw")
curated_result = train_and_eval(keep_mask, "curated")
# --- endregion ---

# --- region: report ---
# The payoff, stated as one number: does curating — on FEWER episodes — lift the
# held-out success rate? And WHERE (the far/near split shows the sloppy episodes
# were poisoning the hard starts). rerun gets the same two bars to eyeball.
delta = curated_result["success_rate"] - raw_result["success_rate"]
for result in (raw_result, curated_result):
    print(f"{result['tag']:8s}: {result['n_frames']:6d} frames -> success {result['success_rate']:.3f}  "
          f"(near {result['near_success'][0]}/{result['near_success'][1]}, "
          f"far {result['far_success'][0]}/{result['far_success'][1]})")
print(f"delta (curated - raw): {delta:+.3f}")
if args.rerun:
    rr.set_time("episode", sequence=len(episodes))
    rr.log("payoff/success/raw", rr.Scalars([raw_result["success_rate"]]))
    rr.log("payoff/success/curated", rr.Scalars([curated_result["success_rate"]]))

metrics = {
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "break_mode": args.break_mode or "none",
    "n_episodes": int(len(episodes)),
    "n_reached_goal": int(reached.sum()),
    "n_kept": int(len(kept)),
    "mean_disagreement_raw": round(float(ep_disagree.mean()), 6),
    "mean_disagreement_kept": round(float(ep_disagree[np.isin(episodes, kept)].mean()), 6),
    "coverage_raw": round(coverage(ep_starts), 6),
    "coverage_kept": round(coverage(kept_starts), 6),
    "raw_success_rate": round(raw_result["success_rate"], 6),
    "curated_success_rate": round(curated_result["success_rate"], 6),
    "delta_success_rate": round(delta, 6),
    "raw_final_train_loss": round(raw_result["final_train_loss"], 6),
    "curated_final_train_loss": round(curated_result["final_train_loss"], 6),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}")

# Write the curated dataset so you can re-run chapter 1.1 on it directly:
#   python curriculum/phase1_imitation/ch1.1_bc/bc.py --data <out>/curated-dataset
if not args.smoke:
    curated_root = args.out / "curated-dataset"
    if curated_root.exists():
        shutil.rmtree(curated_root)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset.create(repo_id="zero2robot/pusht_curated", fps=PushTEnv.CONTROL_HZ,
                                    features=gen_demos.build_features(False), root=curated_root,
                                    robot_type="pusher_2d", use_videos=False)
    for e in kept:
        for i in np.nonzero(episode_ids == e)[0]:
            dataset.add_frame({"observation.state": obs[i], "action": actions[i], "task": gen_demos.TASK})
        dataset.save_episode()
    dataset.finalize()
    print(f"curated dataset: {curated_root}  (re-train ch1.1 on it: bc.py --data {curated_root})")
if args.rerun:
    print(f"recording: {args.out / 'curate.rrd'} — open it with: rerun {args.out / 'curate.rrd'}")
# --- endregion ---
