"""zero2robot 3.7 — Datasets at Scale.

Real robot policies are trained on BIG, MESSY, CROSS-EMBODIMENT data: Open
X-Embodiment (>1M trajectories, 22 robots), DROID (76k trajectories, 564
scenes). You cannot download that on a T4, and you do not need to — every hard
problem it poses is already visible in the two little datasets YOU made. This
file makes the scaled-up reality concrete on the learner's own demos, in three
moves, from scratch:

  (1) WRANGLE / CROSS-EMBODIMENT: load your PushT (2-D pusher) and ALOHA (6-D
      bimanual) demos — two embodiments, two action spaces, one pile — and hit
      the exact problems OXE hits: heterogeneous action dims, per-embodiment
      normalization, a shared "10-number" state whose numbers MEAN different
      things. Zero-pad + action_mask (the same trick you built in ch1.7).
  (2) AUGMENT (MimicGen-style): grow the PushT set by perturbing each source
      demo's object/pusher pose and RE-SOLVING it with the scripted expert —
      a genuinely new, physically valid demo (kept only if the expert still
      succeeds). This is the data engine, honestly: no fabricated actions.
  (3) MEASURE: train the ch1.1 BC policy on N source demos, then on N +
      augmented, and roll both out. Data is the policy (ch1.2), scaled: more
      effective demos -> higher success. Even a modest gain is the lesson.

Everything is offline: no OXE/DROID download, no Hub. The source demos come
from the same gen_demos.py that fed ch1.1; the augmented demos are written to
--out (gitignored). OXE/DROID at true scale is the Scale Lab (meta.yaml).

Run it:      python curriculum/phase3_advanced/ch3.7_scale_data/scale_data.py --seed 0
CI smoke:    python curriculum/phase3_advanced/ch3.7_scale_data/scale_data.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import shutil
import sys
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as the other chapters).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.aloha_cube import gen_demos as aloha_gen_demos  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv, gen_demos as pusht_gen_demos  # noqa: E402
from curriculum.common.envs.pusht import ScriptedExpert, wrap_angle  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch3.7-scale-data"))
parser.add_argument("--pusht_data", type=Path, default=None,
                    help="PushT LeRobot demos; default: a fresh coverage-starved set under --out. "
                         "Pass your own only if it holds exactly --source_episodes demos (else it is rebuilt).")
parser.add_argument("--aloha_data", type=Path, default=None,
                    help="ALOHA LeRobot demos (the second embodiment for the wrangling lesson); "
                         "default: a fresh set under --out")
parser.add_argument("--source_episodes", type=int, default=12,
                    help="source demos per embodiment (deliberately SMALL: coverage-starved so augmentation can help)")
parser.add_argument("--aug_per_demo", type=int, default=8,
                    help="MimicGen-style variants to re-solve per PushT source demo (T4: 8 | smoke: 1)")
parser.add_argument("--aug_pos_sigma", type=float, default=0.08,
                    help="std (m) of the object/pusher pose perturbation before re-solving (wider = more coverage)")
parser.add_argument("--aug_yaw_sigma", type=float, default=0.50,
                    help="std (rad) of the object yaw perturbation before re-solving")
parser.add_argument("--epochs", type=int, default=500)  # cpu-laptop: minutes | smoke: 3
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3, help="peak Adam lr; cosine-decays to 0 over --epochs")
parser.add_argument("--hidden_dim", type=int, default=512)
parser.add_argument("--eval_episodes", type=int, default=50)  # T4: 50 | smoke: 5
parser.add_argument("--seed", type=int, default=0, help="seeds the demos, the augmentation, the split, and training")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true",
                    help="tiny hermetic CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # seeds python/numpy/torch; returns the numpy Generator the augmentation draws from
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    (args.source_episodes, args.aug_per_demo, args.epochs, args.eval_episodes,
     args.hidden_dim, args.device) = 4, 1, 3, 5, 64, "cpu"
    args.pusht_data = args.aloha_data = None  # force the hermetic default below
# Demos default to a CHAPTER-PRIVATE path under --out. This chapter needs its OWN
# coverage-starved 12-demo set; defaulting into a shared path (e.g. ch1.1's 500-demo
# outputs/pusht-demos) would either fight that chapter's data or get rebuilt out from
# under it. Chapter-private means each --seed (and each exercise --out) gets its own demos.
args.pusht_data = args.pusht_data or args.out / "pusht-demos"
args.aloha_data = args.aloha_data or args.out / "aloha-demos"
banner("ch3.7-scale-data", device=args.device)  # report the device the run ACTUALLY uses
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch3.7-scale-data", spawn=False)
    rr.save(str(args.out / "scale_data.rrd"))
# --- endregion ---

# --- region: wrangle ---
# The cross-embodiment reality, on YOUR data. Two embodiments, two datasets, two
# action spaces. A real cross-embodiment mix (RT-X, OpenVLA) stacks dozens of
# these; the problems are identical and already visible with two.
ACT_DIM_MAX = 6  # pusht acts in 2 dims, aloha in 6 — the shared tensor is padded to the max


def ensure_dataset(path: Path, gen, episodes: int, seed: int) -> None:
    """Generate a LeRobot demo dataset if missing (offline, deterministic), and
    REBUILD it if a cached one was built for a different (episodes, seed) than this
    run asks for. Reusing a mismatched cache would silently train on the wrong demos
    while metrics.json reports THIS run's config, and either field can drift:
      * episodes — the whole lesson is the COVERAGE-STARVED regime (a handful of
        source demos). A fatter cache (say ch1.1's 500-demo set) would erase it.
      * seed — a cache from another --seed is different data; the demos must match
        the seed metrics.json records.
    gen re-derives byte-identical demos from (episodes, seed) alone, so a rebuild is
    cheap and deterministic. The default path is chapter-private (see setup), so we
    never read or clobber a dataset another chapter wrote."""
    spec = {"episodes": episodes, "seed": seed}
    stamp = path / "z2r_demospec.json"  # the (episodes, seed) the cached demos were built for
    if stamp.is_file() and json.loads(stamp.read_text()) != spec:
        shutil.rmtree(path)  # cached demos were built for a different run -> stale
    if not stamp.is_file():
        if path.exists():
            shutil.rmtree(path)  # a partial/foreign dir with no stamp -> clean before regen
        gen.main(["--episodes", str(episodes), "--seed", str(seed), "--out", str(path), "--no-video"])
        stamp.write_text(json.dumps(spec) + "\n")


def load_lerobot(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read a LeRobot v3 dataset into (obs, action, episode_index) numpy arrays —
    exactly the wrangling every training stack does, laid bare."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # heavy import, after cheap failures
    frames = LeRobotDataset("local/demos", root=path).hf_dataset.with_format("numpy")
    return (np.stack(frames["observation.state"]).astype(np.float32),
            np.stack(frames["action"]).astype(np.float32),
            np.asarray(frames["episode_index"]))


ensure_dataset(args.pusht_data, pusht_gen_demos, args.source_episodes, args.seed)
ensure_dataset(args.aloha_data, aloha_gen_demos, args.source_episodes, args.seed)
pusht_obs, pusht_act, pusht_ep = load_lerobot(args.pusht_data)
aloha_obs, aloha_act, aloha_ep = load_lerobot(args.aloha_data)

# The heterogeneity, made numeric. Normalization stats are PER EMBODIMENT: a
# pusher's 1 m/s and a gripper's open/close command do not share a scale, so one
# global normalizer would crush one embodiment's signal. This is why OXE-scale
# training normalizes per-dataset, and why "what transfers" is a real question —
# both states are 10 numbers, but idx 2 is a block's x here and a gripper's
# closedness there. A shared policy transfers STRUCTURE (a small MLP, an action
# head), never the SEMANTICS of a raw dimension.
embodiments = []
for name, obs, act, act_dim in (("pusht", pusht_obs, pusht_act, PushTEnv.ACT_DIM),
                                ("aloha", aloha_obs, aloha_act, 6)):
    embodiments.append({
        "name": name, "act_dim": act_dim, "frames": int(len(obs)),
        "action_min": act.min(0).round(4).tolist(), "action_max": act.max(0).round(4).tolist(),
    })
    print(f"[{name}] {len(obs)} frames, action_dim={act_dim}, "
          f"action range per dim {act.min(0).round(2)}..{act.max(0).round(2)}")

# Zero-pad both into ONE action tensor with an action_mask marking the real dims
# (the ch1.7 trick). This is the honest cost of mixing embodiments: the model
# always emits 6 numbers; the mask says which ones this embodiment actually uses.
n_total = len(pusht_act) + len(aloha_act)
mixed_action = np.zeros((n_total, ACT_DIM_MAX), np.float32)
action_mask = np.zeros((n_total, ACT_DIM_MAX), np.float32)
mixed_action[:len(pusht_act), :PushTEnv.ACT_DIM] = pusht_act
action_mask[:len(pusht_act), :PushTEnv.ACT_DIM] = 1.0
mixed_action[len(pusht_act):, :6] = aloha_act
action_mask[len(pusht_act):, :6] = 1.0
(args.out / "cross_embodiment.json").write_text(json.dumps({
    "embodiments": embodiments, "mixed_frames": n_total, "padded_action_dim": ACT_DIM_MAX,
    "note": "shared 10-dim state, but dims mean different things per embodiment; normalize per embodiment",
}, indent=2) + "\n")
print(f"mixed pile: {n_total} frames, padded action dim {ACT_DIM_MAX}, "
      f"mean action_mask density {action_mask.mean():.3f} (pusht wastes 4 of 6 dims)")
# --- endregion ---

# --- region: augment ---
# MimicGen-style augmentation of the PushT demos. For each source demo we read
# its FIRST state, perturb the object + pusher pose, drop the env into that
# perturbed start, and RE-SOLVE with the same scripted expert that made the
# demos. The result is a genuinely new trajectory in the real physics — not a
# fabricated one — and we keep it ONLY if the expert still succeeds. That
# success filter is the honesty gate: an augmented demo the solver cannot finish
# is not a demo, it is noise.
_JOINTS = ("tee_x", "tee_y", "tee_yaw", "pusher_x", "pusher_y")


def pusht_state_obs(env: PushTEnv) -> np.ndarray:
    """The 10-dim PushT observation from public env props (mirrors pusht_env._obs);
    used to record the augmented demo's frames without touching env internals."""
    px, py = env.pusher_pos
    tx, ty, tyaw = env.tee_pose
    gx, gy, gyaw = env.TARGET_POSE
    return np.array([px, py, tx, ty, np.sin(tyaw), np.cos(tyaw),
                     gx, gy, np.sin(gyaw), np.cos(gyaw)], dtype=np.float32)


def set_pusht_start(env: PushTEnv, seed: int, tee_xy, tee_yaw: float, pusher_xy) -> None:
    """Reset env (fresh counters), then place it at a chosen start via the public
    MuJoCo model/data + documented joint names — the MimicGen 'new object pose'."""
    env.reset(seed)  # resets step/success counters and mjData
    adr = {j: env.model.joint(j).qposadr[0] for j in _JOINTS}
    q = env.data.qpos
    q[adr["tee_x"]], q[adr["tee_y"]] = tee_xy
    q[adr["tee_yaw"]] = tee_yaw
    q[adr["pusher_x"]], q[adr["pusher_y"]] = pusher_xy
    env.data.ctrl[:] = 0.0
    mujoco.mj_forward(env.model, env.data)


def solve_from(env: PushTEnv, seed: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Roll the scripted expert from the env's current start; return (obs, act, success)."""
    expert = ScriptedExpert(noise=0.0, seed=seed)
    obs_list, act_list, done = [], [], False
    obs = pusht_state_obs(env)
    while not done:
        action = expert.action(env)
        obs_list.append(obs)
        act_list.append(action)
        obs, _, done, info = env.step(action)
    return np.asarray(obs_list, np.float32), np.asarray(act_list, np.float32), bool(info["success"])


aug_env = PushTEnv()
aug_obs_all, aug_act_all, aug_ep_all = [], [], []
attempts, kept = 0, 0
next_ep = int(pusht_ep.max()) + 1  # augmented demos get fresh episode ids after the source demos
for src in np.unique(pusht_ep):
    first = pusht_obs[pusht_ep == src][0]  # this source demo's initial state
    base_tee, base_yaw = first[2:4], float(np.arctan2(first[4], first[5]))
    base_pusher = first[0:2]
    for _ in range(args.aug_per_demo):
        attempts += 1
        tee_xy = base_tee + rng.normal(0.0, args.aug_pos_sigma, size=2)
        # rescale the block onto ~the spawn annulus (a hair wider than the env's own
        # [0.10, 0.24] so augmentation also probes its edges): keep direction, clamp radius
        radius = np.hypot(*tee_xy)
        tee_xy = tee_xy * np.clip(radius, 0.08, 0.26) / (radius + 1e-9)
        tee_yaw = wrap_angle(base_yaw + rng.normal(0.0, args.aug_yaw_sigma))
        pusher_xy = np.clip(base_pusher + rng.normal(0.0, args.aug_pos_sigma, size=2), -0.30, 0.30)
        if np.linalg.norm(pusher_xy - tee_xy) < PushTEnv._PUSHER_CLEAR:  # nudge the pusher clear of the block
            away = (pusher_xy - tee_xy) / (np.linalg.norm(pusher_xy - tee_xy) + 1e-9)
            pusher_xy = np.clip(tee_xy + away * (PushTEnv._PUSHER_CLEAR + 0.02), -0.30, 0.30)
        set_pusht_start(aug_env, args.seed + int(src), tee_xy, tee_yaw, pusher_xy)
        o, a, ok = solve_from(aug_env, args.seed + int(src))
        if ok:  # the success filter: only physically valid, solved demos join the pile
            kept += 1
            aug_obs_all.append(o)
            aug_act_all.append(a)
            aug_ep_all.append(np.full(len(o), next_ep))
            next_ep += 1
aug_yield = kept / attempts if attempts else 0.0
print(f"augmentation: {kept}/{attempts} perturbed re-solves succeeded (yield {aug_yield:.2f}), "
      f"+{kept} demos on top of {len(np.unique(pusht_ep))} source demos")
# --- endregion ---

# --- region: measure ---
# Data is the policy (ch1.2), scaled. Train the SAME small BC MLP from ch1.1 on
# the source demos alone, then on source + augmented, and compare rollout
# success. Everything past this point is the ch1.1 recipe, deliberately.
class BCPolicy(nn.Module):
    """3-layer MLP, obs float32[10] -> action float32[2], with normalization
    baked in as buffers (ch1.1). Identical for both training runs so the only
    variable is the DATA — the whole point of the measurement."""

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

    def forward(self, obs):
        normalized_obs = 2.0 * (obs - self.obs_min) / self.obs_range - 1.0
        normalized_action = self.net(normalized_obs.clamp(-1.0, 1.0))
        return (normalized_action + 1.0) / 2.0 * self.act_range + self.act_min


def train_and_eval(obs: np.ndarray, act: np.ndarray, tag: str) -> tuple[float, int]:
    """Fit BC on (obs, act) with the ch1.1 recipe, roll it out, return (success_rate, frames)."""
    # Reset the weight-init RNG so BOTH arms start from IDENTICAL weights. With the batch
    # order also fixed (shuffle seed below), the training DATA is the only thing that differs
    # between source-only and source+augmented — which is the entire point of the measurement.
    torch.manual_seed(args.seed)
    obs_min, act_min = obs.min(0), act.min(0)
    obs_range = np.where((obs.max(0) - obs_min) < 1e-4, np.float32(1.0), obs.max(0) - obs_min)
    act_range = np.where((act.max(0) - act_min) < 1e-4, np.float32(1.0), act.max(0) - act_min)
    policy = BCPolicy(args.hidden_dim, obs_min, obs_range, act_min, act_range).to(device)
    obs_t = torch.from_numpy(obs).to(device)
    act_t = torch.from_numpy(act).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    shuffle = torch.Generator().manual_seed(args.seed)  # same seed -> same batch order for both arms
    for epoch in range(args.epochs):
        for batch in torch.randperm(len(obs_t), generator=shuffle).split(args.batch_size):
            loss = nn.functional.mse_loss(policy(obs_t[batch]), act_t[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    env = PushTEnv()
    policy.eval()
    successes = 0
    for episode in range(args.eval_episodes):
        obs_now, done, info = env.reset(seed=10_000 + args.seed + episode), False, {}  # held-out eval seeds
        while not done:
            with torch.no_grad():
                action = policy(torch.from_numpy(obs_now).to(device).unsqueeze(0))[0].cpu().numpy()
            obs_now, _, done, info = env.step(action)
        successes += bool(info["success"])
    rate = successes / args.eval_episodes
    print(f"[{tag}] {len(obs)} frames -> eval success {successes}/{args.eval_episodes} = {rate:.2f}")
    return rate, len(obs)


source_rate, source_frames = train_and_eval(pusht_obs, pusht_act, "source-only")
if aug_obs_all:
    all_obs = np.concatenate([pusht_obs, *aug_obs_all])
    all_act = np.concatenate([pusht_act, *aug_act_all])
else:  # no augmented demo survived the success filter (rare); the honest fallback is source-only
    all_obs, all_act = pusht_obs, pusht_act
aug_rate, aug_frames = train_and_eval(all_obs, all_act, "source+augmented")

if args.rerun:  # the data-scale curve: success vs training-set size, both arms
    rr.set_time("dataset_size", sequence=source_frames)
    rr.log("scale/success_rate", rr.Scalars([source_rate]))
    rr.set_time("dataset_size", sequence=aug_frames)
    rr.log("scale/success_rate", rr.Scalars([aug_rate]))
    rr.log("scale/aug_yield", rr.Scalars([aug_yield]), static=True)
# --- endregion ---

# --- region: emit ---
metrics = {
    "aug_demos_kept": int(kept),
    "aug_frames": int(aug_frames - source_frames),
    "aug_yield": round(aug_yield, 6),
    "augmented_success_rate": round(aug_rate, 6),
    "embodiment_act_dims": {"aloha": 6, "pusht": int(PushTEnv.ACT_DIM)},
    "mixed_frames": int(n_total),
    "scale_effect": round(aug_rate - source_rate, 6),  # the honest headline: does more data help?
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "source_demos": int(len(np.unique(pusht_ep))),
    "source_frames": int(source_frames),
    "source_success_rate": round(source_rate, 6),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"scale effect: source {source_rate:.2f} -> augmented {aug_rate:.2f} "
      f"(delta {aug_rate - source_rate:+.2f}) — data is the policy, scaled")
print(f"metrics: {args.out / 'metrics.json'} + cross_embodiment.json")
if args.rerun:
    print(f"recording: {args.out / 'scale_data.rrd'} — open it with: rerun {args.out / 'scale_data.rrd'}")
# --- endregion ---
