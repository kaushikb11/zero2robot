"""zero2robot 5.8 — The Real Loop: Teleop, Record, Train, Deploy on a Real Arm's Body (in sim).

Every method in this course ran against a toy env we wrote. This chapter runs the
WHOLE LeRobot loop against the SO-101's REAL morphology — the same $150 arm G1 sends
you to buy — loaded from google-deepmind/mujoco_menagerie (`robotstudio_so101`): six
`sts3215` position-servo joints, the real link lengths, the real joint limits, a bundled
wrist camera. Free-tier, on a CPU, with the EXACT record format and control loop you would
run on the metal:

  (1) FETCH the SO-101 model reproducibly — download the Menagerie MJCF + 18 STL meshes into
      a gitignored cache (integrity-checked, skip-if-present). No binaries in git (invariant #5).
  (2) DRIVE a scripted expert on the SO-101 + a box: a deterministic reach — point the arm at
      the box (shoulder_pan from the box's azimuth) and lower the gripper to it. This stands in
      for you teleoperating a leader arm; the joint targets ARE the actions a mouse/leader emits.
  (3) RECORD a REAL LeRobotDataset — the actual LeRobotDataset.create -> add_frame ->
      save_episode -> finalize API (v3.0), STATE-only (use_videos=False) so the run is
      byte-reproducible. Byte-for-byte the format ch0.4 wrote and `lerobot-record` writes.
  (4) TRAIN behavior cloning on the recorded dataset — ch1.1's MLP, retargeted to the SO-101's
      9-D observation (6 joints + box xyz) and 6-D joint-position action. Load the dataset BACK
      and fit it; the training never sees the expert, only the recording.
  (5) DEPLOY the clone into the sim SO-101 (policy -> d.ctrl) and EVALUATE with ch1.6 rigor: a
      success RATE over held-out box placements, against no-op and random baselines.

THE HEADLINE (a mechanism claim, seed-robust — not SOTA): the recorded-then-cloned BC policy
reproduces the scripted reach — success clearly above no-op / random — i.e. the real LeRobot loop
closes end-to-end on the real arm's body. We report the DIRECTION (clone >> baselines), not a %:
MuJoCo contact + servo settling are bitwise on ONE CPU but not across arches (ch1.6).

WHAT THE TWIN CANNOT GIVE YOU (the honest core): you drove, recorded, trained, and redeployed on
the SO-101's real kinematics, in the real LeRobot format + loop. What is MISSING is the reality
gap — servo backlash, friction, latency, camera noise, calibration drift. Menagerie's servo gains
are NOT the real STS3215 gains (the MJCF says so); this is morphology + the loop, not torque-level
fidelity. Those gaps are the whole reason hardware exists, and they stay reading: that is G1.

  --break obs_swap : the ch0.4 lesson, on the metal. Your DEPLOY observation wires box_x/box_y in a
      different order than your RECORDING did. Loss is perfect, training is clean — and the arm
      reaches the wrong way. A wiring bug the metric can't see until you roll it out.

Run it:      python curriculum/phase5_practitioner/ch5.8_real_loop/real_loop.py --seed 0
Break it:    python curriculum/phase5_practitioner/ch5.8_real_loop/real_loop.py --seed 0 --break obs_swap
CI smoke:    python curriculum/phase5_practitioner/ch5.8_real_loop/real_loop.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as the other chapters).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.assert_parity import assert_parity  # noqa: E402
from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.export_onnx import export_policy  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

import mujoco  # noqa: E402

OBS_DIM = 9   # 6 joint angles + box (x, y, z)
ACT_DIM = 6   # 6 joint-position targets (shoulder_pan..gripper) — the SO-101's actuators
STATE_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll",
               "gripper", "box_x", "box_y", "box_z"]
JOINT_NAMES = STATE_NAMES[:6]
TASK = "Reach the box with the SO-101 gripper."

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch5.8-real_loop"))
parser.add_argument("--cache", type=Path, default=Path("outputs/menagerie_so101"),
                    help="where the fetched SO-101 Menagerie model is cached (gitignored; ~17MB, one-time)")
parser.add_argument("--menagerie_ref", default="main",
                    help="git ref of google-deepmind/mujoco_menagerie to fetch from (UN-PINNED like G1; the two XMLs are sha-checked)")
parser.add_argument("--demos", type=int, default=64, help="scripted-expert reach episodes to record")  # smoke: 4
parser.add_argument("--epochs", type=int, default=400)  # cpu-laptop: ~1 min | smoke: 2
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--hidden_dim", type=int, default=256)
parser.add_argument("--steps", type=int, default=60, help="control steps per episode (10 Hz); the arm settles onto the box")
parser.add_argument("--eval_episodes", type=int, default=30)  # held-out box placements | smoke: 2
parser.add_argument("--success_tol", type=float, default=0.08, help="gripper-to-box distance (m) counted as a reach")
parser.add_argument("--seed", type=int, default=0, help="seeds the box placements, the init, and the shuffle")
parser.add_argument("--break", dest="break_mode", choices=("obs_swap",), default=None,
                    help="obs_swap = deploy observation wires box_x/box_y opposite to the recording (the ch0.4 lesson)")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true",
                    help="tiny hermetic CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; the MuJoCo env draws its own PCG64 per episode below
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.demos, args.epochs, args.eval_episodes, args.steps, args.device = 4, 2, 2, 20, "cpu"
banner("ch5.8-real_loop", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch5.8-real_loop", spawn=False)
    rr.save(str(args.out / "real_loop.rrd"))
# --- endregion ---

# --- region: fetch ---
# Reproducibly fetch the SO-101 model — Apache-2.0, from google-deepmind/mujoco_menagerie
# (`robotstudio_so101`). We NEVER commit meshes (invariant #5): the MJCF + 18 STL assets download
# once into a gitignored cache, integrity-checked, and skip on every run after. We pull through the
# jsdelivr CDN (a GitHub mirror keyed by the same ref) rather than raw.githubusercontent — same
# bytes, but built to serve many small files without the burst rate-limiting a from-scratch fetch
# trips. The two XMLs we parse are sha-pinned (they define the robot); the binary meshes are
# size-checked (a rate-limited HTML error page is a few hundred bytes — a real STL is tens of KB),
# and the final proof of a good fetch is that MuJoCo compiles the model. The commit is UN-PINNED on
# purpose (G1's doctrine: the upstream moves), so an author bumps --menagerie_ref without this file.
_RAW = "https://cdn.jsdelivr.net/gh/google-deepmind/mujoco_menagerie@{ref}/robotstudio_so101/{path}"
_XML_SHA = {  # sha256 of the two MJCFs we depend on; a changed robot definition trips this
    "so101.xml": "5ad49f2b45c083baac9ffe5d4d3213a5da7eac8039095bb2df177a697aae8308",
    "scene_box.xml": "14c3826f23889587e63b88ef9a0e7d72318ffe13078fad8287c56dbdb9ba9ad8",
}
_ASSETS = [  # the 18 STL meshes so101.xml references (kept as a literal list — no directory crawl)
    "waveshare_mounting_plate_so101_v2.stl", "sts3215_03a_v1.stl", "motor_holder_so101_base_v1.stl",
    "wrist_roll_follower_so101_v1.stl", "moving_jaw_so101_v1.stl", "base_motor_holder_so101_v1.stl",
    "upper_arm_so101_v1.stl", "wrist_roll_pitch_so101_v2.stl", "under_arm_so101_v1.stl",
    "rotation_pitch_so101_v1.stl", "motor_holder_so101_wrist_v1.stl", "sts3215_03a_no_horn_v1.stl",
    "base_so101_v2.stl", "moving_jaw_so101_gripper_v1.stl", "wrist_roll_follower_so101_camera_mount.stl",
    "wrist_roll_follower_so101_gripper_part0_v1.stl", "moving_jaw_so101_gripper_part0_v1.stl",
    "moving_jaw_so101_gripper_part1_v1.stl",
]


def _download(path: str, dest: Path, ref: str) -> None:
    """Fetch one file with a few retries (raw.githubusercontent rate-limits bursts)."""
    url = _RAW.format(ref=ref, path=path)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                dest.write_bytes(resp.read())
            return
        except Exception as err:  # noqa: BLE001 — network flake / rate-limit: back off, retry, then surface
            if attempt == 4:
                sys.exit(f"failed to fetch {url}: {err}")
            time.sleep(1.5 * (attempt + 1))  # raw.githubusercontent throttles bursts; ease off


def fetch_so101(cache: Path, ref: str) -> Path:
    """Download the SO-101 MJCF + meshes into `cache` (skip-if-present), then compile to prove it."""
    scene = cache / "scene_box.xml"
    assets = cache / "assets"
    have_all = scene.is_file() and all((assets / a).is_file() for a in _ASSETS)
    if not have_all:
        assets.mkdir(parents=True, exist_ok=True)
        for name, want in _XML_SHA.items():
            _download(name, cache / name, ref)
            got = hashlib.sha256((cache / name).read_bytes()).hexdigest()
            if got != want:
                sys.exit(f"{name} sha256 {got} != expected {want} (menagerie_ref={ref!r} moved the robot def)")
        for a in _ASSETS:
            _download(f"assets/{a}", assets / a, ref)
            if (assets / a).stat().st_size < 10_000:  # an HTML error page, not a mesh
                sys.exit(f"asset {a} is {(assets / a).stat().st_size} bytes — a corrupt/rate-limited download; re-run")
    model = mujoco.MjModel.from_xml_path(str(scene))  # the real integrity check: does it compile?
    assert (model.nu, model.nq) == (6, 13), f"unexpected SO-101 model: nu={model.nu} nq={model.nq}"
    (cache / "manifest.json").write_text(json.dumps(  # a pointer, never the meshes themselves
        {"ref": ref, "source": "google-deepmind/mujoco_menagerie/robotstudio_so101",
         "xml_sha256": _XML_SHA, "assets": sorted(_ASSETS)}, indent=2, sort_keys=True) + "\n")
    return scene


scene_path = fetch_so101(args.cache, args.menagerie_ref)
print(f"SO-101 model: {scene_path} (cached; ref={args.menagerie_ref})")
# --- endregion ---

# --- region: env ---
# The SO-101 reach env, wrapping the Menagerie model. This is the "real arm's body": we do NOT
# simplify the kinematics — the six sts3215 position servos, link lengths, and joint limits are
# the manufacturer's. reset() respawns the box on the floor in a reachable patch (via its
# freejoint, exactly as ch1.3's env writes cube qpos), and obs/step/dist mirror the PushT/ALOHA
# env contract so the BC loop below is unchanged from ch1.1. Bitwise-deterministic on one CPU.
HOME = np.array([0.0, -1.57, 1.57, 1.0, 0.0, 0.0])  # a raised, out-of-the-way rest pose
FRAME_SKIP = 10  # 100 Hz physics (timestep 0.005) / 10 Hz control


class SO101ReachEnv:
    def __init__(self, scene_xml: Path):
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self._box_qadr = self.model.jnt_qposadr[self.model.body("box").jntadr[0]]  # box freejoint qpos start (=6)
        self._tip = self.model.site("gripperframe").id
        self._box = self.model.body("box").id
        self._lo = self.model.actuator_ctrlrange[:, 0]
        self._hi = self.model.actuator_ctrlrange[:, 1]

    def reset(self, seed: int) -> np.ndarray:
        rng = np.random.Generator(np.random.PCG64(seed))
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:6] = HOME
        bx, by = rng.uniform(0.26, 0.31), rng.uniform(-0.14, 0.14)  # a reachable floor patch in front
        self.data.qpos[self._box_qadr:self._box_qadr + 3] = [bx, by, 0.03]
        self.data.qpos[self._box_qadr + 3:self._box_qadr + 7] = [1, 0, 0, 0]
        self.data.ctrl[:] = HOME
        mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.data.qpos[:6], self.data.body(self._box).xpos]).astype(np.float32)

    def step(self, ctrl: np.ndarray) -> np.ndarray:
        self.data.ctrl[:] = np.clip(ctrl, self._lo, self._hi)  # position servos track the target
        for _ in range(FRAME_SKIP):
            mujoco.mj_step(self.model, self.data)
        return self._obs()

    def gripper_to_box(self) -> float:
        return float(np.linalg.norm(np.array(self.data.site(self._tip).xpos) - np.array(self.data.body(self._box).xpos)))


env = SO101ReachEnv(scene_path)
# --- endregion ---

# --- region: expert ---
# The scripted expert = your teleoperation, scripted so CI can diff runs. A leader arm hands you
# joint targets; here a tiny controller computes them. The reach is mostly a fixed "lower-to-the-
# table" pose (shoulder_lift/elbow/wrist), STEERED by shoulder_pan toward the box's azimuth — the
# one joint that must depend on where the box is. It commands the TARGET pose every step and lets
# the position servos interpolate (a stable feedback law, not an open-loop trajectory — that is why
# the clone learns it without drifting). PAN_GAIN is a one-number fit of pan -> tip azimuth.
REACH_POSE = np.array([0.2, 0.3, 0.7])  # shoulder_lift, elbow_flex, wrist_flex — lowers the gripper to the table
PAN_GAIN = 0.87


def expert_action(box_xyz: np.ndarray) -> np.ndarray:
    """The 6-D joint-position target that reaches `box_xyz`. Deterministic in the box position."""
    pan = -np.arctan2(box_xyz[1], box_xyz[0]) / PAN_GAIN  # aim shoulder_pan at the box's azimuth
    return np.array([pan, REACH_POSE[0], REACH_POSE[1], REACH_POSE[2], 0.0, 0.3], dtype=np.float32)
# --- endregion ---

# --- region: record ---
# Record a REAL LeRobotDataset — the identical create -> add_frame -> save_episode -> finalize
# sequence ch0.4 and `lerobot-record` use, STATE-only (use_videos=False) so the bytes are
# reproducible on CPU. Episode i places the box with env seed (seed + i): a different, repeatable
# reach each time. We store the obs we ACTED ON (pre-step) — the off-by-one ch0.4 warned about.
def build_features() -> dict:
    return {"observation.state": {"dtype": "float32", "shape": (OBS_DIM,), "names": STATE_NAMES},
            "action": {"dtype": "float32", "shape": (ACT_DIM,), "names": JOINT_NAMES}}


def record_demos(dataset_root: Path) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # lazy: pulls in torch/datasets

    if dataset_root.exists():  # regenerate every run: a stale dir from another --seed is silent wrong data (ch1.1)
        shutil.rmtree(dataset_root)
    dataset = LeRobotDataset.create(repo_id="zero2robot/so101_reach", fps=10, features=build_features(),
                                    root=dataset_root, robot_type="so101_follower", use_videos=False)
    for i in range(args.demos):
        obs = env.reset(args.seed + i)
        target = expert_action(obs[6:9])  # the reach target for THIS box (constant over the episode)
        for _ in range(args.steps):
            dataset.add_frame({"observation.state": obs, "action": target, "task": TASK})
            obs = env.step(target)
        dataset.save_episode()
    dataset.finalize()  # compute stats, write meta/*.parquet — a dataset you could push to the Hub


dataset_root = args.out / "dataset"
record_demos(dataset_root)
# --- endregion ---

# --- region: model ---
# Load the recorded dataset BACK and train BC on it — the clone never sees the expert, only the
# recording (the whole point of the loop). The model is ch1.1's MLP, retargeted 9 -> 6. One change
# matters for a robot: the six joints move over VERY different ranges (shoulder_lift swings ~1.7 rad,
# shoulder_pan ~0.5), so we compute the MSE in NORMALIZED action space — otherwise the big joints
# dominate the loss and the tiny box-dependent pan signal (the only thing that matters) is ignored.
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

frames = LeRobotDataset("zero2robot/so101_reach", root=dataset_root).hf_dataset.with_format("numpy")
obs_np = np.stack(frames["observation.state"]).astype(np.float32)  # (N, 9)
act_np = np.stack(frames["action"]).astype(np.float32)             # (N, 6)
obs_min, obs_max = obs_np.min(0), obs_np.max(0)
obs_range = np.where(obs_max - obs_min < 1e-4, np.float32(1.0), obs_max - obs_min)
act_min, act_max = act_np.min(0), act_np.max(0)
act_range = np.where(act_max - act_min < 1e-4, np.float32(1.0), act_max - act_min)


class BCPolicy(nn.Module):
    """obs float32[9] -> action float32[6]. Same 3-layer MLP as ch1.1; normalization lives inside
    as buffers so the checkpoint / ONNX carry their own stats and the deploy loop feeds raw obs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, ACT_DIM),
        )
        for name, stat in [("obs_min", obs_min), ("obs_range", obs_range),
                           ("act_min", act_min), ("act_range", act_range)]:
            self.register_buffer(name, torch.from_numpy(np.asarray(stat, np.float32)))

    def net_norm(self, obs: torch.Tensor) -> torch.Tensor:  # raw obs -> NORMALIZED action (train target)
        normalized = 2.0 * (obs - self.obs_min) / self.obs_range - 1.0
        return self.net(normalized.clamp(-1.0, 1.0))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:   # raw obs -> RAW joint targets (deploy / ONNX)
        return (self.net_norm(obs) + 1.0) / 2.0 * self.act_range + self.act_min


policy = BCPolicy(args.hidden_dim).to(device)
# --- endregion ---

# --- region: train ---
optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
train_obs = torch.from_numpy(obs_np).to(device)
train_act = torch.from_numpy(act_np).to(device)
# The MSE target is the NORMALIZED action, so every joint weighs equally (see the model region).
target_norm = 2.0 * (train_act - policy.act_min) / policy.act_range - 1.0
shuffle = torch.Generator().manual_seed(args.seed)
train_loss, global_step = float("nan"), 0
for epoch in range(args.epochs):
    epoch_loss, num_batches = 0.0, 0
    for batch in torch.randperm(len(train_obs), generator=shuffle).split(args.batch_size):
        loss = nn.functional.mse_loss(policy.net_norm(train_obs[batch]), target_norm[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss, num_batches = epoch_loss + loss.item(), num_batches + 1
        if args.rerun:
            rr.set_time("step", sequence=global_step)
            rr.log("policy/loss/train", rr.Scalars([loss.item()]))
        global_step += 1
    train_loss = epoch_loss / num_batches
    if epoch % 50 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:3d}  train_loss {train_loss:.5f}")
torch.save(policy, args.out / "bc_policy.pt")
# --- endregion ---

# --- region: deploy ---
# Deploy the clone into the sim SO-101 (policy -> d.ctrl) and evaluate with ch1.6 rigor: a success
# RATE over held-out box placements (seeds 10_000+, never trained on), not one hero rollout. We run
# the clone AND two baselines that must FAIL — a no-op that holds the rest pose, and a random-target
# flail — so the headline is a DIRECTION (clone >> baselines), the seed-robust claim. --break obs_swap
# feeds the clone an observation whose box_x/box_y are transposed vs the recording: the ch0.4 lesson,
# now on a real arm — training is spotless, the arm reaches the wrong way.
def rollout(kind: str, episode: int) -> tuple[bool, float, np.ndarray]:
    obs = env.reset(10_000 + episode)  # held-out placements, disjoint from the seed+i training set
    rng = np.random.Generator(np.random.PCG64(args.seed * 1000 + episode))
    tips = []
    for _ in range(args.steps):
        model_in = obs.copy()
        if kind == "clone" and args.break_mode == "obs_swap":
            model_in[6], model_in[7] = model_in[7], model_in[6]  # deploy obs != record obs
        if kind == "noop":
            ctrl = HOME.astype(np.float32)
        elif kind == "random":
            ctrl = rng.uniform(env._lo, env._hi).astype(np.float32)
        else:
            with torch.no_grad():
                ctrl = policy(torch.from_numpy(model_in).to(device).unsqueeze(0))[0].cpu().numpy()
        obs = env.step(ctrl)
        tips.append(np.array(env.data.site(env._tip).xpos))
    return env.gripper_to_box() < args.success_tol, env.gripper_to_box(), np.array(tips)


def success_rate(kind: str) -> float:
    return sum(rollout(kind, e)[0] for e in range(args.eval_episodes)) / args.eval_episodes


clone_rate = success_rate("clone")
noop_rate = success_rate("noop")
random_rate = success_rate("random")
print(f"eval: clone {clone_rate:.2f} | no-op {noop_rate:.2f} | random {random_rate:.2f}  "
      f"(success = gripper within {args.success_tol} m of the box, break={args.break_mode or 'none'})")

# Capture two gripper-tip paths for the toy NOW, while the policy is still on `device` (the ONNX
# export below moves it to CPU): the clone reaching the box, and the same box mis-wired (obs_swap).
_, _, clone_tips = rollout("clone", 0)
saved_break, args.break_mode = args.break_mode, "obs_swap"
_, _, swap_tips = rollout("clone", 0)
args.break_mode = saved_break
box0 = env.reset(10_000)[6:9]
# --- endregion ---

# --- region: report ---
# The full loop ends where every chapter does: export the clone to ONNX (contract v1) and prove
# torch and onnxruntime agree, so the SO-101 policy could drop into the browser playground.
policy.eval()
onnx_path = export_policy(policy, OBS_DIM, ACT_DIM, args.out / "bc_policy.onnx")
parity_delta = assert_parity(policy, onnx_path, OBS_DIM)

metrics = {
    "act_dim": ACT_DIM,
    "break_mode": args.break_mode or "none",
    "clone_success_rate": round(clone_rate, 6),
    "clone_beats_baselines": bool(clone_rate > max(noop_rate, random_rate)),  # the DIRECTION the gates assert
    "demos": args.demos,
    "epochs": args.epochs,
    "eval_episodes": args.eval_episodes,
    "final_train_loss": round(train_loss, 6),
    "noop_success_rate": round(noop_rate, 6),
    "obs_dim": OBS_DIM,
    "parity_delta": round(parity_delta, 6),
    "random_success_rate": round(random_rate, 6),
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "success_tol": args.success_tol,
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

# demo/vizdata.json: the money picture for the toy — a clone rollout that reaches vs a break
# rollout that misses, as gripper-tip paths the TSX replays over the loop diagram (captured above).
vizdata = {
    "task": TASK, "success_tol": args.success_tol, "box": [round(float(v), 4) for v in box0],
    "clone_rate": metrics["clone_success_rate"], "noop_rate": metrics["noop_success_rate"],
    "random_rate": metrics["random_success_rate"],
    "clone_tip_path": clone_tips.round(4).tolist(),   # (steps, 3) — reaches the box
    "break_tip_path": swap_tips.round(4).tolist(),    # (steps, 3) — obs_swap: reaches the wrong way
}
(args.out / "demo").mkdir(parents=True, exist_ok=True)
(args.out / "demo" / "vizdata.json").write_text(json.dumps(vizdata) + "\n")

if args.rerun:
    rr.log("eval/clone_success_rate", rr.Scalars([clone_rate]), static=True)
    rr.log("eval/noop_success_rate", rr.Scalars([noop_rate]), static=True)
    rr.log("eval/random_success_rate", rr.Scalars([random_rate]), static=True)
    for t, tip in enumerate(clone_tips):  # the clone's gripper reaching the box, over sim time
        rr.set_time("sim_time", duration=t / 10.0)
        rr.log("world/robot/gripper_tip", rr.Points3D([tip], radii=[0.01], colors=(230, 102, 90)))
        rr.log("world/objects/box", rr.Points3D([box0], radii=[0.02], colors=(90, 205, 100)), static=True)

print(f"exported {onnx_path} — torch/onnx parity delta {parity_delta:.2e}")
print(f"metrics: {args.out / 'metrics.json'}  |  vizdata: {args.out / 'demo' / 'vizdata.json'}")
print(f"clone beats baselines: {metrics['clone_beats_baselines']}  "
      f"(clone {clone_rate:.2f} vs no-op {noop_rate:.2f} / random {random_rate:.2f})")
if args.rerun:
    print(f"recording: {args.out / 'real_loop.rrd'} — open it with: rerun {args.out / 'real_loop.rrd'}")
# --- endregion ---
