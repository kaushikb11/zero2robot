"""zero2robot 3.6 — Full Circle: Run Your ch1.1 Policy in the Engine You Built.

Since chapter 0.1 you have trusted `mj_step`. Chapters 3.3-3.5 opened the box:
DYNAMICS (ch3.3, semi-implicit Euler), JOINTS as constraints you SOLVE FOR (ch3.4,
the Baumgarte distance link), CONTACT as the one-sided complementarity you project
(ch3.5, `solve_contacts`). This chapter closes the circle: we RE-CREATE PushT
inside that from-scratch numpy engine and run the SAME behavior-cloning policy you
trained in ch1.1 — the one that learned in MuJoCo — in the engine YOU built.

It PARTLY works, and the gap is the lesson. Your engine is a SIMPLIFICATION (the
T-block is two point masses on a ch3.4 rigid link, contact is frictionless
normal-only, the pusher is an idealized velocity servo), so the policy transfers
IMPERFECTLY: lower success, and trajectories that start identical and drift apart.
That sim-to-SIM gap is the same animal as the sim-to-REAL gap of ch2.6 — a policy
meeting dynamics its training never showed it. We MEASURE it two honest ways:
  * TRANSFER: BC success in MuJoCo (ground truth) vs in your engine (closed-loop).
  * DIVERGENCE: from ONE shared start, replay MuJoCo's EXACT actions in your
    engine and watch the block poses pull apart (open-loop — pure dynamics gap).
A perfect match would be suspicious; it is not what you get.

Run it:      python compare.py --policy outputs/ch1.1-bc/bc_policy.ts.pt --seed 0
             python compare.py --policy <ts.pt> --episodes 50 --block_damp 20
CI smoke:    python compare.py --smoke --seed 0 --no-rerun   # fresh policy, two runs byte-identical

No trained policy yet? Train ch1.1's OWN canonical policy (500 demos) — this chapter
runs that exact checkpoint, it does NOT train its own:
  python curriculum/common/envs/pusht/gen_demos.py --episodes 500 --seed 0 --out outputs/pusht-demos --no-video
  python curriculum/phase1_imitation/ch1.1_bc/bc.py --data outputs/pusht-demos --out outputs/ch1.1-bc --device cpu --no-rerun
"""

# --- region: setup ---
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Loose script from the repo root (same pattern as ch3.3-3.5); put the root on
# sys.path so curriculum.common resolves. The engine here is pure numpy, and the
# policy runs in torch.no_grad() eval on CPU — both bitwise deterministic. The
# MuJoCo reference side is bitwise deterministic on CPU too (root CLAUDE.md #2),
# so --seed 0 run twice produces byte-identical metrics.json.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.device import banner  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv, wrap_angle  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="seeds nothing random in the sim (resets are seeded per-episode to MATCH MuJoCo); shifts the held-out eval seed block")
parser.add_argument("--policy", type=Path, default=Path("outputs/ch1.1-bc/bc_policy.ts.pt"), help="TorchScript ch1.1 BC policy (obs[10]->action[2]); if missing, a fresh seeded policy is used with a loud warning")
parser.add_argument("--episodes", type=int, default=50, help="eval episodes; each uses the SAME seed in both sims so initial states coincide")
parser.add_argument("--pusher_mass", type=float, default=4.0, help="the engine's pusher inertia — the hyperparameter that widens/closes the sim-to-sim gap (heavier = firmer push, closer to MuJoCo)")
parser.add_argument("--block_damp", type=float, default=6.0, help="viscous drag on the block (emulates MuJoCo's joint frictionloss/damping that keeps PushT quasi-static)")
parser.add_argument("--baumgarte", type=float, default=40.0, help="ch3.4 link-stabilization frequency (rad/s) holding the two-mass T-block rigid under contact impulses")
parser.add_argument("--device", choices=["cpu"], default="cpu", help="CPU only: pure-numpy engine + torch eval; the flag exists for banner/tier parity and honest wall-clock")
parser.add_argument("--smoke", action="store_true", help="fixed short hermetic run for CI (fresh seeded policy, 2 episodes, 40-step horizon); two runs must match byte-for-byte")
parser.add_argument("--out", type=Path, default=Path("outputs/ch3.6-compare"))
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)  # recording is the default; opt OUT, not in
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd recording (CI smoke)")
args = parser.parse_args()

banner("ch3.6-compare", device=args.device)  # tier + measured wall-clock, printed first
num_episodes = 2 if args.smoke else args.episodes  # smoke length is FIXED so CI can diff runs exactly
horizon = 40 if args.smoke else PushTEnv.MAX_STEPS  # cap the smoke rollout so CI is fast
args.out.mkdir(parents=True, exist_ok=True)
# --- endregion ---

# --- region: policy ---
# ch1.1's policy, reloaded and RUN — not retrained. bc.py saved a TorchScript
# copy (bc_policy.ts.pt) that carries its own code AND its own normalization
# buffers, so we load it with torch.jit.load and never need bc.py's BCPolicy
# class on the path. The contract is exactly ch1.1's: raw obs float32[10] in,
# raw action float32[2] (pusher velocity, [-1,1]) out. Nothing about the policy
# changes; only the world it acts in does.
BCPolicy = None  # only defined for the smoke fallback below (keeps torch.jit the default path)


def load_policy(path: Path) -> torch.nn.Module:
    """Load the trained ch1.1 TorchScript policy, or a fresh seeded fallback.

    A trained checkpoint is the whole point (the FULL CIRCLE). But smoke/CI is
    hermetic — no trained artifact on disk — so when the file is absent we build
    a fresh, SEEDED, untrained policy: it transfers badly in BOTH sims, which is
    fine for a determinism check (the pipeline runs, two runs match). Real
    transfer numbers require a real ch1.1 checkpoint; we say so, loudly.
    """
    if path.is_file() and not args.smoke:
        policy = torch.jit.load(str(path), map_location="cpu")
        policy.eval()
        print(f"policy: loaded trained ch1.1 checkpoint {path}")
        return policy
    if not args.smoke:
        print(f"WARNING: no policy at {path} — using a FRESH UNTRAINED policy. "
              "Transfer numbers are meaningless until you point --policy at a trained bc_policy.ts.pt.")
    # A small MLP standing in for ch1.1's policy (smoke determinism only). It is NOT
    # ch1.1's BCPolicy — no in-model normalization, a narrower net — and never needs to
    # be: smoke only checks the pipeline runs and two runs match byte-for-byte.
    global BCPolicy
    import torch.nn as nn

    class BCPolicy(nn.Module):  # noqa: F811  (smoke-only stand-in; self-contained per doctrine)
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(PushTEnv.OBS_DIM, 64), nn.ReLU(),
                nn.Linear(64, 64), nn.ReLU(),
                nn.Linear(64, PushTEnv.ACT_DIM),
            )

        def forward(self, obs):
            return torch.tanh(self.net(obs))  # keep the raw action in [-1,1]

    torch.manual_seed(args.seed)  # deterministic init -> deterministic smoke
    policy = BCPolicy().eval()
    return policy


policy = load_policy(args.policy)


def policy_action(obs: np.ndarray) -> np.ndarray:
    """One obs[10] -> action[2] step through the policy (the ch1.1 rollout call)."""
    with torch.no_grad():
        batch = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)  # (10,) -> (1,10)
        action = policy(batch)[0].cpu().numpy()
    return np.clip(action, -1.0, 1.0).astype(np.float32)
# --- endregion ---

# --- region: engine ---
# PushT, RE-CREATED in the engine you built. The T-block is TWO point masses —
# the bar center (= the body origin PushT reports as tee_xy) and the stem center
# 0.06 m below it in the body frame — held rigid by ONE ch3.4 distance link. Its
# yaw is EMERGENT: the orientation of the line between the two masses. The pusher
# is a disk (a point mass with a radius). Pushing the bar mass off the block's
# center of mass turns the whole dumbbell — that is how a frictionless normal
# contact (ch3.5) can rotate a T at all. Masses/geometry come straight from
# pusht.xml (bar 0.06 kg, stem 0.045 kg, welded; stem 0.06 m below the bar).
#
# HONEST SIMPLIFICATIONS (the sources of the sim-to-sim gap): two point masses,
# not the true two-box rigid body (approximate inertia, less rotational leverage
# at the bar tips than MuJoCo); frictionless normal contact only, planar friction
# faked as a viscous block drag; an idealized velocity-servo pusher, no continuous
# collision. None is a bug — each is a modeling choice, and their sum is the gap.

BAR_MASS, STEM_MASS = 0.06, 0.045          # pusht.xml geom masses
STEM_OFFSET = 0.06                          # body-frame distance bar-center -> stem-center
BAR_RADIUS, STEM_RADIUS = 0.032, 0.026      # disk approximations of the two boxes
PUSHER_RADIUS = 0.015                        # pusht.xml pusher cylinder radius
DT, SUBSTEPS = 0.01, 10                      # match PushTEnv: 100 Hz physics, 10 Hz control (FRAME_SKIP=10)
PUSHER_GAIN = 20.0                           # velocity-servo gain (pusht.xml actuator kv=20)


def block_yaw(p_bar: np.ndarray, p_stem: np.ndarray) -> float:
    """Recover the T's yaw from the two masses: the body-frame line bar->stem is
    (0, -L), so the world angle of (stem - bar) is yaw - pi/2. Invert it."""
    d = p_stem - p_bar
    return wrap_angle(float(np.arctan2(d[1], d[0])) + np.pi / 2.0)


def reset_engine(seed: int) -> dict:
    """Build the engine PushT state, sampling the SAME draws as PushTEnv.reset so
    the initial block/pusher pose is bit-for-bit the one MuJoCo starts from."""
    rng = np.random.Generator(np.random.PCG64(seed))  # identical RNG to pusht_env.reset
    r = rng.uniform(*PushTEnv._SPAWN_R)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    tee_xy = np.array([r * np.cos(phi), r * np.sin(phi)])
    tee_yaw = rng.uniform(-np.pi, np.pi)
    while True:  # same rejection sampling, same order -> same pusher spawn
        pusher_xy = rng.uniform(-PushTEnv._PUSHER_BOUND, PushTEnv._PUSHER_BOUND, size=2)
        if np.linalg.norm(pusher_xy - tee_xy) > PushTEnv._PUSHER_CLEAR:
            break
    p_bar = tee_xy.astype(float)
    p_stem = tee_xy + np.array([STEM_OFFSET * np.sin(tee_yaw), -STEM_OFFSET * np.cos(tee_yaw)])
    # State = 3 point masses [pusher, bar, stem]; only the two block masses have velocity to start.
    q = np.stack([pusher_xy.astype(float), p_bar, p_stem])
    v = np.zeros((3, 2))
    m = np.array([args.pusher_mass, BAR_MASS, STEM_MASS])
    return {"q": q, "v": v, "m": m, "radius": np.array([PUSHER_RADIUS, BAR_RADIUS, STEM_RADIUS]),
            "streak": 0, "success": False}


def engine_obs(state: dict) -> np.ndarray:
    """The pusht obs[10] contract, read out of the engine state (target fixed at origin)."""
    px, py = state["q"][0]
    tx, ty = state["q"][1]                       # bar center == PushT body origin
    tyaw = block_yaw(state["q"][1], state["q"][2])
    return np.array([px, py, tx, ty, np.sin(tyaw), np.cos(tyaw), 0.0, 0.0, 0.0, 1.0], dtype=np.float32)


def link_force(q: np.ndarray, v: np.ndarray, m: np.ndarray, baumgarte: float) -> np.ndarray:
    """ch3.4's distance constraint, specialized to the ONE bar<->stem link. Solve
    the scalar (J M^-1 J^T) lambda = -(J M^-1 f_ext + Jdot v) - Baumgarte for the
    Lagrange multiplier, return J^T lambda on the two block masses (bar=1, stem=2)."""
    d = q[1] - q[2]
    dv = v[1] - v[2]
    g = 0.5 * (d @ d - STEM_OFFSET ** 2)         # 0 when the link is exactly rest length
    gdot = d @ dv
    jdotv = dv @ dv
    inv_mass = 1.0 / m[1] + 1.0 / m[2]
    a = inv_mass * (d @ d)                        # J M^-1 J^T (scalar, SPD)
    b = -jdotv - (2.0 * baumgarte * gdot + baumgarte ** 2 * g)  # f_ext=0 in-plane; drag added by caller
    lam = b / a
    force = np.zeros((3, 2))
    force[1] = d * lam
    force[2] = -d * lam
    return force


# --- ch3.5 contact, reused: the body-body overlap test + solve_contacts VERBATIM.
# PushT is top-down, so there is NO floor contact (ch3.5's y=0 table is dropped);
# the only contacts are the pusher (body 0) against each block mass (bodies 1, 2).
def detect_pusher_contacts(q: np.ndarray, radius: np.ndarray) -> list[dict]:
    """ch3.5 detect_contacts' body-body branch, specialized to pusher-vs-block."""
    contacts = []
    for j in (1, 2):  # pusher (i=0) against bar (1) and stem (2)
        d = q[0] - q[j]
        dist = float(np.linalg.norm(d))
        overlap = radius[0] + radius[j] - dist
        if overlap > -1.0e-3 and dist > 1e-12:
            contacts.append({"i": 0, "j": j, "n": d / dist, "depth": overlap})
    return contacts


def _normal_velocity(v: np.ndarray, c: dict) -> float:
    """ch3.5 verbatim: relative velocity of the contact pair along the normal."""
    vn = float(v[c["i"]] @ c["n"])
    if c["j"] >= 0:
        vn -= float(v[c["j"]] @ c["n"])
    return vn


def solve_contacts(v: np.ndarray, minv: np.ndarray, contacts: list[dict],
                   restitution: float, baumgarte: float, dt: float, iters: int) -> np.ndarray:
    """ch3.5's projected Gauss-Seidel for contact impulses, copied VERBATIM."""
    vn_pre = [_normal_velocity(v, c) for c in contacts]
    lam = np.zeros(len(contacts))
    for _ in range(iters):
        for idx, c in enumerate(contacts):
            i, j, n = c["i"], c["j"], c["n"]
            k_eff = minv[i] + (minv[j] if j >= 0 else 0.0)
            target = -restitution * min(vn_pre[idx], 0.0) + (baumgarte / dt) * c["depth"]
            vn = _normal_velocity(v, c)
            dlam = (target - vn) / k_eff
            new = max(0.0, lam[idx] + dlam)
            dlam, lam[idx] = new - lam[idx], new
            v[i] += (dlam * minv[i]) * n
            if j >= 0:
                v[j] -= (dlam * minv[j]) * n
    return v


def step_engine(state: dict, action: np.ndarray) -> None:
    """One 10 Hz control step: hold `action` for SUBSTEPS physics ticks. Each tick
    assembles ch3.3 (semi-implicit) + ch3.4 (link force) + ch3.5 (contact solve)."""
    action = np.clip(action, -1.0, 1.0).astype(float)
    q, v, m, radius = state["q"], state["v"], state["m"], state["radius"]
    minv = 1.0 / m
    for _ in range(SUBSTEPS):
        # ch3.4 link force on the two block masses + viscous drag (faked planar friction).
        f = link_force(q, v, m, args.baumgarte)
        f[1] -= args.block_damp * v[1] * m[1]
        f[2] -= args.block_damp * v[2] * m[2]
        # ch3.3 semi-implicit velocity update (new velocity, then position below).
        v[1:] += DT * f[1:] * minv[1:, None]
        # pusher: velocity servo toward the commanded velocity (finite so contact can resist).
        v[0] += DT * PUSHER_GAIN * (action - v[0])
        # ch3.5 contact solve: pusher pushes the block masses (frictionless normal, restitution 0).
        contacts = detect_pusher_contacts(q, radius)
        if contacts:
            solve_contacts(v, minv, contacts, 0.0, 0.2, DT, 10)
        q += DT * v  # integrate all three masses
    # success bookkeeping, mirroring PushTEnv.step (pos+ang tolerance held SUCCESS_HOLD steps).
    tee_xy = q[1]
    pos_err = float(np.linalg.norm(tee_xy))
    ang_err = abs(block_yaw(q[1], q[2]))
    in_tol = pos_err < PushTEnv.POS_TOL and ang_err < PushTEnv.ANG_TOL
    state["streak"] = state["streak"] + 1 if in_tol else 0
    if state["streak"] >= PushTEnv.SUCCESS_HOLD:
        state["success"] = True
    state["pos_err"], state["ang_err"] = pos_err, ang_err
# --- endregion ---

# --- region: rollouts ---
# Three rollouts per episode, all from the SAME seed so both sims start identical:
#   * MuJoCo closed-loop  -> ground-truth success + the exact action sequence.
#   * engine  closed-loop -> your-engine success (the transfer number).
#   * engine  open-loop   -> replay MuJoCo's actions from the shared start and
#     measure how the block poses DIVERGE (pure dynamics gap, no policy feedback).
# Poses are (x, y, yaw) of the block; divergence is position + angle error vs MuJoCo.


def mj_pose(env: PushTEnv) -> np.ndarray:
    return env.tee_pose  # (x, y, yaw), yaw already wrapped


def run_mujoco(seed: int) -> dict:
    """BC policy closed-loop in MuJoCo. Returns success, the action trace, and the block-pose trace."""
    env = PushTEnv()
    obs = env.reset(seed=seed)
    actions, poses = [], [mj_pose(env)]
    done, success, step = False, False, 0
    while not done and step < horizon:
        action = policy_action(obs)
        actions.append(action)
        obs, _, done, info = env.step(action)
        poses.append(mj_pose(env))
        success = bool(info["success"])
        step += 1
    return {"success": success, "actions": actions, "poses": np.array(poses)}


def run_engine_closed(seed: int) -> bool:
    """BC policy closed-loop in YOUR engine (each sim visits its own states)."""
    state = reset_engine(seed)
    for _ in range(horizon):
        step_engine(state, policy_action(engine_obs(state)))
        if state["success"]:
            return True
    return False


def replay_engine(seed: int, actions: list[np.ndarray]) -> np.ndarray:
    """Open-loop: same start, MuJoCo's EXACT actions -> the engine's block-pose trace."""
    state = reset_engine(seed)
    poses = [np.array([*state["q"][1], block_yaw(state["q"][1], state["q"][2])])]
    for action in actions:
        step_engine(state, action)
        poses.append(np.array([*state["q"][1], block_yaw(state["q"][1], state["q"][2])]))
    return np.array(poses)


def pose_divergence(mj_poses: np.ndarray, eng_poses: np.ndarray) -> tuple[float, float]:
    """Mean position (m) and angle (rad) gap between the two sims' block-pose traces."""
    n = min(len(mj_poses), len(eng_poses))
    pos = np.linalg.norm(mj_poses[:n, :2] - eng_poses[:n, :2], axis=1)
    ang = np.abs([wrap_angle(a - b) for a, b in zip(mj_poses[:n, 2], eng_poses[:n, 2])])
    return float(pos.mean()), float(ang.mean())
# --- endregion ---

# --- region: report ---
# Run every episode in all three modes and aggregate. The deliverable is the
# comparison table: the BC policy's success IN MUJOCO vs IN YOUR ENGINE (the
# transfer), and the mean block-pose DIVERGENCE (the dynamics gap). Expect the
# engine success BELOW MuJoCo and the divergence to grow across the episode —
# the honest sim-to-sim gap, the same shape as ch2.6's sim-to-real gap.
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch3.6-compare", spawn=False)
    rr.save(str(args.out / "compare.rrd"))
    rr.log("world/target", rr.Points3D([[0.0, 0.0, 0.0]], radii=[0.02], colors=[[90, 205, 100]]), static=True)

mj_successes, eng_successes = 0, 0
pos_divs, ang_divs = [], []
for episode in range(num_episodes):
    seed = 10_000 + args.seed + episode  # ch1.1's held-out eval seed block (never a trained start)
    mj = run_mujoco(seed)
    eng_pose = replay_engine(seed, mj["actions"])
    pos_div, ang_div = pose_divergence(mj["poses"], eng_pose)
    mj_successes += mj["success"]
    eng_successes += run_engine_closed(seed)
    pos_divs.append(pos_div)
    ang_divs.append(ang_div)
    if args.rerun:
        rr.log(f"world/mujoco/ep{episode}", rr.LineStrips3D([[[*p[:2], 0.0] for p in mj["poses"]]], colors=[[115, 128, 242]]), static=True)
        rr.log(f"world/engine/ep{episode}", rr.LineStrips3D([[[*p[:2], 0.0] for p in eng_pose]], colors=[[217, 76, 64]]), static=True)
        rr.set_time("episode", sequence=episode)
        rr.log("divergence/position", rr.Scalars([pos_div]))
        rr.log("divergence/angle", rr.Scalars([ang_div]))

mj_rate = mj_successes / num_episodes
eng_rate = eng_successes / num_episodes
# Stale-checkpoint guard: ch1.1's canonical policy (500 demos) scores ~0.62 in MuJoCo —
# the sim it TRAINED in. If a loaded (non-smoke) checkpoint scores 0 there, it is untrained
# or STALE (e.g. a 3-epoch toy run left at the default path), NOT a sim-to-sim gap. Say
# so loudly rather than silently report a meaningless 0/0 transfer.
if not args.smoke and args.policy.is_file() and mj_rate == 0.0:
    print(f"WARNING: the loaded policy {args.policy} scored 0% even in MuJoCo — the sim it was "
          "TRAINED in (ch1.1's canonical reference is ~0.62). This checkpoint looks UNTRAINED or STALE; "
          "re-train ch1.1 to convergence and point --policy at its bc_policy.ts.pt. The sim-to-sim "
          "numbers below measure a broken policy, not your engine's dynamics gap.")
metrics = {
    "episodes": num_episodes,
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "policy": str(args.policy) if (args.policy.is_file() and not args.smoke) else "fresh-untrained",
    "mj_success_rate": round(mj_rate, 6),          # ground truth: the policy in the sim it trained in
    "engine_success_rate": round(eng_rate, 6),     # the transfer: the SAME policy in the engine you built
    "mean_pos_divergence_m": round(float(np.mean(pos_divs)), 6),   # open-loop block-position gap
    "mean_ang_divergence_rad": round(float(np.mean(ang_divs)), 6), # open-loop block-yaw gap
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

print(f"ran {num_episodes} episodes (horizon {horizon}) of the ch1.1 BC policy in BOTH sims")
print(f"{'':<22}{'MuJoCo (truth)':>16}{'your engine':>16}")
print(f"{'BC success rate':<22}{mj_rate:>16.3f}{eng_rate:>16.3f}")
print(f"open-loop divergence (same start, same actions): "
      f"position {np.mean(pos_divs):.4f} m, angle {np.mean(ang_divs):.4f} rad (mean over the episode)")
transfer = eng_rate / mj_rate if mj_rate > 0 else float("nan")
print(f"sim-to-sim transfer: your engine keeps {transfer:.0%} of MuJoCo's success — "
      "the gap is your engine's simplifications, the same shape as ch2.6's sim-to-real gap")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'compare.rrd'} — open it with: rerun {args.out / 'compare.rrd'}")
# config hash of the run knobs, for the wallclock ledger (author: paste into wallclock.csv)
_cfg = f"{args.episodes}|{args.pusher_mass}|{args.block_damp}|{args.baumgarte}|{horizon}"
print(f"config_hash: {hashlib.md5(_cfg.encode()).hexdigest()[:12]}")
# --- endregion ---
