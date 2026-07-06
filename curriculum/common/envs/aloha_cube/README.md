# AlohaCube — Phase-1 bimanual cube-transfer environment

An ALOHA-style **bimanual cube-transfer** task: a right end-effector picks up a
cube, carries it to the middle, hands it off to a left end-effector, which
delivers it to a target. It is the foundation environment for **chapter 1.3
(ACT)** — the mandatory mid-air handoff is exactly the temporally-coordinated
structure ACT's action chunking is designed to exploit.

Infra-grade env, same discipline and API as `../pusht/` (decision 004):
deterministic CPU MuJoCo, `reset(seed)`/`step(action)`/`render_frame`, a
scripted expert, and a local LeRobot-v3 demo generator.

## Files

| file | what |
|---|---|
| `aloha_cube.xml` | MJCF scene: table, cube on a planar joint, two planar end-effectors (each = 2 actuated slides + a 1-DOF gripper), fixed delivery target, walls, two grasp `weld` constraints |
| `aloha_cube_env.py` | `AlohaCubeEnv` — deterministic env (reset/step/render_frame, optional rerun) |
| `scripted_expert.py` | `ScriptedExpert` / `expert_action` — six-phase pick→handoff→place state machine |
| `gen_demos.py` | CLI demo-dataset generator (LeRobot v3, fully local) |
| `test_aloha_cube_env.py` | env-local determinism + handoff-contract test (runs in `make check`) |

## Observation — `float32[10]` (`AlohaCubeEnv.OBS_DIM = 10`)

World-frame meters; grip channels are closedness in `[0, 1]`.

| idx | name | notes |
|---|---|---|
| 0 | `right_ee_x` | right end-effector center |
| 1 | `right_ee_y` | |
| 2 | `right_grip` | 0 = open … 1 = fully closed |
| 3 | `left_ee_x` | left end-effector center |
| 4 | `left_ee_y` | |
| 5 | `left_grip` | 0 = open … 1 = fully closed |
| 6 | `cube_x` | transfer cube |
| 7 | `cube_y` | |
| 8 | `target_x` | fixed `-0.30` |
| 9 | `target_y` | fixed `0.00` |

There is no explicit "which arm holds the cube" flag: grasp is inferable from
gripper closedness + geometry, and ACT's real signal is the top-down image
(`--video`, see below). The target is constant but kept in the obs so a
randomized-target variant stays obs-compatible.

## Action — `float32[6]` (`AlohaCubeEnv.ACT_DIM = 6`)

`[right_vx, right_vy, right_grip, left_vx, left_vy, left_grip]`, clipped to
`[-1, 1]`.

- Velocity channels are end-effector **target velocities** [m/s] via MuJoCo
  velocity actuators.
- Grip channels command each gripper: **+1 = close, −1 = open** (mapped to the
  finger position actuators).

Control is 10 Hz over 100 Hz physics (`FRAME_SKIP = 10`, `timestep = 0.01`).

## Reward and success

```
dist   = ||cube_xy - target_xy||          # m
reward = -(dist / DIST_SCALE)             # shaped, in [-1, 0]  (DIST_SCALE = 0.7)
         + 1.0 once, on the step success latches
```

`info["success"]` latches when `dist < POS_TOL (0.04 m)` for
`SUCCESS_HOLD = 5` consecutive control steps. Episodes end at success or
`MAX_STEPS = 400` (40 s). `info` also exposes `right_held` / `left_held`.

## The task is genuinely bimanual (reach split)

Slide-joint ranges make the handoff mandatory, not optional:

- **Right arm** reaches world x ∈ `[-0.05, 0.45]`; the **cube spawns** at
  x ∈ `[0.15, 0.38]` (right-only) and the **target** sits at x = `-0.30`
  (unreachable by the right arm).
- **Left arm** reaches world x ∈ `[-0.45, 0.05]`; it can reach the target but
  **cannot reach the cube's spawn zone**.
- The reaches overlap only in `[-0.05, 0.05]` — the **handoff band**.

So a policy must pick with the right arm, hand off near the middle, and place
with the left. `test_reach_forces_handoff` pins this.

## Determinism (CI-enforced, root CLAUDE.md invariant 2)

`reset(seed)` draws the cube position and a small per-arm y-jitter from
`np.random.Generator(PCG64(seed))` — no global RNG, no wall clock. Two fresh
envs reset with the same seed produce **byte-identical** observations; same
seed + same action sequence produces byte-identical rollouts (grasp welds
toggle deterministically as a pure function of state). Verified by
`test_aloha_cube_env.py`.

## Honesty — what is simplified, and why

This is a *simplified* ALOHA that holds the free-tier + determinism floor
(invariants 1–2). Two deliberate abstractions, in the same spirit as PushT
emulating planar friction with joint `frictionloss`:

1. **Planar end-effectors, not 14-DOF arms.** Each "arm" is an end-effector on
   two velocity-actuated slides plus a 1-DOF gripper (6-D action total). No
   links, no IK — the policy commands the end-effector directly, like PushT's
   pusher. This keeps CPU rollouts fast (median ~27 control steps/episode) and
   trainable on a T4 / CPU laptop, and sidesteps the contact-rich, harder-to-
   reproduce dynamics of a full ViperX model.
2. **Grasp is a `weld` constraint, not frictional pinch.** When a gripper is
   closed and within `GRASP_R = 0.035 m` of the cube, the env welds the cube to
   that gripper (snapping the cube to the gripper center so the weld starts
   satisfied); opening releases it. This is CPU-deterministic and robust across
   seeds. It is **not** frictional grasping — a slipping-contact grasp model is
   a Scale Lab / later-chapter topic, not a free-tier default.

Consequences to teach honestly: episodes are short, the "grasp" cannot fail
mid-carry, and the cube does not rotate (planar slide-x/slide-y only). The
coordination lesson (pick → handoff → place, with a split reach) is real; the
manipulation physics is intentionally tame.

The MuJoCo Menagerie ALOHA model exists but pulls a 14-DOF bimanual ViperX
setup whose contact-rich grasping is neither free-tier-cheap nor as cleanly
CPU-reproducible; this authored scene is the honest free-tier substitute for
teaching ACT. Upgrading to the Menagerie model would be a pinned
`upstream-pin-check` + Scale Lab, not a feature PR.

## Scripted expert — measured success

Six-phase state machine: `R_APPROACH → R_GRASP → R_CARRY → HANDOFF → RELEASE →
L_CARRY` (proportional pursuit toward waypoints; welds latched via
`env.right_held` / `env.left_held`).

Measured on this pinned stack (mujoco 3.10.0, numpy 2.4.6):

| setting | seeds | success |
|---|---|---|
| `noise = 0.0` | 0..49 | **50/50 (100%)** |
| `noise = 0.05` | 0..49 | 50/50 (100%) |

Episode length (noiseless): median 27 control steps, min 26, max 30.

CI-enforced (fast lane, every PR): ≥8/10 on seeds 0..9
(`test_expert_transfers_and_succeeds`). The ≥45/50 bar on seeds 0..49
(`test_expert_success_full`) is a `-m slow` test — run it locally; it moves
into CI once a `-m slow` lane exists (same status as PushT).

## Demo datasets

```
.venv/bin/python curriculum/common/envs/aloha_cube/gen_demos.py \
    --episodes 50 --seed 0 --out /path/to/dataset [--noise 0.05] [--video]
```

Writes a **LeRobot dataset v3.0** via the pinned `lerobot` package
(`LeRobotDataset.create(...)` + `add_frame`/`save_episode`/`finalize`), fully
local — no hub, no network. Features: `observation.state` (float32[10]),
`action` (float32[6]) + auto `timestamp`/`frame_index`/`episode_index`/`index`/
`task_index`; `--video` adds `observation.image` as a 96×96 top-down mp4
(needed for ACT's image encoder). Episode `i` uses seed `--seed + i` for reset
and expert noise, so a dataset is bit-for-bit reproducible from its CLI
arguments. **No dataset is committed to git** — regenerate locally or publish to
the HF Hub `zero2robot/` org.

## Rerun

```python
env = AlohaCubeEnv()
env.enable_rerun(path="episode.rrd")   # or spawn=True; off by default
```

Logs per the repo-wide conventions: `world/objects/{cube,target}`,
`world/robot/{right_ee,left_ee}` (Boxes3D/Cylinders3D + per-step Transform3D),
`policy/action`, `train/{success,dist,right_held,left_held}`, timeline
`sim_time`. `import rerun` is lazy — importing the env costs nothing without it.
