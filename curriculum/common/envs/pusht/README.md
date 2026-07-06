# PushT — Phase-0 reference environment

Push a T-shaped block to a fixed target pose (position + orientation) with a
velocity-controlled cylindrical pusher. Shared by chapters 0.4/1.1/1.2, the
playground, and the grader (decision 004). Learners rebuild `pusht.xml` from
scratch in ch0.2 — this copy is the reference.

## Files

| file | what |
|---|---|
| `pusht.xml` | MJCF scene: table, T-block (2 welded boxes on slide-slide-hinge), pusher (cylinder on 2 actuated slides), visual-only target T, walls |
| `pusht_env.py` | `PushTEnv` — deterministic env (reset/step/render_frame, optional rerun) |
| `scripted_expert.py` | `ScriptedExpert` / `expert_action` — waypoint push controller |
| `gen_demos.py` | CLI demo-dataset generator (LeRobot v3, fully local) |

## Observation — `float32[10]` (`PushTEnv.OBS_DIM = 10`)

Chapter 1.1's BC trains on exactly this layout. Everything is in world-frame
meters / radians.

| idx | name | notes |
|---|---|---|
| 0 | `pusher_x` | pusher center |
| 1 | `pusher_y` | |
| 2 | `tee_x` | block body origin = center of the T's bar |
| 3 | `tee_y` | |
| 4 | `sin(tee_yaw)` | yaw encoded as sin/cos (no wrap discontinuity) |
| 5 | `cos(tee_yaw)` | |
| 6 | `target_x` | fixed `0.0` |
| 7 | `target_y` | fixed `0.0` |
| 8 | `sin(target_yaw)` | fixed `0.0` (target yaw = 0) |
| 9 | `cos(target_yaw)` | fixed `1.0` |

The target pose is constant in this phase but kept in the obs so downstream
policies have the full task state (and so a randomized-goal variant is an
obs-compatible change).

## Action — `float32[2]` (`PushTEnv.ACT_DIM = 2`)

Pusher **target velocity** `[vx, vy]` in m/s, clipped to `[-1, 1]`, applied
through MuJoCo velocity actuators and held for `FRAME_SKIP = 10` physics steps
(10 Hz control over 100 Hz physics, `timestep = 0.01`).

## Reward and success

```
pos_err = ||tee_xy - target_xy||          # m
ang_err = |wrap(tee_yaw - target_yaw)|    # rad
reward  = -0.5 * (pos_err / 0.5 + ang_err / pi)   # shaped, in [-1, 0]
          + 1.0 once, on the step success latches
```

`info["success"]` latches when `pos_err < POS_TOL (0.03 m)` **and**
`ang_err < ANG_TOL (0.20 rad)` for `SUCCESS_HOLD = 5` consecutive control
steps. Episodes end at success or `MAX_STEPS = 300` (30 s of sim time).

## Determinism (CI-enforced, root CLAUDE.md invariant 2)

`reset(seed)` draws the block pose (annulus around the target) and pusher
position (rejection-sampled clear of the block) from
`np.random.Generator(PCG64(seed))`. Two fresh envs reset with the same seed
produce **byte-identical** observations; same seed + same action sequence
produces byte-identical rollouts. The expert's optional exploration noise is
seeded the same way. Physics detail that keeps things quasi-static and
CPU-stable: the block never touches the table plane — planar friction is
emulated with joint `frictionloss` + damping (see comments in `pusht.xml`).

## Scripted expert — measured success

Two-phase state machine (approach behind a contact point, then straight push
strokes). Translation strokes push through the block's center of mass toward
the target with a yaw-proportional lateral offset (steering while pushing);
rotation strokes push tangentially at a bar tip.

Measured with `MAX_STEPS = 300` on this pinned stack (mujoco 3.10.0,
numpy 2.4.6):

| setting | seeds | success |
|---|---|---|
| `noise = 0.0` | 0..49 | **50/50 (100%)** |
| `noise = 0.08` | 0..49 | 50/50 (100%) |

Median episode length (noiseless): ~60 control steps; max observed: 194.

CI-enforced (fast lane, every PR): ≥7/10 on seeds 0..9
(`test_expert_success`). The ≥40/50 bar on seeds 0..49
(`test_expert_success_full`) is a `-m slow` test — run it locally/manually; no
scheduled slow CI lane runs it yet (it moves into CI once a `-m slow` lane exists).

## Demo datasets

```
.venv/bin/python curriculum/common/envs/pusht/gen_demos.py \
    --episodes 50 --seed 0 --out /path/to/dataset [--noise 0.08] [--video]
```

Writes a **LeRobot dataset v3.0** via the pinned `lerobot` package
(`LeRobotDataset.create(...)` + `add_frame`/`save_episode`/`finalize`),
fully local — no hub, no network. Features: `observation.state` (float32[10]),
`action` (float32[2]) + auto `timestamp`/`frame_index`/`episode_index`/
`index`/`task_index`; `--video` adds `observation.image` as 96×96 top-down
mp4 (AV1). Episode `i` uses seed `--seed + i` for reset and expert noise, so
a dataset is bit-for-bit reproducible from its CLI arguments.

Golden fixture: `tests/envs/fixtures/pusht_mini` (2 episodes, no video, 68 KB)
= `--episodes 2 --seed 0 --no-video`. `tests/envs/test_dataset_golden.py`
regenerates it and requires exact value equality — it pins the dataset schema
and the physics at once. The HF reference dataset `lerobot/pusht` remains the
learner-facing default in chapter prose (decision 004).

## Rerun

```python
env = PushTEnv()
env.enable_rerun(path="episode.rrd")   # or spawn=True; off by default
```

Logs per the repo-wide conventions: `world/objects/tee`, `world/objects/target`,
`world/robot/pusher` (Boxes3D/Cylinders3D + per-step Transform3D),
`policy/action`, `train/{success,pos_err,ang_err}`, timeline `sim_time`.
`import rerun` is lazy — importing the env costs nothing without it.
