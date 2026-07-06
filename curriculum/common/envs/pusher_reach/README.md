# Pusher-reach — Phase-2 reference environment

Drive a planar 2-link arm's fingertip onto a randomly-placed target by applying
a torque to each joint. The canonical **dense-reward** continuous-control task
(OpenAI Gym's "Reacher" lineage) and ch2.2's SAC teaching env.

Built to the same infra-grade discipline as Cartpole / PushT (reset/step/obs
API, bitwise CPU determinism, a `--seed`-able rollout demo), so ch2.1 (PPO) and
ch2.2 (SAC) reuse the same env pattern and the same continuous-action policy
head.

## Why this env for SAC (the off-policy bargain)

Cartpole (ch2.1, PPO) pays a **sparse-ish alive bonus**: `+1` per step, with no
per-step gradient telling the policy *which direction* is better — which suits
an on-policy method that learns from whole trajectories. Pusher-reach is the
opposite: a **dense** per-step signal (`-distance`) that an off-policy learner
can bootstrap on from a replay buffer full of old transitions. That dense reward
is exactly what SAC exploits, and exactly why the chapter pairs the two envs —
same continuous-torque action space, opposite reward structure.

## Files

| file | what |
|---|---|
| `pusher_reach.xml` | MJCF scene: 2-link arm (two hinge joints about +z) + `fingertip` site, one torque motor per joint, a mocap-body target, no collisions, no joint limits |
| `pusher_reach_env.py` | `PusherReachEnv` — deterministic env (reset/step/render_frame, optional rerun) + `reach_action` scripted IK+PD baseline + `--seed`-able rollout demo `main()` |

## Observation — `float32[8]` (`PusherReachEnv.OBS_DIM = 8`)

ch2.2's SAC trains on exactly this layout. Each joint angle is encoded as
`(cos, sin)` so the observation is continuous through `±π` (no wrap seam a
Gaussian policy would trip on).

| idx | name | notes |
|---|---|---|
| 0 | `cos(shoulder_angle)` | |
| 1 | `sin(shoulder_angle)` | |
| 2 | `cos(elbow_angle)` | |
| 3 | `sin(elbow_angle)` | |
| 4 | `shoulder_angvel` | rad/s |
| 5 | `elbow_angvel` | rad/s |
| 6 | `fingertip_to_target_x` | m; `target_x − fingertip_x` |
| 7 | `fingertip_to_target_y` | m; `target_y − fingertip_y` |

Entries 6–7 carry the dense signal: **their norm is the distance the reward
penalizes**. The arm-relative vector has no wrap discontinuity, so it is left raw
rather than sin/cos-encoded.

## Action — `float32[2]` (`PusherReachEnv.ACT_DIM = 2`)

Joint **torques** `[tau_shoulder, tau_elbow]`, clipped to `[-1, 1]` and applied
through the MuJoCo motors (`gear 0.5` ⇒ ±0.5 N·m), held for `FRAME_SKIP = 2`
physics steps (50 Hz control over 100 Hz physics, `timestep = 0.01 s`).

**Design note for the ch2.2 author — torque control, not position targets.**
Torque control is the classic Reacher/SAC setup and keeps the same
"policy outputs a raw actuator command" shape as ch2.1's cartpole (force). If
you want a gentler task, position/velocity actuators are a one-line MJCF +
`step` change; flagged here so torque is a deliberate choice.

## Reward and success (DENSE)

```
dist   = ||fingertip_xy − target_xy||        # m
reward = −dist                               # every step (the dense signal)
         + SUCCESS_BONUS (1.0) once, on the step success first latches
```

`info["success"]` latches when `dist < SUCCESS_TOL (0.02 m)` and stays `True`;
the `+1.0` bonus is paid once. Return is dominated by how fast the fingertip
closes on the target and how tightly it holds there — a smooth, informative
signal at *every* step, which is what off-policy value bootstrapping wants.

**Design note — pure `−dist`, minimal shaping.** A control-cost term
(`−λ‖action‖²`, as in Gym Reacher-v2) is deliberately *omitted* to keep the
reward minimal and the "return ≈ −(distance integrated over the episode)"
reading exact. If ch2.2 wants to teach the classic reach+effort trade-off, add
one line in `step()`; flagged as a deliberate choice.

## Termination vs truncation

| flag | fires when | default |
|---|---|---|
| `info["terminated"]` | `terminate_on_success=True` **and** success latched | **off** (`terminate_on_success=False`) |
| `info["truncated"]` | `step_count >= MAX_STEPS` (100 ≈ 2 s) | always ends the episode here by default |

Default is **Reacher-style: no early termination** — the episode always runs the
full horizon and the arm must *hold* at the target, not just touch it. This
gives SAC a richer dense signal (staying on target keeps paying ≈0 reward vs.
drifting away). Flip `PusherReachEnv(terminate_on_success=True)` for a
reach-and-stop variant. **Flagged for the ch2.2 author** — the "hold" default is
the more standard SAC-reacher setup, but reach-and-stop is a legitimate contrast.

`step()` returns the Cartpole/PushT 4-tuple `(obs, reward, done, info)` with
`done = terminated or truncated`; `info` also carries `dist` and `success`.

## Determinism (CI-enforced, root CLAUDE.md invariant 2)

`reset(seed)` draws the two joint angles (`uniform[-π, π)`) and the target
(uniform annulus `r ∈ [0.05, 0.19] m` around the base) via
`np.random.Generator(PCG64(seed))` — no global RNG, no wall clock; joint
velocities start at rest. Two fresh envs reset with the same seed produce
**byte-identical** observations; same seed + same action sequence produces
byte-identical trajectories. The scene has **no collisions and no joint limits**
(a pure articulated 2-body chain — no contact/constraint solver), and the target
is a **mocap body** (never in `qpos`, so it cannot perturb the arm), so the
dynamics are cheap (a full 100-step episode is microseconds) and reproducible.
Gravity is perpendicular to the plane of motion, so the arm never sags. Verified
in `test_pusher_reach_env.py` (`test_reset_bitwise_determinism`,
`test_step_determinism`).

## Baseline returns (the bar SAC must clear)

Measured on the pinned stack (mujoco 3.10.0, numpy 2.4.6), 20 episodes, seed 0:

| policy | mean final distance (m) | mean return | success |
|---|---|---|---|
| `zero` (no torque) | 0.179 ± 0.096 | −17.9 ± 9.6 | 0/20 |
| `random` (uniform `[-1, 1]`) | 0.176 ± 0.100 | −17.6 ± 9.5 | 0/20 |
| `scripted` (`reach_action`) | **0.0001 ± 0.0001** | **−2.4 ± 2.3** | **20/20** |

`reach_action(env)` solves the closed-form 2-link inverse kinematics
(elbow-down branch) and PD-controls the joints to those angles — a non-learned
reference. SAC should match or beat it. `test_scripted_beats_random` CI-locks
the reward's directionality (scripted final distance `< 0.25×` random, and
`≥9/10` reaches).

```
.venv/bin/python curriculum/common/envs/pusher_reach/pusher_reach_env.py \
    --policy scripted --episodes 20 --seed 0        # or --policy random | zero
```

## Rerun

```python
env = PusherReachEnv()
env.enable_rerun(path="episode.rrd")   # or spawn=True; off by default
```

Logs per the repo-wide conventions: `world/robot/arm` (LineStrips3D
base→elbow→fingertip), `world/robot/fingertip` and `world/objects/target`
(Points3D + per-step updates), `policy/action`, `train/{dist,success}`, timeline
`sim_time`. `import rerun` is lazy — importing the env costs nothing without it.
Pass `--rerun episode.rrd` to the demo to capture a rollout.
