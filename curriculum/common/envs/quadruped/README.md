# Quadruped — Phase-2 locomotion reference environment

Make a minimal 4-legged robot **stand**, then **walk forward**. A box torso on a
floating (free-joint) base carries four identical two-joint legs — a hip hinge +
a knee hinge, **8 actuated joints** — each ending in a foot that touches a
friction floor under gravity. The policy commands the 8 joints; a good policy
holds the torso up at height and drives it forward in +x.

This is the locomotion env for **ch2.4 (Reward Design)** — the reward below is
built to be shaped and hacked — **ch2.5 (Locomotion: The Quadruped Walks)** — the
task a policy is trained on — and **ch2.7 (Domain Randomization)** — the base env
that gets perturbed. It is the third infra-grade Phase-2 env, built to the same
discipline as cartpole (ch2.1) and pusher-reach (ch2.2): identical
`reset`/`step`/`obs` API, the same continuous-action Gaussian-policy shape, and
the same bitwise-CPU-determinism guarantee — so the RL chapters reuse the pattern
unchanged.

## This is a from-scratch quadruped, NOT mujoco_menagerie

`quadruped.xml` is a deliberately tiny **hand-written** robot: a box torso and
four 2-DOF capsule legs, ~14 geoms total. It is **not** a Go1/ANYmal import. A
menagerie model would need a decision-log dependency entry (root CLAUDE.md) and
is far heavier than the free-tier teaching env needs. A from-scratch quadruped is
in the doctrine's from-scratch spirit, stays microsecond-cheap, and — crucially —
stays **bitwise-deterministic on CPU even with ground contact**. The honest cost:
the gait dynamics are cartoonish (a sagittal-plane 2-DOF leg, no ab/adduction, no
real foot), so this teaches *locomotion RL structure* — reward shaping,
stand→walk, sim-to-sim robustness — not a transferable gait for real hardware
(that is a Scale Lab).

## Files

| file | what |
|---|---|
| `quadruped.xml` | MJCF scene: box torso on a free joint + 4 two-joint legs (8 hinge joints), foot spheres + a friction floor, one PD position servo per joint. Only the four foot-floor pairs collide. |
| `quadruped_env.py` | `QuadrupedEnv` — deterministic env (reset/step/render_frame, optional rerun) + `stand_action` / `trot_action` scripted baselines + `--seed`-able rollout demo `main()` |

## Observation — `float32[23]` (`QuadrupedEnv.OBS_DIM = 23`)

ch2.5's policy trains on exactly this layout. Frame convention: **x forward,
y left, z up**.

| idx | name | units / notes |
|---|---|---|
| 0..7 | joint angles | rad; order `FL_hip, FL_knee, FR_hip, FR_knee, HL_hip, HL_knee, HR_hip, HR_knee` |
| 8..15 | joint velocities | rad/s; same joint order |
| 16 | torso height | m; world z of the torso center |
| 17..19 | torso up-vector | world-frame torso body z-axis; `(0,0,1)` = perfectly upright. **Entry 19 alone is the "uprightness"** the reward and fall-check read. |
| 20..22 | torso linear velocity | m/s, world frame; **entry 20 is the forward velocity `vx`** |

The up-vector (not a raw quaternion) encodes orientation with no sign/wrap seam a
Gaussian policy would trip on, and its z-component is directly the term the reward
and termination read.

## Action — `float32[8]` (`QuadrupedEnv.ACT_DIM = 8`)

A **target-angle offset** for each joint around the nominal standing pose,
clipped to `[-1, 1]` and scaled by `ACTION_SCALE = 0.5 rad`. The env commands
`DEFAULT_POSE + ACTION_SCALE * action` to the 8 **PD position servos** (`kp 20`,
`kv 0.5` in the MJCF), held for `FRAME_SKIP = 4` physics steps (50 Hz control over
200 Hz physics, `timestep = 0.005 s`). Same joint order as the observation.

**Design note for the ch2.4/2.5 author — position (residual) control, not
torque.** `action = residual around a default crouch` is the standard legged-RL
setup: it makes a hand-scripted stand/trot stable and gives a learned policy an
easy "do nothing = stand" anchor. Torque control (à la pusher-reach) is a
one-line actuator swap — replace the `<position>` actuators with `<motor>` and
feed `ctrl` raw — flagged here so position control is a deliberate choice. The
`kp/kv`, `ACTION_SCALE`, `FRAME_SKIP`, and control rate are the knobs to reach
for if a chapter wants a harder task.

## Reward — five named, **shapeable** terms (the ch2.4 contract)

`step()` returns each term separately in `info["reward_terms"]`, and the total is
their sum, so ch2.4 can reweight or ablate them one at a time:

```
forward = W_FORWARD * clip(vx, -MAX_VX, MAX_VX)     # reward moving forward (+x)
upright = W_UPRIGHT * up_z                          # reward the torso staying level
height  = -W_HEIGHT * (height - TARGET_HEIGHT)^2    # penalize crouching / bouncing
alive   = W_ALIVE                                   # per-step "not fallen" bonus
ctrl    = -W_CTRL * sum(action^2)                   # penalize large joint commands
reward  = forward + upright + height + alive + ctrl
```

| term | weight (default) | what it drives |
|---|---|---|
| `forward` | `W_FORWARD = 1.0` | forward progress; `vx` clipped to `MAX_VX = 1.0 m/s` so a lunge can't be farmed |
| `upright` | `W_UPRIGHT = 0.2` | keep the torso z-axis pointing up |
| `height` | `W_HEIGHT = 5.0` | hold ride height near `TARGET_HEIGHT = 0.25 m` |
| `alive` | `W_ALIVE = 0.2` | a per-step bonus for not having fallen |
| `ctrl` | `W_CTRL = 0.001` | small penalty on `‖action‖²` (don't thrash) |

The defaults make **standing already positive** (`alive + upright` dominate,
`≈ 0.4`/step) and **walking strictly better** (the `forward` term stacks on top).
That is exactly the surface ch2.4 explores: which weights make a walk *emerge*,
and which get **hacked** — e.g. crank `W_FORWARD` and a policy learns to dive
forward and faceplant (high `vx` for a few frames, then it falls), or drop
`W_HEIGHT` and it belly-crawls. **Flagged for the ch2.4 author:** these weights
are the teaching payload, tuned only enough that the scripted baselines behave;
treat them as the starting point to break, not as sacred.

## Termination vs truncation

| flag | fires when | meaning |
|---|---|---|
| `info["terminated"]` | `torso height < FALL_HEIGHT (0.14 m)` **or** `up_z < UPRIGHT_MIN (0.4)` | the robot fell / flipped — a real failure |
| `info["truncated"]` | `step_count >= MAX_STEPS (500 ≈ 10 s)` still upright | a time limit, **not** a failure — bootstrap the value here |

`step()` returns the cartpole/pusher-reach 4-tuple `(obs, reward, done, info)`
with `done = terminated or truncated`; `info` also carries `reward_terms`,
`height`, `up_z`, and `forward_vel` for logging. **Flagged for the ch2.5 author:**
`FALL_HEIGHT` / `UPRIGHT_MIN` are the termination thresholds — the crouch rides at
~0.25 m and up_z ~1.0, so these leave wide margin; tighten them for a stricter
"must stay tall/level" task.

## Determinism, WITH ground contact (CI-enforced, root CLAUDE.md invariant 2)

`reset(seed)` places the torso at `STAND_HEIGHT = 0.257 m` (feet just touching),
upright and at rest, and sets the 8 leg joints to `DEFAULT_POSE` plus small
seeded joint-angle noise (`uniform[-0.05, 0.05] rad`) via
`np.random.Generator(PCG64(seed))` — no global RNG, no wall clock. Two fresh envs
reset with the same seed produce **byte-identical** observations; same seed +
same action sequence produces byte-identical trajectories.

Unlike cartpole/pusher-reach (which avoid contact entirely), this env **has**
ground contact — the whole point of locomotion — so it pins what a contact solve
needs to stay reproducible (see `quadruped.xml`):

- `integrator="implicitfast"`, `solver="Newton"` with **fixed** `iterations="50"`
  / `ls_iterations="20"` (no tolerance-early-exit variability), `cone="pyramidal"`
  — a fully specified, single-threaded contact solve.
- explicit foot **friction** (`1.0 0.02 0.001`) and contact softness
  (`solref="0.01 1"`, `solimp="0.9 0.95 0.001"`) on the foot default, so behaviour
  never rides on library defaults.
- the **only** collidable pairs are the four foot-floor contacts (torso and leg
  links carry `contype/conaffinity 0`); a fall is caught by the height/orientation
  termination, not by the torso hitting the floor.

MuJoCo's CPU `mj_step` is single-threaded and branch-deterministic, so with the
solver fully specified two runs are bitwise-identical, contacts and all. Verified
twice-run in `test_quadruped_env.py` (`test_step_determinism_contact_rich`,
`test_scripted_trot_determinism`) — a full 500-step trot episode spends **490/500
steps in foot-floor contact** and reproduces to the byte. As always, this is the
**bitwise-CPU** tier; GPU/MJX training is only the statistical tier (see
`curriculum/common/seeding.py`).

## Baseline returns (the bar a learned policy must clear)

Measured on the pinned stack (mujoco 3.10.0, numpy 2.4.6), 20 episodes, seed 0:

| policy | mean episode length | mean forward distance | mean return |
|---|---|---|---|
| `random` (uniform `[-1, 1]`) | 400.9 ± 125.8 | −0.30 ± 0.20 m | 142.7 ± 50.5 |
| `stand` (`stand_action`, zero residual) | **500.0 ± 0.0** (full horizon) | −0.01 ± 0.00 m | 199.4 ± 0.2 |
| `trot` (`trot_action`, open-loop gait) | **500.0 ± 0.0** (full horizon) | **+2.14 ± 0.15 m** | **306.0 ± 7.3** |

Reading: the reward rewards **staying up** (both scripted policies ride out the
full horizon while random falls ~100 steps early, and even a standing robot
out-returns random 199 vs 143) **and moving forward** (the trot's `forward` term
lifts its return to 306, well above a mere stand). `test_scripted_beats_random`
CI-locks this: stand never falls and beats random on length, and the trot walks
`> 1 m` and `> random + 1 m` forward.

- `stand_action(env)` returns a zero residual — the PD servos hold the nominal
  crouch. The trivial "just stand" reference; a learned policy must at least match
  it before it earns the forward term.
- `trot_action(env)` is an **open-loop** diagonal trot (pairs `{FL, HR}` and
  `{FR, HL}` in antiphase; hips sweep fore/aft, knees flex on the swing half). No
  feedback, not learned — yet it both stays up and walks. `hip_amp / knee_amp /
  freq` are keyword args if a chapter wants a faster/slower reference gait.

```
.venv/bin/python curriculum/common/envs/quadruped/quadruped_env.py \
    --policy trot --episodes 20 --seed 0        # or --policy stand | random
```

## Rerun

```python
env = QuadrupedEnv()
env.enable_rerun(path="episode.rrd")   # or spawn=True; off by default
```

Logs per the repo-wide conventions: `world/robot/torso` (Boxes3D + per-step
Transform3D), `world/robot/feet` (Points3D — the contact points), `policy/action`,
`train/{forward_vel,height,up_z}`, timeline `sim_time`. `import rerun` is lazy —
importing the env costs nothing without it. Pass `--rerun episode.rrd` to the demo
to capture a rollout.

## Design choices flagged for the chapter authors

- **Leg DOF (2/leg = hip + knee, sagittal plane).** No ab/adduction, so the robot
  can't strafe or recover sideways — it walks in a straight line. Minimal by
  design; a third hip DOF is a bigger MJCF change and a harder task.
- **Position (residual) control vs torque.** Position chosen for a stable
  scripted baseline and an easy stand anchor; torque is a one-line actuator swap.
- **Reward-term weights.** Tuned only enough that the baselines behave — they are
  ch2.4's payload to reshape/hack, not fixed values.
- **Termination thresholds** (`FALL_HEIGHT = 0.14`, `UPRIGHT_MIN = 0.4`) leave
  wide margin around the ~0.25 m / up_z~1.0 crouch; tighten for a stricter task.
- **Contact-determinism settings** (`solver=Newton`, fixed iteration counts,
  `cone=pyramidal`, explicit foot friction/solref/solimp, feet-only collision) —
  ch2.7 should keep these when randomizing; randomize friction/mass/etc. **on top
  of** this pinned solver, and the bitwise guarantee still holds per fixed seed.
