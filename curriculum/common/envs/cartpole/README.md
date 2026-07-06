# Cartpole — Phase-2 reference environment

Balance a hinged pole upright by pushing a cart along a rail. The classic
control task and ch2.1's PPO smoke env (CleanRL philosophy: the first thing you
run PPO on is CartPole). Underactuated — one actuator, two degrees of freedom:
you can only move the cart, never the pole directly.

Built to the same infra-grade discipline as PushT / AlohaCube (reset/step/obs
API, bitwise CPU determinism), so ch2.1 (PPO) and later ch2.2 (SAC) reuse the
same env pattern.

## Files

| file | what |
|---|---|
| `cartpole.xml` | MJCF scene: cart on a slide joint (the rail) + pole on a hinge, single force actuator, no collisions |
| `cartpole_env.py` | `CartpoleEnv` — deterministic env (reset/step/render_frame, optional rerun) + `balance_action` scripted baseline + `--seed`-able rollout demo `main()` |

## Observation — `float32[5]` (`CartpoleEnv.OBS_DIM = 5`)

ch2.1's PPO trains on exactly this layout. The pole angle is encoded as
`(cos, sin)` of the angle from upright, so the observation is continuous through
the vertical (no ±π wrap seam a Gaussian policy would trip on).

| idx | name | notes |
|---|---|---|
| 0 | `cart_pos` | m; `0` = rail center, `+` toward `+x` |
| 1 | `cart_vel` | m/s |
| 2 | `cos(pole_angle)` | `+1.0` exactly upright |
| 3 | `sin(pole_angle)` | `0.0` upright; `+` when the tip leans toward `+x` |
| 4 | `pole_angvel` | rad/s |

## Action — `float32[1]` (`CartpoleEnv.ACT_DIM = 1`)

A **continuous** horizontal force on the cart, clipped to `[-1, 1]` and applied
through the MuJoCo motor (`gear 10` ⇒ ±10 N), held for `FRAME_SKIP = 2` physics
steps (50 Hz control over 100 Hz physics, `timestep = 0.01 s`).

**Design note for the ch2.1 author — continuous, not discrete.** CartPole is
canonically *discrete* (push-left / push-right), but this env is continuous to
suit PPO's Gaussian policy head, which is the same head the pusher-reach task
(the chapter's second env) and later continuous-control chapters use. Keeping the
action space consistent means the PPO artifact's policy/critic code carries over
unchanged from cartpole to pusher-reach — no discrete-vs-continuous branch to
teach twice. If you *want* the discrete classic for a Categorical-policy contrast,
that is a one-line actuator/`step` change; flagged here so it is a deliberate choice.

## Reward and termination

```
reward = +1.0 for every step survived   # classic CartPole "alive bonus"
```

Episode **return equals the number of steps survived** (capped at
`MAX_STEPS = 500`). Balancing longer is the only way to score higher, so the
reward directly rewards balancing — no shaping to tune, low variance, and the
baseline is trivially interpretable (return = seconds-times-50 upright).

**Design note — alive bonus, not a `cos(angle)` dense reward.** Both were on the
table. The alive bonus is the canonical CartPole-v1 signal, it makes
"return == episode length" exact (so the random-vs-learned gap is obvious in one
number), and it forces PPO to actually learn the value-bootstrap / advantage
machinery the chapter teaches (a dense per-step `cos` reward is so easy it hides
whether the algorithm works). If a later section wants a smoother curriculum, a
`cos(pole_angle)` dense reward is a one-line change in `step()`; flagged as a
deliberate choice.

`step()` returns the PushT/AlohaCube 4-tuple `(obs, reward, done, info)` with
`done = terminated or truncated`. `info` carries the two flags **separately** —
PPO must not bootstrap across a real termination but *must* bootstrap across a
time-limit truncation:

| flag | fires when | PPO handling |
|---|---|---|
| `info["terminated"]` | pole fell (`|pole_angle| > 0.2095 rad ≈ 12°`) **or** cart off rail (`|cart_pos| > 2.4 m`) | real failure — value target is just the reward, no bootstrap |
| `info["truncated"]` | `step_count >= MAX_STEPS` (500), pole still up | time limit — bootstrap the critic's value of the final obs |

`info` also carries `pole_angle` and `cart_pos` for logging.

## Determinism (CI-enforced, root CLAUDE.md invariant 2)

`reset(seed)` draws all four state variables (`cart_pos`, `cart_vel`,
`pole_angle`, `pole_angvel`) i.i.d. from `uniform[-0.05, 0.05]` via
`np.random.Generator(PCG64(seed))` — no global RNG, no wall clock. Two fresh
envs reset with the same seed produce **byte-identical** observations; same seed
+ same action sequence produces byte-identical trajectories. The scene has **no
collisions** (the pole never touches anything, the cart is confined to its slide
joint), so the dynamics are a pure inverted pendulum with no contact solver —
cheap (a full 500-step episode is microseconds) and reproducible. Verified in
`test_cartpole_env.py` (`test_reset_bitwise_determinism`, `test_step_determinism`).

## Baseline returns (the bar PPO must clear)

Measured on the pinned stack (mujoco 3.10.0, numpy 2.4.6), 20 episodes, seed 0:

| policy | mean return (= episode length) | notes |
|---|---|---|
| `zero` (no force, free-fall) | ~37.5 | the pole just falls |
| `random` (uniform `[-1, 1]`) | ~32.6 ± 14.0 | the floor PPO must beat |
| `scripted` (`balance_action`) | **500.0 ± 0.0** | linear feedback balancer, rides out the full horizon on every seed |

`balance_action(env)` is a textbook linear pole balancer (positive gain on angle
+ angle-rate, gentle pull back to center). It is a non-learned reference — PPO
should match or beat it. The `test_scripted_beats_random` test asserts the
scripted balancer averages `>5×` random and `>400` steps, so the reward's
directionality is CI-locked.

```
.venv/bin/python curriculum/common/envs/cartpole/cartpole_env.py \
    --policy scripted --episodes 20 --seed 0        # or --policy random | zero
```

## Rerun

```python
env = CartpoleEnv()
env.enable_rerun(path="episode.rrd")   # or spawn=True; off by default
```

Logs per the repo-wide conventions: `world/robot/cart`, `world/objects/pole`
(Boxes3D/Capsules3D + per-step Transform3D), `policy/action`,
`train/{pole_angle,cart_pos}`, timeline `sim_time`. `import rerun` is lazy —
importing the env costs nothing without it. Pass `--rerun episode.rrd` to the
demo to capture a rollout.
