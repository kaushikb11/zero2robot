"""zero2robot 3.9 — Plan Through Your Engine: Sampling-Based MPC (CEM / MPPI).

Every controller you have built so far LEARNED. BC (ch1.1) cloned a demonstrator;
PPO (ch2.1) and SAC (ch2.2) learned a policy by trial and error over thousands of
episodes. All of them turn experience into a network of weights, then read an
action out of it. This chapter throws the network away.

You spent chapters 3.3-3.6 opening `mj_step` — dynamics, joints as constraints,
contact you solve for — until the simulator stopped being a black box and became
an ENGINE you understand. An engine does one thing: given a state and an action,
it tells you the next state. That is exactly what you need to PLAN. So instead of
learning what to do, we SEARCH for it, live, at every control step:

    1. from the current state, SAMPLE N candidate action sequences over a horizon H
    2. ROLL each one forward through the model (the engine) and SCORE it by a cost
    3. move the sampled distribution toward the good ones, and repeat a few times
    4. execute the first action of the best plan, step the real world one tick,
       and re-plan from the state you actually landed in (RECEDING horizon)

That is Model Predictive Control. Two update rules for step 3 live in this file and
they are ~15 lines apart: CEM refits a Gaussian to the ELITE fraction; MPPI takes a
softmax-WEIGHTED average of every sample. Same loop, one idea differs — read the diff.

The task is cartpole SWING-UP: the pole starts hanging straight DOWN, and the only
actuator pushes the cart sideways. Underactuated — you cannot torque the pole up
directly; you must pump it. No demonstrator, no reward signal to learn from, no
training run. Just a model and a search. It swings up anyway.

Run it:      python curriculum/phase3_advanced/ch3.9_mpc/mpc.py --seed 0
Try MPPI:    python .../mpc.py --seed 0 --method mppi
Break it:    python .../mpc.py --seed 0 --break horizon   (too short to plan the pump)
CI smoke:    python .../mpc.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch2.1 / ch3.6).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.device import banner  # noqa: E402
from curriculum.common.envs.cartpole import CartpoleEnv, wrap_angle  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="seeds the sampling RNG and the start jitter; CPU MuJoCo + numpy are bitwise deterministic, so --seed 0 run twice matches byte-for-byte")
parser.add_argument("--method", choices=("cem", "mppi"), default="cem", help="the step-3 update: cem (refit a Gaussian to the elites) or mppi (softmax-weighted average of all samples)")
parser.add_argument("--samples", type=int, default=64, help="N action sequences sampled per re-plan; cpu: 64 | more compute: 256. --break samples drops it to 3 (too few to find the pump)")
parser.add_argument("--horizon", type=int, default=25, help="H control steps each plan looks ahead (0.5 s at 50 Hz). --break horizon drops it to 3 (too short to see the swing pay off)")
parser.add_argument("--iters", type=int, default=2, help="refinement iterations per re-plan (sample -> score -> update, repeated)")
parser.add_argument("--steps", type=int, default=120, help="control steps the real episode runs (2.4 s at 50 Hz)")
parser.add_argument("--elite_frac", type=float, default=0.1, help="CEM: fraction of samples kept as the elite set the next Gaussian is fit to")
parser.add_argument("--init_std", type=float, default=0.9, help="std of the action-noise the planner samples with (both methods)")
parser.add_argument("--temperature", type=float, default=0.3, help="MPPI: softmax temperature; low = trust the best samples more, high = average more broadly")
parser.add_argument("--break", dest="break_mode", choices=("horizon", "samples"), default=None,
                    help="Break It: cripple the plan. 'horizon' makes it too short-sighted to see the swing pay off; 'samples' gives it too few tries to find the pump")
parser.add_argument("--device", choices=("cpu",), default="cpu", help="CPU only: the model rollouts are mj_step on tiny cartpole, microsecond-cheap; the flag exists for banner/tier parity")
parser.add_argument("--smoke", action="store_true", help="tiny hermetic run for CI (8 samples, horizon 5, 10 steps, both cheap); two runs must match byte-for-byte")
parser.add_argument("--out", type=Path, default=Path("outputs/ch3.9-mpc"))
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)  # recording is the default; opt OUT, not in
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd recording (CI smoke)")
args = parser.parse_args()

banner("ch3.9-mpc", device=args.device)  # tier + measured wall-clock, printed first
if args.break_mode == "horizon":
    args.horizon = 3   # too short: the plan cannot see the swing come back around to upright
if args.break_mode == "samples":
    args.samples = 3   # too few: the search almost never stumbles on an energy-pumping sequence
if args.smoke:  # pin everything the CI byte-compare depends on; keep it tiny AND deterministic
    args.samples, args.horizon, args.iters, args.steps = 8, 5, 1, 10
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.Generator(np.random.PCG64(args.seed))  # the ONE rng: start jitter + all sampling
# --- endregion ---

# --- region: model ---
# The MODEL you plan through. MPC's whole premise is that you HAVE one: a thing
# that maps (state, action) -> next state, that you can fast-forward and rewind at
# will. Here it is a second copy of the cartpole simulator — the same mj_step you
# took apart in ch3.3-3.6. We keep the "real world" (env) and the "model" (sim)
# as SEPARATE objects on purpose: the planner may only touch `sim`; the true state
# only ever advances through `env.step`. Here model == world (a PERFECT model);
# ch3.6 already measured what happens when they differ — hold that thought for the
# ceiling at the end of the chapter.
env = CartpoleEnv()   # the real world: we read its state and apply one action per tick
sim = CartpoleEnv()   # the imagined model: the planner rolls candidate plans through THIS
CONTROL_SUBSTEPS = CartpoleEnv.FRAME_SKIP  # each control step = this many mj_step physics ticks (50 Hz control)
SLIDER = sim.model.joint("slider").qposadr[0]  # cart position index in qpos/qvel
HINGE = sim.model.joint("hinge").qposadr[0]    # pole angle index (0 = upright, pi = hanging down)


def get_state(e: CartpoleEnv) -> tuple[np.ndarray, np.ndarray]:
    """Snapshot the full dynamical state (qpos, qvel) — everything mj_step needs to continue."""
    return e.data.qpos.copy(), e.data.qvel.copy()


def set_state(e: CartpoleEnv, qpos: np.ndarray, qvel: np.ndarray) -> None:
    """Rewind the model to a saved state. mj_forward refreshes derived quantities
    (site positions, etc.) so the next mj_step continues from EXACTLY here."""
    e.data.qpos[:] = qpos
    e.data.qvel[:] = qvel
    e.data.ctrl[:] = 0.0
    mujoco.mj_forward(e.model, e.data)


def step_model(e: CartpoleEnv, action: float) -> None:
    """Advance the model one control step: hold `action` for CONTROL_SUBSTEPS ticks."""
    e.data.ctrl[0] = float(np.clip(action, -1.0, 1.0))
    for _ in range(CONTROL_SUBSTEPS):
        mujoco.mj_step(e.model, e.data)


def state_cost(e: CartpoleEnv, action: float) -> float:
    """The TASK, written down as a cost to minimize — there is no learning here, you
    just SAY what 'good' means. Swing-up wants the pole UPRIGHT, the cart near center,
    and the motion SETTLED so it does not whirl straight through the top. This one
    function scores both the imagined rollouts (the planner) and the realized step."""
    theta = wrap_angle(float(e.data.qpos[HINGE]))       # angle from upright; 0 up, +-pi down
    x = float(e.data.qpos[SLIDER])                       # cart offset from rail center
    return ((1.0 - np.cos(theta))                        # 0 upright, 2 hanging down (the dominant term)
            + 0.1 * x * x                                # keep the cart near center
            + 0.01 * float(e.data.qvel[HINGE]) ** 2      # settle the pole (don't spin through the top)
            + 0.005 * float(e.data.qvel[SLIDER]) ** 2    # settle the cart
            + 0.001 * float(action) ** 2)                # mild effort penalty


def upright_cos(e: CartpoleEnv) -> float:
    """cos(angle-from-upright): +1 exactly up, -1 hanging down. The honest 'is it up?' readout."""
    return float(np.cos(wrap_angle(float(e.data.qpos[HINGE]))))
# --- endregion ---

# --- region: planner ---
# The one function MPC turns on. Given the current true state and a warm-started
# mean plan, it samples N action sequences, rolls each through the MODEL, scores
# them, and refines the sampling distribution `iters` times. CEM and MPPI differ
# ONLY in how the scores update the mean (the two branches below) — everything
# around them is identical. That is the "same file, one idea differs" of the course.


def rollout_cost(q0: np.ndarray, v0: np.ndarray, seq: np.ndarray) -> float:
    """Total cost of one candidate action sequence, IMAGINED through the model from
    (q0, v0). We REWIND the model to the shared start, then step the whole horizon,
    accumulating state_cost — this is the single call MPC makes N x H times a step."""
    set_state(sim, q0, v0)
    total = 0.0
    for action in seq:
        step_model(sim, action)
        total += state_cost(sim, float(action))
    return total


def plan(q0: np.ndarray, v0: np.ndarray, mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (refined mean plan, the sampled first-actions) for ONE control step.
    `mean` is warm-started from last step's plan, shifted one tick — most of the
    thinking carries over, so a couple of iterations is enough."""
    mean = mean.copy()
    std = np.full(args.horizon, args.init_std)
    first_samples = None
    for _ in range(args.iters):
        # sample N sequences around the current mean, clipped to the actuator range
        noise = rng.standard_normal((args.samples, args.horizon))
        samples = np.clip(mean[None, :] + std[None, :] * noise, -1.0, 1.0)  # (N, H)
        costs = np.array([rollout_cost(q0, v0, seq) for seq in samples])    # (N,) score each plan
        if first_samples is None:
            first_samples = samples[:, 0].copy()  # kept only to visualize the fan-out
        if args.method == "cem":
            # CEM: keep the ELITE fraction (the lowest-cost plans) and refit the
            # Gaussian to them. The mean marches toward the elites; the std shrinks
            # as they agree, so the search focuses in on the good region.
            n_elite = max(1, int(args.elite_frac * args.samples))
            elite = samples[np.argsort(costs)[:n_elite]]
            mean = elite.mean(axis=0)
            std = elite.std(axis=0) + 0.05  # floor keeps it from collapsing to zero
        else:  # mppi
            # MPPI: no hard cutoff. Weight EVERY sample by exp(-cost / temperature)
            # (a softmax over negative cost) and take the weighted average. Good
            # plans dominate smoothly; nothing is thrown away.
            weights = np.exp(-(costs - costs.min()) / args.temperature)
            weights = weights / weights.sum()
            mean = (weights[:, None] * samples).sum(axis=0)
    return mean, first_samples
# --- endregion ---

# --- region: control ---
# The receding-horizon loop. Start the pole hanging DOWN (the swing-up problem),
# then at every tick: plan from the state we ACTUALLY landed in, execute only the
# first action, and shift the plan forward to warm-start the next one. Planning
# re-runs every step precisely because the model is never perfect and the world
# drifts — you re-decide from where you really are, not where you meant to be.


def start_hanging(e: CartpoleEnv) -> None:
    """Reset, then place the pole straight down with a tiny seeded jitter (so seeds
    differ) and the cart near center at rest — the swing-up initial condition."""
    e.reset(seed=args.seed)
    qpos, qvel = get_state(e)
    qpos[HINGE] = np.pi + rng.uniform(-0.05, 0.05)   # hanging down (+-pi from upright)
    qpos[SLIDER] = rng.uniform(-0.05, 0.05)
    qvel[:] = 0.0
    set_state(e, qpos, qvel)


def run_mpc(log: bool) -> dict:
    """One MPC episode. Returns the executed trace + the headline metrics."""
    start_hanging(env)
    mean = np.zeros(args.horizon)  # cold start: the first plan is centered on 'do nothing'
    cost_sum, cos_trace, fan = 0.0, [], []
    for t in range(args.steps):
        q0, v0 = get_state(env)             # plan from the TRUE current state
        mean, first_samples = plan(q0, v0, mean)
        action = float(mean[0])             # execute only the first action of the plan
        step_model(env, action)             # the real world advances one tick
        cost_sum += state_cost(env, action)
        cos_trace.append(upright_cos(env))
        if t < 40:                          # keep an early slice of the sampled fan-out for the demo
            fan.append({"step": t, "first_actions": [round(a, 4) for a in first_samples.tolist()],
                        "chosen": round(action, 4), "cos": round(cos_trace[-1], 4)})
        if log:                             # the swing-up curve: cos climbs from -1 (down) to +1 (up)
            cart_x = float(env.data.qpos[SLIDER])
            rr.set_time("control_step", sequence=t)
            rr.log("world/robot/cart", rr.Transform3D(translation=(cart_x, 0.0, 0.0)))
            rr.log("world/objects/pole", rr.Transform3D(translation=(cart_x, 0.0, 0.0),
                   rotation=rr.RotationAxisAngle(axis=(0, 1, 0), radians=float(env.data.qpos[HINGE]))))
            rr.log("mpc/upright_cos", rr.Scalars([cos_trace[-1]]))
            rr.log("mpc/step_cost", rr.Scalars([state_cost(env, action)]))
            rr.log("mpc/action", rr.Scalars([action]))
        mean = np.concatenate([mean[1:], [0.0]])  # receding horizon: shift the plan, pad a fresh tail
    return summarize(cost_sum, cos_trace) | {"fan": fan}


def run_random() -> dict:
    """The no-plan baseline: apply uniform-random actions (no model, no search).
    It flails — the honest floor MPC has to clear to earn the 'planning works' claim."""
    start_hanging(env)
    cost_sum, cos_trace = 0.0, []
    for _ in range(args.steps):
        action = float(rng.uniform(-1.0, 1.0))
        step_model(env, action)
        cost_sum += state_cost(env, action)
        cos_trace.append(upright_cos(env))
    return summarize(cost_sum, cos_trace)


def summarize(cost_sum: float, cos_trace: list[float]) -> dict:
    """Mean per-step cost (the smooth number) + upright fraction over the settle
    window (did it END up balanced? — the last quarter of the episode)."""
    settle = np.array(cos_trace[-max(1, args.steps // 4):])
    return {"mean_cost": round(cost_sum / max(1, len(cos_trace)), 6),
            "upright_frac": round(float((settle > 0.9).mean()), 6)}  # >0.9 == within ~25 deg of straight up
# --- endregion ---

# --- region: report ---
# Run MPC, then the random baseline from the SAME start, and lay them side by
# side. The headline is a MECHANISM claim: with a model and a search — and no
# learning at all — MPC swings the pole up and holds it (upright_frac -> 1.0, cost
# far below the flailing baseline). --break horizon / --break samples cripple the
# plan and the pole never comes up (upright_frac -> 0.0): planning needs to look
# far enough ahead, with enough tries, to find the pump. Both are seed-robust.
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch3.9-mpc", spawn=False)
    rr.save(str(args.out / "mpc.rrd"))
    rr.log("world/objects/pole", rr.Capsules3D(lengths=[1.0], radii=[0.02], colors=(204, 89, 76)), static=True)

mpc = run_mpc(log=args.rerun)     # the real episode, logging the swing-up curve to rerun
baseline = run_random()           # the no-plan floor, from the same start

metrics = {
    "break_mode": args.break_mode,
    "horizon": args.horizon,
    "iters": args.iters,
    "method": args.method,
    "mpc_mean_cost": mpc["mean_cost"],
    "mpc_upright_frac": mpc["upright_frac"],
    "random_mean_cost": baseline["mean_cost"],
    "random_upright_frac": baseline["upright_frac"],
    "samples": args.samples,
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "steps": args.steps,
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
(args.out / "fan.json").write_text(json.dumps(mpc["fan"]) + "\n")  # sampled fan-out for the demo/vizdata

label = f"{args.method.upper()}" + (f" [--break {args.break_mode}]" if args.break_mode else "")
print(f"task: cartpole SWING-UP (pole starts DOWN), {args.steps} control steps, NO learning")
print(f"{'':<22}{'mean cost':>12}{'upright frac':>14}")
print(f"{label:<22}{mpc['mean_cost']:>12.3f}{mpc['upright_frac']:>14.2f}")
print(f"{'random (no plan)':<22}{baseline['mean_cost']:>12.3f}{baseline['upright_frac']:>14.2f}")
if args.break_mode is None:
    print(f"planning through a model beat the no-plan baseline with ZERO training — "
          f"{args.method.upper()} sampled {args.samples} plans/step, looked {args.horizon} steps ahead")
else:
    print(f"crippled plan (--break {args.break_mode}): the pole never comes up (upright {mpc['upright_frac']:.2f}) — "
          "too short-sighted / too few tries to find the swing")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'mpc.rrd'} — open it with: rerun {args.out / 'mpc.rrd'}")
# --- endregion ---
