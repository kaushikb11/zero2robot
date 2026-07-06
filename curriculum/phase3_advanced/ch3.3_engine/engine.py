"""zero2robot 3.3 — Build a Physics Engine I: Unconstrained Dynamics.

Since chapter 0.1 you have trusted `mj_step`: write ctrl, call the black box,
read the new state. This chapter opens the box. We build the unconstrained
rigid-body update from scratch in numpy — no MuJoCo for the dynamics — and
confront the one honesty every physics engine has to make peace with:
INTEGRATORS ACCUMULATE ERROR, and ENERGY DRIFT is how you measure it.

Three integrators, one conservative system (a body in orbit / on a spring
where total energy should never change), and a number that tells the truth:
- explicit EULER gains energy every step and the orbit spirals outward,
- SEMI-IMPLICIT (symplectic) Euler keeps energy BOUNDED — it wobbles around
  the true value forever but never runs away,
- RK4 is far more ACCURATE per step but is NOT symplectic — its tiny error
  still creeps one direction over long runs.
Every MuJoCo integrator choice flows from this measured trade. Chapter 3.4
adds constraints on top of exactly this state + step interface.

Run it:      python engine.py                 # all 3 integrators on the default orbit
             python engine.py --system spring --integrator all
CI smoke:    python engine.py --smoke --seed 0 --no-rerun   # two runs byte-identical
"""

# --- region: setup ---
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch0.1). This chapter
# is pure numpy — no torch, no mujoco — so the whole run is bitwise deterministic
# on CPU: same seed, same bytes, every time (the exercises lean on that).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.device import banner  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="seeds the initial-condition jitter (orbit eccentricity, spring stretch)")
parser.add_argument("--system", choices=["orbit", "spring", "freefall"], default="orbit", help="which conservative system to integrate")
parser.add_argument("--integrator", choices=["all", "euler", "semi_implicit", "rk4"], default="all", help="'all' runs the three-way comparison — the whole point")
parser.add_argument("--dt", type=float, default=0.01, help="timestep in sim seconds; raise it and every integrator's drift grows")
parser.add_argument("--steps", type=int, default=2000)  # any laptop: microseconds
parser.add_argument("--smoke", action="store_true", help="fixed 600-step run for CI; two runs must match byte-for-byte")
parser.add_argument("--device", choices=["cpu"], default="cpu", help="this engine is pure-numpy CPU; the flag exists for banner/tier parity")
parser.add_argument("--out", type=Path, default=Path("outputs/ch3.3-engine"))
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)  # recording is the default; opt OUT, not in
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd recording (CI smoke)")
args = parser.parse_args()

banner("ch3.3-engine", device=args.device)  # tier + measured wall-clock, printed first
num_steps = 600 if args.smoke else args.steps  # smoke length is FIXED so CI can diff runs exactly
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)  # PCG64 — the only source of randomness in this file
# --- endregion ---

# --- region: systems ---
# A "system" is the physics: an initial state (position q, velocity v, mass m),
# a FORCE law force(q, v) -> np.ndarray, and the true total ENERGY energy(q, v)
# that a perfect integrator would hold constant forever. Positions and
# velocities are 3-vectors so a planar orbit reads as an (x, y) curve in rerun
# and a body could later swing in 3D. Chapter 3.4 keeps this exact (q, v, m,
# force) shape and adds a constraint force term — nothing here has to change.


def build_system(name: str, rng: np.random.Generator) -> dict:
    """Return {q0, v0, m, force, energy} for a conservative test system.

    The seed jitters ONE initial-condition knob per system (orbit shape, spring
    stretch, launch angle) so `--seed` is load-bearing while the physics — and
    therefore the energy-drift ordering — stays the same across seeds.
    """
    if name == "orbit":
        # A unit point mass orbiting a fixed attractor at the origin under an
        # inverse-square pull (Newtonian gravity, GM = mu = 1). Circular when
        # speed == 1; the seed nudges the speed so the true orbit is a closed
        # ellipse — the honest baseline the spiral of Euler will violate.
        mu, m = 1.0, 1.0
        speed = 1.0 + rng.uniform(-0.05, 0.05)
        q0 = np.array([1.0, 0.0, 0.0])
        v0 = np.array([0.0, speed, 0.0])

        def force(q, v):
            return -mu * m * q / (np.linalg.norm(q) ** 3)  # points at the attractor, ~1/r^2

        def energy(q, v):
            return 0.5 * m * (v @ v) - mu * m / np.linalg.norm(q)  # kinetic + gravitational potential

    elif name == "spring":
        # A mass on an ideal (Hooke) spring: force pulls straight back toward
        # the origin, strength proportional to stretch. Undamped, so it should
        # oscillate at fixed amplitude — total energy dead flat — forever.
        k, m = 4.0, 1.0  # angular frequency sqrt(k/m) = 2 rad/s
        stretch = 1.0 + rng.uniform(-0.1, 0.1)
        q0 = np.array([stretch, 0.0, 0.0])
        v0 = np.array([0.0, 0.0, 0.0])

        def force(q, v):
            return -k * q  # Hooke's law: restoring, linear in displacement

        def energy(q, v):
            return 0.5 * m * (v @ v) + 0.5 * k * (q @ q)  # kinetic + spring potential

    else:  # freefall — a non-oscillatory sanity check (see the honest caveat below)
        # A body launched under constant gravity. Energy is conserved too, but
        # the motion never RETURNS, so the symplectic advantage does not show:
        # every integrator's energy error here grows linearly and one-signed.
        # That contrast is the lesson — drift is about BOUNDED systems.
        g, m = 9.81, 1.0
        vx = 3.0 + rng.uniform(-0.5, 0.5)
        q0 = np.array([0.0, 0.0, 10.0])
        v0 = np.array([vx, 0.0, 0.0])
        gravity = np.array([0.0, 0.0, -g])

        def force(q, v):
            return m * gravity  # constant downward pull

        def energy(q, v):
            return 0.5 * m * (v @ v) + m * g * q[2]  # kinetic + height potential

    return {"q0": q0, "v0": v0, "m": m, "force": force, "energy": energy}
# --- endregion ---

# --- region: integrators ---
# Each integrator advances (q, v) by one timestep dt given the force law and
# mass. They differ only in WHICH state they evaluate the force/velocity at —
# and that one difference is the whole difference between an orbit that holds
# and an orbit that flies apart. All three are one screen of numpy.


def euler_step(q, v, m, force, dt):
    """Explicit (forward) Euler: evaluate everything at the OLD state.

    q_{n+1} = q_n + dt * v_n ;  v_{n+1} = v_n + dt * a_n. Simplest possible
    update — and it feeds energy into a bounded system every single step.
    """
    a = force(q, v) / m
    return q + dt * v, v + dt * a


def semi_implicit_step(q, v, m, force, dt):
    """Semi-implicit (symplectic) Euler: update velocity FIRST, then step
    position with the NEW velocity. One line reordered from explicit Euler,
    and the payoff is bounded energy — this is (a cousin of) what MuJoCo uses.
    """
    a = force(q, v) / m
    v_next = v + dt * a
    q_next = q + dt * v_next  # <-- the whole trick: NEW velocity, not old
    return q_next, v_next


def rk4_step(q, v, m, force, dt):
    """Classical 4th-order Runge-Kutta on the first-order system
    (q, v)' = (v, force(q, v) / m). Four force samples per step, weighted
    1-2-2-1: far more accurate per step than Euler, but NOT symplectic — its
    small error still accumulates one direction over long horizons.
    """
    def deriv(q, v):
        return v, force(q, v) / m

    k1q, k1v = deriv(q, v)
    k2q, k2v = deriv(q + 0.5 * dt * k1q, v + 0.5 * dt * k1v)
    k3q, k3v = deriv(q + 0.5 * dt * k2q, v + 0.5 * dt * k2v)
    k4q, k4v = deriv(q + dt * k3q, v + dt * k3v)
    q_next = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return q_next, v_next


INTEGRATORS = {"euler": euler_step, "semi_implicit": semi_implicit_step, "rk4": rk4_step}
COLORS = {"euler": [217, 76, 64], "semi_implicit": [76, 175, 80], "rk4": [64, 115, 217]}
# --- endregion ---

# --- region: simulate ---
# Run one integrator over the whole horizon and record, at every step, WHERE
# the body is (the trajectory) and HOW MUCH total energy it has (the honesty
# metric). Pure function of its inputs — no rerun, no globals — so the smoke
# run is bitwise reproducible and the exercises can call it directly.


def simulate(step_fn, system: dict, dt: float, steps: int):
    """Integrate `system` for `steps` steps. Returns (trajectory, energies).

    trajectory: (steps + 1, 3) positions; energies: (steps + 1,) total energy.
    """
    q = system["q0"].astype(float).copy()
    v = system["v0"].astype(float).copy()
    m, force, energy = system["m"], system["force"], system["energy"]

    trajectory = np.empty((steps + 1, 3))
    energies = np.empty(steps + 1)
    trajectory[0], energies[0] = q, energy(q, v)
    for i in range(steps):
        q, v = step_fn(q, v, m, force, dt)
        trajectory[i + 1], energies[i + 1] = q, energy(q, v)
    return trajectory, energies


def energy_drift(energies: np.ndarray) -> dict:
    """Relative energy error vs the starting energy — the drift numbers.

    rel_final: signed drift at the end (Euler: large +, semi-implicit: ~0).
    rel_max:   worst absolute excursion over the whole run (bounded for a
               symplectic integrator, unbounded for explicit Euler).
    """
    e0 = energies[0]
    scale = abs(e0) if e0 != 0.0 else 1.0
    return {
        "e0": float(e0),
        "e_final": float(energies[-1]),
        "rel_final": float((energies[-1] - e0) / scale),
        "rel_max": float(np.max(np.abs(energies - e0)) / scale),
    }


def log_rerun(name: str, trajectory: np.ndarray, energies: np.ndarray) -> None:
    """Draw the path (static) and stream energy-vs-step for one integrator."""
    rr.log(f"world/{name}", rr.LineStrips3D([trajectory], colors=[COLORS[name]]), static=True)
    for i, e in enumerate(energies):
        rr.set_time("step", sequence=i)
        rr.log(f"energy/{name}", rr.Scalars([float(e)]))  # overlaid drift curves, one per integrator
# --- endregion ---

# --- region: report ---
# Run the chosen integrator(s), print the drift table, and write metrics.json.
# The comparison is the deliverable: three energy numbers that say, in order,
# "runs away", "holds", "holds best" — the measured reason engines pick one.
names = list(INTEGRATORS) if args.integrator == "all" else [args.integrator]
system = build_system(args.system, rng)

if args.rerun:
    rr.init("zero2robot/ch3.3-engine", spawn=False)
    rr.save(str(args.out / "engine.rrd"))
    rr.log("world/attractor", rr.Points3D([[0.0, 0.0, 0.0]], radii=[0.03]), static=True)  # the orbit's focus

drifts: dict[str, dict] = {}
for name in names:
    trajectory, energies = simulate(INTEGRATORS[name], system, args.dt, num_steps)
    drifts[name] = energy_drift(energies)
    drifts[name]["final_pos"] = [round(float(x), 6) for x in trajectory[-1]]
    if args.rerun:
        log_rerun(name, trajectory, energies)

metrics = {
    "system": args.system,
    "dt": args.dt,
    "steps": num_steps,
    "seed": args.seed,
    # 12 decimals, not fewer: RK4's drift is ~1e-11, and rounding it to zero
    # would erase the very signal the dt-order exercise measures.
    "drift": {k: {kk: round(vv, 12) if isinstance(vv, float) else vv for kk, vv in v.items()} for k, v in drifts.items()},
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

sim_seconds = num_steps * args.dt
print(f"integrated {args.system} for {num_steps} steps of {args.dt} s -> {sim_seconds:.2f} s of sim time")
print(f"{'integrator':<14}{'E0':>12}{'E_final':>14}{'rel_final':>14}{'rel_max':>14}")
for name in names:
    d = drifts[name]
    print(f"{name:<14}{d['e0']:>12.4f}{d['e_final']:>14.4f}{d['rel_final']:>+14.4e}{d['rel_max']:>14.4e}")
if args.integrator == "all" and args.system in ("orbit", "spring"):
    # The headline, stated as an ordering so it survives any seed on any tier.
    ok = abs(drifts["euler"]["rel_final"]) > abs(drifts["semi_implicit"]["rel_max"]) > drifts["rk4"]["rel_max"]
    verdict = "as expected" if ok else "UNEXPECTED — investigate"
    print(f"energy drift ordering  euler >> semi_implicit(bounded) > rk4:  {verdict}")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'engine.rrd'} — open it with: rerun {args.out / 'engine.rrd'}")
# --- endregion ---
