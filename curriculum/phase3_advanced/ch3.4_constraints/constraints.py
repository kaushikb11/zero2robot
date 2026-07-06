"""zero2robot 3.4 — Build a Physics Engine II: Joints & Constraints.

Chapter 3.3 built the unconstrained update: state (position q, velocity v,
mass m), a force law force(q, v), three integrators, and ENERGY DRIFT as the
honesty metric. Bodies fell freely. A JOINT is the opposite of free: a pendulum
bob may only move on a circle a fixed distance L from its pivot; a chain of bobs
is a chain of such distance locks. That "may only" is a CONSTRAINT, g(q) = 0,
and a constraint is not a force you write down — it is a force you SOLVE FOR.

We keep ch3.3's (q, v, m, force) shape and its integrator registry verbatim. The
new idea is one function: given the constraints, solve the linear system
    J M^-1 J^T lambda = -(Jdot v + external)        (+ stabilization)
for the Lagrange multipliers lambda, and hand J^T lambda back to the integrator
as an ADDED force term. Constraint = added force. That is the whole seam.

Then the honesty metric, parallel to ch3.3's energy drift: a naive constraint
DRIFTS — the pendulum's length grows numerically, the bob creeps off its circle.
Baumgarte STABILIZATION feeds the violation back as a restoring term and holds
it. We MEASURE the length error with/without stabilization on a pendulum and on
the chaotic double pendulum. The drift you just fought by hand is exactly why
MuJoCo's constraints are SOFT and stabilized — chapter 3.5 meets that head-on.

Run it:      python constraints.py                       # double pendulum, none vs baumgarte
             python constraints.py --system triple --integrator rk4
CI smoke:    python constraints.py --smoke --seed 0 --no-rerun   # two runs byte-identical

The double pendulum is CHAOTIC: tiny changes in the seed diverge fast. It is
still BITWISE deterministic given the seed (pure numpy) — the exercises lean on
that. What is seed-robust is the ORDERING (naive drifts >> stabilized holds),
not the exact trajectory.
"""

# --- region: setup ---
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr

# Loose script from the repo root (same pattern as ch3.3); put the root on
# sys.path so curriculum.common resolves. Pure numpy — no torch, no mujoco for
# the dynamics — so every run is bitwise reproducible on CPU: same seed, same
# bytes. np.linalg.solve is deterministic for the same inputs on one machine.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.device import banner  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="seeds the initial-angle jitter; the chain is chaotic so this diverges fast (still bitwise-deterministic)")
parser.add_argument("--system", choices=["pendulum", "double", "triple"], default="double", help="a 1-, 2-, or 3-link chain of distance constraints")
parser.add_argument("--integrator", choices=["euler", "semi_implicit", "rk4"], default="semi_implicit", help="ch3.3's integrators, reused verbatim; semi-implicit is the sane default for constraints")
parser.add_argument("--stabilization", choices=["all", "none", "baumgarte"], default="all", help="'all' runs the none-vs-baumgarte comparison — the whole point")
parser.add_argument("--baumgarte", type=float, default=20.0, help="Baumgarte frequency omega (rad/s); the feedback is 2*omega*gdot + omega^2*g, critically damped")
parser.add_argument("--dt", type=float, default=0.005, help="timestep in sim seconds; the constrained system is stiff, so smaller than ch3.3")
parser.add_argument("--steps", type=int, default=4000)  # any laptop: milliseconds
parser.add_argument("--smoke", action="store_true", help="fixed 800-step run for CI; two runs must match byte-for-byte")
parser.add_argument("--device", choices=["cpu"], default="cpu", help="pure-numpy CPU; the flag exists for banner/tier parity")
parser.add_argument("--out", type=Path, default=Path("outputs/ch3.4-constraints"))
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)  # recording is the default; opt OUT, not in
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd recording (CI smoke)")
args = parser.parse_args()

banner("ch3.4-constraints", device=args.device)  # tier + measured wall-clock, printed first
num_steps = 800 if args.smoke else args.steps  # smoke length is FIXED so CI can diff runs exactly
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)  # PCG64 — the only source of randomness in this file
# --- endregion ---

# --- region: systems ---
# A "system" is now a CHAIN of point masses linked by distance constraints. We
# keep ch3.3's (q, v, m) shape and generalize it to N particles: q and v are
# (N, 3) stacks of positions/velocities, m is (N, 1) so force / m broadcasts per
# particle exactly as ch3.3's scalar divide did. The new fields are the
# constraint TOPOLOGY: `pairs` (which particle is tied to which; the pivot end
# is a fixed anchor, encoded as partner -1) and `lengths` (each link's rest
# length L). Motion lives in the x-y plane (gravity along -y) so a swinging
# chain reads as an (x, y) curve in rerun, same as ch3.3's planar orbit.

PIVOT = np.array([0.0, 0.0, 0.0])  # the fixed anchor the first link hangs from
GRAVITY = np.array([0.0, -9.81, 0.0])


def build_system(name: str, rng: np.random.Generator) -> dict:
    """Return {q0, v0, m, pairs, lengths} for an n-link pendulum chain.

    Links start stretched along +x (max gravitational torque — the dramatic
    drop) with the whole chain tilted by a small seeded angle. The seed jitters
    ONE knob (that launch angle) so --seed is load-bearing; because the chain is
    chaotic the trajectory diverges across seeds, but the drift ORDERING does
    not (naive drifts, stabilized holds — checked seeds 0/1/2).
    """
    n_links = {"pendulum": 1, "double": 2, "triple": 3}[name]
    tilt = rng.uniform(-0.05, 0.05)  # radians off horizontal; the only randomness
    direction = np.array([np.cos(-tilt), np.sin(-tilt), 0.0])  # slightly below +x

    length = 1.0  # every link has rest length 1
    q0 = np.array([PIVOT + (i + 1) * length * direction for i in range(n_links)])
    v0 = np.zeros((n_links, 3))  # released from rest
    m = np.ones((n_links, 1))  # unit point masses
    # Link k ties particle k to its predecessor; particle 0's predecessor is the
    # fixed pivot, encoded as partner index -1.
    pairs = [(i, i - 1) for i in range(n_links)]  # (i, -1) for the first link
    lengths = np.full(n_links, length)
    return {"q0": q0, "v0": v0, "m": m, "pairs": pairs, "lengths": lengths}


def external_force(system: dict, q: np.ndarray) -> np.ndarray:
    """Gravity on every particle: m * g. Shape (N, 3), same as ch3.3's force."""
    return system["m"] * GRAVITY  # (N,1) * (3,) -> (N,3)
# --- endregion ---

# --- region: constraints ---
# THE new idea. A distance constraint reads g_k(q) = 1/2 (|d_k|^2 - L_k^2) = 0,
# where d_k is the vector along link k (bob minus pivot, or bob minus bob). Its
# Jacobian J_k = dg_k/dq is just d_k on the moving end(s). Differentiate the
# constraint twice — gdot = J v, gddot = J a + Jdot v — set gddot = 0, and the
# unknown constraint force J^T lambda drops out of a small linear system:
#     (J M^-1 J^T) lambda = -(J M^-1 f_ext + Jdot v)
# Solve for the Lagrange multipliers lambda, apply J^T lambda as an added force.
# For a distance constraint Jdot v works out to |d_dot|^2 = |v_i - v_j|^2.


def constraint_terms(system: dict, q: np.ndarray, v: np.ndarray):
    """Assemble J and the per-constraint g, gdot, and Jdot v for the current state.

    J is (C, 3N) — one row per constraint, the link vector written into the
    moving particle's 3 columns (and minus it into its partner's, if not the
    pivot). g is the position error, gdot = J v the velocity error, jdotv the
    velocity-squared term that makes gddot = J a + Jdot v exact.
    """
    pairs, lengths = system["pairs"], system["lengths"]
    n_particles, n_con = q.shape[0], len(pairs)
    jac = np.zeros((n_con, 3 * n_particles))
    g = np.zeros(n_con)
    gdot = np.zeros(n_con)
    jdotv = np.zeros(n_con)
    for k, (i, j) in enumerate(pairs):
        if j < 0:  # first link: partner is the fixed pivot (zero velocity)
            d, dv = q[i] - PIVOT, v[i]
            jac[k, 3 * i:3 * i + 3] = d
        else:  # link between two moving bobs
            d, dv = q[i] - q[j], v[i] - v[j]
            jac[k, 3 * i:3 * i + 3] = d
            jac[k, 3 * j:3 * j + 3] = -d
        g[k] = 0.5 * (d @ d - lengths[k] ** 2)  # 0 when the link is exactly length L
        gdot[k] = d @ dv                          # = J v for this row
        jdotv[k] = dv @ dv                        # Jdot v = |d_dot|^2
    return jac, g, gdot, jdotv


def constraint_force(system: dict, q: np.ndarray, v: np.ndarray, baumgarte: float) -> np.ndarray:
    """Solve for the Lagrange multipliers and return the constraint force (N, 3).

    baumgarte = 0 is the NAIVE solve (enforce gddot = 0 and hope g stays 0). A
    positive omega replaces gddot = 0 with the critically-damped target
    gddot + 2*omega*gdot + omega^2*g = 0, feeding the drift back as a restoring
    term — the whole difference between a length that creeps and one that holds.
    """
    jac, g, gdot, jdotv = constraint_terms(system, q, v)
    minv = np.repeat(1.0 / system["m"].reshape(-1), 3)  # diag(M^-1) as a (3N,) vector
    f_ext = external_force(system, q).reshape(-1)       # (3N,)
    a_mat = jac @ (minv[:, None] * jac.T)               # J M^-1 J^T, (C, C), SPD
    b = -(jac @ (minv * f_ext) + jdotv)                 # naive right-hand side
    if baumgarte > 0.0:
        b -= 2.0 * baumgarte * gdot + baumgarte ** 2 * g  # Baumgarte feedback
    lam = np.linalg.solve(a_mat, b)                     # the Lagrange multipliers
    return (jac.T @ lam).reshape(q.shape)               # J^T lambda, back to (N, 3)


def make_force(system: dict, baumgarte: float):
    """Bundle gravity + the solved constraint force into ch3.3's force(q, v) shape.

    THIS is the seam: the integrator sees one force law; inside it, the
    constraint is an added term solved on the fly. Nothing in the integrators
    changes from ch3.3.
    """
    def force(q, v):
        return external_force(system, q) + constraint_force(system, q, v, baumgarte)
    return force
# --- endregion ---

# --- region: integrators ---
# ch3.3's three integrators, reused VERBATIM (the doctrine's "repetition is the
# lesson"): they advance (q, v) by dt given any force law and mass, and never
# knew or cared that `force` now hides a linear solve. q, v are (N, 3); m is
# (N, 1); force returns (N, 3); every operation below broadcasts per particle.


def euler_step(q, v, m, force, dt):
    """Explicit (forward) Euler: everything evaluated at the OLD state."""
    a = force(q, v) / m
    return q + dt * v, v + dt * a


def semi_implicit_step(q, v, m, force, dt):
    """Semi-implicit (symplectic) Euler: new velocity first, then step position
    with it. ch3.3's bounded-energy default; the sane base for constraints too.
    """
    a = force(q, v) / m
    v_next = v + dt * a
    q_next = q + dt * v_next  # NEW velocity, not old
    return q_next, v_next


def rk4_step(q, v, m, force, dt):
    """Classical RK4 on (q, v)' = (v, force / m). Four force samples per step —
    which here means four constraint solves per step, all handled by `force`.
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
COLORS = {"none": [217, 76, 64], "baumgarte": [76, 175, 80]}  # red drifts, green holds
# --- endregion ---

# --- region: simulate ---
# Run one (integrator, stabilization) choice over the whole horizon and record,
# at every step: WHERE the last bob is (the trajectory), how badly the links are
# STRETCHED (the constraint violation — the honesty metric, parallel to ch3.3's
# energy), and the total ENERGY (which a good integrator keeps bounded even for
# the chaotic double pendulum). Pure function of its inputs — bitwise reproducible.


def energy(system: dict, q: np.ndarray, v: np.ndarray) -> float:
    """Total mechanical energy: kinetic + gravitational potential (U = -m g.q).

    Ideal constraint forces do NO work (they act along the link, perpendicular
    to allowed motion), so a perfect solve conserves this. Drift here is a
    second, independent honesty check on the constraint force.
    """
    kinetic = 0.5 * float(np.sum(system["m"] * v * v))
    potential = -float(np.sum(system["m"] * (q @ GRAVITY)[:, None]))
    return kinetic + potential


def constraint_violation(system: dict, q: np.ndarray) -> float:
    """Worst link-length error over the chain: max_k | |d_k| - L_k |.

    This is the number the whole chapter turns on. Zero means every link is
    exactly its rest length; a growing value is the pendulum literally coming
    apart in the arithmetic.
    """
    worst = 0.0
    for k, (i, j) in enumerate(system["pairs"]):
        d = q[i] - (PIVOT if j < 0 else q[j])
        worst = max(worst, abs(float(np.linalg.norm(d)) - system["lengths"][k]))
    return worst


def simulate(step_fn, system: dict, force, dt: float, steps: int):
    """Integrate for `steps`. Returns (tip_trajectory, violations, energies).

    tip_trajectory: (steps + 1, 3) — the LAST bob's path (what you watch swing).
    """
    q = system["q0"].astype(float).copy()
    v = system["v0"].astype(float).copy()
    m = system["m"]

    tip = np.empty((steps + 1, 3))
    violations = np.empty(steps + 1)
    energies = np.empty(steps + 1)
    tip[0], violations[0], energies[0] = q[-1], constraint_violation(system, q), energy(system, q, v)
    for i in range(steps):
        q, v = step_fn(q, v, m, force, dt)
        tip[i + 1] = q[-1]
        violations[i + 1] = constraint_violation(system, q)
        energies[i + 1] = energy(system, q, v)
    return tip, violations, energies


def log_rerun(name: str, tip: np.ndarray, violations: np.ndarray, energies: np.ndarray) -> None:
    """Draw the tip's path (static) and stream the constraint-violation and
    energy curves for one stabilization choice — the honesty curves, overlaid.
    """
    rr.log(f"world/{name}", rr.LineStrips3D([tip], colors=[COLORS[name]]), static=True)
    for i, (viol, e) in enumerate(zip(violations, energies)):
        rr.set_time("step", sequence=i)
        rr.log(f"violation/{name}", rr.Scalars([float(viol)]))  # the hero panel
        rr.log(f"energy/{name}", rr.Scalars([float(e)]))
# --- endregion ---

# --- region: report ---
# Run the chosen stabilization(s), print the drift table, write metrics.json.
# The deliverable is the comparison: WITHOUT stabilization the max link-length
# error is orders of magnitude larger than WITH it — the measured reason a real
# engine never trusts the naive acceleration-level solve alone.
modes = {"none": 0.0, "baumgarte": args.baumgarte}
names = list(modes) if args.stabilization == "all" else [args.stabilization]
system = build_system(args.system, rng)
step_fn = INTEGRATORS[args.integrator]

if args.rerun:
    rr.init("zero2robot/ch3.4-constraints", spawn=False)
    rr.save(str(args.out / "constraints.rrd"))
    rr.log("world/pivot", rr.Points3D([PIVOT], radii=[0.03]), static=True)

results: dict[str, dict] = {}
# Characteristic energy scale for the drift ratio: the chain starts at y ~ 0 so
# its initial energy is ~0 — a useless normalizer. m*g*L (summed) is the natural
# scale of the potential swings, and it is fixed, so energy_rel_max is honest.
escale = float(np.sum(system["m"]) * np.linalg.norm(GRAVITY) * np.sum(system["lengths"]))
for name in names:
    force = make_force(system, modes[name])
    tip, violations, energies = simulate(step_fn, system, force, args.dt, num_steps)
    results[name] = {
        "max_violation": float(np.max(violations)),
        "final_violation": float(violations[-1]),
        "energy_rel_max": float(np.max(np.abs(energies - energies[0])) / escale),
        "tip_final": [round(float(x), 6) for x in tip[-1]],
    }
    if args.rerun:
        log_rerun(name, tip, violations, energies)

metrics = {
    "system": args.system,
    "integrator": args.integrator,
    "dt": args.dt,
    "steps": num_steps,
    "seed": args.seed,
    "baumgarte": args.baumgarte,
    # 12 decimals: a well-stabilized run's violation is ~1e-4 and rounding it to
    # fewer digits would erase the signal the exercises measure.
    "results": {k: {kk: (round(vv, 12) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in results.items()},
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

sim_seconds = num_steps * args.dt
print(f"integrated {args.system} ({args.integrator}) for {num_steps} steps of {args.dt} s -> {sim_seconds:.2f} s of sim time")
print(f"{'stabilization':<16}{'max |len err|':>16}{'final |len err|':>18}{'energy rel_max':>16}")
for name in names:
    r = results[name]
    print(f"{name:<16}{r['max_violation']:>16.4e}{r['final_violation']:>18.4e}{r['energy_rel_max']:>16.4e}")
if args.stabilization == "all":
    # The headline, stated as an ordering so it survives any seed on any tier
    # (measured factor is 5x-17x across pendulum/double/triple; 3x is the robust floor).
    ok = results["none"]["max_violation"] > 3.0 * results["baumgarte"]["max_violation"]
    verdict = "as expected" if ok else "UNEXPECTED — investigate"
    print(f"constraint drift  none >> baumgarte (stabilization holds the links):  {verdict}")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'constraints.rrd'} — open it with: rerun {args.out / 'constraints.rrd'}")
# --- endregion ---
