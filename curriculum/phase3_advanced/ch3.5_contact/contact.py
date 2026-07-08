"""zero2robot 3.5 — Build a Physics Engine III: Contact (the hard part).

Chapter 3.3 built the unconstrained update: state (position q, velocity v,
mass m), a force law force(q, v), integrators, and ENERGY DRIFT as the honesty
metric. Chapter 3.4 added JOINTS as constraints g(q) = 0 you SOLVE FOR, with
CONSTRAINT DRIFT as the metric. A joint is a constraint that is always on — the
rod is always exactly length L. CONTACT is the constraint that switches on and
off: the floor pushes UP on a body only while it is touching, and it never
PULLS it down. That one-sidedness is an INEQUALITY, not an equality —

    gap >= 0   (no penetration)   AND   force >= 0   (only push, never pull)
    AND  gap * force = 0          (no push once separated)

— a COMPLEMENTARITY problem, and it is where every sim artifact you have seen
since ch0.1 (jitter, sinking, phantom energy, the timestep that explodes) comes
from. We build two contact models from scratch, on ch3.3's exact (q, v, m) shape,
and MEASURE which artifacts each one produces:

- PENALTY: a stiff spring-damper on penetration depth. Dead simple — a contact
  is just an ADDED force, so ch3.3's integrator steps it unchanged. But stiff =
  unstable: the body SINKS by mg/k at rest, JITTERS while it settles, and if dt
  gets anywhere near 2*sqrt(m/k) the whole thing EXPLODES.
- LCP-FLAVORED: solve the complementarity directly for the contact IMPULSE that
  makes the body stop pressing in (projected Gauss-Seidel, one small clamp per
  contact). It HOLDS the body on the table — no sink, no jitter — and is stable
  at timesteps where penalty diverges. It is not a force; it is a velocity-level
  solve OUTSIDE the plain integrator, which is the whole contrast.

The penalty-vs-complementarity trade you fight here is exactly WHY MuJoCo uses a
SOFT, regularized, convex contact model — a principled middle path. Chapter 3.6
reuses `detect_contacts` + `solve_contacts` for PushT (a pusher CONTACTING a
T-block is body-body contact, the same code path the `stack` scene exercises).

Run it:      python contact.py                          # drop a body, penalty vs lcp
             python contact.py --scene bounce --contact lcp
             python contact.py --scene stack             # few contacts (PGS)
CI smoke:    python contact.py --smoke --seed 0 --no-rerun   # two runs byte-identical

HONEST LIMITS (author: keep — the lesson is the trade, not a perfect solver):
  * NORMAL contact only — NO FRICTION. Bodies are point masses with a radius
    (pucks/balls), so there is no rotation and no friction cone. Sliding and the
    friction pyramid are ch3.6's problem; the normal complementarity is the core.
  * The LCP-flavored solve is a fixed-iteration projected Gauss-Seidel, not a
    true LCP pivot — good enough to HOLD the stack, and honestly labelled.
"""

# --- region: setup ---
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import rerun as rr

# Loose script from the repo root (same pattern as ch3.3/3.4); put the root on
# sys.path so curriculum.common resolves. Pure numpy — no torch, no mujoco for
# the dynamics — so every run is bitwise reproducible on CPU: same seed, same
# bytes. Contact stays deterministic because the contact LIST is built in a fixed
# order (floor contacts by body index, then body-body pairs by (i, j)) and the
# Gauss-Seidel solve runs a FIXED iteration count — no data-dependent branching.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.device import banner  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--seed", type=int, default=0, help="seeds the drop-height jitter; load-bearing but the artifact ORDERING (penalty penetrates/jitters >> lcp holds) is seed-robust")
parser.add_argument("--scene", choices=["drop", "bounce", "stack"], default="drop", help="drop=settle on the table; bounce=restitution; stack=a few bodies (floor + body-body contacts)")
parser.add_argument("--contact", choices=["all", "penalty", "lcp"], default="all", help="'all' runs the penalty-vs-lcp comparison — the whole point")
parser.add_argument("--stiffness", type=float, default=1.0e4, help="penalty spring constant k; bigger k = less sinking but a smaller stable dt (dt_crit ~ 2*sqrt(m/k))")
parser.add_argument("--damping", type=float, default=-1.0, help="penalty spring DAMPER c; the ONLY thing setting penalty's (uncontrollable) restitution. <0 uses the scene default (drop/stack: 80 settles; bounce: 0 rings and INJECTS energy)")
parser.add_argument("--restitution", type=float, default=-1.0, help="lcp bounce coefficient e in [0,1]; <0 means use the scene default (drop/stack: 0, bounce: 0.7)")
parser.add_argument("--baumgarte", type=float, default=0.2, help="lcp position-correction gain beta; feeds penetration back as a separation velocity so the body does not sink")
parser.add_argument("--iters", type=int, default=20, help="projected Gauss-Seidel sweeps per lcp step (fixed for determinism); enough for the stack's few contacts")
parser.add_argument("--dt", type=float, default=0.002, help="timestep in sim seconds; raise it past penalty's dt_crit and penalty EXPLODES while lcp holds")
parser.add_argument("--steps", type=int, default=2000)  # any laptop: milliseconds
parser.add_argument("--smoke", action="store_true", help="fixed 800-step run for CI; two runs must match byte-for-byte")
parser.add_argument("--device", choices=["cpu"], default="cpu", help="pure-numpy CPU; the flag exists for banner/tier parity")
parser.add_argument("--out", type=Path, default=Path("outputs/ch3.5-contact"))
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)  # recording is the default; opt OUT, not in
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip the .rrd recording (CI smoke)")
args = parser.parse_args()

banner("ch3.5-contact", device=args.device)  # tier + measured wall-clock, printed first
num_steps = 800 if args.smoke else args.steps  # smoke length is FIXED so CI can diff runs exactly
args.out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(args.seed)  # PCG64 — the only source of randomness in this file

GRAVITY = np.array([0.0, -9.81])  # 2D world; the table is the line y = 0, normal points +y
FLOOR_N = np.array([0.0, 1.0])
# --- endregion ---

# --- region: scenes ---
# A "scene" is a set of point masses, each with a collision RADIUS (a puck/ball —
# no rotation, so no friction cone: normal contact is the whole lesson). We keep
# ch3.3's (q, v, m) shape and stack it to N bodies: q, v are (N, 2), m is (N, 1)
# so force / m broadcasts per body exactly as ch3.3's scalar divide did. The new
# fields are the geometry `radius` (N,) and the material `restitution` (the LCP
# bounce coefficient). The seed jitters ONE knob — the drop height — so --seed is
# load-bearing while the physics, and the penalty-vs-lcp ordering, stay put.


def build_scene(name: str, rng: np.random.Generator, restitution_override: float, damping_override: float) -> dict:
    """Return {q0, v0, m, radius, restitution, damping} for a contact test scene.

    The material `restitution` is what lcp uses directly; penalty has no such knob
    — it can only APPROXIMATE a bounce through its `damping`, badly, which is the
    whole point (drop/stack want it settled and damped; bounce wants it to ring).
    """
    radius = 0.1
    jitter = rng.uniform(-0.02, 0.02)  # the only randomness: a small drop-height nudge
    if name == "stack":
        # Three equal balls stacked exactly touching, released from rest. A good
        # solver HOLDS the tower (floor + two body-body contacts, solved together);
        # penalty lets it settle by sinking and jittering. This is the body-body
        # contact path ch3.6 reuses (a pusher on a T-block is the same math).
        n, rest, damp = 3, 0.0, 80.0
        q0 = np.array([[0.0, (2 * i + 1) * radius + jitter] for i in range(n)])
    else:  # drop / bounce — one ball falling onto the table
        n = 1
        q0 = np.array([[0.0, 1.0 + jitter]])  # a metre up, straight above the table
        # bounce: restitution 0.7 and NO penalty damping (so the discrete spring
        # rings and INJECTS energy — the classic penalty artifact). drop: settle.
        rest, damp = (0.7, 0.0) if name == "bounce" else (0.0, 80.0)
    v0 = np.zeros((n, 2))  # released from rest
    m = np.ones((n, 1))    # unit point masses
    rest = restitution_override if restitution_override >= 0.0 else rest
    damp = damping_override if damping_override >= 0.0 else damp
    return {"q0": q0, "v0": v0, "m": m, "radius": np.full(n, radius), "restitution": rest, "damping": damp}
# --- endregion ---

# --- region: contacts ---
# THE geometry step, shared by BOTH solvers and by ch3.6. A contact is a place
# where two surfaces overlap (or nearly do): who is involved (body i, and body j
# or the floor as j = -1), the unit NORMAL n along which they may only push apart,
# and the PENETRATION depth (how far they already overlap, >= 0). We detect a
# little BEFORE touching (a small margin) so a resting contact — depth ~ 0 — is
# still on the list and the solver can hold it there. Order is fixed (floor by
# body index, then pairs by (i, j)) so the whole run is bitwise deterministic.

CONTACT_MARGIN = 1.0e-3  # detect contacts this far before actual overlap (holds resting bodies)


def detect_contacts(q: np.ndarray, radius: np.ndarray) -> list[dict]:
    """Return the list of active contacts for the current positions."""
    contacts = []
    for i in range(q.shape[0]):  # body vs the floor (y = 0)
        gap = q[i, 1] - radius[i]  # signed distance from the ball's underside to the table
        if gap < CONTACT_MARGIN:
            contacts.append({"i": i, "j": -1, "n": FLOOR_N, "depth": -gap})
    for i in range(q.shape[0]):  # body vs body
        for j in range(i + 1, q.shape[0]):
            d = q[i] - q[j]
            dist = float(np.linalg.norm(d))
            overlap = radius[i] + radius[j] - dist
            if overlap > -CONTACT_MARGIN and dist > 1e-12:
                contacts.append({"i": i, "j": j, "n": d / dist, "depth": overlap})
    return contacts


def _normal_velocity(v: np.ndarray, c: dict) -> float:
    """Relative velocity of the contact pair along the normal (>0 separating)."""
    vn = float(v[c["i"]] @ c["n"])
    if c["j"] >= 0:
        vn -= float(v[c["j"]] @ c["n"])
    return vn
# --- endregion ---

# --- region: penalty ---
# PENALTY contact: pretend every contact is a stiff spring-damper pushing the
# surfaces apart, force = k*penetration - c*(closing speed), and CLAMP it to
# push-only (max(0, .)) so it never pulls — the one-sidedness, enforced by hand.
# That is it: a contact becomes an ADDED force, so ch3.3's integrator steps it
# with ZERO changes. The price is everywhere in the metrics: the body sinks to
# where k*depth balances gravity (depth = mg/k), it rings while it settles, and a
# dt past ~2*sqrt(m/k) turns the spring into a bomb. Simplicity you pay for later.


def penalty_force(q: np.ndarray, v: np.ndarray, m: np.ndarray, radius: np.ndarray,
                  k: float, c: float) -> np.ndarray:
    """Gravity + the spring-damper contact force, in ch3.3's force(q, v) shape."""
    f = m * GRAVITY  # (N,1)*(2,) -> (N,2), same broadcast as ch3.3
    for contact in detect_contacts(q, radius):
        depth = contact["depth"]
        if depth <= 0.0:
            continue  # a spring only pushes while actually compressed (not in the margin)
        vn = _normal_velocity(v, contact)
        fn = max(0.0, k * depth - c * vn)  # push-only: never yank a separating body back
        f[contact["i"]] += fn * contact["n"]
        if contact["j"] >= 0:
            f[contact["j"]] -= fn * contact["n"]  # equal and opposite on the partner
    return f


def semi_implicit_step(q, v, m, force, dt):
    """ch3.3's symplectic Euler, reused VERBATIM (repetition is the lesson): new
    velocity first, then step position with it. The integrator never learns that
    `force` now hides contact detection — a contact is just a force to it.
    """
    a = force(q, v) / m
    v_next = v + dt * a
    q_next = q + dt * v_next  # NEW velocity, not old
    return q_next, v_next
# --- endregion ---

# --- region: lcp ---
# LCP-FLAVORED contact: instead of a spring, solve the complementarity directly.
# Predict the velocity under gravity, then find the contact IMPULSE lambda >= 0
# that leaves each contact non-penetrating: the post-solve normal velocity must
# be >= a small target (separate, or hold, plus a Baumgarte push that bleeds off
# any leftover penetration) AND lambda >= 0 (push-only) AND they are complementary
# (no impulse once separating). For one contact that is a single clamp; for the
# stack's few contacts we sweep them Gauss-Seidel style, a fixed number of times.
# This is NOT a force added to the integrator — it is a velocity projection that
# REPLACES the plain step. That structural difference is the chapter's point.


def solve_contacts(v: np.ndarray, minv: np.ndarray, contacts: list[dict],
                   restitution: float, baumgarte: float, dt: float, iters: int) -> np.ndarray:
    """Projected Gauss-Seidel for the contact impulses. Mutates and returns v."""
    vn_pre = [_normal_velocity(v, c) for c in contacts]  # approach speed, for restitution
    lam = np.zeros(len(contacts))  # accumulated normal impulse per contact, clamped >= 0
    for _ in range(iters):  # FIXED count -> deterministic; enough sweeps for a few contacts
        for idx, c in enumerate(contacts):
            i, j, n = c["i"], c["j"], c["n"]
            k_eff = minv[i] + (minv[j] if j >= 0 else 0.0)  # normal effective inverse-mass
            # Target normal velocity: bounce (-e*approach, only if it was closing)
            # plus a gentle Baumgarte push proportional to how deep we already are.
            target = -restitution * min(vn_pre[idx], 0.0) + (baumgarte / dt) * c["depth"]
            vn = _normal_velocity(v, c)
            dlam = (target - vn) / k_eff              # impulse to hit the target this sweep
            new = max(0.0, lam[idx] + dlam)           # project: total impulse stays push-only
            dlam, lam[idx] = new - lam[idx], new
            v[i] += (dlam * minv[i]) * n              # apply the delta impulse to both bodies
            if j >= 0:
                v[j] -= (dlam * minv[j]) * n
    return v


def lcp_step(q, v, m, radius, restitution, baumgarte, dt, iters):
    """One contact-solved step: gravity predict, then project the velocity onto
    the non-penetration set, then integrate position with the corrected velocity.
    """
    v = v + dt * GRAVITY                       # semi-implicit predict (gravity is mass-independent)
    contacts = detect_contacts(q, radius)
    if contacts:
        minv = (1.0 / m).reshape(-1)           # diag(M^-1) as a (N,) vector
        v = solve_contacts(v, minv, contacts, restitution, baumgarte, dt, iters)
    return q + dt * v, v
# --- endregion ---

# --- region: simulate ---
# Run one contact model over the whole horizon and record, at every step: the
# body HEIGHT (what you watch settle or bounce), the worst PENETRATION depth (the
# honesty metric — a body should NOT sink through the table), and the total ENERGY
# (a good inelastic contact only ever REMOVES energy; a penalty spring INJECTS it
# on impact and rings). Pure function of its inputs — bitwise reproducible.


def energy(q: np.ndarray, v: np.ndarray, m: np.ndarray) -> float:
    """Total mechanical energy: kinetic + gravitational potential (U = m g y)."""
    kinetic = 0.5 * float(np.sum(m * v * v))
    potential = float(np.sum(m[:, 0] * 9.81 * q[:, 1]))
    return kinetic + potential


def max_penetration(q: np.ndarray, radius: np.ndarray) -> float:
    """Worst overlap in the scene right now: floor penetration or body-body."""
    worst = 0.0
    for i in range(q.shape[0]):
        worst = max(worst, radius[i] - q[i, 1])            # into the floor
        for j in range(i + 1, q.shape[0]):
            dist = float(np.linalg.norm(q[i] - q[j]))
            worst = max(worst, radius[i] + radius[j] - dist)  # into each other
    return worst


def simulate(step, scene: dict, dt: float, steps: int):
    """Integrate for `steps`. Returns (heights, penetrations, energies) arrays."""
    q = scene["q0"].astype(float).copy()
    v = scene["v0"].astype(float).copy()
    m, radius = scene["m"], scene["radius"]

    heights = np.empty((steps + 1, q.shape[0]))
    penetrations = np.empty(steps + 1)
    energies = np.empty(steps + 1)
    heights[0], penetrations[0], energies[0] = q[:, 1], max_penetration(q, radius), energy(q, v, m)
    for t in range(steps):
        q, v = step(q, v)
        heights[t + 1] = q[:, 1]
        penetrations[t + 1] = max_penetration(q, radius)
        energies[t + 1] = energy(q, v, m)
    return heights, penetrations, energies


def contact_quality(penetrations: np.ndarray, energies: np.ndarray, radius: float) -> dict:
    """The honesty numbers, parallel to ch3.3 energy drift / ch3.4 constraint drift.

    max_penetration_frac: worst sink as a fraction of the body radius (want ~0).
    rest_penetration_frac: the STATIC sink once settled (penalty: ~mg/k; lcp: ~0).
    energy_excess: worst energy ABOVE the start, over the start (a correct
                   inelastic contact never exceeds it; a penalty spring does).
    rest_jitter: penetration wobble over the settled tail (penalty rings; lcp flat).
    """
    e0 = energies[0]
    scale = abs(e0) if e0 != 0.0 else 1.0
    tail = penetrations[3 * len(penetrations) // 4:]  # last quarter — "settled" window
    finite = np.isfinite(energies).all() and np.isfinite(penetrations).all()
    return {
        "max_penetration_frac": float(np.max(penetrations) / radius) if finite else float("inf"),
        "rest_penetration_frac": float(penetrations[-1] / radius) if finite else float("inf"),
        "energy_excess": float(np.max(energies - e0) / scale) if finite else float("inf"),
        "rest_jitter": float(np.std(tail) / radius) if finite else float("inf"),
        "blew_up": not finite,
    }


COLORS = {"penalty": [217, 76, 64], "lcp": [76, 175, 80]}  # red sinks/jitters, green holds


def log_rerun(name: str, heights: np.ndarray, penetrations: np.ndarray, energies: np.ndarray) -> None:
    """Stream the body height, penetration (the hero panel), and energy curves."""
    for t in range(len(penetrations)):
        rr.set_time("step", sequence=t)
        rr.log(f"height/{name}", rr.Scalars([float(h) for h in heights[t]]))
        rr.log(f"penetration/{name}", rr.Scalars([float(penetrations[t])]), colors=[COLORS[name]])
        rr.log(f"energy/{name}", rr.Scalars([float(energies[t])]), colors=[COLORS[name]])
# --- endregion ---

# --- region: report ---
# Run the chosen model(s), print the contact-quality table, write metrics.json.
# The deliverable is the comparison: penalty SINKS + JITTERS (and, past dt_crit,
# EXPLODES) while the LCP-flavored solve HOLDS the body on the table with almost
# no penetration and no phantom energy — the measured reason a real engine never
# ships a bare penalty contact, and the reason MuJoCo softens the hard LCP.
scene = build_scene(args.scene, rng, args.restitution, args.damping)
radius0 = float(scene["radius"][0])


def run_penalty(dt: float):
    def force(q, v):
        return penalty_force(q, v, scene["m"], scene["radius"], args.stiffness, scene["damping"])
    return simulate(lambda q, v: semi_implicit_step(q, v, scene["m"], force, dt), scene, dt, num_steps)


def run_lcp(dt: float):
    def step(q, v):
        return lcp_step(q, v, scene["m"], scene["radius"], scene["restitution"], args.baumgarte, dt, args.iters)
    return simulate(step, scene, dt, num_steps)


RUNNERS = {"penalty": run_penalty, "lcp": run_lcp}
names = list(RUNNERS) if args.contact == "all" else [args.contact]

if args.rerun:
    rr.init("zero2robot/ch3.5-contact", spawn=False)
    rr.save(str(args.out / "contact.rrd"))
    rr.log("world/table", rr.LineStrips2D([[[-1.0, 0.0], [1.0, 0.0]]]), static=True)  # the table line

with np.errstate(all="ignore"):  # a diverging penalty spring overflows on purpose; we measure it, not crash on it
    quality: dict[str, dict] = {}
    for name in names:
        heights, penetrations, energies = RUNNERS[name](args.dt)
        quality[name] = contact_quality(penetrations, energies, radius0)
        if args.rerun and np.isfinite(energies).all():
            log_rerun(name, heights, penetrations, energies)

    # Stability-vs-dt probe: the headline artifact. Penalty is stable only while
    # dt < ~2*sqrt(m/k); lcp has no such spring, so it holds at every dt here.
    stability = {}
    if args.contact == "all":
        dt_crit = 2.0 * np.sqrt(float(scene["m"][0, 0]) / args.stiffness)
        for name in names:
            biggest_stable = 0.0
            for mult in (1, 2, 4, 8, 16, 32):
                _, _, e = RUNNERS[name](args.dt * mult)
                if np.isfinite(e).all() and np.max(e - e[0]) / (abs(e[0]) or 1.0) < 10.0:
                    biggest_stable = args.dt * mult
            stability[name] = biggest_stable
        stability["dt_crit_penalty"] = float(dt_crit)

metrics = {
    "scene": args.scene, "contact": args.contact, "dt": args.dt, "steps": num_steps,
    "seed": args.seed, "stiffness": args.stiffness, "restitution": scene["restitution"],
    # 12 decimals: a good lcp run's penetration is ~1e-3 of the radius and rounding
    # it away would erase the very signal the exercises measure.
    "quality": {k: {kk: (round(vv, 12) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in quality.items()},
    "stability": {k: (round(v, 12) if isinstance(v, float) else v) for k, v in stability.items()},
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

print(f"scene {args.scene} ({num_steps} steps of {args.dt} s = {num_steps * args.dt:.2f} s sim), radius {radius0}")
print(f"{'contact':<10}{'max pen/r':>12}{'rest pen/r':>12}{'energy+':>12}{'jitter/r':>12}{'stable?':>10}")
for name in names:
    r = quality[name]
    flag = "EXPLODED" if r["blew_up"] else "ok"
    print(f"{name:<10}{r['max_penetration_frac']:>12.4e}{r['rest_penetration_frac']:>12.4e}{r['energy_excess']:>+12.3e}{r['rest_jitter']:>12.3e}{flag:>10}")
if args.contact == "all":
    # The headline is an ORDERING, so it survives any seed/tier: penalty drives the
    # DEEPER impact penetration AND rests INSIDE the table, while lcp never sinks past
    # it. (Energy and jitter are the bounce scene's headline; penetration is the
    # robust rock — it holds this ordering on every scene and seed.)
    ok = quality["lcp"]["max_penetration_frac"] < quality["penalty"]["max_penetration_frac"] and \
        quality["lcp"]["rest_penetration_frac"] <= quality["penalty"]["rest_penetration_frac"] + 1e-9
    verdict = "as expected" if ok else "UNEXPECTED — investigate"
    print(f"contact quality  penalty sinks+jitters >> lcp holds:  {verdict}")
    print(f"stability: penalty stable up to dt={stability['penalty']:.4g} s (dt_crit~{stability['dt_crit_penalty']:.4g}); lcp up to dt={stability['lcp']:.4g} s")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'contact.rrd'} — open it with: rerun {args.out / 'contact.rrd'}")
# config hash of the run knobs, for the wallclock ledger (author: paste into wallclock.csv)
_cfg = f"{args.scene}|{args.dt}|{args.stiffness}|{args.damping}|{args.baumgarte}|{args.iters}|{num_steps}"
print(f"config_hash: {hashlib.md5(_cfg.encode()).hexdigest()[:12]}")
# --- endregion ---
