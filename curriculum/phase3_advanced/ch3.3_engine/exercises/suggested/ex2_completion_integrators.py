"""SUGGESTED exercise candidate (humans promote) — code-completion, ch3.3.

Objective tested: the two integrators differ from explicit Euler by ONE idea
each. Explicit Euler is given, complete and correct, as your reference. Fill in
the two blanks so the semi-implicit and RK4 steppers match engine.py exactly.

State is (position q, velocity v); `accel(q, v)` returns the acceleration
force(q, v) / m. Each `*_step` returns (q_next, v_next).

checks.py compares your completed functions against the reference steppers on a
batch of random states; while a blank is unfilled they raise NotImplementedError
and the check SKIPS. Estimated learner time: 20 minutes.
"""

import numpy as np

METADATA = {"type": "code-completion", "chapter": "ch3.3-engine"}


def euler_step(q, v, accel, dt):
    """Explicit (forward) Euler — GIVEN. Everything is evaluated at the OLD
    state: new position uses the old velocity, new velocity uses the old accel.
    """
    a = accel(q, v)
    return q + dt * v, v + dt * a


def semi_implicit_step(q, v, accel, dt):
    """Semi-implicit (symplectic) Euler. ONE line changes from explicit Euler:
    update the velocity first, then step the position with the NEW velocity.
    That single reorder is what keeps energy bounded on an oscillatory system.
    """
    a = accel(q, v)
    v_next = v + dt * a
    # TODO: step the position using v_next (the UPDATED velocity), not v.
    q_next = None  # <- replace: q + dt * v_next
    if q_next is None:
        raise NotImplementedError("complete the semi-implicit position update")
    return q_next, v_next


def rk4_step(q, v, accel, dt):
    """Classical RK4 on the first-order system (q, v)' = (v, accel(q, v)).
    Four samples of the derivative, weighted 1-2-2-1. k1 is done for you; fill
    k2, k3, k4 (the midpoint and endpoint samples) and the weighted combine.
    """
    def deriv(q, v):
        return v, accel(q, v)

    k1q, k1v = deriv(q, v)
    # TODO: k2 at the midpoint using k1; k3 at the midpoint using k2; k4 at the
    # endpoint using k3. Then combine q + dt/6 * (k1 + 2*k2 + 2*k3 + k4).
    k2q, k2v = (None, None)  # <- deriv(q + 0.5*dt*k1q, v + 0.5*dt*k1v)
    k3q, k3v = (None, None)  # <- deriv(q + 0.5*dt*k2q, v + 0.5*dt*k2v)
    k4q, k4v = (None, None)  # <- deriv(q + dt*k3q,     v + dt*k3v)
    if any(x is None for x in (k2q, k2v, k3q, k3v, k4q, k4v)):
        raise NotImplementedError("complete the RK4 midpoint/endpoint samples and combine")
    q_next = q + (dt / 6.0) * (k1q + 2.0 * k2q + 2.0 * k3q + k4q)
    v_next = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    return q_next, v_next


if __name__ == "__main__":
    # A tiny self-check: integrate a unit spring (accel = -q) one step and print.
    def accel(q, v):
        return -q

    q0, v0, dt = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0]), 0.05
    for name, step in (("euler", euler_step), ("semi_implicit", semi_implicit_step), ("rk4", rk4_step)):
        try:
            q1, v1 = step(q0, v0, accel, dt)
            print(f"{name:<14} q -> {np.round(q1, 6)}  v -> {np.round(v1, 6)}")
        except NotImplementedError as exc:
            print(f"{name:<14} not yet: {exc}")
