"""SUGGESTED exercise candidate (humans promote) — code-completion, ch3.4.

Objective tested: the two lines that ARE the constraint solver. The Jacobian
assembly and the linear solve are given, complete and correct. You fill in the
two ideas that make it work:

  1. the Jdot v term — differentiating g = 1/2(|d|^2 - L^2) twice gives
     gddot = J a + Jdot v, and for a distance constraint Jdot v is |d_dot|^2;
     leave it out and gddot = 0 is solved against the wrong right-hand side.
  2. the Baumgarte feedback — replacing the target gddot = 0 with the
     critically-damped gddot + 2*omega*gdot + omega^2*g = 0, which drives an
     existing violation back to zero instead of merely freezing its acceleration.

State is a chain: positions q and velocities v are (N, 3); `pairs[k] = (i, j)`
ties particle i to particle j (j = -1 means the fixed pivot); `lengths[k]` is
link k's rest length; `minv` is diag(M^-1) as a (3N,) vector; `f_ext` is the
external force flattened to (3N,). Returns the constraint force, shape (N, 3).

checks.py compares your completed solve against the reference on random chains;
while a blank is unfilled it raises NotImplementedError and the check SKIPS.
Estimated learner time: 25 minutes.
"""

import numpy as np

METADATA = {"type": "code-completion", "chapter": "ch3.4-constraints"}

PIVOT = np.array([0.0, 0.0, 0.0])


def constraint_force(q, v, pairs, lengths, minv, f_ext, baumgarte):
    n_particles, n_con = q.shape[0], len(pairs)
    jac = np.zeros((n_con, 3 * n_particles))
    g = np.zeros(n_con)     # position error   g   = 1/2(|d|^2 - L^2)
    gdot = np.zeros(n_con)  # velocity error   gdot = J v = d . d_dot
    jdotv = np.zeros(n_con)  # the Jdot v term  (you complete this)
    for k, (i, j) in enumerate(pairs):
        if j < 0:  # first link: partner is the fixed pivot (zero velocity)
            d, dv = q[i] - PIVOT, v[i]
            jac[k, 3 * i:3 * i + 3] = d
        else:
            d, dv = q[i] - q[j], v[i] - v[j]
            jac[k, 3 * i:3 * i + 3] = d
            jac[k, 3 * j:3 * j + 3] = -d
        g[k] = 0.5 * (d @ d - lengths[k] ** 2)
        gdot[k] = d @ dv
        # TODO (1): the Jdot v term for a distance constraint is |d_dot|^2.
        jdotv_k = None  # <- replace: dv @ dv
        if jdotv_k is None:
            raise NotImplementedError("complete the Jdot v term (hint: |d_dot|^2)")
        jdotv[k] = jdotv_k

    a_mat = jac @ (minv[:, None] * jac.T)      # J M^-1 J^T, (C, C), SPD
    b = -(jac @ (minv * f_ext) + jdotv)        # naive right-hand side (gddot = 0)
    if baumgarte > 0.0:
        # TODO (2): add the critically-damped Baumgarte feedback so the target
        # becomes gddot + 2*omega*gdot + omega^2*g = 0. Subtract it from b.
        feedback = None  # <- replace: 2.0 * baumgarte * gdot + baumgarte**2 * g
        if feedback is None:
            raise NotImplementedError("complete the Baumgarte feedback term")
        b -= feedback
    lam = np.linalg.solve(a_mat, b)            # the Lagrange multipliers
    return (jac.T @ lam).reshape(q.shape)      # J^T lambda, back to (N, 3)


if __name__ == "__main__":
    # A tiny self-check: a 2-link chain, hanging, solve the constraint force once.
    q = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    v = np.zeros((2, 3))
    pairs, lengths = [(0, -1), (1, 0)], np.array([1.0, 1.0])
    minv = np.ones(6)
    f_ext = (np.array([[0.0, -9.81, 0.0]] * 2)).reshape(-1)
    try:
        fc = constraint_force(q, v, pairs, lengths, minv, f_ext, baumgarte=20.0)
        print("constraint force:\n", np.round(fc, 6))
    except NotImplementedError as exc:
        print("not yet:", exc)
