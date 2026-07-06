"""SUGGESTED exercise candidate (humans promote) — code-completion, ch3.5.

Objective tested: the two lines that ARE contact. Both are about the ONE-SIDEDNESS
of a contact — the table pushes but never pulls — expressed twice, once in each
solver family:

  1. the PENALTY push-only clamp — a spring force k*depth - c*(closing speed) that
     is CLAMPED to be non-negative, so a separating body is never yanked back. Drop
     the clamp and the "spring" glues the body to the floor (a contact that pulls).
  2. the LCP non-penetration PROJECTION — the accumulated normal impulse must stay
     >= 0 (push-only) as we sweep it. That single max(0, .) is the complementarity
     condition; without it the solve is an equality constraint (a weld), not a contact.

The geometry (detect, normals, effective masses) is given, complete and correct.
You fill in the two clamps. checks.py compares your completed functions against the
reference on random configurations; while a blank is unfilled it raises
NotImplementedError and the check SKIPS.
Estimated learner time: 25 minutes.
"""

METADATA = {"type": "code-completion", "chapter": "ch3.5-contact"}


def penalty_normal_force(depth, closing_speed, k, c):
    """The scalar normal force a penalty contact applies (>= 0), given how deep the
    surfaces overlap (`depth` > 0) and how fast they are closing (`closing_speed`,
    the normal velocity; negative when separating).
    """
    # TODO (1): a penalty contact pushes with a spring minus a damper,
    # k*depth - c*closing_speed — but a contact only PUSHES. Clamp it so the force
    # is never negative (it must never pull a separating body back to the surface).
    force = None  # <- replace: max(0.0, k * depth - c * closing_speed)
    if force is None:
        raise NotImplementedError("complete the penalty push-only force (hint: max(0, k*depth - c*closing_speed))")
    return force


def project_impulse(lam_accumulated, delta):
    """Projected Gauss-Seidel update for one contact's normal impulse. `delta` is
    the impulse this sweep wants to add; the TOTAL accumulated impulse must stay
    push-only. Return the (possibly clamped) new accumulated impulse.
    """
    # TODO (2): the non-penetration / no-pull complementarity condition: the total
    # normal impulse may never go negative. Clamp the accumulated value at 0.
    new_lam = None  # <- replace: max(0.0, lam_accumulated + delta)
    if new_lam is None:
        raise NotImplementedError("complete the impulse projection (hint: max(0, lam + delta))")
    return new_lam


if __name__ == "__main__":
    # A tiny self-check: a body pressing in (depth 0.01, closing at 0.5 m/s) should
    # get a positive push; a body already separating fast should get zero.
    try:
        print("pressing in :", penalty_normal_force(0.01, 0.5, k=1.0e4, c=30.0))   # ~85
        print("separating  :", penalty_normal_force(0.01, 100.0, k=1.0e4, c=30.0))  # clamped to 0
        print("impulse clamp:", project_impulse(0.2, -0.5))  # would-be negative -> 0
    except NotImplementedError as exc:
        print("not yet:", exc)
