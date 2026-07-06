"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch1.6.

Here is the single most common way a success rate lies. You evaluate a policy,
it succeeds on 0 of 20 held-out episodes, and you report "0% success, +/- 0%" —
a point estimate with a standard error of zero. That interval says you are
CERTAIN the true success rate is exactly 0. You are not: you flipped a coin 20
times, saw no heads, and concluded the coin never lands heads.

THE BUG. `report_ci` below computes the naive WALD interval,

    p_hat +- z * sqrt( p_hat * (1 - p_hat) / n )

which is what most people write first. It is fine in the middle (p_hat near 0.5)
and broken at the edges: at k=0 or k=n the p_hat*(1-p_hat) term is 0, so the
half-width is 0 and the interval collapses to a single point — claiming certainty
from a handful of trials. It can also report bounds below 0 or above 1.

Replace the Wald interval with the WILSON score interval (the one harness.py
uses), which never collapses and never leaves [0, 1]:

    center = (p_hat + z^2/(2n)) / (1 + z^2/n)
    half   = (z / (1 + z^2/n)) * sqrt( p_hat*(1-p_hat)/n + z^2/(4 n^2) )
    return (max(0, center - half), min(1, center + half))

Fix it so the check passes (0 of 20 must return a POSITIVE upper bound):

    pytest curriculum/phase1_imitation/ch1.6_harness/exercises/suggested/checks.py -k ex4
"""

import math

METADATA = {"type": "bug-hunt", "chapter": "ch1.6-harness"}

Z95 = 1.959963985  # 0.975 standard-normal quantile (95% two-sided)


def report_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% CI for a success rate k/n. Return (lo, hi).

    BUG: this is the Wald interval. At k=0 (or k=n) it returns a zero-width
    interval — "0% success, no uncertainty" — a lie from 20 coin flips. Swap it
    for the Wilson interval from the docstring so the band stays honest at the
    boundary.
    """
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    half = z * math.sqrt(p_hat * (1.0 - p_hat) / n)   # <-- Wald: collapses to 0 at k=0 / k=n
    return (p_hat - half, p_hat + half)
