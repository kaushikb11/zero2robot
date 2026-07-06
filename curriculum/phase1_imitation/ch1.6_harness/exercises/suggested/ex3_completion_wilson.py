"""SUGGESTED exercise candidate (humans promote) — code-completion, ch1.6.

Every success rate in this book should ship with a confidence interval, and the
one harness.py uses is the WILSON score interval. It is the whole reason a "0.40"
becomes a "[0.24, 0.58]". You will implement it from the formula.

Given k successes in n Bernoulli trials, p_hat = k/n, and z the 0.975 standard-
normal quantile (z = 1.959964 for a 95% two-sided interval), the Wilson interval
is centered at

    center = (p_hat + z^2 / (2n)) / (1 + z^2 / n)

with half-width

    half = (z / (1 + z^2 / n)) * sqrt( p_hat*(1 - p_hat)/n + z^2 / (4 n^2) )

and the interval is (center - half, center + half), clamped into [0, 1]. Unlike
the naive Wald interval p_hat +- z*sqrt(p_hat(1-p_hat)/n), this one stays inside
[0, 1] and is never zero-width at k=0 or k=n — which is exactly where a success
rate most needs an honest band.

YOUR JOB: implement `wilson_ci` from the formulas above (return (lo, hi), clamped
to [0, 1]; for n == 0 return the whole interval (0.0, 1.0)). Then:

    pytest curriculum/phase1_imitation/ch1.6_harness/exercises/suggested/checks.py -k ex3

The checks compare you against the textbook values (Brown, Cai & DasGupta 2001):
0 of 10 -> [0, 0.2775] and 5 of 10 -> [0.2366, 0.7634].

(You will want `import math` for the square root.)
"""

METADATA = {"type": "code-completion", "chapter": "ch1.6-harness"}

Z95 = 1.959963985  # 0.975 standard-normal quantile (95% two-sided)


def wilson_ci(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials. Return (lo, hi),
    each clamped into [0, 1]; return (0.0, 1.0) when n == 0.

    Replace the NotImplementedError with the center +- half formulas in the
    module docstring.
    """
    raise NotImplementedError("implement the Wilson score interval (see the module docstring)")
