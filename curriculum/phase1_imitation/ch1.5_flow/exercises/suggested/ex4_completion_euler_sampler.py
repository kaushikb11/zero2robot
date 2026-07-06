"""SUGGESTED exercise candidate (humans promote) — code-completion, ch1.5.

Sampling in flow matching is just integrating an ODE: start at a noise point (flow
time t = 0) and follow the learned velocity field forward to t = 1, where the
sample lives. With plain forward Euler and `steps` equal-size steps of dt = 1/steps,

    x <- x + dt * velocity(x, t)        for t = 0, dt, 2*dt, ..., (steps-1)*dt

That single update, repeated, IS flow.py's whole sampler — no schedule, no injected
noise, no posterior algebra (compare ch1.4's reverse posterior mean, which needed
all three). The straighter the true paths, the fewer steps this needs.

YOUR JOB: implement `euler_sample` from the rule above. `velocity(x, t)` is given to
you as a callable — call it once per step with the current point and the current
scalar time. Then:

    pytest curriculum/phase1_imitation/ch1.5_flow/exercises/suggested/checks.py -k ex4
"""

from typing import Callable

import numpy as np

METADATA = {"type": "code-completion", "chapter": "ch1.5-flow"}


def euler_sample(velocity: Callable[[np.ndarray, float], np.ndarray],
                 x0_noise: np.ndarray, steps: int) -> np.ndarray:
    """Integrate the velocity field from noise (t=0) to a sample (t=1) with forward
    Euler. `x0_noise` is the starting noise (B, D); `velocity(x, t)` returns the
    (B, D) velocity at point x and scalar time t; `steps` is the number of Euler
    steps. Return the final (B, D) point at t = 1.

    Replace the NotImplementedError with the loop: step size dt = 1/steps, and at
    each i in 0..steps-1 evaluate the velocity at time i*dt and take x <- x + dt*v.
    """
    raise NotImplementedError("implement forward-Euler ODE integration (see the module docstring)")
