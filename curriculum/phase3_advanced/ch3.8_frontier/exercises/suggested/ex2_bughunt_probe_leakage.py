"""SUGGESTED exercise candidate (humans promote) — bug-hunt (fast), ch3.8.

THE probing bug, isolated: evaluating a linear probe on the SAME rows it was fit on.
A linear probe has enough capacity to partly memorize its training rows, so scoring it
on those rows inflates the number — and, worst of all, makes even a RANDOM,
uninformative layer look like it "encodes" the target. The whole point of a probe is a
HELD-OUT read: fit on one split, score on another. probe.py splits 50/50; this
isolated version has the split bug.

Everything here is a self-contained numpy fixture (random features that carry NO signal
about the target), so this gate is fast and needs no torch or checkpoint. On leakage-
free code a random layer must score near 0; the buggy version scores far higher.

FIND THE BUG in `probe_r2` below (it scores on the training rows), then fix it to score
on the held-out rows. `checks.py` gates on the signature (still leaking) and then
verifies your fix drives the random-feature R^2 back down.

Before you fix it, write one sentence: why does scoring a probe on the very rows it was
fit on make even a random, signal-free layer look like it "encodes" the target — and why
does moving to a held-out split make that false signal collapse to ~0?

Estimated learner time: 15 minutes.
"""

import numpy as np

METADATA = {"type": "bug-hunt", "chapter": "ch3.8-frontier", "fast": True}

RIDGE = 1.0


def probe_r2(feats: np.ndarray, target: np.ndarray) -> float:
    """Fit a closed-form ridge linear probe on the FIRST half of the rows, return its
    R^2. A leakage-free probe scores on the SECOND (held-out) half.

    BUG: this fits on the train split but then scores on that SAME train split, so the
    R^2 is optimistic — a random, signal-free layer will look informative. The fix:
    predict and score on the held-out rows (`x[cut:]`, `target[cut:]`).
    """
    n, cut = len(feats), len(feats) // 2
    x = np.concatenate([feats, np.ones((n, 1), np.float64)], axis=1)
    xtr, ytr = x[:cut], target[:cut].astype(np.float64)
    w = np.linalg.solve(xtr.T @ xtr + RIDGE * np.eye(x.shape[1]), xtr.T @ ytr)
    x_score, y_score = xtr, ytr
    pred = x_score @ w
    resid = ((y_score - pred) ** 2).sum()
    total = ((y_score - y_score.mean()) ** 2).sum()
    return float(1.0 - resid / total) if total > 0 else 0.0


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((200, 32))             # random features: NO signal about target
    target = rng.standard_normal(200)                  # independent of feats
    r2 = probe_r2(feats, target)
    print(f"random-feature probe R^2: {r2:.3f}")
    print("FIXED (held-out ~0)" if r2 < 0.2 else "STILL LEAKING — a signal-free layer should NOT score this high")
