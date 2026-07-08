"""SUGGESTED exercise candidate (humans promote) — code-completion (fast), ch3.8.

The core move of "reading a checkpoint" is the linear probe: fit a linear map from a
frozen layer's activations to a KNOWN factor on a train split, then score it on a
held-out split. probe.py runs this closed-form (ridge normal equations) so it is
deterministic — no optimizer, no seed to chase. This exercise asks you to write it.

The contract (the same math probe.py's `linear_probe` runs, regression variant):

    x   = [feats | 1]                       # append a bias column
    w   = solve(Xtr^T Xtr + ridge*I,  Xtr^T ytr)   # ridge normal equations, TRAIN split
    yte_pred = Xte @ w                       # predict on the HELD-OUT split
    R^2 = 1 - SS_res/SS_tot                  # on the held-out split

Split the rows 50/50 (`cut = len(feats)//2`; train = first half, held-out = second),
fit on train, score R^2 on held-out. Complete `linear_probe_r2` so the check passes (it
compares against a reference on features that DO linearly carry the target).

    pytest curriculum/phase3_advanced/ch3.8_frontier/exercises/suggested/checks.py -k ex3
"""

import numpy as np

METADATA = {"type": "code-completion", "chapter": "ch3.8-frontier", "fast": True}


def linear_probe_r2(feats: np.ndarray, target: np.ndarray, ridge: float) -> float:
    """Held-out R^2 of a closed-form ridge linear probe.

    feats  : (N, D) frozen layer activations
    target : (N,)   the known factor to recover
    ridge  : lambda for the ridge normal equations (stabilizes the solve)
    returns: R^2 on the held-out (second) half of the rows

    TODO: implement the four steps from the docstring — append a bias column, solve the
    ridge normal equations on the TRAIN split, predict on the HELD-OUT split, and return
    its R^2 (1 - SS_res/SS_tot). Use np.linalg.solve; do NOT score on the train rows.
    """
    raise NotImplementedError("implement the held-out ridge linear probe (see the docstring)")


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((200, 8))
    target = feats @ rng.standard_normal(8) + 0.1 * rng.standard_normal(200)  # linear in feats
    print(f"held-out R^2 (should be high, ~0.9+): {linear_probe_r2(feats, target, 1.0):.3f}")
