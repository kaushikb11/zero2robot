"""SUGGESTED exercise candidate (humans promote) — code-completion, ch1.7.

The leakage probe in vla_data.py turns each instruction's token ids into a
BAG-OF-WORDS vector: a length-`vocab_size` count of how many times each id appears,
IGNORING the PAD id (pad carries no information and would otherwise dominate). That
vector is the only thing the linear probe sees — so if the instruction words encode
the action (the `--break leak` case), this bag is where it leaks in.

    bag[v] = number of times token id v appears in `tokens`, for v != pad_id
             (pad slots are skipped entirely)

YOUR JOB: implement `bag_of_words` from the rule above. `tokens` is a (T,) int array
of ids; return a (vocab_size,) float64 count vector with the pad id's slot left at 0.

    pytest curriculum/phase1_imitation/ch1.7_vla_data/exercises/suggested/checks.py -k ex4
"""

import numpy as np

METADATA = {"type": "code-completion", "chapter": "ch1.7-vla-data"}


def bag_of_words(tokens: np.ndarray, vocab_size: int, pad_id: int = 0) -> np.ndarray:
    """Count token ids into a (vocab_size,) float64 vector, skipping `pad_id`.

    Replace the NotImplementedError: start from zeros(vocab_size), and for each id in
    `tokens` that is not `pad_id`, increment that id's slot by 1.
    """
    raise NotImplementedError("implement the bag-of-words count (see the module docstring)")
