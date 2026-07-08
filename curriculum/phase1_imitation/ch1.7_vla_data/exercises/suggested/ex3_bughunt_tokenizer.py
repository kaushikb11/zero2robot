"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch1.7.

A word-level tokenizer has ONE job: turn a string into a fixed-length array of ids,
wrapped in the special tokens the model expects. The contract vla_data.py's
Tokenizer promises is exact:

    encode(text) = [BOS, id(w0), id(w1), ..., EOS, PAD, PAD, ...]   length MAX_TOKENS

with BOS FIRST, EOS after the last real word, unknown words -> UNK, and PAD filling
the rest. Get the wrapping wrong and every downstream shape still looks fine — the
arrays are the right length — but the model sees a sentence that starts in the wrong
place, and (worse) a bag-of-words built from it silently miscounts.

Before you read on, look only at `encode` and write one sentence: the arrays are
the right length and every shape downstream still fits — so what does a
bag-of-words built from this output silently miscount, and why will no shape
check flag it?

THE BUG. `encode` below FORGETS to prepend the BOS token: it returns
[id(w0), ..., EOS, PAD, ...]. It still returns length MAX_TOKENS, still runs, still
tokenizes OOV words to UNK. Find it and fix it so the check passes (token[0] must be
BOS, and the EOS must sit right after the last real word).

    pytest curriculum/phase1_imitation/ch1.7_vla_data/exercises/suggested/checks.py -k ex3
"""

import numpy as np

METADATA = {"type": "bug-hunt", "chapter": "ch1.7-vla-data"}

MAX_TOKENS = 12
SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]      # ids 0..3
VOCAB = SPECIALS + ["cube", "left", "push", "target", "the", "to"]  # ids 4..9
STOI = {w: i for i, w in enumerate(VOCAB)}


def encode(text: str) -> np.ndarray:
    """String -> (MAX_TOKENS,) int64 ids: [BOS] words [EOS] then PAD.

    BUG: the BOS token is never prepended, so the sentence starts at its first word
    and every id is shifted one slot to the left. At t=0 the array must hold BOS.
    """
    ids = [STOI.get(w, STOI["<unk>"]) for w in text.split()]
    ids.append(STOI["<eos>"])
    ids = ids[:MAX_TOKENS] + [STOI["<pad>"]] * (MAX_TOKENS - len(ids))
    return np.asarray(ids[:MAX_TOKENS], dtype=np.int64)
