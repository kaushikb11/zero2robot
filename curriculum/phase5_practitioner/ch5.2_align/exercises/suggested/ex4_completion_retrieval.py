"""SUGGESTED exercise candidate (humans promote) — code-completion, ch5.2.

Implement the MEASUREMENT: retrieval@1. Given text-query embeddings and a gallery of
image embeddings (both L2-normalized, so a dot product is a cosine), rank the gallery for
each query, take the top-1, and report the fraction whose CLASS matches the query's class.
This is exactly how align.py turns "a shared embedding space" into a number you can trust.

The self-contained check compares your accuracy to a reference on a fixed fixture, within
meta.yaml's abs_tol. Fill in `retrieval_at1`. Do NOT change the signature.
"""

import numpy as np


def retrieval_at1(text_emb: np.ndarray, image_emb: np.ndarray,
                  gal_cls: np.ndarray, qry_cls: np.ndarray) -> float:
    """text_emb: (Q, D) query embeddings. image_emb: (G, D) gallery embeddings (both
    L2-normalized). gal_cls: (G,) gallery class ids. qry_cls: (Q,) query class ids.
    Return the fraction of queries whose TOP-1 gallery image has the same class.

    Steps:
      1) sims = text_emb @ image_emb.T          # (Q, G) cosine of each query to each gallery item
      2) top1 = sims.argmax(axis=1)             # index of the best gallery item per query
      3) return float((gal_cls[top1] == qry_cls).mean())
    """
    raise NotImplementedError("implement retrieval@1 — see the docstring")
