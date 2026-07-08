"""SUGGESTED exercise candidate (humans promote) — code-completion, ch5.2.

Implement the heart of the chapter: SYMMETRIC InfoNCE over a batch's image-text cosine
matrix. You are given a batch of L2-normalized image embeddings and text embeddings and
a learned log-temperature. Build the B x B matrix of scaled cosines, then average the
cross-entropy in BOTH directions — image->text (each row's correct caption) AND
text->image (each column's correct frame) — against the arange labels.

This is the CLIP training objective (openai/CLIP model.py: logits_per_image /
logits_per_text, two cross-entropies). The self-contained check compares your loss to a
reference on a fixed fixture, within meta.yaml's abs_tol.

Fill in `symmetric_info_nce`. Do NOT change the signature.
"""

import torch
import torch.nn.functional as F  # noqa: F401  (the learner completes the loss using F)


def symmetric_info_nce(img_e: torch.Tensor, txt_e: torch.Tensor,
                       logit_scale: torch.Tensor) -> torch.Tensor:
    """img_e, txt_e: (B, D) L2-normalized. logit_scale: scalar log-temperature.
    Return 0.5 * (CE(image->text) + CE(text->image)) over the scaled cosine matrix.

    Steps:
      1) scale = logit_scale.exp()                          # temperature^-1
      2) logits = scale * img_e @ txt_e.t()                 # (B, B): row i vs every caption
      3) labels = torch.arange(B)                           # the positive for row i is column i
      4) return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))
    """
    raise NotImplementedError("implement symmetric InfoNCE — see the docstring")
