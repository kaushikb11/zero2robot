"""SUGGESTED exercise candidate (humans promote) — code-completion, ch5.3.

The aligned encoder is the whole reason pixel-BC controls, and its objective is a symmetric
InfoNCE (the ch5.2 mechanism): given a batch of image features and their partner (state)
features, pull each image onto its OWN partner and push it off every other example's. The
matches are the diagonal of the batch-vs-batch similarity matrix.

The contract (exactly what pixels.py's `align_encoder` does):
  - L2-normalize each image feature and each partner feature (row-wise).
  - similarity logits = (image @ partner.T) / temperature      -> (B, B)
  - the correct partner for row i is column i (labels = 0..B-1)
  - loss = 0.5 * ( cross_entropy(logits, labels)          # image -> partner
                 + cross_entropy(logits.T, labels) )       # partner -> image

Implement `infonce` below. The check pins it against a reference on a fixed batch.

    pytest curriculum/phase5_practitioner/ch5.3_pixels/exercises/suggested/checks.py -k ex3
"""

import torch
import torch.nn.functional as F  # noqa: F401  (the learner completes the loss using F)

METADATA = {"type": "code-completion", "chapter": "ch5.3-pixels"}


def infonce(image_feat: torch.Tensor, partner_feat: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric InfoNCE over a batch of (image, partner) feature pairs.

    image_feat, partner_feat: (B, dim). Returns a scalar loss. The i-th image matches the
    i-th partner; every other pair in the batch is a negative.
    """
    raise NotImplementedError("implement symmetric InfoNCE — see the contract in the docstring")
    # Hints:
    #   img = F.normalize(image_feat, dim=1); par = F.normalize(partner_feat, dim=1)
    #   logits = img @ par.T / temperature
    #   labels = torch.arange(len(image_feat), device=image_feat.device)
    #   return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
