"""SUGGESTED exercise candidate (humans promote) — code-completion, ch5.1.

Objective tested: the one reshape that turns an image into ViT tokens. A ViT sees no picture —
it sees a SEQUENCE of flat patch vectors. Your job: cut a batch of (B, H, W, C) images into a
GRID x GRID grid of PATCH x PATCH x C patches and flatten each to a vector, returning
(B, GRID*GRID, PATCH*PATCH*C). The grid must be row-major: token t = (grid_row, grid_col),
and EACH output row must be ONE spatially-contiguous patch.

The trap (this is the bug ex2 makes you diagnose): to keep a patch contiguous you must split
H -> (grid_row, patch_row) and W -> (grid_col, patch_col), then PERMUTE the two grid axes in
front of the two pixel axes BEFORE flattening. Skip the permute and the grid axis interleaves
with the pixel axis — every "patch" becomes a scramble. It will still have the right SHAPE, so
nothing errors; that is exactly why the bug is dangerous.

Implement `patchify` below (pure numpy, no torch needed), then run the checks:
    pytest curriculum/phase5_practitioner/ch5.1_vit/exercises/suggested/checks.py -k ex3
Estimated learner time: 15 minutes.
"""

import numpy as np

PATCH = 8   # pixels per patch side
GRID = 8    # patches per side (so images are 64x64 and there are 64 tokens)


def patchify(images: np.ndarray, patch: int = PATCH) -> np.ndarray:
    """(B, H, W, C) -> (B, (H//patch)*(W//patch), patch*patch*C).

    Each output row is one contiguous patch, in row-major grid order. Remove the
    NotImplementedError and write it. HINT: images.reshape(B, gh, patch, gw, patch, C),
    then np.transpose to put (gh, gw) before (patch, patch), then reshape to (B, gh*gw, -1).
    """
    raise NotImplementedError("write patchify: reshape -> transpose the grid axes to the front -> reshape")
