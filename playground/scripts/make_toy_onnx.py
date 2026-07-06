"""Generate a toy 2-layer MLP policy in ONNX for the WASM spike.

Emits playground/public/models/toy_policy.onnx conforming to tensor contract v1
via curriculum.common.export_onnx.export_policy — the same stamping path real
checkpoints use, so this toy producer cannot drift from the contract
(playground/src/policy/contracts.ts mirrors it on the browser side).

Run with the repo venv, from anywhere:
  .venv/bin/python playground/scripts/make_toy_onnx.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from curriculum.common.export_onnx import export_policy  # noqa: E402

OBS_DIM = 4   # [pusher_x, pusher_y, box_x, box_y] — placeholder PushT scene
ACT_DIM = 2   # [force_x, force_y]
HIDDEN = 32

OUT_PATH = Path(__file__).resolve().parent.parent / "public" / "models" / "toy_policy.onnx"


class ToyPolicy(nn.Module):
    """Two linear layers with a tanh in between. Randomly initialized —
    this validates the pipeline, not the policy."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, ACT_DIM),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.net(observation)


def main() -> None:
    torch.manual_seed(0)
    model = ToyPolicy()
    export_policy(model, OBS_DIM, ACT_DIM, OUT_PATH)

    # Print a reference output so browser inference can be sanity-checked
    # (assert_parity.py is the systematic version of this for real policies).
    with torch.no_grad():
        ref = model(torch.ones(1, OBS_DIM))
    print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    print(f"reference: f(ones) = {ref.numpy().tolist()}")


if __name__ == "__main__":
    main()
