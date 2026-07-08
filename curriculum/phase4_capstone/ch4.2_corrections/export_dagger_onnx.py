"""Export the trained ch4.2 DAgger best-round policy to ONNX (tensor contract v1).

dagger.py runs DAgger on PushT and `torch.save`s the BEST round's policy (Ross et
al.: return the best over rounds, not the last) to outputs/ch4.2-corrections/:

  dagger_policy.pt   the best-round BCPolicy (a WHOLE-MODULE torch.save, plus a
                     dagger_policy.ts.pt TorchScript trace for the browser-teleop
                     follow-up). This exporter turns it into the contract-v1 ONNX
                     the live DAgger PushT toy drives with.

The policy is the ch1.1 behavior-cloning MLP (obs[10] -> action[2]) with the
per-dim normalization living INSIDE the module as buffers, so the exported ONNX is
a self-contained stateless map — that IS tensor contract v1 (obs[1,10] ->
action[1,2]) with no wrapper (see curriculum/common/export_onnx.py). This is a
FOLLOW-UP script (kept out of dagger.py's readable loop, per the embed.yaml note
that the ONNX export is wired when the chapter graduates from spike to drop);
dagger.py is UNCHANGED — it already saves dagger_policy.pt.

Run with the repo venv, from the repo root:
  .venv/bin/python curriculum/phase4_capstone/ch4.2_corrections/export_dagger_onnx.py
Smoke (no checkpoint; random-init policy, proves the export+parity path):
  .venv/bin/python curriculum/phase4_capstone/ch4.2_corrections/export_dagger_onnx.py --smoke
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from curriculum.common.assert_parity import assert_parity  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv  # noqa: E402
from curriculum.common.export_onnx import export_policy  # noqa: E402

OBS_DIM, ACT_DIM = PushTEnv.OBS_DIM, PushTEnv.ACT_DIM  # 10, 2 — PushT, not pusher_reach

# The one demo panel: checkpoint stem -> exported filename (also the site path).
ARMS = {"dagger_policy": "dagger.onnx"}
SITE_MODELS = REPO_ROOT / "site" / "public" / "models"


class BCPolicy(nn.Module):
    """Byte-for-byte the dagger.py BCPolicy (reactive 3-layer MLP with the ch1.1
    per-dim normalization carried as buffers), redefined here because dagger.py
    runs argparse/training at import time and cannot be imported as a module. The
    buffers are loaded with the weights, so the exported ONNX bakes normalization
    IN — a self-contained obs[1,10] -> action[1,2] contract-v1 map, no wrapper."""

    def __init__(self, hidden_dim: int, stats: dict[str, np.ndarray]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, ACT_DIM),
        )
        for name, value in stats.items():
            self.register_buffer(name, torch.from_numpy(value))  # saved with the weights, never trained

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        normalized = (2.0 * (observation - self.obs_min) / self.obs_range - 1.0).clamp(-1.0, 1.0)
        return (self.net(normalized) + 1.0) / 2.0 * self.act_range + self.act_min


def _dummy_stats() -> dict[str, np.ndarray]:
    """Identity normalization buffers (right names + shapes) so the module can be
    constructed before load_state_dict overwrites them with the trained stats. For
    the smoke path these ARE the buffers — an identity map still exercises the
    export+parity pipeline (only the weights differ from a real run)."""
    return {
        "obs_min": np.zeros(OBS_DIM, np.float32),
        "obs_range": np.ones(OBS_DIM, np.float32),
        "act_min": np.zeros(ACT_DIM, np.float32),
        "act_range": np.ones(ACT_DIM, np.float32),
    }


def load_policy(ckpt: Path, hidden_dim: int) -> BCPolicy:
    """Rebuild the architecture and load the best-round weights. dagger.py does a
    WHOLE-MODULE torch.save(best_policy, ...); accept that (take its state_dict) or
    a bare state_dict, then load into a fresh in-process BCPolicy for a clean graph."""
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = loaded.state_dict() if isinstance(loaded, nn.Module) else loaded
    policy = BCPolicy(hidden_dim, _dummy_stats())
    policy.load_state_dict(state)  # overwrites the identity buffers with the trained normalization
    return policy


def smoke_policy(seed: int = 0, hidden_dim: int = 64) -> tuple[BCPolicy, int, int]:
    """A random-init policy — no checkpoint, no env rollout needed. Exercises the
    exact BCPolicy architecture + export path in CI (hermetic). Matches the
    test_export_pipeline_smoke.py contract: returns (policy, obs_dim, act_dim)."""
    torch.manual_seed(seed)
    return BCPolicy(hidden_dim, _dummy_stats()).eval(), OBS_DIM, ACT_DIM


def export_arm(policy: BCPolicy, out: Path, onnx_name: str, copy_site: bool) -> None:
    onnx_path = export_policy(policy, OBS_DIM, ACT_DIM, out)
    parity_delta = assert_parity(policy, onnx_path, OBS_DIM)  # torch vs onnxruntime before it ships
    print(f"exported {onnx_path} — torch/onnx parity delta {parity_delta:.2e}")
    if copy_site:
        SITE_MODELS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx_path, SITE_MODELS / onnx_name)
        print(f"provisioned {SITE_MODELS / onnx_name} (git-ignored — not committed)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt_dir", type=Path, default=Path("outputs/ch4.2-corrections"),
                        help="dir holding dagger_policy.pt saved by dagger.py")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/ch4.2-corrections"))
    parser.add_argument("--hidden_dim", type=int, default=256)  # dagger.py default
    parser.add_argument("--smoke", action="store_true",
                        help="skip the checkpoint; export a random-init policy (CI parity smoke)")
    parser.add_argument("--no-copy", dest="copy_site", action="store_false", default=True,
                        help="skip copying the .onnx file into site/public/models/")
    args = parser.parse_args()

    for stem, onnx_name in ARMS.items():
        if args.smoke:
            policy, _, _ = smoke_policy(seed=0, hidden_dim=args.hidden_dim)
        else:
            ckpt = args.ckpt_dir / f"{stem}.pt"
            if not ckpt.is_file():
                sys.exit(f"no checkpoint at {ckpt} — train it first:\n"
                         f"  .venv/bin/python curriculum/phase4_capstone/ch4.2_corrections/dagger.py --seed 0 --device cpu")
            policy = load_policy(ckpt, args.hidden_dim)
        export_arm(policy, args.out_dir / onnx_name, onnx_name, args.copy_site and not args.smoke)


if __name__ == "__main__":
    main()
