"""Export the trained ch1.4 diffusion denoiser to ONNX under tensor contract v2.

diffusion.py trains an eps-prediction PushT policy and writes the denoiser CORE to
outputs/ch1.4-diffusion/diffusion_denoiser.onnx (3 inputs: noisy_action, timestep,
observation -> predicted_noise) — but with NO contract metadata and an int64
timestep, so the browser cannot drive it. That raw ONNX is the source of truth for
the weights; this is a FOLLOW-UP script (kept out of diffusion.py's readable loop,
per the embed.yaml contract note) that turns it into a contract-v2 artifact the
sampler-aware runtime can load:

  - rebuild diffusion.py's Denoiser and load the trained weights straight from the
    raw ONNX initializers (net.*, obs_min, obs_range) — the obs normalization is
    baked inside the net, exactly as diffusion.py exports it;
  - recompute the per-dim action mean/std from the SAME deterministic demos
    diffusion.py trained on (the un-standardization stats the runtime needs —
    diffusion.py applies them OUTSIDE the net, so they must ship as metadata);
  - recompute the cosine noise schedule (betas) EXACTLY as diffusion.py's
    make_schedule non-broken branch — the ddpm reverse loop consumes it;
  - export with export_sampler_policy(sampler="ddpm", ...) (stamps
    z2r_contract_version=v2 + the io spec + z2r_sampler=ddpm + betas/x0_clip) and
    prove torch==onnx parity on random (point, timestep, obs) BEFORE the file ships;
  - copy the .onnx to site/public/models/ (git-ignored, provisioned like
    bc_policy.onnx / flow_velocity.onnx; never committed).

Contract v2 carries flow_time as a float32 scalar; diffusion's timestep is an
integer STEP INDEX, carried through that same float input (sinusoidal_embed does
t.float() internally, so an integer-valued float is byte-identical). The reverse
DDPM loop lives in the RUNTIME (playground/src/policy/sampler.ts ddpmSample),
proven equal to diffusion.py's p_sample_loop by the JS-vs-Python parity check.

Run with the repo venv, from the repo root (same env diffusion.py trained under):
  HF_HUB_OFFLINE=1 HF_TOKEN= .venv/bin/python \
    curriculum/phase1_imitation/ch1.4_diffusion/export_diffusion_onnx.py

Prereq: diffusion.py has been trained so outputs/ch1.4-diffusion/{diffusion_denoiser.onnx,demos} exist:
  HF_HUB_OFFLINE=1 HF_TOKEN= .venv/bin/python \
    curriculum/phase1_imitation/ch1.4_diffusion/diffusion.py --seed 0 --device cpu
"""

import argparse
import math
import os
import shutil
import sys
from pathlib import Path

# Same offline posture diffusion.py's Colab/CI runs under; harmless if already set.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import onnx
import torch
import torch.nn as nn
from onnx import numpy_helper

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from curriculum.common.envs.pusht import PushTEnv  # noqa: E402
from curriculum.common.export_onnx import (  # noqa: E402
    SAMPLER_DDPM,
    assert_sampler_parity,
    export_sampler_policy,
)

OBS_DIM, ACT_DIM = PushTEnv.OBS_DIM, PushTEnv.ACT_DIM  # 10, 2
TIME_DIM = 32       # must match diffusion.py.TIME_DIM
X0_CLIP = 3.0       # must match diffusion.py.X0_CLIP (the manifold clamp during sampling)

# The site fetches this path (git-ignored, provisioned like bc_policy.onnx).
SITE_MODEL = REPO_ROOT / "site" / "public" / "models" / "diffusion_denoiser.onnx"


def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Byte-for-byte diffusion.py.sinusoidal_embed — the denoiser's time features.
    Note: NO time-scale factor (unlike flow.py); t is the integer step index."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class Denoiser(nn.Module):
    """Byte-for-byte diffusion.py.Denoiser (conditioned variant). Redefined here
    because diffusion.py runs argparse/training at import time and cannot be
    imported; only the parameter/buffer layout matters — we load the trained
    weights from diffusion.py's raw ONNX, then re-export the CORE under contract
    v2. forward takes a FLOAT timestep here (the graph input is float32); the
    embedding does t.float() so an integer-valued float matches the training int."""

    def __init__(self, x_dim: int, cond_dim: int, hidden: int):
        super().__init__()
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + TIME_DIM + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )
        # Registered so load_state_dict restores the baked-in obs normalization.
        self.register_buffer("obs_min", torch.zeros(cond_dim))
        self.register_buffer("obs_range", torch.ones(cond_dim))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        parts = [x_t, sinusoidal_embed(t, TIME_DIM)]
        parts.append((2.0 * (cond - self.obs_min) / self.obs_range - 1.0).clamp(-1.0, 1.0))
        return self.net(torch.cat(parts, dim=1))


def cosine_betas(steps: int) -> np.ndarray:
    """Byte-for-byte diffusion.py.make_schedule cosine (non-broken) branch, betas
    only. float32 throughout (torch default dtype), exactly as the chapter."""
    u = torch.linspace(0, steps, steps + 1) / steps
    acp_full = torch.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2
    acp_full = acp_full / acp_full[0].clone()
    betas = (1 - acp_full[1:] / acp_full[:-1]).clamp(1e-8, 0.999)
    return betas.numpy()


def load_weights_from_onnx(net: Denoiser, raw_onnx: Path) -> None:
    """Load diffusion.py's trained weights straight from its raw ONNX initializers.
    torch.onnx.export (dynamo=False) names initializers by their state_dict keys
    (net.0.weight, ..., obs_min, obs_range), so this is a direct, strict load."""
    proto = onnx.load(str(raw_onnx))
    state = {init.name: torch.from_numpy(numpy_helper.to_array(init).copy())
             for init in proto.graph.initializer}
    missing, _ = net.load_state_dict(state, strict=False)
    if missing:
        raise SystemExit(f"weights missing from {raw_onnx}: {missing}")


def action_stats(demos: Path) -> tuple[np.ndarray, np.ndarray]:
    """Recompute the per-dim action mean/std EXACTLY as diffusion.py's data region:
    load the same deterministic demos, standardize stats over all frames."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # heavy import

    frames = LeRobotDataset("local/pusht-demos", root=demos).hf_dataset.with_format("numpy")
    actions = np.stack(frames["action"]).astype(np.float32)  # (N, 2)
    act_mean = actions.mean(0)
    act_std = np.where(actions.std(0) < 1e-4, np.float32(1.0), actions.std(0))
    return act_mean, act_std


def assert_ddpm_step_parity(net: Denoiser, onnx_path: Path, num_steps: int,
                            n: int = 8, seed: int = 0, tol: float = 1e-4) -> float:
    """Extra check: torch==onnx at the INTEGER step indices the ddpm loop actually
    feeds (0..num_steps-1), which assert_sampler_parity's [0,1] draws never reach."""
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    max_delta = 0.0
    with torch.no_grad():
        for step in list(range(num_steps))[:: max(1, num_steps // n)]:
            px = rng.standard_normal((1, ACT_DIM)).astype(np.float32)
            po = rng.standard_normal((1, OBS_DIM)).astype(np.float32)
            pt = np.array([float(step)], dtype=np.float32)
            torch_v = net(torch.from_numpy(px), torch.from_numpy(pt), torch.from_numpy(po)).numpy()
            onnx_v = session.run(None, {"point": px, "flow_time": pt, "observation": po})[0]
            max_delta = max(max_delta, float(np.abs(torch_v - onnx_v).max()))
    if not max_delta < tol:
        raise AssertionError(f"ddpm integer-step parity FAILED: {max_delta:.3e} >= {tol:.1e}")
    return max_delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("outputs/ch1.4-diffusion"),
                        help="diffusion.py output dir (holds diffusion_denoiser.onnx + demos/)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output .onnx (default: <run>/diffusion_denoiser_v2.onnx)")
    parser.add_argument("--num_steps", type=int, default=100,
                        help="default denoising_steps stamped into the contract (diffusion.py default)")
    parser.add_argument("--no-copy", dest="copy_site", action="store_false", default=True,
                        help="skip copying the .onnx into site/public/models/")
    args = parser.parse_args()

    raw_onnx = args.run / "diffusion_denoiser.onnx"
    demos = args.run / "demos"
    out = args.out or (args.run / "diffusion_denoiser_v2.onnx")
    if not raw_onnx.is_file():
        sys.exit(f"no raw denoiser ONNX at {raw_onnx} — train it first:\n"
                 f"  HF_HUB_OFFLINE=1 HF_TOKEN= .venv/bin/python "
                 f"curriculum/phase1_imitation/ch1.4_diffusion/diffusion.py --seed 0 --device cpu")
    if not (demos / "meta" / "info.json").is_file():
        sys.exit(f"no demos at {demos} — diffusion.py writes them next to diffusion_denoiser.onnx.")

    # Rebuild the net at diffusion.py's hidden width (read off the first Linear) and
    # load the trained weights from the raw ONNX.
    proto = onnx.load(str(raw_onnx))
    inits = {i.name: numpy_helper.to_array(i) for i in proto.graph.initializer}
    hidden = int(inits["net.0.weight"].shape[0])
    net = Denoiser(ACT_DIM, OBS_DIM, hidden)
    load_weights_from_onnx(net, raw_onnx)

    act_mean, act_std = action_stats(demos)
    betas = cosine_betas(args.num_steps)
    print(f"action stats (from {demos.name}): mean={act_mean.tolist()} std={act_std.tolist()}")
    print(f"cosine schedule: {args.num_steps} betas in [{betas.min():.2e}, {betas.max():.3f}]")

    onnx_path = export_sampler_policy(
        net, OBS_DIM, ACT_DIM, args.num_steps, act_mean, act_std, out,
        sampler=SAMPLER_DDPM, betas=betas, x0_clip=X0_CLIP,
    )
    # Prove torch and onnxruntime agree on the denoiser BEFORE it ships.
    delta = assert_sampler_parity(net, onnx_path, OBS_DIM, ACT_DIM)
    step_delta = assert_ddpm_step_parity(net, onnx_path, args.num_steps)
    print(f"exported {onnx_path} (contract v2 ddpm, hidden={hidden}, num_steps={args.num_steps}, "
          f"x0_clip={X0_CLIP}); torch/onnx parity {delta:.2e} (uniform-t) / {step_delta:.2e} (integer-t)")

    if args.copy_site:
        SITE_MODEL.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(onnx_path, SITE_MODEL)
        print(f"provisioned {SITE_MODEL} (git-ignored — not committed)")


if __name__ == "__main__":
    main()
