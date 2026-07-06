"""Export a trained torch policy to ONNX under tensor contract v1.

TENSOR CONTRACT v1 (mirrored by playground/src/policy/contracts.ts — change
only in a coordinated cross-package PR, see 03-engineering/artifact-pipeline.md):

  input : name "observation", float32, shape [1, obs_dim]
  output: name "action",      float32, shape [1, act_dim]
  ONNX metadata_props:
    z2r_contract_version = "v1"
    z2r_obs_dim          = "<int>"   (decimal string, matches graph shape)
    z2r_act_dim          = "<int>"

The model interface is deliberately trivial: model(obs) where obs is a
float32 tensor of shape [1, obs_dim], returning [1, act_dim]. Policies with
richer interfaces (action chunks, images) get a later contract version.

Pipeline: train -> export_policy(...) -> assert_parity(...) -> drag the .onnx
into the playground, which refuses mismatched contract versions with a
human-readable error.

CLI: inspect an exported file's contract metadata (exit 1 if non-conformant):

    python curriculum/common/export_onnx.py path/to/policy.onnx
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import torch

CONTRACT_VERSION = "v1"
INPUT_NAME = "observation"
OUTPUT_NAME = "action"
META_VERSION_KEY = "z2r_contract_version"
META_OBS_DIM_KEY = "z2r_obs_dim"
META_ACT_DIM_KEY = "z2r_act_dim"

# --- contract v2 (sampler-aware runtime; see below) --------------------------
SAMPLER_CONTRACT_VERSION = "v2"
POINT_NAME = "point"
TIME_NAME = "flow_time"
VELOCITY_NAME = "velocity"
META_SAMPLER_KEY = "z2r_sampler"
META_NUM_STEPS_KEY = "z2r_num_steps"
META_ACT_MEAN_KEY = "z2r_act_mean"
META_ACT_STD_KEY = "z2r_act_std"
META_BETAS_KEY = "z2r_betas"        # ddpm ONLY: the forward-noise schedule
META_X0_CLIP_KEY = "z2r_x0_clip"    # ddpm ONLY: clamp on the predicted clean sample
SAMPLER_SPACE = "flow-euler"        # ch1.5 flow: schedule-free forward-Euler ODE
SAMPLER_DDPM = "ddpm"               # ch1.4 diffusion: ancestral DDPM reverse loop
SAMPLER_SPACES = (SAMPLER_SPACE, SAMPLER_DDPM)


def export_policy(
    model: torch.nn.Module, obs_dim: int, act_dim: int, out_path: str | Path
) -> Path:
    """Export `model` to `out_path` as contract-v1 ONNX. Returns out_path.

    Exports on CPU in eval mode with a fixed zeros dummy input (the export
    only traces shapes; assert_parity checks values afterwards). The model is
    moved to CPU in place — export after training is done.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = model.eval().to("cpu")
    dummy = torch.zeros(1, obs_dim, dtype=torch.float32)
    with torch.no_grad():
        out = model(dummy)
    if tuple(out.shape) != (1, act_dim):
        raise ValueError(
            f"model(obs[1, {obs_dim}]) returned shape {tuple(out.shape)}, "
            f"expected [1, {act_dim}] — contract v1 wants model(obs)->action"
        )

    torch.onnx.export(
        model,
        (dummy,),
        str(out_path),
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamo=False,  # classic exporter: static [1, dim] shapes per contract v1
    )

    # Stamp the tensor contract into ModelProto.metadata_props.
    proto = onnx.load(str(out_path))
    for key, value in (
        (META_VERSION_KEY, CONTRACT_VERSION),
        (META_OBS_DIM_KEY, str(obs_dim)),
        (META_ACT_DIM_KEY, str(act_dim)),
    ):
        entry = proto.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out_path))
    return out_path


# ===========================================================================
# TENSOR CONTRACT v2 — the SAMPLER-AWARE runtime (generative policies: ch1.4
# diffusion, ch1.5 flow). Mirrored by playground/src/policy/contracts.ts
# (validateSamplerContract) + sampler.ts (eulerSample) — change only in a
# coordinated cross-package PR.
#
#   inputs :
#     "point"       float32 [1, act_dim]   the current point x_t (standardized action space)
#     "flow_time"   float32 [1]            the scalar integration time t in [0, 1]
#     "observation" float32 [1, obs_dim]   the conditioning obs (obs norm baked INSIDE the net)
#   output:
#     "velocity"    float32 [1, act_dim]   the predicted velocity at (point, t | obs)
#   ONNX metadata_props:
#     z2r_contract_version = "v2"
#     z2r_obs_dim, z2r_act_dim = "<int>"   (match the graph shapes)
#     z2r_sampler   = "flow-euler"
#     z2r_num_steps = "<int>"              (default velocity-net evals per action)
#     z2r_act_mean  = "m0,m1,..."          (per-dim mean to un-standardize the sample)
#     z2r_act_std   = "s0,s1,..."          (per-dim std  to un-standardize the sample)
#
# Unlike v1 (a single stateless model(obs)->action step), the browser RUNS the
# sampler: seeded noise -> num_steps forward-Euler velocity evals -> denorm with
# act_mean/act_std. Here we only export the velocity-net CORE + stamp the sampler
# metadata; the integrator lives in the runtime (proven by the JS/Python parity
# check). REUSABLE for ch1.4: a diffusion denoiser fits the same 3-input core.
# ===========================================================================


def export_sampler_policy(
    velocity_net: torch.nn.Module,
    obs_dim: int,
    act_dim: int,
    num_steps: int,
    act_mean,
    act_std,
    out_path: str | Path,
    sampler: str = SAMPLER_SPACE,
    betas=None,
    x0_clip: float | None = None,
) -> Path:
    """Export a velocity/denoiser net to `out_path` as contract-v2 ONNX.

    `net(point[1,act], flow_time[1], observation[1,obs]) -> velocity[1,act]` — the
    shared 3-in/1-out core of flow.py's VelocityNet.forward AND diffusion.py's
    Denoiser.forward (for ddpm the output is the predicted noise eps, and
    flow_time carries the integer step index as a float; its embedding does
    t.float() internally so an integer-valued float is identical). act_mean/act_std
    are the per-dim un-standardization stats (length act_dim) the runtime applies
    after sampling.

    For `sampler="ddpm"` the reverse loop needs the forward-noise schedule: pass
    `betas` (length num_steps) and `x0_clip` (diffusion.py X0_CLIP), stamped as
    z2r_betas / z2r_x0_clip. flow-euler needs neither. Exports on CPU in eval mode
    with fixed dummy inputs; returns out_path. (assert_sampler_parity checks values.)
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if sampler not in SAMPLER_SPACES:
        raise ValueError(f"sampler must be one of {SAMPLER_SPACES}, got {sampler!r}")
    act_mean = np.asarray(act_mean, dtype=np.float64).reshape(-1)
    act_std = np.asarray(act_std, dtype=np.float64).reshape(-1)
    if act_mean.shape != (act_dim,) or act_std.shape != (act_dim,):
        raise ValueError(
            f"act_mean/act_std must have shape ({act_dim},), got "
            f"{act_mean.shape}/{act_std.shape}"
        )
    if not (act_std > 0).all():
        raise ValueError(f"act_std must be strictly positive, got {act_std.tolist()}")
    if not (isinstance(num_steps, int) and num_steps >= 1):
        raise ValueError(f"num_steps must be a positive int, got {num_steps!r}")

    if sampler == SAMPLER_DDPM:
        if betas is None or x0_clip is None:
            raise ValueError("sampler='ddpm' requires betas (length num_steps) and x0_clip")
        betas = np.asarray(betas, dtype=np.float64).reshape(-1)
        if betas.shape != (num_steps,):
            raise ValueError(
                f"betas must have shape ({num_steps},) to match num_steps, got {betas.shape}"
            )
        if not ((betas > 0) & (betas < 1)).all():
            raise ValueError(f"betas must lie in (0, 1), got range [{betas.min()}, {betas.max()}]")
        if not float(x0_clip) > 0:
            raise ValueError(f"x0_clip must be a positive float, got {x0_clip!r}")

    velocity_net = velocity_net.eval().to("cpu")
    dummy = (
        torch.zeros(1, act_dim, dtype=torch.float32),  # point
        torch.zeros(1, dtype=torch.float32),           # flow_time (scalar per sample)
        torch.zeros(1, obs_dim, dtype=torch.float32),  # observation
    )
    with torch.no_grad():
        out = velocity_net(*dummy)
    if tuple(out.shape) != (1, act_dim):
        raise ValueError(
            f"net(point[1,{act_dim}], t[1], obs[1,{obs_dim}]) returned "
            f"{tuple(out.shape)}, expected [1, {act_dim}] — contract v2 wants "
            f"net(point, flow_time, observation)->velocity"
        )

    torch.onnx.export(
        velocity_net,
        dummy,
        str(out_path),
        input_names=[POINT_NAME, TIME_NAME, INPUT_NAME],
        output_names=[VELOCITY_NAME],
        dynamo=False,  # classic exporter: static [1, dim] shapes per contract v2
    )

    proto = onnx.load(str(out_path))
    csv = lambda a: ",".join(repr(float(v)) for v in a)  # noqa: E731  full-precision round-trip
    meta_items = [
        (META_VERSION_KEY, SAMPLER_CONTRACT_VERSION),
        (META_OBS_DIM_KEY, str(obs_dim)),
        (META_ACT_DIM_KEY, str(act_dim)),
        (META_SAMPLER_KEY, sampler),
        (META_NUM_STEPS_KEY, str(num_steps)),
        (META_ACT_MEAN_KEY, csv(act_mean)),
        (META_ACT_STD_KEY, csv(act_std)),
    ]
    if sampler == SAMPLER_DDPM:
        meta_items += [
            (META_BETAS_KEY, csv(betas)),
            (META_X0_CLIP_KEY, repr(float(x0_clip))),
        ]
    for key, value in meta_items:
        entry = proto.metadata_props.add()
        entry.key = key
        entry.value = value
    onnx.checker.check_model(proto)
    onnx.save(proto, str(out_path))
    return out_path


def assert_sampler_parity(
    velocity_net: torch.nn.Module,
    onnx_path: str | Path,
    obs_dim: int,
    act_dim: int,
    n: int = 32,
    seed: int = 0,
    tol: float = 1e-4,
) -> float:
    """Assert torch and onnxruntime agree on the VELOCITY NET over n seeded random
    (point, flow_time, observation) triples; return the max |velocity delta|.
    Proves the serialization path before the file ships (v2 sibling of
    assert_parity). Raises AssertionError on tol breach.
    """
    import onnxruntime as ort  # heavy; only when actually checking

    velocity_net = velocity_net.eval().to("cpu")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    max_delta = 0.0
    with torch.no_grad():
        for _ in range(n):
            px = rng.standard_normal((1, act_dim)).astype(np.float32)
            pt = rng.uniform(0.0, 1.0, size=(1,)).astype(np.float32)
            po = rng.standard_normal((1, obs_dim)).astype(np.float32)
            torch_v = velocity_net(
                torch.from_numpy(px), torch.from_numpy(pt), torch.from_numpy(po)
            ).cpu().numpy()
            onnx_v = session.run(
                None, {POINT_NAME: px, TIME_NAME: pt, INPUT_NAME: po}
            )[0]
            max_delta = max(max_delta, float(np.abs(torch_v - onnx_v).max()))
    if not max_delta < tol:
        raise AssertionError(
            f"sampler parity FAILED for {onnx_path}: max velocity delta "
            f"{max_delta:.3e} >= tol {tol:.1e} over {n} seeded triples (seed={seed})"
        )
    return max_delta


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect an exported policy's tensor-contract metadata.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("onnx_path", help="path to an exported .onnx policy")
    args = parser.parse_args()

    proto = onnx.load(args.onnx_path)
    meta = {p.key: p.value for p in proto.metadata_props}
    version = meta.get(META_VERSION_KEY)
    obs_dim = meta.get(META_OBS_DIM_KEY)
    act_dim = meta.get(META_ACT_DIM_KEY)

    print(f"{args.onnx_path}")
    print(f"  {META_VERSION_KEY} = {version}")
    print(f"  {META_OBS_DIM_KEY} = {obs_dim}")
    print(f"  {META_ACT_DIM_KEY} = {act_dim}")

    if version == SAMPLER_CONTRACT_VERSION:
        sampler = meta.get(META_SAMPLER_KEY)
        keys = [META_SAMPLER_KEY, META_NUM_STEPS_KEY, META_ACT_MEAN_KEY, META_ACT_STD_KEY]
        if sampler == SAMPLER_DDPM:
            keys += [META_BETAS_KEY, META_X0_CLIP_KEY]
        for k in keys:
            print(f"  {k} = {meta.get(k)}")
        if not (obs_dim or "").isdigit() or not (act_dim or "").isdigit():
            print("NOT CONFORMANT: obs/act dims missing or not decimal integers.")
            return 1
        if sampler not in SAMPLER_SPACES:
            print(f"NOT CONFORMANT: expected {META_SAMPLER_KEY} in {SAMPLER_SPACES}.")
            return 1
        if not (meta.get(META_NUM_STEPS_KEY) or "").isdigit():
            print(f"NOT CONFORMANT: {META_NUM_STEPS_KEY} missing or not a decimal integer.")
            return 1
        if sampler == SAMPLER_DDPM and (not meta.get(META_BETAS_KEY) or not meta.get(META_X0_CLIP_KEY)):
            print(f"NOT CONFORMANT: {SAMPLER_DDPM} requires {META_BETAS_KEY} + {META_X0_CLIP_KEY}.")
            return 1
        print(f"conformant: tensor contract v2 (sampler-aware, {sampler})")
        return 0

    if version != CONTRACT_VERSION:
        print(
            f"NOT CONFORMANT: expected {META_VERSION_KEY}={CONTRACT_VERSION!r}, "
            f"found {version!r}. Re-export with curriculum/common/export_onnx.py."
        )
        return 1
    if not (obs_dim or "").isdigit() or not (act_dim or "").isdigit():
        print("NOT CONFORMANT: obs/act dims missing or not decimal integers.")
        return 1
    print("conformant: tensor contract v1")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
