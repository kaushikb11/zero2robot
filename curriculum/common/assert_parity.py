"""Torch-vs-ONNX parity gate for exported policies (artifact-pipeline.md).

The gate: N seeded random observations through the torch model (eval, CPU)
and through onnxruntime; max |action delta| must be < 1e-4. Every chapter
runs this immediately after export_onnx — a policy that fails parity never
reaches the playground.

CLI (run from the repo root):

    python curriculum/common/assert_parity.py policy.onnx policy.ts.pt

policy.ts.pt is a TorchScript module (`torch.jit.script(model).save(...)`): it
carries its own code, so the CLI reloads it in this fresh process with no
access to your model class. A whole-module `torch.save(model, ...)` file also
works IF its defining class is importable here (the CLI falls back to
torch.load) — but prefer the TorchScript file, which never needs the class.
Neither a bare state_dict works: the CLI cannot rebuild your architecture.
obs_dim is read from the ONNX graph input.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def assert_parity(
    model: torch.nn.Module,
    onnx_path: str | Path,
    obs_dim: int,
    n: int = 32,
    seed: int = 0,
    tol: float = 1e-4,
) -> float:
    """Assert torch and ONNX agree on n seeded obs; return the max |delta|.

    Runs the torch model in eval mode on CPU and the ONNX file through
    onnxruntime's CPU provider. Raises AssertionError with the measured delta
    if max |torch_action - onnx_action| >= tol.
    """
    model = model.eval().to("cpu")
    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    rng = np.random.default_rng(seed)

    max_delta = 0.0
    with torch.no_grad():
        for _ in range(n):
            obs = rng.standard_normal((1, obs_dim)).astype(np.float32)
            torch_action = model(torch.from_numpy(obs)).cpu().numpy()
            onnx_action = session.run(None, {input_name: obs})[0]
            delta = float(np.max(np.abs(torch_action - onnx_action)))
            max_delta = max(max_delta, delta)

    if not max_delta < tol:
        raise AssertionError(
            f"parity FAILED for {onnx_path}: max action delta {max_delta:.3e} "
            f">= tol {tol:.1e} over {n} seeded obs (seed={seed})"
        )
    return max_delta


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Check torch-vs-ONNX parity for an exported policy.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("onnx_path", help="exported .onnx policy")
    parser.add_argument(
        "torch_path",
        help="TorchScript module (torch.jit.script(model).save(...)); a whole "
        "torch.save(model, ...) module also works if its class is importable",
    )
    parser.add_argument("--n", type=int, default=32, help="number of seeded obs")
    parser.add_argument("--seed", type=int, default=0, help="rng seed for the obs")
    parser.add_argument("--tol", type=float, default=1e-4, help="max |delta| allowed")
    args = parser.parse_args()

    # TorchScript first (self-contained, cross-process); fall back to a
    # whole-module pickle, which needs the defining class importable here.
    try:
        model = torch.jit.load(args.torch_path, map_location="cpu")
    except RuntimeError:
        model = torch.load(args.torch_path, map_location="cpu", weights_only=False)
    session = ort.InferenceSession(
        args.onnx_path, providers=["CPUExecutionProvider"]
    )
    obs_dim = int(session.get_inputs()[0].shape[1])

    max_delta = assert_parity(
        model, args.onnx_path, obs_dim, n=args.n, seed=args.seed, tol=args.tol
    )
    print(
        f"parity OK: max action delta {max_delta:.3e} < tol {args.tol:.1e} "
        f"over {args.n} seeded obs (seed={args.seed})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(_main())
