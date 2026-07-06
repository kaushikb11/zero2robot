"""onnxruntime-only policy wrapper for the scoring harness.

Loads a validated ONNX policy through onnxruntime's CPU provider and exposes a
single `act(obs) -> action` call. No torch, no pickle, no network: the only
executor is onnxruntime.InferenceSession on the local file (policy.yaml:
"load_via: onnxruntime only"). Deterministic: onnxruntime CPU inference is a
pure function of the weights and the input.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


class OnnxPolicy:
    """A contract-v1 ONNX policy: obs[obs_dim] -> action[act_dim], CPU only."""

    def __init__(self, onnx_path: str | Path) -> None:
        # Pin to a single intra-op thread: deterministic and matches the
        # sandbox's small CPU budget (policy.yaml cpu_limit). Multi-threaded
        # onnxruntime reductions can reorder and are not bitwise-stable.
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(onnx_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.obs_dim = int(self.session.get_inputs()[0].shape[1])
        self.act_dim = int(self.session.get_outputs()[0].shape[1])

    def act(self, obs: np.ndarray) -> np.ndarray:
        """Map a single obs (shape [obs_dim]) to an action (shape [act_dim])."""
        batch = np.asarray(obs, dtype=np.float32).reshape(1, self.obs_dim)
        action = self.session.run(None, {self.input_name: batch})[0]
        return np.asarray(action, dtype=np.float32).reshape(self.act_dim)
