"""zero2robot 5.7 — Quantize a Policy by Hand: INT8 Is a Scale, Not a Rounding.

You trained a behavior-cloning policy in ch1.1 whose weights live in [-0.5, 0.5].
"Deploy it in int8" sounds like a rounding: cast every float to the nearest whole
number. Do that and `round(0.31) = 0` — the whole policy collapses to zeros and the
robot goes limp. INT8 quantization is not a rounding; it is a SCALE: pick a real number
s, store the small integer `round(w / s)`, and recover `w ~= s * round(w / s)`. The int8
grid is the SAME 256 levels for everyone; the scale maps those levels onto YOUR tensor's
range. This file builds it from scratch on a small PushT MLP — no torch.quantization, no
onnxruntime PTQ, just numpy integers you can read:

  (1) SYMMETRIC INT8 WEIGHTS. A per-tensor scale from a weight's range, quantize -> int8
      -> dequantize, and measure the round-trip error. Then do it PER OUTPUT CHANNEL — one
      scale per row — and watch most of that error vanish, because one fat-tailed channel
      no longer sets the step size for all the others.
  (2) STATIC ACTIVATION CALIBRATION. Weights you quantize offline; activations you cannot —
      you don't know their range until data flows. So run a CALIBRATION set through the fp32
      net, collect each layer's activation range (min-max, or a percentile clip that ignores
      one freak outlier), derive a scale, and run the WHOLE pass in integers (int8 @ int8).
  (3) THE DEPLOYMENT TRIANGLE. Size (KB) vs action-error-vs-fp32 (MSE) vs CPU latency (ms),
      across FP32 -> per-tensor INT8 -> per-channel INT8, each rolled out with ch1.6 rigor
      (a success delta only counts once its Wilson interval clears fp32's).

THE HEADLINE (deterministic on CPU — int8 arithmetic is bitwise): per-tensor INT8
SPIKES the action error; switching to per-channel RECOVERS most of it, at ~4x smaller
size. The granularity of the scale is the whole idea.

Two honesties this chapter refuses to fudge:
  * The SIZE win is guaranteed (int8 weights are 1 byte, not 4). The LATENCY win is NOT:
    on a laptop CPU with no fused int8 kernel, dequantize overhead can make naive int8
    SLOWER than fp32. We measure it and report whichever wins — which is why the deploy
    reading-track reaches for a real int8 runtime like TensorRT.
  * `--break bad_calib` calibrates the activation scales on a NARROW slice of states the
    policy never deploys in: the scales come out too small, real activations saturate at
    +-127, and the full-integer error explodes — proof a calibration set must match the
    deployment distribution.

Run it:      python curriculum/phase5_practitioner/ch5.7_quantize/quantize.py --seed 0
Break it:    python curriculum/phase5_practitioner/ch5.7_quantize/quantize.py --seed 0 --break bad_calib
CI smoke:    python curriculum/phase5_practitioner/ch5.7_quantize/quantize.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as the other chapters).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv, ScriptedExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

OBS_DIM, ACT_DIM = PushTEnv.OBS_DIM, PushTEnv.ACT_DIM
QMAX = 127                    # symmetric signed int8 uses [-127, 127]; we skip -128 so |q| is symmetric
EVAL_BASE_SEED = 10_000       # held-out eval seeds; demos use [seed, seed+demos), calib uses [seed+1000, ...)
PERCENTILE = 99.9             # the activation clip: ignore the top 0.1% so one freak state can't set the scale

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, default=Path("outputs/ch5.7-quantize"))
parser.add_argument("--seed", type=int, default=0, help="seeds demos, training, calibration, and the rollout")
parser.add_argument("--demos", type=int, default=80, help="scripted-expert episodes to train the BC policy on")  # smoke: 6
parser.add_argument("--calib_episodes", type=int, default=8, help="held-out episodes whose activations calibrate the scales")  # smoke: 2
parser.add_argument("--hidden_dim", type=int, default=128, help="MLP width — small on purpose so quant error is VISIBLE")
parser.add_argument("--epochs", type=int, default=300)   # cpu-laptop: ~0.5 min | smoke: 3
parser.add_argument("--eval_episodes", type=int, default=24, help="held-out rollouts PER config for the success CI (ch1.6)")  # smoke: 4
parser.add_argument("--calib", choices=("percentile", "minmax"), default="percentile",
                    help="activation-range rule: percentile clips the tail (robust); minmax lets one outlier set the scale")
parser.add_argument("--policy", type=Path, default=None,
                    help="load a trained BC policy (e.g. outputs/ch1.1-bc/bc_policy.pt) instead of training inline")
parser.add_argument("--break", dest="break_mode", choices=("bad_calib",), default=None,
                    help="bad_calib: calibrate on a NARROW near-goal slice -> scales too small -> activations saturate -> full-int8 error explodes")
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # everything here runs on CPU numpy/torch: bitwise-reproducible, no GPU nondeterminism to caveat
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.demos, args.calib_episodes, args.hidden_dim = 6, 2, 32
    args.epochs, args.eval_episodes = 3, 4
banner("ch5.7-quantize", device="cpu")  # int8 quantization is a CPU story; there is no --device knob
args.out.mkdir(parents=True, exist_ok=True)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch5.7-quantize", spawn=False)
    rr.save(str(args.out / "quantize.rrd"))
# --- endregion ---

# --- region: data ---
# Collect scripted-expert demos exactly as ch1.1/ch1.6 do: per-frame (obs, action), plus the
# block-to-target distance we tag each frame with — the narrow --break slice reads it. Episode
# i uses reset seed (base + i), so calib/eval seed ranges never overlap the training demos.
def collect(num_episodes: int, base_seed: int):
    env = PushTEnv()
    obs_all, act_all, dist_all = [], [], []
    for i in range(num_episodes):
        obs = env.reset(base_seed + i)
        expert = ScriptedExpert(noise=0.0, seed=base_seed + i)
        done = False
        while not done:
            action = expert.action(env)
            obs_all.append(obs)
            act_all.append(action)
            dist_all.append(float(math.hypot(obs[2], obs[3])))  # obs[2:4] = block (x, y); target sits at the origin
            obs, _, done, _ = env.step(action)
    return (np.asarray(obs_all, np.float32), np.asarray(act_all, np.float32), np.asarray(dist_all, np.float32))


train_obs, train_act, _ = collect(args.demos, args.seed)
calib_obs, _, calib_dist = collect(args.calib_episodes, args.seed + 1000)   # held-out states to calibrate on
eval_obs, _, _ = collect(args.calib_episodes, EVAL_BASE_SEED)               # held-out states to score action error on
print(f"data: {len(train_obs)} train frames / {len(calib_obs)} calib / {len(eval_obs)} eval "
      f"({args.demos}/{args.calib_episodes}/{args.calib_episodes} episodes)")
# --- endregion ---

# --- region: train ---
# The policy under the knife: the ch1.1 behavior-cloning MLP, small (hidden 128) so the
# quantization error is large enough to SEE. Normalization lives inside the module as
# buffers (ch1.1's contract), so `policy(raw_obs)` just works. We only quantize the three
# Linear layers of `net`; the normalization stays fp32 (it is cheap and not the lesson).
class BCPolicy(nn.Module):
    def __init__(self, hidden_dim: int, stats: dict[str, np.ndarray]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, ACT_DIM),
        )
        for name, value in stats.items():
            self.register_buffer(name, torch.from_numpy(value))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        normalized = (2.0 * (obs - self.obs_min) / self.obs_range - 1.0).clamp(-1.0, 1.0)
        return (self.net(normalized) + 1.0) / 2.0 * self.act_range + self.act_min


def norm_stats(obs: np.ndarray, act: np.ndarray) -> dict[str, np.ndarray]:
    obs_min, act_min = obs.min(0), act.min(0)
    # constant dims (the fixed target) carry range 0; give them range 1 so they map to a
    # constant instead of dividing by zero (ch1.1's guard).
    obs_range = np.where(obs.max(0) - obs_min < 1e-4, np.float32(1.0), obs.max(0) - obs_min)
    act_range = np.where(act.max(0) - act_min < 1e-4, np.float32(1.0), act.max(0) - act_min)
    return {"obs_min": obs_min, "obs_range": obs_range, "act_min": act_min, "act_range": act_range}


def train_bc(obs: np.ndarray, act: np.ndarray, hidden_dim: int, epochs: int, seed: int) -> BCPolicy:
    torch.manual_seed(seed)
    policy = BCPolicy(hidden_dim, norm_stats(obs, act))
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    obs_t, act_t = torch.from_numpy(obs), torch.from_numpy(act)
    shuffle = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for batch in torch.randperm(len(obs_t), generator=shuffle).split(256):
            loss = nn.functional.mse_loss(policy(obs_t[batch]), act_t[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
    return policy.eval()


if args.policy is not None:
    policy = torch.load(args.policy, map_location="cpu", weights_only=False).eval()
    print(f"loaded policy from {args.policy}")
else:
    policy = train_bc(train_obs, train_act, args.hidden_dim, args.epochs, args.seed)
    print(f"trained BC policy (hidden {args.hidden_dim}, {args.epochs} epochs) on {args.demos} demos")

# Pull everything into numpy so BOTH the fp32 and the int8 forward pass run in the SAME
# framework — a fair latency race whose only difference is dtype + dequant overhead.
sd = policy.state_dict()
LAYERS = [(sd["net.0.weight"].numpy(), sd["net.0.bias"].numpy()),
          (sd["net.2.weight"].numpy(), sd["net.2.bias"].numpy()),
          (sd["net.4.weight"].numpy(), sd["net.4.bias"].numpy())]  # (W[out,in], b[out]) per Linear
NORM = {k: sd[k].numpy() for k in ("obs_min", "obs_range", "act_min", "act_range")}


def normalize(obs: np.ndarray) -> np.ndarray:  # (N, 10) raw -> (N, 10) in [-1, 1], the net's real input
    return np.clip(2.0 * (obs - NORM["obs_min"]) / NORM["obs_range"] - 1.0, -1.0, 1.0)


def denormalize(y: np.ndarray) -> np.ndarray:  # (N, 2) in net space -> raw action
    return (y + 1.0) / 2.0 * NORM["act_range"] + NORM["act_min"]


def forward_fp32(obs: np.ndarray) -> np.ndarray:
    x = normalize(obs)
    for i, (W, b) in enumerate(LAYERS):
        x = x @ W.T + b
        if i < len(LAYERS) - 1:
            x = np.maximum(x, 0.0)  # ReLU
    return denormalize(x)
# --- endregion ---

# --- region: quantize ---
# The core. A symmetric int8 scale maps a real range [-r, r] onto the integer grid
# [-127, 127]: s = r / 127, and q = round(w / s) clamped to the grid. Dequantize is one
# multiply, w_hat = s * q. The ONLY choice is what "r" ranges over:
#   per-tensor  : r = max|W| over the WHOLE matrix -> one scale. One fat channel sets the
#                 step size for every channel, so small-magnitude rows lose all resolution.
#   per-channel : r = max|W[o]| per OUTPUT ROW -> one scale each. Every row gets a step
#                 size matched to its own range. This is the recovery.
def quantize_weight(W: np.ndarray, per_channel: bool):
    r = np.abs(W).max(axis=1, keepdims=True) if per_channel else np.abs(W).max()  # (out,1) or scalar
    scale = np.maximum(np.asarray(r, np.float32) / QMAX, 1e-8)                     # never divide by zero
    q = np.clip(np.round(W / scale), -QMAX, QMAX).astype(np.int8)
    return q, scale  # scale broadcasts back over `in`: w_hat = q * scale


def weight_roundtrip_err(per_channel: bool) -> float:
    """Mean |W - dequant(quant(W))| across the three layers. Per-channel is guaranteed
    <= per-tensor: a per-row scale is a strict refinement of a single one."""
    errs = []
    for W, _ in LAYERS:
        q, scale = quantize_weight(W, per_channel)
        errs.append(np.abs(W - q.astype(np.float32) * scale).mean())
    return float(np.mean(errs))


# THE MISCONCEPTION killed in one line: rounding weights to int8 with NO scale collapses
# them — everything in (-0.5, 0.5) rounds to 0. The scale is what saves the signal.
NAIVE_ROUND_ZERO_FRAC = float(np.mean([(np.round(W) == 0).mean() for W, _ in LAYERS]))


# Activations we cannot quantize offline — we don't know their range until data flows. So run
# the calibration states through the fp32 net, record each layer's INPUT range, and set the
# scale r/127 with r = max|a| (minmax) or the 99.9th percentile of |a| (percentile ignores a
# single freak activation that would otherwise blow the scale up).
def calibrate(calib_input: np.ndarray, rule: str) -> list[float]:
    scales, x = [], normalize(calib_input)
    for i, (W, b) in enumerate(LAYERS):
        r = np.abs(x).max() if rule == "minmax" else np.percentile(np.abs(x), PERCENTILE)
        scales.append(float(max(r / QMAX, 1e-8)))  # this layer's activation scale, from its INPUT
        x = x @ W.T + b
        if i < len(LAYERS) - 1:
            x = np.maximum(x, 0.0)
    return scales


# The --break: calibrate on a NARROW slice — only the frames where the block already sits
# near the target, where the policy barely moves and activations are small. The scales come
# out too tight; at real deployment the activations run off the end of the grid and clamp.
if args.break_mode == "bad_calib":
    near_goal = calib_dist <= np.quantile(calib_dist, 0.25)  # the 25% closest-to-goal states
    calib_input, calib_rule = calib_obs[near_goal], "minmax"  # narrow AND minmax: no percentile safety net either
    print(f"[break bad_calib] calibrating on {int(near_goal.sum())}/{len(calib_obs)} near-goal frames "
          f"(block dist <= {np.quantile(calib_dist, 0.25):.3f}) — a distribution the policy never deploys in")
else:
    calib_input, calib_rule = calib_obs, args.calib
ACT_SCALES = calibrate(calib_input, calib_rule)


def forward_int8(obs: np.ndarray, per_channel: bool, quant_act: bool) -> np.ndarray:
    """One int8 forward, two modes. quant_act=False is WEIGHT-ONLY (the dominant real-world
    PTQ mode: int8 weights, dequantized on the fly, fp32 matmul) — the deployment triangle.
    quant_act=True is FULL INTEGER (activations quantized with the STATIC calibrated scales,
    int8 @ int8 -> int32 accumulate) — the path the --break attacks. Bias stays fp32."""
    x = normalize(obs)
    for i, (W, b) in enumerate(LAYERS):
        Wq, s_w = quantize_weight(W, per_channel)
        if quant_act:
            s_x = ACT_SCALES[i]
            x_int = np.clip(np.round(x / s_x), -QMAX, QMAX).astype(np.int32)   # quantize activations (may CLAMP)
            acc = x_int @ Wq.astype(np.int32).T                                # exact integer matmul
            x = acc.astype(np.float32) * (s_x * s_w.reshape(-1)) + b           # dequant: s_x * s_w per output channel
        else:
            x = x @ (Wq.astype(np.float32) * s_w).T + b                        # dequant weights, fp32 matmul
        if i < len(LAYERS) - 1:
            x = np.maximum(x, 0.0)
    return denormalize(x)
# --- endregion ---

# --- region: deploy ---
# The deployment triangle: for each config measure (a) storage size, (b) action error vs
# the fp32 policy, (c) per-call CPU latency, (d) task success with a Wilson interval so a
# "drop" only counts once it clears fp32's band (ch1.6).
Z95 = 1.959963985  # 0.975 standard-normal quantile; the 95% two-sided Wilson interval, no scipy


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials (ch1.6's from-scratch CI)."""
    if n == 0:
        return (0.0, 1.0)
    p, z2 = k / n, Z95 * Z95
    denom = 1.0 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = (Z95 / denom) * math.sqrt(p * (1.0 - p) / n + z2 / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def model_size_bytes(int8: bool, per_channel: bool) -> int:
    """Storage the model actually costs. fp32: 4 bytes/param. int8: 1 byte/weight + the
    fp32 scales (1 per tensor, or 1 per output channel) + fp32 bias (unquantized here)."""
    total = 0
    for W, b in LAYERS:
        total += (W.size * 4 if not int8 else W.size + (W.shape[0] if per_channel else 1) * 4) + b.size * 4
    return total


def action_mse_vs_fp32(fn) -> float:
    return float(np.mean((fn(eval_obs) - forward_fp32(eval_obs)) ** 2))


def latency_ms(fn) -> float:
    """Median single-observation latency (batch=1, the real deployment shape). Wall clock is
    never bit-reproducible, so --smoke skips it (0.0) to keep metrics.json byte-stable; the
    real number comes from a full run + wallclock-bench."""
    if args.smoke:
        return 0.0
    one = eval_obs[:1]
    for _ in range(20):
        fn(one)  # warm up so the first-call cost doesn't skew the median
    samples = []
    for _ in range(200):
        t0 = time.perf_counter()
        fn(one)
        samples.append((time.perf_counter() - t0) * 1e3)
    return float(np.median(samples))


def rollout_success(fn, n_episodes: int) -> tuple[int, int]:
    """Roll `fn` out on held-out start seeds; return (successes, n). Bit-reproducible."""
    env, k = PushTEnv(), 0
    for e in range(n_episodes):
        obs = env.reset(EVAL_BASE_SEED + 500 + e)  # a held-out band, disjoint from training + eval-obs seeds
        done, info = False, {}
        while not done:
            obs, _, done, info = env.step(fn(obs[None])[0])
        k += bool(info["success"])
    return k, n_episodes


def measure(fn, int8: bool, per_channel: bool) -> dict:
    k, n = rollout_success(fn, args.eval_episodes)
    lo, hi = wilson_ci(k, n)
    return {"size_kb": round(model_size_bytes(int8, per_channel) / 1024, 3),
            "action_mse": round(action_mse_vs_fp32(fn), 8), "latency_ms": round(latency_ms(fn), 4),
            "success": k, "n": n, "success_rate": round(k / n, 4),
            "ci_lo": round(lo, 4), "ci_hi": round(hi, 4)}


# The triangle is WEIGHT-ONLY int8 (quant_act=False): the clean, seed-robust headline.
triangle = {
    "fp32": measure(lambda o: forward_fp32(o), False, False),
    "per_tensor_int8": measure(lambda o: forward_int8(o, per_channel=False, quant_act=False), True, False),
    "per_channel_int8": measure(lambda o: forward_int8(o, per_channel=True, quant_act=False), True, True),
}
# The all-integer path (part 2): per-channel weights + STATIC calibrated activations. Under a
# good calibration it pays a modest extra floor; --break bad_calib makes its error explode.
full_int8 = measure(lambda o: forward_int8(o, per_channel=True, quant_act=True), True, True)
for name, cfg in list(triangle.items()) + [("full_int8(+act)", full_int8)]:
    print(f"  {name:18s}  {cfg['size_kb']:6.2f} KB  mse {cfg['action_mse']:.2e}  {cfg['latency_ms']:6.3f} ms  "
          f"success {cfg['success']}/{cfg['n']} [{cfg['ci_lo']:.2f}, {cfg['ci_hi']:.2f}]")

wt_err_per_tensor = weight_roundtrip_err(per_channel=False)
wt_err_per_channel = weight_roundtrip_err(per_channel=True)
# --- endregion ---

# --- region: report ---
# The gated, seed-robust headline is the DIRECTION (int8 arithmetic is bitwise on CPU): per-
# tensor int8 spikes the action error, per-channel recovers most of it, at ~4x smaller size.
# The success CIs are the honesty layer — does that error reach the task, judged vs fp32's band.
pt, pc, fp = triangle["per_tensor_int8"], triangle["per_channel_int8"], triangle["fp32"]
mse_recovery = pt["action_mse"] / max(pc["action_mse"], 1e-12)  # > 1 means per-channel recovered weight-quant error
size_ratio = fp["size_kb"] / pc["size_kb"]
faster = pc["latency_ms"] < fp["latency_ms"]

metrics = {
    "break": args.break_mode or "none",
    "calib_rule": calib_rule,
    "fp32_action_mse": fp["action_mse"],
    "fp32_latency_ms": fp["latency_ms"],
    "fp32_size_kb": fp["size_kb"],
    "fp32_success_ci_hi": fp["ci_hi"],
    "fp32_success_ci_lo": fp["ci_lo"],
    "fp32_success_rate": fp["success_rate"],
    "full_int8_action_mse": full_int8["action_mse"],  # per-channel weights + calibrated activations; the --break target
    "full_int8_success_rate": full_int8["success_rate"],
    "int8_faster_than_fp32": bool(faster),            # HONEST: usually False on a laptop CPU (no fused int8 kernel)
    "mse_recovery_ratio": round(mse_recovery, 3),     # per-tensor mse / per-channel mse; >> 1 is the headline
    "naive_round_zero_frac": round(NAIVE_ROUND_ZERO_FRAC, 4),  # rounding with NO scale collapses this fraction to 0
    "per_channel_action_mse": pc["action_mse"],
    "per_channel_latency_ms": pc["latency_ms"],
    "per_channel_size_kb": pc["size_kb"],
    "per_channel_success_ci_hi": pc["ci_hi"],
    "per_channel_success_ci_lo": pc["ci_lo"],
    "per_channel_success_rate": pc["success_rate"],
    "per_tensor_action_mse": pt["action_mse"],
    "per_tensor_latency_ms": pt["latency_ms"],
    "per_tensor_size_kb": pt["size_kb"],
    "per_tensor_success_ci_hi": pt["ci_hi"],
    "per_tensor_success_ci_lo": pt["ci_lo"],
    "per_tensor_success_rate": pt["success_rate"],
    "seed": args.seed,
    "size_ratio_fp32_over_int8": round(size_ratio, 3),  # ~4x, the guaranteed win
    "smoke": bool(args.smoke),
    "weight_roundtrip_err_per_channel": round(wt_err_per_channel, 8),
    "weight_roundtrip_err_per_tensor": round(wt_err_per_tensor, 8),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

# The quantization-dial toy reads this: three configs driving three live bars (size drops,
# action-mse spikes then recovers, latency), plus the two teaching one-liners.
vizdata = {
    "seed": args.seed, "break": args.break_mode or "none", "calib_rule": calib_rule,
    "naive_round_zero_frac": round(NAIVE_ROUND_ZERO_FRAC, 4),
    "weight_roundtrip_err": {"per_tensor": round(wt_err_per_tensor, 8), "per_channel": round(wt_err_per_channel, 8)},
    "full_int8": {"action_mse": full_int8["action_mse"], "success_rate": full_int8["success_rate"]},
    "configs": [{"label": "FP32", **triangle["fp32"]},
                {"label": "per-tensor INT8", **triangle["per_tensor_int8"]},
                {"label": "per-channel INT8", **triangle["per_channel_int8"]}],
}
(args.out / "demo").mkdir(parents=True, exist_ok=True)
(args.out / "demo" / "vizdata.json").write_text(json.dumps(vizdata) + "\n")

print(f"\nweight round-trip error: per-tensor {wt_err_per_tensor:.2e} -> per-channel {wt_err_per_channel:.2e} "
      f"({wt_err_per_tensor / max(wt_err_per_channel, 1e-12):.1f}x smaller)")
print(f"naive 'round to int8' (no scale) collapses {NAIVE_ROUND_ZERO_FRAC:.0%} of weights to zero — the scale is the idea")
print(f"HEADLINE: per-tensor mse {pt['action_mse']:.2e} -> per-channel {pc['action_mse']:.2e} "
      f"= {mse_recovery:.1f}x recovery, at {size_ratio:.1f}x smaller than fp32")
print(f"LATENCY (honest): fp32 {fp['latency_ms']:.3f} ms vs per-channel int8 {pc['latency_ms']:.3f} ms — "
      f"int8 is {'FASTER' if faster else 'NOT faster (dequant overhead, no fused kernel)'} on this CPU")
print(f"full-integer path (calibrated activations): action mse {full_int8['action_mse']:.2e}"
      f"{'  <-- EXPLODED by bad calibration' if args.break_mode == 'bad_calib' else ''}")

if args.rerun:
    for i, (name, cfg) in enumerate(list(triangle.items()) + [("full_int8", full_int8)]):
        rr.set_time("config", sequence=i)
        rr.log("triangle/size_kb", rr.Scalars([cfg["size_kb"]]))
        rr.log("triangle/action_mse", rr.Scalars([cfg["action_mse"]]))
        rr.log("triangle/latency_ms", rr.Scalars([cfg["latency_ms"]]))
        rr.log("triangle/success_rate", rr.Scalars([cfg["success_rate"]]))
    rr.log("weight/roundtrip_per_tensor", rr.Scalars([wt_err_per_tensor]), static=True)
    rr.log("weight/roundtrip_per_channel", rr.Scalars([wt_err_per_channel]), static=True)
    print(f"recording: {args.out / 'quantize.rrd'} — open it with: rerun {args.out / 'quantize.rrd'}")
print(f"metrics: {args.out / 'metrics.json'}  |  vizdata: {args.out / 'demo' / 'vizdata.json'}")
# --- endregion ---
