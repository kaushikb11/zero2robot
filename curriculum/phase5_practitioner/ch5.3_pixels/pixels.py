"""zero2robot 5.3 — Control From Pixels: Visuomotor Behavior Cloning.

Chapter 1.1 cloned PushT from the 10-number STATE vector and it just worked. Chapter 1.8 bolted a
vision channel onto a VLA and `--break blind` proved the policy never used it — because PushT is
solvable from state, and the random-init encoder had nothing to add. This chapter removes the
escape hatch: the policy gets ONLY pixels, a live 64x64 frame, no state. Now the encoder is the
whole ballgame.

We build a tiny ViT twice, changing one thing — whether it was pre-aligned. ALIGNED: a ViT
contrastively pre-aligned to the scene geometry (the ch5.2 recipe, re-contained: BACKGROUND-
SUBTRACT the ~98%-constant frame — ch5.2's load-bearing trick — then symmetric InfoNCE
image<->state), rebuilt DETERMINISTICALLY from the seed (no checkpoint in git). RANDOM: the
identical ViT, never aligned — a fixed random projection of pixels.

THE MEASURED HEADLINE is the CONTROL-USEFULNESS PROBE (the honest, reproducible bar): freeze
each encoder, fit a small action-regression probe on its features, read the HELD-OUT val MSE.
Aligned features let the probe recover the expert action with LOWER error than random (aligned
< random val_mse — a seed-robust DIRECTION). That is the payoff ch1.8's `--break blind` could
only assert: an aligned backbone puts the geometry a controller needs INTO the features.

THE HIGHER BAR — closed-loop rollout from pixels alone (ch1.6 Wilson intervals) — we also run and
report HONESTLY. At this toy scale (64x64, a from-scratch ViT, single-frame BC of a non-Markovian
expert) full pixel control sits near the floor for BOTH encoders; that ceiling is why real VLAs
(OpenVLA) use a pretrained SigLIP backbone and why closing it is the Scale Lab. Export to ONNX (v1).

Run it:      python curriculum/phase5_practitioner/ch5.3_pixels/pixels.py --seed 0
Trap it:     python curriculum/phase5_practitioner/ch5.3_pixels/pixels.py --seed 0 --train_encoder
CI smoke:    python curriculum/phase5_practitioner/ch5.3_pixels/pixels.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as the other chapters).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.assert_parity import assert_parity  # noqa: E402
from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht.pusht_env import PushTEnv  # noqa: E402
from curriculum.common.envs.pusht.scripted_expert import ScriptedExpert  # noqa: E402
from curriculum.common.export_onnx import export_policy  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

IMG_HW = 64          # the free-tier pixel floor (ch1.7 rendered 96; the loop wants 64)
PATCH = 8            # 8x8 patches -> 8x8 = 64 image tokens, plus a CLS
OBS_DIM = IMG_HW * IMG_HW * 3   # 12288: the flat image IS the observation (contract v1)
ACT_DIM = PushTEnv.ACT_DIM      # 2: pusher velocity
TEMP = 0.07          # InfoNCE temperature for the contrastive alignment (CLIP's init)
Z95 = 1.959963985    # 0.975 standard-normal quantile — the 95% Wilson interval (ch1.6)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch5.3-pixels"))
parser.add_argument("--episodes", type=int, default=90)     # demos to render. cpu: 90 | smoke: 4
parser.add_argument("--frame_stride", type=int, default=2, help="keep every Nth frame (near-duplicates)")
parser.add_argument("--dim", type=int, default=96)          # ViT width. T4: 96 | 4090: 384 | smoke: 32
parser.add_argument("--depth", type=int, default=3)         # attention blocks. T4: 3 | smoke: 1
parser.add_argument("--heads", type=int, default=3)         # attention heads (dim must divide by this)
parser.add_argument("--hidden", type=int, default=256)      # policy-head / probe MLP width
parser.add_argument("--align_epochs", type=int, default=40)  # contrastive pretrain. cpu: 40 | smoke: 2
parser.add_argument("--probe_epochs", type=int, default=150)  # control-usefulness probe fit. cpu: 150 | smoke: 3
parser.add_argument("--bc_epochs", type=int, default=300)   # head BC epochs. cpu: 300 | smoke: 3
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--eval_episodes", type=int, default=20)  # per encoder. T4: 30 | smoke: 3
parser.add_argument("--train_encoder", action="store_true",
                    help="THE TRAP: unfreeze the aligned encoder during BC. At free-tier scale it "
                         "overfits the tiny demo set and eval collapses (exercise ex2)")
parser.add_argument("--seed", type=int, default=0, help="seeds demos, the ViT init, alignment, the probe, and the BC head")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true",
                    help="tiny hermetic CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # seeds python/numpy/torch; the ViT init draws from torch's RNG below
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.episodes, args.dim, args.depth, args.heads = 4, 32, 1, 2
    args.align_epochs, args.probe_epochs, args.bc_epochs, args.eval_episodes, args.device = 2, 3, 3, 3, "cpu"
banner("ch5.3-pixels", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch5.3-pixels", spawn=False)
    rr.save(str(args.out / "pixels.rrd"))
# --- endregion ---

# --- region: data ---
# Replay the scripted PushT expert in-process (ch1.7 pattern) and cache a 64x64 top-down FRAME
# per kept step alongside state and action. The state is collected ONLY to align the encoder +
# score the probe; it is never fed to the policy. Deterministic: episode e uses seed (seed+e).
def collect(episodes: int, seed: int, stride: int):
    env = PushTEnv()
    frames, states, actions = [], [], []
    for e in range(episodes):
        obs = env.reset(seed + e)
        expert = ScriptedExpert(noise=0.0, seed=seed + e)
        step, done = 0, False
        while not done:
            action = expert.action(env)
            if step % stride == 0:  # consecutive frames are near-duplicates
                frames.append(env.render_frame(IMG_HW, IMG_HW))
                states.append(obs.astype(np.float32))
                actions.append(action[:ACT_DIM].astype(np.float32))
            obs, _, done, _ = env.step(action)
            step += 1
    return (np.asarray(frames, np.uint8), np.asarray(states, np.float32),
            np.asarray(actions, np.float32))


frames_np, states_np, actions_np = collect(args.episodes, args.seed, args.frame_stride)
num_frames = len(frames_np)
frames_t = torch.from_numpy(frames_np).to(device).float()   # (N, 64, 64, 3) in [0, 255]
states_t = torch.from_numpy(states_np).to(device)           # (N, 10) — alignment + probe target only
actions_t = torch.from_numpy(actions_np).to(device)         # (N, 2)
# BACKGROUND SUBTRACTION (ch5.2's load-bearing finding): a raw 64x64 PushT frame is ~98% constant
# table, which collapses a tiny from-scratch ViT until you subtract the mean frame (ch5.2 measured
# a linear probe go chance -> ~1.0). We center every frame on this background (rides in a buffer).
frame_mean = torch.from_numpy((frames_np.astype(np.float32) / 127.5 - 1.0).mean(0)).to(device)  # (64,64,3)
# Action denorm stats (ch1.1): map the head's tanh output back to raw velocity.
act_min = actions_np.min(0)
act_range = np.where(actions_np.max(0) - act_min < 1e-4, np.float32(1.0), actions_np.max(0) - act_min)
# Held-out split for the control-usefulness PROBE: 75% fit / 25% val, seeded (numpy Generator from
# set_seed). No val frame is ever fit on, so a low val MSE means the features GENERALIZE.
probe_perm = rng.permutation(num_frames)
n_val = max(8, int(0.25 * num_frames))
probe_tr = torch.from_numpy(probe_perm[:-n_val]).to(device)
probe_va = torch.from_numpy(probe_perm[-n_val:]).to(device)
print(f"dataset: {args.episodes} episodes / {num_frames} frames @ {IMG_HW}x{IMG_HW} "
      f"({len(probe_tr)} probe-fit / {len(probe_va)} probe-val); state used ONLY to align + probe")
# --- endregion ---

# --- region: model ---
# A tiny ViT — the SAME block shape ch1.8's VLA uses, re-derived here (chapters re-contain their
# backbone; the repetition is the lesson). Stride-8 conv -> 64 tokens, a learned CLS + positions,
# a few pre-norm self-attention blocks, read the CLS.
class Block(nn.Module):
    """One pre-norm transformer block: multi-head self-attention + an MLP, nn.Linear only."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.last_attn = None  # CLS attention over patches, for the saliency viz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, dim = x.shape
        h, hd = self.heads, dim // self.heads
        qkv = self.qkv(self.ln1(x)).reshape(B, L, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # each (B, h, L, hd)
        attn = ((q @ k.transpose(-2, -1)) / math.sqrt(hd)).softmax(dim=-1)
        self.last_attn = attn[:, :, 0, 1:].mean(1).detach()  # (B, 64): CLS over the patch grid
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(B, L, dim))
        return x + self.mlp(self.ln2(x))


class ViTEncoder(nn.Module):
    """(B,64,64,3) float pixels -> (B, dim) CLS feature (random until aligned). Subtracts the
    background (frame_mean) FIRST so the ViT sees the block, not the table."""

    def __init__(self, dim: int, depth: int, heads: int, frame_mean: torch.Tensor) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=PATCH, stride=PATCH)   # -> (B, dim, 8, 8)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(0.02 * torch.randn(1, 1 + (IMG_HW // PATCH) ** 2, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.register_buffer("frame_mean", frame_mean)       # background, subtracted below

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = images / 127.5 - 1.0                             # (B, 64, 64, 3) in [-1, 1]
        x = (x - self.frame_mean).permute(0, 3, 1, 2)        # center on background, then (B, 3, 64, 64)
        x = self.patch(x).flatten(2).transpose(1, 2)         # (B, 64, dim) patch tokens
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x[:, 0])                            # the CLS row: the image feature


class PixelPolicy(nn.Module):
    """The deployable policy: a FROZEN ViT encoder + a thin trained adapter + head. forward takes
    the flat image as the contract-v1 observation, so the whole pixels->action path exports as one
    ONNX graph (the encoder is INSIDE). Normalization rides in buffers."""

    def __init__(self, encoder: ViTEncoder, dim: int, hidden: int,
                 feat_mean, feat_std, act_min, act_range) -> None:
        super().__init__()
        self.encoder = encoder
        self.adapter = nn.Linear(dim, hidden)   # the thin adapter onto the frozen features
        self.head = nn.Linear(hidden, ACT_DIM)
        for name, stat in [("feat_mean", feat_mean), ("feat_std", feat_std),
                           ("act_min", act_min), ("act_range", act_range)]:
            self.register_buffer(name, torch.as_tensor(stat, dtype=torch.float32))

    def head_forward(self, feat_std: torch.Tensor) -> torch.Tensor:
        # standardized feature -> (B, hidden) -> tanh action in [-1, 1] -> raw velocity
        a = torch.tanh(self.head(torch.relu(self.adapter(feat_std))))
        return (a + 1.0) / 2.0 * self.act_range + self.act_min

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # (B, 12288) flat image -> (B, 2) action. The frozen encoder is INSIDE the policy.
        feat = self.encoder(obs.reshape(-1, IMG_HW, IMG_HW, 3))
        return self.head_forward((feat - self.feat_mean) / self.feat_std)
# --- endregion ---

# --- region: align ---
# THE ch5.2 RECIPE, re-contained and COMPACT (5.2 teaches it in full). Symmetric InfoNCE pulls each
# image's CLS feature onto its own scene state and off every other example's. The encoder is told
# only "these pixels and this geometry belong together" — enough to make the features carry WHERE
# the block and pusher are. Deterministic from the seed (CPU), so re-running reproduces it — no git.
def align_encoder(encoder: ViTEncoder, epochs: int) -> float:
    # the contrastive PARTNER tower: projects the 10-D state into the ViT's feature space, pulling
    # image features onto the scene geometry. Thrown away after alignment (only the encoder is kept).
    state_enc = nn.Sequential(nn.Linear(PushTEnv.OBS_DIM, 128), nn.ReLU(), nn.Linear(128, args.dim)).to(device)
    params = list(encoder.parameters()) + list(state_enc.parameters())
    opt = torch.optim.Adam(params, lr=args.lr)
    shuffle = torch.Generator().manual_seed(args.seed + 7)
    loss_val = float("nan")
    for epoch in range(epochs):
        for batch in torch.randperm(num_frames, generator=shuffle).split(args.batch_size):
            if len(batch) < 2:  # InfoNCE needs negatives — a singleton batch has none
                continue
            img = nn.functional.normalize(encoder(frames_t[batch]), dim=1)
            st = nn.functional.normalize(state_enc(states_t[batch]), dim=1)
            logits = img @ st.T / TEMP                        # (b, b) image-state similarities
            labels = torch.arange(len(batch), device=device)  # the diagonal is the match
            loss = 0.5 * (nn.functional.cross_entropy(logits, labels)
                          + nn.functional.cross_entropy(logits.T, labels))
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = loss.item()
            if args.rerun:
                rr.log("align/infonce", rr.Scalars([loss_val]))
    return loss_val
# --- endregion ---

# --- region: probe ---
# THE HEADLINE, measured honestly. Freeze the encoder, featurize every frame, fit a small MLP that
# regresses the EXPERT ACTION from the frozen feature — then read the HELD-OUT val MSE: can a
# controller read the action off these features at all? Both encoders get the IDENTICAL probe init,
# so the only difference is what the encoder learned. Aligned < random val MSE is the DIRECTION.
@torch.no_grad()
def encode_all(encoder: ViTEncoder) -> torch.Tensor:
    encoder.eval()
    return torch.cat([encoder(frames_t[i:i + 256]) for i in range(0, num_frames, 256)])


def control_probe(encoder: ViTEncoder, tag: str) -> float:
    feats = encode_all(encoder)                              # (N, dim) frozen features
    mean, std = feats[probe_tr].mean(0), feats[probe_tr].std(0).clamp_min(1e-4)
    feats_std = (feats - mean) / std                         # standardize on the FIT split only
    torch.manual_seed(args.seed + 5)                         # identical probe init for aligned + random
    probe = nn.Sequential(nn.Linear(args.dim, args.hidden), nn.ReLU(),
                          nn.Linear(args.hidden, ACT_DIM)).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=args.lr)
    shuffle = torch.Generator().manual_seed(args.seed + 6)
    val_mse = float("nan")
    for epoch in range(args.probe_epochs):
        for batch in probe_tr[torch.randperm(len(probe_tr), generator=shuffle)].split(args.batch_size):
            loss = nn.functional.mse_loss(probe(feats_std[batch]), actions_t[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_mse = nn.functional.mse_loss(probe(feats_std[probe_va]), actions_t[probe_va]).item()
        if args.rerun:
            rr.log(f"probe/{tag}/val_mse", rr.Scalars([val_mse]))
    return val_mse
# --- endregion ---

# --- region: train ---
# ch1.1's BC loop, obs swapped for frozen features. Freeze the encoder, featurize every frame ONCE
# (fast: the head trains on vectors, not pixels), standardize, fit the adapter+head with plain MSE.
# `--train_encoder` is THE TRAP: it unfreezes the encoder end-to-end, where the tiny set overfits.
def build_and_train(encoder: ViTEncoder, trainable: bool, tag: str) -> tuple[PixelPolicy, float]:
    feats = encode_all(encoder)                               # (N, dim) frozen features
    feat_mean, feat_std = feats.mean(0), feats.std(0).clamp_min(1e-4)
    policy = PixelPolicy(encoder, args.dim, args.hidden,
                         feat_mean.cpu().numpy(), feat_std.cpu().numpy(), act_min, act_range).to(device)
    if trainable:  # TRAP: encoder joins the optimizer and overfits the pixels
        params, feats_std = list(policy.parameters()), None
        for p in policy.encoder.parameters():
            p.requires_grad_(True)
    else:          # frozen: only the adapter+head learn, on precomputed features
        for p in policy.encoder.parameters():
            p.requires_grad_(False)
        params = list(policy.adapter.parameters()) + list(policy.head.parameters())
        feats_std = (feats - feat_mean) / feat_std
    opt = torch.optim.Adam(params, lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.bc_epochs)
    shuffle = torch.Generator().manual_seed(args.seed + 1)
    loss_val = float("nan")
    for epoch in range(args.bc_epochs):
        policy.train()
        for batch in torch.randperm(num_frames, generator=shuffle).split(args.batch_size):
            if trainable:                       # recompute features through the live encoder
                pred = policy(frames_t[batch].reshape(len(batch), -1))
            else:                               # head-only on the standardized frozen features
                pred = policy.head_forward(feats_std[batch])
            loss = nn.functional.mse_loss(pred, actions_t[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_val = loss.item()
        sched.step()
        if args.rerun:
            rr.log(f"bc/{tag}/loss", rr.Scalars([loss_val]))
    return policy, loss_val
# --- endregion ---

# --- region: eval ---
# Rollouts from PIXELS. Each step renders the live 64x64 frame, flattens it to the contract-v1
# observation, and steps the env with the policy's action — the state MuJoCo tracks is never shown
# to the policy. Report a Wilson 95% interval (ch1.6): pixel-BC success is noisy and a bare % lies.
def wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p, z = k / n, Z95
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


@torch.no_grad()
def rollout(policy: PixelPolicy, ep_seed: int, record: bool = False):
    policy.eval()
    pdev = next(policy.parameters()).device  # export_onnx moves the policy to cpu in place — follow it
    env = PushTEnv()
    env.reset(ep_seed)
    done, info, ret, traj = False, {}, 0.0, []
    while not done:
        frame = env.render_frame(IMG_HW, IMG_HW)
        obs = torch.from_numpy(frame).to(pdev).float().reshape(1, -1)  # (1, 12288)
        action = policy(obs)[0].cpu().numpy()
        if record:
            px, py = env.pusher_pos
            tx, ty, tyaw = env.tee_pose
            traj.append([round(float(px), 4), round(float(py), 4), round(float(tx), 4),
                         round(float(ty), 4), round(float(tyaw), 4)])
        _, reward, done, info = env.step(action)
        ret += reward
    return bool(info["success"]), ret, traj


def evaluate(policy: PixelPolicy, tag: str) -> dict:
    outcomes = [rollout(policy, 10_000 + args.seed + ep) for ep in range(args.eval_episodes)]
    k = sum(s for s, _, _ in outcomes)
    lo, hi = wilson_ci(k, args.eval_episodes)
    mean_return = float(np.mean([r for _, r, _ in outcomes]))
    print(f"eval[{tag:8s}] pixels-only: success {k}/{args.eval_episodes} = {k / args.eval_episodes:.2f}  "
          f"95% CI [{lo:.2f}, {hi:.2f}]  mean_return {mean_return:.2f}")
    if args.rerun:
        rr.log(f"eval/{tag}/success_rate", rr.Scalars([k / args.eval_episodes]))
        rr.log(f"eval/{tag}/ci", rr.Scalars([lo, hi]))
    return {"success_rate": k / args.eval_episodes, "ci_lo": lo, "ci_hi": hi, "mean_return": mean_return}
# --- endregion ---

# --- region: report ---
# One ViT architecture, two histories. FIRST the headline: the control-usefulness probe on each
# FROZEN encoder (measured before BC, so the --train_encoder trap can't touch it). THEN the higher
# bar: freeze, clone the head, roll out from pixels. The RANDOM ViT is the same class, never aligned.
aligned_enc = ViTEncoder(args.dim, args.depth, args.heads, frame_mean).to(device)
align_loss = align_encoder(aligned_enc, args.align_epochs)
torch.manual_seed(args.seed + 3)  # the RANDOM encoder: same class, a fresh independent init
random_enc = ViTEncoder(args.dim, args.depth, args.heads, frame_mean).to(device)
probe_aligned = control_probe(aligned_enc, "aligned")   # THE HEADLINE: held-out action-regression MSE
probe_random = control_probe(random_enc, "random")
probe_gap = probe_random - probe_aligned                # > 0 == aligned features are more control-useful
print(f"control-usefulness probe val_mse: aligned {probe_aligned:.4f}  random {probe_random:.4f}  "
      f"(gap {probe_gap:+.4f}) — {'aligned features carry the control signal' if probe_gap > 0 else 'no aligned advantage'}")
aligned_policy, bc_loss_aligned = build_and_train(aligned_enc, args.train_encoder, "aligned")
random_policy, bc_loss_random = build_and_train(random_enc, False, "random")
aligned = evaluate(aligned_policy, "aligned")
random = evaluate(random_policy, "random")
# Export the ALIGNED policy (the deployable one) to ONNX contract v1 and prove parity.
onnx_path = export_policy(aligned_policy, OBS_DIM, ACT_DIM, args.out / "pixels_policy.onnx")
parity_delta = assert_parity(aligned_policy, onnx_path, OBS_DIM)
print(f"exported {onnx_path} — torch/onnx parity delta {parity_delta:.2e}")
metrics = {
    "align_final_loss": round(align_loss, 6),
    "aligned_ci_hi": round(aligned["ci_hi"], 6),
    "aligned_ci_lo": round(aligned["ci_lo"], 6),
    "aligned_mean_return": round(aligned["mean_return"], 6),
    "aligned_success_rate": round(aligned["success_rate"], 6),
    "bc_final_loss_aligned": round(bc_loss_aligned, 6),
    "bc_final_loss_random": round(bc_loss_random, 6),
    "dim": args.dim,
    "eval_episodes": args.eval_episodes,
    "num_frames": int(num_frames),
    "parity_delta": round(parity_delta, 6),
    "probe_mse_gap": round(probe_gap, 6),                # HEADLINE: random_val_mse - aligned_val_mse; > 0 every seed
    "probe_val_mse_aligned": round(probe_aligned, 6),
    "probe_val_mse_random": round(probe_random, 6),
    "random_ci_hi": round(random["ci_hi"], 6),
    "random_ci_lo": round(random["ci_lo"], 6),
    "random_mean_return": round(random["mean_return"], 6),
    "random_success_rate": round(random["success_rate"], 6),
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "success_gap": round(aligned["success_rate"] - random["success_rate"], 6),  # HIGHER bar (may floor both: Scale Lab)
    "train_encoder": bool(args.train_encoder),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"HEADLINE (probe val_mse): aligned {probe_aligned:.4f} vs random {probe_random:.4f}, gap {probe_gap:+.4f} "
      f"(aligned<random is the direction). rollout: {aligned['success_rate']:.2f} vs {random['success_rate']:.2f} (Scale Lab)")
# demo/vizdata.json: side-by-side rollout from the SAME perturbed start, aligned vs random, for the
# load-bearing-vision toy. Geometry only (no binaries); a separate file so it never perturbs metrics.
viz_seed = 20_000 + args.seed
a_ok, a_ret, a_traj = rollout(aligned_policy, viz_seed, record=True)
r_ok, r_ret, r_traj = rollout(random_policy, viz_seed, record=True)
saliency = aligned_policy.encoder.blocks[-1].last_attn[0].reshape(IMG_HW // PATCH, IMG_HW // PATCH)
vizdata = {
    "provenance": f"pixels.py seed {args.seed}, device {args.device}, "
                  f"{'smoke' if args.smoke else 'default'} config; recorded replay, geometry only",
    "seed": args.seed,
    "world_half_extent_m": 0.45,
    "target": {"x": 0.0, "y": 0.0, "yaw": 0.0},
    "tee": {"bar_half": [0.06, 0.015], "stem_half": [0.015, 0.045], "stem_offset_y": -0.06},
    "labels": ["pusher_x", "pusher_y", "tee_x", "tee_y", "tee_yaw"],
    "aligned": {"success": a_ok, "mean_return": round(a_ret, 4), "frames": a_traj},
    "random": {"success": r_ok, "mean_return": round(r_ret, 4), "frames": r_traj},
    "saliency": {"grid": (IMG_HW // PATCH), "weights": saliency.cpu().double().numpy().round(6).tolist(),
                 "note": "aligned encoder: CLS attention over the 8x8 patch grid for the last frame"},
    "meta": {k: metrics[k] for k in ("probe_val_mse_aligned", "probe_val_mse_random", "probe_mse_gap",
                                     "aligned_success_rate", "random_success_rate", "num_frames", "dim")},
}
demo_dir = Path(__file__).resolve().parent / "demo"
demo_dir.mkdir(exist_ok=True)
(demo_dir / "vizdata.json").write_text(json.dumps(vizdata, indent=2) + "\n")
print(f"metrics: {args.out / 'metrics.json'}  |  vizdata: {demo_dir / 'vizdata.json'}")
if args.rerun:
    print(f"recording: {args.out / 'pixels.rrd'} — open it with: rerun {args.out / 'pixels.rrd'}")
# --- endregion ---
