"""zero2robot 5.1 — Patches & Attention: A ViT From Scratch.

Every VLA in Phase 1 conditioned on VISION through a FROZEN, random-init CNN
(ch1.7's FrozenVisionEncoder) — a fixed projection we were honest about: not
perception, just a compact stand-in. Phase 5 replaces it. This chapter builds the
architecture a real backbone (SigLIP, DINOv2) is made of — a Vision Transformer —
from scratch, over cached 64x64 PushT frames, and MEASURES that a trained ViT's
representation actually carries scene structure a frozen random one does not:

  (1) PATCHIFY: cut each 64x64 image into an 8x8 grid of 8x8x3 patches (64 tokens).
      A ViT has no convolution and no pixels-as-a-picture prior — a patch is a flat
      192-vector, and the model only knows two patches are neighbors because we TAG
      each with a LEARNED positional embedding. Get the reshape wrong and the patch
      grid interleaves with the pixel axis: patches scramble, silently (--break).
  (2) THE ViT: Linear patch-embed -> prepend a CLS token -> add learned positions ->
      a stack of pre-norm self-attention Blocks (the SAME block ch1.8 fused with;
      re-derived here, ~30 lines). The CLS row after the blocks is one vector that has
      attended over all 64 patches: the pooled scene representation.
  (3) LINEAR PROBE (the measurement): freeze the backbone, read a CHEAP scene fact off
      the CLS feature with a closed-form linear probe — which QUADRANT the PushT block
      sits in (a deterministic label from sim state, no annotation). A ViT trained on
      that task beats a random-init ViT of the SAME shape, which beats the majority
      guess. We report the DIRECTION (trained > random > majority), seed-robust — never
      an exact %: MuJoCo rasterization is not bitwise across CPU arches and the probe is
      a noisy small-held-out metric (ch1.6). Training also leans on a MODEST lr + an LR
      WARMUP — drop the warmup (--warmup 1) and some seeds pin at chance for the whole run
      (the from-scratch transformer's cold-start pathology; warmup makes it reliable).

The misconception this chapter kills: "a ViT sees the image as a picture." It does not — it
sees a permutation-invariant BAG of patch vectors plus learned position TAGS. Two --break
demos make that concrete, and both land on the SAME surprising fact:
  * --break patch_interleave — the classic patchify RESHAPE bug (patch-grid axis interleaved
    with the pixel axis). It globally permutes the patches, and a coarse scene fact is a
    bag-of-patches property, so the quadrant probe is UNMOVED — the bug is SILENT to your
    accuracy. That is the footgun: a patchify bug hides from the metric and only shows in the
    attention map. Predict the probe drop, run it, and explain why there isn't one.
  * --break shuffle_pos — scramble the position TAGS with the PIXELS UNTOUCHED. The coarse
    probe shrugs (a bag barely needs order), but the CLS attention map scrambles: the model
    was never looking at a picture, only at tokens wearing position tags.

Run it:      python curriculum/phase5_practitioner/ch5.1_vit/vit.py --seed 0
Break it:    python curriculum/phase5_practitioner/ch5.1_vit/vit.py --seed 0 --break patch_interleave
CI smoke:    python curriculum/phase5_practitioner/ch5.1_vit/vit.py --smoke --seed 0 --no-rerun
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

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht.pusht_env import PushTEnv  # noqa: E402
from curriculum.common.envs.pusht.scripted_expert import ScriptedExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

IMG_HW = 64          # the free-tier floor: 64x64 frames (NOT ch1.7's 96 — half the pixels)
PATCH = 8            # 8x8 patches -> an 8x8 = 64-patch grid, +1 CLS = 65 tokens
GRID = IMG_HW // PATCH          # 8 patches per side
NUM_PATCHES = GRID * GRID       # 64 image tokens
PATCH_DIM = PATCH * PATCH * 3   # 192 numbers per flattened patch
NUM_QUADRANTS = 4    # the probe target: which quadrant of the table the block sits in

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch5.1-vit"))
parser.add_argument("--episodes", type=int, default=40, help="scripted-expert episodes to cache frames from")  # smoke: 3
parser.add_argument("--frame_stride", type=int, default=4, help="keep every Nth control frame (neighbors near-duplicate)")
parser.add_argument("--dim", type=int, default=96)      # transformer width. T4: 128 | 4090: 384 | smoke: 48
parser.add_argument("--depth", type=int, default=2)     # pre-norm attention blocks. deeper needs more DATA (scale lab) | smoke: 1
parser.add_argument("--heads", type=int, default=3)     # attention heads (dim must divide by this)
parser.add_argument("--epochs", type=int, default=120)  # cpu-laptop: ~40 s | smoke: 2
parser.add_argument("--warmup", type=int, default=15,
                    help="linear LR-warmup epochs. Drop it (--warmup 1) and some seeds pin at chance all run (measured) — the real lesson")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=1e-3,
                    help="a MODEST lr on purpose: too hot and the from-scratch ViT plateaus at chance for hundreds of steps (measured)")
parser.add_argument("--seed", type=int, default=0, help="seeds the demos, the frame render, and every init")
parser.add_argument("--break", dest="break_mode",
                    choices=("patch_interleave", "shuffle_pos"), default=None,
                    help="patch_interleave = the buggy patchify reshape (patches globally permute; SILENT to the "
                         "coarse probe — that is the lesson); shuffle_pos = permute the position tags, pixels "
                         "untouched (attention map scrambles; refutes 'a ViT sees a picture')")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true",
                    help="tiny hermetic CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; every model init below draws from torch's RNG in a fixed order
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.episodes, args.dim, args.depth, args.heads = 3, 48, 1, 3
    args.epochs, args.warmup, args.device = 2, 1, "cpu"
banner("ch5.1-vit", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
# One CPU generator feeds every stochastic draw (batch order) in a fixed order; tensors
# then move to `device`: same seed -> byte-identical CPU run (ch1.8's determinism recipe).
gen = torch.Generator().manual_seed(args.seed)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch5.1-vit", spawn=False)
    rr.save(str(args.out / "vit.rrd"))
# --- endregion ---

# --- region: data ---
# Cache frames ONCE from the scripted expert (ch1.7's replay-in-process pattern), at the
# free-tier 64x64. Each cached frame carries a CHEAP label read straight off sim state:
# which quadrant the T-block occupies. No human annotation, no learned label — the probe
# target is a deterministic function of the state that produced the pixels.
def quadrant(tee_x: float, tee_y: float) -> int:
    """0=NE, 1=NW, 2=SW, 3=SE. The target sits at the origin; the block spawns in an
    annulus around it, so the sign of (tee_x, tee_y) is an unambiguous scene fact."""
    top, right = tee_y >= 0.0, tee_x >= 0.0
    return {(True, True): 0, (True, False): 1, (False, False): 2, (False, True): 3}[(top, right)]


def cache_frames(episodes: int, seed: int, stride: int):
    """Replay the scripted PushT expert and keep every Nth frame + its quadrant label +
    the episode it came from. Deterministic given seed (ch1.7's contract)."""
    env = PushTEnv()
    frames, labels, ep_index = [], [], []
    for e in range(episodes):
        obs = env.reset(seed + e)
        expert = ScriptedExpert(noise=0.0, seed=seed + e)
        step, done = 0, False
        while not done:
            action = expert.action(env)
            if step % stride == 0:  # obs is the CURRENT state; render_frame renders that same state
                frames.append(env.render_frame(IMG_HW, IMG_HW))
                labels.append(quadrant(float(obs[2]), float(obs[3])))  # obs[2:4] = tee_x, tee_y
                ep_index.append(e)
            obs, _, done, _ = env.step(action)
            step += 1
    return (np.asarray(frames, np.uint8), np.asarray(labels, np.int64), np.asarray(ep_index, np.int64))


frames, labels, ep_index = cache_frames(args.episodes, args.seed, args.frame_stride)
# Split by EPISODE (not by frame) so near-duplicate frames from one episode never straddle
# train/test — the honest way to measure a representation (ch1.6). Last ~25% of episodes -> test.
test_ep = ep_index >= int(math.ceil(args.episodes * 0.75))
train_idx = np.where(~test_ep)[0]
test_idx = np.where(test_ep)[0]
if len(test_idx) == 0:  # tiny smoke budgets can leave no held-out episode; fall back to a frame split
    test_idx, train_idx = train_idx[-len(train_idx) // 3:], train_idx[: -len(train_idx) // 3]
# Frames as float in [-1, 1], on device, kept whole so patchify runs on the batch (B, 64, 64, 3).
images = (torch.from_numpy(frames).to(device).float() / 127.5 - 1.0)
labels_t = torch.from_numpy(labels).to(device)
majority_baseline = float(np.bincount(labels[test_idx], minlength=NUM_QUADRANTS).max() / len(test_idx))
print(f"cached {len(frames)} frames ({len(train_idx)} train / {len(test_idx)} test), "
      f"quadrant balance {np.bincount(labels, minlength=NUM_QUADRANTS).tolist()}")
# --- endregion ---

# --- region: patches ---
# A ViT sees NO image — it sees a SEQUENCE of flat patch vectors. patchify cuts the
# (B, H, W, 3) image into an 8x8 grid of 8x8x3 patches and flattens each to 192 numbers.
# The grid is row-major: token t=(grid_row, grid_col). The reshape is the whole trick and
# the whole trap: to keep each patch's pixels contiguous you must PERMUTE the grid axes in
# front of the within-patch axes. Skip the permute and the patch-grid axis interleaves with
# the pixel axis — "patches" become a global permutation of the real ones. The trap is that
# this is SILENT to a coarse probe (a permutation of a bag is the same bag), so the quadrant
# accuracy barely moves; the damage only shows in the attention map. That is the deliberate
# failure the exercise has you write, predict, and diagnose (--break patch_interleave).
def patchify(images: torch.Tensor, patch: int, interleave_bug: bool = False) -> torch.Tensor:
    """(B, H, W, C) -> (B, num_patches, patch*patch*C). Correct: grid axes first, then
    pixel axes, so each row of the output is ONE spatially-contiguous patch."""
    B, H, W, C = images.shape
    gh, gw = H // patch, W // patch
    x = images.reshape(B, gh, patch, gw, patch, C)      # split H->(gh,patch), W->(gw,patch)
    if not interleave_bug:
        x = x.permute(0, 1, 3, 2, 4, 5)                 # (B, gh, gw, patch, patch, C): patch contiguous
    # BUG PATH (no permute): flattening (B, gh, patch, gw, patch, C) mixes grid-row with
    # patch-row into the token axis and grid-col into the feature axis — patches interleave.
    return x.reshape(B, gh * gw, patch * patch * C)
# --- endregion ---

# --- region: model ---
# The ViT, from scratch. A transformer Block is the SAME shape ch1.8 fused vision+language
# with: pre-norm multi-head self-attention (Q/K/V are nn.Linear, a scaled-dot-product
# softmax mixes tokens, an output projection), then a per-token MLP, both on residuals. No
# transformers, no einops. Here there is no key padding — the sequence is a fixed 1+64 tokens.
class Block(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.last_attn = None  # (B, L, L) head-averaged attention, kept for the rollout viz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, dim = x.shape
        h, hd = self.heads, dim // self.heads
        qkv = self.qkv(self.ln1(x)).reshape(B, L, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # each (B, h, L, hd)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)  # (B, h, L, L)
        attn = scores.softmax(dim=-1)
        self.last_attn = attn.mean(1).detach()              # (B, L, L): average over heads
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(B, L, dim))
        return x + self.mlp(self.ln2(x))


class TinyViT(nn.Module):
    """Patch-embed -> [CLS, patch_0..patch_63] + learned positions -> pre-norm blocks. The
    CLS row after the blocks is the pooled scene representation; `head` reads the quadrant."""

    def __init__(self, dim: int, depth: int, heads: int, num_classes: int) -> None:
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_DIM, dim)               # Conv2d(3,dim,8,stride=8) is the SAME op
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(0.02 * torch.randn(1, NUM_PATCHES + 1, dim))  # LEARNED position tags
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def features(self, images: torch.Tensor, interleave_bug: bool = False,
                 shuffle_pos: torch.Tensor | None = None) -> torch.Tensor:
        B = images.shape[0]
        tokens = self.patch_embed(patchify(images, PATCH, interleave_bug))  # (B, 64, dim)
        seq = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1)        # prepend CLS -> (B, 65, dim)
        pos = self.pos
        if shuffle_pos is not None:  # --break shuffle_pos: permute the 64 PATCH position tags (keep CLS at 0)
            pos = torch.cat([self.pos[:, :1], self.pos[:, 1:][:, shuffle_pos]], dim=1)
        seq = seq + pos
        for blk in self.blocks:
            seq = blk(seq)
        return self.norm(seq[:, 0])   # the CLS row: pooled over all 64 patches

    def forward(self, images: torch.Tensor, interleave_bug: bool = False) -> torch.Tensor:
        return self.head(self.features(images, interleave_bug))
# --- endregion ---

# --- region: train ---
# Train the ViT end-to-end to classify the block's quadrant. Nothing about the label needs
# a ViT — a linear probe on raw pixels could do it — the POINT is what the ATTENTION learns:
# to route the CLS token to the patches that actually contain the block. Cross-entropy, Adam,
# and a LINEAR LR-WARMUP: a from-scratch transformer cold-starts badly, and without warmup
# some seeds pin at chance for the whole run (measured — try --warmup 1). Warmup + a modest lr
# are why the model trains RELIABLY across seeds. --break patch_interleave trains through the
# buggy (globally-permuted) patchify — it still fits the coarse label (the trap), but its
# attention never lines up with the image.
INTERLEAVE = args.break_mode == "patch_interleave"
torch.manual_seed(args.seed)  # trained ViT init
model = TinyViT(args.dim, args.depth, args.heads, NUM_QUADRANTS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
train_images, train_labels = images[train_idx], labels_t[train_idx]
loss_fn = nn.CrossEntropyLoss()
train_loss, step = float("nan"), 0
for epoch in range(args.epochs):
    for group in optimizer.param_groups:  # linear warmup, then hold: lr ramps 0 -> args.lr over --warmup epochs
        group["lr"] = args.lr * min(1.0, (epoch + 1) / args.warmup)
    model.train()
    epoch_loss, nb = 0.0, 0
    for batch in torch.randperm(len(train_idx), generator=gen).split(args.batch_size):
        logits = model(train_images[batch], INTERLEAVE)
        loss = loss_fn(logits, train_labels[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss, nb = epoch_loss + loss.item(), nb + 1
        if args.rerun:
            rr.set_time("step", sequence=step)
            rr.log("vit/loss/train", rr.Scalars([loss.item()]))
        step += 1
    train_loss = epoch_loss / nb
    if epoch % 5 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:3d}  ce {train_loss:.4f}")

# A random-init ViT of the SAME shape, never trained — the reference the probe must beat.
torch.manual_seed(args.seed + 1)
random_vit = TinyViT(args.dim, args.depth, args.heads, NUM_QUADRANTS).to(device)
# --break shuffle_pos: a FIXED permutation of the 64 patch position tags (pixels untouched).
shuffle = torch.randperm(NUM_PATCHES, generator=gen).to(device) if args.break_mode == "shuffle_pos" else None
# --- endregion ---

# --- region: probe ---
# THE MEASUREMENT. Freeze each backbone, extract its CLS feature on train+test, and fit a
# CLOSED-FORM linear probe (least-squares onto one-hot quadrants — deterministic, no SGD,
# ch1.7's lstsq trick). Test accuracy of that probe is representation QUALITY: how linearly
# the scene fact falls out of the pooled feature. Trained ViT > random-init ViT > majority.
@torch.no_grad()
def cls_features(vit: TinyViT, imgs: torch.Tensor) -> np.ndarray:
    vit.eval()
    return vit.features(imgs, INTERLEAVE, shuffle).cpu().double().numpy()


def linear_probe(feat_tr: np.ndarray, y_tr: np.ndarray, feat_te: np.ndarray, y_te: np.ndarray) -> float:
    onehot = np.zeros((len(y_tr), NUM_QUADRANTS))
    onehot[np.arange(len(y_tr)), y_tr] = 1.0
    X = np.concatenate([feat_tr, np.ones((len(feat_tr), 1))], axis=1)  # + intercept column
    W, *_ = np.linalg.lstsq(X, onehot, rcond=None)
    pred = (np.concatenate([feat_te, np.ones((len(feat_te), 1))], axis=1) @ W).argmax(1)
    return float((pred == y_te).mean())


y_tr, y_te = labels[train_idx], labels[test_idx]
probe_trained = linear_probe(cls_features(model, images[train_idx]), y_tr,
                             cls_features(model, images[test_idx]), y_te)
probe_random = linear_probe(cls_features(random_vit, images[train_idx]), y_tr,
                            cls_features(random_vit, images[test_idx]), y_te)


# Attention ROLLOUT: how the CLS token's attention flows to each patch through the whole
# stack. Add the residual (0.5*A + 0.5*I), renormalize, multiply the per-layer matrices, and
# read the CLS row over the 64 patches -> an 8x8 map. Trained: concentrates on the block.
# Random-init: washes out near-uniform. (Abnar & Zuidema's rollout, the standard ViT viz.)
@torch.no_grad()
def cls_rollout(vit: TinyViT, imgs: torch.Tensor) -> np.ndarray:
    vit.eval()
    vit.features(imgs, INTERLEAVE, shuffle)  # populate each block's last_attn
    rolled = None
    for blk in vit.blocks:
        a = blk.last_attn
        a = 0.5 * a + 0.5 * torch.eye(a.shape[-1], device=a.device)[None]
        a = a / a.sum(-1, keepdim=True)
        rolled = a if rolled is None else a @ rolled
    grid = rolled[:, 0, 1:].reshape(-1, GRID, GRID)  # CLS -> patches, as an 8x8 map
    return (grid / grid.amax(dim=(1, 2), keepdim=True)).cpu().double().numpy()  # per-frame normalized
# --- endregion ---

# --- region: report ---
metrics = {
    "break_mode": args.break_mode or "none",
    "chance": round(1.0 / NUM_QUADRANTS, 6),
    "depth": int(args.depth),
    "dim": int(args.dim),
    "heads": int(args.heads),
    "majority_baseline": round(majority_baseline, 6),   # predict the most common quadrant
    "num_frames": int(len(frames)),
    "num_patches": int(NUM_PATCHES),
    "num_test": int(len(test_idx)),
    "num_train": int(len(train_idx)),
    "patch_size": int(PATCH),
    "probe_acc_random": round(probe_random, 6),         # random-init ViT, same shape
    "probe_acc_trained": round(probe_trained, 6),       # the headline representation
    "probe_gap": round(probe_trained - probe_random, 6),  # DIRECTION: > 0 is the seed-robust claim
    "seed": int(args.seed),
    "smoke": bool(args.smoke),
    "train_final_ce": round(train_loss, 6),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

# demo/vizdata.json: for a few test frames spanning quadrants, the frame + the CLS-attention
# 8x8 map for the trained AND the random-init ViT — everything the attention-map viewer needs.
viz_pick = np.unique(np.linspace(0, len(test_idx) - 1, min(4, len(test_idx))).astype(int))
viz_idx = test_idx[viz_pick]
attn_tr = cls_rollout(model, images[viz_idx])
attn_rand = cls_rollout(random_vit, images[viz_idx])
vizdata = {
    "grid": GRID, "patch": PATCH, "img_hw": IMG_HW, "quadrants": ["NE", "NW", "SW", "SE"],
    "break_mode": args.break_mode or "none",
    "probe_acc_trained": metrics["probe_acc_trained"], "probe_acc_random": metrics["probe_acc_random"],
    "thumb_hw": IMG_HW // 2,  # frames are emitted at 32x32 (every 2nd pixel) to bound JSON size; the toy scales up
    "frames": [
        {"quadrant": int(labels[i]),
         "image": frames[i][::2, ::2].tolist(),         # 32x32x3 uint8 thumbnail; the toy scales it
         "attn_trained": attn_tr[j].round(4).tolist(),  # 8x8, normalized to [0,1]
         "attn_random": attn_rand[j].round(4).tolist()}
        for j, i in enumerate(viz_idx)
    ],
}
(args.out / "demo").mkdir(parents=True, exist_ok=True)
(args.out / "demo" / "vizdata.json").write_text(json.dumps(vizdata) + "\n")

if args.rerun:
    rr.log("probe/trained", rr.Scalars([probe_trained]), static=True)
    rr.log("probe/random", rr.Scalars([probe_random]), static=True)
    rr.log("probe/majority", rr.Scalars([majority_baseline]), static=True)
    for j, i in enumerate(viz_idx):  # the money picture: attention on the block vs washed out
        rr.set_time("viz_frame", sequence=j)
        rr.log("viz/frame", rr.Image(frames[i]))
        rr.log("viz/attn_trained", rr.Image((attn_tr[j] * 255).astype(np.uint8)))
        rr.log("viz/attn_random", rr.Image((attn_rand[j] * 255).astype(np.uint8)))

print(f"\nlinear probe (quadrant acc): trained {probe_trained:.2f}  |  random-init {probe_random:.2f}  "
      f"|  majority {majority_baseline:.2f}  |  chance {1.0 / NUM_QUADRANTS:.2f}")
print(f"probe_gap (trained - random) = {metrics['probe_gap']:+.2f}  [break={args.break_mode or 'none'}]")
print(f"wrote {args.out / 'metrics.json'} + {args.out / 'demo' / 'vizdata.json'}")
if args.rerun:
    print(f"recording: {args.out / 'vit.rrd'} — open it with: rerun {args.out / 'vit.rrd'}")
# --- endregion ---
