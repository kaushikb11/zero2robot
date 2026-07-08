"""zero2robot 5.2 — Why Aligned: Contrastive Vision-Language Pretraining.

Chapter 5.1 asked "why SEE?" and trained a tiny ViT with a SUPERVISED probe —
labels (which quadrant the block is in) pull the pixels into a feature a linear
head can read. This chapter asks a harder question: can an image and the WORDS
that describe it land in the SAME embedding space, with NO labels at all — only
the knowledge of which caption came with which frame? That is contrastive
vision-language pretraining, the recipe behind CLIP and the vision backbones real
VLAs (SigLIP in SmolVLA/OpenVLA) are built on. We build it from scratch:

  (1) TWO TOWERS. An IMAGE tower (a tiny ViT — patch-embed a 64x64 frame into 64
      tokens, a CLS token, learned positions, a few pre-norm attention blocks) and
      a trivial TEXT tower (word-level tokenizer -> embed -> one block -> masked
      mean-pool). Each projects to one small, L2-normalized embedding.
  (2) SYMMETRIC INFONCE. In a batch of B (frame, caption) pairs the B x B cosine matrix
      should be bright on its diagonal (each frame matches ITS caption) and dark off it.
      Cross-entropy in BOTH directions — image->text AND text->image — pulls the shared
      space together, scaled by a LEARNED temperature. The only supervision is the PAIRING.
  (3) ALIGNMENT = RETRIEVAL. We measure the space by retrieval: type "the block is near the
      top left corner" and rank held-out frames by cosine. A good space puts top-left frames
      on top. The contrastive encoder beats BOTH a SUPERVISED-probe encoder (ch5.1's recipe:
      it learns only its LABEL, recovering the quadrant but losing the rest of the caption)
      AND a RANDOM-init encoder — the DIRECTION, seed-robust, not an exact number.

THE MISCONCEPTION this chapter kills: "contrastive learning needs labels." It does not — the
supervised baseline DOES use quadrant labels and still loses to the label-free contrastive tower.
Two free-tier notes (prose): these 64x64 frames are ~98% constant background, so we
SUBTRACT the mean frame (the ViT must see the block); and contrastive wants a big batch
(128-256) so each frame gets many in-batch negatives — a harder, better lesson.

Run it:      python curriculum/phase5_practitioner/ch5.2_align/align.py --seed 0
Break it:    python curriculum/phase5_practitioner/ch5.2_align/align.py --seed 0 --break noneg
CI smoke:    python curriculum/phase5_practitioner/ch5.2_align/align.py --smoke --seed 0 --no-rerun
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
import torch.nn.functional as F

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as the other chapters).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht.pusht_env import PushTEnv  # noqa: E402
from curriculum.common.envs.pusht.scripted_expert import ScriptedExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

IMG_HW = 64          # cache + train at the free-tier floor (NOT ch1.7's 96); patch 8 -> 8x8 = 64 tokens
PATCH = 8            # 8x8 pixel patches -> 64 image tokens
MAX_TOKENS = 14      # fixed caption length: [BOS] + words + [EOS], then <pad>
PAD_ID = 0           # <pad> is token id 0 by construction (the tokenizer below)
NUM_QUAD = 4         # 2x2 block-position quadrants — ch5.1's cheap label, derived from sim state

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch5.2-align"))
parser.add_argument("--episodes", type=int, default=80, help="PushT expert episodes to cache frames from. cpu: 80 | smoke: 4")
parser.add_argument("--frame_stride", type=int, default=3, help="keep every Nth frame (consecutive frames are near-duplicates)")
parser.add_argument("--dim", type=int, default=96, help="tower width. T4: 96 | 4090: 256 | smoke: 32")
parser.add_argument("--depth", type=int, default=3, help="image-ViT attention blocks. T4: 3 | smoke: 1")
parser.add_argument("--heads", type=int, default=3, help="attention heads (dim must divide by this)")
parser.add_argument("--embed_dim", type=int, default=64, help="shared image/text embedding width (aligned + random towers)")
parser.add_argument("--epochs", type=int, default=30, help="contrastive epochs. cpu: ~3 min | smoke: 2")
parser.add_argument("--sup_epochs", type=int, default=15, help="supervised-probe pretrain epochs for the ch5.1 baseline")
parser.add_argument("--batch_size", type=int, default=128, help="in-batch negatives (contrastive wants 128-256). smoke: 16")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=0, help="seeds the demos, the tower inits, and the batch order")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--break", dest="break_mode", choices=("noneg",), default=None,
                    help="noneg = pull each pair together with NO in-batch negatives (the classic InfoNCE bug): "
                         "the space collapses and retrieval, while still above chance, is measurably worse")
parser.add_argument("--smoke", action="store_true", help="tiny hermetic CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # seeds python/numpy/torch; tower inits draw from torch's RNG below
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.episodes, args.dim, args.depth, args.heads, args.embed_dim = 4, 32, 1, 2, 32
    args.epochs, args.sup_epochs, args.batch_size, args.device = 2, 2, 16, "cpu"
banner("ch5.2-align", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
gen = torch.Generator().manual_seed(args.seed)  # one CPU generator feeds every batch shuffle
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch5.2-align", spawn=False)
    rr.save(str(args.out / "align.rrd"))
# --- endregion ---

# --- region: data ---
# ~2-4k (frame, caption) pairs from PushT. We replay the scripted expert in-process (ch1.7
# pattern) and render 64x64 frames; the block sweeps the table as the expert pushes it to
# center, so the cache spans every quadrant and radius. The CAPTION is generated from the
# block's position in SIM STATE — a cheap label, no human annotation (the shared-arc trick):
# a quadrant word (2x2 from the sign of tee_xy) and a near/far word (radius, split at the
# median). Contrastive pretraining never SEES the quadrant/near-far labels — it only knows
# which caption came with which frame; the labels only train the SUPERVISED baseline + SCORE.
QUAD_WORDS = ["bottom left", "bottom right", "top left", "top right"]  # index by quadrant id 0..3
NF_WORDS = ["near", "far"]                                             # index by the near/far bit
TEMPLATES = [                                                          # paraphrases, one picked per frame
    "the block is {nf} the {q} corner",
    "the tee sits {nf} the {q}",
    "push toward the block {nf} the {q}",
    "the t piece is {nf} the {q} region",
]


def collect_frames(episodes: int, seed: int, stride: int):
    """Replay the PushT expert; return (frames uint8 (N,64,64,3), tee_xy (N,2))."""
    env = PushTEnv()
    frames, tee = [], []
    for e in range(episodes):
        obs = env.reset(seed + e)
        expert = ScriptedExpert(noise=0.0, seed=seed + e)
        step, done = 0, False
        while not done:
            action = expert.action(env)
            if step % stride == 0:                       # subsample near-duplicate frames
                frames.append(env.render_frame(IMG_HW, IMG_HW))
                tee.append(obs[2:4].copy())              # obs idx 2,3 = tee_x, tee_y (PushT obs layout)
            obs, _, done, _ = env.step(action)
            step += 1
    return np.asarray(frames, np.uint8), np.asarray(tee, np.float32)


frames, tee = collect_frames(args.episodes, args.seed, args.frame_stride)
num_pairs = len(frames)
# Background subtraction: a raw frame is ~98% constant table, so center on the MEAN frame so the
# ViT sees the block (a linear probe on raw pixels is at chance; on centered pixels near-perfect).
frame_mean = (frames.astype(np.float32) / 127.5 - 1.0).mean(0)   # (64, 64, 3) — subtracted in ImageTower
radius = np.hypot(tee[:, 0], tee[:, 1])
radius_split = float(np.median(radius))                  # data-driven near/far boundary (deterministic given frames)
quadrant = ((tee[:, 1] >= 0).astype(np.int64) * 2 + (tee[:, 0] >= 0).astype(np.int64))  # 0..3 (top=+y, right=+x)
near_far = (radius >= radius_split).astype(np.int64)     # 0=near, 1=far
cls = quadrant * 2 + near_far                            # 0..7 fine class (quadrant x near/far)
variant = (args.seed + np.arange(num_pairs)) % len(TEMPLATES)   # one paraphrase per frame, seed-deterministic
captions = [TEMPLATES[variant[i]].format(nf=NF_WORDS[near_far[i]], q=QUAD_WORDS[quadrant[i]])
            for i in range(num_pairs)]
# --- endregion ---

# --- region: language ---
# The whole tokenizer, from scratch (re-contained from ch1.7 — chapters do not import each
# other): a fixed WORD-LEVEL vocab (no BPE, no HF tokenizers), 4 special ids, pad/truncate.
# The corpus is CLOSED, so the vocab is fixed (no OOV). A real CLIP/SigLIP uses a 30k+ subword
# tokenizer; here the point is the mechanism (text -> ids -> a learned embedding), laid bare.
class Tokenizer:
    def __init__(self, corpus: list[str]) -> None:
        words = sorted({w for text in corpus for w in text.split()})
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + words   # ids 0..3 are the specials
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> np.ndarray:
        ids = [self.stoi["<bos>"]] + [self.stoi.get(w, 1) for w in text.split()] + [self.stoi["<eos>"]]
        ids = ids[:MAX_TOKENS] + [PAD_ID] * (MAX_TOKENS - len(ids))
        return np.asarray(ids[:MAX_TOKENS], dtype=np.int64)


# The vocab is the closure of the templates over every (quadrant, near/far) fill.
corpus = [t.format(nf=nf, q=q) for t in TEMPLATES for nf in NF_WORDS for q in QUAD_WORDS]
tokenizer = Tokenizer(corpus)
tokens = np.stack([tokenizer.encode(c) for c in captions])       # (N, MAX_TOKENS) int64


class TextTower(nn.Module):
    """The TRIVIAL tower: embed the token ids, run ONE attention block so words can talk,
    masked-mean-pool to one vector, and project into the shared space. Kept deliberately
    small — the interesting tower is the image ViT; text just needs to name the scene."""

    def __init__(self, vocab_size: int, dim: int, heads: int, embed_dim: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.pos = nn.Parameter(0.02 * torch.randn(1, MAX_TOKENS, dim))
        self.block = Block(dim, heads)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, embed_dim)

    def forward(self, tok: torch.Tensor) -> torch.Tensor:
        pad = tok == PAD_ID                                      # (B, L) True where padded
        x = self.block(self.embed(tok) + self.pos, pad)
        keep = (~pad).float()[:, :, None]                        # mask pads OUT of the mean
        pooled = (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        return self.proj(self.norm(pooled))                      # (B, embed_dim), pre-L2-norm
# --- endregion ---

# --- region: vision ---
# The image tower is a tiny ViT — the SAME shape ch5.1 built and the SAME attention block
# ch1.8 uses, re-derived here (chapters re-contain their backbone; do NOT import ch5.1).
class Block(nn.Module):
    """One pre-norm transformer block: multi-head self-attention (Q/K/V from nn.Linear +
    a scaled-dot-product softmax) then a per-token MLP. key_pad masks padded text keys;
    the image tower has no padding and passes None."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, key_pad: torch.Tensor | None) -> torch.Tensor:
        B, L, dim = x.shape
        h, hd = self.heads, dim // self.heads
        qkv = self.qkv(self.ln1(x)).reshape(B, L, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                          # each (B, h, L, hd)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)       # (B, h, L, L)
        if key_pad is not None:
            scores = scores.masked_fill(key_pad[:, None, None, :], float("-inf"))
        x = x + self.proj((scores.softmax(-1) @ v).transpose(1, 2).reshape(B, L, dim))
        return x + self.mlp(self.ln2(x))


class ImageTower(nn.Module):
    """Patch-embed a 64x64 frame into 8x8 = 64 tokens, prepend a CLS token, add learned
    positions, run the attention blocks, read the fused scene off CLS, project to the
    shared space. This IS ch5.1's ViT; here its CLS output must MATCH language, not a label."""

    def __init__(self, dim: int, depth: int, heads: int, embed_dim: int, frame_mean: np.ndarray) -> None:
        super().__init__()
        n_tok = (IMG_HW // PATCH) ** 2                            # 64 patches
        self.patch = nn.Conv2d(3, dim, PATCH, stride=PATCH)      # (B,3,64,64) -> (B,dim,8,8)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(0.02 * torch.randn(1, n_tok + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, embed_dim)
        self.register_buffer("frame_mean", torch.from_numpy(frame_mean))  # background, subtracted below

    def forward(self, images_uint8: torch.Tensor) -> torch.Tensor:
        x = images_uint8.to(torch.float32) / 127.5 - 1.0         # (B,64,64,3) in [-1,1]
        x = (x - self.frame_mean).permute(0, 3, 1, 2)            # center on background, then (B,3,64,64)
        x = self.patch(x).flatten(2).transpose(1, 2)             # (B, 64, dim)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1) + self.pos
        for blk in self.blocks:
            x = blk(x, None)
        return self.proj(self.norm(x[:, 0]))                     # (B, embed_dim), pre-L2-norm
# --- endregion ---

# --- region: contrastive ---
class Aligner(nn.Module):
    """Two towers + a LEARNED temperature. encode_* returns L2-normalized embeddings so a
    dot product IS a cosine. logit_scale is stored in log space and init to CLIP's 1/0.07. The
    SUPERVISED baseline builds one with embed_dim=NUM_QUAD, so its image tower IS the classifier."""

    def __init__(self, vocab_size: int, dim: int, depth: int, heads: int, embed_dim: int,
                 frame_mean: np.ndarray) -> None:
        super().__init__()
        self.image = ImageTower(dim, depth, heads, embed_dim, frame_mean)
        self.text = TextTower(vocab_size, dim, heads, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

    def encode_image(self, imgs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image(imgs), dim=-1)      # onto the unit sphere: a dot product IS a cosine

    def encode_text(self, tok: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text(tok), dim=-1)


def info_nce(img_e: torch.Tensor, txt_e: torch.Tensor, logit_scale: torch.Tensor,
             negatives: bool) -> torch.Tensor:
    """Symmetric InfoNCE over a batch's B x B cosine matrix. The B matched pairs are the
    positives (the diagonal); every OTHER caption in the batch is a NEGATIVE — that is why a
    bigger batch is a harder, better lesson, and why contrastive needs no labels (the negatives
    come free from PAIRING). Cross-entropy is applied in BOTH directions — image->text AND
    text->image — the symmetric pull CLIP uses, scaled by the learned temperature.
    THE BUG (--break noneg): keep the positive pull but DROP the negatives (just push each pair's
    cosine to 1). With nothing pushing non-matches apart the space collapses and retrieval,
    though still above chance, is measurably worse — negatives are the whole point."""
    if not negatives:
        return (1.0 - (img_e * txt_e).sum(-1)).mean()        # positives only -> collapse (the deliberate failure)
    scale = logit_scale.exp().clamp(max=100.0)               # clamp like CLIP so temperature can't blow up
    logits = scale * img_e @ txt_e.t()                       # (B, B): logits[i,j] = cos(image_i, text_j) / temp
    labels = torch.arange(len(logits), device=logits.device)  # the positive for row i is column i
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def train_contrastive(model: Aligner, imgs: torch.Tensor, tok: torch.Tensor, epochs: int,
                      negatives: bool, freeze_image: bool, tag: str) -> float:
    if freeze_image:                                         # the supervised baseline aligns text to a FROZEN image tower
        for p in model.image.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    loss_val = float("nan")
    for epoch in range(epochs):
        total, nb = 0.0, 0
        for batch in torch.randperm(len(imgs), generator=gen).split(args.batch_size):
            loss = info_nce(model.encode_image(imgs[batch]), model.encode_text(tok[batch]),
                            model.logit_scale, negatives)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total, nb = total + loss.item(), nb + 1
        loss_val = total / nb
        if args.rerun:
            rr.set_time("epoch", sequence=epoch)
            rr.log(f"contrastive/{tag}/loss", rr.Scalars([loss_val]))
            rr.log(f"contrastive/{tag}/temperature", rr.Scalars([1.0 / model.logit_scale.exp().clamp(max=100.0).item()]))
    return loss_val


def supervised_pretrain(model: Aligner, imgs: torch.Tensor, quad: torch.Tensor, epochs: int) -> None:
    """ch5.1's recipe, re-contained: train the image tower to CLASSIFY the block quadrant.
    With embed_dim=NUM_QUAD the tower's output IS the 4-way logits — no extra head. We freeze
    it and align text to it; it recovers the quadrant but LOSES the near/far half of the
    caption the contrastive tower keeps. Supervised != aligned."""
    opt = torch.optim.Adam(model.image.parameters(), lr=args.lr)
    for _ in range(epochs):
        for batch in torch.randperm(len(imgs), generator=gen).split(args.batch_size):
            loss = F.cross_entropy(model.image(imgs[batch]), quad[batch])
            opt.zero_grad()
            loss.backward()
            opt.step()
# --- endregion ---

# --- region: eval ---
# ALIGNMENT = RETRIEVAL. Split the cache into disjoint held-out GALLERY (images) and QUERY
# (captions) sets so no caption can trivially retrieve its own frame. For each query caption,
# rank gallery frames by cosine, take the top-1: is it the RIGHT class? retrieval@1 (fine,
# 8-way) demands quadrant AND near/far; (quad, 4-way) is coarser. Chance ~1/8 and ~1/4.
@torch.no_grad()
def retrieval(model: Aligner, gal_imgs: torch.Tensor, qry_tok: torch.Tensor,
              gal_cls: torch.Tensor, qry_cls: torch.Tensor) -> tuple[float, float]:
    img_e = model.encode_image(gal_imgs)                     # (G, D)
    top1 = (model.encode_text(qry_tok) @ img_e.t()).argmax(dim=1)   # best-matching frame per query
    fine = (gal_cls[top1] == qry_cls).float().mean().item()
    quad = ((gal_cls[top1] // 2) == (qry_cls // 2)).float().mean().item()
    return fine, quad


# Held-out split: shuffle (seeded), last 30% held-out, halved into disjoint gallery / query;
# the rest (70%) trains the towers. No held-out frame is ever trained on.
perm = rng.permutation(num_pairs)
n_hold = max(4, int(0.30 * num_pairs))
train_idx, hold = perm[:-n_hold], perm[-n_hold:]
gal_idx, qry_idx = hold[: len(hold) // 2], hold[len(hold) // 2:]
imgs_t = torch.from_numpy(frames).to(device)
tok_t = torch.from_numpy(tokens).to(device)
quad_t = torch.from_numpy(quadrant).to(device)
cls_t = torch.from_numpy(cls).to(device)
tr_imgs, tr_tok, tr_quad = imgs_t[train_idx], tok_t[train_idx], quad_t[train_idx]
gal_imgs, gal_cls = imgs_t[gal_idx], cls_t[gal_idx]
qry_tok, qry_cls = tok_t[qry_idx], cls_t[qry_idx]
print(f"pairs: {num_pairs} ({len(train_idx)} train / {len(gal_idx)} gallery / {len(qry_idx)} query), "
      f"vocab {tokenizer.vocab_size}, radius_split {radius_split:.3f}, break={args.break_mode or 'none'}")

# THREE encoders, one retrieval task. (1) ALIGNED: both towers, symmetric InfoNCE.
negatives = args.break_mode != "noneg"
torch.manual_seed(args.seed)
aligned = Aligner(tokenizer.vocab_size, args.dim, args.depth, args.heads, args.embed_dim, frame_mean).to(device)
loss_aligned = train_contrastive(aligned, tr_imgs, tr_tok, args.epochs, negatives, False, "aligned")
at1_aligned, quad_aligned = retrieval(aligned, gal_imgs, qry_tok, gal_cls, qry_cls)

# (2) RANDOM: both towers random-init, NEVER trained — the floor. Retrieval ~ chance.
torch.manual_seed(args.seed + 1)
random_model = Aligner(tokenizer.vocab_size, args.dim, args.depth, args.heads, args.embed_dim, frame_mean).to(device)
at1_random, quad_random = retrieval(random_model, gal_imgs, qry_tok, gal_cls, qry_cls)

# (3) SUPERVISED: ch5.1's probe. embed_dim=NUM_QUAD makes the image tower the 4-way quadrant
# classifier; pretrain it on LABELS, freeze it, align text to it. Uses labels the contrastive
# tower never saw — and still loses, because a label-shaped space can't hold the whole caption.
torch.manual_seed(args.seed + 2)
supervised = Aligner(tokenizer.vocab_size, args.dim, args.depth, args.heads, NUM_QUAD, frame_mean).to(device)
supervised_pretrain(supervised, tr_imgs, tr_quad, args.sup_epochs)
train_contrastive(supervised, tr_imgs, tr_tok, args.epochs, True, True, "supervised")
at1_supervised, quad_supervised = retrieval(supervised, gal_imgs, qry_tok, gal_cls, qry_cls)
temp_final = 1.0 / aligned.logit_scale.exp().clamp(max=100.0).item()  # the learned temperature
print(f"retrieval@1 a/s/r  fine {at1_aligned:.3f}/{at1_supervised:.3f}/{at1_random:.3f}  "
      f"quad {quad_aligned:.3f}/{quad_supervised:.3f}/{quad_random:.3f}")
# --- endregion ---

# --- region: report ---
# The toy: type an instruction, watch which cached frames light up — aligned vs random, side by
# side. We emit one canonical query per quadrant, the top-5 gallery frames each encoder returns,
# and every gallery frame's block position + class so the site draws each scene (NO image binaries).
CANON = [TEMPLATES[0].format(nf="near", q=QUAD_WORDS[q]) for q in range(NUM_QUAD)]
canon_tok = torch.from_numpy(np.stack([tokenizer.encode(c) for c in CANON])).to(device)


@torch.no_grad()
def top5(model: Aligner) -> list:
    sims = model.encode_text(canon_tok) @ model.encode_image(gal_imgs).t()   # (4, G)
    return [[int(j) for j in row] for row in sims.argsort(1, descending=True)[:, :5].cpu().numpy()]


gal_tee = tee[gal_idx]
vizdata = {
    "provenance": (f"curriculum/phase5_practitioner/ch5.2_align/align.py, seed {args.seed}, "
                   f"--device {args.device}; retrieval over {len(gal_idx)} held-out gallery frames. "
                   "Aligned = symmetric InfoNCE; random = untrained init. Positions are the block's "
                   "tee_xy in sim state (no camera binaries)."),
    "quadrant_words": QUAD_WORDS,
    "gallery": [{"idx": i, "tee_x": round(float(gal_tee[i, 0]), 4), "tee_y": round(float(gal_tee[i, 1]), 4),
                 "quadrant": int(gal_cls[i].item() // 2), "near_far": int(gal_cls[i].item() % 2)}
                for i in range(len(gal_idx))],
    "queries": [{"quadrant": q, "instruction": CANON[q]} for q in range(NUM_QUAD)],
    "aligned_top5": top5(aligned),
    "random_top5": top5(random_model),
}
(args.out / "demo").mkdir(exist_ok=True)
(args.out / "demo" / "vizdata.json").write_text(json.dumps(vizdata, indent=2) + "\n")
if args.rerun:
    for tag, f, q in (("aligned", at1_aligned, quad_aligned), ("supervised", at1_supervised, quad_supervised),
                      ("random", at1_random, quad_random)):
        rr.log(f"retrieval/{tag}/at1_fine", rr.Scalars([f]), static=True)
        rr.log(f"retrieval/{tag}/at1_quadrant", rr.Scalars([q]), static=True)

metrics = {
    "break_mode": args.break_mode or "none",
    "embed_dim": int(args.embed_dim),
    "final_contrastive_loss": round(loss_aligned, 6),
    "num_gallery": int(len(gal_idx)),
    "num_pairs": int(num_pairs),
    "num_query": int(len(qry_idx)),
    "radius_split": round(radius_split, 6),
    "retrieval_at1_aligned": round(at1_aligned, 6),        # fine (8-way): quadrant AND near/far — the headline
    "retrieval_at1_random": round(at1_random, 6),
    "retrieval_at1_supervised": round(at1_supervised, 6),
    "retrieval_quad_at1_aligned": round(quad_aligned, 6),  # coarse (4-way): quadrant only
    "retrieval_quad_at1_random": round(quad_random, 6),
    "retrieval_quad_at1_supervised": round(quad_supervised, 6),
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "temperature_learned": round(temp_final, 6),           # the learned softmax temperature (1 / exp(logit_scale))
    "vocab_size": int(tokenizer.vocab_size),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"aligned {at1_aligned:.3f} >> supervised {at1_supervised:.3f} > random {at1_random:.3f} "
      f"(retrieval@1 fine); temperature {temp_final:.4f}")
print(f"metrics: {args.out / 'metrics.json'}  |  vizdata: {args.out / 'demo' / 'vizdata.json'}")
if args.rerun:
    print(f"recording: {args.out / 'align.rrd'} — open it with: rerun {args.out / 'align.rrd'}")
# --- endregion ---
