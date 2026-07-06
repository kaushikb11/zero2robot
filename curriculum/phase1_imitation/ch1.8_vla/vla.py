"""zero2robot 1.8 — Tokens Meet Torques: The Tiny VLA, Part II (train it).

Chapter 1.7 built the DATA: a multi-task pile of (instruction_tokens, image_features,
state) -> action examples, with a from-scratch word-level tokenizer and a FROZEN,
random-init tiny CNN standing in for a real vision backbone. Chapter 1.5 built the
ACTION HEAD: flow matching — learn the velocity of the straight noise->data line,
then sample an action by integrating an ODE. This file FUSES them into one
language-conditioned policy and trains it, entirely from scratch:

  (1) A TINY VLM BACKBONE (from scratch, NO transformers lib): embed the instruction
      token ids, project the frozen image feature and the state into the same width,
      lay them out as one sequence [CLS, vision, state, tok_0..tok_15], and run a few
      self-attention blocks (attention IS a few nn.Linear + a softmax — built here).
      The CLS output is one vector that has SEEN words, pixels-as-features, and numbers
      at once: the fused conditioning representation.
  (2) The ch1.5 FLOW-MATCHING head, now conditioned on that fused vector instead of a
      bare state. Same velocity objective, same forward-Euler ODE sampler.

Train on ch1.7's .npz, evaluate on the PushT (and ALOHA) envs with ch1.6's error-bar
rigor (a Wilson interval, because VLA success is noisy). THE HONEST LESSON (measured,
not asserted): a tiny from-scratch VLA with a RANDOM vision encoder is WEAK. On PushT
it works — but largely from the state, which already determines the expert action; the
random-vision channel barely helps (Break It: --break blind zeros vision and success
hardly moves). ALOHA's bimanual handoff it essentially cannot do. That gap is exactly
why real VLAs (SmolVLA, OpenVLA) bolt on a PRETRAINED backbone and a subword LM — the
Scale Lab. From-scratch teaches the mechanism; adapt-pretrained is what performs.

Run it:      python curriculum/phase1_imitation/ch1.8_vla/vla.py --seed 0
Break it:    python curriculum/phase1_imitation/ch1.8_vla/vla.py --seed 0 --break blind
CI smoke:    python curriculum/phase1_imitation/ch1.8_vla/vla.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as the other chapters).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.aloha_cube.aloha_cube_env import AlohaCubeEnv  # noqa: E402
from curriculum.common.envs.pusht.pusht_env import PushTEnv  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

STATE_DIM = 10       # both envs expose a 10-number state (ch1.7)
ACT_DIM = 6          # the padded shared action width (pusht uses dims 0:2, aloha 0:6)
MAX_TOKENS = 16      # fixed instruction length (ch1.7's tokenizer)
IMG_HW = 96          # camera size both envs render (ch1.7)
PAD_ID = 0           # <pad> is token id 0 by construction (ch1.7 tokenizer)
TIME_DIM = 32        # sinusoidal flow-time embedding width (ch1.5)
TIME_SCALE = 1000.0  # flow time lives in [0,1]; scale up so the embed has resolution (ch1.5)
Z95 = 1.959963985    # 0.975 standard-normal quantile — the 95% Wilson interval (ch1.6)
VLA_DATA = Path(__file__).resolve().parents[1] / "ch1.7_vla_data" / "vla_data.py"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=Path("outputs/ch1.8-vla-data"),
                    help="dataset dir (ch1.7's vla_dataset.npz + manifest.json); regenerated via ch1.7 if absent")
parser.add_argument("--episodes_per_task", type=int, default=60,
                    help="demos/task when regenerating: ch1.7's inspection default (12) starves the policy; the VLA needs more")
parser.add_argument("--out", type=Path, default=Path("outputs/ch1.8-vla"))
parser.add_argument("--model_dim", type=int, default=64)     # transformer width. T4: 128 | smoke: 16
parser.add_argument("--layers", type=int, default=2)         # self-attention blocks. T4: 4 | smoke: 1
parser.add_argument("--heads", type=int, default=4)          # attention heads (model_dim must divide by this)
parser.add_argument("--hidden", type=int, default=128)       # velocity-head MLP width. T4: 256 | smoke: 16
parser.add_argument("--flow_steps", type=int, default=6,
                    help="Euler steps that integrate the action ODE at eval; straight paths need few (ch1.5). smoke: 2")
parser.add_argument("--epochs", type=int, default=200)       # cpu-laptop: ~2 min | smoke: 2
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--eval_episodes", type=int, default=12,  # per task; few episodes is noisy (ch1.6). T4: 30 | smoke: 2
                    help="PushT eval episodes; ALOHA uses half (it is far harder and mostly fails)")
parser.add_argument("--seed", type=int, default=0, help="seeds the data, the frozen encoder, inits, and the ODE sampler")
parser.add_argument("--break", dest="break_mode", choices=("blind",), default=None,
                    help="blind = zero the image feature (train+eval); PushT success barely moves — the VLA never used vision")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; the FROZEN ENCODER we rebuild below draws from torch's RNG next
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.model_dim, args.layers, args.heads, args.hidden = 16, 1, 2, 16
    args.flow_steps, args.epochs, args.eval_episodes, args.device = 2, 2, 2, "cpu"
# Mirror ch1.7's frozen-encoder recipe so we can rebuild the IDENTICAL encoder for
# live eval frames (ch1.7 saved features, not weights or frames — see the vision region).
CONV_WIDTH = 8 if args.smoke else 16
BLIND = args.break_mode == "blind"
banner("ch1.8-vla", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
# One CPU generator feeds every stochastic draw (flow noise, batch order, the sampler)
# in a fixed order; tensors then move to `device`: same seed -> byte-identical CPU run.
gen = torch.Generator().manual_seed(args.seed)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch1.8-vla", spawn=False)
    rr.save(str(args.out / "vla.rrd"))
# --- endregion ---

# --- region: data ---
# Consume ch1.7's dataset (do NOT re-derive it). If the .npz is absent, regenerate it
# by running ch1.7 at the matching seed/smoke config — the same seed makes the frozen
# encoder we rebuild below match the one that produced these features.
npz_path = args.data / "vla_dataset.npz"
if not npz_path.is_file():
    print(f"no dataset at {npz_path} — regenerating via ch1.7 (episodes_per_task={args.episodes_per_task}) ...")
    subprocess.run([sys.executable, str(VLA_DATA), "--seed", str(args.seed), "--device", "cpu",
                    "--no-rerun", "--out", str(args.data), "--episodes_per_task", str(args.episodes_per_task)]
                   + (["--smoke"] if args.smoke else []),
                   check=True, cwd=Path(__file__).resolve().parents[3])
d = np.load(npz_path)
tokens_np = d["instruction_tokens"].astype(np.int64)     # (N, 16) token ids
image_feats = d["image_features"].astype(np.float32)     # (N, feature_dim) frozen CNN
states = d["state"].astype(np.float32)                    # (N, 10)
actions = d["action"].astype(np.float32)                 # (N, 6) zero-padded
act_mask_np = d["action_mask"].astype(np.float32)        # (N, 6) 1.0 where the dim is real
task_ids = d["task_id"].astype(np.int64)                 # (N,) 0=pusht, 1=aloha
manifest = json.loads((args.data / "manifest.json").read_text())
vocab = manifest["vocab"]
feature_dim = int(manifest["feature_dim"])
# Guard: the frozen features must come from OUR seed, or the encoder we rebuild for
# eval will not match them (a silent train/eval vision mismatch).
ch17_metrics = json.loads((args.data / "metrics.json").read_text())
assert ch17_metrics["seed"] == args.seed, f"ch1.7 data was built at seed {ch17_metrics['seed']}, not {args.seed}"
# We rebuild ch1.7's frozen encoder from its DEFAULT/smoke recipe (CONV_WIDTH + this
# feature_dim); a dataset built at any other config would not match. Fail loudly rather
# than train on features our eval encoder cannot reproduce.
assert feature_dim == (16 if args.smoke else 64), (
    f"ch1.7 data has feature_dim {feature_dim}; ch1.8 rebuilds the encoder at the "
    f"{'smoke' if args.smoke else 'default'} recipe (feature_dim {16 if args.smoke else 64}). "
    f"Regenerate: rm -rf {args.data} and re-run.")

# Normalization (ch1.5/1.6 pattern): standardize the flowed actions (per-dim, over the
# VALID entries only, so pusht's zero-pad does not corrupt aloha's stats), min-max the
# state, and standardize the image feature. Stats become model buffers so a checkpoint
# carries them. constant dims get range 1 (a constant, not a divide-by-zero).
valid = act_mask_np.astype(bool)
act_mean = np.array([actions[valid[:, j], j].mean() if valid[:, j].any() else 0.0 for j in range(ACT_DIM)], np.float32)
act_std = np.array([actions[valid[:, j], j].std() if valid[:, j].any() else 1.0 for j in range(ACT_DIM)], np.float32)
act_std = np.where(act_std < 1e-4, np.float32(1.0), act_std)
state_min = states.min(0)
state_range = np.where(states.max(0) - state_min < 1e-4, np.float32(1.0), states.max(0) - state_min)
feat_mean = image_feats.mean(0)
feat_std = np.where(image_feats.std(0) < 1e-4, np.float32(1.0), image_feats.std(0))
STATS = {"act_mean": act_mean, "act_std": act_std, "state_min": state_min,
         "state_range": state_range, "feat_mean": feat_mean, "feat_std": feat_std}
act_mean_t = torch.from_numpy(act_mean).to(device)
act_std_t = torch.from_numpy(act_std).to(device)
print(f"dataset: {len(states)} examples ({int((task_ids == 0).sum())} pusht / {int((task_ids == 1).sum())} aloha), "
      f"vocab {len(vocab)}, feature_dim {feature_dim}, blind={BLIND}")
# --- endregion ---

# --- region: vision_language ---
# The frozen encoder, rebuilt IDENTICALLY to ch1.7's (same class, same seed, same first
# torch-RNG draw after set_seed — envs/experts touch no torch RNG, verified). At eval we
# render a live frame and must project it exactly as training did; ch1.7 saved the
# features but not the weights, so the recipe IS the interface we carry forward.
class FrozenVisionEncoder(nn.Module):
    def __init__(self, width: int, out_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.ReLU(),              # 96 -> 48
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.ReLU(),      # 48 -> 24
            nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1), nn.ReLU(),  # 24 -> 12
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(4 * width, out_dim)
        for p in self.parameters():
            p.requires_grad_(False)  # FROZEN — never trained here or in ch1.7
        self.eval()

    @torch.no_grad()
    def forward(self, images_uint8: torch.Tensor) -> torch.Tensor:
        x = images_uint8.to(torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0
        return self.head(self.stem(x).flatten(1))


encoder = FrozenVisionEncoder(CONV_WIDTH, feature_dim).to(device)  # FIRST torch-RNG use: matches ch1.7


def encode_instruction(text: str) -> np.ndarray:
    """Word-level tokenizer (ch1.7): [BOS] ids [EOS], OOV -> <unk>, pad/truncate."""
    stoi = {w: i for i, w in enumerate(vocab)}
    ids = [stoi["<bos>"]] + [stoi.get(w, stoi["<unk>"]) for w in text.split()] + [stoi["<eos>"]]
    ids = ids[:MAX_TOKENS] + [PAD_ID] * (MAX_TOKENS - len(ids))
    return np.asarray(ids[:MAX_TOKENS], dtype=np.int64)
# --- endregion ---

# --- region: model ---
# The tiny VLM + flow head, from scratch. No transformers, no einops — a transformer
# block is a from-scratch multi-head attention (Q,K,V projections, a scaled dot-product
# softmax, an output projection) plus an MLP, with pre-norm residuals. That is the whole
# "backbone"; the lesson is the FUSION, not a big architecture.
def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous flow time (B,) -> (B, dim) sinusoidal features (ch1.5 / ch1.4)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


class Block(nn.Module):
    """One pre-norm transformer block: self-attention that lets vision, state, and each
    word token exchange information, then a per-token MLP. Built from nn.Linear only."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.last_attn = None  # CLS-token attention over the sequence, for the rerun viz

    def forward(self, x: torch.Tensor, key_pad: torch.Tensor) -> torch.Tensor:
        B, L, dim = x.shape
        h, hd = self.heads, dim // self.heads
        qkv = self.qkv(self.ln1(x)).reshape(B, L, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # each (B, h, L, hd)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd)   # (B, h, L, L)
        scores = scores.masked_fill(key_pad[:, None, None, :], float("-inf"))  # ignore <pad> keys
        attn = scores.softmax(dim=-1)
        self.last_attn = attn[:, :, 0, :].mean(1).detach()   # (B, L): how CLS attends to each input
        x = x + self.proj((attn @ v).transpose(1, 2).reshape(B, L, dim))
        return x + self.mlp(self.ln2(x))


class TinyVLA(nn.Module):
    """Fuse instruction + vision + state -> one conditioning vector, then predict the
    flow velocity of the action conditioned on it. The sequence is
    [CLS, vision, state, tok_0..tok_15]; the CLS output (after the blocks) is the fused
    representation the flow head sees. blind=True zeros the vision input (Break It)."""

    def __init__(self, vocab_size: int, feat_dim: int, dim: int, layers: int, heads: int,
                 hidden: int, stats: dict) -> None:
        super().__init__()
        self.blind = BLIND
        self.tok_embed = nn.Embedding(vocab_size, dim, padding_idx=PAD_ID)
        self.vision_proj = nn.Linear(feat_dim, dim)
        self.state_proj = nn.Linear(STATE_DIM, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(0.02 * torch.randn(1, 3 + MAX_TOKENS, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.vel = nn.Sequential(
            nn.Linear(ACT_DIM + TIME_DIM + dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, ACT_DIM),
        )
        for name, value in stats.items():
            self.register_buffer(name, torch.from_numpy(value))

    def fuse(self, tokens: torch.Tensor, img_feat: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        feat = (img_feat - self.feat_mean) / self.feat_std
        if self.blind:  # Break It: the policy gets NO vision — must solve from words + state
            feat = torch.zeros_like(feat)
        st = (2.0 * (state - self.state_min) / self.state_range - 1.0).clamp(-1.0, 1.0)
        seq = torch.cat([
            self.cls.expand(B, -1, -1),        # a learned query that reads out the fusion
            self.vision_proj(feat)[:, None],   # the frozen image feature, as one token
            self.state_proj(st)[:, None],      # the state, as one token
            self.tok_embed(tokens),            # the instruction, one token per word id
        ], dim=1) + self.pos
        key_pad = torch.zeros(B, 3 + MAX_TOKENS, dtype=torch.bool, device=tokens.device)
        key_pad[:, 3:] = tokens == PAD_ID      # attention ignores padded instruction slots
        for blk in self.blocks:
            seq = blk(seq, key_pad)
        return self.norm(seq[:, 0])            # the CLS row: the fused conditioning vector

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.vel(torch.cat([x_t, sinusoidal_embed(t * TIME_SCALE, TIME_DIM), cond], dim=1))


def flow_loss(model: TinyVLA, x0: torch.Tensor, mask: torch.Tensor,
              tokens: torch.Tensor, img: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """ch1.5's conditional flow-matching loss, now conditioned on the fused VLM vector
    and MASKED to each embodiment's real action dims (pusht's padded 2:6 never train)."""
    t = torch.rand(len(x0), generator=gen).to(device)
    noise = torch.randn(x0.shape, generator=gen).to(device)
    x_t = (1.0 - t)[:, None] * noise + t[:, None] * x0
    target_v = x0 - noise                                     # velocity of the straight noise->data line
    pred = model.velocity(x_t, t, model.fuse(tokens, img, state))
    # Average the velocity MSE over each example's VALID dims, THEN over examples, so a
    # 2-D PushT frame and a 6-D ALOHA frame weigh the same. Summing instead lets the
    # higher-DOF embodiment dominate the gradient and starve the other task (measured).
    return ((((pred - target_v) ** 2) * mask).sum(1) / mask.sum(1)).mean()


@torch.no_grad()
def sample_action(model: TinyVLA, cond: torch.Tensor, steps: int) -> torch.Tensor:
    """Sample by integrating the velocity ODE from noise (ch1.5), in standardized space."""
    x = torch.randn((cond.shape[0], ACT_DIM), generator=gen).to(device)
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((cond.shape[0],), i * dt, device=device)
        x = x + dt * model.velocity(x, t, cond)              # forward Euler along the field
    return x
# --- endregion ---

# --- region: train ---
# Standardize actions once (masked), then train the fused policy with the flow loss.
torch.manual_seed(args.seed)  # policy init reproducible, independent of the frozen encoder above
gen.manual_seed(args.seed)    # fresh flow-noise stream for training
x0_all = torch.from_numpy((actions - act_mean) / act_std).to(device) * torch.from_numpy(act_mask_np).to(device)
mask_t = torch.from_numpy(act_mask_np).to(device)
tokens_t = torch.from_numpy(tokens_np).to(device)
feats_t = torch.from_numpy(image_feats).to(device)
states_t = torch.from_numpy(states).to(device)
policy = TinyVLA(len(vocab), feature_dim, args.model_dim, args.layers, args.heads, args.hidden, STATS).to(device)
optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
shuffle = torch.Generator().manual_seed(args.seed + 1)  # torch-side RNG for batch order
train_loss, step = float("nan"), 0
for epoch in range(args.epochs):
    epoch_loss, nb = 0.0, 0
    for batch in torch.randperm(len(states_t), generator=shuffle).split(args.batch_size):
        loss = flow_loss(policy, x0_all[batch], mask_t[batch], tokens_t[batch], feats_t[batch], states_t[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss, nb = epoch_loss + loss.item(), nb + 1
        if args.rerun:
            rr.set_time("step", sequence=step)
            rr.log("policy/loss/train", rr.Scalars([loss.item()]))
        step += 1
    train_loss = epoch_loss / nb
    if epoch % 10 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:3d}  flow_mse {train_loss:.5f}")
# --- endregion ---

# --- region: eval ---
# Loss measures velocity fits; rollouts measure the task. At each env step: render a
# frame, encode it with the FROZEN encoder, fuse with the (fixed, per-task) instruction
# and current state, and SAMPLE an action by ODE integration. Report a Wilson 95%
# interval on the success rate — VLA success is noisy and a bare % lies (ch1.6).
TASKS = [(PushTEnv, 0, 2), (AlohaCubeEnv, 1, 6)]  # (env class, task_id, real action dims)


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials (ch1.6; numpy-free math)."""
    if n == 0:
        return (0.0, 1.0)
    p, z = k / n, Z95
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


@torch.no_grad()
def rollout(model: TinyVLA, env_cls, tok_row: np.ndarray, act_dim: int, ep_seed: int) -> tuple[bool, float]:
    env = env_cls()
    obs = env.reset(ep_seed)
    gen.manual_seed(ep_seed)  # seed the sampler from the episode: reproducible AND order-independent
    tok = torch.from_numpy(tok_row).to(device).unsqueeze(0)
    done, info, ret = False, {}, 0.0
    while not done:
        feat = encoder(torch.from_numpy(env.render_frame(IMG_HW, IMG_HW)[None]).to(device))
        cond = model.fuse(tok, feat, torch.from_numpy(obs[None]).to(device))
        x = sample_action(model, cond, args.flow_steps)
        action = (x * act_std_t + act_mean_t)[0, :act_dim].cpu().numpy().clip(-1.0, 1.0)
        obs, reward, done, info = env.step(action)
        ret += reward
    return bool(info["success"]), ret


def evaluate(model: TinyVLA, env_cls, task_id: int, act_dim: int, episodes: int, tag: str) -> tuple[int, int, float]:
    instruction = manifest["tasks"][task_id]["templates"][0]  # a fixed, held-in instruction for this task
    tok_row = encode_instruction(instruction)
    outcomes = [rollout(model, env_cls, tok_row, act_dim, 10_000 + args.seed + ep) for ep in range(episodes)]
    k = sum(s for s, _ in outcomes)
    mean_return = float(np.mean([r for _, r in outcomes]))
    lo, hi = wilson_ci(k, episodes)
    # mean_return separates a policy that DRIVES TOWARD the goal (0% success but better
    # shaped reward) from one that wanders — the honest learning signal when success is 0.
    print(f"eval[{tag:16s}] {manifest['tasks'][task_id]['name']:6s}: success {k}/{episodes} = "
          f"{k / episodes:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  mean_return {mean_return:.2f}")
    if args.rerun:
        rr.log(f"eval/{tag}/success_rate", rr.Scalars([k / episodes]))
        rr.log(f"eval/{tag}/ci", rr.Scalars([lo, hi]))
        rr.log(f"eval/{tag}/mean_return", rr.Scalars([mean_return]))
    return k, episodes, mean_return


# A fixed random-init reference (ch1.5 pattern): shows the trained number is signal.
torch.manual_seed(args.seed + 2)
baseline = TinyVLA(len(vocab), feature_dim, args.model_dim, args.layers, args.heads, args.hidden, STATS).to(device)
policy.eval()
kp, np_, ret_p = evaluate(policy, PushTEnv, 0, 2, args.eval_episodes, "trained")
kb, nb_, ret_b = evaluate(baseline, PushTEnv, 0, 2, args.eval_episodes, "untrained")
ka, na, ret_a = evaluate(policy, AlohaCubeEnv, 1, 6, max(1, args.eval_episodes // 2), "trained")

# Fused-attention viz: how much the CLS token attends to vision vs state vs each word,
# read off the last block for one PushT example — the picture of what the fusion uses.
if args.rerun:
    tok0 = torch.from_numpy(encode_instruction(manifest["tasks"][0]["templates"][0])).to(device).unsqueeze(0)
    policy.fuse(tok0, feats_t[:1], states_t[:1])
    rr.log("fusion/cls_attention", rr.BarChart(policy.blocks[-1].last_attn[0].cpu().double().numpy()))
# --- endregion ---

# --- region: report ---
pusht_rate, base_rate = kp / np_, kb / nb_
aloha_rate = ka / na
metrics = {
    "aloha_mean_return": round(ret_a, 6),
    "aloha_success_ci_hi": round(wilson_ci(ka, na)[1], 6),
    "aloha_success_ci_lo": round(wilson_ci(ka, na)[0], 6),
    "aloha_success_rate": round(aloha_rate, 6),
    "baseline_pusht_mean_return": round(ret_b, 6),
    "baseline_pusht_success_rate": round(base_rate, 6),
    "blind": BLIND,
    "break_mode": args.break_mode or "none",
    "epochs": args.epochs,
    "eval_episodes": args.eval_episodes,
    "feature_dim": feature_dim,
    "final_train_loss": round(train_loss, 6),
    "flow_steps": args.flow_steps,
    "model_dim": args.model_dim,
    "num_examples": int(len(states)),
    "pusht_mean_return": round(ret_p, 6),
    "pusht_success_ci_hi": round(wilson_ci(kp, np_)[1], 6),
    "pusht_success_ci_lo": round(wilson_ci(kp, np_)[0], 6),
    "pusht_success_rate": round(pusht_rate, 6),
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "vocab_size": int(len(vocab)),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"\nPushT trained {pusht_rate:.2f} vs untrained {base_rate:.2f}; ALOHA trained {aloha_rate:.2f}")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'vla.rrd'} — open it with: rerun {args.out / 'vla.rrd'}")
# --- endregion ---
