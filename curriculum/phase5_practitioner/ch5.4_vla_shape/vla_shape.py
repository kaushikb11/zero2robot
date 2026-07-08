"""zero2robot 5.4 — The Production VLA Shape: Prefix, Suffix, and the Action Expert.

Chapter 1.8 built a tiny VLA as ONE tower ([CLS, vision, state, tokens] -> read CLS -> ONE flowed
action). Production VLAs (pi0, SmolVLA, OpenVLA-OFT) have a different SHAPE; this chapter graduates
ch1.8 into it, from scratch, PushT-only, no transformers lib. A PREFIX (the VLM: vision + state +
instruction tokens, bidirectional, no CLS) and a SUFFIX (the ACTION EXPERT: H learned action-query
tokens = the noised action CHUNK + the flow time; each reads the whole prefix and emits one step's
velocity -> a CHUNK of H actions, ch1.3). Three pieces the prose walks: the BLOCK-ATTENTION MASK
(prefix<->prefix full; suffix->prefix full — the cross-attention; suffix<->suffix intra-chunk; prefix
NEVER reads suffix, so it is KV-cacheable — pi0); the EXPERT AS SEPARATE WEIGHTS on the suffix
positions (the pi0 "mixture": one shared attention, per-tower Q/K/V/MLP — NOT a layer on a CLS
vector); the ch1.5 FLOW HEAD per token.

THE HEADLINE IS A MECHANISM (ch1.8's honesty). The expert's ONLY window onto the state, pixels, and
words is the suffix->prefix cross-attention. CUT that block (--break cut_cross) at inference and the
trained expert goes blind: its HELD-OUT VELOCITY FIT collapses toward the unconditional prior
(seed-robust: +0.6..+1.0 MSE) — routing, not just parameters, is load-bearing. The PushT ROLLOUT is
the HIGHER bar and FLOORS for BOTH masks: from-scratch on ch1.7's FROZEN RANDOM backbone can't drive
PushT (ch1.8's warning); ch5.2's aligned encoder is the upgrade that makes the pixels load-bearing.

Run it:      python curriculum/phase5_practitioner/ch5.4_vla_shape/vla_shape.py --seed 0
Break it:    python curriculum/phase5_practitioner/ch5.4_vla_shape/vla_shape.py --seed 0 --break cut_cross
CI smoke:    python curriculum/phase5_practitioner/ch5.4_vla_shape/vla_shape.py --smoke --seed 0 --no-rerun
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

# Chapter scripts run from the repo root; put it on sys.path so `curriculum.common` resolves.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht.pusht_env import PushTEnv  # noqa: E402
from curriculum.common.envs.pusht.scripted_expert import ScriptedExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

STATE_DIM = 10       # PushT's 10-number state (pusht_env.py); the whole prefix "state" token
ACT_DIM = 2          # PushT action [vx, vy] (no ALOHA here -> no action padding/masking)
IMG_HW = 64          # the free-tier pixel floor (ch5.1/5.3; the frozen CNN pools any size)
INSTR_TOKENS = 12    # fixed instruction length: [BOS] + words + [EOS], then padded (ch1.7)
PAD_ID = 0           # <pad> is token id 0 by construction (ch1.7 tokenizer)
TIME_DIM, TIME_SCALE = 32, 1000.0  # sinusoidal flow-time embed (ch1.5); t in [0,1] scaled up for resolution
Z95 = 1.959963985    # 0.975 standard-normal quantile — the 95% Wilson interval (ch1.6)
# PushT paraphrases (ch1.7's pile). Language barely varies PushT's action (ch1.7 R^2~0): the
# load-bearing prefix signal is the STATE, not the words — we say so honestly.
TEMPLATES = ["push the t block onto the target", "slide the tee onto the goal",
             "move the t shape to the target pose", "push the block until it covers the target"]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch5.4-vla-shape"))
parser.add_argument("--episodes", type=int, default=60, help="scripted-expert PushT demos to collect")  # smoke: 4
parser.add_argument("--frame_stride", type=int, default=2, help="keep every Nth control frame (near-duplicates)")
parser.add_argument("--horizon", type=int, default=8, help="H: action-chunk length the expert emits (ch1.3)")  # smoke: 4
parser.add_argument("--model_dim", type=int, default=64)     # tower width. T4: 128 | smoke: 16
parser.add_argument("--layers", type=int, default=2)         # shared attention blocks. T4: 4 | smoke: 1
parser.add_argument("--heads", type=int, default=4)          # attention heads (model_dim must divide by this)
parser.add_argument("--feature_dim", type=int, default=64)   # frozen-CNN visual feature width (ch1.7). smoke: 16
parser.add_argument("--flow_steps", type=int, default=6, help="Euler steps for the action ODE at eval (ch1.5)")  # smoke: 2
parser.add_argument("--epochs", type=int, default=150)       # cpu-laptop: ~2-3 min | smoke: 2
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--eval_episodes", type=int, default=12, help="held-out PushT rollout episodes")  # smoke: 2
parser.add_argument("--seed", type=int, default=0, help="seeds the demos, the frozen encoder, inits, and the sampler")
parser.add_argument("--break", dest="break_mode", choices=("cut_cross",), default=None,
                    help="cut_cross = sever suffix->prefix at inference: roll out the SAME trained expert blind to "
                         "the VLM. Every run logs the full-vs-cut held-out flow-MSE gap regardless.")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true", help="tiny hermetic CPU run; two runs -> byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()
set_seed(args.seed)  # seeds python/numpy/torch; the FROZEN ENCODER below draws torch's RNG first
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.episodes, args.horizon, args.model_dim, args.layers, args.heads = 4, 4, 16, 1, 2
    args.feature_dim, args.flow_steps, args.epochs, args.eval_episodes, args.device = 16, 2, 2, 2, "cpu"
CONV_WIDTH = 8 if args.smoke else 16     # frozen-CNN first-conv width (ch1.7 recipe, re-contained)
EVAL_CUT = args.break_mode == "cut_cross"  # --break: roll out the trained expert with cross-attention severed
H = args.horizon
banner("ch5.4-vla-shape", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
gen = torch.Generator().manual_seed(args.seed)  # one CPU RNG, fixed draw order -> byte-identical (ch1.8)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch5.4-vla-shape", spawn=False)
    rr.save(str(args.out / "vla_shape.rrd"))
# --- endregion ---
# --- region: data ---
# ch1.7's recipe, PushT-only and re-contained (no ALOHA -> no padding): replay the scripted expert;
# keep every Nth 64x64 frame + state + action + episode index (the index lets us build CHUNKS). A frozen
# RANDOM CNN featurizes them — ch1.7's stand-in, NOT perception (ch5.2 is the aligned upgrade).
class FrozenVisionEncoder(nn.Module):
    """ch1.7's conv stack, rebuilt: (B,64,64,3) uint8 -> (B, feature_dim), random-init + FROZEN. The
    SAME instance featurizes training frames AND live eval frames (no train/eval mismatch to guard)."""

    def __init__(self, width: int, out_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(4 * width, out_dim)
        for p in self.parameters():
            p.requires_grad_(False)  # FROZEN — never trained here
        self.eval()

    @torch.no_grad()
    def forward(self, images_uint8: torch.Tensor) -> torch.Tensor:
        x = images_uint8.to(torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0
        return self.head(self.stem(x).flatten(1))


encoder = FrozenVisionEncoder(CONV_WIDTH, args.feature_dim).to(device)  # FIRST torch-RNG use


def tokenize(text: str, stoi: dict) -> np.ndarray:
    """ch1.7's word-level tokenizer over a FIXED vocab (no BPE, no HF): [BOS] ids [EOS], OOV-><unk>."""
    ids = [stoi["<bos>"]] + [stoi.get(w, 1) for w in text.split()] + [stoi["<eos>"]]
    ids = ids[:INSTR_TOKENS] + [PAD_ID] * (INSTR_TOKENS - len(ids))
    return np.asarray(ids[:INSTR_TOKENS], dtype=np.int64)


def collect(episodes: int, seed: int, stride: int):
    env = PushTEnv()
    frames, states, actions, ep_index = [], [], [], []
    for e in range(episodes):
        obs = env.reset(seed + e)
        expert = ScriptedExpert(noise=0.0, seed=seed + e)
        step, done = 0, False
        while not done:
            action = expert.action(env)
            if step % stride == 0:
                frames.append(env.render_frame(IMG_HW, IMG_HW))
                states.append(obs.astype(np.float32))
                actions.append(action[:ACT_DIM].astype(np.float32))
                ep_index.append(e)
            obs, _, done, _ = env.step(action)
            step += 1
    return (np.asarray(frames, np.uint8), np.asarray(states, np.float32),
            np.asarray(actions, np.float32), np.asarray(ep_index, np.int64))


frames_np, states_np, actions_np, ep_np = collect(args.episodes, args.seed, args.frame_stride)
N = len(frames_np)
VOCAB = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted({w for t in TEMPLATES for w in t.split()})
STOI = {w: i for i, w in enumerate(VOCAB)}
tokens_np = np.stack([tokenize(TEMPLATES[(args.seed + int(e)) % len(TEMPLATES)], STOI) for e in ep_np])
# CHUNK targets (ch1.3): frame i -> its next H expert actions; pad the episode tail (masked out).
chunks_np = np.zeros((N, H, ACT_DIM), np.float32)
cmask_np = np.zeros((N, H), np.float32)
for e in np.unique(ep_np):
    idx = np.nonzero(ep_np == e)[0]
    for j, f in enumerate(idx):
        valid = min(H, len(idx) - j)
        chunks_np[f, :valid] = actions_np[idx[j:j + valid]]
        chunks_np[f, valid:] = actions_np[idx[-1]]
        cmask_np[f, :valid] = 1.0
# Split by EPISODE (last 25%) so near-duplicate frames never straddle train/test (ch1.6/ch5.1).
test_ep = ep_np >= int(math.ceil(args.episodes * 0.75))
train_idx, test_idx = np.where(~test_ep)[0], np.where(test_ep)[0]
if len(test_idx) == 0:  # tiny smoke budgets can leave no held-out episode; fall back to a frame split
    test_idx, train_idx = train_idx[-max(1, len(train_idx) // 3):], train_idx[:-max(1, len(train_idx) // 3)]
image_feats = torch.cat([encoder(torch.from_numpy(frames_np[i:i + 256]).to(device))
                         for i in range(0, N, 256)])                       # (N, feature_dim), frozen
# Normalize on the TRAIN split (ch1.5/1.8): standardize actions + feature, min-max state; const->1.
_safe = lambda a: np.where(a < 1e-4, np.float32(1.0), a).astype(np.float32)  # noqa: E731
tr_acts = chunks_np[train_idx][cmask_np[train_idx].astype(bool)]
st_tr, ft = states_np[train_idx], image_feats[torch.from_numpy(train_idx).to(device)]
STATS = {"act_mean": tr_acts.mean(0).astype(np.float32), "act_std": _safe(tr_acts.std(0)),
         "state_min": st_tr.min(0).astype(np.float32), "state_range": _safe(st_tr.max(0) - st_tr.min(0)),
         "feat_mean": ft.mean(0).cpu().numpy().astype(np.float32), "feat_std": _safe(ft.std(0).cpu().numpy())}
act_mean_t = torch.from_numpy(STATS["act_mean"]).to(device)
act_std_t = torch.from_numpy(STATS["act_std"]).to(device)
tokens_t = torch.from_numpy(tokens_np).to(device)
states_t = torch.from_numpy(states_np).to(device)
chunks_t = torch.from_numpy((chunks_np - STATS["act_mean"]) / STATS["act_std"]).to(device)
cmask_t = torch.from_numpy(cmask_np).to(device)
print(f"dataset: {args.episodes} PushT episodes / {N} frames ({len(train_idx)} train / {len(test_idx)} test), "
      f"vocab {len(VOCAB)}, feature_dim {args.feature_dim}, horizon {H}")
# --- endregion ---
# --- region: model ---
# The two-tower VLA: P prefix [vision, state, tokens] then H suffix (action-expert) positions.
P = 2 + INSTR_TOKENS  # prefix length: 1 vision + 1 state + L instruction tokens


def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous flow time (B,) -> (B, dim) sinusoidal features (ch1.5)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


def block_mask(cut_cross: bool, dev) -> torch.Tensor:
    """THE block-attention mask, an additive bias (0 = allowed, -inf = blocked), (S, S). Row =
    query, col = key. prefix=[0,P), suffix=[P,P+H). prefix<->prefix full (the VLM fusion);
    suffix->prefix full (the expert READS the VLM — the cross-attention); suffix<->suffix full
    (the H chunk steps coordinate); prefix->suffix BLOCKED (the prefix never reads the actions, so
    it is action-independent and KV-cacheable — pi0). cut_cross drops ONLY the suffix->prefix block."""
    S = P + H
    allowed = torch.zeros(S, S, dtype=torch.bool, device=dev)
    allowed[:P, :P] = True            # prefix <-> prefix
    allowed[P:, P:] = True            # suffix <-> suffix (intra-chunk)
    if not cut_cross:
        allowed[P:, :P] = True        # suffix -> prefix (the cross-attention we can cut)
    return torch.where(allowed, 0.0, float("-inf"))


def per_tower(x: torch.Tensor, f_pre, f_suf) -> torch.Tensor:
    """Run f_pre on the first P (prefix) positions and f_suf on the rest (suffix), then re-join —
    the 'separate weights, shared sequence' trick in one line."""
    return torch.cat([f_pre(x[:, :P]), f_suf(x[:, P:])], dim=1)


class ExpertBlock(nn.Module):
    """ONE pre-norm self-attention over the joint [prefix|suffix] sequence — but the prefix and the
    suffix tokens each own their Q/K/V, output projection, MLP, and norms (the pi0 mixture): the
    softmax under the block mask is SHARED, the weights are NOT. That makes the expert a tower."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.ln1_pre, self.ln1_suf = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ln2_pre, self.ln2_suf = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv_pre, self.qkv_suf = nn.Linear(dim, 3 * dim), nn.Linear(dim, 3 * dim)
        self.proj_pre, self.proj_suf = nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.mlp_pre = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.mlp_suf = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.last_attn = None  # (B, S, S) head-averaged attention, for the routing viz

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, S, dim = x.shape
        h, hd = self.heads, dim // self.heads
        qkv = per_tower(per_tower(x, self.ln1_pre, self.ln1_suf), self.qkv_pre, self.qkv_suf)
        q, k, v = qkv.reshape(B, S, 3, h, hd).permute(2, 0, 3, 1, 4)          # each (B, h, S, hd)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(hd) + bias[:, None]    # + (B,1,S,S) mask
        attn = scores.softmax(dim=-1)
        self.last_attn = attn.mean(1).detach()
        x = x + per_tower((attn @ v).transpose(1, 2).reshape(B, S, dim), self.proj_pre, self.proj_suf)
        return x + per_tower(per_tower(x, self.ln2_pre, self.ln2_suf), self.mlp_pre, self.mlp_suf)


class TwoTowerVLA(nn.Module):
    """Prefix [vision, state, instruction] + suffix [H action-query tokens = noised action chunk +
    flow time]. Shared masked attention, per-tower weights; each suffix output -> one step's flow
    velocity, so the model emits a CHUNK of H velocities in one pass."""

    def __init__(self, vocab: int, feat_dim: int, dim: int, layers: int, heads: int, stats: dict) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab, dim, padding_idx=PAD_ID)
        self.vision_proj = nn.Linear(feat_dim, dim)
        self.state_proj = nn.Linear(STATE_DIM, dim)
        self.prefix_pos = nn.Parameter(0.02 * torch.randn(1, P, dim))
        self.action_in = nn.Linear(ACT_DIM, dim)                          # the noised action -> a token
        self.action_query = nn.Parameter(0.02 * torch.randn(1, H, dim))    # "which chunk step am I"
        self.time_mlp = nn.Sequential(nn.Linear(TIME_DIM, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([ExpertBlock(dim, heads) for _ in range(layers)])
        self.norm_suf = nn.LayerNorm(dim)
        self.vel_head = nn.Linear(dim, ACT_DIM)                            # per suffix token -> velocity
        for name, value in stats.items():
            self.register_buffer(name, torch.from_numpy(value))

    def prefix(self, tokens, feat, state):
        feat_n = (feat - self.feat_mean) / self.feat_std
        state_n = (2.0 * (state - self.state_min) / self.state_range - 1.0).clamp(-1.0, 1.0)
        return torch.cat([self.vision_proj(feat_n)[:, None], self.state_proj(state_n)[:, None],
                          self.tok_embed(tokens)], dim=1) + self.prefix_pos

    def forward(self, tokens, feat, state, x_t, t, cut_cross):
        B = tokens.shape[0]
        tvec = self.time_mlp(sinusoidal_embed(t * TIME_SCALE, TIME_DIM))[:, None]  # (B, 1, dim)
        suf = self.action_in(x_t) + self.action_query + tvec              # (B, H, dim)
        seq = torch.cat([self.prefix(tokens, feat, state), suf], dim=1)   # (B, P+H, dim)
        bias = block_mask(cut_cross, seq.device).expand(B, P + H, P + H).clone()
        key_pad = torch.zeros(B, P + H, dtype=torch.bool, device=seq.device)
        key_pad[:, 2:P] = tokens == PAD_ID                                # ignore padded instruction slots
        bias = bias.masked_fill(key_pad[:, None, :], float("-inf"))
        for blk in self.blocks:
            seq = blk(seq, bias)
        return self.vel_head(self.norm_suf(seq[:, P:]))                   # (B, H, ACT_DIM) velocity


# ch1.5's conditional flow matching, over an H-step CHUNK, conditioned through the mask (ch1.5/1.8).
def flow_loss(model, chunk, cmask, tokens, feat, state):
    t = torch.rand(len(chunk), generator=gen).to(device)
    noise = torch.randn(chunk.shape, generator=gen).to(device)
    x_t = (1.0 - t)[:, None, None] * noise + t[:, None, None] * chunk
    pred = model(tokens, feat, state, x_t, t, False)                     # ALWAYS trained with full routing
    per_step = ((pred - (chunk - noise)) ** 2).mean(-1)                  # (B, H) velocity MSE per step
    return (per_step * cmask).sum() / cmask.sum()                        # ignore padded chunk steps


@torch.no_grad()
def sample_chunk(model, tokens, feat, state, steps, cut_cross):
    """Integrate the velocity ODE from noise to a chunk of H actions (ch1.5), standardized space."""
    x = torch.randn((tokens.shape[0], H, ACT_DIM), generator=gen).to(device)
    for i in range(steps):
        t = torch.full((tokens.shape[0],), i / steps, device=device)
        x = x + (1.0 / steps) * model(tokens, feat, state, x, t, cut_cross)
    return x
# --- endregion ---

# --- region: train ---
torch.manual_seed(args.seed)   # policy init reproducible, independent of the frozen encoder above
gen.manual_seed(args.seed)     # fresh flow-noise stream for training
policy = TwoTowerVLA(len(VOCAB), args.feature_dim, args.model_dim, args.layers, args.heads, STATS).to(device)
optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=args.lr)
shuffle = torch.Generator().manual_seed(args.seed + 1)  # torch-side RNG for batch order
train_t = torch.from_numpy(train_idx).to(device)
train_loss, step = float("nan"), 0
for epoch in range(args.epochs):
    epoch_loss, nb = 0.0, 0
    for order in torch.randperm(len(train_idx), generator=shuffle).split(args.batch_size):
        batch = train_t[order]
        loss = flow_loss(policy, chunks_t[batch], cmask_t[batch], tokens_t[batch],
                         image_feats[batch], states_t[batch])
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
# The HEADLINE (byte-reproducible): the trained weights' HELD-OUT velocity fit, full mask vs the SAME
# weights SEVERED, on a FIXED (t, noise) pair (paired, no rendering) — severing raises the MSE toward
# the unconditional prior. The PushT ROLLOUT is the HIGHER bar and FLOORS for both masks (Wilson, ch1.6).
test_t = torch.from_numpy(test_idx).to(device)
eval_gen = torch.Generator().manual_seed(args.seed + 99)        # fixed (t, noise) for the paired MSE
t_eval = torch.rand(len(test_idx), generator=eval_gen).to(device)
noise_eval = torch.randn((len(test_idx), H, ACT_DIM), generator=eval_gen).to(device)
x_t_eval = (1.0 - t_eval)[:, None, None] * noise_eval + t_eval[:, None, None] * chunks_t[test_t]
target_eval, cmask_eval = chunks_t[test_t] - noise_eval, cmask_t[test_t]


@torch.no_grad()
def held_out_flow_mse(cut_cross: bool) -> float:
    pred = policy(tokens_t[test_t], image_feats[test_t], states_t[test_t], x_t_eval, t_eval, cut_cross)
    per_step = ((pred - target_eval) ** 2).mean(-1)
    return float(((per_step * cmask_eval).sum() / cmask_eval.sum()).item())


def wilson_ci(k: int, n: int) -> tuple[float, float]:  # 95% Wilson score interval (ch1.6)
    if n == 0:
        return (0.0, 1.0)
    p, z = k / n, Z95
    denom = 1.0 + z * z / n
    center, half = (p + z * z / (2 * n)) / denom, (z / denom) * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


eval_tok = torch.from_numpy(tokenize(TEMPLATES[0], STOI)).to(device).unsqueeze(0)


@torch.no_grad()
def rollout(cut_cross: bool, ep_seed: int, record: bool = False):
    policy.eval()
    env = PushTEnv()
    obs = env.reset(ep_seed)
    gen.manual_seed(ep_seed)   # seed the sampler from the episode: reproducible AND order-independent
    done, info, ret, traj = False, {}, 0.0, []
    while not done:
        feat = encoder(torch.from_numpy(env.render_frame(IMG_HW, IMG_HW)[None]).to(device))
        state = torch.from_numpy(obs[None]).to(device)
        chunk = sample_chunk(policy, eval_tok, feat, state, args.flow_steps, cut_cross)[0]  # (H, 2)
        for a in (chunk * act_std_t + act_mean_t).cpu().numpy():         # execute the chunk open-loop
            if done:
                break
            if record:
                px, py = env.pusher_pos
                tx, ty, tyaw = env.tee_pose
                traj.append([round(float(px), 4), round(float(py), 4), round(float(tx), 4),
                             round(float(ty), 4), round(float(tyaw), 4)])
            obs, reward, done, info = env.step(a.clip(-1.0, 1.0))
            ret += reward
    return bool(info["success"]), ret, traj


policy.eval()
mse_full, mse_cut = held_out_flow_mse(False), held_out_flow_mse(True)      # the byte-reproducible headline
# Roll out under the --break-chosen mask (full by default, severed under --break); floors either way.
outs = [rollout(EVAL_CUT, 10_000 + args.seed + ep) for ep in range(args.eval_episodes)]
succ = sum(s for s, _, _ in outs)
ci_lo, ci_hi = wilson_ci(succ, args.eval_episodes)
mean_ret = float(np.mean([r for _, r, _ in outs]))
print(f"eval[{'cut' if EVAL_CUT else 'full'}] PushT: {succ}/{args.eval_episodes} = {succ / args.eval_episodes:.2f}"
      f"  95% CI [{ci_lo:.2f}, {ci_hi:.2f}]  mean_return {mean_ret:.2f}")
print(f"HEADLINE (held-out flow MSE): full {mse_full:.4f}  cut-cross {mse_cut:.4f}  gap {mse_cut - mse_full:+.4f}"
      f" — {'routing is load-bearing' if mse_cut > mse_full else 'NO collapse — reframe!'}")
# --- endregion ---
# --- region: report ---
metrics = {
    # HEADLINE: flow_mse_gap = cut - full; > 0 == severing suffix->prefix collapses the held-out fit
    "flow_mse_full": round(mse_full, 6), "flow_mse_cut": round(mse_cut, 6),
    "flow_mse_gap": round(mse_cut - mse_full, 6),
    # the rolled-out policy (full by default, severed under --break); PushT rollout floors either way
    "reported_success_rate": round(succ / args.eval_episodes, 6), "reported_mean_return": round(mean_ret, 6),
    "reported_ci_lo": round(ci_lo, 6), "reported_ci_hi": round(ci_hi, 6), "eval_cut": bool(EVAL_CUT),
    "break_mode": args.break_mode or "none", "final_train_loss": round(train_loss, 6),
    "epochs": args.epochs, "eval_episodes": args.eval_episodes, "horizon": H, "model_dim": args.model_dim,
    "num_frames": int(N), "num_train": int(len(train_idx)), "num_test": int(len(test_idx)),
    "prefix_len": int(P), "seed": args.seed, "smoke": bool(args.smoke),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

# demo/vizdata.json: the MASK (full + cut) heatmaps + a recorded rollout under each (the toy toggle).
viz_seed = 20_000 + args.seed
f_ok, f_ret, f_traj = rollout(False, viz_seed, record=True)
c_ok, c_ret, c_traj = rollout(True, viz_seed, record=True)
suffix_row = policy.blocks[-1].last_attn[0, P:].mean(0).cpu().double().numpy()  # avg action-token attention
vizdata = {  # PushT rollout geometry is standard (ch5.3's toy renders it; frame rows = pusher_xy, tee_xyw)
    "provenance": f"vla_shape.py seed {args.seed}, {args.device}, {'smoke' if args.smoke else 'default'}; replay geometry only",
    "seed": args.seed, "prefix_len": P, "horizon": H, "world_half_extent_m": 0.45,
    "labels_seq": ["vision", "state"] + [f"tok{j}" for j in range(INSTR_TOKENS)] + [f"act{j}" for j in range(H)],
    "mask_full": (block_mask(False, device) == 0.0).int().cpu().tolist(),   # 1 = allowed to attend
    "mask_cut": (block_mask(True, device) == 0.0).int().cpu().tolist(),     # cut drops the suffix->prefix block
    "suffix_attention": [round(float(x), 6) for x in suffix_row],           # where the expert actually looks
    "target": {"x": 0.0, "y": 0.0, "yaw": 0.0},
    "tee": {"bar_half": [0.06, 0.015], "stem_half": [0.015, 0.045], "stem_offset_y": -0.06},
    "full": {"success": f_ok, "mean_return": round(f_ret, 4), "frames": f_traj},  # both rollouts floor at free-tier
    "cut": {"success": c_ok, "mean_return": round(c_ret, 4), "frames": c_traj},
    "meta": {k: metrics[k] for k in ("flow_mse_full", "flow_mse_cut", "flow_mse_gap", "num_frames")},
}
demo_dir = Path(__file__).resolve().parent / "demo"
demo_dir.mkdir(exist_ok=True)
(demo_dir / "vizdata.json").write_text(json.dumps(vizdata, indent=2) + "\n")
if args.rerun:
    rr.log("mask/full", rr.Image((np.array(vizdata["mask_full"]) * 255).astype(np.uint8)), static=True)
    rr.log("mask/cut", rr.Image((np.array(vizdata["mask_cut"]) * 255).astype(np.uint8)), static=True)
    rr.log("routing/suffix_attention", rr.BarChart(np.asarray(suffix_row)))
print(f"metrics: {args.out / 'metrics.json'}  |  vizdata: {demo_dir / 'vizdata.json'}")
print(f"two-tower shape: prefix {P} (vision+state+{INSTR_TOKENS} tok) | suffix {H} (action expert) — "
      f"cut suffix->prefix and the expert goes blind")
if args.rerun:
    print(f"recording: {args.out / 'vla_shape.rrd'} — open it with: rerun {args.out / 'vla_shape.rrd'}")
# --- endregion ---
