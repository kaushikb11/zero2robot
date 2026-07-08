"""zero2robot 3.8 — Reading the Frontier: probe a released checkpoint's insides.

By Phase 3 you have BUILT every piece a frontier robot policy is made of: the VLM
fusion that turns an instruction + an image feature + a state into one vector (1.7 /
1.8), from-scratch multi-head attention (1.8), and a flow-matching action head (1.5).
A released frontier VLA — Physical Intelligence's pi0 (openpi), NVIDIA's GR00T N1 —
is those same pieces, scaled and pretrained. So the last skill of the phase is not
building; it is READING: opening a checkpoint someone else trained and working out
what is inside it. This file teaches the four mechanical moves "reading a checkpoint"
always comes down to, and they are identical whether the file on disk is 20 KB or 14 GB:

  (1) LOAD    — pour a saved state_dict into an architecture skeleton you hold.
  (2) INSPECT — enumerate the modules, their shapes, and their parameter counts.
  (3) HOOK    — run a forward pass and CAPTURE a hidden layer's activations; you
                cannot edit a released model's forward(), so you register a hook.
  (4) PROBE   — fit a LINEAR map from those activations to a KNOWN factor and read
                its score: what has this layer actually learned to represent?

We run these on an ACCESSIBLE checkpoint: a tiny language-conditioned policy this file
trains once, deterministically, on a synthetic routing task (an instruction token
selects WHICH state coordinate drives the action — a stand-in for "which skill"), then
RELOADS it from disk and probes. We probe our own tiny checkpoint, not pi0, for one
honest reason: a real pi0 / GR00T checkpoint is multi-GB and not offline-probeable on
the free tier. The MECHANICS transfer 1:1; the prose does the guided READING of the
real frontier (openpi / pi0, GR00T's dual system) as a STUDY-tier segment.

The finding, measured, and the honest twist: the fused layer recovers the task ID
almost perfectly (~1.0) — but so does a RANDOM-init checkpoint of the same shape. A
probe that recovers an INPUT (the instruction token is right there in the sequence)
proves little; a linear read of almost any projection can do it. The probe that MEANS
something recovers a COMPUTED quantity — the value of the state coordinate the task
ROUTES to — and there training shows up loudly: R^2 ~0.90 trained vs ~0.16 random.
That gap is the readout, and "did the probe just recover an input?" is exactly the
question you must ask before believing a probe of pi0's fused token.

Run it:    python curriculum/phase3_advanced/ch3.8_frontier/probe.py --seed 0
CI smoke:  python curriculum/phase3_advanced/ch3.8_frontier/probe.py --smoke --seed 0 --no-rerun
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

# Chapter artifacts run as loose scripts from the repo root; put the root on sys.path
# so `curriculum.common` resolves (same pattern as every other chapter).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

NUM_TASKS = 4       # how many "skills" the instruction can select
STATE_DIM = 6       # the proprioceptive-ish state vector
ACT_DIM = 2         # the action the policy regresses
MAX_TOK = 4         # fixed instruction length: [BOS, task, EOS, PAD]
BOS, EOS, PAD = 1, 2, 0
TASK_BASE = 3       # task t uses token id TASK_BASE + t; vocab is TASK_BASE + NUM_TASKS
VOCAB = TASK_BASE + NUM_TASKS

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch3.8-frontier"))
parser.add_argument("--model_dim", type=int, default=32)   # fused width. T4: 64 | smoke: 8
parser.add_argument("--heads", type=int, default=4)        # attention heads (model_dim % heads == 0)
parser.add_argument("--hidden", type=int, default=64)      # action-head MLP width. smoke: 16
parser.add_argument("--train_examples", type=int, default=2048)  # synthetic pile size. smoke: 64
parser.add_argument("--steps", type=int, default=400)      # Adam steps to train the checkpoint. smoke: 20
parser.add_argument("--probe_examples", type=int, default=512,   # held-out rows the probe fits on. smoke: 64
                    help="fresh examples for the probe (never seen in checkpoint training)")
parser.add_argument("--probe_ridge", type=float, default=1.0,
                    help="ridge lambda for the closed-form linear probe (keeps the normal equations stable)")
parser.add_argument("--lr", type=float, default=1e-2)
parser.add_argument("--seed", type=int, default=0, help="seeds the synthetic data, the init, training, and the probe split")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)
torch.set_num_threads(1)  # single-threaded => bitwise-reproducible CPU reductions (the probe R^2 reads out
#                           the trained weights to 6 decimals, so multi-thread reduction jitter would break
#                           the twice-run byte-compare — the fused features are that sensitive to the last bits)
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.model_dim, args.heads, args.hidden = 8, 2, 16
    args.train_examples, args.steps, args.probe_examples, args.device = 64, 20, 64, "cpu"
banner("ch3.8-frontier", device="cpu")  # pure-CPU chapter: honest cpu tier, never the host's mps/cuda (the reference metrics + the only measured wall-clock are the cpu tier)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
CONFIG = {"model_dim": args.model_dim, "heads": args.heads, "hidden": args.hidden}  # identifies the checkpoint shape
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch3.8-frontier", spawn=False)
    rr.save(str(args.out / "probe.rrd"))
# --- endregion ---

# --- region: checkpoint ---
# The checkpoint we will READ. It is a tiny language-conditioned policy: a fused
# transformer token (a CLS that reads instruction + state, exactly the 1.8 fusion) into
# a small action head. This is the ONE model definition in the file; everything after
# treats it as an opaque released checkpoint loaded from disk.
class TinyPolicy(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.heads = heads
        self.tok_embed = nn.Embedding(VOCAB, dim, padding_idx=PAD)
        self.state_proj = nn.Linear(STATE_DIM, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(0.02 * torch.randn(1, 2 + MAX_TOK, dim))
        self.ln = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)          # the FUSED layer we will probe
        self.head = nn.Sequential(nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, ACT_DIM))
        self.last_attn = None                  # CLS attention over the sequence, kept for inspection

    def forward(self, tokens: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        B = tokens.shape[0]
        seq = torch.cat([self.cls.expand(B, -1, -1),
                         self.state_proj(state)[:, None],
                         self.tok_embed(tokens)], dim=1) + self.pos  # [CLS, state, tok_0..3]
        h, hd = self.heads, seq.shape[-1] // self.heads
        qkv = self.qkv(self.ln(seq)).reshape(B, 2 + MAX_TOK, 3, h, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = ((q @ k.transpose(-2, -1)) / math.sqrt(hd)).softmax(dim=-1)
        self.last_attn = attn[:, :, 0, :].mean(1).detach()          # (B, L): CLS -> each input token
        seq = seq + self.proj((attn @ v).transpose(1, 2).reshape(B, 2 + MAX_TOK, seq.shape[-1]))
        return self.head(self.norm(seq[:, 0]))                      # fused CLS -> action


def make_batch(gen: torch.Generator, n: int):
    """Synthetic routing task: task t is drawn uniformly; the instruction encodes t; the
    action is a FIXED linear function of the state SELECTED by t. To predict it the fused
    token must encode which task is active (that is what the probe will later recover)."""
    task = torch.randint(0, NUM_TASKS, (n,), generator=gen)
    state = torch.randn(n, STATE_DIM, generator=gen)
    tokens = torch.full((n, MAX_TOK), PAD, dtype=torch.long)
    tokens[:, 0], tokens[:, 1], tokens[:, 2] = BOS, TASK_BASE + task, EOS
    routed = state[torch.arange(n), task]                           # the task-selected coordinate
    other = state[torch.arange(n), (task + 3) % STATE_DIM]
    action = torch.stack([routed, -other], dim=1)                  # deterministic rule; no noise
    return tokens, state, action, task, routed


ckpt_path = args.out / f"tiny_policy_{args.device}.pt"   # device in the filename: a cpu cache and a gpu cache never collide. GPU/CPU kernels differ in the last bits, so silently reloading GPU-trained weights for a --device cpu run would break this chapter's byte-identical reload claim (the probe R^2 reads those bits to 6 decimals)
data_gen = torch.Generator().manual_seed(args.seed)                 # one stream for the training pile
tokens, state, action, _, _ = make_batch(data_gen, args.train_examples)


def train_and_save() -> None:
    """Stand-in for "someone trained and released a checkpoint": train the tiny policy,
    then serialize its state_dict + the config that identifies its shape."""
    torch.manual_seed(args.seed)                                   # reproducible init
    model = TinyPolicy(args.model_dim, args.heads, args.hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    tok_d, st_d, act_d = tokens.to(device), state.to(device), action.to(device)
    for step in range(args.steps):
        loss = ((model(tok_d, st_d) - act_d) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if args.rerun:
            rr.set_time("step", sequence=step)
            rr.log("checkpoint/train_mse", rr.Scalars([loss.item()]))
        if step % 100 == 0 or step == args.steps - 1:
            print(f"train step {step:4d}  mse {loss.item():.5f}")
    torch.save({"state_dict": model.state_dict(), "config": CONFIG}, ckpt_path)


# Rebuild the checkpoint only when it is absent or its shape no longer matches this
# config (so a smoke run never loads a full run's weights). Then always RE-LOAD from
# disk — move (1), the one you make with any released checkpoint you did not train.
if not ckpt_path.is_file() or torch.load(ckpt_path, weights_only=False)["config"] != CONFIG:
    train_and_save()
blob = torch.load(ckpt_path, weights_only=False)
policy = TinyPolicy(args.model_dim, args.heads, args.hidden).to(device)   # a fresh, weightless skeleton
policy.load_state_dict(blob["state_dict"], strict=True)                   # strict=True: every key must match
policy.eval()
print(f"loaded checkpoint {ckpt_path} ({sum(p.numel() for p in policy.parameters()):,} params); "
      f"config {blob['config']}")
# --- endregion ---

# --- region: inspect ---
# Move (2): before you run a released model you read its shape. named_parameters() is the
# table of contents — every weight, where it lives, how big. (For a real VLA this is how
# you spot the vision tower vs the LM vs the action expert, and where the parameters sit.)
total_params = sum(p.numel() for p in policy.parameters())
module_params = {name: sum(p.numel() for p in mod.parameters())
                 for name, mod in policy.named_children()}
print("\narchitecture (module: params):")
for name, count in module_params.items():
    shapes = ", ".join(f"{n.split('.')[-1]}{tuple(p.shape)}" for n, p in policy.named_parameters()
                       if n.startswith(name + "."))
    print(f"  {name:12s} {count:7,d}  [{shapes}]")
print(f"  {'TOTAL':12s} {total_params:7,d}")
# --- endregion ---

# --- region: forward ---
# Move (3): capture a hidden layer's activations on a FRESH batch. You cannot edit a
# released forward(), so you HOOK the layer — here `norm`, whose output is the fused CLS
# vector every action is read from. We also keep the CLS attention the block recorded.
probe_gen = torch.Generator().manual_seed(args.seed + 100)          # held-out from the training pile
p_tokens, p_state, _, p_task, p_routed = make_batch(probe_gen, args.probe_examples)

captured: dict[str, torch.Tensor] = {}
handle = policy.norm.register_forward_hook(lambda _m, _i, out: captured.__setitem__("fused", out.detach()))
with torch.no_grad():
    policy(p_tokens.to(device), p_state.to(device))
handle.remove()
features = captured["fused"].cpu().numpy()                          # (probe_examples, model_dim)
cls_attention = policy.last_attn[0].cpu().numpy()                   # how CLS attends to [CLS, state, tok_0..3]
print(f"\nhooked policy.norm -> fused features {features.shape}; "
      f"CLS attends most to input index {int(cls_attention.argmax())} "
      f"(0=CLS, 1=state, 2..5=instruction tokens)")
# --- endregion ---

# --- region: probe ---
# Move (4): a LINEAR probe. Fit a linear map from the frozen features to a KNOWN factor
# on a train split, score it on a held-out split. A HIGH score means the layer already
# represents that factor linearly — the model "knows" it. We ask two questions: does the
# fused token encode WHICH TASK is active (classification accuracy), and does it encode
# the ROUTED coordinate's value (regression R^2)? A RANDOM-init checkpoint of the same
# shape is the control: any score above it is what TRAINING put there.
def linear_probe(feats: np.ndarray, target: np.ndarray, ridge: float, classify: bool) -> float:
    n, cut = len(feats), len(feats) // 2                            # deterministic 50/50 split
    x = np.concatenate([feats, np.ones((n, 1), np.float64)], axis=1)  # append a bias column
    xtr, xte = x[:cut], x[cut:]
    ytr_raw, yte_raw = target[:cut], target[cut:]
    y = np.eye(NUM_TASKS)[ytr_raw] if classify else ytr_raw.reshape(-1, 1).astype(np.float64)
    w = np.linalg.solve(xtr.T @ xtr + ridge * np.eye(x.shape[1]), xtr.T @ y)  # ridge normal equations
    pred = xte @ w
    if classify:
        return float((pred.argmax(1) == yte_raw).mean())           # held-out accuracy
    resid = ((yte_raw.reshape(-1, 1) - pred) ** 2).sum()
    total = ((yte_raw - yte_raw.mean()) ** 2).sum()
    return float(1.0 - resid / total) if total > 0 else 0.0        # held-out R^2


# The random-init control: same architecture, never trained (seed offset so it is a
# genuinely different init, the 1.5/1.8 baseline pattern).
torch.manual_seed(args.seed + 7)
control = TinyPolicy(args.model_dim, args.heads, args.hidden).to(device)
control.eval()
c_captured: dict[str, torch.Tensor] = {}
c_handle = control.norm.register_forward_hook(lambda _m, _i, out: c_captured.__setitem__("fused", out.detach()))
with torch.no_grad():
    control(p_tokens.to(device), p_state.to(device))
c_handle.remove()
control_features = c_captured["fused"].cpu().numpy()

task_np, routed_np = p_task.numpy(), p_routed.numpy().astype(np.float64)
trained_task_acc = linear_probe(features, task_np, args.probe_ridge, classify=True)
trained_coord_r2 = linear_probe(features, routed_np, args.probe_ridge, classify=False)
control_task_acc = linear_probe(control_features, task_np, args.probe_ridge, classify=True)
control_coord_r2 = linear_probe(control_features, routed_np, args.probe_ridge, classify=False)
chance_task_acc = 1.0 / NUM_TASKS
print("\nlinear probe of the fused layer (held-out split):")
print(f"  task-id accuracy   trained {trained_task_acc:.3f}  vs random {control_task_acc:.3f}  "
      f"(chance {chance_task_acc:.3f}) <- trivially decodable from EITHER: the token is an input")
print(f"  routed-coord R^2   trained {trained_coord_r2:.3f}  vs random {control_coord_r2:.3f}  "
      f"<- the REAL readout: only training makes the fused token encode the routed VALUE")
if args.rerun:
    rr.log("probe/task_id_accuracy", rr.BarChart(np.array([trained_task_acc, control_task_acc, chance_task_acc])))
    rr.log("probe/routed_coord_r2", rr.BarChart(np.array([trained_coord_r2, control_coord_r2])))
    rr.log("forward/cls_attention", rr.BarChart(cls_attention.astype(np.float64)))
# --- endregion ---

# --- region: report ---
metrics = {
    "chance_task_acc": round(chance_task_acc, 6),
    "cls_attends_to_index": int(cls_attention.argmax()),
    "control_coord_r2": round(control_coord_r2, 6),
    "control_task_acc": round(control_task_acc, 6),
    "model_dim": args.model_dim,
    "probe_examples": args.probe_examples,
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "total_params": int(total_params),
    "trained_coord_r2": round(trained_coord_r2, 6),
    "trained_task_acc": round(trained_task_acc, 6),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"\nRead-out: task-id is decodable from EITHER checkpoint ({trained_task_acc:.2f} vs "
      f"{control_task_acc:.2f}) — that just recovers an input. The routed-coordinate R^2 is the "
      f"real signal: {trained_coord_r2:.2f} trained vs {control_coord_r2:.2f} random. That is the "
      f"same probe — and the same caveat — you would apply to pi0's fused token. Read the prose.")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'probe.rrd'} — open it with: rerun {args.out / 'probe.rrd'}")
# --- endregion ---
