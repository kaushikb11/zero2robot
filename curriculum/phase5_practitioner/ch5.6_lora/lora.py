"""zero2robot 5.6 — LoRA From Scratch: Adapt a Frozen Policy.

Real robots ship as FROZEN checkpoints (pi0, GR00T, SmolVLA) — gigabytes of pretrained
weights you did not train and do not want to overwrite. To teach one a NEW skill you have
two honest options: fine-tune ALL its weights (a full copy per skill), or bolt on a tiny
trainable ADAPTER and leave the original weights untouched. This chapter builds the second
option from scratch — LoRA, Low-Rank Adaptation — the single most-used fine-tuning trick in
the modern stack, in ~40 lines, and MEASURES what it actually buys.

The subject is a compact, state-based conditioned policy (the ch3.8 routing task: an
instruction token selects WHICH state coordinates drive the action). We PRETRAIN it on three
skills, HOLD ONE OUT, torch.save + reload + FREEZE it, then teach it the held-out skill three
ways in one run:

  (1) FROZEN   — no adaptation. Zero-shot on the unseen skill: it cannot do it.
  (2) FULL-FT  — unfreeze every weight and train (the ceiling, and the cost).
  (3) LoRA     — freeze W, add a thin trainable rank-r bypass to the action head's output
                 projection:  y = W x  +  (alpha/r) * B(A x),  with A[d->r], B[r->d] and B
                 ZERO-INITIALIZED, so at step 0 the adapted policy is BITWISE the frozen one.
                 Only A and B train — around one percent of the parameters.

THE HEADLINE (state-based => bitwise-reproducible on CPU, seed-robust): sweep the rank r and
the held-out fit RISES then PLATEAUS at full fine-tuning's ceiling — the visible ELBOW. A
rank-4 LoRA training ~1% of the weights recovers MOST of full-FT's held-out fit. The
misconception this kills: "fewer trainable parameters must mean a worse fit." The elbow says
no.

THE HONEST TWIST, measured not asserted (and it refutes a second, tempting intuition): freezing
W does NOT automatically protect the old skill. We watch an in-distribution skill (TASK_A)
while we adapt, and its fit COLLAPSES under LoRA just as it does under full-FT — often more.
A single low-rank LINEAR adapter is added to EVERY input, and it cannot gate itself off for the
old skill, so the new skill's correction bleeds onto the old one. "Frozen weights" is not
"frozen behavior." LoRA's real win here is parameter efficiency, not free memory.

Break it (--break rand_init_B): initialize B the way you would init ANY other Linear (kaiming)
instead of zeroing it — the single most natural mistake. Now B(A x) != 0 at step 0, so the
adapter is NOT a no-op: the frozen policy is already perturbed before a single gradient step,
and the "adapted == frozen at step 0" invariant is broken (step0_frozen_gap goes from EXACTLY
0.0 to clearly nonzero, seed-robust). Adaptation still recovers the held-out fit here — the
lesson is not that the outcome collapses but that you no longer BEGIN from the model you paid
to pretrain. Zero-init B is what makes LoRA a no-op at step 0; that is the whole reason to zero it.

Run it:      python curriculum/phase5_practitioner/ch5.6_lora/lora.py --seed 0
Break it:    python curriculum/phase5_practitioner/ch5.6_lora/lora.py --seed 0 --break rand_init_B
CI smoke:    python curriculum/phase5_practitioner/ch5.6_lora/lora.py --smoke --seed 0 --no-rerun
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

# Chapter artifacts run as loose scripts from the repo root; put the root on sys.path
# so `curriculum.common` resolves (same pattern as every other chapter).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

NUM_SKILLS = 4       # how many "skills" the instruction token can select
STATE_DIM = 8        # the proprioceptive-ish state vector
ACT_DIM = 6          # each action dim reads a distinct skill-selected coordinate (a higher-rank rule
#                      than a single coordinate — that spread is what makes the rank ELBOW visible)
HELD_OUT = NUM_SKILLS - 1   # skill 3 is held OUT of pretraining; the adapter must teach it
TASK_A = 0                  # an in-distribution skill; we watch whether adaptation FORGETS it
HEADLINE_RANK = 4           # the rank the frozen/full/lora named arms compare at
SWEEP_RANKS = [0, 1, 2, 4, 8, 16]   # r=0 is the frozen base (no adapter); "full" fine-tune is appended

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch5.6-lora"))
parser.add_argument("--dim", type=int, default=32)         # instruction-token embed width. smoke: 8
parser.add_argument("--hidden", type=int, default=256)     # action-head MLP width (base ~78K params; wide
#                      enough that a rank-r head adapter is ~r/hidden of the weights -> rank-4 ~1.3%). smoke: 32
parser.add_argument("--alpha", type=float, default=2.0,    # LoRA scale; the update is (alpha/r) * B(A x)
                    help="LoRA scaling numerator; scaling = alpha/r (peft's default relation)")
parser.add_argument("--train_examples", type=int, default=2048)  # pretrain pile (skills 0..2). smoke: 64
parser.add_argument("--adapt_examples", type=int, default=1024)  # held-out adaptation pile (skill 3). smoke: 64
parser.add_argument("--eval_examples", type=int, default=1024)   # fresh held-out + task-A eval piles. smoke: 64
parser.add_argument("--pretrain_steps", type=int, default=800)   # Adam steps to pretrain the base. smoke: 20
parser.add_argument("--adapt_steps", type=int, default=400)      # Adam steps per adaptation arm. smoke: 20
parser.add_argument("--lr", type=float, default=3e-3)
parser.add_argument("--seed", type=int, default=0, help="seeds the synthetic data, every init, and each arm's training")
parser.add_argument("--break", dest="break_mode", choices=("rand_init_B",), default=None,
                    help="rand_init_B = init the LoRA B matrix like any other Linear (kaiming) instead of "
                         "zeroing it; the adapter is no longer a no-op at step 0 (adapted != frozen), so you "
                         "begin adaptation from a perturbed copy of the pretrained model (measured: gap 0 -> >0)")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)
torch.set_num_threads(1)  # single-threaded => bitwise-reproducible CPU reductions. The held-out R^2 reads out
#                           the trained weights to 6 decimals, so multi-thread reduction jitter would break the
#                           twice-run byte-compare. This is a STATE-based task: no rendering, so CPU IS bitwise.
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.dim, args.hidden = 8, 32
    args.train_examples, args.adapt_examples, args.eval_examples = 64, 64, 64
    args.pretrain_steps, args.adapt_steps, args.device = 20, 20, "cpu"
RAND_B = args.break_mode == "rand_init_B"
banner("ch5.6-lora", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch5.6-lora", spawn=False)
    rr.save(str(args.out / "lora.rrd"))
# --- endregion ---

# --- region: model ---
# The base policy we will FREEZE and adapt. It is a compact, state-based conditioned MLP
# (the ch3.8 routing task, no attention needed here): an instruction token picks a "skill",
# we concatenate its embedding with the raw state, and a small MLP maps that to an action.
# `head` — the action head's OUTPUT projection — is what LoRA will wrap. The trunk (fc1, fc2)
# stays frozen too; the adapter must reconstruct the held-out skill from the features the
# trunk already computes, which is exactly why rank matters.
class TinyPolicy(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(NUM_SKILLS, dim)   # skill id -> a conditioning vector
        self.fc1 = nn.Linear(dim + STATE_DIM, hidden)    # action-head trunk, layer 1
        self.fc2 = nn.Linear(hidden, hidden)             # action-head trunk, layer 2
        self.head = nn.Linear(hidden, ACT_DIM)           # action-head readout  (LoRA-wrapped)

    def forward(self, skill: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([self.tok_embed(skill), state], dim=-1)  # fuse instruction + state
        h = F.silu(self.fc1(cond))
        h = F.silu(self.fc2(h))
        return self.head(h)


def make_batch(gen: torch.Generator, n: int, skills: torch.Tensor):
    """Synthetic routing task (ch3.8, widened): each example draws a skill from `skills`; each
    of the ACT_DIM action dims reads a DISTINCT coordinate the skill selects (coordinate
    (skill + j) mod STATE_DIM, with an alternating sign). No noise — the rule is deterministic,
    so a good fit is a real R^2 near 1. Pretraining on skills {0,1,2} never touches skill 3's
    coordinates, so the held-out skill is genuinely unseen (poor zero-shot)."""
    idx = torch.randint(0, len(skills), (n,), generator=gen)
    skill = skills[idx]
    state = torch.randn(n, STATE_DIM, generator=gen)
    rows = torch.arange(n)
    cols = (skill[:, None] + torch.arange(ACT_DIM)) % STATE_DIM   # (n, ACT_DIM) selected coordinates
    signs = torch.where(torch.arange(ACT_DIM) % 2 == 0, 1.0, -1.0)
    action = state[rows[:, None], cols] * signs                  # the deterministic action rule
    return skill.to(device), state.to(device), action.to(device)
# --- endregion ---

# --- region: lora ---
# LoRA, from scratch. Wrap a FROZEN nn.Linear with a thin trainable low-rank bypass:
#     y = W x  +  (alpha / r) * B (A x),   A: (r, in)   B: (out, r)
# W (and its bias) never train — only A and B do, r*(in+out) numbers instead of in*out.
# The init is the whole trick (this mirrors peft's reset_lora_parameters): A gets the usual
# nn.Linear kaiming init, B is ZEROED — so B(A x) = 0 at step 0 and the wrapped layer is
# BITWISE the frozen one. Training then grows the bypass FROM the pretrained model, not from
# noise. --break rand_init_B zeroes nothing: B starts random and the frozen prior is already
# corrupted before the first gradient step (the deliberate failure the exercise measures).
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, rand_init_B: bool = False) -> None:
        super().__init__()
        self.base = base                       # the frozen pretrained projection
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.r = r
        self.scaling = alpha / r if r > 0 else 0.0
        if r > 0:
            self.A = nn.Parameter(torch.empty(r, base.in_features))
            self.B = nn.Parameter(torch.empty(base.out_features, r))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # A: the default nn.Linear init
            if rand_init_B:                                     # BREAK: the natural mistake — init B like any
                nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))  # other Linear instead of zeroing it. Now B(A x)
            else:                                              # != 0 at step 0: the adapter is NOT a no-op and the
                nn.init.zeros_(self.B)                         # frozen policy is perturbed before any training.
            #                                                    zero-init B: B(A x) = 0 => adapted == frozen at step 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.r > 0:                                          # W x + (alpha/r) * B(A x)
            y = y + self.scaling * F.linear(F.linear(x, self.A), self.B)
        return y


def build_arm(state_dict: dict, arm: str, rank: int) -> nn.Module:
    """Load the frozen pretrained weights into a fresh policy, then configure ONE arm:
      * "frozen"/"lora" with rank 0 -> nothing trains (the zero-shot base),
      * "full"                      -> every weight trains (full fine-tuning),
      * "lora"  with rank > 0       -> W frozen, only the head's A/B adapter trains."""
    model = TinyPolicy(args.dim, args.hidden).to(device)
    model.load_state_dict(state_dict)
    if arm == "full":
        return model                                           # all params keep requires_grad=True
    for p in model.parameters():                               # freeze the whole base
        p.requires_grad_(False)
    if arm == "lora" and rank > 0:                             # wrap the action head's output projection
        model.head = LoRALinear(model.head, rank, args.alpha, RAND_B)
    return model.to(device)                                    # the fresh A/B adapter follows the base's device
# --- endregion ---

# --- region: pretrain ---
# Pretrain the base on skills {0,1,2}, HOLDING OUT skill 3. Skill 3's token embedding never
# gets a gradient and the trunk never sees its coordinates, so the reloaded base cannot do it
# zero-shot — that gap is what the three arms then try to close.
def r2(pred: np.ndarray, target: np.ndarray) -> float:
    resid = float(((target - pred) ** 2).sum())
    total = float(((target - target.mean(0)) ** 2).sum())
    return 1.0 - resid / total if total > 0 else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, skill: torch.Tensor, state: torch.Tensor, action: torch.Tensor):
    model.eval()
    pred = model(skill, state).cpu().double().numpy()
    tgt = action.cpu().double().numpy()
    return r2(pred, tgt), float(((tgt - pred) ** 2).mean())


def fit(model: nn.Module, skill, state, action, steps: int, tag: str) -> None:
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:  # r=0 / frozen: nothing to train
        return
    opt = torch.optim.Adam(params, lr=args.lr)
    for step in range(steps):
        loss = ((model(skill, state) - action) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if args.rerun:
            rr.set_time("step", sequence=step)
            rr.log(f"train/{tag}/mse", rr.Scalars([loss.item()]))


pretrain_gen = torch.Generator().manual_seed(args.seed)          # the pretraining pile (skills 0..2)
train_skills = torch.tensor([s for s in range(NUM_SKILLS) if s != HELD_OUT])
pt_skill, pt_state, pt_action = make_batch(pretrain_gen, args.train_examples, train_skills)

ckpt_path = args.out / "base_policy.pt"
torch.manual_seed(args.seed)                                    # reproducible init
base = TinyPolicy(args.dim, args.hidden).to(device)
fit(base, pt_skill, pt_state, pt_action, args.pretrain_steps, "pretrain")
torch.save(base.state_dict(), ckpt_path)                        # "ship" the checkpoint...
frozen_state = torch.load(ckpt_path, weights_only=True)         # ...and reload it, as any released policy
base_total = sum(p.numel() for p in base.parameters())
print(f"pretrained + reloaded base ({base_total:,} params); held out skill {HELD_OUT}")

# The two held-out eval piles, fixed across every arm: the UNSEEN skill (the target of
# adaptation) and an in-distribution skill TASK_A (the retention / forgetting probe).
eval_gen = torch.Generator().manual_seed(args.seed + 100)
ho_skill, ho_state, ho_action = make_batch(eval_gen, args.eval_examples, torch.tensor([HELD_OUT]))
ta_skill, ta_state, ta_action = make_batch(eval_gen, args.eval_examples, torch.tensor([TASK_A]))
# The adaptation pile: skill 3 ONLY (what a practitioner has for the new skill).
adapt_gen = torch.Generator().manual_seed(args.seed + 1)
ad_skill, ad_state, ad_action = make_batch(adapt_gen, args.adapt_examples, torch.tensor([HELD_OUT]))
# --- endregion ---

# --- region: sweep ---
# One rank dial, three regimes. For each rank we rebuild the arm from the SAME frozen weights,
# adapt on skill-3 data, and read two fits: the held-out skill (did we LEARN it?) and TASK_A
# (did we FORGET an old one?). r=0 is the frozen zero-shot; "full" is full fine-tuning. The
# elbow lives in held_out_r2 vs rank; the (honest) forgetting lives in task_a_r2.
def trainable_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_arm(arm: str, rank: int) -> dict:
    torch.manual_seed(args.seed + 7)                            # same adapter/full init draw across ranks
    model = build_arm(frozen_state, arm, rank)
    trainable = trainable_count(model)
    # Step 0, BEFORE any update: for zero-init-B LoRA this is bitwise the frozen policy; the
    # break (and full-FT) moves it. We record how far step-0 already is from the frozen output.
    with torch.no_grad():
        step0 = model(ho_skill, ho_state).cpu().double().numpy()
        frozen_out = base(ho_skill, ho_state).cpu().double().numpy()
    step0_gap = float(np.abs(step0 - frozen_out).max())
    step0_r2 = r2(step0, ho_action.cpu().double().numpy())
    fit(model, ad_skill, ad_state, ad_action, args.adapt_steps, f"{arm}{rank}")
    ho_r2, ho_mse = evaluate(model, ho_skill, ho_state, ho_action)
    ta_r2, ta_mse = evaluate(model, ta_skill, ta_state, ta_action)
    return {"arm": arm, "rank": rank, "trainable": trainable,
            "trainable_pct": round(100.0 * trainable / base_total, 4),
            "held_out_r2": round(ho_r2, 6), "held_out_mse": round(ho_mse, 6),
            "task_a_r2": round(ta_r2, 6), "task_a_mse": round(ta_mse, 6),
            "step0_heldout_r2": round(step0_r2, 6), "step0_frozen_gap": round(step0_gap, 6)}


sweep = [run_arm("lora", r) for r in SWEEP_RANKS] + [run_arm("full", args.hidden)]
frozen = next(r for r in sweep if r["arm"] == "lora" and r["rank"] == 0)
lora = next(r for r in sweep if r["arm"] == "lora" and r["rank"] == HEADLINE_RANK)
full = next(r for r in sweep if r["arm"] == "full")
print("\nrank dial (adapt the held-out skill; held_out fit rises then plateaus; watch task_A):")
print(f"  {'arm':6s} {'rank':>5s} {'trainable':>10s} {'%':>7s} {'held_out R2':>12s} {'task_A R2':>11s}")
for row in sweep:
    name = "frozen" if (row["arm"] == "lora" and row["rank"] == 0) else row["arm"]
    rk = "full" if row["arm"] == "full" else str(row["rank"])
    print(f"  {name:6s} {rk:>5s} {row['trainable']:>10,d} {row['trainable_pct']:>6.2f}% "
          f"{row['held_out_r2']:>12.3f} {row['task_a_r2']:>11.3f}")
# --- endregion ---

# --- region: report ---
# The headline, stated as the reproducible DIRECTION (mechanism, not a third decimal):
#  * held_out fit: full-FT is the ceiling; a small-rank LoRA recovers MOST of it (the elbow),
#    training ~1% of the weights — "fewer trainable params" does NOT mean "worse fit".
#  * forgetting (the honest twist): freezing W does NOT protect TASK_A. Its fit collapses under
#    LoRA as it does under full-FT, because the additive low-rank adapter fires on EVERY input
#    and cannot gate itself off for the old skill. LoRA buys parameter efficiency, not memory.
recovered = (lora["held_out_r2"] - frozen["held_out_r2"]) / max(
    1e-9, full["held_out_r2"] - frozen["held_out_r2"])   # fraction of full-FT's held-out gain LoRA recovers
metrics = {
    "alpha": args.alpha,
    "base_total_params": int(base_total),
    "break_mode": args.break_mode or "none",
    "frozen_heldout_r2": frozen["held_out_r2"],      # zero-shot on the unseen skill: poor
    "frozen_task_a_r2": frozen["task_a_r2"],         # the pretrained retention baseline: high
    "full_heldout_r2": full["held_out_r2"],          # full-FT ceiling on the held-out skill
    "full_task_a_r2": full["task_a_r2"],
    "full_task_a_forget": round(frozen["task_a_r2"] - full["task_a_r2"], 6),   # >0: full-FT forgot TASK_A
    "full_trainable_pct": full["trainable_pct"],
    "headline_rank": HEADLINE_RANK,
    "held_out_skill": HELD_OUT,
    "hidden": args.hidden,
    "lora_heldout_r2": lora["held_out_r2"],          # small-rank LoRA held-out fit
    "lora_recovered_frac": round(recovered, 6),      # fraction of full-FT's held-out gain LoRA recovers (headline)
    "lora_step0_frozen_gap": lora["step0_frozen_gap"],  # 0.0 clean (adapted==frozen@step0); >0 under --break
    "lora_step0_heldout_r2": lora["step0_heldout_r2"],  # == frozen_heldout_r2 clean; worse under --break
    "lora_task_a_r2": lora["task_a_r2"],
    "lora_task_a_forget": round(frozen["task_a_r2"] - lora["task_a_r2"], 6),   # >0: LoRA ALSO forgot TASK_A
    "lora_trainable_pct": lora["trainable_pct"],
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "sweep_heldout_r2": [row["held_out_r2"] for row in sweep],
    "sweep_ranks": [("full" if row["arm"] == "full" else row["rank"]) for row in sweep],
    "sweep_task_a_r2": [row["task_a_r2"] for row in sweep],
    "sweep_trainable_pct": [row["trainable_pct"] for row in sweep],
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

# demo/vizdata.json: the rank dial. Two synced readouts per rank — params-trained %
# (climbing) and held-out fit (rising then PLATEAUING, the elbow) — plus the TASK_A trace
# (collapsing under adaptation) and the full-FT reference line.
viz_rows = [r for r in sweep if r["arm"] == "lora"]  # the LoRA rank dial (r=0..16)
vizdata = {
    "held_out_skill": HELD_OUT, "task_a_skill": TASK_A, "headline_rank": HEADLINE_RANK,
    "base_total_params": int(base_total), "break_mode": args.break_mode or "none",
    "ranks": [r["rank"] for r in viz_rows],
    "trainable_pct": [r["trainable_pct"] for r in viz_rows],
    "held_out_r2": [r["held_out_r2"] for r in viz_rows],
    "task_a_r2": [r["task_a_r2"] for r in viz_rows],
    "full": {"held_out_r2": full["held_out_r2"], "task_a_r2": full["task_a_r2"],
             "trainable_pct": full["trainable_pct"]},
    "frozen": {"held_out_r2": frozen["held_out_r2"], "task_a_r2": frozen["task_a_r2"]},
}
(args.out / "demo").mkdir(parents=True, exist_ok=True)
(args.out / "demo" / "vizdata.json").write_text(json.dumps(vizdata) + "\n")

if args.rerun:
    rr.log("sweep/held_out_r2", rr.BarChart(np.array([r["held_out_r2"] for r in viz_rows])))
    rr.log("sweep/trainable_pct", rr.BarChart(np.array([r["trainable_pct"] for r in viz_rows])))
    rr.log("forgetting/task_a_r2", rr.BarChart(np.array([frozen["task_a_r2"], lora["task_a_r2"], full["task_a_r2"]])))

print(f"\nheadline: rank-{HEADLINE_RANK} LoRA trains {lora['trainable_pct']:.2f}% of the weights and recovers "
      f"{recovered * 100:.0f}% of full-FT's held-out gain ({frozen['held_out_r2']:.2f} -> "
      f"{lora['held_out_r2']:.2f} vs full {full['held_out_r2']:.2f}).")
print(f"forgetting (the honest twist): task_A R^2  frozen {frozen['task_a_r2']:.2f}  ->  LoRA "
      f"{lora['task_a_r2']:.2f}  vs  full-FT {full['task_a_r2']:.2f}.  Freezing W did NOT protect the old skill.")
print(f"step 0: LoRA output is {lora['step0_frozen_gap']:.2e} from the frozen policy "
      f"[{'zero-init B: adapted == frozen' if not RAND_B else 'BREAK rand_init_B: prior already corrupted'}].")
print(f"metrics: {args.out / 'metrics.json'}  |  vizdata: {args.out / 'demo' / 'vizdata.json'}")
if args.rerun:
    print(f"recording: {args.out / 'lora.rrd'} — open it with: rerun {args.out / 'lora.rrd'}")
# --- endregion ---
