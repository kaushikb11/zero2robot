"""zero2robot 1.7 — Tokens Meet Torques: The Tiny VLA, Part I (the data).

A vision-language-action policy eats three things at once — an instruction, an
image, and a state — and predicts an action. Chapter 1.8 trains that policy;
this file builds the DATA it will train on, and every piece a VLA data stack
needs is here, from scratch:

  (1) MULTI-TASK: assemble demos from TWO different tasks and embodiments — the
      PushT pusher and the ALOHA bimanual cube handoff — into one pile.
  (2) INSTRUCTION TEMPLATING: attach a natural-language instruction to each demo
      from paraphrase templates, and turn it into token ids with a hand-built
      word-level TOKENIZER (fixed vocab, no HF tokenizers).
  (3) FROZEN VISION ENCODER: run every 96x96 camera frame through a small,
      random-init, FROZEN CNN (an nn.Conv2d stack, built here) to get the visual
      features a downstream policy conditions on.
  (4) PIPELINE: emit unified (instruction_tokens, image_features, state) ->
      action examples as one multi-task dataset.

No policy is trained here — that is 1.8. The lesson is how the examples are made,
and (Break It) how easily a careless template leaks the answer into the words so
a policy would never learn to look at the image.

Run it:      python curriculum/phase1_imitation/ch1.7_vla_data/vla_data.py --seed 0
Break it:    python curriculum/phase1_imitation/ch1.7_vla_data/vla_data.py --seed 0 --break leak
CI smoke:    python curriculum/phase1_imitation/ch1.7_vla_data/vla_data.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
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
from curriculum.common.envs.aloha_cube.scripted_expert import ScriptedExpert as AlohaExpert  # noqa: E402
from curriculum.common.envs.pusht.pusht_env import PushTEnv  # noqa: E402
from curriculum.common.envs.pusht.scripted_expert import ScriptedExpert as PushtExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

STATE_DIM = 10       # both envs expose a 10-number state (see each env's docstring)
ACT_DIM_MAX = 6      # pusht acts in 2 dims, aloha in 6 — we pad pusht up to the max
MAX_TOKENS = 16      # fixed instruction length: [BOS] + words + [EOS], then padded
IMG_HW = 96          # the top-down camera size both envs render

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch1.7-vla-data"))
parser.add_argument("--episodes_per_task", type=int, default=12)  # cpu-laptop: 12 | smoke: 2
parser.add_argument("--frame_stride", type=int, default=2,
                    help="keep every Nth control frame (real VLA data subsamples too)")
parser.add_argument("--feature_dim", type=int, default=64,
                    help="width of the frozen CNN's output — the visual feature a policy conditions on")
parser.add_argument("--conv_width", type=int, default=16,
                    help="channels in the first conv; doubles each of the three stages")
parser.add_argument("--log_examples", type=int, default=32, help="how many examples to stream to rerun")
parser.add_argument("--seed", type=int, default=0, help="seeds the demos and the frozen CNN's random init")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--break", dest="break_mode", choices=("leak",), default=None,
                    help="leak = templates that name the action direction; a policy could then ignore the image")
parser.add_argument("--smoke", action="store_true",
                    help="tiny hermetic CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # seeds python/numpy/torch; the frozen CNN init draws from torch's RNG
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.episodes_per_task, args.feature_dim, args.conv_width = 2, 16, 8
    args.device, args.log_examples = "cpu", 8
banner("ch1.7-vla-data", device=args.device)  # report the device the run ACTUALLY uses
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch1.7-vla-data", spawn=False)
    rr.save(str(args.out / "vla_data.rrd"))
# --- endregion ---

# --- region: tasks ---
# Two tasks, two embodiments, ONE pile — that is the multi-task point. A real VLA
# (RT-X, OpenVLA) mixes dozens of datasets; the idea is already visible with two.
# Each task brings its OWN env, its OWN scripted expert (the SAME experts
# gen_demos.py writes the LeRobot v3 datasets with — we replay them in-process here
# so we can render frames without a video-decode dependency), its action
# dimensionality, and a set of paraphrase instructions.
TASKS = [
    {
        "name": "pusht",
        "env": PushTEnv, "expert": PushtExpert, "act_dim": PushTEnv.ACT_DIM,
        "templates": [
            "push the t block onto the target",
            "slide the tee onto the goal",
            "move the t shape to the target pose",
            "push the block until it covers the target",
        ],
    },
    {
        "name": "aloha",
        "env": AlohaCubeEnv, "expert": AlohaExpert, "act_dim": AlohaCubeEnv.ACT_DIM,
        "templates": [
            "transfer the cube to the other arm",
            "hand the cube from the right arm to the left arm",
            "pick up the cube and pass it to the left gripper",
            "carry the cube across and place it on the target",
        ],
    },
]


def collect_task(task: dict, episodes: int, seed: int, stride: int):
    """Replay the scripted expert for `episodes` episodes; return per-frame states,
    padded actions, 96x96 frames, and the episode index of each frame.

    Deterministic given seed: episode e uses env+expert seed (seed + e), exactly
    the reproducibility contract gen_demos.py relies on. Actions live in the task's
    native dimensionality (2 for pusht, 6 for aloha); we zero-pad to ACT_DIM_MAX so
    both tasks share one action tensor — the honest cost of mixing embodiments.
    """
    env = task["env"]()
    states, actions, frames, ep_index = [], [], [], []
    for e in range(episodes):
        obs = env.reset(seed + e)
        expert = task["expert"](noise=0.0, seed=seed + e)
        step, done = 0, False
        while not done:
            action = expert.action(env)
            if step % stride == 0:  # subsample: consecutive frames are near-duplicates
                padded = np.zeros(ACT_DIM_MAX, dtype=np.float32)
                padded[: task["act_dim"]] = action[: task["act_dim"]]
                states.append(obs.astype(np.float32))
                actions.append(padded)
                frames.append(env.render_frame(IMG_HW, IMG_HW))
                ep_index.append(e)
            obs, _, done, _ = env.step(action)
            step += 1
    return (np.asarray(states, np.float32), np.asarray(actions, np.float32),
            np.asarray(frames, np.uint8), np.asarray(ep_index, np.int64))
# --- endregion ---

# --- region: language ---
# A VLA reads words, and words must become integers. We build the WHOLE tokenizer
# from scratch: a fixed WORD-LEVEL vocab (no BPE, no HF tokenizers), four special
# ids, pad/truncate to MAX_TOKENS. It is tiny (~40 words) on purpose — a real VLA
# uses a 30k+ subword vocab from a pretrained language model, which is one of the
# things chapter 1.8 reaches for. Here the point is the MECHANISM, laid bare.
_SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]  # ids 0..3, by construction
# --break leak appends one of these compass words per FRAME. They must be IN vocab:
# an out-of-vocab word would collapse to <unk> and leak nothing.
_DIRECTIONS = ["east", "northeast", "north", "northwest",
               "west", "southwest", "south", "southeast"]


class Tokenizer:
    """Word-level tokenizer over a FIXED vocab. Deterministic: the vocab is the
    sorted set of corpus words, so the same corpus always yields the same ids."""

    def __init__(self, corpus: list[str]) -> None:
        words = sorted({w for text in corpus for w in text.split()})
        self.itos = _SPECIALS + words              # index -> string
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> np.ndarray:
        # [BOS] word ids... [EOS], unknown words -> <unk>, then pad/truncate.
        ids = [self.stoi["<bos>"]]
        ids += [self.stoi.get(w, self.stoi["<unk>"]) for w in text.split()]
        ids.append(self.stoi["<eos>"])
        ids = ids[:MAX_TOKENS] + [self.stoi["<pad>"]] * (MAX_TOKENS - len(ids))
        return np.asarray(ids[:MAX_TOKENS], dtype=np.int64)


def direction_word(action_xy: np.ndarray) -> str:
    """8-way compass word for a 2D action vector; the leak template appends it."""
    sector = int(np.round(np.arctan2(action_xy[1], action_xy[0]) / (np.pi / 4))) % 8
    return _DIRECTIONS[sector]
# --- endregion ---

# --- region: vision ---
class FrozenVisionEncoder(nn.Module):
    """A from-scratch conv stack: (B, 96, 96, 3) uint8 -> (B, feature_dim). It is
    RANDOM-INIT and FROZEN — never trained here or in 1.8.

    Deliberately bare: Conv/ReLU and a global average pool, no BatchNorm (its
    batch-dependent stats would break determinism) and no attention. The honest
    claim: a frozen random CNN is a fixed nonlinear projection of the pixels. It
    still preserves coarse spatial layout — roughly WHERE the block, cube, and arms
    are — so its output is a usable visual feature, and already more compact and
    policy-friendly than raw pixels. What it does NOT give you is what a PRETRAINED
    backbone (DINOv2 / SigLIP, as in OpenVLA and SmolVLA) does: features aligned to
    objects and to language, transferable across scenes. Closing that gap is the
    reason chapter 1.8 bolts on a real backbone.
    """

    def __init__(self, width: int, feature_dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1), nn.ReLU(),              # 96 -> 48
            nn.Conv2d(width, 2 * width, 3, stride=2, padding=1), nn.ReLU(),      # 48 -> 24
            nn.Conv2d(2 * width, 4 * width, 3, stride=2, padding=1), nn.ReLU(),  # 24 -> 12
            nn.AdaptiveAvgPool2d(1),                                             # -> (4*width, 1, 1)
        )
        self.head = nn.Linear(4 * width, feature_dim)
        for p in self.parameters():
            p.requires_grad_(False)  # FROZEN: the perception is fixed, not learned
        self.eval()

    @torch.no_grad()
    def forward(self, images_uint8: torch.Tensor) -> torch.Tensor:
        # (B, 96, 96, 3) uint8 -> (B, 3, 96, 96) float in [-1, 1] -> (B, feature_dim)
        x = images_uint8.to(torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0
        return self.head(self.stem(x).flatten(1))


def encode_frames(encoder: FrozenVisionEncoder, frames: np.ndarray,
                  device: torch.device, batch: int = 256) -> np.ndarray:
    feats = []
    for i in range(0, len(frames), batch):
        chunk = torch.from_numpy(frames[i:i + batch]).to(device)
        feats.append(encoder(chunk).cpu().numpy())
    return np.concatenate(feats).astype(np.float32)
# --- endregion ---

# --- region: pipeline ---
# 1) MULTI-TASK: gather both piles and stack them into one dataset.
per_task = [collect_task(t, args.episodes_per_task, args.seed, args.frame_stride) for t in TASKS]
states = np.concatenate([p[0] for p in per_task])
actions = np.concatenate([p[1] for p in per_task])
frames = np.concatenate([p[2] for p in per_task])
task_id = np.concatenate([np.full(len(p[0]), i, np.int64) for i, p in enumerate(per_task)])
ep_id = np.concatenate([p[3] for p in per_task])
num_examples = len(states)

# 2) INSTRUCTION TEMPLATING + TOKENIZING. The vocab is fixed and known up front —
# every template word, the compass words, and "moving" (used by the leak template).
corpus = [t for task in TASKS for t in task["templates"]] + _DIRECTIONS + ["moving"]
tokenizer = Tokenizer(corpus)
# Pick one paraphrase per EPISODE, deterministically from the seed, and reuse its
# wording for every frame of that episode — real demos are annotated once per
# episode. --break leak overrides this per FRAME (it names the move; see below).
instructions = []
for i in range(num_examples):
    task = TASKS[task_id[i]]
    variant = (args.seed + task_id[i] * 7919 + ep_id[i]) % len(task["templates"])
    text = task["templates"][variant]
    if args.break_mode == "leak":
        # THE MISCONCEPTION: the annotator "helpfully" writes which way to move into
        # the instruction. Now the WORDS carry the action, so a policy can read the
        # answer off language and never look at the pixels. Measured by the probe below.
        text = f"{text} moving {direction_word(actions[i, :2])}"
    instructions.append(text)
tokens = np.stack([tokenizer.encode(t) for t in instructions])
specials = {tokenizer.stoi[s] for s in ("<pad>", "<bos>", "<eos>")}
content = tokens[~np.isin(tokens, list(specials))]
oov_rate = float((content == tokenizer.stoi["<unk>"]).mean()) if content.size else 0.0

# 3) FROZEN VISION ENCODER: pixels -> features the policy will condition on.
encoder = FrozenVisionEncoder(args.conv_width, args.feature_dim).to(device)
image_features = encode_frames(encoder, frames, device)

# 4) LEAKAGE PROBE — the Break-It's measured signature. How much of the action can a
# LINEAR model read out of the instruction tokens ALONE, per task? Build a
# bag-of-words instruction feature, least-squares onto the action, report R^2 of the
# fit. Normal templates name the TASK, not the move, so within a task the words
# barely vary and the probe explains ~none of the per-frame action (R^2 ~ 0). Leak
# templates name the direction every frame, so R^2 jumps: the answer is decodable
# from language, and a policy trained on it would learn to ignore vision entirely.
def language_action_r2(rows: np.ndarray, act_dim: int) -> float:
    bag = np.zeros((len(rows), tokenizer.vocab_size), np.float64)
    for r, i in enumerate(rows):
        for tok in tokens[i]:
            if tok != tokenizer.stoi["<pad>"]:
                bag[r, tok] += 1.0
    X = np.concatenate([bag, np.ones((len(rows), 1))], axis=1)   # + intercept column
    Y = actions[rows, :act_dim].astype(np.float64)
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    ss_res = float(((Y - X @ W) ** 2).sum())
    ss_tot = float(((Y - Y.mean(0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 1e-9 else 0.0


r2_by_task = [language_action_r2(np.where(task_id == i)[0], t["act_dim"])
              for i, t in enumerate(TASKS)]
action_from_language_r2 = float(np.mean(r2_by_task))

# 5) EMIT the unified dataset. A documented .npz (a real stack streams LeRobot v3
# from the Hub); the action_mask marks which action dims are valid per embodiment.
action_mask = np.zeros((num_examples, ACT_DIM_MAX), np.float32)
for i, t in enumerate(TASKS):
    action_mask[task_id == i, : t["act_dim"]] = 1.0
np.savez(args.out / "vla_dataset.npz", instruction_tokens=tokens,
         image_features=image_features, state=states, action=actions,
         action_mask=action_mask, task_id=task_id)
manifest = {
    "schema": {
        "instruction_tokens": ["num_examples", MAX_TOKENS, "int64 token ids"],
        "image_features": ["num_examples", args.feature_dim, "float32 (frozen CNN)"],
        "state": ["num_examples", STATE_DIM, "float32"],
        "action": ["num_examples", ACT_DIM_MAX, "float32 (zero-padded)"],
        "action_mask": ["num_examples", ACT_DIM_MAX, "1.0 where the action dim is real"],
        "task_id": ["num_examples", "int64 index into tasks[]"],
    },
    "tasks": [{"id": i, "name": t["name"], "act_dim": t["act_dim"],
               "templates": t["templates"], "count": int((task_id == i).sum())}
              for i, t in enumerate(TASKS)],
    "vocab": tokenizer.itos,
    "max_tokens": MAX_TOKENS,
    "feature_dim": args.feature_dim,
    "frozen_vision_encoder": "from-scratch conv stack, random-init, never trained",
    "break_mode": args.break_mode or "none",
}
(args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

# rerun: the multi-task MIX (how many examples per task) and a stream of sampled
# examples — the frame, its frozen features, state, action, and task id.
if args.rerun:
    for i, t in enumerate(TASKS):
        rr.log(f"data/task_mix/{t['name']}", rr.Scalars([float((task_id == i).sum())]), static=True)
    sample = np.unique(np.linspace(0, num_examples - 1, min(args.log_examples, num_examples)).astype(int))
    for out_step, i in enumerate(sample):
        rr.set_time("example", sequence=out_step)
        rr.log("data/image", rr.Image(frames[i]))
        rr.log("data/image_features", rr.BarChart(image_features[i].astype(np.float64)))
        rr.log("data/state", rr.Scalars(states[i].astype(np.float64)))
        rr.log("data/action", rr.Scalars(actions[i].astype(np.float64)))
        rr.log("data/task_id", rr.Scalars([float(task_id[i])]))

metrics = {
    "action_from_language_r2": round(action_from_language_r2, 6),  # the leak signature
    "break_mode": args.break_mode or "none",
    "feature_dim": int(args.feature_dim),
    "image_feature_mean": round(float(image_features.mean()), 6),   # frozen-encoder fingerprint
    "image_feature_std": round(float(image_features.std()), 6),
    "max_tokens": int(MAX_TOKENS),
    "num_examples": int(num_examples),
    "num_examples_aloha": int((task_id == 1).sum()),
    "num_examples_pusht": int((task_id == 0).sum()),
    "oov_rate": round(oov_rate, 6),
    "r2_aloha": round(r2_by_task[1], 6),
    "r2_pusht": round(r2_by_task[0], 6),
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "vocab_size": int(tokenizer.vocab_size),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"dataset: {num_examples} examples "
      f"({metrics['num_examples_pusht']} pusht / {metrics['num_examples_aloha']} aloha), "
      f"vocab {tokenizer.vocab_size}, feature_dim {args.feature_dim}")
print(f"language->action R^2: {action_from_language_r2:.3f} "
      f"(pusht {r2_by_task[0]:.3f}, aloha {r2_by_task[1]:.3f}), break={args.break_mode or 'none'}")
print(f"wrote {args.out / 'vla_dataset.npz'} + manifest.json + metrics.json")
if args.rerun:
    print(f"recording: {args.out / 'vla_data.rrd'} — open it with: rerun {args.out / 'vla_data.rrd'}")
# --- endregion ---
