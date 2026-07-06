"""zero2robot 1.3 — ACT: Commit to the Chunk.

Behavior cloning (1.1) predicts ONE action per observation, and chapter 1.2
showed the ceiling that leaves you at: at 10 Hz, a single-step policy re-decides
everything every 0.1 s and, at a state where two futures are equally good,
averages them into a bad third one. ACT (Zhao et al., "Learning Fine-Grained
Bimanual Manipulation with Low-Cost Hardware") changes the OUTPUT: predict a
CHUNK of the next K actions in one forward pass, so the policy commits to a
short plan instead of re-averaging every step. The bimanual handoff in
aloha_cube is exactly the coordination a chunk can hold and a single step cannot.

Three ideas, all in this file:
  1. ACTION CHUNKING  — the network maps obs -> (K, act_dim); training regresses
     the next K expert actions (L1), padded at episode ends.
  2. A TINY TRANSFORMER, from scratch — the obs is split into four entity tokens
     (right arm / left arm / cube / target); a hand-rolled attention ENCODER
     lets them attend to each other, and a DECODER of K learned query tokens
     cross-attends to that memory to emit the chunk. No `transformers`, no timm.
  3. TEMPORAL ENSEMBLING — at eval, every step is covered by several overlapping
     chunks predicted at earlier steps; we average them with exponential weights
     for a smooth, committed trajectory instead of jerky chunk seams.

SIMPLIFIED from real ACT (flagged honestly, taught in prose): real ACT is a
CVAE — a style encoder over the action sequence produces a latent z that the
decoder conditions on, to model demonstrator multimodality. We DROP the CVAE
(no z, no KL term): a deterministic chunking policy is the pedagogical core and
still beats single-step BC here. We also train on the STATE obs (10 numbers),
not images, so there is no ResNet backbone. Both are real omissions, not
approximations — see prose "What we cut".

Run it:      python curriculum/phase1_imitation/ch1.3_act/act.py --seed 0
Break it:    python curriculum/phase1_imitation/ch1.3_act/act.py --seed 0 --break no_ensemble
CI smoke:    python curriculum/phase1_imitation/ch1.3_act/act.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch1.1's bc.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.aloha_cube import AlohaCubeEnv, gen_demos  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

HEADS = 4          # attention heads (fixed; model_dim is the scale knob)
ENC_LAYERS = 2     # encoder self-attention blocks over the 4 entity tokens
DEC_LAYERS = 2     # decoder blocks (self-attn over queries + cross-attn to memory)
OBS_DIM, ACT_DIM = AlohaCubeEnv.OBS_DIM, AlohaCubeEnv.ACT_DIM

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=None,
                    help="LeRobot aloha_cube dataset; omitted => generate --num_demos scripted demos")
parser.add_argument("--out", type=Path, default=Path("outputs/ch1.3-act"))
parser.add_argument("--chunk_size", type=int, default=8,
                    help="K: actions predicted per forward pass. aloha episodes are ~27 steps, so keep K well under that (measured: K=8 beats K=16 here, both crush K=1)")  # smoke: 4
parser.add_argument("--model_dim", type=int, default=128)  # T4: 256 | smoke: 16
parser.add_argument("--num_demos", type=int, default=50)   # T4: 200 | smoke: 4
parser.add_argument("--epochs", type=int, default=400)     # cpu-laptop: minutes | smoke: 3
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3, help="peak Adam lr; cosine-decays to 0 over --epochs")
parser.add_argument("--ensemble_m", type=float, default=0.1,
                    help="temporal-ensembling decay: larger = concentrate weight on the OLDEST overlapping prediction (commit to earlier plans); ~0 = uniform (real ACT uses ~0.01)")
parser.add_argument("--eval_episodes", type=int, default=25)  # T4: 50 | smoke: 2 — few episodes is noisy (ch1.6)
parser.add_argument("--seed", type=int, default=0, help="seeds demo generation, the init, and the shuffle")
parser.add_argument("--break", dest="break_mode",
                    choices=("no_chunk", "no_ensemble", "open_loop"), default=None,
                    help="Break It: a real ACT misconception with a measured signature (see the eval region)")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())  # cpu: deterministic (statistical repro on GPU/mps)
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # seeds python/numpy/torch; returns the numpy Generator demo-gen draws from
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.chunk_size, args.model_dim, args.num_demos = 4, 16, 4
    args.epochs, args.eval_episodes, args.device = 3, 2, "cpu"
if args.break_mode == "no_chunk":
    args.chunk_size = 1  # the whole ACT idea, ablated: predict a single action, like 1.1's BC
banner("ch1.3-act", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch1.3-act", spawn=False)
    rr.save(str(args.out / "act.rrd"))
# --- endregion ---

# --- region: data ---
# ACT trains on the SAME scripted-expert demos as everything else in Phase 1;
# the only new step is reshaping per-frame actions into per-frame CHUNKS.
if args.data is None:
    # Regenerate every run (never reuse a leftover dir): a cache from a different
    # --seed / --num_demos would silently train on the wrong data. gen_demos is
    # deterministic, so same args -> bit-identical demos whether built or rebuilt.
    args.data = args.out / "demos"
    if args.data.exists():
        shutil.rmtree(args.data)
    gen_demos.main(["--episodes", str(args.num_demos), "--seed", str(args.seed),
                    "--out", str(args.data), "--no-video"])
if not (args.data / "meta" / "info.json").is_file():
    sys.exit(f"no dataset at {args.data} — generate one first:\n"
             f"  python curriculum/common/envs/aloha_cube/gen_demos.py "
             f"--episodes 50 --seed 0 --out {args.data} --no-video")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402  (heavy import — after cheap failures)

frames = LeRobotDataset("local/aloha_cube", root=args.data).hf_dataset.with_format("numpy")
obs = np.stack(frames["observation.state"]).astype(np.float32)   # (N, 10) — layout in aloha_cube_env.py
actions = np.stack(frames["action"]).astype(np.float32)          # (N, 6)  — already clipped to [-1, 1]
episode_ids = np.asarray(frames["episode_index"])                # (N,) which demo each frame came from

# Chunk targets: for frame i, the next K expert actions within ITS episode. Near
# an episode's end there are fewer than K left, so pad by repeating the last
# action and record a 0/1 mask so those padded steps carry no gradient. This is
# what turns a per-step dataset into a per-chunk one (real ACT does the same).
K = args.chunk_size
chunk_targets = np.zeros((len(obs), K, ACT_DIM), dtype=np.float32)
chunk_mask = np.zeros((len(obs), K), dtype=np.float32)
for e in np.unique(episode_ids):
    idx = np.nonzero(episode_ids == e)[0]  # frame indices of this episode, in order
    ep_actions = actions[idx]
    for j, frame in enumerate(idx):
        valid = min(K, len(idx) - j)
        chunk_targets[frame, :valid] = ep_actions[j:j + valid]
        chunk_targets[frame, valid:] = ep_actions[-1]  # pad (masked out below)
        chunk_mask[frame, :valid] = 1.0

# Normalization stats over obs only; actions are already in [-1, 1] by the env's
# action contract, so the chunk head regresses them directly (no action denorm).
obs_min = obs.min(0)
obs_range = np.where(obs.max(0) - obs_min < 1e-4, np.float32(1.0), obs.max(0) - obs_min)
obs_t = torch.from_numpy(obs).to(device)
chunk_t = torch.from_numpy(chunk_targets).to(device)
mask_t = torch.from_numpy(chunk_mask).to(device)
print(f"dataset: {len(np.unique(episode_ids))} episodes / {len(obs)} frames, "
      f"chunk_size={K}, model_dim={args.model_dim}")
# --- endregion ---

# --- region: model ---
# A transformer block is attention + a feed-forward net, each wrapped in a
# residual connection with pre-norm (LayerNorm before the sublayer — the stable
# variant). We hand-roll it from nn.MultiheadAttention so every piece is visible;
# nothing here is imported from a transformer library.
class EncoderBlock(nn.Module):
    """Self-attention over the entity tokens, then a per-token FFN."""

    def __init__(self, dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, HEADS, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, tokens, dim)
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]  # tokens attend to each other
        return x + self.ff(self.norm2(x))


class DecoderBlock(nn.Module):
    """Query tokens self-attend, then cross-attend to the encoder memory, then FFN."""

    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(dim, HEADS, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, HEADS, batch_first=True)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, q: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        h = self.norm1(q)
        q = q + self.self_attn(h, h, h, need_weights=False)[0]       # chunk steps coordinate
        h = self.norm2(q)
        q = q + self.cross_attn(h, memory, memory, need_weights=False)[0]  # read the scene
        return q + self.ff(self.norm3(q))


class ACTPolicy(nn.Module):
    """obs float32[B,10] -> action chunk float32[B,K,6]. The whole architecture.

    The obs is not a sequence, so we MAKE one: split the 10 numbers into four
    entity tokens (right arm, left arm, cube, target), each padded to width 3 and
    projected to model_dim. Self-attention over those four tokens is where the
    policy reasons about relationships — where is the cube relative to each
    gripper, which arm should act. K learned query tokens then cross-attend to
    that memory; each query becomes one action in the chunk.
    """

    def __init__(self, dim: int, chunk_size: int, obs_min, obs_range):
        super().__init__()
        self.token_proj = nn.Linear(3, dim)              # shared across the 4 tokens
        self.type_embed = nn.Parameter(torch.zeros(1, 4, dim))       # "which entity am I"
        self.query_embed = nn.Parameter(0.02 * torch.randn(1, chunk_size, dim))  # "which step am I"
        self.encoder = nn.ModuleList(EncoderBlock(dim) for _ in range(ENC_LAYERS))
        self.decoder = nn.ModuleList(DecoderBlock(dim) for _ in range(DEC_LAYERS))
        self.head = nn.Linear(dim, ACT_DIM)              # each query -> one 6-D action
        for name, stat in [("obs_min", obs_min), ("obs_range", obs_range)]:
            self.register_buffer(name, torch.from_numpy(stat))  # saved with the weights, never trained

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        obs_n = (2.0 * (observation - self.obs_min) / self.obs_range - 1.0).clamp(-1.0, 1.0)
        tokens = torch.stack([                            # (B, 4, 3) entity tokens
            obs_n[:, 0:3],                                # right arm: x, y, grip
            obs_n[:, 3:6],                                # left arm:  x, y, grip
            F.pad(obs_n[:, 6:8], (0, 1)),                 # cube:      x, y, (pad)
            F.pad(obs_n[:, 8:10], (0, 1)),                # target:    x, y, (pad)
        ], dim=1)
        memory = self.token_proj(tokens) + self.type_embed
        for block in self.encoder:
            memory = block(memory)
        queries = self.query_embed.expand(observation.shape[0], -1, -1)
        for block in self.decoder:
            queries = block(queries, memory)
        return self.head(queries)                         # (B, K, 6)


policy = ACTPolicy(args.model_dim, K, obs_min, obs_range).to(device)
# --- endregion ---

# --- region: train ---
# Plain loop, one policy, no DataLoader. The loss is L1 (ACT's choice — sharper
# than MSE on multimodal action data) between the predicted chunk and the K
# expert actions, averaged over the VALID (unpadded) steps only.
def chunk_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    per_step = (pred - target).abs().mean(-1)     # (B, K) mean over the 6 action dims
    return (per_step * mask).sum() / mask.sum()   # ignore padded steps


optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
# Decay the lr to 0 over the run so the last epochs settle instead of bouncing.
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
shuffle = torch.Generator().manual_seed(args.seed)  # torch-side RNG: same seed -> same batch order
train_loss, global_step = float("nan"), 0
for epoch in range(args.epochs):
    epoch_loss, num_batches = 0.0, 0
    for batch in torch.randperm(len(obs_t), generator=shuffle).split(args.batch_size):
        loss = chunk_l1(policy(obs_t[batch]), chunk_t[batch], mask_t[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss, num_batches = epoch_loss + loss.item(), num_batches + 1
        if args.rerun:
            rr.set_time("step", sequence=global_step)
            rr.log("policy/loss/train", rr.Scalars([loss.item()]))
        global_step += 1
    scheduler.step()
    train_loss = epoch_loss / num_batches
    if epoch % 50 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:4d}  chunk_l1 {train_loss:.5f}")
# --- endregion ---

# --- region: eval ---
# Loss measured imitation on dataset states; rollouts measure the task. At each
# step the policy predicts a fresh chunk, but every step is ALSO covered by
# chunks predicted at earlier steps. Temporal ensembling averages all of those
# overlapping predictions for the current step with exponential weights (older
# predictions weighted exp(-m * age)), so the executed trajectory is smooth and
# committed rather than jerking at chunk boundaries. --break ablates this:
#   no_chunk    K=1 (set in setup): single-step BC through a transformer — the
#               chunking benefit is gone; measure how far success falls.
#   no_ensemble execute only the freshly predicted chunk's first action, no
#               averaging across overlapping chunks.
#   open_loop   commit to the whole chunk (run all K, then re-query): the naive
#               "why bother ensembling" version, jerky at every chunk seam.
MAX_T = AlohaCubeEnv.MAX_STEPS


@torch.no_grad()
def predict_chunk(net: ACTPolicy, observation: np.ndarray) -> np.ndarray:
    obs_batch = torch.from_numpy(observation).to(device).unsqueeze(0)  # (10,) -> (1, 10)
    return net(obs_batch)[0].cpu().numpy()                             # (K, 6)


def rollout(net: ACTPolicy, seed: int, mode: str, tag: str, episode: int) -> tuple[bool, float]:
    env = AlohaCubeEnv()
    observation = env.reset(seed)
    all_time = np.zeros((MAX_T, MAX_T + K, ACT_DIM), dtype=np.float32)  # [query time, target time]
    populated = np.zeros((MAX_T, MAX_T + K), dtype=bool)
    committed: list[np.ndarray] = []
    done, episode_return, info, t = False, 0.0, {}, 0
    while not done and t < MAX_T:
        if mode == "open_loop":
            if not committed:                                    # re-query only when the plan runs out
                committed = list(predict_chunk(net, observation))
            action = committed.pop(0)
        else:
            chunk = predict_chunk(net, observation)
            all_time[t, t:t + K] = chunk
            populated[t, t:t + K] = True
            if mode == "no_ensemble":
                action = chunk[0]
            else:  # temporal ensembling: average every chunk that predicted step t
                votes = all_time[:t + 1, t][populated[:t + 1, t]]    # oldest first
                weights = np.exp(-args.ensemble_m * np.arange(len(votes)))
                action = (votes * (weights / weights.sum())[:, None]).sum(0)
        observation, reward, done, info = env.step(action)
        episode_return += reward
        t += 1
        if args.rerun:
            rr.set_time("sim_time", duration=episode * (MAX_T / AlohaCubeEnv.CONTROL_HZ) + env.data.time)
            rr.log(f"eval/{tag}/action", rr.Scalars(action.astype(np.float64)))
            rr.log(f"eval/{tag}/dist", rr.Scalars([info["dist"]]))
    return bool(info["success"]), episode_return


def evaluate(net: ACTPolicy, mode: str, tag: str) -> tuple[float, float]:
    outcomes = [rollout(net, 20_000 + args.seed + ep, mode, tag, ep)  # held-out seeds, never in demos
                for ep in range(args.eval_episodes)]
    success_rate = float(np.mean([s for s, _ in outcomes]))
    mean_return = float(np.mean([r for _, r in outcomes]))
    if args.rerun:
        rr.log(f"eval/{tag}/success_rate", rr.Scalars([success_rate]))
    print(f"eval[{tag:11s}]: success {success_rate:.2f}  mean_return {mean_return:.3f}")
    return success_rate, mean_return


eval_mode = args.break_mode if args.break_mode in ("no_ensemble", "open_loop") else "ensemble"
baseline_success, baseline_return = evaluate(ACTPolicy(args.model_dim, K, obs_min, obs_range).to(device),
                                             "ensemble", "untrained")  # random-init reference
success_rate, mean_return = evaluate(policy, eval_mode, "trained")
# --- endregion ---

# --- region: report ---
# ONNX/contract note: contract v1 is model(obs[1,10]) -> action[1,6], a single
# stateless step. This policy outputs a CHUNK [1, K, 6] and its temporal
# ensembling is a stateful runtime loop, so it does NOT fit contract v1 — a
# chunked policy needs a later contract version (chunk output + an ensembling
# buffer the browser runtime owns). We still export the chunk core and check
# torch/onnxruntime agree, so the serialization path is proven; we just don't
# claim v1 conformance or hand it to the ch1.1 playground.
if not args.smoke:
    import onnxruntime as ort  # noqa: E402  (heavy; only when actually exporting)

    onnx_path = args.out / "act_policy.onnx"
    policy.eval().to("cpu")
    dummy = torch.zeros(1, OBS_DIM)
    torch.onnx.export(policy, (dummy,), str(onnx_path), input_names=["observation"],
                      output_names=["action_chunk"], dynamo=False)  # output [1, K, 6] — NOT contract v1
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    probe = np.random.default_rng(0).standard_normal((1, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        delta = float(np.abs(policy(torch.from_numpy(probe)).numpy()
                             - session.run(None, {"observation": probe})[0]).max())
    assert delta < 1e-4, f"torch/onnx chunk parity failed: {delta:.2e}"
    print(f"exported {onnx_path} (chunk output [1,{K},{ACT_DIM}], NOT contract v1); parity {delta:.2e}")
    policy.to(device)

metrics = {
    "baseline_mean_return": round(baseline_return, 6),
    "baseline_success_rate": round(baseline_success, 6),
    "break_mode": args.break_mode or "none",
    "chunk_size": K,
    "epochs": args.epochs,
    "final_train_loss": round(train_loss, 6),
    "mean_return": round(mean_return, 6),
    "model_dim": args.model_dim,
    "num_demos": len(np.unique(episode_ids)),
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "success_rate": round(success_rate, 6),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}  (trained vs untrained baseline: "
      f"return {mean_return:.3f} vs {baseline_return:.3f})")
if args.rerun:
    print(f"recording: {args.out / 'act.rrd'} — open it with: rerun {args.out / 'act.rrd'}")
# --- endregion ---
