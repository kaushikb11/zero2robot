"""zero2robot 1.9 — Graduation Bridge I: LeRobot for Real.

In 1.3 you built ACT from scratch: a hand-rolled encoder/decoder, a chunking
head, temporal ensembling — ~380 lines you can read end to end. That was the
point: you now KNOW what ACT is. This chapter is the graduation bridge. The same
policy already lives in the official `lerobot` stack (the one the field actually
ships), and instantiating it there is ~20 lines. This file trains that official
ACT on YOUR dataset, evaluates it against your from-scratch 1.3 run on the SAME
env and seeds, and shows you the publish path to the Hugging Face Hub.

This is the first chapter where importing a framework is the LESSON, not a
violation of it. Everywhere else in Phase 1 the doctrine forbids frameworks that
hide the loop, because you were learning the loop. Here you have learned it, so
`from lerobot.policies.act... import ACTPolicy` is exactly what you should reach
for. The trade you are being shown, concretely and measured:

  1. THE OFFICIAL POLICY   — `ACTConfig` + `ACTPolicy` + `make_act_pre_post_
     processors` is the whole model. We MATCH 1.3's choices for a fair compare:
     use_vae=False (1.3 dropped the CVAE) and the same lr. The ResNet backbone
     lives in here too (we feed state, so it stays dormant), and normalization is
     a processor pipeline built from the dataset's own stats. Flip use_vae=True
     for the full CVAE ACT — on this tiny state task it regularizes hard and wants
     more data/epochs, so we leave it off to isolate the algorithm, not the trick.
  2. THE SAME DATA         — your 1.3 LeRobot-v3 dataset loads unchanged. The one
     honest wrinkle: ACT's state-only schema wants `observation.environment_
     state`; our recorder emits `observation.state`. Bridging is one rename.
  3. AN HONEST COMPARISON  — official-ACT vs from-scratch-1.3-ACT on held-out
     seeds, with the Wilson intervals from 1.6, because a single success number
     is a lie without them. Neither "wins" by construction — you read the bars.

OFFLINE BY DEFAULT: no token, no network. `--publish` is the documented Hub path
and is HUMAN-GATED — offline (or without HF_TOKEN) it DRY-RUNS: it prints the
exact push_to_hub calls it would make and saves the policy locally instead.

Run it:      python curriculum/phase1_imitation/ch1.9_bridge/bridge.py --seed 0
Break it:    python curriculum/phase1_imitation/ch1.9_bridge/bridge.py --seed 0 --break train_dist
Publish:     python .../bridge.py --seed 0 --publish   # dry-runs unless HF_TOKEN + network
CI smoke:    python .../bridge.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch1.3's act.py).
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.aloha_cube import AlohaCubeEnv, gen_demos  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

OBS_DIM, ACT_DIM = AlohaCubeEnv.OBS_DIM, AlohaCubeEnv.ACT_DIM
REPO_ID = "local/aloha_cube"          # local dataset id; never resolved against the Hub
ACT_PY = ROOT / "curriculum/phase1_imitation/ch1.3_act/act.py"  # the from-scratch baseline
HELD_OUT_BASE = 20_000                # 1.3's held-out seed base; NEVER a demo seed (0..num_demos-1)
Z95 = 1.959963984540054               # 95% Wilson z (same constant as 1.6's harness)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=None,
                    help="a LeRobot-v3 dataset (your ch0.4 recording, or pusht/aloha_cube gen_demos); "
                         "omitted => generate --num_demos scripted aloha_cube demos, exactly as 1.3 does")
parser.add_argument("--out", type=Path, default=Path("outputs/ch1.9-bridge"))
parser.add_argument("--chunk_size", type=int, default=8,
                    help="K: actions per forward pass (ACTConfig.chunk_size). Matches 1.3's default")  # smoke: 4
parser.add_argument("--model_dim", type=int, default=128)  # ACTConfig.dim_model; T4: 256 | smoke: 16
parser.add_argument("--num_demos", type=int, default=50)   # T4: 200 | smoke: 6
parser.add_argument("--epochs", type=int, default=150)     # passes over the dataset; cpu-laptop: minutes | smoke: 3
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--lr", type=float, default=1e-3, help="AdamW lr (lerobot's ACT preset is 1e-5; we match 1.3's 1e-3 for the short run + fair compare)")
parser.add_argument("--eval_episodes", type=int, default=20)  # noisy at 20 (that's 1.6's whole point) | smoke: 2
parser.add_argument("--seed", type=int, default=0, help="seeds demo generation, the init, the shuffle, and both evals")
parser.add_argument("--break", dest="break_mode", choices=("train_dist",), default=None,
                    help="Break It: evaluate the official ACT on the TRAINING seeds instead of held-out ones — "
                         "the 1.6 sin, made concrete on the official stack (see the eval region)")
parser.add_argument("--repo-id", default="zero2robot/aloha_cube_act",
                    help="Hub repo id used ONLY by --publish (and only its dry-run, offline)")
parser.add_argument("--publish", action="store_true",
                    help="HUMAN-GATED Hub publish. Offline or without HF_TOKEN it DRY-RUNS: prints the "
                         "push_to_hub calls and saves the policy locally. Never required by CI")
parser.add_argument("--no-compare", dest="compare", action="store_false", default=True,
                    help="skip the from-scratch 1.3 subprocess (report official-only)")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())  # cpu: deterministic
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; lerobot's DataLoader + VAE draw from torch
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.chunk_size, args.model_dim, args.num_demos = 4, 16, 6
    args.epochs, args.eval_episodes, args.device = 3, 2, "cpu"
K = args.chunk_size
banner("ch1.9-bridge", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch1.9-bridge", spawn=False)
    rr.save(str(args.out / "bridge.rrd"))
# --- endregion ---

# --- region: data ---
# Reuse 1.3's exact demo source so the two ACTs train on bit-identical data.
if args.data is None:
    args.data = args.out / "demos"
    if not (args.data / "meta" / "info.json").is_file():
        gen_demos.main(["--episodes", str(args.num_demos), "--seed", str(args.seed),
                        "--out", str(args.data), "--no-video"])
if not (args.data / "meta" / "info.json").is_file():
    sys.exit(f"no dataset at {args.data} — generate one first:\n"
             f"  python curriculum/common/envs/aloha_cube/gen_demos.py "
             f"--episodes 50 --seed 0 --out {args.data} --no-video")

# Heavy imports AFTER the cheap failures — the whole official stack, in one place.
from lerobot.configs.types import FeatureType, PolicyFeature  # noqa: E402
from lerobot.datasets.lerobot_dataset import (  # noqa: E402
    LeRobotDataset, LeRobotDatasetMetadata)
from lerobot.datasets.utils import dataset_to_policy_features  # noqa: E402
from lerobot.policies.act.configuration_act import ACTConfig  # noqa: E402
from lerobot.policies.act.modeling_act import ACTPolicy  # noqa: E402
from lerobot.policies.act.processor_act import make_act_pre_post_processors  # noqa: E402

ds_meta = LeRobotDatasetMetadata(REPO_ID, root=args.data)
features = dataset_to_policy_features(ds_meta.features)
# THE BRIDGE, in two lines. Our state IS the whole world (arms+cube+target), so
# it is ACT's `environment_state`, not proprioceptive `observation.state`. ACT's
# state-only schema requires that key; re-typing it to ENV is the entire remap.
features["observation.environment_state"] = PolicyFeature(
    FeatureType.ENV, features.pop("observation.state").shape)
output_features = {k: v for k, v in features.items() if v.type is FeatureType.ACTION}
input_features = {k: v for k, v in features.items() if k not in output_features}
# Normalization stats travel WITH the data (this is what the --break skips in 1.6
# spirit — here we always pass them). Re-key state stats to the bridged name.
stats = dict(ds_meta.stats)
stats["observation.environment_state"] = stats.pop("observation.state")
print(f"dataset: {ds_meta.total_episodes} episodes / {ds_meta.total_frames} frames "
      f"@ {ds_meta.fps} Hz; bridged observation.state -> observation.environment_state")
# --- endregion ---

# --- region: official ---
# THE ENTIRE OFFICIAL POLICY. Contrast this block with all of 1.3's act.py: the
# encoder, decoder, chunking head, CVAE, and temporal ensembler are ALL in here,
# behind three constructors. temporal_ensemble_coeff=0.01 (with n_action_steps=1)
# is the real ACT ensembling we hand-rolled in 1.3. use_vae=False MATCHES 1.3 (which
# cut the CVAE) so the comparison is apples-to-apples — flip it to True for the full
# CVAE ACT. This is the payoff of having built it yourself: every argument is legible.
config = ACTConfig(
    input_features=input_features, output_features=output_features,
    chunk_size=K, n_action_steps=1, temporal_ensemble_coeff=0.01,
    dim_model=args.model_dim, n_heads=8, dim_feedforward=2 * args.model_dim,
    n_encoder_layers=1, n_decoder_layers=1, use_vae=False,
    dropout=0.1, optimizer_lr=args.lr, device=args.device)
policy = ACTPolicy(config, dataset_stats=stats).to(device)
preprocessor, postprocessor = make_act_pre_post_processors(config, dataset_stats=stats)
n_params = sum(p.numel() for p in policy.parameters())
print(f"official lerobot ACT: {n_params:,} params (use_vae={config.use_vae}, "
      f"chunk_size={K}, dim_model={args.model_dim})")
# --- endregion ---

# --- region: train ---
# lerobot builds the action CHUNK for us: delta_timestamps asks the dataset for K
# future action frames per sample, plus an `action_is_pad` mask at episode ends —
# the exact bookkeeping we wrote by hand in 1.3, now a one-liner. The preprocessor
# normalizes with the dataset stats; policy.forward returns the L1(+KL) loss.
delta_timestamps = {"action": [i / ds_meta.fps for i in range(K)]}
dataset = LeRobotDataset(REPO_ID, root=args.data, delta_timestamps=delta_timestamps)
loader = torch.utils.data.DataLoader(
    dataset, batch_size=args.batch_size, shuffle=True, num_workers=0,
    generator=torch.Generator().manual_seed(args.seed))  # seeded order => reproducible on CPU
optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
policy.train()
train_loss, global_step = float("nan"), 0
for epoch in range(args.epochs):
    epoch_loss, num_batches = 0.0, 0
    for batch in loader:
        # The device-probe inside ACT reads observation.state even when it uses
        # env_state, so we carry both keys (same tensor); only env_state is a
        # model input. preprocessor() normalizes, batches, and moves to device.
        batch["observation.environment_state"] = batch["observation.state"]
        batch = preprocessor(batch)
        loss, _ = policy.forward(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss, num_batches = epoch_loss + loss.item(), num_batches + 1
        if args.rerun:
            rr.set_time("step", sequence=global_step)
            rr.log("official/loss/train", rr.Scalars([loss.item()]))
        global_step += 1
    train_loss = epoch_loss / num_batches
    if epoch % 25 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:4d}  loss {train_loss:.5f}")
# --- endregion ---

# --- region: eval ---
# Booleans in, statistics out — the 1.6 discipline. A bare success rate hides its
# own uncertainty; the Wilson interval is the honest error bar at these few
# episodes. --break train_dist swaps the held-out seeds for the TRAINING seeds:
# the policy has effectively memorized those exact starts, so the number inflates
# and you would wrongly crown the official stack. That is the 1.6 sin, committed
# on the real ACT — the eval, not the model, is what breaks.
def wilson_ci(k: int, n: int) -> tuple[float, float]:
    """95% Wilson score interval for k successes in n trials (no scipy)."""
    if n == 0:
        return (0.0, 0.0)
    p, z2 = k / n, Z95 * Z95
    center = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = (Z95 * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / (1 + z2 / n)
    return (max(0.0, center - half), min(1.0, center + half))


@torch.no_grad()
def rollout(seed: int, episode: int) -> tuple[bool, float]:
    policy.reset()  # clears the temporal-ensembling buffer between episodes
    env = AlohaCubeEnv()
    observation = env.reset(seed)
    done, episode_return, info = False, 0.0, {}
    while not done:
        obs_t = torch.from_numpy(observation)
        batch = preprocessor({"observation.state": obs_t, "observation.environment_state": obs_t})
        action = postprocessor(policy.select_action(batch))  # un-normalized env action
        observation, reward, done, info = env.step(action.squeeze(0).cpu().numpy())
        episode_return += reward
        if args.rerun:
            rr.set_time("sim_time", duration=episode * (AlohaCubeEnv.MAX_STEPS
                        / AlohaCubeEnv.CONTROL_HZ) + env.data.time)
            rr.log("eval/official/dist", rr.Scalars([info["dist"]]))
    return bool(info["success"]), episode_return


# Held-out by default (base 20000, disjoint from demo seeds 0..num_demos-1);
# --break train_dist evaluates on the demo seeds themselves (in-distribution).
eval_base = args.seed if args.break_mode == "train_dist" else HELD_OUT_BASE + args.seed
outcomes = [rollout(eval_base + ep, ep) for ep in range(args.eval_episodes)]
successes = sum(s for s, _ in outcomes)
official_success = successes / args.eval_episodes
official_return = float(np.mean([r for _, r in outcomes]))
lo, hi = wilson_ci(successes, args.eval_episodes)
if args.rerun:
    rr.log("eval/official/success_rate", rr.Scalars([official_success]))
print(f"official ACT:  success {official_success:.2f}  [{lo:.2f}, {hi:.2f}]  "
      f"return {official_return:.3f}  ({'TRAIN-DIST' if args.break_mode else 'held-out'} "
      f"seeds, n={args.eval_episodes})")

# The from-scratch baseline: run 1.3's act.py as a subprocess on the SAME dataset
# and seed, then read its held-out metrics.json. Same env, same seed formula
# (base 20000), same episode count => an apples-to-apples comparison.
scratch = None
if args.compare and ACT_PY.is_file():
    fs_out = args.out / "from_scratch"
    cmd = [sys.executable, str(ACT_PY), "--data", str(args.data), "--seed", str(args.seed),
           "--device", "cpu", "--no-rerun", "--out", str(fs_out)]
    cmd += ["--smoke"] if args.smoke else [
        "--chunk_size", str(K), "--model_dim", str(args.model_dim),
        "--num_demos", str(args.num_demos), "--epochs", str(args.epochs),
        "--eval_episodes", str(args.eval_episodes)]
    print("\n[compare] training the from-scratch 1.3 ACT on the same data ...")
    subprocess.run(cmd, check=True, cwd=ROOT)
    scratch = json.loads((fs_out / "metrics.json").read_text())
    print(f"from-scratch 1.3 ACT:  success {scratch['success_rate']:.2f}  "
          f"return {scratch['mean_return']:.3f}  (held-out seeds, n={args.eval_episodes})")
    # Read the BARS, not the point estimates (1.6): is the from-scratch rate inside
    # the official Wilson interval? Inside => indistinguishable at this n; outside =>
    # a real gap. Either way the durable headline is the CODE ratio, not a winner.
    inside = lo <= scratch["success_rate"] <= hi
    verdict = ("their intervals overlap — indistinguishable at this n"
               if inside else "the from-scratch rate is OUTSIDE the official CI — a real gap at this n")
    print(f"compare: {verdict}. The headline is the CODE ratio (~380 lines by hand "
          f"vs ~20 official), not the winner.")
# --- endregion ---

# --- region: publish ---
# HUMAN-GATED. Offline or without a token this DRY-RUNS: it prints the exact
# push_to_hub calls and serializes the policy locally (proving the save path)
# WITHOUT any network. A real publish needs `huggingface-cli login` (HF_TOKEN)
# and HF_HUB_OFFLINE unset — never CI, never the default path.
import os  # noqa: E402

local_model = args.out / "lerobot_act"
have_token = bool(os.environ.get("HF_TOKEN")) and os.environ.get("HF_HUB_OFFLINE") not in ("1", "true")
if args.publish and have_token:
    policy.push_to_hub(args.repo_id)          # real push (needs network + token)
    dataset.push_to_hub(f"{args.repo_id}_dataset")
    print(f"published policy -> hf.co/{args.repo_id} and dataset -> {args.repo_id}_dataset")
else:
    policy.save_pretrained(local_model)       # local serialization, always offline-safe
    reason = "no --publish" if not args.publish else "offline / no HF_TOKEN"
    print(f"\n[publish DRY-RUN — {reason}] saved policy to {local_model}. To publish for real:")
    print("    huggingface-cli login   # sets HF_TOKEN")
    print(f"    policy.push_to_hub({args.repo_id!r})")
    print(f"    dataset.push_to_hub({args.repo_id + '_dataset'!r})")
# --- endregion ---

# --- region: report ---
metrics = {
    "break_mode": args.break_mode or "none",
    "chunk_size": K,
    "epochs": args.epochs,
    "eval_episodes": args.eval_episodes,
    "num_demos": ds_meta.total_episodes,
    "official_ci_hi": round(hi, 6),
    "official_ci_lo": round(lo, 6),
    "official_final_train_loss": round(train_loss, 6),
    "official_mean_return": round(official_return, 6),
    "official_params": n_params,
    "official_success_rate": round(official_success, 6),
    "scratch_mean_return": round(scratch["mean_return"], 6) if scratch else None,
    "scratch_success_rate": round(scratch["success_rate"], 6) if scratch else None,
    "seed": args.seed,
    "smoke": bool(args.smoke),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"\nmetrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'bridge.rrd'} — open it with: rerun {args.out / 'bridge.rrd'}")
# --- endregion ---
