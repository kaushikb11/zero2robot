"""zero2robot 3.1 — World Models I: Learning the Simulator.

For thirty chapters you trusted `mj_step`: hand the sim an action, it hands back
the next state. A world model LEARNS that map from data — the model that dreams
the next frame instead of simulating it. Here we build a tiny Dreamer-style
latent dynamics model from scratch (torch + numpy, no libraries) and make one
distinction the whole field turns on:

  RECONSTRUCTION — decode a latent that HAS SEEN the observation. Easy: it is an
    autoencoder passing state through a recurrent bottleneck. Low error, always.
  PREDICTION    — roll the latent dynamics FORWARD on actions alone, never seeing
    the future observation, then decode. THIS is "did it learn the simulator?".
    The honest test: does k-step prediction beat COPY-LAST (assume nothing moves)?
    At k=1 copy-last wins — nothing moved yet. As the horizon grows, copy-last
    rots while the model, integrating the actions through its dynamics, holds
    lower error. But WHERE does that gap come from? The eval prints the honest
    breakdown: the win is the PUSHER (whose next position is a trivial integral of
    the commanded velocity); on the OBJECT dims (the T-block's contact dynamics —
    what PushT is about) this tiny model does NOT yet beat copy-last. It learned
    the easy half. That split — easy kinematics won, hard dynamics not yet — is
    the honest lesson, and the reason 3.2 and the pixel Scale Lab exist.

WHY STATE, NOT PIXELS (the honest free-tier call — the map calls this "the chapter
about why world models eat compute, deliberately small"). We ran the feasibility
spike on 32x32 PushT pixels: rendering is cheap and RECONSTRUCTION is beautiful,
but pixel-MSE prediction can NOT beat copy-last at a tiny model's scale — a T and
a pusher are a few pixels on a static background, so copy-last's error is tiny and
the model's reconstruction blur floor never dips below it (measured: no crossover
through k=22). Beating it needs a sharper decoder = more compute = not free-tier.
That IS the "world models eat compute" wall, and it is the Scale Lab. On the
low-dim STATE the same recipe wins in aggregate (measured ~2.3x lower than
copy-last, 2.18-2.45x across seeds 0-2) — though, honestly, that win is the pusher
kinematics, not the block dynamics (see the eval breakdown). The MECHANISM —
encoder, latent dynamics, decoder, prior-vs-posterior — is what we teach here,
honestly and in seconds. See meta.yaml `scale_lab` and "What we cut".

Run it:      python curriculum/phase3_advanced/ch3.1_world_models/wm.py --seed 0
Peek-cheat:  python curriculum/phase3_advanced/ch3.1_world_models/wm.py --seed 0 --break peek
CI smoke:    python curriculum/phase3_advanced/ch3.1_world_models/wm.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch1.4's diffusion.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv  # noqa: E402
from curriculum.common.envs.pusht.scripted_expert import ScriptedExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

OBS_DIM, ACT_DIM = PushTEnv.OBS_DIM, PushTEnv.ACT_DIM
EVAL_SEED_BASE = 10_000     # val episodes drawn from here — disjoint from train seeds (0..episodes)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch3.1-world-models"))
# Scale knobs, nanoGPT-style, free-tier default first. (--resolution belongs to the
# PIXEL Scale Lab — see the header; the free-tier chapter models the low-dim state.)
parser.add_argument("--latent_dim", type=int, default=16)   # stochastic latent width. T4: 32 | smoke: 4
parser.add_argument("--hidden_dim", type=int, default=128)  # deterministic GRU state. T4: 256 | smoke: 16
parser.add_argument("--embed_dim", type=int, default=64)    # encoder output width. T4: 128 | smoke: 8
parser.add_argument("--seq_len", type=int, default=24)      # training sequence length (steps). T4: 32 | smoke: 8
parser.add_argument("--episodes", type=int, default=120)    # PushT sequences collected. T4: 400 | smoke: 6
parser.add_argument("--epochs", type=int, default=60)       # cpu-laptop: ~1-2 min | smoke: 3
parser.add_argument("--batch_size", type=int, default=30)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--horizon", type=int, default=12, help="k: open-loop prediction steps evaluated vs copy-last")
parser.add_argument("--context", type=int, default=3, help="frames observed to warm the state before predicting")
parser.add_argument("--noise", type=float, default=0.5, help="scripted-expert exploration noise (m/s) — dataset diversity")
parser.add_argument("--val_episodes", type=int, default=30)  # held-out sequences for recon/prediction (ch1.6: few = noisy)
parser.add_argument("--seed", type=int, default=0, help="seeds demos, inits, batch order — CPU run is byte-reproducible")
parser.add_argument("--break", dest="break_mode", choices=("peek",), default=None,
                    help="Break It: 'peek' lets the prediction rollout SEE each true frame (posterior, not prior) — "
                         "the classic world-model eval bug where 'prediction' is secretly reconstruction")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())  # cpu: bitwise-reproducible
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch globals (model inits draw from these)
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.latent_dim, args.hidden_dim, args.embed_dim = 4, 16, 8
    args.seq_len, args.episodes, args.epochs = 8, 6, 3
    args.horizon, args.context, args.val_episodes, args.device = 4, 2, 3, "cpu"
banner("ch3.1-world-models", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
# One CPU generator feeds batch order; model init draws from torch's global RNG
# (seeded above). Same seed, cpu => byte-identical run (root CLAUDE.md invariant 2).
shuffle_gen = torch.Generator().manual_seed(args.seed + 1)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch3.1-world-models", spawn=False)
    rr.save(str(args.out / "world_models.rrd"))
# --- endregion ---

# --- region: data ---
# The world model trains on SEQUENCES, not single steps: it must learn how one
# state flows into the next under an action, so the data is (state_0..T, action_0..T-1)
# rollouts. We reuse the ch1.1 PushT env and the scripted expert (with exploration
# noise for coverage) — the SAME simulator we are about to learn to imitate.


def collect(num, seed_base):
    """Roll `num` scripted-expert episodes, each cropped to seq_len+1 states and
    seq_len actions. Deterministic: episode i uses env+expert seed (seed_base + i),
    so the whole dataset is reproducible from the CLI args alone (like gen_demos)."""
    states, actions = [], []
    env = PushTEnv()
    need = args.seq_len + 1
    i = 0
    while len(states) < num:
        seed = seed_base + i
        i += 1
        obs = env.reset(seed)
        expert = ScriptedExpert(noise=args.noise, seed=seed)
        obs_seq, act_seq, done = [obs], [], False
        while not done and len(obs_seq) < need:
            action = expert.action(env)
            obs, _, done, _ = env.step(action)
            obs_seq.append(obs)
            act_seq.append(action)
        if len(obs_seq) == need:  # keep only full-length sequences (short successes dropped)
            states.append(np.stack(obs_seq))
            actions.append(np.stack(act_seq))
    return np.stack(states).astype(np.float32), np.stack(actions).astype(np.float32)


train_states, train_actions = collect(args.episodes, seed_base=0)
val_states, val_actions = collect(args.val_episodes, seed_base=EVAL_SEED_BASE + args.seed)

# Standardize states to zero-mean/unit-std so every dim carries comparable weight in
# the MSE (constant dims — the fixed target pose — get std 1, never divided to inf).
# The sampler and losses live in this normalized space; we only de-normalize to report.
obs_mean = train_states.reshape(-1, OBS_DIM).mean(0)
obs_std = train_states.reshape(-1, OBS_DIM).std(0)
obs_std = np.where(obs_std < 1e-4, np.float32(1.0), obs_std)


def to_tensor(states, actions):
    norm = (states - obs_mean) / obs_std
    return torch.from_numpy(norm).to(device), torch.from_numpy(actions).to(device)


train_obs, train_act = to_tensor(train_states, train_actions)
val_obs, val_act = to_tensor(val_states, val_actions)
print(f"dataset: {len(train_obs)} train / {len(val_obs)} val sequences of {args.seq_len} steps "
      f"(latent_dim={args.latent_dim}, hidden_dim={args.hidden_dim})")
# --- endregion ---


# --- region: model ---
# A deterministic-latent RSSM-lite (the skeleton of Dreamer's world model, minus
# the stochastic-KL machinery — see "What we cut"). Four learned pieces:
#   encoder  obs_t         -> embed_t              (features from an observation)
#   gru      (z,a), h_{t-1} -> h_t                 (the DETERMINISTIC recurrent state)
#   posterior [h_t, embed_t] -> z_t                (latent that HAS SEEN obs_t: for reconstruction)
#   prior     [h_t]          -> zhat_t             (latent PREDICTED from h_t alone: for prediction)
#   decoder  [h_t, z_t]    -> obs_t                (reconstruct the observation)
# The prior/posterior split is the whole lesson: at train time the posterior teaches
# the decoder to reconstruct; the prior learns to MATCH the posterior WITHOUT the
# observation, so at test time we can roll the prior forward on actions alone.


class WorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, latent_dim, hidden_dim, embed_dim):
        super().__init__()
        self.latent_dim, self.hidden_dim = latent_dim, hidden_dim
        self.encoder = nn.Sequential(nn.Linear(obs_dim, embed_dim), nn.ReLU(),
                                     nn.Linear(embed_dim, embed_dim))
        self.gru = nn.GRUCell(latent_dim + act_dim, hidden_dim)
        self.posterior = nn.Sequential(nn.Linear(hidden_dim + embed_dim, hidden_dim), nn.ReLU(),
                                       nn.Linear(hidden_dim, latent_dim))
        self.prior = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(hidden_dim + latent_dim, embed_dim), nn.ReLU(),
                                     nn.Linear(embed_dim, obs_dim))

    def _init(self, batch):
        h = torch.zeros(batch, self.hidden_dim, device=next(self.parameters()).device)
        z = torch.zeros(batch, self.latent_dim, device=h.device)
        return h, z

    def observe(self, obs_seq, act_seq):
        """Teacher-forced pass over a whole sequence. At each step advance the GRU,
        form the posterior latent from (state, observation), and decode. Returns the
        reconstruction loss and the DYNAMICS loss (prior matching posterior) — the
        two terms that, minimized together, make the prior a one-step simulator."""
        batch, steps = obs_seq.shape[0], obs_seq.shape[1]
        embed = self.encoder(obs_seq)                          # (B, T, embed) — encode every frame at once
        h, z = self._init(batch)
        recon_loss, dyn_loss = 0.0, 0.0
        for t in range(steps):
            if t > 0:  # advance the deterministic state with the PREVIOUS latent + action
                h = self.gru(torch.cat([z, act_seq[:, t - 1]], dim=-1), h)
            zhat = self.prior(h)                               # predicted latent (no obs)
            z = self.posterior(torch.cat([h, embed[:, t]], dim=-1))  # corrected latent (sees obs)
            recon = self.decoder(torch.cat([h, z], dim=-1))
            recon_loss = recon_loss + F.mse_loss(recon, obs_seq[:, t])
            if t > 0:  # train the prior to hit the posterior it can't see (detached target)
                dyn_loss = dyn_loss + F.mse_loss(zhat, z.detach())
        return recon_loss / steps, dyn_loss / max(steps - 1, 1)

    @torch.no_grad()
    def warm_state(self, obs_seq, act_seq):
        """Filter through `context` observed steps to get (h, z) at the last one —
        the state the model predicts FROM. Uses the posterior: warmup gets to see."""
        batch = obs_seq.shape[0]
        embed = self.encoder(obs_seq)
        h, z = self._init(batch)
        for t in range(obs_seq.shape[1]):
            if t > 0:
                h = self.gru(torch.cat([z, act_seq[:, t - 1]], dim=-1), h)
            z = self.posterior(torch.cat([h, embed[:, t]], dim=-1))
        return h, z

    @torch.no_grad()
    def imagine(self, h, z, future_actions, true_future=None):
        """Open-loop rollout: step the GRU on actions alone and take the PRIOR latent
        each step (no observation) — dreaming forward. Decode to predicted states.
        `true_future` is only read under --break peek, where we illegally re-filter
        the posterior on the real frame: 'prediction' that secretly reconstructs."""
        preds = []
        for k in range(future_actions.shape[1]):
            h = self.gru(torch.cat([z, future_actions[:, k]], dim=-1), h)
            if true_future is not None:  # BROKEN: peek at the frame we are meant to predict
                z = self.posterior(torch.cat([h, self.encoder(true_future[:, k])], dim=-1))
            else:
                z = self.prior(h)
            preds.append(self.decoder(torch.cat([h, z], dim=-1)))
        return torch.stack(preds, dim=1)                       # (B, K, obs_dim)
# --- endregion ---


# --- region: train ---
# One loss, two terms: reconstruction (the decoder must recreate the state the
# posterior saw) + dynamics (the prior must predict that posterior latent from the
# recurrent state alone). The dynamics term is what turns an autoencoder into a
# simulator — without it the prior is untrained and prediction is noise.
torch.manual_seed(args.seed)  # model init reproducible, independent of data collection above
model = WorldModel(OBS_DIM, ACT_DIM, args.latent_dim, args.hidden_dim, args.embed_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
recon_hist, global_step = float("nan"), 0
for epoch in range(args.epochs):
    epoch_recon, epoch_dyn, num_batches = 0.0, 0.0, 0
    for batch in torch.randperm(len(train_obs), generator=shuffle_gen).split(args.batch_size):
        recon_loss, dyn_loss = model.observe(train_obs[batch], train_act[batch])
        optimizer.zero_grad()
        (recon_loss + dyn_loss).backward()
        optimizer.step()
        epoch_recon += recon_loss.item()
        epoch_dyn += dyn_loss.item()
        num_batches += 1
        if args.rerun:
            rr.set_time("step", sequence=global_step)
            rr.log("train/recon", rr.Scalars([recon_loss.item()]))
            rr.log("train/dyn", rr.Scalars([dyn_loss.item()]))
        global_step += 1
    recon_hist = epoch_recon / num_batches
    if epoch % 15 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:3d}  recon {recon_hist:.5f}  dyn {epoch_dyn / num_batches:.5f}")
# --- endregion ---


# --- region: eval ---
# The headline. RECONSTRUCTION: decode the posterior on val states — the model
# recreating what it sees (its floor). PREDICTION: warm the state on `context`
# frames, then imagine `horizon` steps on actions alone and score each k against
# COPY-LAST (repeat the last observed state). Copy-last is a strong, honest baseline
# on slow dynamics: at k=1 almost nothing moved, so it wins. The world model wins
# once the horizon is long enough that integrating the actions beats assuming
# stasis — the CROSSOVER, and the widening gap after it, is "it learned to step."
model.eval()
with torch.no_grad():
    val_recon, val_dyn = model.observe(val_obs, val_act)
    val_recon = float(val_recon)

ctx_obs = val_obs[:, : args.context + 1]                       # frames 0..context (observed)
ctx_act = val_act[:, : args.context]                           # actions between them
future_act = val_act[:, args.context : args.context + args.horizon]
future_obs = val_obs[:, args.context + 1 : args.context + 1 + args.horizon]  # ground truth to predict
last_obs = val_obs[:, args.context]                            # copy-last baseline state

h0, z0 = model.warm_state(ctx_obs, ctx_act)
peek = future_obs if args.break_mode == "peek" else None       # --break peek: illegal re-filtering
pred = model.imagine(h0, z0, future_act, true_future=peek)     # (B, K, obs_dim)

wm_err = ((pred - future_obs) ** 2).mean(dim=(0, 2)).cpu().numpy()          # per-k world-model MSE
copy_err = ((last_obs[:, None] - future_obs) ** 2).mean(dim=(0, 2)).cpu().numpy()  # per-k copy-last MSE
crossover_k = next((k + 1 for k in range(args.horizon) if wm_err[k] < copy_err[k]), 0)
wm_mean, copy_mean = float(wm_err.mean()), float(copy_err.mean())

# WHERE the win lives — the honest breakdown. Split obs into the fast PUSHER dims
# (whose next state is a trivial first-order integral of the commanded velocity) and
# the OBJECT dims (the T-block pose — the actual contact/pushing dynamics PushT is
# ABOUT). The aggregate win is carried by the pusher; the object dynamics are the hard
# half this tiny free-tier model does NOT yet beat copy-last on. The chapter must say so.
PUSHER_DIMS, OBJECT_DIMS = [0, 1], [2, 3, 4, 5]   # pusht obs layout: pusher xy | tee xy + yaw(sin,cos)
def _mse(dims):  # (world-model, copy-last) mean MSE over the given obs dims + all k
    w = ((pred[..., dims] - future_obs[..., dims]) ** 2).mean().item()
    c = ((last_obs[:, None][..., dims] - future_obs[..., dims]) ** 2).mean().item()
    return w, c
wm_push, copy_push = _mse(PUSHER_DIMS)
wm_obj, copy_obj = _mse(OBJECT_DIMS)

print(f"reconstruction (posterior, sees the frame): val recon {val_recon:.5f}")
print("prediction (prior rollout, actions only) vs copy-last, per horizon k:")
for k in range(args.horizon):
    flag = "  <- world model wins" if wm_err[k] < copy_err[k] else ""
    print(f"  k={k + 1:2d}  world_model {wm_err[k]:.5f}  copy_last {copy_err[k]:.5f}{flag}")
    if args.rerun:
        rr.set_time("horizon", sequence=k + 1)
        rr.log("eval/pred/world_model", rr.Scalars([float(wm_err[k])]))
        rr.log("eval/pred/copy_last", rr.Scalars([float(copy_err[k])]))
print(f"crossover at k={crossover_k or '>horizon'} | mean world_model {wm_mean:.5f} vs "
      f"copy_last {copy_mean:.5f} ({copy_mean / max(wm_mean, 1e-9):.2f}x lower error)")
print(f"  where the win lives: PUSHER dims wm {wm_push:.5f} vs copy {copy_push:.5f} "
      f"({copy_push / max(wm_push, 1e-9):.1f}x lower — learned to integrate the commanded velocity) | "
      f"OBJECT dims wm {wm_obj:.5f} vs copy {copy_obj:.5f} "
      + ("(wm wins)" if wm_obj < copy_obj else "(COPY-LAST WINS — the block/contact dynamics are the hard "
         "half this tiny model does not yet beat; that gap is the honest lesson, not a failure)"))

if args.rerun:  # a readable rollout: predicted vs true pusher_x on one val sequence (de-normalized)
    px_pred = pred[0, :, 0].cpu().numpy() * obs_std[0] + obs_mean[0]
    px_true = future_obs[0, :, 0].cpu().numpy() * obs_std[0] + obs_mean[0]
    for k in range(args.horizon):
        rr.set_time("horizon", sequence=k + 1)
        rr.log("eval/rollout/pusher_x_predicted", rr.Scalars([float(px_pred[k])]))
        rr.log("eval/rollout/pusher_x_true", rr.Scalars([float(px_true[k])]))
# --- endregion ---


# --- region: report ---
metrics = {
    "break_mode": args.break_mode or "none",
    "context": args.context,
    "copy_last_pred_mean": round(copy_mean, 6),
    "crossover_k": crossover_k,
    "epochs": args.epochs,
    "final_train_recon": round(recon_hist, 6),
    "hidden_dim": args.hidden_dim,
    "horizon": args.horizon,
    "latent_dim": args.latent_dim,
    "num_train_sequences": len(train_obs),
    "pred_ratio_copy_over_wm": round(copy_mean / max(wm_mean, 1e-9), 6),
    "seed": args.seed,
    "seq_len": args.seq_len,
    "smoke": bool(args.smoke),
    "val_recon": round(val_recon, 6),
    "world_model_pred_mean": round(wm_mean, 6),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}  (world model predicts {metrics['pred_ratio_copy_over_wm']}x "
      f"lower than copy-last; crossover k={crossover_k or '>horizon'})")
if args.rerun:
    print(f"recording: {args.out / 'world_models.rrd'} — open it with: rerun {args.out / 'world_models.rrd'}")
# --- endregion ---
