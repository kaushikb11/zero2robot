"""zero2robot 3.2 — World Models II: Acting in Imagination.

In 3.1 you LEARNED a simulator: an encoder/GRU/prior/decoder that steps forward
on actions alone instead of calling mj_step. It learned the EASY half (the pusher
kinematics) and NOT the hard half (the T-block's contact dynamics — copy-last
still beat it there). Hold onto that finding; it is the spine of this chapter.

Now we do the Dreamer move: learn a POLICY *inside* the learned model. We freeze
the world model, then train an actor (+ a critic) purely on IMAGINED rollouts —
roll the prior forward, decode each dreamed state, score its reward, and
backpropagate that imagined return into the actor. The real PushT sim is NEVER
touched during policy learning. That is the whole promise of model-based RL: if
you trust your dream, you can learn to act without paying for real experience.

THE HONEST QUESTION this chapter is built to measure — the IMAGINATION GAP.
The reward here depends on the BLOCK pose (get the T to the target). But 3.1's
model got the block dynamics WRONG. So the actor optimizes a reward computed in a
dream whose hard half is hallucinated: it can look like a champion in imagination
and then FAIL in the real sim, because the imagined block moved and the real one
did not. We deploy the SAME policy in both and print the gap. Whatever it is, it
is the lesson: imagination is only as good as your world model.

Run it:      python curriculum/phase3_advanced/ch3.2_dreamer/dreamer.py --seed 0
CI smoke:    python curriculum/phase3_advanced/ch3.2_dreamer/dreamer.py --smoke --seed 0 --no-rerun
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
# sys.path so `curriculum.common` resolves (same pattern as ch3.1's wm.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.pusht import PushTEnv  # noqa: E402
from curriculum.common.envs.pusht.scripted_expert import ScriptedExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

OBS_DIM, ACT_DIM = PushTEnv.OBS_DIM, PushTEnv.ACT_DIM
EVAL_SEED_BASE = 10_000        # eval starts drawn here — disjoint from train seeds (0..episodes)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch3.2-dreamer"))
# World-model knobs (same skeleton as 3.1). Free-tier default first, nanoGPT-style.
parser.add_argument("--latent_dim", type=int, default=16)   # stochastic-latent width. T4: 32 | smoke: 4
parser.add_argument("--hidden_dim", type=int, default=128)  # GRU deterministic state. T4: 256 | smoke: 16
parser.add_argument("--embed_dim", type=int, default=64)    # encoder width. T4: 128 | smoke: 8
parser.add_argument("--seq_len", type=int, default=24)      # training sequence length. T4: 32 | smoke: 8
parser.add_argument("--episodes", type=int, default=120)    # PushT sequences collected. T4: 400 | smoke: 6
parser.add_argument("--wm_epochs", type=int, default=60)    # world-model epochs. cpu: ~0.4 min | smoke: 3
parser.add_argument("--batch_size", type=int, default=30)
parser.add_argument("--wm_lr", type=float, default=1e-3)
parser.add_argument("--context", type=int, default=3, help="frames observed to warm (h,z) before imagining")
parser.add_argument("--noise", type=float, default=0.5, help="scripted-expert exploration noise — dataset coverage")
# Actor-in-imagination knobs.
parser.add_argument("--imag_horizon", type=int, default=15)  # steps dreamt per policy update. T4: 15 | smoke: 4
parser.add_argument("--imag_iters", type=int, default=400)   # policy gradient steps. cpu: ~0.4 min | smoke: 3
parser.add_argument("--imag_batch", type=int, default=64)    # imagined trajectories per update. T4: 256 | smoke: 8
parser.add_argument("--actor_lr", type=float, default=1e-4,
                    help="small on purpose: the analytic dynamics-gradient through a frozen GRU is high-variance")
parser.add_argument("--gamma", type=float, default=0.99, help="imagined-return discount")
parser.add_argument("--lam", type=float, default=0.95, help="lambda-return bias/variance knob (Dreamer)")
parser.add_argument("--ent_coef", type=float, default=1e-3, help="entropy bonus — keep the imagined policy exploring")
parser.add_argument("--eval_episodes", type=int, default=30)  # held-out start states. T4: 30 | smoke: 3
parser.add_argument("--eval_horizon", type=int, default=40)   # steps rolled in imagination AND reality. smoke: 6
parser.add_argument("--seed", type=int, default=0, help="seeds demos, inits, batch order — CPU run is byte-reproducible")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())  # cpu: bitwise-reproducible
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch globals (model inits draw from these)
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.latent_dim, args.hidden_dim, args.embed_dim = 4, 16, 8
    args.seq_len, args.episodes, args.wm_epochs = 8, 6, 3
    args.imag_horizon, args.imag_iters, args.imag_batch = 4, 3, 8
    args.context, args.eval_episodes, args.eval_horizon, args.device = 2, 3, 6, "cpu"
banner("ch3.2-dreamer", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
shuffle_gen = torch.Generator().manual_seed(args.seed + 1)  # feeds WM batch order (cpu => byte-identical)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch3.2-dreamer", spawn=False)
    rr.save(str(args.out / "dreamer.rrd"))
# --- endregion ---

# --- region: data ---
# Same data as 3.1: the world model trains on SEQUENCES (state_0..T, action_0..T-1)
# rolled from the scripted expert (with exploration noise for coverage) on the SAME
# PushT sim we are learning to dream. The actor never sees this data directly — it
# only ever trains on trajectories the FROZEN world model imagines.


def collect(num, seed_base):
    """Roll `num` scripted-expert episodes, cropped to seq_len+1 states / seq_len
    actions. Deterministic: episode i uses env+expert seed (seed_base + i)."""
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
        if len(obs_seq) == need:
            states.append(np.stack(obs_seq))
            actions.append(np.stack(act_seq))
    return np.stack(states).astype(np.float32), np.stack(actions).astype(np.float32)


train_states, train_actions = collect(args.episodes, seed_base=0)

# Standardize states (zero-mean/unit-std); constant dims (the fixed target pose) get
# std 1. The model, the losses, and the imagined reward all live in this normalized
# space; we de-normalize only to compute the physical reward and to report.
obs_mean = train_states.reshape(-1, OBS_DIM).mean(0)
obs_std = train_states.reshape(-1, OBS_DIM).std(0)
obs_std = np.where(obs_std < 1e-4, np.float32(1.0), obs_std)
mean_t = torch.tensor(obs_mean, device=device)
std_t = torch.tensor(obs_std, device=device)

train_obs = torch.from_numpy((train_states - obs_mean) / obs_std).to(device)
train_act = torch.from_numpy(train_actions).to(device)
print(f"dataset: {len(train_obs)} sequences of {args.seq_len} steps "
      f"(latent_dim={args.latent_dim}, hidden_dim={args.hidden_dim})")
# --- endregion ---


# --- region: model ---
# The 3.1 world model, unchanged in spirit: a deterministic-latent RSSM-lite.
#   encoder  obs      -> embed
#   gru      (z,a),h  -> h        (deterministic recurrent state)
#   posterior [h,embed] -> z      (latent that HAS SEEN obs — filtering)
#   prior     [h]       -> zhat   (latent PREDICTED from h alone — dreaming)
#   decoder   [h,z]     -> obs
# `step()` is the one new affordance 3.2 leans on: advance the state one tick on an
# action, take the PRIOR latent (no observation), decode. That is one tick of a dream.


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
        return h, torch.zeros(batch, self.latent_dim, device=h.device)

    def observe(self, obs_seq, act_seq):
        """Teacher-forced pass: reconstruction loss + dynamics loss (prior matching
        the detached posterior). Minimized together, they make the prior a one-step
        simulator — exactly as in 3.1. This is all the world model ever trains on."""
        batch, steps = obs_seq.shape[0], obs_seq.shape[1]
        embed = self.encoder(obs_seq)
        h, z = self._init(batch)
        recon_loss, dyn_loss = 0.0, 0.0
        for t in range(steps):
            if t > 0:
                h = self.gru(torch.cat([z, act_seq[:, t - 1]], dim=-1), h)
            zhat = self.prior(h)
            z = self.posterior(torch.cat([h, embed[:, t]], dim=-1))
            recon = self.decoder(torch.cat([h, z], dim=-1))
            recon_loss = recon_loss + F.mse_loss(recon, obs_seq[:, t])
            if t > 0:
                dyn_loss = dyn_loss + F.mse_loss(zhat, z.detach())
        return recon_loss / steps, dyn_loss / max(steps - 1, 1)

    def warm(self, obs_seq, act_seq):
        """Filter through observed frames to the last one's (h, z) — the state a
        dream (or a real deployment) starts FROM. Uses the posterior: warmup sees."""
        embed = self.encoder(obs_seq)
        h, z = self._init(obs_seq.shape[0])
        for t in range(obs_seq.shape[1]):
            if t > 0:
                h = self.gru(torch.cat([z, act_seq[:, t - 1]], dim=-1), h)
            z = self.posterior(torch.cat([h, embed[:, t]], dim=-1))
        return h, z

    def step(self, h, z, action):
        """One tick of a dream: advance the GRU on the action, take the PRIOR latent
        (no observation), decode. Differentiable in `action` — that is what lets the
        imagined return flow a gradient back into the actor."""
        h = self.gru(torch.cat([z, action], dim=-1), h)
        z = self.prior(h)
        return h, z, self.decoder(torch.cat([h, z], dim=-1))
# --- endregion ---


# --- region: reward ---
# The reward is a KNOWN function of the state (PushT: drive the T-block to the
# target pose), so we do NOT learn a reward head — we compute it analytically from
# the DECODED observation. That is the honest crux: in imagination this reads the
# block pose the world model HALLUCINATED, the very dims 3.1 measured it gets wrong.
# The same function scores the real trajectory in eval, so the gap is pure state
# divergence, not two different reward definitions.
POS_SCALE, ANG_SCALE = 0.5, float(np.pi)   # from PushTEnv.step: reward = -0.5*(pos_err/0.5 + ang_err/pi)


def reward_from_obs(norm_obs):
    """Shaped PushT reward from a normalized observation (any leading batch shape).
    De-normalize, then read tee_xy and tee_yaw (target is the fixed origin)."""
    obs = norm_obs * std_t + mean_t
    pos_err = torch.linalg.vector_norm(obs[..., 2:4], dim=-1)          # ||tee_xy - target(0,0)||
    ang_err = torch.abs(torch.atan2(obs[..., 4], obs[..., 5]))         # |wrap(tee_yaw - 0)|
    return -0.5 * (pos_err / POS_SCALE + ang_err / ANG_SCALE), pos_err, ang_err
# --- endregion ---


# --- region: agent ---
# The actor and critic both read the world-model STATE [h, z] — the same features
# available in a dream AND on the real robot (there we recover [h,z] by filtering
# the posterior on real frames). A tanh-squashed Gaussian keeps actions in the
# env's [-1, 1] and stays reparameterized, so the imagined return differentiates
# straight through the (frozen) dynamics into the policy — Dreamer's analytic
# actor gradient, no REINFORCE estimator needed at this scale.


class ActorCritic(nn.Module):
    def __init__(self, state_dim, act_dim, hidden_dim):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                   nn.Linear(hidden_dim, act_dim))
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Sequential(nn.Linear(state_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                                    nn.Linear(hidden_dim, 1))

    def act(self, state, sample=True):
        """Return (action in [-1,1], base-Gaussian entropy). sample=False acts on the
        mean — the deterministic policy used for BOTH imagined and real eval."""
        mean = self.actor(state)
        std = torch.exp(self.log_std)
        pre = mean + std * torch.randn_like(mean) if sample else mean
        entropy = (0.5 + 0.5 * np.log(2 * np.pi) + self.log_std).sum()
        return torch.tanh(pre), entropy

    def value(self, state):
        return self.critic(state).squeeze(-1)
# --- endregion ---


# --- region: train_wm ---
# Step 1: learn the simulator, exactly as in 3.1. Reconstruction + dynamics loss.
torch.manual_seed(args.seed)  # model init reproducible, independent of data collection
wm = WorldModel(OBS_DIM, ACT_DIM, args.latent_dim, args.hidden_dim, args.embed_dim).to(device)
wm_opt = torch.optim.Adam(wm.parameters(), lr=args.wm_lr)
global_step = 0
for epoch in range(args.wm_epochs):
    epoch_recon, num_batches = 0.0, 0
    for batch in torch.randperm(len(train_obs), generator=shuffle_gen).split(args.batch_size):
        recon_loss, dyn_loss = wm.observe(train_obs[batch], train_act[batch])
        wm_opt.zero_grad()
        (recon_loss + dyn_loss).backward()
        wm_opt.step()
        epoch_recon += recon_loss.item()
        num_batches += 1
        if args.rerun:
            rr.set_time("step", sequence=global_step)
            rr.log("wm/recon", rr.Scalars([recon_loss.item()]))
            rr.log("wm/dyn", rr.Scalars([dyn_loss.item()]))
        global_step += 1
    if epoch % 15 == 0 or epoch == args.wm_epochs - 1:
        print(f"[wm] epoch {epoch:3d}  recon {epoch_recon / num_batches:.5f}")
# --- endregion ---


# --- region: train_actor ---
# Step 2: FREEZE the world model and learn a policy INSIDE it. Every gradient the
# actor sees comes from a dream — start from real filtered states, then roll the
# frozen prior forward under the actor's own actions, decode, and score the reward.
# The lambda-return (Dreamer) mixes the imagined rewards with the critic's bootstrap;
# the actor maximizes it by analytic gradient, the critic regresses onto it.
for p in wm.parameters():
    p.requires_grad_(False)   # frozen: gradients flow THROUGH it to the actions, never INTO it
state_dim = args.hidden_dim + args.latent_dim
agent = ActorCritic(state_dim, ACT_DIM, args.hidden_dim).to(device)
agent_opt = torch.optim.Adam(agent.parameters(), lr=args.actor_lr)
ctx_obs = train_obs[:, : args.context + 1]     # frames to warm the start state from
ctx_act = train_act[:, : args.context]


def lambda_return(rewards, values):
    """Dreamer's lambda-return: G_t = r_t + gamma*((1-lam)*V_{t+1} + lam*G_{t+1}),
    bootstrapped at the horizon by the critic. rewards: (H,B); values: (H+1,B)."""
    out, last = [], values[-1]
    for t in reversed(range(len(rewards))):
        last = rewards[t] + args.gamma * ((1 - args.lam) * values[t + 1] + args.lam * last)
        out.append(last)
    return torch.stack(out[::-1])   # (H, B)


imag_reward_hist = float("nan")
for it in range(args.imag_iters):
    idx = torch.randint(len(train_obs), (args.imag_batch,), generator=shuffle_gen)
    with torch.no_grad():
        h, z = wm.warm(ctx_obs[idx], ctx_act[idx])   # real start states; detached from the actor graph
    states, rewards, entropies = [torch.cat([h, z], dim=-1)], [], []
    for _ in range(args.imag_horizon):
        action, entropy = agent.act(states[-1], sample=True)
        h, z, obs_pred = wm.step(h, z, action)
        rewards.append(reward_from_obs(obs_pred)[0])
        entropies.append(entropy)
        states.append(torch.cat([h, z], dim=-1))
    # Stop-gradients keep the two objectives from cross-contaminating (canonical Dreamer):
    # the critic regresses V(state) onto the return, so its input states are detached (it
    # must not reshape the actor); the actor maximizes the return, so the critic bootstrap
    # inside it is detached (it must not inflate the critic). The actor's gradient reaches
    # it only through the imagined REWARD path. One optimizer is fine once they no longer cross.
    values = agent.value(torch.stack(states).detach())   # (H+1, B) — grad to the critic only
    returns = lambda_return(rewards, values.detach())     # (H, B) — the reward path carries the actor's analytic grad
    actor_loss = -(returns.mean() + args.ent_coef * torch.stack(entropies).mean())
    critic_loss = 0.5 * F.mse_loss(values[:-1], returns.detach())
    agent_opt.zero_grad()
    (actor_loss + critic_loss).backward()
    nn.utils.clip_grad_norm_(agent.parameters(), 100.0)
    agent_opt.step()
    imag_reward_hist = torch.stack(rewards).mean().item()   # mean per-step imagined reward under current actor
    if args.rerun:
        rr.set_time("imag_iter", sequence=it)
        rr.log("imag/reward_per_step", rr.Scalars([imag_reward_hist]))
        rr.log("imag/critic_loss", rr.Scalars([critic_loss.item()]))
    if it % max(args.imag_iters // 5, 1) == 0 or it == args.imag_iters - 1:
        print(f"[imag] iter {it:4d}  imagined reward/step {imag_reward_hist:+.4f}  "
              f"critic {critic_loss.item():.4f}")
# --- endregion ---


# --- region: eval ---
# The headline. Deploy the SAME deterministic policy in two worlds and compare.
#   IMAGINED: warm (h,z) on the start frame, then roll the prior forward under the
#     actor for eval_horizon steps — a pure dream, decoded, reward from the decode.
#   REAL: filter (h,z) on true frames each step, act, step the true PushT sim. The
#     reward uses the SAME function on the TRUE state. Success is the env's own flag.
# If imagined return >> real return, the actor learned to please a dream whose block
# dynamics 3.1 measured are wrong. If they agree, pushing transferred. Either is the
# lesson; we do not tune until the story is clean.
agent.eval()
imag_returns, real_returns, real_success = [], [], []
imag_final_pos, real_final_pos = [], []
for ep in range(args.eval_episodes):
    seed = EVAL_SEED_BASE + args.seed + ep
    env = PushTEnv()
    obs0 = env.reset(seed)
    norm0 = torch.tensor((obs0 - obs_mean) / obs_std, device=device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():   # IMAGINED rollout: one-frame warm, then dream forward
        h, z = wm._init(1)
        z = wm.posterior(torch.cat([h, wm.encoder(norm0)], dim=-1))
        dream_r = []
        for _ in range(args.eval_horizon):
            # Score the CURRENT decoded state BEFORE stepping, the SAME index
            # convention the real rollout uses (score the state you are in, THEN act),
            # so both returns average the same states (0..H-1): an honest gap, not a shift.
            dream_r.append(reward_from_obs(wm.decoder(torch.cat([h, z], dim=-1)))[0].item())
            action, _ = agent.act(torch.cat([h, z], dim=-1), sample=False)
            h, z, _ = wm.step(h, z, action)
        dream_pos = reward_from_obs(wm.decoder(torch.cat([h, z], dim=-1)))[1].item()  # final dreamed tee-dist
    imag_returns.append(float(np.mean(dream_r)))
    imag_final_pos.append(dream_pos)

    with torch.no_grad():   # REAL rollout: filter on true frames, act in the true sim
        h, z = wm._init(1)
        obs, real_r, done = obs0, [], False
        for _ in range(args.eval_horizon):
            norm = torch.tensor((obs - obs_mean) / obs_std, device=device, dtype=torch.float32).unsqueeze(0)
            z = wm.posterior(torch.cat([h, wm.encoder(norm)], dim=-1))
            action, _ = agent.act(torch.cat([h, z], dim=-1), sample=False)
            real_r.append(reward_from_obs(norm)[0].item())
            obs, _, done, info = env.step(action[0].cpu().numpy())
            h = wm.gru(torch.cat([z, action], dim=-1), h)
            if done:
                break
        final_norm = torch.tensor((obs - obs_mean) / obs_std, device=device, dtype=torch.float32)
        real_final_pos.append(reward_from_obs(final_norm)[1].item())
        real_success.append(float(info["success"]))
    real_returns.append(float(np.mean(real_r)))

imag_return = float(np.mean(imag_returns))     # mean per-step reward the actor BELIEVES it earns
real_return = float(np.mean(real_returns))     # mean per-step reward it ACTUALLY earns
gap = imag_return - real_return                # >0 => imagination is rosier than reality
success_rate = float(np.mean(real_success))
print(f"\n=== the imagination gap ({args.eval_episodes} held-out starts, "
      f"{args.eval_horizon}-step rollouts) ===")
print(f"IMAGINED return/step (in the dream) : {imag_return:+.4f}   final tee-dist {np.mean(imag_final_pos):.3f} m")
print(f"REAL     return/step (true PushT)   : {real_return:+.4f}   final tee-dist {np.mean(real_final_pos):.3f} m")
print(f"gap (imagined - real)               : {gap:+.4f}   "
      + ("<- imagination is DELUDED: the policy earns its reward on a hallucinated block"
         if gap > 0.02 else "<- imagined and real roughly agree at this scale"))
print(f"real task success rate              : {success_rate:.2f}   "
      "(the shaped reward can rise while success stays ~0 — the block never actually parks)")
if args.rerun:
    rr.log("eval/imagined_return", rr.Scalars([imag_return]))
    rr.log("eval/real_return", rr.Scalars([real_return]))
# --- endregion ---


# --- region: report ---
metrics = {
    "eval_episodes": args.eval_episodes,
    "eval_horizon": args.eval_horizon,
    "final_imagined_reward_per_step_train": round(imag_reward_hist, 6),
    "gamma": args.gamma,
    "hidden_dim": args.hidden_dim,
    "imag_horizon": args.imag_horizon,
    "imag_iters": args.imag_iters,
    "imagination_gap": round(gap, 6),
    "imagined_final_tee_dist": round(float(np.mean(imag_final_pos)), 6),
    "imagined_return_per_step": round(imag_return, 6),
    "latent_dim": args.latent_dim,
    "real_final_tee_dist": round(float(np.mean(real_final_pos)), 6),
    "real_return_per_step": round(real_return, 6),
    "real_success_rate": round(success_rate, 6),
    "seed": args.seed,
    "smoke": bool(args.smoke),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"\nmetrics: {args.out / 'metrics.json'}  (imagination gap {metrics['imagination_gap']:+.4f} "
      f"return/step; real success {success_rate:.2f})")
if args.rerun:
    print(f"recording: {args.out / 'dreamer.rrd'} — open it with: rerun {args.out / 'dreamer.rrd'}")
# --- endregion ---
