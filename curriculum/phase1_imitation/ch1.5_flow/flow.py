"""zero2robot 1.5 — Generative Policies II: Flow Matching.

Chapter 1.4 fixed the mode-collapse of a squared-error regressor by learning to
DENOISE noise into a sample (DDPM): a noise schedule, an epsilon-prediction net,
and a reverse loop that walks pure noise back to the data one small step at a time.
It works — but that walk is long and fussy: dozens of steps, a schedule to get
right, and fresh noise injected each step that you then have to fight back down.

Flow matching keeps the SAME generative idea — turn noise into a SAMPLE, so the
model commits to one mode instead of averaging two — and swaps the mechanism for a
simpler one. Draw a STRAIGHT LINE from a noise point to a data point, x_t =
(1-t)*noise + t*data; that line's velocity is the constant data - noise. Train one
net to predict that velocity field, then SAMPLE by integrating an ODE forward from
noise: x <- x + dt * v. No schedule, no injected noise, no posterior algebra — and
because the target paths are straight, the integrator reaches the data in FEWER
steps. This is the pi0 / stable-diffusion-3 objective, built from scratch.

Read this beside 1.4. Same file structure, same 2D ring toy, same PushT policy,
same eval — only the OBJECTIVE (predict velocity, not noise) and the SAMPLER
(integrate an ODE, don't run the DDPM posterior) change. ~60 lines differ; the site
renders the diff. The toy makes the same point (flow covers the 8 ring modes; a
same-width MSE regressor collapses to the dead center — objective-driven, robust to
capacity), and it adds one flow measures directly: fewer sampling steps to the modes.

SIMPLIFIED from real flow-matching policies (pi0), flagged in prose: we flow a
SINGLE action (not an action-horizon chunk), the net is a small MLP (not a temporal
U-Net), we condition on the 10-number state (no image/VLM). We integrate with plain
Euler (not a higher-order or adaptive solver). The velocity->sample core is
identical; capacity is deliberately tiny. See "What we cut". Everything here is
torch + numpy — no diffusers, no einops.

Run it:      python curriculum/phase1_imitation/ch1.5_flow/flow.py --seed 0
Break it:    python curriculum/phase1_imitation/ch1.5_flow/flow.py --seed 0 --break few_steps
CI smoke:    python curriculum/phase1_imitation/ch1.5_flow/flow.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import math
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
from curriculum.common.envs.pusht import PushTEnv, gen_demos  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

OBS_DIM, ACT_DIM = PushTEnv.OBS_DIM, PushTEnv.ACT_DIM
TIME_DIM = 32       # width of the sinusoidal time embedding fed to the velocity net
TIME_SCALE = 1000.0  # flow time t lives in [0,1]; scale it up so the sinusoidal embed has real resolution
N_TOY = 2000        # points sampled from the 2D toy target (the ring)
EFF_STEPS = 5       # a deliberately SMALL step budget — flow's headline: it covers the modes with few steps

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=None,
                    help="LeRobot PushT dataset (your ch0.4 teleop, or omit => generate --num_demos scripted demos)")
parser.add_argument("--out", type=Path, default=Path("outputs/ch1.5-flow"))
parser.add_argument("--flow_steps", type=int, default=100,
                    help="T: Euler steps that integrate the ODE from noise to data. Fewer => curved-path error (Break It). T4: 100 | smoke: 4")
parser.add_argument("--model_dim", type=int, default=128)   # velocity-net MLP width. T4: 256 | smoke: 16
parser.add_argument("--num_demos", type=int, default=100)   # scripted PushT demos. T4: 500 | smoke: 4
parser.add_argument("--epochs", type=int, default=300)      # cpu-laptop: minutes | smoke: 3
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3, help="constant Adam lr (decaying it to 0 undertrains the velocity net here — measured, as in ch1.4)")
parser.add_argument("--eval_episodes", type=int, default=20)  # T4: 50 | smoke: 2 — few episodes is noisy (ch1.6)
parser.add_argument("--seed", type=int, default=0, help="seeds demos, inits, the noise draws, and the ODE sampler")
parser.add_argument("--break", dest="break_mode", choices=("few_steps", "wrong_target"), default=None,
                    help="Break It: a real flow-matching misconception with a measured signature (see the toy/eval regions)")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())  # cpu: deterministic (statistical repro on GPU/mps)
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch globals (model inits draw from these)
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.flow_steps, args.model_dim, args.num_demos = 4, 16, 4
    args.epochs, args.eval_episodes, args.device = 3, 2, "cpu"
if args.break_mode == "few_steps":
    args.flow_steps = 2  # the misconception "2 steps is plenty" — the curved marginal field is under-integrated
BROKEN_TARGET = args.break_mode == "wrong_target"  # flip the velocity target's sign: flow AWAY from the data
TOY_ITERS = 30 if args.smoke else 1500  # the 2D toy is cheap; give it enough steps to reveal the modes
banner("ch1.5-flow", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
# One CPU RNG feeds every stochastic draw (noise, batch indices, the sampler) in a
# fixed order, then tensors move to `device`: same seed -> byte-identical CPU run.
gen = torch.Generator().manual_seed(args.seed)
def randn(shape):  # noqa: E306  standard normal on the CPU gen, moved to the run device
    return torch.randn(shape, generator=gen).to(device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch1.5-flow", spawn=False)
    rr.save(str(args.out / "flow.rrd"))
# --- endregion ---

# --- region: core ---
# The flow-matching machinery, shared by toy and policy. Nothing from a generative
# library — the interpolation, the loss, and the ODE sampler ARE the method. Note
# what is MISSING versus ch1.4: no noise schedule, no betas/alphas, no posterior
# variance. A straight line needs none of it.


def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Continuous flow time (B,) -> (B, dim) sinusoidal features (the same time
    embedding as ch1.4 / transformers). Lets one network condition smoothly on
    WHERE along the noise->data path it is, without a separate weight per time."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


def interpolate(x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """The conditional flow path: a STRAIGHT LINE from a noise point (t=0) to a data
    point x0 (t=1). x_t = (1-t)*noise + t*x0. This one line replaces ch1.4's entire
    noise schedule + q_sample — the path is fixed by geometry, not a learned/tuned
    schedule. Its velocity (below) is the constant x0 - noise, the same everywhere
    on the line."""
    return (1.0 - t)[:, None] * noise + t[:, None] * x0


def flow_matching_loss(model: nn.Module, x0: torch.Tensor, cond) -> torch.Tensor:
    """The entire training objective (conditional flow matching). Pick a random time
    t in [0,1] per sample, put the sample on its straight noise->data line at t, and
    ask the network to PREDICT THAT LINE'S VELOCITY, x0 - noise. Plain MSE on the
    velocity — but because the target is a velocity toward a SAMPLED data point, not
    the action itself, minimizing it never averages two actions into a third one.
    (ch1.4 predicted the noise instead; this is the ~2-line objective swap.)"""
    t = torch.rand(len(x0), generator=gen)
    noise = torch.randn(x0.shape, generator=gen).to(device)
    x_t = interpolate(x0, noise, t.to(device))
    target_v = (noise - x0) if BROKEN_TARGET else (x0 - noise)  # Break It flips this sign
    return F.mse_loss(model(x_t, t.to(device), cond), target_v)


@torch.no_grad()
def ode_sample_loop(model: nn.Module, shape, cond, steps: int, log=None) -> torch.Tensor:
    """Sampling = integrate the learned velocity field forward, from noise (t=0) to a
    sample (t=1), with plain forward Euler. Each step just follows the arrow: x <- x
    + dt * v. No fresh noise, no posterior mean, no clip — the whole ch1.4 reverse
    step collapses to this. Straight conditional paths mean few steps get you there;
    `log` records the moving cloud to rerun so you can watch noise slide into data."""
    x = randn(shape)  # start at pure noise, flow time t = 0
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((shape[0],), i * dt, device=device)
        x = x + dt * model(x, t, cond)  # forward Euler along the velocity field
        if log is not None:
            rr.set_time("flow_step", sequence=i + 1)
            rr.log(log, rr.Points3D(_xy0(x)))
    return x


class VelocityNet(nn.Module):
    """The velocity predictor, used for BOTH the toy and the policy. In: a point x_t
    on a noise->data line, its flow time t, and (optionally) a conditioning vector
    (the obs for the policy, nothing for the toy). Out: the predicted velocity, x_t's
    shape. Just an MLP — the lesson is the OBJECTIVE and the SAMPLER, not the
    architecture (real flow policies use a U-Net / transformer; see 'What we cut').
    Structurally identical to ch1.4's Denoiser — only the name and what it predicts
    change. When conditioned, obs normalization lives inside as buffers so a
    checkpoint carries its own stats."""

    def __init__(self, x_dim: int, cond_dim: int, hidden: int, obs_min=None, obs_range=None):
        super().__init__()
        self.cond_dim = cond_dim
        self.net = nn.Sequential(
            nn.Linear(x_dim + TIME_DIM + cond_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, x_dim),
        )
        if cond_dim and obs_min is not None:
            self.register_buffer("obs_min", torch.from_numpy(obs_min))
            self.register_buffer("obs_range", torch.from_numpy(obs_range))

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, cond) -> torch.Tensor:
        parts = [x_t, sinusoidal_embed(t * TIME_SCALE, TIME_DIM)]  # scale t in [0,1] up before embedding
        if self.cond_dim:  # normalize obs to [-1, 1] inside the model, then condition on it
            parts.append((2.0 * (cond - self.obs_min) / self.obs_range - 1.0).clamp(-1.0, 1.0))
        return self.net(torch.cat(parts, dim=1))


def _xy0(a) -> np.ndarray:
    """(N, >=2) points -> (N, 3) with z=0 for rerun's 3D point cloud."""
    a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    return np.column_stack([a[:, :2], np.zeros(len(a), dtype=np.float32)]).astype(np.float32)
# --- endregion ---

# --- region: toy ---
# The aha, before any robot. Target: a unit ring — every angle an equally-good
# "mode", the center the single point NO sample occupies, exactly where an
# averaging regressor lands. The ch1.1 left-vs-right collapse in 2D, where we SEE it.
angle = torch.rand(N_TOY, generator=gen) * 2.0 * math.pi
ring = torch.stack([angle.cos(), angle.sin()], dim=1) + 0.03 * torch.randn(N_TOY, 2, generator=gen)
ring = ring.to(device)

# (a) Flow matching: train the velocity net to predict x0 - noise on ring points.
toy = VelocityNet(2, 0, args.model_dim).to(device)
toy_opt = torch.optim.Adam(toy.parameters(), lr=args.lr)
toy_loss = float("nan")
for it in range(TOY_ITERS):
    idx = torch.randint(0, N_TOY, (256,), generator=gen)
    loss = flow_matching_loss(toy, ring[idx], None)
    toy_opt.zero_grad()
    loss.backward()
    toy_opt.step()
    toy_loss = loss.item()

# (b) Regression baseline: a same-WIDTH MLP (smaller than the time-conditioned
# velocity net — which only helps the point), trained to map noise -> a point with
# MSE. Fresh noise says nothing about WHICH ring point, so the MSE-optimal output is
# the average of them all — the empty center. Objective-driven, robust to capacity.
# IDENTICAL to ch1.4's baseline: the collapse is about the loss, not the generator.
regress = nn.Sequential(nn.Linear(2, args.model_dim), nn.SiLU(),
                        nn.Linear(args.model_dim, 2)).to(device)
reg_opt = torch.optim.Adam(regress.parameters(), lr=args.lr)
for it in range(TOY_ITERS):
    idx = torch.randint(0, N_TOY, (256,), generator=gen)
    loss = F.mse_loss(regress(randn((256, 2))), ring[idx])
    reg_opt.zero_grad()
    loss.backward()
    reg_opt.step()

# Sample from both and MEASURE multimodality. `modes_covered` bins on-ring samples
# into 8 angular sectors: flow should light up all 8, regression none.
flow_samples = ode_sample_loop(toy, (N_TOY, 2), None, args.flow_steps,
                               log="toy/flow" if args.rerun else None).cpu().numpy()
with torch.no_grad():
    reg_samples = regress(randn((N_TOY, 2))).cpu().numpy()


def ring_stats(samples: np.ndarray) -> tuple[float, float, int]:
    r = np.hypot(samples[:, 0], samples[:, 1])
    on_ring = np.abs(r - 1.0) < 0.2
    sector = (np.floor((np.arctan2(samples[:, 1], samples[:, 0]) + math.pi) / (2 * math.pi) * 8).astype(int) % 8)
    return float(r.mean()), float(on_ring.mean()), int(len(np.unique(sector[on_ring])))


flow_r, flow_hit, flow_modes = ring_stats(flow_samples)
reg_r, reg_hit, reg_modes = ring_stats(reg_samples)
# The chapter's flow-specific measurement: re-sample the SAME trained net with only
# EFF_STEPS Euler steps. Flow's straight paths still land the modes at a step budget
# where ch1.4's DDPM reverse loop is still a blur — the efficiency claim, measured.
lowstep_samples = ode_sample_loop(toy, (N_TOY, 2), None, EFF_STEPS).cpu().numpy()
_, lowstep_hit, lowstep_modes = ring_stats(lowstep_samples)
print(f"toy multimodality [measured]: flow mean_radius {flow_r:.2f} ring_hit {flow_hit:.2f} modes {flow_modes}/8"
      f"  |  regression mean_radius {reg_r:.2f} ring_hit {reg_hit:.2f} modes {reg_modes}/8")
print(f"toy step efficiency [measured]: flow covers {lowstep_modes}/8 modes with only {EFF_STEPS} Euler steps "
      f"(vs {flow_modes}/8 at {args.flow_steps}) — straight paths need few steps")
if args.rerun:  # forward: watch the ring slide into noise along the straight paths (negative time = before sampling)
    for k in range(7):
        t_k = torch.full((N_TOY,), k / 6.0, device=device)
        rr.set_time("flow_step", sequence=k - 7)
        rr.log("toy/forward", rr.Points3D(_xy0(interpolate(ring, randn((N_TOY, 2)), t_k))))
    rr.log("toy/target", rr.Points3D(_xy0(ring), colors=(140, 140, 150)), static=True)
    rr.log("toy/regression", rr.Points3D(_xy0(reg_samples), colors=(230, 102, 90)), static=True)
# --- endregion ---

# --- region: data ---
# The policy trains on the SAME scripted PushT demos as ch1.1 / ch1.4 — the only
# change from ch1.4 is the objective (predict velocity) and the sampler (integrate).
if args.data is None:
    # Regenerate every run (never reuse a leftover dir): a cache from a different
    # --seed/--num_demos would silently train on the wrong data. gen_demos is
    # deterministic, so same args -> bit-identical demos, built or rebuilt.
    args.data = args.out / "demos"
    if args.data.exists():
        shutil.rmtree(args.data)
    gen_demos.main(["--episodes", str(args.num_demos), "--seed", str(args.seed),
                    "--out", str(args.data), "--no-video"])
if not (args.data / "meta" / "info.json").is_file():
    sys.exit(f"no dataset at {args.data} — record one in ch0.4, or generate demos:\n"
             f"  python curriculum/common/envs/pusht/gen_demos.py "
             f"--episodes 500 --seed 0 --out {args.data} --no-video")

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402  (heavy import — after cheap failures)

frames = LeRobotDataset("local/pusht-demos", root=args.data).hf_dataset.with_format("numpy")
obs = np.stack(frames["observation.state"]).astype(np.float32)   # (N, 10) — layout in pusht_env.py
actions = np.stack(frames["action"]).astype(np.float32)          # (N, 2)  — already clipped to [-1, 1]

# The ODE starts from N(0, I), so the data must sit at ~unit scale or the sampler
# fights a scale mismatch: these expert actions have std ~0.3, and standardizing
# them to zero-mean/unit-std is what takes this policy from 0% to working (measured,
# same as ch1.4). So we STANDARDIZE the flowed actions and MIN-MAX normalize the
# conditioning obs (constant obs dims get range 1 -> a constant, not a divide-by-zero).
obs_min = obs.min(0)
obs_range = np.where(obs.max(0) - obs_min < 1e-4, np.float32(1.0), obs.max(0) - obs_min)
act_mean = actions.mean(0)
act_std = np.where(actions.std(0) < 1e-4, np.float32(1.0), actions.std(0))
obs_t = torch.from_numpy(obs).to(device)
act_t = torch.from_numpy((actions - act_mean) / act_std).to(device)
act_mean_t = torch.from_numpy(act_mean).to(device)
act_std_t = torch.from_numpy(act_std).to(device)
print(f"dataset: {len(np.unique(frames['episode_index']))} episodes / {len(obs)} frames, "
      f"flow_steps={args.flow_steps}, model_dim={args.model_dim}")
# --- endregion ---

# --- region: train ---
# Chapter 1.4's loop, one line changed: the loss is flow_matching_loss (predict the
# velocity of a noised action's straight path, conditioned on the obs) instead of the
# diffusion noise-prediction loss. As in ch1.4 we DON'T cosine-decay the lr — the net
# must keep fitting the velocity at all times t; decaying it undertrains (measured).
torch.manual_seed(args.seed)  # policy init + training-noise stream reproducible, independent of the toy
gen.manual_seed(args.seed)    # give the policy a fresh noise stream (the toy drained the shared gen above)
policy = VelocityNet(ACT_DIM, OBS_DIM, args.model_dim, obs_min, obs_range).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
shuffle = torch.Generator().manual_seed(args.seed + 1)  # torch-side RNG for batch order
train_loss, global_step = float("nan"), 0
for epoch in range(args.epochs):
    epoch_loss, num_batches = 0.0, 0
    for batch in torch.randperm(len(obs_t), generator=shuffle).split(args.batch_size):
        loss = flow_matching_loss(policy, act_t[batch], obs_t[batch])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss, num_batches = epoch_loss + loss.item(), num_batches + 1
        if args.rerun:
            rr.set_time("step", sequence=global_step)
            rr.log("policy/loss/train", rr.Scalars([loss.item()]))
        global_step += 1
    train_loss = epoch_loss / num_batches
    if epoch % 50 == 0 or epoch == args.epochs - 1:
        print(f"epoch {epoch:4d}  flow_mse {train_loss:.5f}")
# --- endregion ---

# --- region: eval ---
# Loss measured on velocity fits; rollouts measure the task. At every env step we
# SAMPLE an action: start from noise, integrate the ODE conditioned on the current
# obs. --break few_steps runs this at flow_steps=2 (set in setup): too few steps
# can't follow the curved MARGINAL field, the action lands off the data, and the
# pusher wanders — a measured drop, not a crash.
def rollout(net: VelocityNet, seed: int, tag: str, episode: int) -> tuple[bool, float]:
    env = PushTEnv()
    obs_now = env.reset(seed)
    gen.manual_seed(seed)  # seed the sampler from the episode seed: reproducible AND order-independent, so
    #                        baseline and trained see the same per-episode noise (a fair, seeded comparison)
    done, episode_return, info = False, 0.0, {}
    while not done:
        cond = torch.from_numpy(obs_now).to(device).unsqueeze(0)          # (10,) -> (1, 10)
        sample = ode_sample_loop(net, (1, ACT_DIM), cond, args.flow_steps)  # in standardized action space
        action = (sample * act_std_t + act_mean_t).clamp(-1.0, 1.0)[0].cpu().numpy()
        obs_now, reward, done, info = env.step(action)
        episode_return += reward
        if args.rerun:
            rr.set_time("sim_time", duration=episode * (PushTEnv.MAX_STEPS / PushTEnv.CONTROL_HZ) + env.data.time)
            rr.log(f"eval/{tag}/action", rr.Scalars(action.astype(np.float64)))
            rr.log(f"eval/{tag}/pos_err", rr.Scalars([info["pos_err"]]))
    return bool(info["success"]), episode_return


def evaluate(net: VelocityNet, tag: str) -> tuple[float, float]:
    # 10_000 + offset: held out from demo seeds (0..num_demos) by construction.
    outcomes = [rollout(net, 10_000 + args.seed + ep, tag, ep) for ep in range(args.eval_episodes)]
    success_rate = float(np.mean([s for s, _ in outcomes]))
    mean_return = float(np.mean([r for _, r in outcomes]))
    if args.rerun:
        rr.log(f"eval/{tag}/success_rate", rr.Scalars([success_rate]))
    print(f"eval[{tag:9s}]: success {success_rate:.2f}  mean_return {mean_return:.3f}")
    return success_rate, mean_return


torch.manual_seed(args.seed + 1)  # a fixed random-init reference, independent of eval-time RNG
baseline = VelocityNet(ACT_DIM, OBS_DIM, args.model_dim, obs_min, obs_range).to(device)
baseline_success, baseline_return = evaluate(baseline, "untrained")
success_rate, mean_return = evaluate(policy, "trained")
# --- endregion ---

# --- region: report ---
# ONNX/contract note: contract v1 is model(obs[1,10]) -> action[1,2], a single
# stateless step. A flow policy does NOT fit v1 twice over: its velocity net takes
# THREE inputs (a point, a flow time, obs) and one action needs flow_steps of them
# chained through an ODE integrator the runtime must own. We export the velocity net
# CORE and check torch/onnxruntime agree, proving the serialization path; a
# sampler-aware contract (v2) is what the browser needs to drive this. Like ch1.3/1.4
# — and cheaper here: the integrator is Euler, no schedule to ship.
if not args.smoke:
    import onnxruntime as ort  # noqa: E402  (heavy; only when actually exporting)

    onnx_path = args.out / "flow_velocity.onnx"
    policy.eval().to("cpu")
    dummy = (torch.zeros(1, ACT_DIM), torch.zeros(1), torch.zeros(1, OBS_DIM))  # timestep is a FLOAT in [0,1]
    torch.onnx.export(policy, dummy, str(onnx_path),
                      input_names=["point", "flow_time", "observation"],
                      output_names=["velocity"], dynamo=False)  # 3 inputs — NOT contract v1
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    px = np.random.default_rng(0).standard_normal((1, ACT_DIM)).astype(np.float32)
    po = np.random.default_rng(1).standard_normal((1, OBS_DIM)).astype(np.float32)
    pt = np.full(1, 0.5, dtype=np.float32)
    with torch.no_grad():
        delta = float(np.abs(policy(torch.from_numpy(px), torch.from_numpy(pt), torch.from_numpy(po)).numpy()
                             - session.run(None, {"point": px, "flow_time": pt, "observation": po})[0]).max())
    assert delta < 1e-4, f"torch/onnx velocity parity failed: {delta:.2e}"
    print(f"exported {onnx_path} (velocity core, 3 inputs — NOT contract v1); parity {delta:.2e}")
    policy.to(device)

metrics = {
    "baseline_mean_return": round(baseline_return, 6),
    "baseline_success_rate": round(baseline_success, 6),
    "break_mode": args.break_mode or "none",
    "epochs": args.epochs,
    "final_train_loss": round(train_loss, 6),
    "flow_steps": args.flow_steps,
    "mean_return": round(mean_return, 6),
    "model_dim": args.model_dim,
    "num_demos": len(np.unique(frames["episode_index"])),
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "success_rate": round(success_rate, 6),
    "toy_final_loss": round(toy_loss, 6),
    "toy_flow_lowstep_modes_covered": lowstep_modes,
    "toy_flow_lowstep_ring_hit": round(lowstep_hit, 6),
    "toy_flow_lowstep_steps": EFF_STEPS,
    "toy_flow_mean_radius": round(flow_r, 6),
    "toy_flow_modes_covered": flow_modes,
    "toy_flow_ring_hit": round(flow_hit, 6),
    "toy_regress_mean_radius": round(reg_r, 6),
    "toy_regress_modes_covered": reg_modes,
    "toy_regress_ring_hit": round(reg_hit, 6),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}  (toy: flow covers {flow_modes}/8 modes, "
      f"regression {reg_modes}/8 and collapses to r={reg_r:.2f})")
if args.rerun:
    print(f"recording: {args.out / 'flow.rrd'} — open it with: rerun {args.out / 'flow.rrd'}")
# --- endregion ---
