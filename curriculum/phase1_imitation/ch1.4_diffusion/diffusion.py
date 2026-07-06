"""zero2robot 1.4 — Generative Policies I: Diffusion.

BC (1.1) and ACT (1.3) both regress the expert's action with a squared/absolute
error, and that objective keeps hitting one failure: when a state maps to several
equally-good actions (go left OR go right), the loss is minimized by their
AVERAGE — and the average of two good pushes is one bad one, into the stuck
middle. Diffusion removes it: instead of predicting the action, it learns to turn
noise into a SAMPLE from the action distribution, so it commits to one mode.

Two passes, smallest first:
  1. A 2D TOY. A deliberately multimodal target (a ring — every direction an
     equally-good mode, the empty center the one place no data sits), denoised
     from scratch: a noise schedule, a tiny MLP that predicts the noise, and the
     reverse sampling loop. Beside it, the SAME MLP trained as a one-shot MSE
     regressor. Measured: diffusion covers all modes, regression collapses to the
     dead center — the whole idea, in rerun, before a robot is in sight.
  2. THE POLICY. Condition that denoiser on the PushT observation, denoise the 2D
     action, train on the scripted demos, sample an action per step in the env.

SIMPLIFIED from real Diffusion Policy (Chi et al.), flagged in prose: we denoise a
SINGLE action (not an action-horizon chunk), the denoiser is a small MLP (not a
temporal U-Net), we condition on the 10-number state (no image/ResNet). The
noise->sample core is identical; capacity is deliberately tiny. See "What we cut".
Everything here is torch + numpy — no diffusers, no einops.

Run it:      python curriculum/phase1_imitation/ch1.4_diffusion/diffusion.py --seed 0
Break it:    python curriculum/phase1_imitation/ch1.4_diffusion/diffusion.py --seed 0 --break few_steps
CI smoke:    python curriculum/phase1_imitation/ch1.4_diffusion/diffusion.py --smoke --seed 0 --no-rerun
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
TIME_DIM = 32       # width of the sinusoidal timestep embedding fed to the denoiser
N_TOY = 2000        # points sampled from the 2D toy target (the ring)
X0_CLIP = 3.0       # clamp the predicted clean sample to +/-3 std during sampling (manifold guard)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path, default=None,
                    help="LeRobot PushT dataset (your ch0.4 teleop, or omit => generate --num_demos scripted demos)")
parser.add_argument("--out", type=Path, default=Path("outputs/ch1.4-diffusion"))
parser.add_argument("--denoising_steps", type=int, default=100,
                    help="T: forward-noising / reverse-sampling steps. Fewer under-denoises the action (Break It). T4: 100 | smoke: 4")
parser.add_argument("--model_dim", type=int, default=128)   # denoiser MLP width. T4: 256 | smoke: 16
parser.add_argument("--num_demos", type=int, default=100)   # scripted PushT demos. T4: 500 | smoke: 4
parser.add_argument("--epochs", type=int, default=300)      # cpu-laptop: minutes | smoke: 3
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--lr", type=float, default=1e-3, help="constant Adam lr (decaying it to 0 undertrains the denoiser here — measured)")
parser.add_argument("--eval_episodes", type=int, default=20)  # T4: 50 | smoke: 2 — few episodes is noisy (ch1.6)
parser.add_argument("--seed", type=int, default=0, help="seeds demos, inits, the noise draws, and the sampler")
parser.add_argument("--break", dest="break_mode", choices=("few_steps", "wrong_schedule"), default=None,
                    help="Break It: a real diffusion misconception with a measured signature (see the toy/eval regions)")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())  # cpu: deterministic (statistical repro on GPU/mps)
parser.add_argument("--smoke", action="store_true",
                    help="tiny self-contained CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch globals (model inits draw from these)
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.denoising_steps, args.model_dim, args.num_demos = 4, 16, 4
    args.epochs, args.eval_episodes, args.device = 3, 2, "cpu"
if args.break_mode == "few_steps":
    args.denoising_steps = 2  # the misconception "2 steps is plenty" — under-denoised, blurry actions
BROKEN_SCHEDULE = args.break_mode == "wrong_schedule"
TOY_ITERS = 30 if args.smoke else 1500  # the 2D toy is cheap; give it enough steps to reveal the modes
banner("ch1.4-diffusion", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
# One CPU RNG feeds every stochastic draw (noise, batch indices, the sampler) in a
# fixed order, then tensors move to `device`: same seed -> byte-identical CPU run.
gen = torch.Generator().manual_seed(args.seed)
def randn(shape):  # noqa: E306  standard normal on the CPU gen, moved to the run device
    return torch.randn(shape, generator=gen).to(device)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch1.4-diffusion", spawn=False)
    rr.save(str(args.out / "diffusion.rrd"))
# --- endregion ---

# --- region: core ---
# The diffusion machinery, shared by toy and policy. Nothing from a diffusion
# library — the schedule, the noising, and the reverse sampler ARE the method.


def sinusoidal_embed(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Integer timesteps (B,) -> (B, dim) sinusoidal features (the DDPM/transformer
    time embedding). Lets one network condition smoothly on WHICH noise level it
    is undoing, without a separate weight per step."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / half)
    ang = t.float()[:, None] * freqs[None]
    return torch.cat([ang.sin(), ang.cos()], dim=1)


def make_schedule(steps: int, device: torch.device, broken: bool = False) -> dict:
    """Precompute the forward-process constants. `acp` is alpha-bar (cumprod of
    1-beta): the fraction of ORIGINAL signal still present at each step. A good
    schedule drives acp from ~1 (clean) to ~0 (pure noise) so sampling can honestly
    start from N(0, I); the cosine schedule (Nichol & Dhariwal) reaches ~0 even at
    modest step counts. `broken` is the Break It: near-zero betas leave acp~1, so
    the model only ever removes a whisper of noise and never learns the trip back
    from pure noise."""
    if broken:
        betas = torch.linspace(1e-6, 1e-4, steps, device=device)
    else:
        u = torch.linspace(0, steps, steps + 1, device=device) / steps
        acp_full = torch.cos((u + 0.008) / 1.008 * math.pi / 2) ** 2
        acp_full = acp_full / acp_full[0].clone()
        betas = (1 - acp_full[1:] / acp_full[:-1]).clamp(1e-8, 0.999)
    alphas = 1.0 - betas
    acp = torch.cumprod(alphas, dim=0)
    # acp_prev (acp shifted one step) feeds the reverse posterior; post_sigma is
    # the noise injected per reverse step — the true posterior variance beta-tilde,
    # smaller than beta_t, which samples noticeably cleaner than beta_t.
    acp_prev = torch.cat([torch.ones(1, device=device), acp[:-1]])
    post_var = betas * (1.0 - acp_prev) / (1.0 - acp)
    return {"steps": steps, "betas": betas, "alphas": alphas, "acp": acp,
            "acp_prev": acp_prev, "sqrt_acp": acp.sqrt(),
            "sqrt_one_minus_acp": (1.0 - acp).sqrt(),
            "post_sigma": post_var.clamp_min(1e-20).sqrt()}


def q_sample(x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor, sch: dict) -> torch.Tensor:
    """Forward process: jump straight to noise level t in one shot (no loop needed).
    x_t = sqrt(acp_t) * x0 + sqrt(1-acp_t) * noise."""
    return sch["sqrt_acp"][t][:, None] * x0 + sch["sqrt_one_minus_acp"][t][:, None] * noise


def diffusion_loss(model: nn.Module, x0: torch.Tensor, cond, sch: dict) -> torch.Tensor:
    """The entire training objective. Pick a random noise level per sample, noise
    x0 to it, and ask the network to PREDICT THE NOISE it added (epsilon-prediction).
    Plain MSE on that noise — but because the target is the noise, not the action,
    minimizing it never averages two actions into a third one."""
    t = torch.randint(0, sch["steps"], (len(x0),), generator=gen)
    noise = torch.randn(x0.shape, generator=gen)
    x_t = q_sample(x0, t.to(device), noise.to(device), sch)
    return F.mse_loss(model(x_t, t.to(device), cond), noise.to(device))


@torch.no_grad()
def p_sample_loop(model: nn.Module, shape, cond, sch: dict, log=None) -> torch.Tensor:
    """Reverse process: start from pure noise and walk it back to a sample, one
    denoising step at a time (ancestral DDPM). Each step subtracts the model's
    predicted noise, rescales, and re-injects a little fresh noise (except the last
    step). `log` records the shrinking cloud to rerun so you can watch noise become
    structure."""
    x = randn(shape)
    for step in reversed(range(sch["steps"])):
        t = torch.full((shape[0],), step, dtype=torch.long, device=device)
        eps = model(x, t, cond)
        # Recover the predicted CLEAN sample x0 from the noise and CLIP it: a tiny
        # model can predict a wild x0 early on, and clamping to the data's scale
        # keeps the trajectory on the manifold (Diffusion Policy does this — here it
        # is the step from a policy that fails to one that works). Then step to
        # x_{t-1} via the DDPM posterior mean of (x0, x).
        acp, acp_prev = sch["acp"][step], sch["acp_prev"][step]
        x0 = ((x - sch["sqrt_one_minus_acp"][step] * eps) / sch["sqrt_acp"][step]).clamp(-X0_CLIP, X0_CLIP)
        mean = (sch["betas"][step] * acp_prev.sqrt() / (1.0 - acp) * x0
                + (1.0 - acp_prev) * sch["alphas"][step].sqrt() / (1.0 - acp) * x)
        x = mean + (sch["post_sigma"][step] * randn(shape) if step > 0 else 0.0)
        if log is not None:
            rr.set_time("denoise_step", sequence=sch["steps"] - step)
            rr.log(log, rr.Points3D(_xy0(x)))
    return x


class Denoiser(nn.Module):
    """The noise-predictor, used for BOTH the toy and the policy. In: a noisy
    sample x_t, its noise level t, and (optionally) a conditioning vector (the obs
    for the policy, nothing for the toy). Out: the predicted noise, x_t's shape.
    Just an MLP — the lesson is the OBJECTIVE and the SAMPLER, not the architecture
    (real Diffusion Policy uses a U-Net; see 'What we cut'). When conditioned, obs
    normalization lives inside as buffers so a checkpoint carries its own stats."""

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
        parts = [x_t, sinusoidal_embed(t, TIME_DIM)]
        if self.cond_dim:  # normalize obs to [-1, 1] inside the model, then condition on it
            parts.append((2.0 * (cond - self.obs_min) / self.obs_range - 1.0).clamp(-1.0, 1.0))
        return self.net(torch.cat(parts, dim=1))


def _xy0(a) -> np.ndarray:
    """(N, >=2) points -> (N, 3) with z=0 for rerun's 3D point cloud."""
    a = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
    return np.column_stack([a[:, :2], np.zeros(len(a), dtype=np.float32)]).astype(np.float32)


schedule = make_schedule(args.denoising_steps, device, broken=BROKEN_SCHEDULE)
# --- endregion ---

# --- region: toy ---
# The aha, before any robot. Target: a unit ring — every angle an equally-good
# "mode", the center the single point NO sample occupies, exactly where an
# averaging regressor lands. The ch1.1 left-vs-right collapse in 2D, where we SEE it.
angle = torch.rand(N_TOY, generator=gen) * 2.0 * math.pi
ring = torch.stack([angle.cos(), angle.sin()], dim=1) + 0.03 * torch.randn(N_TOY, 2, generator=gen)
ring = ring.to(device)

# (a) Diffusion: train the denoiser to predict the noise on ring points.
toy = Denoiser(2, 0, args.model_dim).to(device)
toy_opt = torch.optim.Adam(toy.parameters(), lr=args.lr)
toy_loss = float("nan")
for it in range(TOY_ITERS):
    idx = torch.randint(0, N_TOY, (256,), generator=gen)
    loss = diffusion_loss(toy, ring[idx], None, schedule)
    toy_opt.zero_grad()
    loss.backward()
    toy_opt.step()
    toy_loss = loss.item()

# (b) Regression baseline: a same-WIDTH MLP (smaller than the timestep-conditioned
# denoiser — which only helps the point), trained to map noise -> a point with MSE.
# Fresh noise says nothing about WHICH ring point, so the MSE-optimal output is the
# average of them all — the empty center. Objective-driven, robust to capacity.
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
# into 8 angular sectors: diffusion should light up all 8, regression none.
diff_samples = p_sample_loop(toy, (N_TOY, 2), None, schedule,
                             log="toy/diffusion" if args.rerun else None).cpu().numpy()
with torch.no_grad():
    reg_samples = regress(randn((N_TOY, 2))).cpu().numpy()


def ring_stats(samples: np.ndarray) -> tuple[float, float, int]:
    r = np.hypot(samples[:, 0], samples[:, 1])
    on_ring = np.abs(r - 1.0) < 0.2
    sector = (np.floor((np.arctan2(samples[:, 1], samples[:, 0]) + math.pi) / (2 * math.pi) * 8).astype(int) % 8)
    return float(r.mean()), float(on_ring.mean()), int(len(np.unique(sector[on_ring])))


diff_r, diff_hit, diff_modes = ring_stats(diff_samples)
reg_r, reg_hit, reg_modes = ring_stats(reg_samples)
print(f"toy multimodality [measured]: diffusion mean_radius {diff_r:.2f} ring_hit {diff_hit:.2f} modes {diff_modes}/8"
      f"  |  regression mean_radius {reg_r:.2f} ring_hit {reg_hit:.2f} modes {reg_modes}/8")
if args.rerun:  # forward: watch structured data melt into noise (negative time = before the reverse pass)
    for step in range(0, schedule["steps"], max(1, schedule["steps"] // 6)):
        t = torch.full((N_TOY,), step, dtype=torch.long, device=device)
        rr.set_time("denoise_step", sequence=step - schedule["steps"])
        rr.log("toy/forward", rr.Points3D(_xy0(q_sample(ring, t, randn((N_TOY, 2)), schedule))))
    rr.log("toy/target", rr.Points3D(_xy0(ring), colors=(140, 140, 150)), static=True)
    rr.log("toy/regression", rr.Points3D(_xy0(reg_samples), colors=(230, 102, 90)), static=True)
# --- endregion ---

# --- region: data ---
# The policy trains on the SAME scripted PushT demos as ch1.1 — the only change
# from BC is the head (a denoiser) and the loss (predict noise, not the action).
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

# The sampler starts from N(0, I), so the data must sit at ~unit scale or the
# reverse process fights a scale mismatch: these expert actions have std ~0.3, and
# standardizing them to zero-mean/unit-std is what takes this policy from 0% to
# working (measured). So we STANDARDIZE the diffused actions and MIN-MAX normalize
# the conditioning obs (constant obs dims get range 1 -> a constant, not a divide-by-zero).
obs_min = obs.min(0)
obs_range = np.where(obs.max(0) - obs_min < 1e-4, np.float32(1.0), obs.max(0) - obs_min)
act_mean = actions.mean(0)
act_std = np.where(actions.std(0) < 1e-4, np.float32(1.0), actions.std(0))
obs_t = torch.from_numpy(obs).to(device)
act_t = torch.from_numpy((actions - act_mean) / act_std).to(device)
act_mean_t = torch.from_numpy(act_mean).to(device)
act_std_t = torch.from_numpy(act_std).to(device)
print(f"dataset: {len(np.unique(frames['episode_index']))} episodes / {len(obs)} frames, "
      f"denoising_steps={args.denoising_steps}, model_dim={args.model_dim}")
# --- endregion ---

# --- region: train ---
# Chapter 1.1's loop, one line changed: the loss is diffusion_loss (predict the
# noise on a noised action, conditioned on the obs) instead of MSE on the action.
# Unlike BC/ACT we DON'T cosine-decay the lr — the denoiser must keep fitting noise
# at all levels; decaying it undertrains (measured: it drops below the baseline).
torch.manual_seed(args.seed)  # policy init + training-noise stream reproducible, independent of the toy
gen.manual_seed(args.seed)    # give the policy a fresh noise stream (the toy drained the shared gen above)
policy = Denoiser(ACT_DIM, OBS_DIM, args.model_dim, obs_min, obs_range).to(device)
optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
shuffle = torch.Generator().manual_seed(args.seed + 1)  # torch-side RNG for batch order
train_loss, global_step = float("nan"), 0
for epoch in range(args.epochs):
    epoch_loss, num_batches = 0.0, 0
    for batch in torch.randperm(len(obs_t), generator=shuffle).split(args.batch_size):
        loss = diffusion_loss(policy, act_t[batch], obs_t[batch], schedule)
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
        print(f"epoch {epoch:4d}  diffusion_mse {train_loss:.5f}")
# --- endregion ---

# --- region: eval ---
# Loss measured denoising on dataset states; rollouts measure the task. At every
# env step we SAMPLE an action: start from noise, run the reverse loop conditioned
# on the current obs. --break few_steps runs this at denoising_steps=2 (set in
# setup): too few steps can't resolve the action, it comes out under-denoised, and
# the pusher wanders — a measured drop, not a crash.
def rollout(net: Denoiser, seed: int, tag: str, episode: int) -> tuple[bool, float]:
    env = PushTEnv()
    obs_now = env.reset(seed)
    gen.manual_seed(seed)  # seed the sampler from the episode seed: reproducible AND order-independent, so
    #                        baseline and trained see the same per-episode noise (a fair, seeded comparison)
    done, episode_return, info = False, 0.0, {}
    while not done:
        cond = torch.from_numpy(obs_now).to(device).unsqueeze(0)          # (10,) -> (1, 10)
        sample = p_sample_loop(net, (1, ACT_DIM), cond, schedule)         # in standardized action space
        action = (sample * act_std_t + act_mean_t).clamp(-1.0, 1.0)[0].cpu().numpy()
        obs_now, reward, done, info = env.step(action)
        episode_return += reward
        if args.rerun:
            rr.set_time("sim_time", duration=episode * (PushTEnv.MAX_STEPS / PushTEnv.CONTROL_HZ) + env.data.time)
            rr.log(f"eval/{tag}/action", rr.Scalars(action.astype(np.float64)))
            rr.log(f"eval/{tag}/pos_err", rr.Scalars([info["pos_err"]]))
    return bool(info["success"]), episode_return


def evaluate(net: Denoiser, tag: str) -> tuple[float, float]:
    # 10_000 + offset: held out from demo seeds (0..num_demos) by construction.
    outcomes = [rollout(net, 10_000 + args.seed + ep, tag, ep) for ep in range(args.eval_episodes)]
    success_rate = float(np.mean([s for s, _ in outcomes]))
    mean_return = float(np.mean([r for _, r in outcomes]))
    if args.rerun:
        rr.log(f"eval/{tag}/success_rate", rr.Scalars([success_rate]))
    print(f"eval[{tag:9s}]: success {success_rate:.2f}  mean_return {mean_return:.3f}")
    return success_rate, mean_return


torch.manual_seed(args.seed + 1)  # a fixed random-init reference, independent of eval-time RNG
baseline = Denoiser(ACT_DIM, OBS_DIM, args.model_dim, obs_min, obs_range).to(device)
baseline_success, baseline_return = evaluate(baseline, "untrained")
success_rate, mean_return = evaluate(policy, "trained")
# --- endregion ---

# --- region: report ---
# ONNX/contract note: contract v1 is model(obs[1,10]) -> action[1,2], a single
# stateless step. A diffusion policy does NOT fit v1 twice over: its denoiser takes
# THREE inputs (noisy action, timestep, obs) and one action needs denoising_steps
# of them chained through a reverse loop the runtime must own. We export the
# denoiser CORE and check torch/onnxruntime agree, proving the serialization path;
# a sampler-aware contract (v2) is what the browser needs to drive this. Like ch1.3.
if not args.smoke:
    import onnxruntime as ort  # noqa: E402  (heavy; only when actually exporting)

    onnx_path = args.out / "diffusion_denoiser.onnx"
    policy.eval().to("cpu")
    dummy = (torch.zeros(1, ACT_DIM), torch.zeros(1, dtype=torch.long), torch.zeros(1, OBS_DIM))
    torch.onnx.export(policy, dummy, str(onnx_path),
                      input_names=["noisy_action", "timestep", "observation"],
                      output_names=["predicted_noise"], dynamo=False)  # 3 inputs — NOT contract v1
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    px = np.random.default_rng(0).standard_normal((1, ACT_DIM)).astype(np.float32)
    po = np.random.default_rng(1).standard_normal((1, OBS_DIM)).astype(np.float32)
    pt = np.zeros(1, dtype=np.int64)
    with torch.no_grad():
        delta = float(np.abs(policy(torch.from_numpy(px), torch.from_numpy(pt), torch.from_numpy(po)).numpy()
                             - session.run(None, {"noisy_action": px, "timestep": pt, "observation": po})[0]).max())
    assert delta < 1e-4, f"torch/onnx denoiser parity failed: {delta:.2e}"
    print(f"exported {onnx_path} (denoiser core, 3 inputs — NOT contract v1); parity {delta:.2e}")
    policy.to(device)

metrics = {
    "baseline_mean_return": round(baseline_return, 6),
    "baseline_success_rate": round(baseline_success, 6),
    "break_mode": args.break_mode or "none",
    "denoising_steps": args.denoising_steps,
    "epochs": args.epochs,
    "final_train_loss": round(train_loss, 6),
    "mean_return": round(mean_return, 6),
    "model_dim": args.model_dim,
    "num_demos": len(np.unique(frames["episode_index"])),
    "seed": args.seed,
    "smoke": bool(args.smoke),
    "success_rate": round(success_rate, 6),
    "toy_diffusion_mean_radius": round(diff_r, 6),
    "toy_diffusion_modes_covered": diff_modes,
    "toy_diffusion_ring_hit": round(diff_hit, 6),
    "toy_final_loss": round(toy_loss, 6),
    "toy_regress_mean_radius": round(reg_r, 6),
    "toy_regress_modes_covered": reg_modes,
    "toy_regress_ring_hit": round(reg_hit, 6),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}  (toy: diffusion covers {diff_modes}/8 modes, "
      f"regression {reg_modes}/8 and collapses to r={reg_r:.2f})")
if args.rerun:
    print(f"recording: {args.out / 'diffusion.rrd'} — open it with: rerun {args.out / 'diffusion.rrd'}")
# --- endregion ---
