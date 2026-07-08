# 1.4: Generative Policies I — Diffusion

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## See it work

Forget the robot for a minute. Here is a cloud of two-dimensional points arranged
in a ring — a unit circle, every direction around it equally likely. It is a
stand-in for a decision with several equally-good answers: at this state you could
go this way, or that way, or any of a hundred ways around, and each is fine. The
one place you should *never* end up is the empty center, which is the average of
all of them and a member of none.

Train two little networks on that ring. They have the same width, they see the same
points. The first is a *regressor*: map an input to a point, minimize squared
error. The second is a *diffusion model*: learn to turn noise into a *sample* from
the ring. Then draw two thousand points from each and look. The diffusion model's
points land on the ring, all the way around it — they occupy **8 of 8** angular
sectors at mean radius **0.87**. The regressor's points sit in a tight blob at the
origin — **0 of 8** sectors, mean radius **0.06**. It collapsed to the one place no
data lives, because the average of "every direction" is "no direction."

That blob is the same failure you have been fighting since chapter 1.1. There, a
behavior-cloning policy at a fork averaged "go left" and "go right" into "drive
into the middle and get stuck." Chapter 1.3 hid the seam with chunking but never
removed the collapse — it is baked into the *objective*. Squared (or absolute)
error asks for the mean, and the mean of a multimodal target is a lie. Diffusion
changes the objective so the network learns to *sample* instead of average. That is
the whole chapter. The ring is where you see it; PushT is where you use it.

## The problem

Every policy so far answers a question of the form "given this observation, what
action?" with one deterministic action, fit by pushing the prediction toward the
demonstrated action under a squared or absolute penalty. That penalty has a precise
and unavoidable optimum: the conditional *mean* action. When the demonstrator was
consistent, the mean is a fine answer. When the demonstrator (or the task) admits
several good actions from the same state, the mean is a point *between* them — and
between two ways around an obstacle is *into* it.

You cannot fix this by making the network bigger or training it longer; a sharper
fit to the mean is still the mean. You have to change what the network is asked to
produce: not the average action, but a *draw* from the distribution of good
actions. A model that can sample "left" half the time and "right" half the time —
and crucially, never "straight into the middle" — is a *generative* model. Diffusion
is the one we build here.

## Build

`diffusion.py` is one file, about 450 lines, in seven regions: setup, core, toy,
data, train, eval, report. The `core` region is the diffusion method itself — a
noise schedule, the forward noising, an epsilon-prediction denoiser, and the
reverse sampling loop — shared by the toy and the policy. Nothing is imported to
hide it: no `diffusers`, no `einops`, no sampler you cannot read.

### The core: noise in, sample out

The forward process $q(x_t \mid x_0)$ jumps straight to any noise level in one
shot, and the reverse loop steps back along the DDPM posterior mean $\tilde{\mu}_t$
— and here they are in code:

$$
\begin{aligned}
x_t &= \sqrt{\bar{\alpha}_t}\,x_0 + \sqrt{1-\bar{\alpha}_t}\;\epsilon,
   & \epsilon &\sim \mathcal{N}(0,\mathbf{I}) \\[4pt]
\tilde{\mu}_t(x_t, x_0) &= \frac{\sqrt{\bar{\alpha}_{t-1}}\,\beta_t}{1-\bar{\alpha}_t}\,x_0
   \;+\; \frac{\sqrt{\alpha_t}\,(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t}\,x_t
\end{aligned}
$$

```
[include-by-region: diffusion.py#core]
```

Diffusion has two processes. The **forward** process takes a clean sample and adds
Gaussian noise in graded steps until nothing is left but noise; `q_sample` jumps
straight to any noise level `t` in one shot, mixing signal and noise so their
powers always sum to one (that is why the coefficients are `sqrt(acp)` and
`sqrt(1-acp)` — exercise 3 breaks exactly this). The schedule that controls how
fast the signal fades is a *cosine* schedule, chosen because it drives the surviving
signal to ~0 by the last step even at modest step counts, so sampling can honestly
begin from pure `N(0, I)`.

The **reverse** process is what we actually want: start from noise and walk *back*
to a sample. `p_sample_loop` runs it one step at a time. At each step the denoiser
predicts the noise in the current point; from that we recover a guess of the clean
sample `x0`, **clip it** to a sane scale (a small model can hallucinate a wild `x0`
early on, and clamping keeps the trajectory on the data manifold — this one line is
the difference between a policy that works and one that doesn't), and step to a
slightly-less-noisy point via the DDPM posterior mean. The denoiser itself
(`Denoiser`) is deliberately just an MLP: a sinusoidal embedding of the timestep so
one network can handle every noise level, an optional conditioning vector, three
linear layers. The lesson is the objective and the sampler, not the architecture.

### The toy: seeing multimodality

```
[include-by-region: diffusion.py#toy]
```

This is the aha, and it is worth running before you read on. We sample the ring,
train the denoiser on it with `diffusion_loss` (noise a point, predict the noise —
that's the entire objective), and *separately* train a same-width MLP as a one-shot
regressor. Then we sample both and measure. `modes_covered` bins the on-ring
samples into eight angular sectors; `mean_radius` says how far from the center they
sit. Diffusion: **8/8** modes, radius **0.87**. Regression: **0/8**, radius
**0.06** — the dead center. Same width, same data, opposite outcome,
because one learned to sample and the other learned to average. Open the recording
and scrub the `denoise_step` timeline to watch a cloud of pure noise resolve into
the ring.

### The policy: conditioning on the observation

```
[include-by-region: diffusion.py#data]
```

The policy trains on the same scripted PushT demos as chapter 1.1. The only new
data step is a normalization the sampler forces on us: the reverse process starts
from `N(0, I)`, so the actions we diffuse must live at unit scale too. These expert
velocities have a standard deviation around 0.3, and leaving them there strands the
sampler in a scale mismatch — it pins the policy at 0% success until we
*standardize* the actions to zero mean and unit variance. So we standardize the
actions we diffuse and min-max normalize the observation we condition on.

```
[include-by-region: diffusion.py#train]
```

The training loop is chapter 1.1's with a single line swapped: the loss is
`diffusion_loss` — noise the demonstrated action to a random level, condition the
denoiser on the observation, predict the noise — instead of MSE on the action.
There is one deliberate difference from BC and ACT worth flagging: we do **not**
cosine-decay the learning rate. A denoiser has to keep fitting noise across every
level, and decaying the rate to zero undertrains it: a decayed run actually finishes
*below* the untrained baseline, while a constant rate clears it across seeds. It is a
small thing that quietly decides whether the chapter works.

```
[include-by-region: diffusion.py#eval]
```

At every environment step we *sample* an action: start from noise, run the reverse
loop conditioned on the current observation, un-standardize, clip to the action
range, step. The sampler is seeded from each episode's reset seed, so evaluation is
reproducible and the trained-vs-baseline comparison is fair (both see the same
per-episode sampling noise).

## Run it

```
python curriculum/phase1_imitation/ch1.4_diffusion/diffusion.py --seed 0 --device cpu
```

<!-- wall-clock table renders from wallclock.csv (cpu-laptop, T4, L40S all measured) -->

The result at the default config (seed 0, CPU):

| | held-out success | mean return |
|---|---|---|
| untrained (random-init denoiser) | 0% | −120 |
| trained diffusion policy | **25%** | **−112** |

The trained policy clears the untrained baseline on both counts, and it holds
across seeds. Be honest about the number, though: 25% is *below* what plain behavior
cloning reaches on the same small budget (around 30%). That is not a bug and it is
not a disappointment — it is the lesson. Single-action diffusion pays for its
generality with *sampling noise*: every action is a draw, and on a task whose best
policy is nearly deterministic, BC's clean conditional mean is simply a steadier
hand. The place diffusion's
sampling *wins* is exactly the place BC's mean *loses* — genuine multimodality —
and PushT's scripted expert is too consistent to show it. The ring already showed
it; chapter 1.5 sharpens the sampler, and real Diffusion Policy (below) wins by
committing a whole horizon of actions per sample instead of one.

```
rerun outputs/ch1.4-diffusion/diffusion.rrd
```

## What we cut

This is real DDPM, trained the real way, but it is **not** the full Diffusion
Policy, and the gaps matter enough to name:

- **We denoise a single action, not an action horizon.** Real Diffusion Policy
  denoises a *chunk* of the next several actions jointly and executes them
  receding-horizon. That is where its temporal coherence — and much of its real
  performance — comes from, and it is why it tolerates sampling noise that hurts our
  single-step version. Adding a horizon dimension to the diffused vector is the
  single highest-value next step, and it is the 4090 Scale Lab the map calls for.
- **The denoiser is an MLP, not a temporal U-Net.** Real Diffusion Policy uses a
  1-D convolutional U-Net over the action horizon with FiLM conditioning. Our MLP is
  legible; a U-Net is what you reach for once there is a horizon to convolve over.
- **We condition on the 10-number state, not images.** No camera, no ResNet.
- **We use plain DDPM sampling.** No DDIM (faster deterministic sampling), no EMA of
  the weights (a standard stabilizer). Both are real, both are omitted for legibility.

None of these is an approximation that silently degrades a number; each is a whole
capability left for later, on purpose, so the noise→sample core is readable. The
"read the real thing" segment walks Chi et al.'s repo so you can see exactly what
these paragraphs left out.

## Break it

Two ablations, each a real diffusion misconception with a measured signature (seed 0,
CPU).

**`--break few_steps` — "two denoising steps is plenty, it's faster."** This forces
`denoising_steps = 2` for the whole run. Watch the trap: the *training loss goes
down* (0.32 → **0.14**) because a 2-step schedule is an easier function to fit — and
the *samples get worse*. The toy drops from 8/8 modes to **7/8** and its radius
falls from 0.87 to **0.70** as points get pulled off the ring; the policy collapses
from 25% to **0%**. Denoising steps are not a speed knob you can turn to nothing:
they are how many chances the reverse process gets to walk noise back to the data,
and two is not enough. The low-loss/bad-sample disconnect is the whole point — your
training curve will not warn you.

**`--break wrong_schedule` — "any increasing noise schedule will do."** This swaps
in near-zero betas, so the forward process barely adds noise and `acp` never
approaches zero. The denoiser is then only ever asked to remove a whisper of noise,
and it never learns the trip back from the pure `N(0, I)` that sampling starts from.
The signature is unmistakable: training loss sits near **1.0** (the model cannot fit
the task at all), the toy overshoots the ring to radius **1.21**, and the
policy is at **0%**. The schedule is not decoration; it defines the very distribution
the sampler begins in.

The transferable lesson: in diffusion the *sampler* and the *schedule* are part of
the model, not afterthoughts. A perfect denoiser with too few steps or a broken
schedule samples garbage, and — the `few_steps` trap again — the training loss is
the last thing that will tell you.

## Read the real thing

"What we cut" named the gaps; this is where you see them in Chi et al.'s code. We
read `real-stanford/diffusion_policy` pinned at commit `5ba07ac`. Three focus
points, each our version set beside theirs — teaching floor against production, not
worse against better.

**The reverse sampler and the denoiser.** Ours is `p_sample_loop` in the `core`
region: a hand-written loop that, at each step, predicts the noise with our MLP
`Denoiser`, recovers and clips `x0`, and steps via the DDPM posterior — the whole
method on the page. The real version is `conditional_sample` in
`diffusion_policy/policy/diffusion_unet_lowdim_policy.py` (lines 59–96 @ `5ba07ac`).
It is the *same* loop — start from `torch.randn`, iterate over timesteps, predict,
step — but each denoising step is delegated to `diffusers`' `DDPMScheduler.step`,
and the network it calls is a `ConditionalUnet1D`
(`diffusion_policy/model/diffusion/conditional_unet1d.py`): a temporal 1-D U-Net
with FiLM conditioning, not an MLP. What they add is a library scheduler you can
swap DDPM for DDIM without touching the loop, and a convolutional net that shares
structure across time. We cut both — our own loop, our own MLP — because there is no
time axis to convolve over and no library worth hiding, which is exactly why reading
it here lands.

**The action horizon.** Our `eval` region samples one action per environment step:
`p_sample_loop(net, (1, ACT_DIM), ...)`. The real `predict_action` (same file, lines
99+) samples a whole *trajectory* of shape `(B, horizon, action_dim)` in one
denoise, then executes only a slice — `action_pred[:, start:end]`, `n_action_steps`
of it — before re-planning (receding-horizon control). This is the single biggest
thing we dropped, and the one that most explains why our 25% trails BC while real
Diffusion Policy leads: denoising a *coherent chunk* both smooths the trajectory and
dilutes the per-action sampling noise that punishes our single-step sampler.

**EMA.** We evaluate the raw trained weights straight out of the `train` region.
Real Diffusion Policy keeps a shadow copy — `EMAModel` in
`diffusion_policy/model/diffusion/ema_model.py` — whose `step` (lines 56–88) blends
each parameter toward the live one under a warmed-up decay (`get_decay`, climbing
toward 0.9999), and evaluates *that*. It is a standard stabilizer: cheap, no effect
on the objective you just learned, a few real points at the end. We omit it so the
training loop stays the handful of lines you can read in one breath.

None of these makes our version wrong; each is production hardening wrapped around
the identical noise→sample core. Read next, in order: `conditional_sample` first
(you already know the loop it runs), then `predict_action` for the horizon slice,
then `conditional_unet1d.py` for the U-Net, and `ema_model.py` last.

## Exercises

Four, in `exercises/`. Two ask you to commit to a prediction before the run is
allowed to answer — the ring's diffusion-vs-regression modes, and what two denoising
steps do to the ring. One is a bug-hunt in the forward-noising equation (one wrong
coefficient, loss still falls, samples still bad). One has you implement the reverse
posterior-mean step from its definition — the one line the whole sampler turns on.

## What's next

You now have a policy that *samples* actions instead of averaging them, and a ring
that shows you why that matters. But DDPM's reverse loop is a long, slightly fussy
walk — dozens of steps, a schedule to get right, a sampler that injects noise you
then have to fight. Chapter 1.5 keeps the exact same file structure and the exact
same task and changes about sixty lines: it replaces the denoising objective with
*flow matching*, the pi0-style objective that learns a straight-line path from noise
to data. Same generative idea, a cleaner and faster sampler — and the site renders
the two files as a diff so you can see precisely what changed.
