# 1.5: Generative Policies II — Flow Matching

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## Read this next to 1.4

Chapter 1.4 solved mode-collapse by learning to *denoise*: add Gaussian noise to an
action in graded steps, train a network to predict the noise, then walk pure noise
back to a sample with the DDPM reverse loop. It works — the ring came out multimodal,
the policy beat its baseline. But the machinery is heavy: a noise schedule you have
to get right, dozens of reverse steps, and fresh noise injected at every one that you
then fight back down. This chapter keeps the *idea* — turn noise into a *sample*, so
the model commits to one mode instead of averaging two — and replaces the machinery
with something you can hold in your head.

Open `flow.py` beside `diffusion.py`. Same seven regions, the same 2D ring toy, the
same PushT policy, the same eval. The *mechanism* differs in just two places: the
**objective** (predict a velocity, not the noise) and the
**sampler** (integrate an ODE, don't run the DDPM posterior). Everything else — the
data pipeline, the training loop, the eval rollouts, the ONNX export — is the ch1.4
code, unchanged. This is the chapter: a single, legible swap, measured.

## The swap, in one picture

Diffusion defines a stochastic process — a schedule of `beta`s, an `alpha`-bar, a
posterior variance — and learns to invert it. Flow matching throws all of that away
and draws a **straight line**. Pick a data point, pick a noise point, and connect
them:

    x_t = (1 - t) * noise + t * data          for t in [0, 1]

At `t = 0` you are at the noise the sampler starts from; at `t = 1` you are at the
data. The velocity of a point sliding along that line is constant — it is just
`data - noise`, the same everywhere on the segment. So the whole learning problem
becomes: **predict that velocity.** Put a point on its line at a random `t`, ask the
network for the line's velocity, penalize the squared error. No schedule, no
posterior, no injected noise — the geometry does the work the schedule used to.

## Build

`flow.py` is one file in the same seven regions as ch1.4: setup, core, toy, data,
train, eval, report. The `core` region is where diffusion and flow diverge — and
notice what is *missing* from it versus ch1.4: no `make_schedule`, no `betas`/
`alphas`, no posterior variance. A straight line needs none of them.

### The core: velocity in, sample out

The objective places a sample on its straight noise→data line at a random time and
regresses the network onto that line's constant velocity $x_0 - \epsilon$ — and
here it is in code:

$$
\mathcal{L}_{\mathrm{FM}} = \mathbb{E}_{t,\,x_0,\,\epsilon}\Big[\;\big\lVert\, v_\theta(x_t, t) - (x_0 - \epsilon) \,\big\rVert^2\;\Big],
\qquad x_t = (1-t)\,\epsilon + t\,x_0
$$

```
[include-by-region: flow.py#core]
```

Three functions carry the method. `interpolate` is the straight noise→data line —
one line of code that replaces ch1.4's entire schedule plus `q_sample`.
`flow_matching_loss` is the whole training objective: draw a random time `t`, place
the sample on its line, and predict the line's velocity `data - noise` with plain
MSE. (Compare `diffusion_loss`, which predicted the *noise* on a *scheduled* noising —
that is the ~2-line objective swap.) `ode_sample_loop` is sampling: start at noise,
and take forward-Euler steps `x <- x + dt * v` up the velocity field to `t = 1`. The
entire DDPM reverse step — recover `x0`, clip it, blend via the posterior mean,
re-inject noise — collapses to that one addition. The network itself, `VelocityNet`,
is structurally *identical* to ch1.4's `Denoiser`: the same MLP, the same sinusoidal
time embedding, the same optional obs conditioning. Only its name and what it
predicts change.

One necessary difference from ch1.4: flow time `t` lives in `[0, 1]` — a continuous
position along the line, not an integer step index — so we scale it up by `TIME_SCALE`
before the sinusoidal embedding, or the embedding would barely vary across the whole
interval.

### The toy: same aha, plus efficiency

```
[include-by-region: flow.py#toy]
```

This is the ch1.4 toy with two lines swapped (`flow_matching_loss` for
`diffusion_loss`, `ode_sample_loop` for `p_sample_loop`) — and it makes the same
point. Flow covers **8/8** angular modes of the ring at radius **0.94**; the
same-width MSE regressor — *identical* code to ch1.4's baseline — covers **0/8** at
radius **0.06**, the dead center. The collapse was never about diffusion; it is about
*sampling vs averaging*, and flow samples.

What is new here is the last measurement. We re-sample the *same trained net* at only
**5** Euler steps and it still covers **8/8** modes. Sweep the step count and the
pattern is clean: full mode coverage appears by **3** steps, ring quality keeps
tightening out to ~20 and then saturates, and only **2** steps is too coarse for
Euler to resolve the ring (that is the `few_steps` break). The point is not just "few
steps work" — it is that **sampling steps are decoupled from training**. Diffusion's
step count is welded to its schedule; change it and you retrain. Flow's step count is
a free choice you make at sampling time, from one trained network. That is what
"straighter and faster" actually cashes out to, and it is measured, not asserted.

### The policy: conditioning on the observation

```
[include-by-region: flow.py#data]
```

The data region is ch1.4's, unchanged: the same scripted PushT demos, the same
normalization. The ODE starts from `N(0, I)` exactly as the DDPM sampler did, so the
actions we flow must sit at unit scale too — standardizing them is what takes this
policy from 0% to working, the same lesson as ch1.4.

```
[include-by-region: flow.py#train]
```

The training loop is ch1.4's with one line changed: the loss is `flow_matching_loss`.
As in ch1.4 we do **not** cosine-decay the learning rate — the velocity net must keep
fitting the field at every time `t`, and decaying it undertrains the net.

```
[include-by-region: flow.py#eval]
```

At every environment step we sample an action by integrating the ODE conditioned on
the current observation, un-standardize, clip, step. The trained policy reaches
**40%** held-out success versus **0%** for the untrained net, and it holds across
seeds (0.40, 0.40, 0.55). Now be precise about what that 40% means, because this is
where flow matching is easy to oversell.

It *edges* chapter 1.4's diffusion policy — ~25% at the *same* 100-demo budget — a
modest but real gap, plausibly because straight-line integration is a steadier hand
than DDPM's noisy reverse walk on a task whose good policy is nearly deterministic. It
does **not**, however, beat chapter 1.1's behavior cloning: BC at its full 500-demo
default reaches **62%**, well above flow here, and only at a matched 100-demo budget
does BC fall back into flow's neighborhood. Flow matching does not buy you a better
policy — it buys you *cheaper sampling at the same quality*, which is a different and
more honest claim. And this is 20 episodes (noisy — chapter 1.6 is about exactly
this), so treat even the diffusion gap as a *directional* win, not a headline. The
rock-solid results are the toy and the step-efficiency above.

## Run it

```
python curriculum/phase1_imitation/ch1.5_flow/flow.py --seed 0 --device cpu
```

<!-- wall-clock table renders from wallclock.csv (all tiers measured) -->

| | held-out success | mean return |
|---|---|---|
| untrained (random-init velocity net) | 0% | −106 |
| trained flow policy | **40%** | **−89** |

```
rerun outputs/ch1.5-flow/flow.rrd
```

## What we cut

This is real conditional flow matching, trained the real way, but it is **not** a
full flow-matching policy like pi0, and the gaps are the same shape as ch1.4's:

- **We flow a single action, not an action horizon.** Real flow policies flow a
  *chunk* of future actions jointly and execute them receding-horizon — the source of
  temporal coherence and much of the real performance.
- **The velocity net is an MLP, not a temporal U-Net / transformer.** Legible here;
  a bigger backbone is what you reach for with a horizon and images.
- **We condition on the 10-number state, not images or language.** pi0 conditions a
  flow-matching action head on a VLM backbone — that is chapter 1.8.
- **We integrate with plain forward Euler.** Real implementations use a higher-order
  or adaptive ODE solver (fewer, better steps). Euler is the readable floor; the toy
  already shows 3–5 steps suffice here.

None of these silently degrades a number; each is a whole capability left for later
so the velocity→sample core stays readable. The "read the real thing" segment walks a
production flow-matching repo so you can see exactly what these paragraphs left out.

## Break it

Two ablations, each a real flow-matching misconception with a measured signature.

**`--break few_steps` — "two Euler steps is plenty."** This forces `flow_steps = 2`
at *sampling* time only. Watch two things. First, the toy degrades — 8/8 modes to
**3/8**, radius 0.94 to **0.51** — because two Euler steps overshoot the curved
*marginal* field the net actually learned (each conditional path is straight, but the
field averaged over which endpoint is not). Second, and this is the honest twist: the
*policy is unharmed* (**45%**, if anything up from 40), because the PushT action
distribution is nearly unimodal and even a crude two-step integration lands close
enough. The same two-step regime that *collapsed* ch1.4's diffusion policy to 0%
leaves flow's policy standing. And note the training loss: it is **bit-identical** to
the full run (0.951594), because `flow_steps` is a *sampler* knob that never touches
the objective. That is the decoupling, made unmistakable.

**`--break wrong_target` — "the sign of the velocity can't matter that much."** This
flips the target to `noise - data`, so the net learns to flow *away* from the data.
The ODE then integrates outward: the toy explodes off the ring to radius **4.12**, the
policy drops to **0%** — and the training loss stays *low* (**0.935**), because the net
fits the flipped target perfectly well. Low loss, wrong direction: the sign of the
velocity *is* the direction of the flow, and the loss curve will not warn you. (The
ch1.4 `wrong_schedule` trap, in flow's language.)

## Read the real thing

The paper that named this objective ships a reference library — **`facebookresearch/flow_matching`**, pinned here at commit `11568d3`. It is not a policy; it is the objective and the sampler you just built, generalized and hardened. Read it in three passes, against your `core` region.

**The path and the velocity target.** Your `interpolate` hard-codes one straight
line, `(1-t)*noise + t*x0`, and `flow_matching_loss` regresses its constant velocity
`x0 - noise`. The library's `AffineProbPath.sample()` in
`flow_matching/path/affine.py` builds the same point as `x_t = sigma_t*x_0 +
alpha_t*x_1` and returns its velocity as `dx_t = d_sigma_t*x_0 + d_alpha_t*x_1`
inside a `PathSample` (`flow_matching/path/path_sample.py`); the training loss is one
line in that file's docstring — `mse_loss(path_sample.dx_t, model(x_t, t))`, your
loss exactly. Your straight line is their `CondOTProbPath`, whose `CondOTScheduler`
(`flow_matching/path/scheduler/scheduler.py`) sets `alpha_t=t, sigma_t=1-t,
d_alpha_t=1, d_sigma_t=-1`. What the library adds is *choice*: a scheduler object so
the path can be any affine schedule (there is a `PolynomialConvexScheduler` for
curved ones), plus a full set of conversions — `target_to_velocity`,
`epsilon_to_velocity`, `velocity_to_epsilon` — so you can train in data-, noise-, or
velocity-space and move between them. You cut every bit of it because this chapter is
about the *one* path where none of it is needed.

**The sampler.** Your `ode_sample_loop` is the whole integrator: `x <- x + dt*v`,
`steps` times. That is precisely the `method="euler"` branch of `ODESolver.sample()`
in `flow_matching/solver/ode_solver.py`, which hands the velocity field to
`torchdiffeq.odeint`. What production adds is the rest of the menu — `dopri5`,
`midpoint`, `heun3`: adaptive and higher-order solvers that reach the data in fewer,
better steps than Euler — plus a `compute_likelihood()` that integrates the field's
divergence for an exact log-likelihood, a capability your loop does not have. The toy
already showed 3–5 Euler steps suffice *here*; a stiffer field is where those solvers
earn their keep.

**The horizon and the VLM — honestly, not here.** This is the cut to be precise
about: this repo has *no* action horizon and *no* VLM, because it is a
generative-modeling library, not a robot policy. Those pieces live one repo further
out — in a flow-matching *policy* like pi0 / openpi, built on this exact objective —
and that is chapter 1.8. What this repo *does* show of the production direction is
`examples/image/`: the same velocity loss wired to a real U-Net (`models/unet.py`),
EMA (`models/ema.py`), and distributed training — the scaffolding, minus the robot.

**Read these, in order.** `examples/2d_flow_matching.ipynb` first — your ring toy in
their hands. Then `flow_matching/path/affine.py` and
`flow_matching/solver/ode_solver.py` — your three `core` functions, generalized.
Then `examples/image/train.py`, to watch the same loss survive a real backbone. The
action head that flows a *chunk* of actions on a VLM backbone is where 1.8 picks up.

## Exercises

Four, in `exercises/`. Two ask you to commit to a prediction before the run answers —
the ring's flow-vs-regression modes (the same result as ch1.4, which is the point),
and flow-vs-diffusion in the few-step regime. One is a bug-hunt in the interpolation
(the two time coefficients are swapped, so the path runs data→noise; the loss still
falls). One has you implement the Euler ODE sampler from its definition — the one
update the whole method turns on.

## What's next

You now have two generative policies that *sample* actions — diffusion and flow — and
a ring that shows why sampling beats averaging, plus a step-efficiency result that
says flow gets there in a handful of straight steps. Chapter 1.8 puts this exact
velocity objective to work as the *action head* of a tiny vision-language-action
model: the same `data - noise` target, now conditioned on a VLM backbone instead of a
ten-number state, flowing a chunk of actions instead of one. The sampler you just
wrote is the last piece before the VLA.
