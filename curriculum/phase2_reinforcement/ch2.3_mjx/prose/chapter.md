# 2.3 — 4096 Robots at Once: PPO on MJX

## You have been training one robot at a time

In chapter 2.1 you built PPO and it worked: the cartpole learned to balance by
acting and watching the consequences. Look at how it collected its experience,
though — eight envs in a plain Python `for` loop, stepped one after another. That
is how you learn the algorithm. It is not how the field runs it.

Real reinforcement learning scales the *other* way: not a bigger network, but more
robots. Thousands of simulated robots stepping in parallel on one accelerator, so
the policy sees millions of transitions per second and a run that would take hours
finishes in minutes. The question this chapter answers is simply: how? And what
does it cost?

## Meet the other framework the field uses

Here is the honest reason this chapter looks different from the ~22 before it. The
tool that runs MuJoCo massively in parallel is **MJX** — MuJoCo-XLA — and MJX is
**JAX**, not PyTorch. There is no parallel MuJoCo in torch; teaching "4096 robots
at once" in torch would be teaching a fiction (see `infra/decisions/015`). So this
is the curriculum's one deliberate excursion into JAX — the same way chapter 1.9
graduated you from hand-rolled training loops to LeRobot. The field genuinely runs
both: torch for policies (ACT, diffusion, VLAs), jax for parallel sim and RL
(Brax, MJX). You should meet the second one once, on purpose.

You are not leaving the algorithm behind. `ppo_mjx.py` is chapter 2.1's PPO — the
same Gaussian policy, the same GAE with the same termination-vs-truncation
bootstrap, the same clipped surrogate and minibatch epochs. What changes is the
*idiom*: from imperative torch to functional JAX.

## The mental-model shift (read this before the code)

If you are torch-native, four things will feel wrong at first, and each is the
point:

- **Functions are pure. Params are data.** A torch module *owns* its weights and
  you mutate them in place (`opt.step()`). In JAX the network is a pure function
  and its parameters are a plain pytree you pass *in* and get *out*:
  `model.apply(params, obs)`. Every update returns a *new* params pytree; nothing
  is mutated.
- **Randomness is explicit.** No hidden global seed. You carry a `PRNGKey` and
  `split` it every time you sample. That is *why* CPU JAX replays bit-for-bit: the
  randomness is threaded, not ambient.
- **`vmap` is the parallelism.** You write the env and the rollout for *one*
  world. `jax.vmap` turns it into N worlds by adding a leading batch axis — no env
  loop. This one transform is how a single device runs 4096 cartpoles.
- **`jit` compiles the whole step.** `jax.jit` traces the entire
  rollout + GAE + update into one XLA program and compiles it. The first call is
  slow (compilation); every call after is the payoff. That compile-once /
  run-fast shape is exactly why throughput *cliffs*, as you are about to measure.

The file keeps this primer inline so you can refer to it beside the code:

```
[include-by-region: ppo_mjx.py#primer]
```

## The env, re-expressed as pure functions

The cartpole is the same MJCF you have used since chapter 2.1
(`common/envs/cartpole/cartpole.xml`), now loaded into MJX with `mjx.put_model`.
We do *not* reuse the `CartpoleEnv` class — that one is imperative C-MuJoCo plus
numpy. Instead reset, step, and obs become pure functions on an `mjx.Data`, so
`vmap` and `jit` can batch them over thousands of worlds:

```
[include-by-region: ppo_mjx.py#env]
```

The last two lines are the whole trick: `jax.vmap(env_step)` is a function that
steps *all N envs at once*. `tree_select` is the parallel autoreset — where an
episode ended, it swaps in that env's fresh reset, leaf by leaf across the whole
`mjx.Data` pytree.

## The same PPO, now functional

The rollout is a `lax.scan` over timesteps instead of a Python loop, but every
line maps to something you wrote in 2.1: sample an action, step the envs, record
the transition, and store the bootstrap value of the real next state for the GAE.
The termination-vs-truncation distinction you fought for in 2.1 is here unchanged
— `truncated` (time ran out) still bootstraps; `terminated` (the pole fell) still
does not:

```
[include-by-region: ppo_mjx.py#rollout]
```

The update is PPO's clipped surrogate, now written as a pure loss function that
`jax.value_and_grad` differentiates and `optax` applies — the same objective, the
same advantage normalization / value clipping / entropy terms:

```
[include-by-region: ppo_mjx.py#update]
```

And the payoff of the JAX idiom: the entire rollout → GAE → K epochs of minibatch
SGD is *one* jitted function. Call it once per iteration; XLA runs the whole thing
as a single compiled program:

```
[include-by-region: ppo_mjx.py#build]
```

## Run it

```
python curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py --seed 0
```

The default is a **small, CPU-jax, free-tier** config — 64 parallel envs — and it
learns cartpole in about **0.49 min (measured, cpu-laptop)**:

```
iter  10/36  return   54.4     80,445 env-steps/s
iter  20/36  return  216.6     81,794 env-steps/s
iter  35/36  return  312.5     81,078 env-steps/s
eval: mean return 407.2 over 64 envs (random ~10-30, cap 500)   # seed 0
```

(Seed 1 solves outright at 499.7 — RL is graded on the seeded band, not one run;
chapter 1.6's determinism tiering is why.) It is honestly small: 64 envs is not
the 4096 in the title. Which is the whole next lesson.

## The wall-clock cliff

MJX's headline is throughput. Measure it directly — `--sweep` times one update
step at each env count and reports env-steps per second:

```
python curriculum/phase2_reinforcement/ch2.3_mjx/ppo_mjx.py --sweep 16,64,256,1024
```

```
   num_envs   env-steps/sec   sec/step
         16          50,154      0.041
         64          83,638      0.098
        256         104,544      0.313      <- CPU peak
       1024          76,387      1.716      <- past the plateau, throughput FALLS
```

There is the cliff. Throughput climbs steeply as you add parallel envs — the
parallelism win is real — but on a CPU laptop it **plateaus around 256 envs and
then reverses**: a handful of cores cannot actually run 1024 worlds at once, so
adding more only adds queueing. This is the honest free-tier result. On a GPU the
same curve keeps climbing to 4096 and beyond, which is exactly why the 4096-robot
regime is this chapter's **Scale Lab** (`--platform gpu --num_envs 4096`, needs a
`jax[cuda]` install), not its free-tier default.

## Throughput is not learning

Here is the trap the throughput number sets, and the deeper lesson. More envs give
you more env-steps per second — so more envs is strictly better, right?

Not for *learning* at a fixed data budget. `--total_steps` is a budget of
env-steps, and each PPO iteration spends `num_envs * num_steps` of it. So doubling
`num_envs` at a fixed budget *halves the number of gradient updates*. Measured, at
the default 300k-step budget:

| num_envs | env-steps/s | gradient updates | eval return |
|---------:|------------:|-----------------:|------------:|
|       64 |      ~84k   |               36 |     **407** |
|      256 |     ~104k+  |                9 |      **90** |

The 256-env run is *faster* and learns *worse* — same data, a quarter of the
updates, nowhere near solved. That is the throughput-vs-gradient-quality tradeoff:
parallel envs buy you data per second, but each gradient is one big, correlated
batch, and past some point more parallelism stops buying you *learning*. On a GPU
you would raise `total_steps` to give the 4096-env run enough updates — the
tradeoff never vanishes, it just moves. (Investigate it yourself in exercise 2.)

## Determinism, honestly (same rule as chapter 1.6)

With `--platform cpu` (the default) and a fixed `--seed`, this file is
**bitwise-reproducible**: two runs produce byte-identical metrics, because every
source of randomness flows from one `PRNGKey`. That is CI-enforced. On a **GPU**,
JAX/XLA kernels are only **statistically** reproducible — same seed, same
qualitative result within the seeded band, not the same bytes. This is the exact
tiering chapter 1.6 taught; the paradigm changed but the honesty rule did not.

## Read the real thing

`ppo_mjx.py` is the from-scratch version; the field's parallel-PPO code is the
same shape, hardened. Three implementations are worth reading, all JAX:

- **PureJaxRL** (`luchris429/purejaxrl`) — the closest match to this file: a single
  `jax.jit`-compiled PPO where the entire rollout-plus-update is one XLA program,
  exactly the idiom you just built.
- **Brax PPO** (`google/brax`, `brax/training/agents/ppo`) — the canonical
  parallel-env PPO, with the vectorized envs and autoreset generalized well past
  one cartpole.
- **MJX** (`google-deepmind/mujoco`, `mjx/`) — the sim underneath everything here:
  read `mjx.step` and `mjx.make_data` to see the physics you vmapped.

The guided-reading segment pins the exact upstream commit to study against.

## Scale Lab

The 4096-robot headline is a **GPU** study, not a free-tier run. With a
`jax[cuda]` install on a 4090 or L40S (`infra/decisions/014`), re-run the sweep out
to 4096 envs — `--platform gpu --num_envs 4096` — and chart where the throughput
curve keeps climbing past the CPU plateau you measured near 256. Then repeat the
throughput-vs-gradient-quality table at a GPU-scale `--total_steps`, large enough
to give the 4096-env run the gradient updates it needs: the tradeoff does not
vanish on a GPU, it moves. These numbers stay **PENDING** in `wallclock.csv` until
measured on a real GPU runner — this chapter does not estimate them.
