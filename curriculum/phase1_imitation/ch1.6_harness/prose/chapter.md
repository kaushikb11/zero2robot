# 1.6: Evaluation Is Hard

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## The number you have been trusting

Three times now you have read a sentence like this one: "flow reaches 40% held-out
success versus 25% for diffusion at the matched budget." Chapter 1.3 ranked ACT
against a baseline; 1.4 ranked diffusion against its untrained self; 1.5 put flow
next to diffusion. Each headline was a success rate over roughly twenty rollouts,
and here is what this chapter makes rigorous: some of those gaps are decisively real
(ACT's 0.88 against an untrained 0.0), and one is genuinely within the noise (flow's
0.40 against diffusion's 0.25, whose difference interval spans zero). You cannot tell
which is which by reading the point estimates alone. This chapter is the method that
can.

Here is the uncomfortable fact. A success rate is a coin you flipped twenty times.
If the coin comes up heads eight times you write "0.40", but you would not bet your
savings that the coin's true rate is exactly 0.40: eight of twenty is entirely
consistent with a coin whose real rate is 0.25, or 0.55. "0.40" is not a number. It
is the center of a band, and the band over twenty flips is embarrassingly wide.
`harness.py` is the tool that draws the band, and then uses it to answer the only
question that matters: when you say policy A beat policy B, did it?

## Build

`harness.py` is one file in five regions: setup, **stats**, policy, eval, and a long
report region that runs the whole argument. The `stats` region is the chapter: it
is maybe forty lines of numpy, and it is the part you must get exactly right, because
a wrong confidence interval taught as canonical is worse than no interval at all.

### The stats: turning k/n into a band

```
[include-by-region: harness.py#stats]
```

Three functions. `wilson_ci` is the workhorse: given `k` successes in `n` trials it
returns a 95% **Wilson score interval**. You may have seen the naive interval first
(`p_hat ± z·sqrt(p_hat(1-p_hat)/n)`, the "Wald" interval) and it is taught first
because it is wrong in an instructive way. At `k = 0` or `k = n` the `p_hat(1-p_hat)`
term is zero, so the interval collapses to a single point: "zero successes in twenty
episodes, therefore the true success rate is exactly zero, no uncertainty." That is
a lie from twenty coin flips, and it is exactly the bug you will hunt in exercise 4.
Wilson solves for the proportion instead of reading it off; it never collapses and
never leaves `[0, 1]`. The file trusts nothing here: an inline assert checks
`wilson_ci` against the textbook value (zero of ten successes → `[0, 0.2775]`, from
Brown, Cai & DasGupta 2001) on every run.

`bootstrap_ci` reaches the same band by a completely different road: resample the
twenty (or two hundred) 0/1 outcomes with replacement a couple thousand times, take
each resample's mean, and read off the 2.5th and 97.5th percentiles. No formula, no
normal approximation: just "what would this eval have said on a slightly different
draw of the same episodes?" It lands on top of Wilson (on the pooled strong policy,
Wilson `[0.173, 0.288]` versus bootstrap `[0.170, 0.285]`). Two derivations, one
band: that agreement is your evidence that neither is lying.

`diff_ci` is the verdict. Given two policies it returns a confidence interval on the
**difference** of their rates: Newcombe's hybrid-score method, built from the two
Wilson intervals. If that interval excludes zero, the ranking is real at this `n`. If
it contains zero, you have not established the ranking, no matter how far apart the
two point estimates look. That interval, not the gap between the point estimates, is
what you report.

### The policies: the ch1.2 experiment, with error bars

```
[include-by-region: harness.py#policy]
```

We need two real policies to rank, and we build the cheapest honest pair: two tiny
behavior-cloning MLPs (the chapter 1.1 network, trimmed), one trained on `--num_demos`
demonstrations and one on far fewer. This is chapter 1.2's "data is the policy"
result exactly, and back then we could only assert that more data helped. Now we can
put an error bar on it. **The policies are weak on purpose.** BC on the tight,
low-diversity scripted demos saturates around 20% success, and that is fine: low,
close success rates are precisely the regime where single numbers lie loudest. The
subject of this chapter is the band, not the robot.

### The eval: seeded suites and a held-out variant

```
[include-by-region: harness.py#eval]
```

`eval_suite` runs `--n_seeds` independent suites of `--eval_episodes` rollouts each,
on held-out start seeds, and returns a `(n_seeds, eval_episodes)` grid of successes.
That grid is the whole data set of the chapter. One row of it is a single
twenty-episode eval: the thing the earlier chapters reported. Ten rows let you see
how far that single number would have swung if you had drawn a different twenty.

`reset_held_out` is the LIBERO idea in a dozen lines. LIBERO tests a policy on task
configurations it never trained on; we do the one-axis version: spawn the block in
an annulus 0.24–0.30 m from the target, farther out than any training demo (the demos
live at 0.10–0.24 m). Same T, same target, same physics; a start the policy has never
seen. It is a genuine held-out distribution, and it is honest about its scope: one
shifted start on one task, not LIBERO's suites of unseen objects and goals.

## Run it

```
python curriculum/phase1_imitation/ch1.6_harness/harness.py --seed 0 --device cpu
```

<!-- wall-clock table renders from wallclock.csv (measured: cpu-laptop 0.6 min, T4 2.65 min, L40S 1.03 min) -->

The report runs four movements, all measured at seed 0 on CPU.

**[1] The swing.** The strong policy's ten suites of twenty scored anywhere from
**0.05** to **0.30** (std 0.087). A single twenty-episode eval (the number the arc
reported) could have landed on any of those. Before you trust one, look at how far
they disagree.

**[2] Two roads, one band.** Pool all two hundred rollouts and the strong policy's
success rate is **0.225**. Wilson puts a 95% interval of `[0.173, 0.288]` on it; the
seeded bootstrap, resampling those same two hundred outcomes, returns `[0.170, 0.285]`.
The analytic formula and the brute-force resample agree to within a third of a
percentage point. When two derivations that share no algebra land on the same band,
you can stop worrying that the band is an artifact of either one.

**[3] Single numbers lie.** Here is the whole chapter in two rows:

| N | strong | weak | difference CI | verdict |
|---|---|---|---|---|
| 20 | 0.30 `[0.15, 0.52]` | 0.20 `[0.08, 0.42]` | `[-0.17, +0.35]` | **not significant** |
| 200 | 0.23 `[0.17, 0.29]` | 0.10 `[0.07, 0.16]` | `[+0.05, +0.19]` | **significant** |

At twenty episodes the strong policy "beat" the weak one 0.30 to 0.20, and that
ranking is **not established**: the difference interval straddles zero. Pool every
suite to two hundred episodes and the same two policies separate cleanly. Nothing
about the policies changed between the rows. The only thing that changed is how many
episodes you counted. This is the flow-vs-diffusion **0.40 vs 0.25** thread made
rigorous: that ranking was a real question twenty episodes could not answer. (The
ACT-vs-baseline and diffusion-vs-untrained gaps, by contrast, the same method
resolves decisively; the interval is what tells you which of the three gaps are real.)
Notice too that the single twenty-episode suite read **0.30** while the true pooled
rate is **0.23**: the one eval you would have run overstated the policy. It was a
lucky twenty.

**[4] Held-out.** The same strong policy scored **0.23** `[0.17, 0.29]` on
train-distribution starts and **0.08** `[0.05, 0.13]` on the held-out variant. The
gap's confidence interval is `[+0.08, +0.21]`: it excludes zero, so the
generalization drop is real, with its own error bar, because it too is a success
rate. "Our policy gets 23%" was true and useless until you asked: on which starts?

```
rerun outputs/ch1.6-harness/harness.rrd
```

## What we simplified

The evaluation ideas here are the real ones, but the harness is a teaching floor:

- **One held-out axis, not a suite.** We shift the block's start radius. LIBERO holds
  out whole families of tasks (unseen objects, goals, spatial arrangements) across
  dozens of tasks. Generalization has many axes; we measure one. The "read the real
  thing" segment walks LIBERO's held-out suites so you see the full version.
- **95% intervals, one comparison at a time.** We ship Wilson (and cross-check with a
  bootstrap). We do not correct for testing many policies at once (a Bonferroni or
  Holm correction), which you would need the moment you rank five policies instead of
  two. Named here, left for later.
- **Success is binary.** We collapse each episode to succeeded / did not. Real eval
  also reports partial progress, time-to-success, and smoothness: distributions, not
  a single Bernoulli. The binomial machinery here is the honest minimum, not the ceiling.
- **The policies are tiny and weak.** By design: the statistics are the subject.

None of these hides a number. Each is a whole dimension of evaluation left visible
for later so the `k/n → band → verdict` core stays readable.

## Break it

**`--break too_few`: "five episodes is a number."** This shrinks every suite to five
episodes and reports the same strong-vs-weak comparison. The difference CI balloons to
about **0.92 wide** and still spans zero: you can read a success rate off five
rollouts, but it is not evidence of anything. This is the main result pushed to its
limit: too few episodes is not a smaller good eval, it is not an eval. The fix is
never a cleverer statistic; it is more episodes.

## Read the real thing

Our held-out eval is one shifted start radius on one task. LIBERO is the same idea
built as a benchmark. `meta.yaml` pins it to `Lifelong-Robot-Learning/LIBERO@8f1084e3`:
read these three files against what you just built.

**The held-out suites.** `reset_held_out` perturbs a single annulus. LIBERO registers
whole suites of held-out tasks in `libero/libero/benchmark/__init__.py`: the
`@register_benchmark` classes `LIBERO_SPATIAL`, `LIBERO_OBJECT`, and `LIBERO_GOAL`,
each holding one axis fixed and varying another: Spatial keeps the objects and goal
and moves the layout, Object swaps the object under a fixed layout, Goal keeps the
scene and changes the instruction. Each suite is a list of `Task` tuples built from
`libero_task_map`, and every task carries its own BDDL scene file and a saved
initial-states file, so a "held-out start" there is not a radius we tweak, it is a
versioned initial-state distribution shipped with the task. That is what one line of
`reset_held_out` stands in for: a great deal of authored variation, pinned so the eval
reproduces.

**The eval loop.** Our `eval_suite` runs `n_seeds × eval_episodes` rollouts and hands
back a boolean grid. Theirs is `evaluate_one_task_success()` in
`libero/lifelong/metric.py`: it spins up parallel environments (`SubprocVectorEnv`),
steps them to `cfg.eval.max_steps`, marks each episode done with a success flag, and
returns `num_success / cfg.eval.n_eval`. The default `n_eval` is **20**
(`libero/configs/eval/default.yaml`): the same twenty this chapter spent four
movements distrusting. `evaluate_success()` wraps that over every task in the suite and
returns `np.array(successes)`, one rate per task; the single-task entrypoint
`libero/lifelong/evaluate.py` computes the same `num_success / env_num` for one task.

**The reported metric.** Here is the gap worth seeing. LIBERO reports the **mean** of
those per-task rates (one number per suite) and the lifelong story layers
forward-transfer and forgetting on top. It does not ship a confidence interval. Twenty
rollouts per task, averaged across tasks, reported as a point estimate: exactly the
object `harness.py` puts a band around with `wilson_ci` and `bootstrap_ci`. Their
machinery (vectorized envs, BDDL suites, saved init states) is the production
scaffolding for running the eval at scale; the `k/n → band → verdict` step is the one
this chapter adds and they leave to the reader. Neither is missing the other's point:
they built the harder thing (a real held-out benchmark), we built the honest error bar
around its headline.

**Read next:** open `libero/lifelong/metric.py` and find the line
`success_rate = num_success / cfg.eval.n_eval`. That divide is the whole of this
chapter: put a Wilson interval around it and you know whether two suite means actually
differ.

## Exercises

Four, in `exercises/`. Two are the harness grading the harness: you predict, before
running, whether a claimed strong-vs-weak ranking is significant at N=20 versus N=200
(it is not, then it is), and whether the held-out drop is real (it is). One has you
implement the Wilson interval from its formula: the CI the whole chapter rests on.
One is a bug-hunt in a plausible confidence interval: it is the Wald interval, which
reports "0% ± 0%" on zero of twenty successes; you replace it with Wilson so the band
stays honest at the boundary.

## What's next

You now have the one tool the rest of the book cannot do without: an honest success
rate. Every ranking from here (PPO's tricks ablated in chapter 2.1, the domain-
randomized policy in 2.7, offline RL against behavior cloning in chapter 4.1)
is a claim that one number beat another, and you now know that a claim like that is
empty without an `n` and a band. The RL fork takes a thirty-minute excerpt of this
chapter (seeded suites and success-rate bands) as its on-ramp before chapter 2.1,
for exactly this reason: you cannot tell whether a reward change helped until you can
tell whether two success rates differ.
