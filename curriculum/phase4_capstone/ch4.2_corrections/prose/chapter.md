# 4.2 — Corrections: Human-in-the-Loop Data

## The death you diagnosed, now the fix

In chapter 1.1 you watched behavior cloning die, and you named the cause:
covariate shift. BC only ever sees the states the demonstrator visited. At deploy
time the policy drifts a little, lands in a state no demo covered, guesses worse,
drifts further — and the block slides past the goal. You could not fix it by
cloning harder, because the policy never *acts* during training, so it never sees
the states its own mistakes create.

Phase 2 fixed this by throwing the demonstrator out and letting the policy act
for reward (PPO, SAC). This chapter fixes it a different way — the way you reach
for when you have an expert but not a reward function, which is most of real
robotics. Keep the demonstrator. Just stop asking it the wrong question.

BC asks the expert: *"show me some good trajectories."* The trouble is those
trajectories only cover the expert's states. DAgger (Dataset Aggregation, Ross,
Gordon & Bagnell 2011) asks the expert a better question: *"here is the state MY
policy just wandered into — what would you do HERE?"* Then it trains on the
answer. The loop is almost embarrassingly direct:

1. **Roll out the current policy** and record the states it actually visits —
   including the drifted, off-distribution ones BC never saw.
2. **Ask the expert** what to do on each of those visited states.
3. **Aggregate** the new `(state, expert-action)` pairs into the dataset.
4. **Retrain** and repeat.

The policy learns to recover from its *own* mistakes, because its own mistakes
are exactly what the dataset now covers.

## The expert is a stand-in for your hand

Here the "expert" is the scripted PushT controller from `common/envs/pusht` — the
same one that generated the demos in chapter 1.1. In this chapter it plays the
role of a **human teleoperator** correcting the robot through the browser
playground. That substitution is honest, not a shortcut: the mechanism is
identical — *label the state the policy is in with a good action*. The browser
follow-up (see `demo/embed.yaml`) swaps the scripted labeler for your hand on the
mouse and changes nothing else in the loop. We use the scripted expert because it
is free, deterministic, and lets the whole chapter run on a CPU laptop; the
lesson transfers verbatim to the human-in-the-loop version.

That substitution also makes DAgger's one hard requirement impossible to miss —
and it is the price you pay for the recovery. Behavior cloning asks the expert
for a fixed batch of demonstrations *once*, offline, and never needs it again.
DAgger needs an expert it can **query on any state the policy wanders into**,
round after round — an *interactive*, queryable expert. When that expert is a
person, those queries are real human labor in the loop. When you have only a
frozen dataset and no way to ask "what would you do *here*", DAgger simply does
not apply — and you are back to the offline-only tools of Phase 1.

## Making covariate shift measurable

There is a catch we have to handle honestly. On the full PushT task, a reactive
MLP trained on enough demos already limps to ~0.24 success — covariate shift is
real but *mild*, and mild effects drown in the seed-to-seed noise of a free-tier
eval (this is the exact trap chapter 1.6 taught you to respect). To *see* the
recovery, we make the covariate shift bite, using chapter 1.6's held-out device:

- The BC demos come from a **narrow region** — the block only ever starts *close*
  to the goal (`--r_max 0.13`). This is the demonstrator's limited practice set.
- We **deploy on the full task**, where the block starts anywhere in the annulus.

BC never saw the far starts. It covariate-shifts hard and fails. And crucially,
*more narrow demos cannot fix it* — they still do not cover the far starts. Only
data collected where the policy actually goes can. That is the gap DAgger fills.

## The shape of the file

`dagger.py` reuses `common/`: the PushT env, its scripted expert, the seeding
helper, the device banner. The policy is chapter 1.1's, unchanged — a reactive
MLP with normalization baked in as buffers. **DAgger changes the data, never the
policy.** That is the whole point.

```
[include-by-region: dagger.py#policy]
```

The corrector collects demos from the narrow practice region, then labels the
policy's on-policy rollouts. The one subtlety is the **beta-mixture**: early
rounds execute mostly the *expert* (`beta` starts at 1.0 and decays), so the
visited states stay near the manifold instead of flooding the dataset with
far-off-manifold flailing; later rounds hand control to the policy and collect
corrections on the states *it* causes.

```
[include-by-region: dagger.py#data]
```

The loop itself is four lines of intent — correct, aggregate, retrain, evaluate —
wrapped around the Wilson intervals from chapter 1.6, because a recovery you
cannot separate from noise is not a recovery.

```
[include-by-region: dagger.py#loop]
```

## Run it

```
python curriculum/phase4_capstone/ch4.2_corrections/dagger.py --seed 0 --device cpu
```

On a CPU laptop this takes about **1.5 min** at the default config — this chapter
lives comfortably on the free-tier floor.

<!-- wall-clock table renders from wallclock.csv (measured: cpu-laptop 1.51 min; the reference run is CPU) -->

The success rate is graded on 200 held-out rollouts (20 suites × 10), each with
its Wilson interval. Measured (seed 0, default config):

```
round 0 (BC    ) success  13/200 = 0.065  CI [0.038, 0.108]   <- covariate shift
round 1 (DAgger) success  17/200 = 0.085  CI [0.054, 0.132]
round 2 (DAgger) success  31/200 = 0.155  CI [0.111, 0.212]
round 3 (DAgger) success  43/200 = 0.215  CI [0.164, 0.277]   <- recovered
round 4 (DAgger) success  17/200 = 0.085  CI [0.054, 0.132]   <- over-iterated

BC 0.065 -> best DAgger3 0.215;  recovery diff CI [+0.08, +0.22]  (excludes 0)
```

BC sits at 0.065 and its interval tops out at 0.11. The best DAgger round reaches
0.215 with a lower bound of 0.16 — the two intervals **do not overlap**, and the
difference interval excludes zero. The policy did not get a bigger network or a
reward function; it got the states it was actually failing in, labeled. That is
the recovery, and it holds on every seed 0–2 (BC ~0.06, best DAgger ~0.18–0.22,
diff CI excludes zero every seed).

## More is not better — select the best round

Look again at round 4: it fell back to 0.085. Aggregating corrections onto a
still-weak reactive policy eventually **floods** the dataset with its own long
failure trajectories, and the fit regresses. The round that peaks is not always
the last — seed 0 peaks at round 3, seed 1 at round 4. This is not a bug to tune
away; it is what Ross et al. tell you to expect, which is why the algorithm
**returns the best policy over rounds**, not the final one.

One honesty note, because honest numbers are this chapter's whole subject:
`dagger.py` selects that best round on the *same* held-out eval it then reports
the BC-vs-DAgger gap against — there is no separate validation split — so the
headline carries a mild winner's-curse selection bias. It is not an artifact. A
*non*-selected round (round 2 at seed 0) already clears BC's interval on its own,
and the recovery survives a Bonferroni correction across the four rounds. A
production loop would hold out a separate split for the selection; we name the
shortcut rather than bury it. `dagger.py` saves the best round's checkpoint and
reports the whole success-vs-round curve. The "how many rounds?" exercise makes
you read it.

## An honest ceiling

DAgger recovers the covariate-shift loss; it does not turn a reactive MLP into a
great PushT policy. The scripted expert is *stateful* (a multi-phase push
controller), and a memoryless clone of it — averaging over a multimodal action
distribution — tops out around 0.2–0.25 no matter how you cover the state space
(that ceiling is chapter 1.3's ACT and 1.4's diffusion policy, which give the
clone a memory and a multimodal head). DAgger's job here is narrow and real:
close the gap between what BC saw and where the policy goes. It closes it.

## Read the real thing

The paper to read beside this file is the original: Ross, Gordon & Bagnell,
*"A Reduction of Imitation Learning and Structured Prediction to No-Regret Online
Learning"* (2011). It is the source of the aggregate-and-retrain loop and of the
best-round result you just watched happen. Read it for the theory `dagger.py`
elides — the no-regret argument for *why* on-policy relabeling bounds an error
that behavior cloning cannot, and the formal role of the `beta` schedule you set
by hand here.

Then read the interactive expert this chapter stubbed out. LeRobot's teleop and
record loop is the production version of `dagger_rollout`: a human on a controller
labeling the states a policy visits, streamed to disk in the very
`(observation, action)` format `collect_bc_demos` writes. Read it for the one line
where "the expert labels the visited state" stops being a scripted function and
becomes a person. The refinement to look up from there is HG-DAgger and the
broader interactive-imitation line, which hand the *human* the decision of when to
intervene — the ergonomic answer to DAgger's one real cost, an expert you can
query at every state the policy reaches.

## What's next

DAgger fixed the states you visit by relabeling them; it still needs an expert with
the answers. The offline-RL primer next drops that assumption — it learns from a fixed
batch of logged, imperfect interactions with no expert to query. From there, ch4.3
(HIL-SERL) is the capstone: it folds the human back in, but as an occasional corrector
of a policy improving from its own reward, not a labeler of every state.
