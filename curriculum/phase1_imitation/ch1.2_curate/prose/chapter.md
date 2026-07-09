# 1.2: Data Is the Policy

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## See it work

Two policies, side by side, on the same held-out starts. They were trained by
the identical code (chapter 1.1's behavior cloning, unchanged) for the
identical number of epochs. The only difference is the dataset. The one on the
left learned from every episode you recorded. The one on the right learned from
the subset that reached the goal: a bit more than half the episodes, but, since
the ones it drops are the long, timed-out failures, only about a quarter of the
frames. Watch the far starts. The left policy shoves the block toward the
target and stalls when it needs to rotate; the right one commits to the turn.
Same method, same compute, different data, and the data is what you can see.

That is the whole chapter. In 1.1 the network was the thing you built and the
data was the thing you had. Here it flips: the data is the thing you build, and
the network is a fixed function that turns it into behavior. Curating is
programming, and this chapter is about doing it with measurements instead of
vibes.

## The problem

Chapter 1.1 ended on a confession. Behavior cloning fits the average action,
and when two demonstrations disagree at the same state, the average of two good
actions can be a bad one. We waved at that flaw and moved on. Now look at what
it does to real data.

Your ch0.4 recordings are not uniform. Some runs you drove cleanly and the
block clicked into place. Some you fumbled: you pushed when you should have
rotated, chased the block into a wall, ran out of patience and the episode
timed out with the T sitting crooked two centimeters off target. Every one of
those frames is a labeled (observation, action) pair, and behavior cloning
believes labels. Show it a demonstration that wandered and never rotated, and
in the states that demonstration visited, you have taught the policy to wander
and not rotate.

So the naive move, "I recorded 500 episodes, train on all 500", is not
obviously right. Some of those episodes are teaching the wrong thing. The
question this chapter makes measurable: can you do better by throwing data
away? And if so, how do you decide what to throw?

## Build

`curate.py` is one file, about 340 lines, in six regions: setup, data,
quality, curate, train, report. It scores every episode, filters on the score,
re-trains 1.1's BC twice (raw, then curated), and reports the gap. No new
machine learning: the model and training loop are chapter 1.1's, deliberately
copied so "re-train on the curated data" is something you can read, not import.

### Setup

```
[include-by-region: curate.py#setup]
```

The one new idea in the flags is the data source. `--data` points at a real
LeRobot dataset, your ch0.4 session. Omit it and the file builds a
reproducible stand-in: 250 "careful" episodes and 250 "sloppy" ones, a shaky
hand modeled as a large action-noise std. The stand-in exists so the chapter's
numbers reproduce on your machine; the lesson is identical on your own
recordings, only the digits move. `--break` is the Break It flag, and we earn
it at the end.

### Data

```
[include-by-region: curate.py#data]
```

Whichever source you use, the file ends this region holding three plain arrays:
observations, actions, and an episode id per frame. The episode is the unit we
are about to judge, not the frame. A single bad frame is noise the average can
absorb; a bad *episode* is a coherent stretch of wrong behavior covering a
whole region of the state space, and that is what curation removes. Note that
both halves of the stand-in are written by the same `gen_demos` as every other
PushT dataset in this book, so the curated set you produce drops straight back
into 1.1 with no format wrangling.

### Quality

This is the chapter's core: three ways to measure demonstration quality using
nothing but the dataset on your disk: no environment, no privileged
information a learner grading their own recordings would not have.

```
[include-by-region: curate.py#quality]
```

**Outcome.** Did the block finish inside the task's own tolerance? The last
recorded frame carries the block's pose; decode it and compare to PushT's
`POS_TOL` and `ANG_TOL`. This is the bluntest signal and the most important
one: a demonstration that did not accomplish the task is a bad source of labels
for accomplishing the task. On the stand-in, **288 of 500 episodes reached the
goal**.

**Disagreement.** For each frame, find the nearest frames from *other*
episodes and measure how much their actions vary. High disagreement means: near
this state, demonstrators chose visibly different things. This is chapter 1.1's
villain made into a number: the multimodality that MSE blurs into mush. Hold
onto how this one behaves; it is the trap.

**Coverage.** What fraction of the arena do your episodes even start in? A
dataset can be large and still blind to whole regions. Curation trades coverage
against quality, and you want to watch that trade, not make it by accident.

### Curate

```
[include-by-region: curate.py#curate]
```

The honest filter is one line: keep the episodes that reached the goal. Notice
the Break It path keeps the *same number* of episodes by a different ranking,
so when we get there, the only variable is *which* episodes, never how many.
That is what makes the comparison fair, and it is the difference between an
experiment and an anecdote.

### Train and report

```
[include-by-region: curate.py#train]
```

This is 1.1, copied. Same three-layer MLP, same in-model min-max normalization,
same cosine-decayed Adam, same held-out reset seeds. We reseed before each of
the two runs so the raw policy and the curated policy start from the identical
initialization: the dataset is the only thing that differs between them. The
eval also splits held-out episodes by starting distance, near versus far,
because *where* the two policies differ turns out to matter more than by how
much.

```
[include-by-region: curate.py#report]
```

## Run it

```
python curriculum/phase1_imitation/ch1.2_curate/curate.py --seed 0 --device cpu
```

<!-- wall-clock table renders from wallclock.csv -->

The result, with the default stand-in:

| | episodes | held-out success | near starts | far starts |
|---|---|---|---|---|
| raw | 500 | 8% | 2/19 | 2/31 |
| curated (outcome) | 288 | 22% | 3/19 | 8/31 |

Both rates are low in absolute terms here (this is deliberately little data, a few
minutes of CPU training), so read the *lift*, not the ceiling: curating nearly tripled
the success rate while *removing* 212 episodes. Read
the two right-hand columns before you celebrate the left one: near the goal the
policies are about the same, and the entire gap is on the far starts, where the
sloppy episodes had been teaching the network to shove-and-stall. The bad data
was not uniformly bad: it was poisoning a specific, identifiable region, and
the outcome filter cut exactly that region out. Open the recording and scrub
`quality/disagreement` and the two `payoff/success` bars:

```
rerun outputs/ch1.2-curate/curate.rrd
```

Scale either dataset up and both numbers
climb; the point is that at *equal* method and compute, the curated data wins,
and it wins by fixing the far-start failures the raw set was quietly teaching.

## Break it

Here is the move that should work and does not. Chapter 1.1 taught you that
disagreement is behavior cloning's enemy. So curate on it directly: keep the
episodes that *agree* most with their neighbors, and throw out the
high-disagreement ones as noise.

```
python curriculum/phase1_imitation/ch1.2_curate/curate.py --seed 0 --device cpu --break low_disagreement
```

It keeps the same 288 episodes the honest filter would, just chosen by lowest
disagreement instead of by outcome. And it makes the policy worse:

| | held-out success | mean disagreement of kept set | far starts |
|---|---|---|---|
| raw | 8% | 0.443 | 2/31 |
| curated (outcome) | 22% | 0.380 | 8/31 |
| break (low_disagreement) | 12% | 0.378 | 5/31 |

Stare at the middle column. The break's kept set has the *lowest* disagreement
of the three: by the logic of chapter 1.1 it should be the *best* data. It
produces a policy well short of honest curation, and it gives up on exactly the
far starts curation rescued (5/31 against curation's 8/31).

The mechanism is the whole lesson. Demonstrator disagreement does not measure
label noise. It measures *difficulty*. The states where good demonstrators
diverge are the hard ones (the far approaches, the rotations, the recoveries)
because those are the states with more than one reasonable thing to do. Rank
episodes by low disagreement and you are ranking them by *easiness*, and you
keep a tidy dataset of near-goal nudges that teaches a policy which fails the
instant the task gets hard. The metric was real. Optimizing it was the mistake.

The transferable warning: a quality signal is not a quality objective. A number
that correlates with good data in your analysis can point somewhere terrible
when you filter on it, because filtering changes the distribution the number
was describing. Outcome (did the task get done) is robust to this because it
is defined by the goal, not by the data's own internal agreement. When you
invent a data-quality score, ask what a dataset that *maximizes* it looks like
before you trust it, because your filter will find that dataset.

## Exercises

Four, in `exercises/`. Two ask you to commit to a prediction before the run is
allowed to answer: curated-versus-raw, and the Break It. One is a bug-hunt
where a single mask polarity error quietly curates the *failures* and every
metric still prints. One has you implement the disagreement signal from its
definition, since it is the number the whole chapter turns on.

## What's next

You now have two levers on a behavior-cloning policy: the method (1.1) and the
data (1.2). You have also seen the ceiling. Even the curated policy tops out
well short of the expert, because it is still averaging, still committing to one
action per state at 10 Hz, still unable to represent "go left OR go right" at a
state where both are correct. No amount of data curation fixes that: it is the
model class. Next chapter the policy stops predicting one instant at a time and
commits to a *chunk* of the future at once, and the multimodality you have spent
two chapters measuring finally gets a model that can hold two plans in mind
instead of averaging them into a bad third one.
