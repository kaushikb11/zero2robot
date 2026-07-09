# 1.1: Behavior Cloning: The Dumbest Thing That Works

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## See it work

Drag the T-block a few centimeters off the path the policy wants, then let
go and watch it come back for the block. This policy has no planner, no
search, no reward function, no idea what a "T" is. It is a three-layer MLP
that was shown a few hundred episodes of pushing and told: when the world
looks like this, do that. Sometimes your drag lands somewhere the
demonstrations covered, and the recovery looks deliberate, almost smug.
Sometimes it doesn't, and the pusher commits (confidently, repeatedly) to
an approach that will never work.

Try to find where it breaks. You will, and quickly, and that boundary you
just found by hand is this chapter's actual subject. The policy is exactly
as good as its demonstrations, no better, and the interesting question is
never whether it works: it's where it stops working and what that looks
like from the inside.

## The problem

In chapter 0.4 you drove this pusher yourself, with your own two hands on
the keys, and you succeeded at a task you could not begin to write down.
Try it: write the rule you used for deciding when to stop pushing the bar
and start nudging the stem. You rotated the block when it needed rotating.
How did you know? You'd have to say something about the angle, and the
contact point, and where the pusher happened to be, and by the third
clause you're describing a controller you never consciously ran.

The scripted expert in `curriculum/common/envs/pusht/scripted_expert.py` is
what writing it down actually costs: a two-phase state machine, contact-point
selection in the block's body frame, a detour circle so the approach never
plows through the block, a steering offset proportional to yaw error. It
took real engineering hours, it is full of constants like `0.045` and
`1.15`, and it works for exactly one task on exactly one block. Nobody
writes that controller for the thousandth task.

Behavior cloning skips the describing. You already produced the only
artifact it needs, a dataset of (observation, action) pairs from chapter
0.4's teleop session, and the claim this chapter tests is almost insulting
in its simplicity: fit a function from one column to the other with mean
squared error, and driving skill falls out. What we build: a single file
that loads your demos, fits an MLP, measures it honestly in rollouts, and
ships the result to the browser. Then we break it in a way no loss curve
will admit to.

## Build

Four regions in dependency order: data, model, train, eval. The whole file
is `bc.py`, about 270 lines, and every line of it is on the page: there is
no framework underneath.

### Setup

One thing to look for: every source of randomness in this file (the
train/val split, the weight init, the batch shuffle) flows from the one
`--seed` flag through `set_seed`.

```
[include-by-region: bc.py#setup]
```

The flags follow the house convention: free-tier defaults, a `--smoke` mode
that runs tiny and fixed so CI can byte-compare two runs, and `--no-rerun`
to skip recording. Two flags are new. `--normalize` you will meet properly
in Break It. `--device` picks the fastest thing your machine has (cuda,
then mps, then cpu); the reference numbers below are from `--device cpu`,
the one configuration we can promise is bitwise-deterministic under
`--seed`. On a GPU or an Apple mps backend the same seed reproduces the
result statistically, not bitwise (your success rate will land near the
number printed here, not exactly on it) and this book does not pretend
otherwise.

### Data

Look for what gets split: episodes, never frames.

```
[include-by-region: bc.py#data]
```

The dataset is LeRobot-format: the same format your ch0.4 teleop session
produced, the same format `lerobot/pusht`'s human demonstrations ship in,
and the same format the reference generator writes. This chapter's numbers
use the scripted expert's demos so they're exactly reproducible on your
machine:

```
python curriculum/common/envs/pusht/gen_demos.py --episodes 500 --seed 0 \
    --out outputs/pusht-demos --no-video
```

Two decisions in this region carry most of the chapter's honesty. First,
the split. Consecutive frames are 0.1 s apart; the block barely moves
between them. Split frame-wise and nearly every validation frame has a
near-twin in training: val loss becomes a memorization test you can only
pass, and it will happily stay low while your policy learns nothing
transferable. Splitting by episode keeps validation what it claims to be:
whole trajectories the network has never seen.

Second, normalization. Raw observations mix meters (±0.35) with sin/cos
(±1); raw actions are velocities. We rescale each dimension to [-1, 1]
using its min and max over the training frames, the same scheme the real
PushT policies use. The stats are computed from the data, which sounds like
a triviality and is actually a loaded gun: the stats are a claim about what
the world looks like, and the code will believe that claim long after it
stops being true. `--normalize full` computes them from all training
episodes. The other setting exists for Break It.

### Model

Look at how little there is to look at.

```
[include-by-region: bc.py#model]
```

Three linear layers and two ReLUs. This is deliberate, and it is not
modesty. The ceiling of behavior cloning is the data, not the network: the
policy can never act better than the demonstrations it averages over, so
capacity spent past "can represent the demonstrator" buys you nothing but a
sharper copy of the same mistakes. Exercise 3 makes you measure this
instead of trusting me: you'll 10x the network's training two different
ways and watch which one the success rate responds to.

One structural decision deserves attention: the normalization stats live
inside the model, as buffers. The forward pass takes raw observations,
normalizes, predicts, denormalizes. That means the checkpoint and the ONNX
export carry their own stats: the browser playground can feed the model
observations straight from the simulator, and there is no separate stats
file to lose, version-skew, or apply twice. Exercise 1 is about the "apply
twice" failure, which every robotics codebase commits eventually.

The clamp in the forward pass never moves a training value: the training
data defines [-1, 1], so nothing in it can leave. Remember that it exists.

### Train

The loop is short enough to read in one breath.

```
[include-by-region: bc.py#train]
```

No DataLoader, no gradient clipping, no early stopping: the dataset is two
tensors in memory, and a "batch" is a slice of a shuffled index permutation.
The one concession to optimization reality is the cosine decay on the
learning rate. Without it the last hundred epochs bounce around the
minimum instead of settling into it, and the difference is measurable: +6
points of success rate at the default configuration (62% with the decay,
56% without).

MSE deserves a sentence of defense and a sentence of prosecution. Defense:
minimizing squared error on actions is maximum likelihood under a Gaussian,
it's not a hack, it's the textbook estimator for "predict the average
action the demonstrator took in this state". Prosecution: it predicts the
AVERAGE action. When demonstrations disagree (approach the block
clockwise or counterclockwise, push the near tip or the far one), the
average of two good actions can be a bad action, sometimes a catastrophic
one. Even our scripted expert disagrees with itself: its action depends on
an internal phase (approaching vs. mid-stroke) that the observation does
not fully reveal, so the same observation carries different labels and the
MLP splits the difference. Hold that thought; it is the entire reason
chapter 1.2 exists.

### Eval

Loss measured how well we imitate on the dataset's states. Rollouts ask the
question we actually care about.

```
[include-by-region: bc.py#eval]
```

The distinction matters because the policy chooses its own future inputs.
One slightly-off action leads to a state slightly off the demonstration
manifold, where the policy is slightly worse, which leads further off: the
compounding spiral is called covariate shift, and it is the disease
behavior cloning dies of. A loss number cannot see it, because the loss is
computed on states the DEMONSTRATOR chose. Rollouts from 50 held-out reset
seeds can. Note the seed arithmetic: demo episode `i` consumed reset seed
`seed + i`, so the evaluation seeds `10_000 + …` are new poses by
construction: we never grade the policy on a start it trained on.

After the rollouts, two lines close the loop the whole book runs on:
`export_policy` writes the ONNX under tensor contract v1, and
`assert_parity` proves torch and onnxruntime produce the same actions
before the file is allowed anywhere near the playground. The measured
parity delta at the default config is about 2e-06, comfortably under the
1e-4 gate.

## Run it

```
python curriculum/phase1_imitation/ch1.1_bc/bc.py --seed 0
```

<!-- wall-clock table renders from wallclock.csv -->

With 500 scripted demos and default flags, the reference run reaches **31/50
held-out episodes (62% success), mean return -36.6, final train loss 0.010,
val loss 0.037**. Open the recording:

```
rerun outputs/ch1.1-bc/bc.rrd
```

Two timelines. On `step`: `policy/loss/train` sawing downward,
`policy/loss/val` flattening out around epoch 300. After that you are
polishing the fit, not learning new behavior. On `sim_time`: fifty eval
episodes laid end to end, `eval/pos_err` and `policy/action` per step. The
healthy signature is pos_err curves that dive and stay down; the sick ones
dive, stall, and saw sideways. Go find two or three of those now (about a
third of episodes fail at this scale; you have plenty to choose from) and
scrub `policy/action` at the stall. That's what averaged indecision looks
like, and diagnosing it from the trace is a skill this book will ask of you
thirty more times.

And 62% is worth staring at. The expert that produced the demonstrations
succeeds on every one of the fifty seeds measured in the env README (100%).
Same observations, same task: we lost a third of the competence in the
copying. Some of that is averaging over the expert's hidden phase, some is
covariate shift off the demo manifold. More demonstrations buy some of it
back (exercise 4 quantifies how much); no amount buys back all of it.

## Break it

Two runs, one flag apart (the smaller network and shorter schedule are
deliberate: the note at the end of this section explains why, and it's
worth the wait):

```
python curriculum/phase1_imitation/ch1.1_bc/bc.py --seed 0 --epochs 300 --hidden_dim 256
python curriculum/phase1_imitation/ch1.1_bc/bc.py --seed 0 --epochs 300 --hidden_dim 256 --normalize narrow
```

`narrow` computes the normalization stats from only the ~20% of training
episodes whose block starts closest to the target, and changes nothing
else. Same 500 episodes, same network, same epochs. This is not an exotic
sabotage; it's the most ordinary bug in robot learning: you computed your
stats the week you were testing the rig with the block placed gently near
the goal, and never recomputed after you started collecting for real.

Now look at what training reports:

| | `--normalize full` | `--normalize narrow` |
|---|---|---|
| final train loss | 0.0161 | 0.0157 |
| final val loss | 0.0345 | 0.0355 |
| rollout success | 29/50 (58%) | 23/50 (46%) |

The `full` column lands at 58%, below Run It's 62%, because these two ablation
runs use the smaller 256-wide net and shorter 300-epoch schedule from the commands
above; the comparison that matters is the two columns against each other, not
against the default.

Read the loss rows first. Nothing flags the narrow run: train loss is
marginally LOWER, val loss is a wash. Both splits are normalized with the
same skewed stats, so both fit and both agree with each other. Every curve
in rerun looks like the healthy screenshot from Run It. If you shipped
policies on loss curves, you would ship this one.

The rollouts tell a different story, and WHERE they tell it is the lesson.
Split the 50 eval episodes by the block's starting distance from the
target: for starts inside the region the stats covered (under 0.15 m), the
narrow policy essentially matches the honest one: 8/19 vs 9/19, a
single-episode difference that is noise at this sample size. For far starts
it gives up a quarter of its competence: 15/31 against the honest run's
20/31. Five of the six episodes the narrow policy loses are far starts: the
damage lands exactly on the states the stats never described, and no metric
computed ON THE DATASET can see it, because the dataset is precisely the
thing that got misdescribed.

The mechanism is that clamp from the model region, awake for the first
time. Min-max stats from near-goal episodes declare that `tee_x` lives in
roughly [-0.13, +0.13]. A block starting at 0.20 m normalizes to 1.5, and
clamps to 1.0. The policy doesn't see a far block; it sees a block pinned
to the edge of a world that ends where the easy demos ended. Open the two
recordings side by side and scrub a far-start episode: in the narrow run,
`policy/action` saturates toward a block-edge that isn't where the block
is, the pusher arrives short, strokes empty table, re-approaches, strokes
again. It isn't confused. It is certain, and it's certain about a world
0.13 meters wide.

The transferable lesson, and it's worth saying plainly: loss curves measure
the dataset you gave them, not the world. Normalization stats are part of
the model. Version them, recompute them when the data changes, and when a
policy trains clean but acts wrong, check what its inputs look like AFTER
normalization: `rr.log` one histogram and this entire class of bug
confesses in seconds.

One more measured fact, because this book doesn't hide inconvenient ones:
train the narrow run at the chapter's full default scale (600 epochs,
hidden_dim 512) and the failure heals. The bigger network, given twice the
schedule, learns a working strategy for blocks pinned to the edge of its
clamped world ("push toward the edge until the block walks into view") and
finishes at 37/50, above the honest default run's 31/50. That should
unsettle you more than the failure did: the bug didn't go away, the policy
papered over it, and nothing in your metrics distinguishes "fixed" from
"compensated". Chapter 1.6 is an entire chapter about why evaluation this
shallow will eventually lie to you.

## Exercises

Four, in `exercises/`: a bug-hunt where every training metric is healthy
and only rollouts complain, the rollout loop with the middle missing, and
two where you commit to a prediction before the run is allowed to tell you
the answer.

## What's next

Your BC policy averages. On PushT the averaging is mostly survivable:
scalar velocities, one expert, its self-disagreements small enough to blur
rather than break. But you saw the residue even here: a third of the
competence gone, stalls where the expert's hidden phase made one
observation carry two labels. Human demonstrations are worse. You never
push the block the same way twice, and when demonstrations genuinely
disagree, the average of two good trajectories is a trajectory through the
space between them, which is to say: through failure. Next chapter the
policy stops predicting the next instant's average and starts committing
to plans (chunks of future actions predicted together) and the stalls
you found in the rerun traces tonight are the first thing that disappears.
