# 0.0: Your First Robot Policy in a Few Minutes

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## See it work

That blue T-shape is being pushed onto the green target by a neural network.
Nobody told it what a "T" is, or where the target is, or which way to shove.
It was shown a few hundred examples of a pushing task being done, and asked to
copy them. And this is the copy, running on a start it was never shown. No
planner, no reward, no search. A stack of matrix multiplies that learned to
push.

You are going to train that network yourself, right now, before we explain a
single thing about how it works. That order is deliberate. The rest of Phase 0
takes the simulator, the data format, and the learning apart piece by piece;
it reads very differently once you've already watched the whole thing succeed.

## The whole loop, once

Everything is in one file, `quickstart.py`, and it does four things in order:
gets demonstrations, trains a network to imitate them, rolls the result out on
fresh starts, and prints how often it worked. Run it:

```
python curriculum/phase0_foundations/ch0.0_quickstart/quickstart.py --seed 0
```

<!-- wall-clock table renders from wallclock.csv -->

**Demonstrations.** A demonstration is just a recording of the task being done
well: at each instant, what the world looked like and what the driver did about
it. Here the "driver" is a scripted expert that ships with the PushT
environment, a hand-tuned controller somebody wrote so you don't have to. We
run it a few hundred times and keep what it saw and did.

```
[include-by-region: quickstart.py#demos]
```

**A tiny network.** Three linear layers (a linear layer is just a matrix
multiply plus an added offset). Its whole job is to map "the world looks like
this" to "so do that". Boring on purpose.

```
[include-by-region: quickstart.py#model]
```

**Training.** Show it a demonstrated state, ask what it would do, penalize the
gap to what the expert actually did, repeat. That's the entire method. The
loop is short enough to read in one sitting.

```
[include-by-region: quickstart.py#train]
```

**The honest test.** Loss on the demonstrations tells you the network memorized
them; it does not tell you the policy can push a block. So we turn it loose on
25 starting positions it never trained on and count the ones where the block
actually reaches the target. And we roll out a *random* policy on the same 25
starts as a floor. If "it works" is going to mean anything, it has to beat
flailing.

```
[include-by-region: quickstart.py#eval]
```

At `--seed 0` the trained policy solves **12 of 25** held-out starts (48%). The
random floor solves **0 of 25**. That gap, from nothing to roughly half, is
the whole point of this chapter, and you produced it in a few minutes on a
laptop with no GPU. (It's seed-dependent: across the first few seeds the policy
lands between about a quarter and a half of the starts. Never zero, always well
clear of the floor.)

## What we skipped, and where it comes back

This was the shortest honest path to a working policy, which means it cut
corners you should know about, and every one of them is a real chapter ahead, not
a hand-wave:

- **The demonstrations came from a script.** Real imitation learning starts
  with *your* demonstrations. In **ch0.4** you drive the pusher yourself and
  record them, and you learn what a dataset physically is.
- **The data lived in memory.** We never wrote a dataset to disk. The actual
  format, the one real robots record into, is **ch0.4** and **ch0.5**.
- **We trained with plain squared error and didn't ask what breaks.** That's
  behavior cloning, and it has failure modes that no loss curve will confess
  to. **ch1.1** is this exact method, built properly, with those failures
  induced and measured, including *why* 48% and not 100%.
- **The simulator was a black box.** What `env.step` actually does (the physics
  loop you just trusted a few hundred thousand times) is **ch0.1**, the very
  next thing.

So the rest of Phase 0 is not new material piled on top of this. It is *this*,
slowed down: now let's understand what you just did.

## Exercise

One, in `exercises/`, and it's worth doing before you move on. All 300 of those
demonstrations succeeded at the task. So did the win really need 300, or would
a handful of perfect demos have done just as well? Commit to an answer, then let
the code settle it.

## What's next

You have a policy that works and no idea why. Chapter 0.1 opens the simulator
you just called `env.step` on and shows you the loop underneath: the first of
the pieces that, put back together, are the thing now pushing a block across
your screen.
