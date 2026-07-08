# G3 — Keep Going: The Living Field

<!-- Graduation bridge module. PROSE ONLY — no artifact, no exercises, no toy,
     no wall-clock table, no meta.yaml. This is the last thing you read.
     Every count/date below is a moving snapshot: hedged "as of 2025/2026" and
     pointed at a living page. No SHAs, no invented URLs. Verify before you cite. -->

## The one skill that does not expire

Here is the uncomfortable thing about a course on a field this young: some of the
specific facts in it will be stale before the ink dries. A checkpoint that was
frontier when 3.8 was written is a baseline a year later. A memory number that
was true on last quarter's LeRobot is off by a flag rename this quarter. The
model names in this very file will age.

That is fine, because the course never sold you the facts. It sold you a habit.
Chapter 3.8 named it the four moves — **load, inspect, hook, probe** — and proved
you could open a fourteen-gigabyte frontier VLA and *read* it, because you had
already written every mechanism inside it by hand: flow matching in 1.5, the
vision-language fuse in 1.7 and 1.8, the LeRobot dataset contract in 1.9. The
reading tracks in this phase — fine-tuning a real 450M policy, data engineering
at a million trajectories, edge deployment — turned that into a stance: *read
the real thing*. When the workflow outgrows the free tier, you don't pretend to
rebuild it; you go read the code that does it, next to the miniature you already
own.

This module is that stance pointed at the field itself. The durable skill is not
knowing today's best policy. It is knowing how to plug into the places the field
argues in public, and keep reading. Here is the map.

## Community: where the field actually talks

The center of gravity for open robot learning, as of 2026, is **Hugging Face
LeRobot** — a library, a Hub, and a Discord. You have been writing its dataset
format since 0.4; the Hub is where that format goes to be *shared*. Push your G2
dataset to the Hub under your namespace and it becomes a thing other people can
pull, train on, and diff against — the same `LeRobotDataset` a stranger's policy
eats. Models and interactive **Spaces** live there too, so "here is my policy,
here is a demo you can click" is one `push_to_hub` away.

The **LeRobot Discord** is where the messy-middle questions get answered — the
ones no paper's method section covers, the ones the reading tracks warned you
would eat your weeks: my camera angle is off-distribution, my norm stats look
wrong, which flag freezes the backbone. Ask there. Answer there when you can;
teaching a thing is how you find out you understood it.

And there is a concrete on-ramp to building *with* people, not just near them:
the **LeRobot Worldwide Hackathon**. The June 2025 edition (14–15 June) drew
**3,000+ registered participants across 100+ local events**, and the teams left
behind roughly **190 datasets** on the Hub under the hackathon org — a public
pile of real teleoperation you can go learn from today. (Those numbers are a
snapshot of one event; check the org page for the current count and the next
edition — a hackathon is the single fastest way to go from "I finished a course"
to "I shipped a thing with a team.")

## Benchmarks: where to prove your own policy

Names on a leaderboard are trivia. The useful question is *where do I take the
policy I built and find out if it is any good?* Four answers, in rising honesty.

**Open X-Embodiment (OXE)** is the canonical corpus — the thing you read in the
data track. As of its 2023 release it pooled **~60 datasets from 34 labs into 1M+
trajectories across 22 embodiments** (527 skills; and it keeps growing, so treat
every count as a snapshot). It is where you go to *co-train*: OXE is the proof,
at scale, of 3.7's thesis that a good mixture beats any single robot's data — the
RT-1-X / RT-2-X gains (roughly +50% and ~3× on emergent skills under the paper's
own protocol) are the headline you now know how to read skeptically.

**RoboArena** is the honest generalization test, and the one worth caring about
most. Central leaderboards let a policy overfit a fixed task list; RoboArena
instead **crowd-sources distributed, double-blind, pairwise real-robot
evaluations** on the shared DROID platform — its first run gathered 600+ pairwise
episodes across seven policies at seven institutions, and it has been running
live (through 2026). This is your policy versus the reality gap, judged by people
who did not build it, on robots you do not control. It is the closest thing the
field has to an honest answer to "does it actually generalize?"

**LIBERO, CALVIN, RoboMimic** are the manipulation suites — where you measure a
specific muscle in simulation before you spend real-robot time. **LIBERO**: 130
language-conditioned tasks in four suites, built to measure *knowledge transfer*
for lifelong learning (does new skill help or clobber the old?). **CALVIN**:
long-horizon chains of language-conditioned skills on a shared tabletop — the
test of stringing sub-tasks together. **RoboMimic**: learning from
demonstrations of *varying quality* — the benchmark that made "the data is the
policy" (1.2) and offline-from-demos (Phase 4) measurable. Pick the one that
matches the claim you want to make about your policy, and report a rate with an
interval, the way 1.6 taught you — never a hero rollout.

## The frontier to track: where generalist policies are going

3.8 gave you the four moves on a stand-in; 5.4 and 5.6 put them on real
architectures. The frontier to keep reading is the **generalist VLA foundation
model**, and as of 2026 the open ones you can actually pull and probe are:
**pi0 / openpi** (Physical Intelligence — the flow-matching action expert on a
PaliGemma backbone), **NVIDIA GR00T N1 / N1.5** (the structural slow-VLM /
fast-action split, humanoid-oriented, ~2–3B params), **OpenVLA** (the earlier
open 7B autoregressive VLA), and **SmolVLA** (Hugging Face's ~450M, pi0-style
flow head — the one small enough to fine-tune on a single consumer card, and the
policy the P1 reading track walks you through). Watch this space the way 3.8
taught you to: not "which won," but *what did they add, and why* — a real
web-scale VLM, a data pyramid, a dual-system rate split. The mechanisms are
yours; the scale is the story.

## Beyond manipulation: an honest scope note

Be clear-eyed about what this course is and is not. It is **manipulation-centric**
— arms, grippers, tabletop. The real field is wider, and pretending otherwise
would betray the honesty rail everything here runs on. Two directions it goes
that you now have the foundation to walk into, both first-class in LeRobot as of
2026:

- **Mobile / navigation.** The **LeKiwi** — an open, low-cost mobile-manipulator
  base (a wheeled platform under a SO-arm, Raspberry Pi–driven) — adds a moving
  frame under the gripper. Now the policy has to decide *where the base goes*,
  not just where the hand goes.
- **Whole-body / humanoids.** The **Unitree G1** (23/29-DoF humanoid) is a
  supported platform for locomanipulation — balance and contact and whole-body
  control, a different and harder control problem than a fixed arm.

And underneath both: **safety on real hardware**. A sim policy that fails is a
number; a humanoid or a mobile base that fails is a thing that moves in the world
near people. Torque limits, e-stops, guarded rollout — this course did not teach
it, and any honest map has to say so and point at it.

## How to actually keep learning

You already have the method; here is the practice.

1. **Read the real thing.** Pick a paper whose result you want. Find its repo,
   pour the checkpoint into a skeleton, and read it the way 3.8 read pi0 — against
   the mechanism you already wrote. Reproducing a method *is* the deepest reading.
2. **Contribute to LeRobot.** Fix a doc, add a dataset, file the issue you hit.
   The library is the field's shared workbench; leave it better.
3. **Enter the hackathon. Share a dataset.** Push your G2 data to the Hub. Enter
   the next Worldwide Hackathon. Building in public with a deadline and a team is
   worth ten solo tutorials.

That is the whole close. Look back at **G1**, where you named the arc you built,
and **G2**, where you shipped a policy of your own. Between them and here, the
claim this course makes about you is small and true: you can **build** the
mechanisms from scratch, **read** any robot-learning paper or checkpoint on the
frontier, **choose** the right tool and the honest eval — and now you know where
to **plug in and keep going**. The field will keep moving. So will you. Go do
embodied AI.

---

*Snapshot honesty: participant/dataset/trajectory counts and model names above
are as of 2025–2026 and will drift — every one is pointed at a living page
(the LeRobot Hub org, the OXE and RoboArena project sites, the benchmark repos)
that carries the current number. Verify before you cite; the shape of the map is
what this module promises to keep true, not the exact figures.*
