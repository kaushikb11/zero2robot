# 2.8: Concepts of ROS, Without ROS

<!-- objectives: rendered from meta.yaml, do not duplicate here -->

## One loop is a lie

Every chapter so far ran your policy in a single `while` loop: read the state,
compute an action, step the environment, repeat. It is the right first picture,
and it is a lie about how real robots run.

On real hardware there is no one loop. A camera driver produces frames at 30 Hz.
A joint-encoder driver produces angles at 1 kHz. Your policy wants a state
estimate at 50 Hz and emits a torque command. A motor controller consumes
commands at 500 Hz and will happily hold the last one if you go quiet. These
things run at *different rates*, on *different threads*, often on *different
computers* — and none of them can afford to block waiting for the others. The
camera cannot stop producing frames because the policy is mid-inference.

So real robot software is not a loop. It is a **graph of concurrent nodes** that
never call each other directly. Each node runs at its own fixed rate and talks to
the rest of the world only by **publishing** and **subscribing** to named
**topics**. That is what ROS is: the industrial formalization of this exact
shape. This chapter builds the shape from scratch — Node, Topic, Rate, in a few
hundred lines of standard-library Python — so that when you meet ROS you
recognize it as machinery you have already written, not magic. No `rospy`, no
`rclpy`, no install. The concepts, without the framework.

## The three primitives

Strip ROS down and three ideas remain. A **Topic** is a named, thread-safe bus:
publishers drop messages in, subscribers read them out, and neither holds a
reference to the other. A **Rate** is a fixed-Hz cadence — the promise that a
node ticks every 20 ms and not whenever the CPU feels like it. A **Node** ties a
piece of work to a rate. None of them knows anything about cartpoles; that is the
whole point of a runtime — the plumbing is generic, and the robot you build on
top is not.

```
[include-by-region: runtime.py#primitives]
```

The subtle design choice is in `Topic`. A control loop does not want a FIFO
backlog of stale sensor readings — it wants the *freshest* estimate, so a
subscriber reads the latest message, not the oldest. But the buffer is bounded,
so a message that gets superseded before the subscriber ever caught up to it is
**dropped**, and we count those drops. That counter is not bookkeeping: it is how
you *see* one node falling behind another. Hold onto it.

## Wiring the sense → think → act loop as a graph

Now the robot. The single loop you have run all along was already three jobs
wearing a trenchcoat: *sense* the state, *think* up an action, *act* on the
world. We give each its own node.

- The **sensor** samples the cartpole state and publishes it on `/obs`.
- The **policy** subscribes to `/obs`, runs the brain on the latest reading, and
  publishes an action on `/action`.
- The **actuator** subscribes to `/action`, applies it, and steps the plant.

The brain behind the policy node is deliberately swappable — it is just a
function from observation to action. By default we run a small **scripted
balancer**: a few lines of linear feedback on the observation, no checkpoint and
no download, so the graph balances the pole from a fresh clone and reproduces
byte-for-byte. It is also the brain the exercises measure. If you finished
chapter 2.1's training run, point `--policy` at the PPO checkpoint you saved and
the graph runs *that* policy instead — the sensor and actuator never notice,
because they only know two topics. That indifference is the lesson: you could
swap in the diffusion policy from chapter 1.4 the same way.

```
[include-by-region: runtime.py#graph]
```

Notice what the actuator does when the policy is quiet: it re-applies the last
action it received. That **zero-order hold** is not a hack — it is what every
real motor controller does between commands, and it is the mechanism that makes
the control rate matter.

## Running the graph: two clocks

Here is the honesty problem a concurrent system forces on us. If each node runs
in its own thread against the wall clock — the real robot-software shape — then
the order in which messages interleave depends on OS scheduling, and two runs
with the same `--seed` will *not* be byte-for-byte identical. That is not a bug
to fix; it is the nature of concurrency, and pretending otherwise would be
dishonest (root CLAUDE.md, invariant 2: we tier determinism honestly).

So the runtime has two schedulers behind one graph:

```
[include-by-region: runtime.py#scheduler]
```

`--clock real` is the authentic one: one thread per node, each ticking at its
rate against `time.monotonic`. Run it and watch the pole balance in real time.
`--clock virtual` replaces threads and sleeps with a single-threaded
discrete-event scheduler over a *simulated* clock — the same nodes, the same bus,
fired in a fixed tie-break order — so a `--seed` run is bit-for-bit reproducible.
CI uses it (via `--smoke`), and so should you when you want to compare two runs.
Same graph, two clocks, one honest story about what is and is not reproducible.

## Run it

```
python curriculum/phase2_reinforcement/ch2.8_runtime/runtime.py --seed 0
```

On a CPU laptop this takes about **0.19 min (measured)** — which for a real-time
runtime is essentially the 10 seconds of simulated time it plays out, because the
whole point of `--clock real` is that it runs at real time. The three nodes spin
up, the pole balances, and the summary reads something like:

```
[ch2.8-runtime] clock=real  policy=scripted balancer (built-in, checkpoint-free)
[ch2.8-runtime] rates: sensor 50 Hz | control 50 Hz | actuator 50 Hz | queue_depth 1
[ch2.8-runtime] BALANCED (pole up the whole run) — 500 control steps in 10.00s sim (10.00s wall)
[ch2.8-runtime] topics: /obs 50.0 msg/s (127 dropped) | /action 50.0 msg/s (132 dropped)
[ch2.8-runtime] mean sense->act latency: 16 ms  (control period 20 ms)
```

Nobody wrote a single control loop. Three concurrent nodes, passing messages,
kept the pole up. And look at the drop counts: even with every rate matched at
50 Hz, a hundred-odd messages were dropped, because thread jitter occasionally
lets the sensor publish a second reading before the policy has read the first —
the older one is superseded and falls off the depth-1 buffer. Latest-wins makes
that completely harmless here: the policy always acts on the *freshest* reading,
so the pole never notices. Hold that thought. In the next section the very same
counter, driven this time by a genuine rate mismatch, is the difference between
balancing and falling.

(All of this jitters run to run — drop counts and latency drift by a handful,
because `--clock real` is not bit-reproducible; run `--clock virtual` twice and
the numbers match exactly. To run your chapter-2.1 policy through the identical
graph, add `--policy outputs/ch2.1-ppo/ppo_agent.pt`.)

## What breaks when a rate is missed

Now the lesson the graph exists to teach. The sensor and the plant keep running
at 50 Hz; slow only the *policy*:

```
python .../runtime.py --seed 0 --break        # drops --control_hz to 5 Hz
```

The pole falls in about a second. Now the same drop counter is screaming — the
50 Hz sensor is overrunning the 5 Hz policy ten to one — but this time the drops
are a *symptom*, not the disease. The disease is latency: sense-to-act jumps to
~100 ms on average — and up to a full 200 ms, one whole period of a 5 Hz policy —
because the actuator keeps re-applying an action computed from state that is by
now a fifth of a second old. A command that stale, on a pole that tips past
recovery in a fraction of a second, is a lost pole. The control rate was never a
detail. It was the thing keeping the robot alive.

The exercises make you measure the cliff yourself: predict the rate at which the
pole first fails (it survives further down than you would guess — a good
controller is robust until it very suddenly is not), and then investigate the
tempting wrong fix. When a slow policy is dropping sensor messages, the obvious
move is to make the queue deeper so the drops stop. Try it: the drops go to zero
and the pole falls at the *exact same step*. A bigger buffer trades drops for
latency; it never buys you a faster controller. That distinction — a delivery
problem is not a control problem — is worth more than any single balanced run.

## Read the real thing

You did not build a toy version of ROS. You built ROS's *shape* — and the
fastest way to prove that is to open the real Python client next to your file. We
pin **`ros2/rclpy`** at commit `eedd8b1`. The package lives one directory in from
the repo root, so the two files to read are `rclpy/rclpy/node.py` and
`rclpy/rclpy/executors.py`. Read them against your regions.

**Your three primitives → `rclpy/rclpy/node.py`.** Your `primitives` region is
`Topic`, `Rate`, and `Node`; your `graph` region wires them into
sensor/policy/actuator. rclpy's `Node` exposes the same three moves as methods.
Where your `SensorNode` calls `obs_topic.publish(...)`, a real node first calls
`create_publisher(self, msg_type, topic, *, qos_profile=qos_profile_default)`;
where your `PolicyNode` reads `obs_topic.latest()`, a real node registers a
`create_subscription(self, msg_type, topic, callback, *,
qos_profile=qos_profile_default, callback_group=None)`. Your `Rate` — a fixed-Hz
tick — is `create_timer(self, timer_period_sec, callback, callback_group=None)`
(this pinned rclpy has no `create_rate`; the periodic timer *is* the rate). The
one line that shows it is not plumbing but a real transport: inside
`create_publisher`, the handle comes from `_rclpy.rclpy_create_publisher(
self.handle, msg_type, topic, qos_profile.get_c_qos_profile())` — a C call down
into DDS, exactly where your `deque` sat.

**Your scheduler → `rclpy/rclpy/executors.py`.** Your `scheduler` region has two
runners: `run_real` (one thread per node against the wall clock) and `run_virtual`
(a single-threaded, fixed-tie-break discrete-event loop that makes a `--seed` run
byte-identical). rclpy's answer is the `Executor`. `SingleThreadedExecutor.spin`
is literally `while ok(): self.spin_once()`, and its `spin_once` is `handler,
entity, node = next(self.wait_for_ready_callbacks(...)); handler()` — pop the next
ready callback, run it. That is your virtual scheduler's `heapq.heappop(...);
node.tick(...)`, except readiness is decided by `_rclpy.rclpy_wait(wait_set,
timeout_nsec)` blocking on the OS, not by your heap of due times.
`MultiThreadedExecutor` is your `run_real`, hardened into a thread pool.

**What they add, and why you were allowed to skip it.** Three things your runtime
omits. **QoS** — `rclpy/rclpy/qos.py`'s `QoSProfile` bundles `history`, `depth`,
`reliability`, `durability`: your `queue_depth` is exactly `depth`, but a real
publisher also picks reliable-vs-best-effort delivery, which over a lossy network
is the difference between your drop counter and a hang. **Discovery** — you handed
each node its topics by hand in `build_graph`; DDS finds publishers and
subscribers across processes and machines at runtime, no wiring. **A real
transport** — your bus is a `deque` behind a `Lock`; theirs is DDS-over-UDP, which
is why the payload must be a typed message and the buffer lives in C. None of it
changes the shape. It changes the blast radius: your graph is three threads on one
laptop; theirs is forty nodes across four computers that have never met.

Read `node.py` first — find your three primitives as three methods — then
`executors.py`, and watch your virtual clock become `spin`.
