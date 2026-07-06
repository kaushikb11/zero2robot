"""zero2robot 2.8 — Concepts of ROS, Without ROS: a pub-sub control runtime.

Real robot software is not one loop. It is a graph of concurrent NODES that do
not call each other directly — they exchange messages on named TOPICS, each
running at its own fixed RATE. A sensor samples state at 50 Hz. A policy thinks
at 50 Hz. An actuator drives the motors at 50 Hz. None of them holds a reference
to the others; they publish and subscribe through a bus. That decoupling is why
you can swap the policy without touching the sensor, log every topic for free,
and reason about a 40-node robot at all. ROS is the industrial formalization of
exactly this shape. This file builds the shape from scratch in stdlib threading
+ queues — 300 lines instead of a ROS install — so you can see WHY it is shaped
this way, and what breaks when a rate is missed.

The graph we wire (the sense -> think -> act loop, decoupled into three nodes):

    SENSOR   reads the cartpole state          --/obs-->    at --sensor_hz
    POLICY   runs the ch2.1 PPO policy          --/action--> at --control_hz
    ACTUATOR applies the action, steps the env               at the plant rate

Run the graph and the pole balances — driven not by one script but by three
concurrent nodes passing messages. Drop the control rate (--break) and watch the
actuator hold a stale action too long while the pole falls: the control RATE is
not a detail, it is the thing keeping the robot alive.

Determinism (honest, per root CLAUDE.md #2): the default --clock real mode runs
each node in its own thread against the wall clock, so message interleaving is
NOT bitwise reproducible — that is real robot software. --clock virtual (forced
by --smoke) replaces threads+sleep with a single-threaded virtual-clock
discrete-event scheduler: the same nodes, the same bus, fired in a fixed order
on a simulated timeline, so a --seed run is byte-for-byte reproducible (CI uses
this).

Run it:      python curriculum/phase2_reinforcement/ch2.8_runtime/runtime.py --seed 0
Reproduce:   python .../runtime.py --seed 0 --clock virtual   (twice -> identical)
Break it:    python .../runtime.py --seed 0 --break           (control rate -> 5 Hz, pole falls)
CI smoke:    python .../runtime.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import heapq
import itertools
import json
import math
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as ch2.1 / tests/).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner, detect_device  # noqa: E402
from curriculum.common.envs.cartpole import CartpoleEnv  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch2.8-runtime"))
parser.add_argument("--duration_s", type=float, default=10.0, help="sim seconds to run (10 s = one 500-step episode); smoke: 2")
parser.add_argument("--sensor_hz", type=float, default=50.0, help="how often the SENSOR node samples state")
parser.add_argument("--control_hz", type=float, default=50.0, help="how often the POLICY node recomputes the action; --break drops this")
parser.add_argument("--queue_depth", type=int, default=1, help="per-topic message buffer; 1 = latest-wins (the control-systems default)")
parser.add_argument("--clock", choices=("real", "virtual"), default="real",
                    help="real: one thread per node vs the wall clock (non-deterministic). virtual: single-thread simulated clock (reproducible; --smoke forces it)")
parser.add_argument("--policy", type=Path, default=Path("outputs/ch2.1-ppo/ppo_agent.pt"),
                    help="trained ch2.1 PPO checkpoint to run as the 'brain'; falls back to a scripted balancer if absent")
parser.add_argument("--break", dest="break_bug", action="store_true",
                    help="Break It (optional demo, NOT the graded exercise): drop --control_hz to 5 Hz so the policy can't keep up -> the pole falls")
parser.add_argument("--seed", type=int, default=0, help="seeds torch, numpy, AND the env reset")
parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=detect_device())
parser.add_argument("--smoke", action="store_true", help="tiny deterministic CPU run for CI; forces --clock virtual, two runs must match byte-for-byte")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

set_seed(args.seed)  # seeds python/numpy/torch; the env reset is seeded explicitly below
if args.break_bug:
    args.control_hz = 5.0  # the demo: a policy that thinks 10x too slowly cannot balance
if args.smoke:  # pin everything the CI byte-compare depends on
    args.clock, args.device, args.duration_s = "virtual", "cpu", 2.0
banner("ch2.8-runtime", device=args.device)
args.out.mkdir(parents=True, exist_ok=True)
device = torch.device(args.device)
ACTUATOR_HZ = float(CartpoleEnv.CONTROL_HZ)  # the plant integrates at a FIXED rate; the actuator can't command faster
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch2.8-runtime", spawn=False)
    rr.save(str(args.out / "runtime.rrd"))
# --- endregion ---

# --- region: primitives ---
# The three ideas ROS formalizes, from scratch. A Message is a timestamped
# payload; a Topic is a thread-safe pub-sub bus; a Rate is a fixed-Hz cadence;
# a Node ties a callback to a rate. Nothing here knows about cartpole — that is
# the point of a runtime: the plumbing is generic, the graph on top is not.


@dataclass
class Message:
    """One published value. `stamp` is when it was created (runtime seconds);
    `origin_stamp` carries the timestamp of the sensor reading a value was
    DERIVED from, so a downstream node can measure end-to-end latency."""
    data: object
    stamp: float
    seq: int
    origin_stamp: float | None = None


class Topic:
    """A named message bus, bounded to `depth` buffered messages. Publishers
    append; a subscriber reads the LATEST (control code wants the freshest
    estimate, not a FIFO backlog — a stale sensor reading is worse than a
    dropped one). A message is DROPPED if it is evicted from the full buffer
    before the subscriber ever caught up to it: a deeper queue absorbs a burst,
    a depth-1 queue drops everything a slow subscriber didn't read in time. That
    drop count is how you SEE a rate mismatch. Thread-safe because in --clock
    real the publisher and subscriber live in different threads."""

    def __init__(self, name: str, depth: int):
        self.name = name
        self._buf: deque[Message] = deque(maxlen=max(1, depth))
        self._lock = threading.Lock()
        self._seq = itertools.count()
        self._read_seq = -1  # high-water mark: newest seq a subscriber has seen
        self.published = 0
        self.dropped = 0

    def publish(self, data: object, stamp: float, origin_stamp: float | None = None) -> None:
        with self._lock:
            if len(self._buf) == self._buf.maxlen and self._buf[0].seq > self._read_seq:
                self.dropped += 1  # the oldest buffered msg falls off UNREAD -> lost
            self._buf.append(Message(data, stamp, next(self._seq), origin_stamp))
            self.published += 1

    def latest(self) -> Message | None:
        """The freshest message, or None if nothing has been published yet.
        Reading the newest advances the read high-water mark past everything
        buffered — a latest-wins subscriber that caught up skips the backlog."""
        with self._lock:
            if not self._buf:
                return None
            msg = self._buf[-1]
            self._read_seq = max(self._read_seq, msg.seq)
            return msg


class Rate:
    """A fixed-Hz cadence. `period` is the seconds between ticks. In --clock
    real, sleep_until() blocks the node's thread just long enough to hold the
    rate; the virtual scheduler ignores it and advances the clock itself."""

    def __init__(self, hz: float):
        self.hz = hz
        self.period = 1.0 / hz

    def sleep_until(self, deadline: float) -> None:
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)


class Node:
    """A unit of computation on the graph: a name, a rate, and a tick(). The
    scheduler calls tick() at the node's rate; the node talks to the rest of the
    graph ONLY through topics it was handed. `priority` breaks ties when two
    nodes are due at the same virtual instant (sensor < policy < actuator, so a
    cycle always samples, then thinks, then acts)."""

    def __init__(self, name: str, hz: float, priority: int):
        self.name = name
        self.rate = Rate(hz)
        self.priority = priority

    def tick(self, now: float) -> None:
        raise NotImplementedError
# --- endregion ---

# --- region: graph ---
def load_policy(path: Path, device: torch.device):
    """Return an obs -> action function: the ch2.1 PPO policy's MEAN action (no
    sampling — this is deployment, not exploration). We rebuild just the
    actor_mean MLP (obs_dim -> 64 -> 64 -> act_dim) and load its weights from the
    checkpoint ch2.1 saved. If no checkpoint exists (a fresh clone / CI), fall
    back to a scripted linear balancer computed from the obs — the runtime lesson
    is identical whichever brain is behind the topic."""
    if path.is_file():
        net = nn.Sequential(
            nn.Linear(CartpoleEnv.OBS_DIM, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, CartpoleEnv.ACT_DIM),
        ).to(device)
        state = torch.load(path, map_location=device)
        net.load_state_dict({k.replace("actor_mean.", ""): v
                             for k, v in state.items() if k.startswith("actor_mean.")})
        net.eval()

        def policy(obs: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                a = net(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))
            return a[0].cpu().numpy()
        return policy, f"ch2.1 PPO checkpoint ({path})"

    def scripted(obs: np.ndarray) -> np.ndarray:
        # Mirrors common.cartpole.balance_action's gains, but reads the OBS (what
        # the sensor published), not the env — the policy node only ever sees
        # messages. obs = [cart_pos, cart_vel, cos(theta), sin(theta), angvel].
        theta = math.atan2(float(obs[3]), float(obs[2]))
        u = 10.0 * theta + 2.0 * float(obs[4]) + 0.4 * float(obs[0]) + 0.8 * float(obs[1])
        return np.array([np.clip(u, -1.0, 1.0)], dtype=np.float32)
    return scripted, "scripted balancer (no checkpoint found)"


class SensorNode(Node):
    """Samples the plant state and publishes it. In a real robot this is a
    camera or an encoder driver; here it reads the cartpole observation. It
    shares the env with the actuator, so it takes the env lock for a clean read
    (a shared resource still needs guarding, even in sim)."""

    def __init__(self, env, lock, obs_topic, hz):
        super().__init__("sensor", hz, priority=0)
        self.env, self.lock, self.obs_topic = env, lock, obs_topic

    def tick(self, now: float) -> None:
        with self.lock:
            obs = self.env._obs()  # the current measurement of the world's state
        self.obs_topic.publish(obs, stamp=now)


class PolicyNode(Node):
    """The brain. Subscribes to /obs, runs the policy on the LATEST reading, and
    publishes an action — carrying forward the obs's timestamp so the actuator
    can measure how stale the decision is. Never touches the env: it only knows
    the two topics, which is exactly why the policy is swappable."""

    def __init__(self, policy_fn, obs_topic, action_topic, hz):
        super().__init__("policy", hz, priority=1)
        self.policy_fn, self.obs_topic, self.action_topic = policy_fn, obs_topic, action_topic

    def tick(self, now: float) -> None:
        msg = self.obs_topic.latest()
        if msg is None:
            return  # nothing sensed yet this run
        action = self.policy_fn(msg.data)
        self.action_topic.publish(action, stamp=now, origin_stamp=msg.stamp)


class ActuatorNode(Node):
    """Applies the latest commanded action and advances the plant one control
    step. It is the ONLY node that steps the env, so it owns the world's clock:
    each tick = one env.step = one control period of sim time. If the policy is
    slow, the newest action is old and gets re-applied (zero-order hold) — that
    held-stale-command is what --break makes visible."""

    def __init__(self, env, lock, action_topic, state, hz):
        super().__init__("actuator", hz, priority=2)
        self.env, self.lock, self.action_topic, self.state = env, lock, action_topic, state

    def tick(self, now: float) -> None:
        msg = self.action_topic.latest()
        action = msg.data if msg is not None else np.zeros(CartpoleEnv.ACT_DIM, dtype=np.float32)
        with self.lock:
            _, _, done, info = self.env.step(action)
            pole_angle = info["pole_angle"]
        self.state["steps"] += 1
        self.state["pole_angle"] = pole_angle
        if msg is not None and msg.origin_stamp is not None:
            self.state["latency_sum"] += now - msg.origin_stamp  # sense -> act delay
            self.state["latency_n"] += 1
        if info["terminated"]:  # the pole fell — a real failure, stop the graph
            self.state["fell"] = True
        if args.rerun:
            rr.set_time("control_step", sequence=self.state["steps"])
            rr.log("plant/pole_angle_rad", rr.Scalars([pole_angle]))
            rr.log("policy/action", rr.Scalars(np.asarray(action, dtype=np.float64)))
            rr.log("bus/obs_published", rr.Scalars([float(self.state["obs_topic"].published)]))
            rr.log("bus/action_published", rr.Scalars([float(self.action_topic.published)]))
# --- endregion ---

# --- region: scheduler ---
def build_graph():
    """Wire the env + policy into the sensor/policy/actuator graph. Returns the
    nodes, the two topics, and the shared run state the actuator writes."""
    env = CartpoleEnv()
    env.reset(seed=args.seed)  # deterministic start (CPU MuJoCo is bitwise-reproducible)
    lock = threading.Lock()
    obs_topic = Topic("/obs", args.queue_depth)
    action_topic = Topic("/action", args.queue_depth)
    policy_fn, policy_src = load_policy(args.policy, device)
    state = {"steps": 0, "fell": False, "pole_angle": 0.0,
             "latency_sum": 0.0, "latency_n": 0, "obs_topic": obs_topic}
    nodes = [
        SensorNode(env, lock, obs_topic, args.sensor_hz),
        PolicyNode(policy_fn, obs_topic, action_topic, args.control_hz),
        ActuatorNode(env, lock, action_topic, state, ACTUATOR_HZ),
    ]
    return nodes, obs_topic, action_topic, state, policy_src


def run_virtual(nodes, state, duration_s):
    """Deterministic discrete-event scheduler: no threads, no wall clock. A heap
    holds (next_due, priority, node); we advance a virtual clock to the earliest
    due node, tick it, and reschedule it one period later. Fixed tie-break =
    fixed interleaving = byte-identical across runs at a given --seed."""
    heap = [(0.0, n.priority, i, n) for i, n in enumerate(nodes)]
    heapq.heapify(heap)
    while heap:
        due, prio, i, node = heapq.heappop(heap)
        if due > duration_s:
            break
        node.tick(due)
        if state["fell"]:
            return due  # pole down: stop the world
        heapq.heappush(heap, (due + node.rate.period, prio, i, node))
    return duration_s


def run_real(nodes, state, duration_s):
    """Threaded scheduler: one thread per node, each looping tick()+sleep at its
    own rate against the wall clock — the real robot-software shape. Timing (and
    so message interleaving) is NOT bitwise reproducible; that is why --smoke and
    reproducible runs use --clock virtual instead."""
    stop = threading.Event()

    def loop(node):
        start = time.monotonic()
        deadline = start
        while not stop.is_set():
            now = time.monotonic() - start
            if now >= duration_s or state["fell"]:
                break
            node.tick(now)
            deadline += node.rate.period
            node.rate.sleep_until(deadline)

    threads = [threading.Thread(target=loop, args=(n,), name=n.name) for n in nodes]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return duration_s
# --- endregion ---

# --- region: report ---
nodes, obs_topic, action_topic, state, policy_src = build_graph()
print(f"[ch2.8-runtime] clock={args.clock}  policy={policy_src}")
print(f"[ch2.8-runtime] rates: sensor {args.sensor_hz:g} Hz | control {args.control_hz:g} Hz | "
      f"actuator {ACTUATOR_HZ:g} Hz | queue_depth {args.queue_depth}")

t0 = time.monotonic()
runner = run_virtual if args.clock == "virtual" else run_real
sim_time = runner(nodes, state, args.duration_s)
wall_s = time.monotonic() - t0

balanced = not state["fell"]
steps = state["steps"]
mean_latency_ms = 1000.0 * state["latency_sum"] / state["latency_n"] if state["latency_n"] else 0.0
# Effective message rate = messages actually published / sim seconds elapsed.
obs_rate = obs_topic.published / sim_time if sim_time > 0 else 0.0
action_rate = action_topic.published / sim_time if sim_time > 0 else 0.0

verdict = "BALANCED (pole up the whole run)" if balanced else f"FELL at {sim_time:.2f}s"
print(f"[ch2.8-runtime] {verdict} — {steps} control steps in {sim_time:.2f}s sim ({wall_s:.2f}s wall)")
print(f"[ch2.8-runtime] topics: /obs {obs_rate:.1f} msg/s ({obs_topic.dropped} dropped) | "
      f"/action {action_rate:.1f} msg/s ({action_topic.dropped} dropped)")
print(f"[ch2.8-runtime] mean sense->act latency: {mean_latency_ms:.1f} ms  "
      f"(control period {1000.0 / args.control_hz:.0f} ms)")

metrics = {
    "balanced": bool(balanced),
    "break_bug": bool(args.break_bug),
    "clock": args.clock,
    "control_hz": round(float(args.control_hz), 4),
    "final_pole_angle": round(float(state["pole_angle"]), 4),
    "mean_latency_ms": round(float(mean_latency_ms), 4),
    "obs_dropped": int(obs_topic.dropped),
    "obs_published": int(obs_topic.published),
    "action_dropped": int(action_topic.dropped),
    "action_published": int(action_topic.published),
    "queue_depth": int(args.queue_depth),
    "seed": args.seed,
    "sensor_hz": round(float(args.sensor_hz), 4),
    "sim_time_s": round(float(sim_time), 4),
    "smoke": bool(args.smoke),
    "steps": int(steps),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
print(f"metrics: {args.out / 'metrics.json'}")
if args.rerun:
    print(f"recording: {args.out / 'runtime.rrd'} — open it with: rerun {args.out / 'runtime.rrd'}")
# --- endregion ---
