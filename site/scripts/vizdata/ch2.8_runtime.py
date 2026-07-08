#!/usr/bin/env python3
"""Regenerate the ch2.8 pub-sub-runtime concept-toy vizdata from runtime.py, seed 0.

The site's RuntimeGraphToy island renders REAL recorded runs of the ch2.8 node
graph — never an invented animation. This generator REUSES runtime.py's OWN
pieces (build_graph, the Sensor/Policy/Actuator nodes, the two Topics, and the
deterministic virtual-clock scheduler run_virtual) so the browser panels are
bit-faithful to the chapter artifact, then dumps a small JSON the island loads.

It captures the TWO measured, graded facts the chapter teaches (meta.yaml
exercise_checks ex1 + ex2), so the toy can make each one interactive:

  1. the CONTROL-RATE CLIFF (ex1) — the SAME graph run at control_hz in
     {50, 25, 20, 15, 10, 5}. The pole BALANCES the whole 501-step run at
     {50,25,20,15,10} and FALLS at 5 Hz; sense->act latency climbs 0 -> ~60 ms
     as the rate drops from 50 to 10 Hz, and past the cliff the pole topples.
     Each rate carries its own recorded cart_x + pole_angle trajectory so the
     rate slider REPLAYS the real run at whatever rate is selected.
  2. the QUEUE-DEPTH vs RATE lesson (ex2) — at every rate we ALSO run the graph
     with a deep (100) /obs queue. A deeper buffer drives obs_dropped -> 0, yet
     the trajectory is BYTE-IDENTICAL to the shallow (depth-1) run: the policy
     always reads the LATEST message, so the buffer changes only the drop COUNT,
     never the control. Throughput != control — the RATE is the killer, not the
     queue. We assert that byte-identity as an honesty gate.

Plus the NODE GRAPH structure + the VIRTUAL CLOCK schedule (the determinism
story), captured from the healthy 50 Hz reference run.

Why we exec a PREFIX of runtime.py instead of `import runtime`
--------------------------------------------------------------
runtime.py is a loose script (no `if __name__ == "__main__"` guard): importing
it runs the WHOLE report — argparse, banner, build_graph, a full scheduler run,
metrics.json. We must NOT modify runtime.py (it is LOC-capped). So we read its
source and exec only the prefix up to the `# --- region: report ---` line — i.e.
setup + primitives + graph + scheduler — in a throwaway namespace, ONCE PER
CONFIG (argparse runs during exec, so each config re-parses a pinned sys.argv).
That hands us runtime.py's OWN `build_graph`, `run_virtual`, and the node
classes, at each pinned config, with zero edits.

Which brain the graph runs (honesty note)
-----------------------------------------
meta.yaml's reference_run is "the pub-sub graph balances the cartpole 501 steps
deterministically". The DETERMINISTIC, checkpoint-free reference brain that hits
exactly that — and that runtime.py declares as its CI fallback — is the scripted
linear balancer (ch2.1's `balance_action` gains, reading the /obs message). The
committed outputs/ch2.1-ppo/ppo_agent.pt is a CI *smoke* stub (1 iteration,
~45-step eval return) that does NOT balance, so pointing the runtime at it would
FAIL the 501-step meta gate. We therefore force the scripted balancer (via a
non-existent --policy path), which balances 501 steps byte-identically on every
run — the honest, reproducible data that matches meta. runtime.py itself notes
the runtime lesson is identical whichever brain sits behind the topic.

What we verify before writing
-----------------------------
We run the healthy 50 Hz config TWICE and assert the trajectory + metrics are
BYTE-identical (the toy's determinism thesis). At every rate we assert the deep
and shallow queue trajectories are byte-identical (the ex2 thesis). Then we gate
the whole sweep + queue comparison against meta.yaml exercise_checks. The script
STOPS on any drift.

    Run:  .venv/bin/python site/scripts/vizdata/ch2.8_runtime.py
    Out:  curriculum/phase2_reinforcement/ch2.8_runtime/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
RUNTIME_PY = REPO / "curriculum" / "phase2_reinforcement" / "ch2.8_runtime" / "runtime.py"
OUT_JSON = REPO / "curriculum" / "phase2_reinforcement" / "ch2.8_runtime" / "demo" / "vizdata.json"

SEED = 0
CUT_MARKER = "# --- region: report ---"

# Pinned reference config = meta.yaml reference_run: 10 s of sim time on the
# virtual clock, sensor + actuator at 50 Hz, latest-wins depth-1 topics.
# 10 s * 50 Hz = 501 control steps (the actuator ticks at t = 0, 0.02, ..., 10.0).
DURATION_S = 10.0
SENSOR_HZ = 50.0
REF_CONTROL_HZ = 50.0
SHALLOW_QUEUE = 1     # exercise_checks.ex2.shallow_queue_depth — latest-wins, drops on overflow
DEEP_QUEUE = 100      # exercise_checks.ex2.deep_queue_depth   — absorbs the burst, hides the drops

# The control-rate sweep the rate slider walks (meta exercise_checks.ex1). The
# pole balances at every rate >= BALANCES_AT_HZ and falls at FALLS_AT_HZ.
SWEEP_HZ = [50.0, 25.0, 20.0, 15.0, 10.0, 5.0]

# meta.yaml exercise_checks — the honesty gate.
META_STEPS = 501            # a balanced run reaches 501 control steps
META_BALANCED_STEPS_MIN = 500   # exercise_checks.ex1.balanced_steps_min
META_BALANCES_AT_HZ = 10.0      # exercise_checks.ex1.balances_at_hz — survives at >= this rate
META_FALLS_AT_HZ = 5.0          # exercise_checks.ex1.falls_at_hz — falls at this rate
META_NODES = ["sensor", "policy", "actuator"]   # embed.yaml graph_overlay.nodes
META_TOPICS = ["/obs", "/action"]               # embed.yaml graph_overlay.topics

# Physical constants copied from the cartpole env for the replay frame (render
# metadata only — not used in the physics, which runtime.py owns).
ANGLE_LIMIT_RAD = 0.2095   # ~12 deg fall threshold
CART_LIMIT_M = 2.4
POLE_LEN_M = 1.0


def exec_runtime_prefix(control_hz: float, queue_depth: int, policy_path: Path,
                        scratch: Path) -> dict:
    """Exec runtime.py up to the report region, in an isolated namespace. argparse
    runs DURING exec, so we pin the config via sys.argv. Returns the populated
    globals dict (runtime.py's own build_graph + run_virtual + node classes)."""
    src = RUNTIME_PY.read_text()
    cut = src.index(CUT_MARKER)
    prefix = src[:cut]

    old_argv = sys.argv
    sys.argv = [
        str(RUNTIME_PY),
        "--seed", str(SEED),
        "--clock", "virtual",     # deterministic discrete-event scheduler (NOT --smoke: we set 10 s)
        "--device", "cpu",
        "--duration_s", str(DURATION_S),
        "--sensor_hz", str(SENSOR_HZ),
        "--control_hz", str(control_hz),
        "--queue_depth", str(queue_depth),
        "--policy", str(policy_path),   # non-existent => scripted balancer fallback (see module docstring)
        "--no-rerun",
        "--out", str(scratch),
    ]
    # A real module object registered in sys.modules — runtime.py's @dataclass
    # Message needs sys.modules[cls.__module__] to resolve during class creation.
    mod = types.ModuleType("runtime_toy_vizgen")
    mod.__file__ = str(RUNTIME_PY)
    sys.modules["runtime_toy_vizgen"] = mod
    try:
        exec(compile(prefix, str(RUNTIME_PY), "exec"), mod.__dict__)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return mod.__dict__


def capture_run(control_hz: float, queue_depth: int, policy_path: Path,
                scratch: Path, want_schedule: bool = False) -> dict:
    """Exec the prefix at (control_hz, queue_depth) and run runtime.py's OWN
    run_virtual, capturing the cartpole trajectory + final topic/latency metrics
    without perturbing the physics. We wrap tick() to READ state after each fire
    (never mutating), so the scheduler is byte-for-byte the chapter's."""
    ns = exec_runtime_prefix(control_hz, queue_depth, policy_path, scratch)
    build_graph = ns["build_graph"]
    run_virtual = ns["run_virtual"]

    nodes, obs_topic, action_topic, state, policy_src = build_graph()
    by_name = {n.name: n for n in nodes}
    actuator = by_name["actuator"]
    env = actuator.env

    cart_x: list[float] = []
    pole_angle: list[float] = []
    schedule: list[tuple[float, str]] = []   # (virtual time, node) firing order — determinism proof
    SCHEDULE_INSTANTS = 3                     # keep the first few virtual instants only (9 firings)
    schedule_cutoff = (SCHEDULE_INSTANTS - 0.5) / SENSOR_HZ

    def wrap(node):
        orig = node.tick

        def ticked(now: float) -> None:
            if want_schedule and now <= schedule_cutoff:
                schedule.append((round(float(now), 5), node.name))
            orig(now)
            if node is actuator:
                cart_x.append(round(float(env.cart_pos), 6))
                pole_angle.append(round(float(env.pole_angle), 6))

        node.tick = ticked  # type: ignore[method-assign]

    for n in nodes:
        wrap(n)

    sim_time = run_virtual(nodes, state, DURATION_S)
    balanced = not state["fell"]

    metrics = {
        "control_hz": round(float(control_hz), 4),
        "queue_depth": int(queue_depth),
        "balanced": balanced,
        "steps": int(state["steps"]),
        # sim step at which the pole fell (== steps for a fallen run), else null
        "fell_at_step": None if balanced else int(state["steps"]),
        "sim_time_s": round(float(sim_time), 4),
        "obs_published": int(obs_topic.published),
        "obs_dropped": int(obs_topic.dropped),
        "action_published": int(action_topic.published),
        "action_dropped": int(action_topic.dropped),
        "mean_latency_ms": round(
            1000.0 * state["latency_sum"] / state["latency_n"] if state["latency_n"] else 0.0, 3
        ),
        # effective per-topic delivery rate = messages published / sim seconds.
        # /obs tracks the sensor (~50 Hz); /action tracks the policy (~control_hz).
        "obs_rate": round(obs_topic.published / sim_time, 1) if sim_time > 0 else 0.0,
        "action_rate": round(action_topic.published / sim_time, 1) if sim_time > 0 else 0.0,
    }
    return {
        "cart_x": cart_x,
        "pole_angle": pole_angle,
        "schedule": schedule,
        "metrics": metrics,
        "policy_src": policy_src,
    }


def main() -> int:
    # Force the scripted balancer: a checkpoint path that does not exist, so
    # load_policy() falls back to the deterministic linear balancer (see docstring).
    scratch = Path(tempfile.mkdtemp(prefix="ch2.8-viz-"))
    no_ckpt = REPO / "outputs" / "ch2.1-ppo" / "__no_such_checkpoint_for_vizgen__.pt"
    assert not no_ckpt.exists(), "expected a non-existent checkpoint path to force the scripted brain"

    fail: list[str] = []

    # ---------------------------------------------------------- reference 50 Hz, twice → determinism
    ref_a = capture_run(REF_CONTROL_HZ, SHALLOW_QUEUE, no_ckpt, scratch, want_schedule=True)
    ref_b = capture_run(REF_CONTROL_HZ, SHALLOW_QUEUE, no_ckpt, scratch, want_schedule=True)
    if ref_a["cart_x"] != ref_b["cart_x"] or ref_a["pole_angle"] != ref_b["pole_angle"]:
        fail.append("virtual-clock run is NOT reproducible: trajectory differs between two runs")
    if ref_a["metrics"] != ref_b["metrics"]:
        fail.append(f"metrics differ between runs: {ref_a['metrics']} != {ref_b['metrics']}")
    if ref_a["schedule"] != ref_b["schedule"]:
        fail.append("scheduler firing order differs between runs (should be a fixed tie-break)")

    # ---------------------------------------------------------- the control-rate sweep (ex1)
    # At each rate: shallow (depth-1) run for the trajectory + drop count, and a
    # deep (depth-100) run whose ONLY difference must be the drop count (ex2).
    print(f"regenerated ch2.8 runtime [seed {SEED}, --clock virtual, scripted balancer] vs meta.yaml:")
    print(f"  brain           : {ref_a['policy_src']}")
    print("  control-rate sweep (sensor+actuator 50 Hz, latest-wins /obs queue):")
    print(f"    {'hz':>4} {'balanced':>9} {'steps':>6} {'obs_drop(q1)':>13} {'obs_drop(q100)':>15} {'latency_ms':>11}")

    sweep_runs = []
    for hz in SWEEP_HZ:
        shallow = capture_run(hz, SHALLOW_QUEUE, no_ckpt, scratch)
        deep = capture_run(hz, DEEP_QUEUE, no_ckpt, scratch)
        ms, md = shallow["metrics"], deep["metrics"]

        # ex2 thesis: a deeper queue changes only the drop COUNT, never the control.
        if shallow["cart_x"] != deep["cart_x"] or shallow["pole_angle"] != deep["pole_angle"]:
            fail.append(f"@{hz:g}Hz deep queue changed the trajectory — it must only change the drop count")
        if md["balanced"] != ms["balanced"] or md["fell_at_step"] != ms["fell_at_step"]:
            fail.append(f"@{hz:g}Hz deep queue changed the outcome (fell_at {ms['fell_at_step']}->{md['fell_at_step']})")
        if abs(md["mean_latency_ms"] - ms["mean_latency_ms"]) > 1e-6:
            fail.append(f"@{hz:g}Hz deep queue changed the latency — it must not")

        print(f"    {hz:>4g} {str(ms['balanced']):>9} {ms['steps']:>6} "
              f"{ms['obs_dropped']:>13} {md['obs_dropped']:>15} {ms['mean_latency_ms']:>11}")

        sweep_runs.append({
            "control_hz": ms["control_hz"],
            "balanced": ms["balanced"],
            "steps": ms["steps"],
            "fell_at_step": ms["fell_at_step"],
            "sim_time_s": ms["sim_time_s"],
            "obs_published": ms["obs_published"],
            "action_published": ms["action_published"],
            "obs_dropped_shallow": ms["obs_dropped"],   # queue_depth 1  (latest-wins default)
            "obs_dropped_deep": md["obs_dropped"],       # queue_depth 100 (deep buffer)
            "action_dropped": ms["action_dropped"],
            "obs_rate": ms["obs_rate"],
            "action_rate": ms["action_rate"],
            "mean_latency_ms": ms["mean_latency_ms"],
            "cart_x": shallow["cart_x"],                 # deep trajectory is identical (asserted above)
            "pole_angle": shallow["pole_angle"],
        })

    by_hz = {r["control_hz"]: r for r in sweep_runs}

    # ------------------------------------------------------------------ honesty gate (ex1)
    for hz, r in by_hz.items():
        if hz >= META_BALANCES_AT_HZ:
            if not r["balanced"] or r["steps"] < META_BALANCED_STEPS_MIN:
                fail.append(f"@{hz:g}Hz expected BALANCED >= {META_BALANCED_STEPS_MIN} steps, got "
                            f"balanced={r['balanced']} steps={r['steps']}")
        if hz == META_FALLS_AT_HZ and r["balanced"]:
            fail.append(f"@{hz:g}Hz expected the pole to FALL (the control-rate cliff)")
    # the reference 50 Hz run must be the clean, 0-drop, 0-latency healthy graph
    ref = by_hz[REF_CONTROL_HZ]
    if ref["steps"] != META_STEPS or ref["obs_published"] != META_STEPS:
        fail.append(f"reference 50 Hz run off {META_STEPS}: steps={ref['steps']} obs={ref['obs_published']}")
    if ref["obs_dropped_shallow"] != 0 or ref["mean_latency_ms"] != 0.0:
        fail.append("reference 50 Hz run must be the clean graph: 0 dropped, 0 ms latency")
    # latency must climb monotonically as the rate drops from 50 to 10 (ex1)
    lat = [by_hz[h]["mean_latency_ms"] for h in (50.0, 25.0, 20.0, 15.0, 10.0)]
    if any(b <= a for a, b in zip(lat, lat[1:])):
        fail.append(f"sense->act latency must climb as the rate drops 50->10 Hz, got {lat}")
    # every rate: the deep queue must drive drops to 0 (ex2, generalized)
    for hz, r in by_hz.items():
        if r["obs_dropped_deep"] != 0:
            fail.append(f"@{hz:g}Hz deep queue must drive obs_dropped to 0, got {r['obs_dropped_deep']}")
    # the cliff run must actually drop /obs at depth-1 (the visible symptom) while a
    # sub-cliff balanced rate keeps balancing despite dropping too
    cliff = by_hz[META_FALLS_AT_HZ]
    if cliff["obs_dropped_shallow"] <= 0:
        fail.append("the 5 Hz cliff run should drop /obs messages at depth-1 (the symptom)")
    # the balanced replay frames must stay inside the fall cone; the cliff replay must exceed it
    for hz, r in by_hz.items():
        max_ang = float(np.max(np.abs(r["pole_angle"]))) if r["pole_angle"] else 0.0
        inside = max_ang <= ANGLE_LIMIT_RAD
        if r["balanced"] and not inside:
            fail.append(f"@{hz:g}Hz a recorded frame exceeds the fall limit — not a balanced run")
        if not r["balanced"] and inside:
            fail.append(f"@{hz:g}Hz the pole 'fell' but no frame exceeds the fall limit")

    # ------------------------------------------------------------------ node graph + schedule gate
    m = ref_a["metrics"]
    if len(ref_a["cart_x"]) != m["steps"]:
        fail.append("captured trajectory length != step count (capture perturbed the run)")
    first_instant = [name for (t, name) in ref_a["schedule"] if t == 0.0]
    if first_instant != META_NODES:
        fail.append(f"first virtual instant firing order {first_instant} != {META_NODES}")

    if fail:
        print("\nSTOP — regenerated runtime does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------------ pack JSON
    graph = {
        "nodes": [
            {"id": "sensor", "label": "sensor", "role": "reads the plant state",
             "hz": SENSOR_HZ, "priority": 0},
            {"id": "policy", "label": "policy", "role": "runs the scripted balancer",
             "hz": REF_CONTROL_HZ, "priority": 1},
            {"id": "actuator", "label": "actuator", "role": "steps the plant",
             "hz": SENSOR_HZ, "priority": 2},
        ],
        "topics": [
            {"id": "/obs", "from": "sensor", "to": "policy"},
            {"id": "/action", "from": "policy", "to": "actuator"},
        ],
    }
    clock = {
        "kind": "virtual",
        "period_s": round(1.0 / SENSOR_HZ, 6),
        "sim_time_s": m["sim_time_s"],
        "sensor_hz": SENSOR_HZ,
        "actuator_hz": SENSOR_HZ,
        # fixed tie-break: every virtual instant fires in this order (sensor < policy
        # < actuator by node priority), so a --seed run is byte-for-byte reproducible.
        "tick_order": META_NODES,
        "schedule_sample": [{"t": t, "node": name} for (t, name) in ref_a["schedule"]],
        "reproducible": True,   # verified above: two runs are byte-identical
    }
    control_rate_sweep = {
        "knob": "control_hz",
        "sensor_hz": SENSOR_HZ,
        "actuator_hz": SENSOR_HZ,
        "balances_at_hz": META_BALANCES_AT_HZ,
        "falls_at_hz": META_FALLS_AT_HZ,
        "default_hz": REF_CONTROL_HZ,
        "runs": sweep_runs,     # ordered high -> low rate (SWEEP_HZ)
    }
    # ex2, at the cliff rate: same run, two queue depths. The trajectory is shared
    # with the sweep's 5 Hz run (byte-identical, asserted) — we carry only the two
    # differing drop counts + the shared outcome so the toy can flip depth in place.
    cliff = by_hz[META_FALLS_AT_HZ]
    queue_depth = {
        "control_hz": META_FALLS_AT_HZ,
        "shallow_depth": SHALLOW_QUEUE,
        "deep_depth": DEEP_QUEUE,
        "obs_dropped_shallow": cliff["obs_dropped_shallow"],
        "obs_dropped_deep": cliff["obs_dropped_deep"],   # 0 — the drop symptom vanishes
        "balanced": cliff["balanced"],                   # still False either way
        "fell_at_step": cliff["fell_at_step"],           # SAME step — the buffer does not rescue
        "mean_latency_ms": cliff["mean_latency_ms"],     # unchanged — the rate, not the queue
    }

    data = {
        "provenance": {
            "source": "curriculum/phase2_reinforcement/ch2.8_runtime/runtime.py",
            "generator": "site/scripts/vizdata/ch2.8_runtime.py",
            "seed": SEED,
            "device": "cpu",
            "clock": "virtual",
            "config": f"--clock virtual, seed {SEED}, {DURATION_S:g} s sim time, sensor+actuator "
                      f"{SENSOR_HZ:g} Hz, control_hz swept over {[int(h) for h in SWEEP_HZ]}, "
                      f"/obs queue_depth {SHALLOW_QUEUE} vs {DEEP_QUEUE}",
            "brain": ref_a["policy_src"],
            "stack": "torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6",
            "note": "Real recorded runs of runtime.py's OWN build_graph + run_virtual "
                    "(the virtual-clock discrete-event scheduler), captured read-only. "
                    "The reference 50 Hz run is verified byte-identical across two runs "
                    "(determinism), and at every rate the deep-queue run is verified "
                    "byte-identical to the shallow-queue run (a deeper buffer changes only "
                    "the drop COUNT, never the control — ex2). Matches meta.yaml "
                    "exercise_checks: balances 501 steps at control_hz>=10, falls at 5 Hz "
                    "(ex1); a depth-100 /obs queue drives obs_dropped->0 yet the pole falls "
                    "at the SAME step (ex2). The brain is the scripted linear balancer, "
                    "runtime.py's declared checkpoint-free CI reference (the committed ch2.1 "
                    "checkpoint is a smoke stub that does not balance); the runtime lesson is "
                    "identical whichever brain sits behind the /action topic.",
        },
        "cartpole": {
            "dt_s": round(1.0 / SENSOR_HZ, 6),
            "angle_limit_rad": ANGLE_LIMIT_RAD,
            "cart_limit_m": CART_LIMIT_M,
            "pole_len_m": POLE_LEN_M,
        },
        "graph": graph,
        "clock": clock,
        "control_rate_sweep": control_rate_sweep,
        "queue_depth": queue_depth,
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml exercise_checks:")
    print("  ex1 control-rate cliff : balances 501 steps at >=10 Hz, falls at 5 Hz; latency climbs 0->~60 ms")
    print("  ex2 queue-depth vs rate: deep /obs queue -> obs_dropped 0, yet the pole falls at the SAME step")
    print("  determinism            : virtual clock reproducible (50 Hz run byte-identical across two runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
