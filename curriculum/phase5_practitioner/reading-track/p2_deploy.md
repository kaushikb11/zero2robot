# P2: Ship It (Edge Deployment & the Last Mile)

<!-- Phase 5 practitioner reading track. PROSE ONLY: no artifact, no exercises,
     no wall-clock table, no meta.yaml. This is a "read the real thing" study
     module: the production edge stack (TensorRT, Jetson) is not free-tier
     buildable, so we read it instead of building it. Commits below are
     UN-PINNED on purpose: verify the path still exists when you read it. -->

## The job nobody put in the paper

You trained a policy. It rolls out in the sim, it exports to ONNX, the parity
check passes. In every chapter so far, that was the finish line. On a real robot
it is the starting line, and the stretch between here and a policy that actually
runs on the arm has a name: **the last mile**. Quantize the network, export it,
serve it, and hit the control rate on the hardware you actually have. That is
four verbs, and each one is a place a demo dies.

It is also a *hireable* skill. "Deployment engineer," "edge ML," and "robot
runtime" are all job titles, and the thing behind them is exactly this chain. We are
not going to build it end to end, because the honest parts of it aren't
free-tier buildable: TensorRT is a closed NVIDIA SDK, and a Jetson is a $250
board you either have on your desk or you don't. Colab's T4 can't stand in: a
cloud A100 is not the thing you deploy *onto*. So this is a reading module. We
read the real stack, next to the three pieces of it the course already built from
scratch, and we tell you exactly where the free lunch stops.

## The chain, concretely

**1. Export: you already did this.** The policy leaves the training script as an
ONNX graph under `curriculum/common/export_onnx.py`'s tensor contract: one
`observation` in, one `action` out, dims stamped in `metadata_props`,
`assert_parity` proving torch and onnxruntime agree to `1e-4`. That file *is* the
handoff. Everything downstream consumes the `.onnx`, and everything downstream
assumes the parity check already passed. If you skipped it, you are debugging the
serializer and the runtime at the same time. Don't.

**2. Measure: p50 and p99, not "fast."** Before you optimize anything, time it.
Load the graph in ONNX Runtime, run inference a few hundred times on real-shaped
inputs, and record the *distribution*: p50 (median) and p99 (tail). The tail is
the one that kills robots. A policy that runs in 12 ms at p50 and 80 ms at p99
will hold the arm fine 99 times and then, once a second, deliver a command a
whole control period late, and chapter 2.8 already showed you what a late
command does to a pole. "Average latency" is the number that hides the crash.

**3. Budget: turn a rate into milliseconds.** A control rate is a latency
budget in disguise. 50 Hz means one command every 20 ms, so your *entire* loop
(read sensors, run the net, post-process, write the command) has 20 ms, and
inference gets some fraction of that. If p99 is over budget, you have exactly
three moves: make the net cheaper (quantize), stop paying inference on every step
(chunk), or lower the rate (and hope the plant tolerates it). The first two are
the real work.

**4. Quantize: INT8, and it's less magic than it sounds.** Quantization maps the
network's float32 weights and activations onto 8-bit integers: pick a scale and a
zero-point per tensor, `q = round(x / scale) + zero_point`, and the hardware's
integer units do the matmul 2–4× faster in a quarter of the memory. The whole
idea is a linear map you could write in three lines, which is the point: once
you've written those lines, the production tool stops being magic and becomes
*bookkeeping about where the scale comes from*. **Dynamic**
quantization computes the activation scale on the fly (good for the
transformer/MLP policies this course trains). **Static** quantization computes it
once, offline, from **calibration data**: a few hundred real observations you
feed through to measure the activation ranges. Static is faster at runtime and is
what a Jetson deployment wants; the price is you have to *have* representative
data and a calibration pass. The accuracy you lose is real and you must
re-measure success rate after quantizing: an INT8 policy is a *different* policy
until the eval suite says otherwise.

**5. Chunk: beat latency by amortizing it.** Here is the move that made modern
VLAs deployable. Instead of predicting one action per inference, predict a
*chunk* (10 to 50 actions) and execute them open-loop while you compute the
next chunk. Your effective control rate is now the *playback* rate of the chunk,
not the inference rate. A policy that needs 60 ms to think can still drive a 50 Hz
arm, because one 60 ms inference bought you 20 actions of runway. This is the same
zero-order-hold idea from chapter 2.8 (the actuator re-applying the last command)
turned into a design tool instead of a failure mode. The catch is the chunk
boundary: naive playback jerks when the next chunk disagrees with the last few
actions of the current one, which is why the real systems overlap and blend
chunks (ACT's temporal ensembling) or inpaint across the seam (real-time
chunking). Latency didn't go away. You hid it behind a queue.

**6. Wire it into a real-time loop.** All of the above lands inside the
concurrent graph you built in chapter 2.8: a sensor node, a policy node pulling
the freshest observation and refilling an action queue, an actuator draining the
queue at a fixed rate. The production version of that graph, for robot policies,
is LeRobot's async-inference server/client split, same shape, hardened for a
GPU box talking to a robot over a network.

## Read the real thing

Un-pinned on purpose: these are living repos, and this module is meant to be
cheap to keep current. Read at `main`; if a path moved, `grep` for the function
name, and never trust a line here over the file in front of you.

**Quantization → `microsoft/onnxruntime`.** Read
`onnxruntime/python/tools/quantization/quantize.py`: the public surface is
`quantize_dynamic()` and `quantize_static()`, with `QuantType` (QInt8 / QUInt8),
`CalibrationDataReader` (the interface *you* implement to feed calibration data),
and `get_qdq_config()`. The `README.md` beside it is the map; the docs page
<https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html>
states the rule of thumb plainly (dynamic for transformers/RNNs, static for CNNs
and edge) and the load-bearing hardware caveat: GPU/TensorRT quantization only
supports S8S8 and only pays off on int8-Tensor-Core silicon (T4 and up). On an
old CPU, quantization can run *slower*. Run it if you have the hardware; measure
before you believe it.

**The GPU runtime → NVIDIA TensorRT.** Not open source, not free-tier, so this is
strictly reading: <https://docs.nvidia.com/deeplearning/tensorrt/latest/>,
specifically the "Working with Quantized Types" chapter of the Developer Guide.
The two workflows to understand are **PTQ calibration** (TensorRT measures
activation ranges from representative data and picks scales, the same
calibration idea as ONNX Runtime's static path) and **QAT** (quantize during
training and import the ranges). ONNX Runtime can hand its graph to the TensorRT
execution provider, which is the bridge from the `.onnx` you exported to an
optimized engine on a Jetson.

**Async policy serving → `huggingface/lerobot`.** This is chapter 2.8's graph as
a shipping product. Read `src/lerobot/async_inference/policy_server.py` and
`robot_client.py` (with `configs.py` and `helpers.py` alongside), and the doc at
`docs/source/async.mdx`. You start it as two processes:
`python -m lerobot.async_inference.policy_server` on the GPU box and
`python -m lerobot.async_inference.robot_client` on the robot, and the client
keeps acting from its action queue, requesting a fresh chunk when the queue drops
below `chunk_size_threshold`, so the robot never stalls waiting for inference.
Before you put that on a network, one security caveat: the gRPC channel
deserializes pickle and has a known unauthenticated remote-code-execution hole
(CVE-2026-25874, unpatched through v0.5.1), so keep the server and client on a
trusted local network and never expose the port to something you do not control.
That threshold is the exact knob this whole module is about: how much runway the
chunk buys you against the p99 you measured. For *why* the seam between chunks is
hard, read "Real-Time Execution of Action Chunking Flow Policies" (arXiv
2506.07339), the paper behind LeRobot's real-time chunking; as of LeRobot v0.5.0,
RTC is a native inference option you can switch on, not just a paper to read.

## What the course builds, and what it doesn't

Three pieces of this chain are durable enough that the course builds them from
scratch, and you already have them:

- **The export contract**: `curriculum/common/export_onnx.py`, driven end to end
  by ch5.8's `real_loop.py` (the record → train → deploy → eval loop that exports
  the `.onnx` and proves parity). The `.onnx` and its parity check are the
  artifact the entire last mile consumes. Quantization, TensorRT, and the LeRobot
  server all start from a file that looks like the one you export.
- **The INT8 quantizer**: ch5.7's `quantize.py`. You built it from scratch in
  numpy you can read: symmetric per-tensor and per-channel scales, static
  activation calibration, and a full-integer forward pass. Not a call into a PTQ
  library, the arithmetic itself.
- **The real-time loop**: chapter 2.8's `runtime.py`. Sense/think/act as
  concurrent nodes, a latency budget you can violate, zero-order hold, and the
  drop counter that tells you a node is falling behind. LeRobot's async server is
  that graph with the blast radius turned up.

Everything *beyond* those (the TensorRT engine build, the Jetson it runs on) is
the production stack, and it is the part that isn't free-tier buildable. That is
not a gap in the course; it is an honest line. The quantization *math* you built
from scratch in ch5.7; the quantization *tooling* is a closed SDK and a $250
board, and no Colab notebook will change that. Knowing exactly where that line
falls (what you build and own,
and what you rent from NVIDIA) is itself the practitioner skill this module is
here to hand you.
