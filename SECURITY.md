# Security Policy

Report vulnerabilities to security@zero2robot.dev (alias lands day 1). No public issues for security reports. Priority surfaces: the grader (executes learner-submitted ONNX — see grader/sandbox/policy.yaml), the leaderboard API, and the site.

Grader threat model (non-negotiable): learner submissions are hostile by default. ONNX deserialization happens only inside the sandbox; resource caps (CPU, memory, wall-clock, no network) enforced at the container level; syscall-level isolation (gVisor-class) required before the leaderboard opens at Drop 4 — container-only isolation is NOT sufficient for internet-facing arbitrary-model execution.
