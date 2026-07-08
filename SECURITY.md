# Security Policy

Report vulnerabilities to security@zero2robot.dev (alias lands day 1). No public issues for security reports. Priority surfaces: the site and the browser playground (in-page MuJoCo sim + ONNX inference).

The grader is an offline exercise auto-checker: it runs a chapter's public checks on the learner's own machine (drives pytest over `exercises/suggested/checks.py`, reads the public bands from `meta.yaml`). There is no submission server, no hosted grading endpoint, and no execution of untrusted third-party models, so there is no network-facing scoring surface to secure.
