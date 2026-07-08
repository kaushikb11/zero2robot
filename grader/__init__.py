"""grader/ — zero2robot's exercise auto-checker (INFRA, not pedagogy).

One job (grader/CLAUDE.md): exercise auto-checking — formalize the "run it
locally" self-check (grader.check): discover a chapter's
exercises/suggested/checks.py and run them offline via pytest, reporting
pass/fail/skip per test. No network, no scoring server, no hidden seeds; this
path only runs pytest on human-owned check files and reads the public
exercise_checks bands from each chapter's meta.yaml.

(The leaderboard submission grading server and its scoring/contract/sandbox
surface were removed as a product decision — see infra/decisions/013.)
"""

__version__ = "0.0.1"
