"""Put the repo root on sys.path for every test under tests/ so in-process
imports of `curriculum.*` (e.g. a chapter's golden-parity test importing
gen_demos) resolve the same way chapter artifacts do. Applies to all of
tests/ (common, envs, suggested); the tests/common one predates this and is
harmless overlap."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
