"""Put the repo root on sys.path so tests import curriculum.common the same
way chapter artifacts do (repo root on sys.path, absolute imports)."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
