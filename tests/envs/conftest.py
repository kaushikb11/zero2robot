import sys
from pathlib import Path

# Make `curriculum.common.envs.*` importable without installing the repo
# (pyproject ships no packages; curriculum/ is a namespace path).
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
