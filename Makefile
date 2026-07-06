# Overridable so CI (system python) and local (venv) share targets: make PY=python check
PY ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff

check: ## lint + ALL ci-cpu pedagogy gates + unit (run before every PR)
	$(RUFF) check .
	$(PY) infra/ci/check_loc_caps.py           # 450 cap on chapter artifacts
	$(PY) infra/ci/check_forbidden_imports.py
	$(PY) infra/ci/check_prose_code_drift.py   # region checksums vs site includes
	$(PY) infra/ci/check_wallclock_provenance.py
	$(PYTEST) curriculum tests infra grader -m "not gpu and not slow" -q
check-full: check ## everything in `check` PLUS the changed-chapter CPU smoke (full ci-cpu parity)
	$(PY) infra/ci/smoke_chapters.py --changed-only --seed 0 --verify-determinism
smoke: ## changed-chapter CPU smoke with determinism verification
	$(PY) infra/ci/smoke_chapters.py --changed-only --seed 0 --verify-determinism
.PHONY: check check-full smoke
