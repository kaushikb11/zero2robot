# Overridable so CI (system python) and local (venv) share targets: make PY=python check
PY ?= .venv/bin/python
PYTEST ?= .venv/bin/pytest
RUFF ?= .venv/bin/ruff

check: ## lint + ALL ci-cpu pedagogy gates + unit (run before every PR)
	$(RUFF) check .
	$(PY) infra/ci/check_loc_caps.py           # 450 cap on chapter artifacts
	$(PY) infra/ci/check_forbidden_imports.py
	$(PY) infra/ci/check_prose_code_drift.py   # region checksums vs site includes
	$(PY) infra/ci/check_prose_includes.py     # include directives fenced; no leaked rtrt
	$(PY) infra/ci/check_demo_assets.py        # demo embeds: no dangling model assets
	$(PY) infra/ci/check_draft_scaffolding.py  # no leftover draft residue in prose/meta
	$(PY) infra/ci/check_wallclock_provenance.py
	$(PY) infra/ci/check_notebook_hashes.py    # notebook_hash in meta matches the artifact (fast static check; execution is the nightly check-notebooks-exec lane)
	$(PYTEST) curriculum tests infra grader -m "not gpu and not slow" -q
check-full: check ## everything in `check` PLUS the changed-chapter CPU smoke (full ci-cpu parity)
	$(PY) infra/ci/smoke_chapters.py --changed-only --seed 0 --verify-determinism
smoke: ## changed-chapter CPU smoke with determinism verification
	$(PY) infra/ci/smoke_chapters.py --changed-only --seed 0 --verify-determinism
check-export-parity: ## torch<->ONNX parity smoke for every export_*_onnx.py (proof the browser path serializes)
	$(PY) infra/ci/check_export_parity.py
check-notebooks-exec: ## EXECUTE every notebook headless on CPU (cpu-smoke). Separate/nightly lane — NOT in `check` (needs the notebooks extra: pip install -e ".[dev,notebooks]")
	$(PY) infra/ci/run_notebooks.py --profile cpu-smoke --fail-fast false --report infra/ci/reports/notebooks.json
full-system-smoke: ## clean-clone whole-product proof: gates + every chapter --smoke + export parity + site build (add --with-notebooks for the notebook lane). Nightly, NOT in `check`
	$(PY) infra/ci/full_system_smoke.py
.PHONY: check check-full smoke check-export-parity check-notebooks-exec full-system-smoke
