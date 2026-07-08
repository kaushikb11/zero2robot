# Architecture

zero2robot is an educational embodied-AI product. **The code is the product** — learners
read every line like a textbook — and the whole repository is arranged, and every rule
below exists, to protect that.

## Repo map

| Path | What it is |
|---|---|
| `curriculum/` | Chapter code + prose. **THE PRODUCT.** One runnable file per chapter, plus its prose, exercises, tests, demo config, and `meta.yaml`. Strictest rules apply. |
| `site/` | The interactive textbook (Astro). Renders each chapter from `curriculum/**/prose/chapter.md` + `meta.yaml` — it never holds prose of its own. Live sim embeds + code panels + exercise blocks. |
| `playground/` | MuJoCo-WASM browser playground, teleop UI, ONNX inference (TypeScript). |
| `grader/` | The exercise auto-checker (`python -m grader.check`) and the hidden-seed leaderboard grading server (Python/FastAPI). |
| `infra/` | CI gates, deploy/publish tooling, telemetry, the decision log, agentic metrics. |
| `notebooks/` | Generated Colab variants of chapter code. **Never hand-edited** — regenerated via the `notebook-tier-test` skill and hash-checked in CI. |
| `datasets/`, `checkpoints/` | Pointers/manifests only (`datasets.yaml`, `models.yaml`). Artifacts live on the Hugging Face Hub. Never commit binaries. |
| `scripts/` | Author-side Hub tooling: `fetch_datasets.py` (setup-time provision, sha-verified, graceful) and `upload_models_hf.py`/`upload_datasets_hf.py` (dry-run-default publish). The model fetcher lives at `site/scripts/fetch_models.py` (it runs in the site build). |
| `outputs/` | Git-ignored. Where chapters write trained policies, rerun logs, and regenerated datasets. |

### Per-chapter layout

```
phase1_imitation/ch1.3_act/
├── act.py                # the artifact (≤450 LOC)
├── prose/chapter.md      # HUMAN-OWNED voice; agents touch only via prose-sync PRs
├── exercises/            # checkable exercises + checks.py (local pytest)
├── tests/                # HUMAN-OWNED correctness tests; agents never modify
├── demo/                 # browser embed config for this chapter
└── meta.yaml             # objectives, wall-clock entries, deps, region hashes, scale-lab ref
```

## The doctrine

These are the invariants that make the code teachable and the product honest. Violating
any of them fails review.

1. **Single file per artifact, ≤450 LOC.** Each chapter builds exactly one runnable file.
   A hook enforces a 450-line hard cap (target ≤400). `infra/ci/check_loc_caps.py`.
2. **No clever abstractions.** No inheritance hierarchies, no config frameworks, no
   decorators for their own sake. If BC and ACT share a data loader, they each keep their
   own copy — the repetition is the lesson (the learner sees exactly what changed).
3. **No forbidden imports in chapter files.** Hook-enforced: `hydra`, `omegaconf`,
   `pytorch_lightning`, `transformers` (except the tiny-VLA chapters that explicitly opt
   in), `stable_baselines3`, sim-hiding `gym` wrappers. Allowed: `torch`, `numpy`,
   `mujoco`, `rerun`, stdlib; MJX chapters add `jax`/`flax`/`optax`.
   `infra/ci/check_forbidden_imports.py`.
4. **Determinism, tiered honestly.** Env resets are **bitwise-reproducible** on CPU MuJoCo
   (enforced in CI). Training is **statistically** reproducible: same seed → same
   qualitative result and metrics within a recorded seeded-run band on the same tier.
   Bitwise *training* reproducibility is **not** promised on GPU (cuDNN/JAX nondeterminism)
   and prose never claims it. `--seed` is mandatory everywhere; ch1.6 teaches this exact
   distinction.
5. **Wall-clock from the ledger.** Every wall-clock claim in prose comes from
   `curriculum/common/wallclock.csv`, measured on real hardware by the `wallclock-bench`
   skill — never estimated. `infra/ci/check_wallclock_provenance.py`.
6. **Region-hash drift gate.** The site renders code panels by including real regions of
   the chapter file (`# --- region: model ---`), never pasted copies. Each region's sha256
   is recorded in `meta.yaml`; CI fails if the rendered page drifts from the source.
   `infra/ci/check_prose_code_drift.py`.
7. **No binaries in git** (root CLAUDE.md #5). Trained policies and datasets live on the
   Hugging Face Hub; the repo holds only manifests (`checkpoints/models.yaml`,
   `datasets/datasets.yaml`) and the fetch/upload scripts. `.gitignore` blocks
   `*.onnx`, `*.pt`, `*.safetensors`, `outputs/`, etc.
8. **Version pins are frozen in feature PRs.** `mujoco`, `mjx`, `lerobot`, `rerun-sdk`,
   `torch` are pinned in `pyproject.toml`; upgrades come only through dedicated
   `upstream-pin-check` PRs, never a feature branch.
9. **Free-tier floor.** Every learner-facing path completes on a Colab T4 or CPU laptop.
   Anything heavier is a clearly-marked, optional **Scale Lab**.

## Data & model provisioning

Binaries never touch git, so trained policies and demo datasets are provisioned from the
Hub at build/setup time, sha256-verified against an in-repo pointer manifest:

- **Models** — `checkpoints/models.yaml` → `site/scripts/fetch_models.py` (used by the site
  build; a missing model degrades the live demo to its poster frame). Author publishes via
  `scripts/upload_models_hf.py` (dry-run default).
- **Datasets** — `datasets/datasets.yaml` → `scripts/fetch_datasets.py`. Most datasets are
  **regenerated** from seeded, deterministic generators already in the repo (`gen_demos.py`,
  `vla_data.py`), so the manifest distinguishes `source: regenerate` (run the command —
  the primary path), `source: fetch` (an external upstream dataset like `lerobot/pusht`,
  which we do not re-host), and `source: reference-only` (OXE/DROID — discussed at scale,
  never downloaded). Optional Hub mirrors let a fresh Colab skip regeneration; the author
  publishes them via `scripts/upload_datasets_hf.py` (dry-run default). A failed fetch
  never breaks a build — it falls back to the regenerate command.

## CI gates

Three lanes (see `.github/workflows/`):

- **ci-cpu** — every PR. `make check` runs: `ruff`, the LOC cap, forbidden-imports,
  prose↔code region-drift, include-directive hygiene, demo-asset triangle, draft-scaffolding
  residue, wall-clock provenance, then `pytest` over `curriculum tests infra grader`
  (`-m "not gpu and not slow"`). `make check-full` adds the changed-chapter CPU smoke with
  twice-run determinism verification.
- **ci-notebook** — nightly, headless execution of the generated Colab notebooks to hold the
  free-tier T4 constraint; notebook hashes are checked so nobody hand-edits a generated file.
- **ci-gpu** — manual/weekly (once a runner exists); GPU wall-clock fan-out via Modal
  (`infra/modal/`, decision 014).

Ownership boundaries baked into the gates: `curriculum/**/prose/` and `curriculum/**/tests/`
are **human-owned** — agents may fix factual/code-sync issues in prose only via `prose-sync`
PRs, and never modify tests or `grader/hidden_seeds/`.
