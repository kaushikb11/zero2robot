#!/usr/bin/env python3
"""Generate the ch1.7 "VLA dataset browser + the language-leak" toy's vizdata.

The site toy (site/src/components/toys/VlaBrowserToy.tsx) is a pure DATA panel:
it browses a few real examples from the multi-task language+vision dataset and
contrasts the leakage probe (clean templates vs the --break leak templates that
name the move direction). This script produces the small, TEXT-ONLY JSON it loads.

Provenance / honesty (the whole point of a generator instead of hand-typed JSON):
every number here comes from the chapter artifact
`curriculum/phase1_imitation/ch1.7_vla_data/vla_data.py` run at seed 0 on CPU with
the default config — the exact reference run recorded in that chapter's meta.yaml.
We run it TWICE (clean + --break leak) into a throwaway temp dir, read its emitted
manifest.json / metrics.json / vla_dataset.npz, and re-serialize a handful of
example rows plus the leakage-probe scalars. We assert every scalar matches
meta.yaml's reference_run before writing; a mismatch aborts (never fabricate).

NO binaries are written or committed: the .npz (which holds only tokens, actions,
state, and frozen-CNN feature vectors — NOT raw camera frames) lives in a temp dir
that is deleted on exit. The 96x96 camera FRAMES are never read here at all; the
toy represents "vision" as the frozen encoder's 64-D feature vector (a bar), which
is exactly what a downstream policy conditions on — not a decoded image.

Run:  python site/scripts/vizdata/ch1.7_vla_data.py
      (uses the repo venv interpreter; ~10s on CPU for the two runs)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]  # site/scripts/vizdata/<file> -> repo root
CHAPTER = REPO_ROOT / "curriculum" / "phase1_imitation" / "ch1.7_vla_data"
ARTIFACT = CHAPTER / "vla_data.py"
OUT_JSON = CHAPTER / "demo" / "vizdata.json"

SEED = 0

# ---------------------------------------------------------------- reference (meta.yaml)
# The chapter's recorded reference_run (seed 0, cpu, default config). We assert the
# freshly measured run reproduces these EXACTLY before writing — the honesty gate.
REFERENCE = {
    "num_examples": 479,
    "num_examples_pusht": 309,
    "num_examples_aloha": 170,
    "vocab_size": 46,
    "feature_dim": 64,
    "max_tokens": 16,
    "oov_rate": 0.0,
    "clean_r2": 0.006083,
    "clean_r2_pusht": 0.011662,
    "clean_r2_aloha": 0.000504,
    "leak_r2": 0.713237,
    "leak_r2_pusht": 0.778152,
    "leak_r2_aloha": 0.648323,
}
TOL = 1e-6  # CPU runs are bitwise-reproducible; the metrics are rounded to 6 dp


def run_artifact(out_dir: Path, leak: bool) -> None:
    """Run vla_data.py at seed 0 on CPU (deterministic) into out_dir. --no-rerun
    keeps it hermetic (no .rrd); --device cpu pins the bitwise-reproducible tier."""
    cmd = [
        sys.executable, str(ARTIFACT),
        "--seed", str(SEED), "--device", "cpu", "--no-rerun",
        "--out", str(out_dir),
    ]
    if leak:
        cmd += ["--break", "leak"]
    subprocess.run(cmd, check=True, cwd=REPO_ROOT,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def load_run(out_dir: Path):
    manifest = json.loads((out_dir / "manifest.json").read_text())
    metrics = json.loads((out_dir / "metrics.json").read_text())
    npz = np.load(out_dir / "vla_dataset.npz")
    return manifest, metrics, npz


def decode(tokens, itos: list[str]) -> str:
    """Token ids -> instruction text (drop the special ids, as the model's language
    channel effectively does)."""
    specials = {"<pad>", "<bos>", "<eos>"}
    return " ".join(itos[t] for t in tokens if itos[t] not in specials)


def close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= TOL


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ch1.7-vizdata-") as tmp:
        tmp = Path(tmp)
        clean_dir, leak_dir = tmp / "clean", tmp / "leak"
        print(f"[ch1.7 vizdata] running {ARTIFACT.name} (clean) …")
        run_artifact(clean_dir, leak=False)
        print(f"[ch1.7 vizdata] running {ARTIFACT.name} (--break leak) …")
        run_artifact(leak_dir, leak=True)

        manifest, cm, cnpz = load_run(clean_dir)
        _, lm, lnpz = load_run(leak_dir)

        itos: list[str] = manifest["vocab"]
        tasks = manifest["tasks"]
        task_id = cnpz["task_id"]

        # -------- honesty gate: the freshly measured run MUST match meta.yaml --------
        checks = {
            "num_examples": cm["num_examples"],
            "num_examples_pusht": cm["num_examples_pusht"],
            "num_examples_aloha": cm["num_examples_aloha"],
            "vocab_size": cm["vocab_size"],
            "feature_dim": cm["feature_dim"],
            "max_tokens": cm["max_tokens"],
            "oov_rate": cm["oov_rate"],
            "clean_r2": cm["action_from_language_r2"],
            "clean_r2_pusht": cm["r2_pusht"],
            "clean_r2_aloha": cm["r2_aloha"],
            "leak_r2": lm["action_from_language_r2"],
            "leak_r2_pusht": lm["r2_pusht"],
            "leak_r2_aloha": lm["r2_aloha"],
        }
        for key, got in checks.items():
            want = REFERENCE[key]
            if not close(got, want):
                print(f"[ch1.7 vizdata] ABORT: {key} = {got!r} != meta reference {want!r}",
                      file=sys.stderr)
                return 1
        # the leak must actually be a leak (defensive; the lesson's rock)
        if not (lm["action_from_language_r2"] > cm["action_from_language_r2"] + 0.3):
            print("[ch1.7 vizdata] ABORT: leak r2 does not exceed clean r2 by the expected gap",
                  file=sys.stderr)
            return 1
        print("[ch1.7 vizdata] reference match OK "
              f"(clean r2 {cm['action_from_language_r2']:.6f} / leak r2 "
              f"{lm['action_from_language_r2']:.6f}; {cm['num_examples']} examples)")

        # -------- pick a template-diverse example per (task, template) --------
        # One representative row per distinct clean instruction, with a non-degenerate
        # action (skip end-of-episode zero actions whose "direction" is ill-defined).
        # Deterministic: we scan each task's rows in order and take the first hit.
        examples = []
        for tid, task in enumerate(tasks):
            act_dim = task["act_dim"]
            seen: set[str] = set()
            for idx in np.where(task_id == tid)[0]:
                idx = int(idx)
                clean_text = decode(cnpz["instruction_tokens"][idx], itos)
                if clean_text in seen:
                    continue
                action = cnpz["action"][idx][:act_dim]
                if float(np.abs(action).max()) < 1e-6:
                    continue  # degenerate (settled) frame — a bad browser exemplar
                seen.add(clean_text)
                leak_text = decode(lnpz["instruction_tokens"][idx], itos)
                # the leaked instruction appends "moving <direction>"; the direction is
                # the last content word — it is the word that carries the action.
                leak_direction = leak_text.split()[-1]
                examples.append({
                    "index": idx,
                    "task": task["name"],
                    "task_id": tid,
                    "act_dim": act_dim,
                    "instruction_clean": clean_text,
                    "instruction_leak": leak_text,
                    "leak_direction": leak_direction,
                    # token ids INCLUDING specials/pad, so the toy can show the real
                    # fixed-length [BOS] words [EOS] pad… layout the tokenizer emits.
                    "tokens_clean": [int(t) for t in cnpz["instruction_tokens"][idx]],
                    "tokens_leak": [int(t) for t in lnpz["instruction_tokens"][idx]],
                    # action in the embodiment's NATIVE dims (unpadded) — the bar chart.
                    "action": [round(float(v), 4) for v in action],
                    # the 10-D proprioceptive state (the "numbers" input).
                    "state": [round(float(v), 4) for v in cnpz["state"][idx]],
                    # the FROZEN CNN's 64-D feature vector — "what the policy sees of the
                    # camera" as a bar, NOT a decoded image (no frame binaries anywhere).
                    "image_features": [round(float(v), 4) for v in cnpz["image_features"][idx]],
                })
            if len(seen) < len(task["templates"]):
                print(f"[ch1.7 vizdata] WARN: only {len(seen)}/{len(task['templates'])} "
                      f"templates sampled for task {task['name']}", file=sys.stderr)

        vizdata = {
            "provenance": (
                "curriculum/phase1_imitation/ch1.7_vla_data/vla_data.py, seed 0, "
                "--device cpu (bitwise-reproducible tier), default config "
                "(episodes_per_task 12, frame_stride 2, feature_dim 64, conv_width 16). "
                "Two runs: clean templates and --break leak. All scalars match this "
                "chapter's meta.yaml reference_run (measured 2026-07-06). Real dataset "
                "rows + real leakage-probe R^2; NO camera-frame binaries — 'vision' is "
                "the frozen encoder's 64-D feature vector. Regenerate: "
                "python site/scripts/vizdata/ch1.7_vla_data.py"
            ),
            "seed": SEED,
            "meta": {
                "num_examples": cm["num_examples"],
                "num_examples_pusht": cm["num_examples_pusht"],
                "num_examples_aloha": cm["num_examples_aloha"],
                "vocab_size": cm["vocab_size"],
                "feature_dim": cm["feature_dim"],
                "max_tokens": cm["max_tokens"],
                "oov_rate": cm["oov_rate"],
            },
            "vocab": itos,  # 46 words — lets the toy render token id -> word
            "specials": {"pad": itos.index("<pad>"), "bos": itos.index("<bos>"),
                         "eos": itos.index("<eos>"), "unk": itos.index("<unk>")},
            "tasks": [{"id": t["id"], "name": t["name"], "act_dim": t["act_dim"],
                       "count": t["count"], "templates": t["templates"]} for t in tasks],
            "examples": examples,
            # the leakage probe — the chapter's headline (a MEASURED data diagnostic,
            # not a trained-policy result). clean names the TASK; leak names the MOVE.
            "leak": {
                "clean": {
                    "r2": cm["action_from_language_r2"],
                    "r2_pusht": cm["r2_pusht"],
                    "r2_aloha": cm["r2_aloha"],
                },
                "leak": {
                    "r2": lm["action_from_language_r2"],
                    "r2_pusht": lm["r2_pusht"],
                    "r2_aloha": lm["r2_aloha"],
                },
            },
        }

        OUT_JSON.write_text(json.dumps(vizdata, indent=2) + "\n")
        size_kb = OUT_JSON.stat().st_size / 1024
        print(f"[ch1.7 vizdata] wrote {OUT_JSON.relative_to(REPO_ROOT)} "
              f"({len(examples)} examples, {size_kb:.1f} KB, text only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
