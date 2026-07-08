#!/usr/bin/env python3
"""Regenerate the ch3.1 World-Models concept-toy vizdata from wm.py, seed 0, cpu.

The site's WorldModelToy island renders REAL prediction curves and the REAL
per-dim pusher-vs-object split that wm.py itself measured — never invented shapes.
Its ONE honest job is to carry the chapter's load-bearing caveat into the browser:
the ~2.3x aggregate "world model beats copy-last" win is carried ENTIRELY by the
trivially-integrable PUSHER dims; on the OBJECT/block dims (the tee pose PushT is
actually about) COPY-LAST WINS. The toy MUST show that split, so this generator
STOPS unless the regenerated numbers still exhibit it (pusher wins / object loses).

Why we EXEC the whole wm.py instead of `import wm`
--------------------------------------------------
wm.py is a loose script (no `if __name__ == "__main__"` guard, no eval function):
the entire encoder/GRU/prior/posterior/decoder build, the training loop, AND the
recon-vs-prediction eval all run at MODULE level, top to bottom. Importing it would
run everything anyway — so we simply exec the whole file in a throwaway namespace
with argv pinned to the deterministic CPU config (seed 0, --no-rerun), then read the
REAL objects wm.py left in that namespace: the per-k arrays `wm_err`/`copy_err`, the
posterior `val_recon`, the crossover, and the four per-group MSEs `wm_push`/
`copy_push`/`wm_obj`/`copy_obj` (over `PUSHER_DIMS`/`OBJECT_DIMS`). No stdout
parsing; the numbers ARE wm.py's own tensors. We do NOT modify wm.py (LOC-capped).

CPU + seed 0 is byte-reproducible (root CLAUDE.md invariant 2; wm.py's docstring
says so), so seed-0 lands inside meta.yaml's seed-sweep bands every run. We gate
against those bands (not a single magic number) and against the mandatory split.

    Run:  .venv/bin/python site/scripts/vizdata/ch3.1_world_models.py
    Out:  curriculum/phase3_advanced/ch3.1_world_models/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
WM_PY = REPO / "curriculum" / "phase3_advanced" / "ch3.1_world_models" / "wm.py"
OUT_JSON = REPO / "curriculum" / "phase3_advanced" / "ch3.1_world_models" / "demo" / "vizdata.json"

SEED = 0

# meta.yaml reference_run seed-sweep bands (seeds 0-2, cpu, default config; measured
# 2026-07-06). seed 0 lands inside every band; we gate against the BANDS, not exact
# magic numbers, because these are sweep bands and seed 0 is one draw inside them.
BAND_VAL_RECON = (0.030, 0.040)      # meta 0.0336-0.0358 (reconstruction floor)
BAND_WM_MEAN = (0.052, 0.063)        # meta 0.0549-0.0608 (k-step prediction mean)
BAND_COPY_MEAN = (0.128, 0.150)      # meta 0.132-0.146 (copy-last mean)
BAND_RATIO = (2.1, 2.55)             # meta 2.18-2.45 (copy/wm aggregate — >2x lower)
CROSSOVER_OK = {2, 3}                # meta crossover_k 2-3 (aggregate)
PUSHER_RATIO_MIN = 5.0               # meta: pusher WM ~7-8x lower than copy-last


def exec_wm() -> dict:
    """Exec the whole wm.py (seed 0, cpu, no rerun) in an isolated namespace and
    return its populated globals — the REAL arrays/scalars wm.py computed."""
    src = WM_PY.read_text()
    scratch = Path(tempfile.mkdtemp(prefix="ch3.1-viz-"))
    old_argv = sys.argv
    # argparse runs DURING exec — pin the deterministic, byte-reproducible config.
    sys.argv = [str(WM_PY), "--seed", str(SEED), "--device", "cpu",
                "--no-rerun", "--out", str(scratch)]
    ns: dict = {"__file__": str(WM_PY), "__name__": "wm_toy_vizgen"}
    try:
        exec(compile(src, str(WM_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def main() -> int:
    ns = exec_wm()

    # The REAL objects wm.py left behind (per-k MSE arrays, posterior recon, the
    # per-group split over PUSHER_DIMS/OBJECT_DIMS). No re-derivation, no parsing.
    wm_err = np.asarray(ns["wm_err"], dtype=float)        # per-k world-model MSE
    copy_err = np.asarray(ns["copy_err"], dtype=float)    # per-k copy-last MSE
    val_recon = float(ns["val_recon"])                    # posterior reconstruction floor
    crossover_k = int(ns["crossover_k"])
    wm_mean = float(ns["wm_mean"])
    copy_mean = float(ns["copy_mean"])
    wm_push, copy_push = float(ns["wm_push"]), float(ns["copy_push"])
    wm_obj, copy_obj = float(ns["wm_obj"]), float(ns["copy_obj"])
    pusher_dims = list(ns["PUSHER_DIMS"])
    object_dims = list(ns["OBJECT_DIMS"])
    args = ns["args"]
    horizon, context = int(args.horizon), int(args.context)

    ratio = copy_mean / max(wm_mean, 1e-9)
    push_ratio = copy_push / max(wm_push, 1e-9)         # >1 => WM wins the pusher
    obj_ratio = copy_obj / max(wm_obj, 1e-9)            # <1 => copy-last wins the object
    ks = list(range(1, horizon + 1))

    # ---------------------------------------------------------------- console echo
    print(f"regenerated wm.py [seed {SEED}, cpu] vs meta.yaml reference_run bands:")
    print(f"  val_recon (posterior)   : {val_recon:.6f}   (meta band 0.0336-0.0358)")
    print(f"  world_model_pred_mean   : {wm_mean:.6f}   (meta band 0.0549-0.0608)")
    print(f"  copy_last_pred_mean     : {copy_mean:.6f}   (meta band 0.132-0.146)")
    print(f"  pred_ratio copy/wm      : {ratio:.3f}x     (meta 2.18-2.45)")
    print(f"  crossover_k (aggregate) : {crossover_k}         (meta 2-3)")
    print("  --- THE HONEST SPLIT (the load-bearing lesson) ---")
    print(f"  PUSHER dims {pusher_dims}: wm {wm_push:.5f} vs copy {copy_push:.5f}  "
          f"-> {push_ratio:.1f}x lower  ({'WM WINS' if wm_push < copy_push else 'wm loses'})")
    print(f"  OBJECT dims {object_dims}: wm {wm_obj:.5f} vs copy {copy_obj:.5f}  "
          f"-> {obj_ratio:.2f}x  ({'COPY-LAST WINS' if wm_obj > copy_obj else 'wm wins'})")

    # ------------------------------------------------------------------ honesty gate
    fail: list[str] = []
    if not BAND_VAL_RECON[0] <= val_recon <= BAND_VAL_RECON[1]:
        fail.append(f"val_recon {val_recon} outside band {BAND_VAL_RECON}")
    if not BAND_WM_MEAN[0] <= wm_mean <= BAND_WM_MEAN[1]:
        fail.append(f"world_model_pred_mean {wm_mean} outside band {BAND_WM_MEAN}")
    if not BAND_COPY_MEAN[0] <= copy_mean <= BAND_COPY_MEAN[1]:
        fail.append(f"copy_last_pred_mean {copy_mean} outside band {BAND_COPY_MEAN}")
    if not BAND_RATIO[0] <= ratio <= BAND_RATIO[1]:
        fail.append(f"pred_ratio {ratio} outside band {BAND_RATIO}")
    if crossover_k not in CROSSOVER_OK:
        fail.append(f"crossover_k {crossover_k} not in {CROSSOVER_OK}")
    # aggregate ordering: copy-last WINS at k=1 (nothing moved yet)
    if not copy_err[0] < wm_err[0]:
        fail.append(f"k=1 copy-last should win: copy {copy_err[0]} !< wm {wm_err[0]}")
    # THE MANDATORY SPLIT — this is why the toy exists:
    #  * pusher: WM beats copy-last by a wide margin (a trivial velocity integral)
    #  * object: COPY-LAST WINS (the contact dynamics the tiny model did NOT learn)
    if not (wm_push < copy_push and push_ratio >= PUSHER_RATIO_MIN):
        fail.append(f"pusher split broken — WM must beat copy-last >{PUSHER_RATIO_MIN}x: "
                    f"wm {wm_push} copy {copy_push} ({push_ratio:.1f}x)")
    if not wm_obj > copy_obj:
        fail.append(f"object split broken — COPY-LAST must win the block dims: "
                    f"wm {wm_obj} copy {copy_obj}")
    if fail:
        print("\nSTOP — regenerated wm.py does NOT match meta.yaml / the honest split:")
        for f in fail:
            print("  x " + f)
        return 1

    # ---------------------------------------------------------------------- pack it
    data = {
        "provenance": {
            "source": "curriculum/phase3_advanced/ch3.1_world_models/wm.py",
            "generator": "site/scripts/vizdata/ch3.1_world_models.py",
            "seed": SEED,
            "device": "cpu",
            "config": "default (latent_dim 16, hidden_dim 128, seq_len 24, episodes 120, "
                      "epochs 60, horizon 12, context 3); byte-reproducible on CPU",
            "stack": "torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6",
            "note": "Real per-k prediction curves and the per-dim pusher-vs-object split "
                    "wm.py itself measured (its own wm_err/copy_err arrays + wm_push/"
                    "copy_push/wm_obj/copy_obj). seed 0 lands inside meta.yaml's seed-0-2 "
                    "bands. THE HONEST LESSON: the aggregate win is carried by the pusher "
                    "kinematics; on the object/block dims COPY-LAST WINS — the tiny model "
                    "learned the easy half, not the hard (contact) half.",
        },
        "horizon": horizon,
        "context": context,
        "k": ks,
        # Panel 1 — the aggregate crossover curves (per horizon k).
        "aggregate": {
            "world_model": [round(float(v), 6) for v in wm_err],
            "copy_last": [round(float(v), 6) for v in copy_err],
            "crossover_k": crossover_k,
            "wm_mean": round(wm_mean, 6),
            "copy_mean": round(copy_mean, 6),
            "ratio_copy_over_wm": round(ratio, 4),
        },
        # Panel 2 — THE HEADLINE. WHERE the aggregate win lives, per obs group.
        "split": {
            "pusher": {
                "label": "pusher xy",
                "sublabel": "commanded-velocity integral — the easy half",
                "dims": pusher_dims,
                "wm": round(wm_push, 6),
                "copy": round(copy_push, 6),
                "ratio_copy_over_wm": round(push_ratio, 2),
                "wm_wins": True,
            },
            "object": {
                "label": "tee pose (block)",
                "sublabel": "contact dynamics — the hard half PushT is about",
                "dims": object_dims,
                "wm": round(wm_obj, 6),
                "copy": round(copy_obj, 6),
                "ratio_copy_over_wm": round(obj_ratio, 2),
                "wm_wins": False,
            },
            "headline": "It learned the easy half (pusher kinematics), not the hard half "
                        "(block dynamics).",
        },
        # Panel 3 — reconstruction (posterior) is easy/low vs prediction; + the pixel
        # Scale Lab caveat (pixels can't beat copy-last free-tier).
        "reconstruction": {
            "val_recon": round(val_recon, 6),
            "prediction_mean": round(wm_mean, 6),
            "prediction_floor": round(float(wm_err.min()), 6),
            "note": "Reconstruction (posterior — it has SEEN the frame) is easy and stays "
                    "low; the real test is PREDICTION (prior rollout on actions alone). "
                    "On PIXELS even prediction can't beat copy-last free-tier — that is the "
                    "Scale Lab ('why world models eat compute').",
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml bands; aggregate crossover honest; THE SPLIT holds: "
          "pusher WINS, object LOSES (copy-last wins the block dynamics).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
