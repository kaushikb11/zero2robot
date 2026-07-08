#!/usr/bin/env python3
"""Regenerate the ch3.8 "Reading the Frontier" concept-toy vizdata from probe.py, seed 0.

The site's ProbeToy island renders REAL numbers — never invented ones. This
generator REUSES probe.py's OWN pieces (the TinyPolicy skeleton it loads from
disk, the inspect table, the forward-hook capture, and the four linear-probe
reads) so the browser panels are bit-faithful to the chapter artifact, then dumps
a small JSON the island loads.

Why we exec a PREFIX of probe.py instead of `import probe`
----------------------------------------------------------
probe.py is a loose script (no `if __name__ == "__main__"` guard): importing it
runs the WHOLE report — argparse, banner, train+save+reload the checkpoint, the
inspect table, the hook, the four probes, metrics.json, and a rerun recording. We
must NOT modify probe.py (it is LOC-capped and its metrics.json is byte-compared
in CI). So we read its source and exec only the prefix up to the
`# --- region: report ---` line — i.e. setup + checkpoint + inspect + forward +
probe — in a throwaway namespace. That hands us probe.py's OWN loaded `policy`,
`module_params` / `total_params`, the hooked `cls_attention`, and the four probe
scalars (`trained_task_acc`, `control_task_acc`, `trained_coord_r2`,
`control_coord_r2`, `chance_task_acc`), at the DEFAULT config (seed 0, cpu,
torch), with zero edits. CPU + single-thread makes these byte-reproducible.

The HONEST core of this toy (author: keep this framing)
-------------------------------------------------------
The load-bearing lesson is that a probe recovering an INPUT proves nothing: the
task-id probe hits ~1.0 on BOTH the trained checkpoint AND a random-init control,
because the instruction token is a literal input to the fused layer. The signal
that MEANS something is the routed-coordinate R^2 — a value the model had to
COMPUTE — which is high only after training (0.90 trained vs 0.16 random). This
generator STOPS if the regenerated numbers drift from meta.yaml's reference_run,
so the toy can never quietly tell a prettier story than the measurement.

    Run:  .venv/bin/python site/scripts/vizdata/ch3.8_frontier.py
    Out:  curriculum/phase3_advanced/ch3.8_frontier/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PROBE_PY = REPO / "curriculum" / "phase3_advanced" / "ch3.8_frontier" / "probe.py"
OUT_JSON = REPO / "curriculum" / "phase3_advanced" / "ch3.8_frontier" / "demo" / "vizdata.json"

SEED = 0
CUT_MARKER = "# --- region: report ---"

# meta.yaml reference_run (seed 0, cpu, torch 2.10.0 / numpy 2.4.6) — the honesty
# gate. CPU single-thread training is byte-reproducible, so these are exact; the
# tolerances only guard float print noise. If the regenerated probe drifts from
# these, STOP: the toy must match the measurement, not a nicer story.
META = {
    "total_params": 7266,
    "cls_attends_to_index": 3,
    "chance_task_acc": 0.25,
    "trained_task_acc": 1.0,
    "control_task_acc": 1.0,
    "trained_coord_r2": 0.903798,
    "control_coord_r2": 0.161236,
}
R2_TOL = 5e-4        # routed-coord R^2: byte-reproducible on cpu; tol guards last-bit float noise
ACC_TOL = 1e-6       # task-id accuracy is an exact fraction (held-out count / count)

# The fused-token sequence probe.py builds: [CLS, state, tok_0..3] where the
# instruction tokens are [BOS, task, EOS, PAD]. So the CLS-attention bars read as:
SEQ_LABELS = ["CLS", "state", "BOS", "task", "EOS", "PAD"]


def exec_probe_prefix() -> dict:
    """Exec probe.py up to the report region, in an isolated namespace. Returns
    the populated globals dict (probe.py's own loaded policy + probe scalars)."""
    src = PROBE_PY.read_text()
    cut = src.index(CUT_MARKER)
    prefix = src[:cut]

    scratch = Path(tempfile.mkdtemp(prefix="ch3.8-viz-"))
    # argparse runs DURING exec — pin the default config on cpu, no rerun path.
    old_argv = sys.argv
    sys.argv = [str(PROBE_PY), "--seed", str(SEED), "--no-rerun",
                "--device", "cpu", "--out", str(scratch)]
    ns: dict = {"__file__": str(PROBE_PY), "__name__": "probe_toy_vizgen"}
    try:
        exec(compile(prefix, str(PROBE_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def main() -> int:
    ns = exec_probe_prefix()
    policy = ns["policy"]
    total_params = int(ns["total_params"])
    module_params = ns["module_params"]
    cls_attention = np.asarray(ns["cls_attention"], dtype=float)
    trained_task_acc = float(ns["trained_task_acc"])
    control_task_acc = float(ns["control_task_acc"])
    trained_coord_r2 = float(ns["trained_coord_r2"])
    control_coord_r2 = float(ns["control_coord_r2"])
    chance_task_acc = float(ns["chance_task_acc"])
    config = dict(ns["CONFIG"])

    # ---------------------------------------------------------------- inspect table
    # Rebuild the per-module shape list EXACTLY as probe.py's inspect region prints
    # it (named_children -> per-param shapes), so the architecture panel is the same
    # table you would print for pi0 or GR00T. Flag the fused `norm` layer — the one
    # the next panel probes.
    modules = []
    for name in module_params:
        shapes = [
            {"name": n.split(".")[-1], "shape": list(p.shape)}
            for n, p in policy.named_parameters()
            if n.startswith(name + ".")
        ]
        modules.append({
            "name": name,
            "params": int(module_params[name]),
            "shapes": shapes,
            "probed": name == "norm",   # the FUSED layer the linear probe reads
        })
    # `named_children()` (probe.py's inspect table) misses the two DIRECT
    # nn.Parameters — the learned CLS token and the positional embeddings — which
    # is why probe.py prints TOTAL separately. Capture them so the panel's total is
    # honest: sum(modules) + these == total_params.
    direct = [
        {"name": n, "params": int(p.numel()), "shape": list(p.shape)}
        for n, p in policy.named_parameters() if "." not in n
    ]
    assert sum(m["params"] for m in modules) + sum(d["params"] for d in direct) == total_params, \
        "module + direct params != total"

    argmax_index = int(cls_attention.argmax())

    # ------------------------------------------------------------------ honesty gate
    print("regenerated probe [seed 0, cpu] vs meta.yaml reference_run:")
    print(f"  total_params        : {total_params:>10d}   (meta {META['total_params']})")
    print(f"  cls_attends_to_index: {argmax_index:>10d}   (meta {META['cls_attends_to_index']}  = the '{SEQ_LABELS[argmax_index]}' token)")
    print(f"  task-id  trained    : {trained_task_acc:>10.6f}   (meta {META['trained_task_acc']})")
    print(f"  task-id  control    : {control_task_acc:>10.6f}   (meta {META['control_task_acc']})  <- ~1.0 on BOTH: a probe recovered an INPUT")
    print(f"  coord R^2 trained   : {trained_coord_r2:>10.6f}   (meta {META['trained_coord_r2']})")
    print(f"  coord R^2 control   : {control_coord_r2:>10.6f}   (meta {META['control_coord_r2']})  <- the SIGNAL: a value the model had to COMPUTE")

    fail = []
    if total_params != META["total_params"]:
        fail.append(f"total_params {total_params} != {META['total_params']}")
    if argmax_index != META["cls_attends_to_index"]:
        fail.append(f"cls_attends_to_index {argmax_index} != {META['cls_attends_to_index']}")
    if abs(trained_task_acc - META["trained_task_acc"]) > ACC_TOL:
        fail.append(f"trained_task_acc {trained_task_acc} != {META['trained_task_acc']}")
    if abs(control_task_acc - META["control_task_acc"]) > ACC_TOL:
        fail.append(f"control_task_acc {control_task_acc} != {META['control_task_acc']}")
    if abs(trained_coord_r2 - META["trained_coord_r2"]) > R2_TOL:
        fail.append(f"trained_coord_r2 {trained_coord_r2} != {META['trained_coord_r2']}")
    if abs(control_coord_r2 - META["control_coord_r2"]) > R2_TOL:
        fail.append(f"control_coord_r2 {control_coord_r2} != {META['control_coord_r2']}")
    # The load-bearing INVARIANTS of the chapter, asserted as directions (not just
    # the third decimal): task-id saturates on BOTH (the input-recovery caveat), and
    # the routed-coord R^2 gap is large (the real, computed signal).
    if abs(trained_task_acc - control_task_acc) > 0.1:
        fail.append(f"task-id should be ~equal trained vs control (input recovered): {trained_task_acc} vs {control_task_acc}")
    if not (trained_coord_r2 - control_coord_r2) >= 0.4:
        fail.append(f"routed-coord R^2 gap must be >= 0.4 (trained >> random): {trained_coord_r2} vs {control_coord_r2}")
    if argmax_index != SEQ_LABELS.index("task"):
        fail.append(f"CLS should attend most to the 'task' token, got index {argmax_index}")
    if fail:
        print("\nSTOP — regenerated probe does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------------------ pack
    data = {
        "provenance": {
            "source": "curriculum/phase3_advanced/ch3.8_frontier/probe.py",
            "generator": "site/scripts/vizdata/ch3.8_frontier.py",
            "seed": SEED,
            "device": "cpu",
            "config": f"default (seed 0, model_dim {config['model_dim']}, heads {config['heads']}, "
                      f"hidden {config['hidden']}, probe_examples 512, probe_ridge 1.0)",
            "stack": "torch 2.10.0 / numpy 2.4.6",
            "note": "Real numbers from probe.py's own loaded checkpoint + inspect "
                    "table + forward hook + four linear probes; matches meta.yaml "
                    "reference_run. The task-id probe hitting ~1.0 on BOTH the "
                    "trained checkpoint and the random-init control is the LESSON, "
                    "not a bug: a probe that recovers an INPUT proves nothing. The "
                    "routed-coordinate R^2 (0.90 trained vs 0.16 random) is the "
                    "signal — a value the model had to COMPUTE.",
            "seed_robustness": "trained coord R^2 seed0 0.90 / seed1 0.81 / seed2 0.77; "
                               "random control 0.16 / 0.23 / 0.20 — trained >> random on "
                               "every seed (gap >= 0.55). task-id 1.0 vs 1.0 on every seed. "
                               "Report the direction, not the third decimal.",
            "wallclock": "not yet measured",
        },
        # (1) THE CAVEAT + THE SIGNAL — the centerpiece. Two probes of the fused
        # layer, trained checkpoint vs random-init control.
        "probe": {
            "task_id": {
                "label": "task-id accuracy",
                "kind": "accuracy",
                "trained": round(trained_task_acc, 6),
                "control": round(control_task_acc, 6),
                "chance": round(chance_task_acc, 6),
                "role": "caveat",   # ~1.0 on BOTH — recovers an INPUT, proves nothing
            },
            "routed_coord": {
                "label": "routed-coordinate R²",
                "kind": "r2",
                "trained": round(trained_coord_r2, 6),
                "control": round(control_coord_r2, 6),
                "role": "signal",   # 0.90 vs 0.16 — a COMPUTED value; the real read
            },
        },
        # (2) THE FOUR MOVES + the checkpoint's insides (inspect table).
        "moves": ["load", "inspect", "hook", "probe"],
        "architecture": {
            "total_params": total_params,
            "config": config,
            "probed_module": "norm",     # the fused CLS layer the hook + probe read
            "modules": modules,
            "direct_params": direct,     # cls token + positional embeddings (not in named_children)
        },
        # (3) The interpretable read: what the fused CLS token attends to.
        "cls_attention": {
            "labels": SEQ_LABELS,
            "values": [round(float(v), 6) for v in cls_attention],
            "argmax_index": argmax_index,
            "argmax_label": SEQ_LABELS[argmax_index],
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml; task-id ~1.0 on BOTH (input recovered); "
          "routed-coord R^2 0.90 trained vs 0.16 random (the computed signal).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
