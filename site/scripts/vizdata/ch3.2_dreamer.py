#!/usr/bin/env python3
"""Regenerate the ch3.2 "imagined_vs_real" concept-toy vizdata from dreamer.py,
seed 0, cpu — the IMAGINATION GAP, told HONESTLY.

The site's ImaginationGapToy island renders REAL numbers dreamer.py itself measured
(seed 0, cpu — the byte-reproducible reference seed), never invented shapes. Its ONE
honest job is to carry the chapter's whole thesis into the browser: a policy trained
entirely INSIDE a learned world model looks like a champion IN IMAGINATION (its
imagined return climbs, its dreamed block parks near the target) and FAILS in reality
(real return floors, the real block barely moves, real task success is 0% on every
seed). Imagination is only as good as your world model — and this one (from 3.1) got
the block dynamics wrong. The toy MUST show that gap, so this generator STOPS unless
the regenerated numbers still exhibit it (imagined >> real, real success ~0).

Sibling of site/scripts/vizdata/ch3.1_world_models.py (EXEC-the-whole-script) and
site/scripts/vizdata/ch4.3_serl.py (gate every scalar against meta.yaml's measured
reference_run, STOP-on-drift).

Why we EXEC the whole dreamer.py (and shim rerun to capture the curves)
----------------------------------------------------------------------
dreamer.py is a loose script (no __main__ guard, no eval function): the world-model
train loop, the actor-in-imagination loop, AND the dual imagined-vs-real eval all run
at MODULE level, top to bottom. Importing it would run everything anyway — so we exec
the whole file in a throwaway namespace with argv pinned to the deterministic CPU
config (seed 0), then read the REAL objects it leaves behind: the eval scalars
`imag_return` / `real_return` / `gap` / `success_rate` and the `imag_final_pos` /
`real_final_pos` lists. No stdout parsing; the numbers ARE dreamer.py's own tensors.

The per-iteration CURVES (the rising imagined-return curve and the falling world-model
losses) live ONLY in the values dreamer.py logs to rerun — it keeps no python arrays
of them (LOC-capped; we do NOT modify the artifact). So we inject a tiny rerun-capture
shim into sys.modules BEFORE exec: dreamer.py's own `import rerun as rr` then binds our
shim, and every `rr.log(entity, rr.Scalars([v]))` is recorded in call order. Those are
dreamer.py's OWN logged scalars, not a re-derivation.

CPU + seed 0 is byte-reproducible (root CLAUDE.md invariant 2; dreamer.py's docstring
says so), so seed 0 lands inside meta.yaml's seed-0-2 bands every run. We gate against
those BANDS (not a single magic number) and against the mandatory gap + 0% success.

    Run:  .venv/bin/python site/scripts/vizdata/ch3.2_dreamer.py
    Out:  curriculum/phase3_advanced/ch3.2_dreamer/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CH = REPO / "curriculum" / "phase3_advanced" / "ch3.2_dreamer"
DREAMER_PY = CH / "dreamer.py"
OUT_JSON = CH / "demo" / "vizdata.json"

SEED = 0

# meta.yaml reference_run seed-sweep bands (seeds 0-2, cpu, default config; measured
# 2026-07-07). seed 0 lands inside every band; we gate against the BANDS, not exact
# magic numbers, because these are sweep bands and seed 0 is one draw inside them.
# (bands are the meta ranges, nudged out by rounding slack — meta rounds to 3 dp.)
BAND_IMAG_RETURN = (-0.230, -0.130)   # meta imagined -0.144..-0.220 per-step (rosier than real EVERY seed)
BAND_REAL_RETURN = (-0.400, -0.375)   # meta real -0.383..-0.392 per-step (the true PushT floor)
BAND_GAP = (0.150, 0.260)             # meta gap +0.172..+0.240 (imagined - real; imagination rosier)
BAND_IMAG_TEE = (0.000, 0.030)        # meta imagined final tee-dist 0.005..0.011 m (dream "parks" the block)
BAND_REAL_TEE = (0.140, 0.180)        # meta real final tee-dist 0.156..0.160 m (real block barely moved)
SUCCESS_MAX = 0.001                   # meta real success 0.00 on EVERY seed — the policy never solves it

# meta.yaml reference_run bands, verbatim, for the toy's "across seeds 0-2" honesty line.
SEED_BAND = {
    "imagined_return": [-0.220, -0.144],
    "real_return": [-0.392, -0.383],
    "gap": [0.172, 0.240],
    "imagined_final_tee_dist": [0.005, 0.011],
    "real_final_tee_dist": [0.156, 0.160],
    "real_success_rate": [0.0, 0.0],
}

TARGET_POINTS = 60   # subsample each curve to ~this many points (keeps the JSON small)


# --------------------------------------------------------------- rerun capture shim
class _Scalars:
    """Stand-in for rr.Scalars([v]) — just carries the logged value(s)."""

    def __init__(self, vals):
        self.vals = list(vals) if isinstance(vals, (list, tuple)) else [vals]


def make_rerun_shim():
    """A minimal module that quacks like `rerun` for exactly the calls dreamer.py
    makes (init/save/set_time/Scalars/log), recording every logged scalar in call
    order per entity. Returns (module, logs) where logs[entity] = [float, ...]."""
    shim = types.ModuleType("rerun")
    logs: dict[str, list[float]] = {}

    def _init(*_a, **_k):
        pass

    def _save(*_a, **_k):
        pass

    def _set_time(*_a, **_k):
        pass

    def _scalars(vals):
        return _Scalars(vals)

    def _log(entity, scalars, **_k):
        v = scalars.vals[0] if isinstance(scalars, _Scalars) else float(scalars)
        logs.setdefault(entity, []).append(float(v))

    shim.init = _init
    shim.save = _save
    shim.set_time = _set_time
    shim.Scalars = _scalars
    shim.log = _log
    return shim, logs


def exec_dreamer() -> tuple[dict, dict]:
    """Exec the whole dreamer.py (seed 0, cpu, rerun-shimmed) in an isolated namespace
    and return (its populated globals, the captured rerun logs)."""
    shim, logs = make_rerun_shim()
    src = DREAMER_PY.read_text()
    scratch = Path(tempfile.mkdtemp(prefix="ch3.2-viz-"))
    old_argv = sys.argv
    old_rerun = sys.modules.get("rerun")
    # argparse runs DURING exec — pin the deterministic, byte-reproducible config.
    # We leave --rerun ON (its default) so the shim captures the per-iter curves.
    sys.argv = [str(DREAMER_PY), "--seed", str(SEED), "--device", "cpu", "--out", str(scratch)]
    sys.modules["rerun"] = shim   # dreamer.py's `import rerun as rr` binds our shim
    ns: dict = {"__file__": str(DREAMER_PY), "__name__": "dreamer_toy_vizgen"}
    try:
        exec(compile(src, str(DREAMER_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
        if old_rerun is not None:
            sys.modules["rerun"] = old_rerun
        else:
            sys.modules.pop("rerun", None)
    return ns, logs


def subsample(xs: list[float], n: int) -> list[int]:
    """Indices that keep the first, the last, and ~n evenly-spaced points between —
    so a long training curve renders as a clean polyline without shipping every step."""
    if len(xs) <= n:
        return list(range(len(xs)))
    idx = sorted({round(i * (len(xs) - 1) / (n - 1)) for i in range(n)})
    return idx


def main() -> int:
    ns, logs = exec_dreamer()

    # ---------------------------------------------- the REAL objects dreamer.py left
    imag_return = float(ns["imag_return"])          # eval: what the actor BELIEVES it earns (dream)
    real_return = float(ns["real_return"])          # eval: what it ACTUALLY earns (true PushT)
    gap = float(ns["gap"])                           # imagined - real (>0 => imagination rosier)
    success_rate = float(ns["success_rate"])         # real task success (the honest 0.00)
    imag_tee = float(np.mean(ns["imag_final_pos"]))  # dreamed final tee-dist (block "parks")
    real_tee = float(np.mean(ns["real_final_pos"]))  # real final tee-dist (block barely moved)
    args = ns["args"]

    # -------------------------------------------------- the REAL curves it logged
    recon = list(logs.get("wm/recon", []))                 # world-model reconstruction (easy half, falls)
    dyn = list(logs.get("wm/dyn", []))                     # prior matching the posterior (the dynamics loss)
    imag_curve = list(logs.get("imag/reward_per_step", []))  # RISES as the actor learns to game the dream

    if not (recon and dyn and imag_curve):
        print("STOP — rerun capture missing curves (recon/dyn/imag). Did the shim bind?")
        return 1

    # ------------------------------------------------------------------ console echo
    print(f"regenerated dreamer.py [seed {SEED}, cpu] vs meta.yaml reference_run bands:")
    print(f"  IMAGINED return/step (dream) : {imag_return:+.4f}   (meta band -0.144..-0.220)")
    print(f"  REAL     return/step (sim)   : {real_return:+.4f}   (meta band -0.383..-0.392)")
    print(f"  gap (imagined - real)        : {gap:+.4f}   (meta band +0.172..+0.240)")
    print(f"  imagined final tee-dist      : {imag_tee:.4f} m (meta 0.005..0.011 — dream parks the block)")
    print(f"  real     final tee-dist      : {real_tee:.4f} m (meta 0.156..0.160 — block barely moved)")
    print(f"  real task success rate       : {success_rate:.3f}   (meta 0.00 EVERY seed)")
    print(f"  imagined-return curve        : {imag_curve[0]:+.4f} -> {imag_curve[-1]:+.4f} "
          f"over {len(imag_curve)} iters ({'RISES' if imag_curve[-1] > imag_curve[0] else 'does NOT rise'})")

    # ------------------------------------------------------------------ honesty gate
    fail: list[str] = []
    if not BAND_IMAG_RETURN[0] <= imag_return <= BAND_IMAG_RETURN[1]:
        fail.append(f"imagined_return {imag_return} outside band {BAND_IMAG_RETURN}")
    if not BAND_REAL_RETURN[0] <= real_return <= BAND_REAL_RETURN[1]:
        fail.append(f"real_return {real_return} outside band {BAND_REAL_RETURN}")
    if not BAND_GAP[0] <= gap <= BAND_GAP[1]:
        fail.append(f"gap {gap} outside band {BAND_GAP}")
    if not BAND_IMAG_TEE[0] <= imag_tee <= BAND_IMAG_TEE[1]:
        fail.append(f"imagined final tee-dist {imag_tee} outside band {BAND_IMAG_TEE}")
    if not BAND_REAL_TEE[0] <= real_tee <= BAND_REAL_TEE[1]:
        fail.append(f"real final tee-dist {real_tee} outside band {BAND_REAL_TEE}")
    # THE MANDATORY STORY — this is why the toy exists:
    #  * imagination is rosier than reality (a large positive gap)
    #  * the imagined-return curve RISES (the policy DID train — H1)
    #  * the real block does NOT park (real tee-dist >> imagined tee-dist)
    #  * real task success is ~0 (the policy NEVER solves the real task)
    if not gap > 0.10:
        fail.append(f"imagination gap {gap} not large/positive — the whole toy is the gap")
    if not imag_return > real_return:
        fail.append(f"imagined {imag_return} !> real {real_return} — imagination must look rosier")
    if not imag_curve[-1] > imag_curve[0]:
        fail.append(f"imagined-return curve did not rise: {imag_curve[0]} -> {imag_curve[-1]}")
    if not real_tee > imag_tee + 0.10:
        fail.append(f"real block moved as much as the dream (real {real_tee} vs imag {imag_tee}) — "
                    "the delusion is the point")
    if not success_rate <= SUCCESS_MAX:
        fail.append(f"real success {success_rate} > {SUCCESS_MAX} — the policy is NOT supposed to solve it")
    if fail:
        print("\nSTOP — regenerated dreamer.py does NOT match meta.yaml / the honest gap:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------- subsample curves
    wm_idx = subsample(recon, TARGET_POINTS)
    im_idx = subsample(imag_curve, TARGET_POINTS)
    wm_steps = [int(i) for i in wm_idx]
    wm_recon = [round(float(recon[i]), 6) for i in wm_idx]
    wm_dyn = [round(float(dyn[i]), 6) for i in wm_idx]
    im_iters = [int(i) for i in im_idx]
    im_reward = [round(float(imag_curve[i]), 6) for i in im_idx]

    # ---------------------------------------------------------------------- pack it
    data = {
        "provenance": {
            "source": "curriculum/phase3_advanced/ch3.2_dreamer/dreamer.py",
            "generator": "site/scripts/vizdata/ch3.2_dreamer.py",
            "seed": SEED,
            "device": "cpu",
            "config": f"default (latent_dim {args.latent_dim}, hidden_dim {args.hidden_dim}, "
                      f"seq_len {args.seq_len}, episodes {args.episodes}, wm_epochs {args.wm_epochs}, "
                      f"imag_horizon {args.imag_horizon}, imag_iters {args.imag_iters}, "
                      f"actor_lr {args.actor_lr}, eval_horizon {args.eval_horizon}, "
                      f"eval_episodes {args.eval_episodes}); byte-reproducible on CPU",
            "stack": "torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6",
            "note": "The two eval bars (imagined vs real return) and the two final tee-distances "
                    "are dreamer.py's OWN eval scalars; the rising imagined-return curve and the "
                    "falling world-model losses are dreamer.py's OWN per-iteration rerun logs, "
                    "captured in call order (no re-derivation). seed 0 lands inside meta.yaml's "
                    "seed-0-2 bands. THE HONEST LESSON — THE IMAGINATION GAP: the policy trained "
                    "entirely inside the world model looks like a champion IN IMAGINATION (imagined "
                    "return climbs, dreamed block parks ~0.01 m from target) and FAILS in reality "
                    "(real return floors, real block barely moves ~0.16 m, real task success 0% on "
                    "EVERY seed). Imagination is only as good as your world model, and this one "
                    "(from 3.1) got the block dynamics wrong. This is NOT 'Dreamer solves PushT' — "
                    "it emphatically does not at free-tier scale.",
        },
        "config": {
            "imag_iters": int(args.imag_iters),
            "imag_horizon": int(args.imag_horizon),
            "wm_epochs": int(args.wm_epochs),
            "eval_episodes": int(args.eval_episodes),
            "eval_horizon": int(args.eval_horizon),
            "latent_dim": int(args.latent_dim),
            "hidden_dim": int(args.hidden_dim),
        },
        # Panel 1 — THE HEADLINE: the imagination gap (the SAME policy, two worlds).
        "gap": {
            "imagined_return": round(imag_return, 6),   # what the actor BELIEVES it earns (dream)
            "real_return": round(real_return, 6),        # what it ACTUALLY earns (true PushT)
            "gap": round(gap, 6),                        # imagined - real (>0 => imagination rosier)
            "imagined_final_tee_dist": round(imag_tee, 6),  # dream "parks" the block near target
            "real_final_tee_dist": round(real_tee, 6),       # reality: block barely moved from ~0.17 m spawn
            "real_success_rate": round(success_rate, 6),     # the imagination-trained policy NEVER solves it
        },
        # Panel 2 — proof the policy DID train: imagined return climbs as the actor
        # learns to drive the (hallucinated) block toward the target. The flat real
        # line is where the SAME policy actually lands — the curve climbs AWAY from it.
        "imagined_return_curve": {
            "iters": im_iters,
            "reward_per_step": im_reward,
            "start": round(float(imag_curve[0]), 6),
            "final": round(float(imag_curve[-1]), 6),
            "real_return": round(real_return, 6),   # the flat reference: reality does not climb
        },
        # Panel 3 — Step 1 (unchanged from 3.1): the world model the actor dreams inside.
        # Reconstruction (easy half) falls fast; the dynamics loss is the prior learning
        # to match the posterior it cannot see.
        "wm_losses": {
            "steps": wm_steps,
            "recon": wm_recon,
            "dyn": wm_dyn,
        },
        # meta.yaml reference_run bands (seeds 0-2) — for the toy's honesty line.
        "seed_band": SEED_BAND,
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml bands; THE GAP holds: imagination looks rosy "
          f"({imag_return:+.3f}/step, dream parks at {imag_tee:.3f} m) while reality floors "
          f"({real_return:+.3f}/step, {real_tee:.3f} m) at {success_rate:.0%} real success.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
