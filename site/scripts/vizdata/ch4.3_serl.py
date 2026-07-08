#!/usr/bin/env python3
"""Regenerate the ch4.3 HIL-SERL concept-toy vizdata — the HONEST sample-efficiency
story: the corrections-as-prior put HIL-SERL below the reach threshold at ZERO
online samples, while from-scratch SAC needs ~10,000 online samples to reach the
same bar. Sibling of site/scripts/vizdata/ch2.2_sac.py (a sample-efficiency curve
toy) and ch4_offline_primer.py / ch4.2_corrections.py (the measured-reference gate).

Why a CURVE, and why the sample AXIS is the hero
------------------------------------------------
SERL's lesson is SAMPLE EFFICIENCY, not a stronger final policy. On this small
dense-reward task both arms top out near the same short-horizon SAC ceiling
(serl.py meta: prior ~0.06 m, scratch-best ~0.086 m — a modest final-height gap);
online fine-tuning HOLDS the prior rather than beating it. A live side-by-side
reach would show both arms reaching similarly and MISREPRESENT the point. The
honest visualization is the LEARNING CURVE: eval reach-distance vs online env
samples, with the threshold marked and the horizontal "0 vs ~10,000 samples" gap
as the hero. That is exactly what serl.py measures (samples_to_threshold).

Why we PARSE the real run (not hand-copy the curve)
---------------------------------------------------
The per-step curve (hil_curve / scratch_curve — the eval mean final-distance at
each online-sample checkpoint) lives ONLY in serl.py's run output
(outputs/ch4.3-serl/metrics.json, seed 0, cpu — the reference-run seed). meta.yaml
records the SCALARS (samples-to-threshold per seed, prior/scratch/hil distances,
success rates, the diff CI) but not the curve. So this generator READS the real
metrics.json for the curve and cross-checks EVERY scalar the toy shows against
meta.yaml's reference_run (the seed-0 slice) — the same STOP-on-drift honesty gate
the ch4.2 / ch4-primer generators use. It never invents a number, and it never
renders a value meta does not back.

The honesty gate (STOP-on-drift + honest-framing asserts)
---------------------------------------------------------
  * metrics.json's seed-0 scalars must match meta.yaml reference_run's seed-0
    slice (samples-to-threshold, threshold, prior/scratch/hil distances, success
    rates, the diff CI) within recording tolerance;
  * the SAMPLE-AXIS signal: HIL clears the threshold at 0 online samples (its
    curve starts below the bar — the prior put it there) and from-scratch's first
    sub-threshold checkpoint is at metrics' scratch_steps_to_threshold, and
    0 << scratch_sts (the win is large);
  * the HONEST-FRAMING guard: the final-height gap (scratch-best - hil-best) is
    MODEST and both sit below the threshold — the win is NOT a stronger final
    policy, it is the sample axis. If that inverts (a big final-height gap), we
    STOP, because the toy would then be selling SERL as a better policy.
If any drift, we STOP.

    Run:  .venv/bin/python site/scripts/vizdata/ch4.3_serl.py
    Out:  curriculum/phase4_capstone/ch4.3_serl/demo/vizdata.json
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CH = REPO / "curriculum" / "phase4_capstone" / "ch4.3_serl"
META_YAML = CH / "meta.yaml"
OUT_JSON = CH / "demo" / "vizdata.json"
# serl.py's real run output (seed 0, cpu — the reference-run seed); the ONLY place
# the per-step eval curve lives. outputs/ is gitignored; the COMMITTED artifact is
# the vizdata.json this emits. Re-run serl.py --seed 0 to refresh the source.
METRICS_JSON = REPO / "outputs" / "ch4.3-serl" / "metrics.json"

# meta.yaml records per-seed scalars rounded to 3 dp (e.g. prior 0.061 vs metrics'
# 0.06096); the run output carries full precision. Guard rounding only.
TOL = 0.005


def approx(a: float, b: float, tol: float = TOL) -> bool:
    return abs(a - b) <= tol


def main() -> int:
    if not METRICS_JSON.exists():
        print(f"STOP — {METRICS_JSON} not found. Run serl.py --seed 0 to produce the "
              "reference-run metrics before regenerating this toy's vizdata.")
        return 1

    m = json.loads(METRICS_JSON.read_text())
    meta = yaml.safe_load(META_YAML.read_text())
    ref = meta["reference_run"]

    seed = int(m["seed"])
    threshold = float(m["threshold"])
    hil_curve = [[int(s), round(float(d), 5)] for s, d in m["hil_curve"]]
    scratch_curve = [[int(s), round(float(d), 5)] for s, d in m["scratch_curve"]]
    hil_sts = int(m["hil_steps_to_threshold"])
    scratch_sts = int(m["scratch_steps_to_threshold"])
    prior_dist = float(m["prior_eval_dist"])
    hil_best = float(m["hil_best_dist"])
    scratch_best = float(m["scratch_best_dist"])
    hil_success = float(m["hil_success_rate"])
    scratch_success = float(m["scratch_success_rate"])
    prior_success = float(m["prior_success_rate"])
    diff_ci = [round(float(m["diff_ci_lo"]), 5), round(float(m["diff_ci_hi"]), 5)]
    gap_significant = bool(m["gap_significant"])
    n_pooled = int(m["n_pooled"])
    corr_episodes = int(m["corr_episodes"])

    # per-seed scalars (all three seeds) from meta — the seed-robust ledger.
    sts_hil = [int(x) for x in ref["samples_to_threshold_hil"]]
    sts_scratch = [int(x) for x in ref["samples_to_threshold_scratch"]]
    prior_dist_seeds = [float(x) for x in ref["prior_eval_dist_per_seed"]]
    prior_succ_seeds = [float(x) for x in ref["prior_success_per_seed"]]
    scratch_best_seeds = [float(x) for x in ref["scratch_best_dist_per_seed"]]
    scratch_succ_seeds = [float(x) for x in ref["scratch_success_per_seed"]]
    hil_best_seeds = [float(x) for x in ref["hil_best_dist_per_seed"]]
    hil_succ_seeds = [float(x) for x in ref["hil_success_per_seed"]]
    random_dist = float(ref["random_baseline_dist"])

    # ===================================================== honesty gate vs meta
    fail: list[str] = []

    # (a) seed-0 scalars in metrics.json match meta's seed-0 slice.
    if hil_sts != sts_hil[seed]:
        fail.append(f"hil_sts {hil_sts} != meta seed{seed} {sts_hil[seed]}")
    if scratch_sts != sts_scratch[seed]:
        fail.append(f"scratch_sts {scratch_sts} != meta seed{seed} {sts_scratch[seed]}")
    if not approx(prior_dist, prior_dist_seeds[seed]):
        fail.append(f"prior_dist {prior_dist} != meta {prior_dist_seeds[seed]}")
    if not approx(scratch_best, scratch_best_seeds[seed]):
        fail.append(f"scratch_best {scratch_best} != meta {scratch_best_seeds[seed]}")
    if not approx(hil_best, hil_best_seeds[seed]):
        fail.append(f"hil_best {hil_best} != meta {hil_best_seeds[seed]}")
    if not approx(hil_success, hil_succ_seeds[seed]):
        fail.append(f"hil_success {hil_success} != meta {hil_succ_seeds[seed]}")
    if not approx(scratch_success, scratch_succ_seeds[seed]):
        fail.append(f"scratch_success {scratch_success} != meta {scratch_succ_seeds[seed]}")
    if not approx(prior_success, prior_succ_seeds[seed]):
        fail.append(f"prior_success {prior_success} != meta {prior_succ_seeds[seed]}")
    # meta records seed0 diff CI as "+0.05..+0.46"; guard to 2 dp.
    if not (approx(diff_ci[0], 0.05, 0.01) and approx(diff_ci[1], 0.46, 0.01)):
        fail.append(f"seed0 diff CI {diff_ci} != meta 'seed0 +0.05..+0.46'")
    if not gap_significant:
        fail.append("seed0 gap not significant, but meta records it SIG")

    # (b) the SAMPLE-AXIS signal — the hero. HIL clears at 0 online samples
    #     (its curve STARTS below the bar — the prior put it there); scratch's
    #     first sub-threshold checkpoint is scratch_sts; and 0 << scratch_sts.
    if not (hil_curve[0][0] == 0 and hil_curve[0][1] < threshold):
        fail.append(f"HIL curve does not start below threshold at step 0: {hil_curve[0]}")
    if hil_sts != 0:
        fail.append(f"hil_sts {hil_sts} != 0 (the prior should clear at zero online samples)")
    first_cross = next((int(s) for s, d in scratch_curve if d < threshold), None)
    if first_cross != scratch_sts:
        fail.append(f"scratch first sub-threshold checkpoint {first_cross} != scratch_sts {scratch_sts}")
    if not (scratch_sts - hil_sts >= 8000):
        fail.append(f"sample-axis gap {scratch_sts - hil_sts} too small — the hero signal is weak")
    if not all(x == 0 for x in sts_hil):
        fail.append(f"HIL samples-to-threshold not 0 on every seed: {sts_hil}")
    if not all(x >= 8000 for x in sts_scratch):
        fail.append(f"scratch samples-to-threshold not seed-robustly large: {sts_scratch}")
    for s, d in zip(range(len(prior_dist_seeds)), prior_dist_seeds):
        if not (d < threshold):
            fail.append(f"seed{s} prior dist {d} does not clear threshold {threshold}")

    # (c) HONEST-FRAMING guard — the win is the SAMPLE axis, NOT a stronger final
    #     policy. The final-height gap must be MODEST and both below threshold.
    final_gap = scratch_best - hil_best  # >0 means HIL's final is nearer, but small
    if not (hil_best < threshold and scratch_best < threshold):
        fail.append(f"final heights not both below threshold (hil {hil_best}, scratch {scratch_best})")
    if abs(final_gap) > threshold:
        fail.append(f"final-height gap {final_gap:.3f} too large — toy would sell SERL as a better "
                    "policy; the honest win is the SAMPLE axis")

    print("ch4.3 HIL-SERL — sample-efficiency curve vs meta.yaml reference_run:")
    print(f"  seed{seed}: HIL clears threshold at {hil_sts} online samples · "
          f"scratch at {scratch_sts} · gap {scratch_sts - hil_sts:,}")
    print(f"  prior {prior_dist:.4f} m · hil-best {hil_best:.4f} m · scratch-best {scratch_best:.4f} m "
          f"(threshold {threshold} m, random {random_dist} m)")
    print(f"  seed-robust samples-to-threshold: HIL {sts_hil} vs scratch {sts_scratch}")
    print(f"  seed0 diff CI (HIL-scratch success): {diff_ci[0]:+.3f}..{diff_ci[1]:+.3f} "
          f"({'SIG' if gap_significant else 'ns'})")

    if fail:
        print("\nSTOP — regenerated ch4.3 toy data does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ============================================================ pack + write
    data = {
        "provenance": {
            "source": "curriculum/phase4_capstone/ch4.3_serl/serl.py run output "
                      "(outputs/ch4.3-serl/metrics.json, seed 0, cpu — the reference-run seed)",
            "generator": "site/scripts/vizdata/ch4.3_serl.py",
            "measured": "2026-07-07, cpu, torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6",
            "config": "DEFAULT — corr_episodes 60, prior_steps 8000, hil_steps 6000, "
                      "scratch_steps 12000, threshold 0.10 m, eval n_seeds 3 x eval_episodes 10 "
                      "= 30 pooled rollouts; the CURVE is seed 0, the ledger is seeds 0-2.",
            "note": "The per-step eval curve (hil_curve / scratch_curve) is PARSED from serl.py's "
                    "real run output; every scalar the toy shows is cross-checked against meta.yaml's "
                    "MEASURED reference_run (seed-0 slice) and the generator STOPS on drift. HONEST "
                    "FRAMING: the win is SAMPLE EFFICIENCY on the ONLINE-sample axis — the "
                    "corrections-as-prior put HIL-SERL below the 0.10 m threshold at ZERO online "
                    "samples, while from-scratch SAC needs ~10,000. It is NOT a stronger final "
                    "policy: both arms top out near the same short-horizon SAC ceiling (final-height "
                    "gap is modest), so online fine-tuning HOLDS the prior rather than beating it. On "
                    "harder tasks + the gated capstone suite (ch4.4) is where the online phase earns "
                    "its keep. Do NOT re-frame this as 'online RL solves what the prior can't.'",
        },
        "demo": "serl_sample_efficiency",
        "seed": seed,
        "seeds": list(range(len(sts_hil))),
        "threshold": threshold,
        "random_dist": random_dist,
        "n_pooled": n_pooled,
        "corr_episodes": corr_episodes,
        # THE CURVE (seed 0) — eval reach distance (m, lower is better) vs online env samples.
        # HIL starts at step 0 (the prior, already below threshold); scratch starts at its
        # first eval checkpoint and crawls down to the same line over ~10k online samples.
        "hil_curve": hil_curve,
        "scratch_curve": scratch_curve,
        "hil_sts": hil_sts,          # 0 — the prior clears at zero online samples
        "scratch_sts": scratch_sts,  # ~10,000 — the horizontal gap IS the sample efficiency
        # seed-0 endpoint scalars (the readout).
        "prior_dist": round(prior_dist, 5),
        "hil_best": round(hil_best, 5),
        "scratch_best": round(scratch_best, 5),
        "hil_success": round(hil_success, 4),
        "scratch_success": round(scratch_success, 4),
        "prior_success": round(prior_success, 4),
        "diff_ci": diff_ci,
        "gap_significant": gap_significant,
        # the seed-robust ledger (all three seeds) — the honest "holds on every seed".
        "per_seed": {
            "sts_hil": sts_hil,
            "sts_scratch": sts_scratch,
            "prior_dist": prior_dist_seeds,
            "prior_success": prior_succ_seeds,
            "scratch_best": scratch_best_seeds,
            "scratch_success": scratch_succ_seeds,
            "hil_best": hil_best_seeds,
            "hil_success": hil_succ_seeds,
            # meta's per-seed diff-CI verdicts (HIL - scratch success), verbatim.
            "diff_ci_note": ref["diff_ci_hil_minus_scratch"],
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml; HIL clears the threshold at 0 online samples on every seed, "
          f"scratch needs ~{sts_scratch[seed]:,}; the win is the SAMPLE axis, not a better final policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
