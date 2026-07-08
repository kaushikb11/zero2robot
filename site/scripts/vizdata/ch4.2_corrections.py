#!/usr/bin/env python3
"""Regenerate the ch4.2 DAgger concept-toy vizdata — the HONEST "BC covariate-shift
floor -> DAgger recovery" numbers, with ch1.6-style Wilson error bars and Newcombe
recovery diff CIs. Sibling of site/scripts/vizdata/ch2.7_dr.py and ch3.3_engine.py.

Why we READ the measured reference_run instead of running dagger.py --smoke
---------------------------------------------------------------------------
dagger.py trains a BC seed policy then runs 4 rounds of rollout -> expert-label ->
aggregate -> retrain, evaluating N=200 pooled episodes per round (see meta.yaml).
The chapter's whole LESSON is that measured, seeds-0-2 result: BC sits at the
covariate-shift floor (~0.06), the best DAgger round recovers to ~0.19-0.22 with a
diff CI that EXCLUDES 0, and the peak round VARIES by seed (3, 4, 4) — over-
iterating a reactive clone REGRESSES (dataset flooding). A --smoke run (bc_demos 6,
1 DAgger iter, 3 rollouts, 3 epochs, N=6) trains an essentially-untrained clone and
would NOT reproduce that measured story — it would invent a different, misleading
picture. So, exactly like ch2.7's DomainRandToy (and ch1.6's EvalBandsToy, which
pin their measured reference_run rather than re-running), this generator transcribes
dagger.py's MEASURED per-seed round_rates from meta.yaml and recomputes the Wilson /
diff-CI statistics the toy shows.

Faithful CIs (no re-implementation)
-----------------------------------
The error bars and recovery diff CIs are computed with dagger.py's OWN wilson_ci /
diff_ci — we exec ONLY the `# --- region: stats ---` block of dagger.py (self-
contained: it needs just `math`) in a throwaway namespace, so the site's bars are
bit-faithful to the chapter artifact. We do NOT modify dagger.py (it is LOC-capped)
and we never run its heavy training/eval path.

The honesty gate (STOP-on-drift vs meta)
----------------------------------------
The transcribed round_rates below are the SOURCE. We re-open meta.yaml and cross-
check every quantitative claim meta makes against numbers RECOMPUTED from them:
  * per-seed recovery diff CI (best round - BC) must equal meta's
    recovery_diff_ci_seed{0,1,2} AND exclude 0 (recovery significant every seed);
  * BC floor across seeds must equal meta's bc_rate_band [0.055, 0.065] and stay
    under exercise_checks.ex1.bc_rate_max (0.12);
  * best DAgger round across seeds must equal meta's best_rate_band [0.185, 0.215]
    and clear exercise_checks.ex1.best_rate_min (0.14);
  * the peak round is (3, 4, 4) — NOT always the last (non-monotonic); seed0's peak
    (round 3) beats its last round by >= ex2.peak_beats_last_min (0.05).
If any drift, we STOP — the toy never renders a number meta does not back.

    Run:  .venv/bin/python site/scripts/vizdata/ch4.2_corrections.py
    Out:  curriculum/phase4_capstone/ch4.2_corrections/demo/vizdata.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CH = REPO / "curriculum" / "phase4_capstone" / "ch4.2_corrections"
DAGGER_PY = CH / "dagger.py"
META_YAML = CH / "meta.yaml"
OUT_JSON = CH / "demo" / "vizdata.json"

# ---------------------------------------------------------------------------
# The MEASURED reference_run, transcribed verbatim from meta.yaml (PROVENANCE:
# 2026-07-07, cpu, torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6; DEFAULT config —
# bc_demos 50, r_max 0.13, dagger_iters 4, rollouts 40, hidden_dim 256,
# epochs 300, eval N = n_seeds 20 * eval_episodes 10 = 200; seeds 0-2). Each entry
# is the per-round pooled success rate (N=200): round 0 = BC, rounds 1..4 = DAgger.
# ---------------------------------------------------------------------------
N_POOLED = 200
SEEDS = [0, 1, 2]
ROUND_RATES: dict[int, list[float]] = {
    0: [0.065, 0.085, 0.155, 0.215, 0.085],  # peak round 3; regresses at round 4 (flooding)
    1: [0.055, 0.075, 0.110, 0.125, 0.185],  # peak round 4 (still climbing at 4)
    2: [0.055, 0.160, 0.105, 0.185, 0.210],  # peak round 4; non-monotonic (dip at round 2)
}
# Round labels: round 0 is the BC seed policy, 1..4 are DAgger correct-aggregate-retrain rounds.
ROUND_LABELS = ["BC", "DAgger 1", "DAgger 2", "DAgger 3", "DAgger 4"]

# The reactive-MLP ceiling this task tops out at (meta / dagger.py prose: ~0.25).
CEILING = 0.25

STATS_BEGIN = "# --- region: stats ---"
STATS_END = "# --- endregion ---"
# We recompute every CI with dagger.py's OWN diff_ci from the measured round_rates,
# so the displayed values are bit-faithful to the artifact: seed0 reproduces meta's
# recovery_diff_ci_seed0 to 1e-6. meta recorded seed1/seed2's diff CIs to coarser,
# hand-rounded precision (e.g. 0.070 / 0.190 / 0.093), so the cross-check tolerance
# accommodates that recording precision — but the HARD gate is that every recovery
# diff CI EXCLUDES 0 (the recovery-significant claim), which is asserted separately.
META_CI_TOL = 0.005


def load_dagger_stats() -> dict:
    """Exec ONLY dagger.py's self-contained `stats` region in a throwaway
    namespace, returning its wilson_ci + diff_ci (faithful to the chapter artifact,
    zero edits, no heavy training path)."""
    src = DAGGER_PY.read_text()
    b = src.index(STATS_BEGIN)
    e = src.index(STATS_END, b)
    block = src[b:e]
    ns: dict = {"math": math}  # the stats region uses math.sqrt; it imports at the setup region we skip
    exec(compile(block, str(DAGGER_PY), "exec"), ns)  # noqa: S102 — our own trusted source, math-only
    return ns


def k_of(rate: float) -> int:
    """Recover the integer success count k for a pooled rate at N=200 (round_rates
    are exact k/200 by construction; assert so a bad transcription STOPS here)."""
    k = round(rate * N_POOLED)
    assert abs(k / N_POOLED - rate) < 1e-9, f"rate {rate} is not an exact k/{N_POOLED}"
    return k


def main() -> int:
    stats = load_dagger_stats()
    wilson_ci = stats["wilson_ci"]
    diff_ci = stats["diff_ci"]

    # ---- per-seed: round rates, Wilson bars, best round, recovery diff CI --------
    per_seed: dict[str, dict] = {}
    bc_rates: list[float] = []
    best_rates: list[float] = []
    peak_rounds: list[int] = []
    recovery_diff: dict[int, list[float]] = {}
    for seed in SEEDS:
        rates = ROUND_RATES[seed]
        ks = [k_of(r) for r in rates]
        cis = [[round(lo, 6), round(hi, 6)] for lo, hi in (wilson_ci(k, N_POOLED) for k in ks)]
        # best round over ALL rounds incl. BC (Ross et al.: return the best, not the last)
        best_round = max(range(len(rates)), key=lambda i: rates[i])
        peak_rounds.append(best_round)
        kb, kbc = ks[best_round], ks[0]
        d_lo, d_hi = diff_ci(kb, N_POOLED, kbc, N_POOLED)  # best - BC (Newcombe)
        recovery_diff[seed] = [round(d_lo, 6), round(d_hi, 6)]
        bc_rates.append(rates[0])
        best_rates.append(rates[best_round])
        per_seed[str(seed)] = {
            "rates": rates,
            "k": ks,
            "ci": cis,
            "bc_rate": rates[0],
            "best_round": best_round,
            "best_rate": rates[best_round],
            "last_round": len(rates) - 1,
            "last_rate": rates[-1],
            "peak_beats_last": round(rates[best_round] - rates[-1], 6),
            "recovery_diff_ci": recovery_diff[seed],
            "significant": bool(d_lo > 0 or d_hi < 0),
        }

    bc_rate_band = [round(min(bc_rates), 6), round(max(bc_rates), 6)]
    best_rate_band = [round(min(best_rates), 6), round(max(best_rates), 6)]

    # ---- seed 0 Bonferroni multiplicity CIs (transcribed from meta HONESTY): the
    #      recovery survives correction over the rounds — a NON-selected round (2)
    #      already clears BC on its own. Widened z (multiplicity), so carried as
    #      meta-provided display constants; we assert the UNcorrected round2/round3
    #      diff CIs also exclude 0 (they are tighter, so if the Bonferroni ones
    #      exclude 0 these must too — a consistency check, not a re-derivation).
    k_bc0 = k_of(ROUND_RATES[0][0])
    r2_lo, r2_hi = diff_ci(k_of(ROUND_RATES[0][2]), N_POOLED, k_bc0, N_POOLED)
    r3_lo, r3_hi = diff_ci(k_of(ROUND_RATES[0][3]), N_POOLED, k_bc0, N_POOLED)
    bonferroni_seed0 = {"round3": [0.064, 0.237], "round2": [0.011, 0.171]}

    # ======================================================= honesty gate vs meta
    meta = yaml.safe_load(META_YAML.read_text())
    ref = meta["reference_run"]
    checks = meta["exercise_checks"]
    ex1, ex2 = checks["ex1"], checks["ex2"]
    fail: list[str] = []

    # (a) transcription integrity: our round_rates equal meta's seedN_round_rates.
    for seed in SEEDS:
        meta_rr = ref[f"seed{seed}_round_rates"]
        if [round(x, 6) for x in meta_rr] != [round(x, 6) for x in ROUND_RATES[seed]]:
            fail.append(f"seed{seed} round_rates {ROUND_RATES[seed]} != meta {meta_rr}")

    # (b) per-seed recovery diff CI matches meta AND excludes 0.
    for seed in SEEDS:
        meta_ci = ref[f"recovery_diff_ci_seed{seed}"]
        ours = recovery_diff[seed]
        if abs(ours[0] - meta_ci[0]) > META_CI_TOL or abs(ours[1] - meta_ci[1]) > META_CI_TOL:
            fail.append(f"seed{seed} recovery diff CI {ours} != meta {meta_ci} (beyond recording tol)")
        if not (ours[0] > 0 or ours[1] < 0):
            fail.append(f"seed{seed} recovery diff CI {ours} does NOT exclude 0")

    # (c) BC floor band + best-round band match meta and clear the exercise checks.
    if bc_rate_band != [round(x, 6) for x in ref["bc_rate_band"]]:
        fail.append(f"bc_rate_band {bc_rate_band} != meta {ref['bc_rate_band']}")
    if best_rate_band != [round(x, 6) for x in ref["best_rate_band"]]:
        fail.append(f"best_rate_band {best_rate_band} != meta {ref['best_rate_band']}")
    if bc_rate_band[1] > float(ex1["bc_rate_max"]) + 1e-9:
        fail.append(f"BC floor max {bc_rate_band[1]} exceeds ex1.bc_rate_max {ex1['bc_rate_max']}")
    if best_rate_band[0] < float(ex1["best_rate_min"]) - 1e-9:
        fail.append(f"best-round min {best_rate_band[0]} below ex1.best_rate_min {ex1['best_rate_min']}")

    # (d) non-monotonic: peak rounds are (3,4,4), and seed0's peak beats its last.
    if peak_rounds != [3, 4, 4]:
        fail.append(f"peak rounds {peak_rounds} != (3,4,4) — the 'peak VARIES by seed' claim broke")
    s0_peak_beats_last = per_seed["0"]["peak_beats_last"]
    if s0_peak_beats_last < float(ex2["peak_beats_last_min"]) - 1e-9:
        fail.append(f"seed0 peak-beats-last {s0_peak_beats_last} below ex2.peak_beats_last_min "
                    f"{ex2['peak_beats_last_min']} — the 'select the best round, not the last' claim broke")

    # (e) Bonferroni consistency: the (tighter) uncorrected round2/round3 diff CIs
    #     must exclude 0 if the meta Bonferroni ones do (the non-selected-round point).
    if not (r2_lo > 0):
        fail.append(f"seed0 uncorrected round2 diff CI [{r2_lo:.4f},{r2_hi:.4f}] does not exclude 0")
    if not (r3_lo > 0):
        fail.append(f"seed0 uncorrected round3 diff CI [{r3_lo:.4f},{r3_hi:.4f}] does not exclude 0")

    print("regenerated ch4.2 DAgger [measured reference_run, seeds 0-2] vs meta.yaml:")
    for seed in SEEDS:
        ps = per_seed[str(seed)]
        print(f"  seed{seed}: rates {ps['rates']}  best {ROUND_LABELS[ps['best_round']]} "
              f"({ps['best_rate']:.3f})  recovery diff CI [{ps['recovery_diff_ci'][0]:+.3f},"
              f"{ps['recovery_diff_ci'][1]:+.3f}]  (meta {ref[f'recovery_diff_ci_seed{seed}']})")
    print(f"  BC floor band {bc_rate_band}  (meta {ref['bc_rate_band']})")
    print(f"  best-round band {best_rate_band}  (meta {ref['best_rate_band']})")
    print(f"  peak rounds {peak_rounds}  (meta note: 3,4,4)")

    if fail:
        print("\nSTOP — regenerated DAgger toy data does NOT match meta.yaml:")
        for f in fail:
            print("  x " + f)
        return 1

    # ================================================================ pack + write
    data = {
        "provenance": {
            "source": "curriculum/phase4_capstone/ch4.2_corrections/dagger.py "
                      "(MEASURED reference_run in meta.yaml — not a live re-run)",
            "generator": "site/scripts/vizdata/ch4.2_corrections.py",
            "measured": "2026-07-07, cpu, torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6",
            "config": "DEFAULT — bc_demos 50, r_max 0.13, dagger_iters 4, rollouts 40, "
                      "hidden_dim 256, epochs 300, eval N = 20 suites * 10 episodes = 200; seeds 0-2",
            "metric": "pooled success rate over N=200 held-out full-annulus episodes per round; "
                      "Wilson score interval + Newcombe recovery diff CI computed with dagger.py's "
                      "OWN wilson_ci / diff_ci (exec of its `stats` region — zero edits).",
            "note": "Transcribed from meta.yaml reference_run; the per-seed recovery diff CIs, the BC "
                    "floor band, the best-round band and the (3,4,4) peak rounds are cross-checked "
                    "against meta and the script STOPS on any drift. A --smoke run (6 demos, 1 iter) "
                    "would NOT reproduce this measured recovery.",
        },
        "n_pooled": N_POOLED,
        "seeds": SEEDS,
        "round_labels": ROUND_LABELS,
        "default_seed": 0,
        "ceiling": CEILING,
        "per_seed": per_seed,
        "bc_rate_band": bc_rate_band,
        "best_rate_band": best_rate_band,
        "peak_rounds": peak_rounds,
        # The winner's-curse HONESTY caveat + the Bonferroni-surviving evidence (meta).
        "winners_curse": {
            "caveat": "The best round is selected on the SAME held-out eval used for the diff CI (no "
                      "separate validation split), so the reported significance carries a mild "
                      "selection bias.",
            "why_not_artifact": "It is NOT an artifact: a NON-selected round (round 2) already clears "
                                "BC on its own, and the recovery survives Bonferroni multiplicity "
                                "correction over the rounds.",
            "bonferroni_seed0": bonferroni_seed0,
        },
        # The mechanism strip (panel 3) — the DAgger loop, plain-language.
        "mechanism": [
            "BC only ever saw the states the EXPERT visited (near starts).",
            "Deployed on the full task, the clone DRIFTS into states no demo covered.",
            "DAgger rolls out the CURRENT policy and labels the states IT visits.",
            "Aggregate those (visited-state, expert-action) pairs into the dataset.",
            "Retrain, and iterate — the clone learns to recover from its own mistakes.",
        ],
        "ceiling_note": "The clone is a reactive MLP (obs -> action, no memory), so even recovered it "
                        "tops out around 25% on this task — DAgger fixes the covariate shift, it does "
                        "not make a reactive policy omniscient.",
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml; BC floor recovers to best round, diff CI excludes 0 every seed; "
          "peak round VARIES (3,4,4), the last round is not the best.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
