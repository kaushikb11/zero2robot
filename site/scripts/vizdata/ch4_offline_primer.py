#!/usr/bin/env python3
"""Regenerate the ch4 offline-RL-primer concept-toy vizdata from the MEASURED
reference_run, reusing offline.py's OWN error-bar code (ch1.6 idiom), seed-robust.

THE OFFLINE-RL HEADLINE, made into a browser toy: on ONE fixed, mixed-quality
dataset, behavior cloning (BC) clones the average and stays near the floor, while
AWAC uses the REWARD to extract an above-average policy and beats it — significant
(the difference CI excludes 0) on every seed, but honestly MODEST (AWAC reaches
~0.09-0.11 m, still far from the scripted expert's ~0.0001 m). And the Break-It:
naive maximize-Q with no data constraint OVERESTIMATES out-of-distribution actions
and its |Q| inflates ~7x on NARROW (expert-only) data, while AWAC's stays bounded.

Why we TRANSCRIBE the reference_run (not re-run offline.py end-to-end)
--------------------------------------------------------------------
offline.py's DEFAULT config trains BC + a twin-Q AWAC critic for 8000 steps and
evaluates 100 rollouts, per seed — and the honest picture needs THREE seeds plus
the --naive --expert_frac 1.0 narrow-data Break-It on three more. That is minutes
of CPU per arm; re-running all of it here would be slow and buys nothing, because
seed-0 CPU is bitwise reproducible and the numbers are already MEASURED and
recorded in meta.yaml's reference_run (provenance there: cpu, torch 2.10.0 /
mujoco 3.10.0 / numpy 2.4.6, measured 2026-07-07). So we transcribe those measured
per-seed success counts + distances + naive |Q| — the SAME honesty policy ch2.3's
MJX toy uses for its wall-clock table.

What we DO reuse from offline.py (so the error bars are the chapter's error bars)
--------------------------------------------------------------------------------
The one thing a toy must not fake is a confidence interval. So we do NOT
re-implement Wilson / Newcombe here: we exec a PREFIX of offline.py (up to the
`# --- region: eval ---` marker, under --smoke so its dataset build is tiny) in a
throwaway namespace and lift its OWN `wilson_ci` + `diff_ci` — the identical
functions the chapter grades with — then apply them to the measured per-seed
success counts. The regenerated diff CIs must reproduce meta.yaml's stated ranges
(seed0 +0.15..+0.34, seed1 +0.04..+0.23, seed2 +0.07..+0.25). We STOP if anything
drifts, or if the honest signal breaks: BC < AWAC every seed, every diff CI
excludes 0, and naive |Q| inflates on narrow data.

    Run:  .venv/bin/python site/scripts/vizdata/ch4_offline_primer.py
    Out:  curriculum/phase4_capstone/ch4_offline_primer/demo/vizdata.json
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OFFLINE_PY = REPO / "curriculum" / "phase4_capstone" / "ch4_offline_primer" / "offline.py"
OUT_JSON = REPO / "curriculum" / "phase4_capstone" / "ch4_offline_primer" / "demo" / "vizdata.json"

SEED = 0
CUT_MARKER = "# --- region: eval ---"

# ---------------------------------------------------------------------------
# meta.yaml reference_run (cpu; torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6;
# DEFAULT config: episodes 200, expert_frac 0.3, steps 8000, n_seeds 5 x
# eval_episodes 20 = 100 pooled rollouts; seeds 0,1,2; measured 2026-07-07).
# These are the MEASURED numbers — the honesty gate, transcribed verbatim.
# ---------------------------------------------------------------------------
SEEDS = [0, 1, 2]
N_POOL = 100  # n_seeds(5) * eval_episodes(20)

# The HEADLINE: BC vs AWAC on the SAME fixed mixed-quality dataset.
BC_SUCCESS = [0.03, 0.07, 0.05]        # per seed; mean 0.050
AWAC_SUCCESS = [0.27, 0.20, 0.21]      # per seed; mean 0.227
BC_DIST = [0.1542, 0.1468, 0.1437]     # mean final fingertip-to-target dist (m)
AWAC_DIST = [0.1089, 0.0997, 0.0889]
# meta's stated per-seed difference CIs (offline - BC), for the drift check.
META_DIFF_CI = [(0.15, 0.34), (0.04, 0.23), (0.07, 0.25)]

# Honest context: what BC must average over, and how far even AWAC still is.
BEHAVIOR_RETURN = {"expert": -2.3, "random": -16.0}   # the mixed quality (episode return)
RANDOM_DIST = 0.176      # a random policy leaves the fingertip ~0.176 m out
EXPERT_DIST = 0.0001     # the scripted expert reaches ~0.0001 m — the ceiling AWAC is FAR from
EXPERT_FRAC = 0.3        # dataset mix used for the headline
BETA = 0.3               # AWAC temperature (the exp(A/beta) weight)

# THE Break-It: --naive --expert_frac 1.0 (NARROW, expert-only data), seeds 0,1,2.
# naive maximize-Q OVERESTIMATES OOD actions -> |Q| inflates ~7x; AWAC stays bounded.
NAIVE_NARROW_ABS_Q = [7.29, 7.41, 6.74]    # naive final mean|Q| over data, per seed
AWAC_NARROW_ABS_Q = 1.08                    # AWAC on the SAME narrow data (seed 0), bounded
NAIVE_EXPERT_FRAC = 1.0

TOL_CI = 0.01   # diff CIs are rounded to 0.01 in meta; guard rounding only


def exec_offline_prefix() -> dict:
    """Exec offline.py up to the eval region, under --smoke, in an isolated
    namespace, and return its globals. --smoke shrinks the in-file dataset build
    to ~20 episodes (seconds), and we only lift the PURE stats functions
    `wilson_ci` + `diff_ci` — the exact code offline.py grades BC vs AWAC with."""
    src = OFFLINE_PY.read_text()
    prefix = src[: src.index(CUT_MARKER)]
    scratch = Path(tempfile.mkdtemp(prefix="ch4-viz-"))
    old_argv = sys.argv
    sys.argv = [str(OFFLINE_PY), "--seed", str(SEED), "--no-rerun",
                "--device", "cpu", "--smoke", "--out", str(scratch)]
    ns: dict = {"__file__": str(OFFLINE_PY), "__name__": "ch4_offline_vizgen"}
    try:
        exec(compile(prefix, str(OFFLINE_PY), "exec"), ns)  # noqa: S102 — our own trusted source
    finally:
        sys.argv = old_argv
    return ns


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def main() -> int:
    ns = exec_offline_prefix()
    wilson_ci = ns["wilson_ci"]   # offline.py's OWN Wilson interval (ch1.6 idiom)
    diff_ci = ns["diff_ci"]       # offline.py's OWN Newcombe difference CI

    bc_k = [round(s * N_POOL) for s in BC_SUCCESS]        # [3, 7, 5]
    awac_k = [round(s * N_POOL) for s in AWAC_SUCCESS]    # [27, 20, 21]
    bc_ci = [wilson_ci(k, N_POOL) for k in bc_k]
    awac_ci = [wilson_ci(k, N_POOL) for k in awac_k]
    # difference CI is offline(AWAC) - BC; excludes 0 (lo > 0) => the gap is real.
    diff = [diff_ci(ak, N_POOL, bk, N_POOL) for ak, bk in zip(awac_k, bc_k)]

    naive_mean = mean(NAIVE_NARROW_ABS_Q)
    inflation = naive_mean / AWAC_NARROW_ABS_Q

    # ------------------------------------------------------------------ honesty gate
    print("ch4 offline primer — regenerated error bars vs meta.yaml reference_run:")
    print(f"  BC   success/seed {BC_SUCCESS}  mean {mean(BC_SUCCESS):.3f}")
    print(f"  AWAC success/seed {AWAC_SUCCESS}  mean {mean(AWAC_SUCCESS):.3f}")
    for s, d, (mlo, mhi) in zip(SEEDS, diff, META_DIFF_CI):
        print(f"  seed{s} diff CI (AWAC-BC): {d[0]:+.2f}..{d[1]:+.2f}   (meta {mlo:+.2f}..{mhi:+.2f})")
    print(f"  naive|Q| narrow/seed {NAIVE_NARROW_ABS_Q}  mean {naive_mean:.2f}  "
          f"vs AWAC {AWAC_NARROW_ABS_Q}  -> {inflation:.1f}x")

    fail: list[str] = []
    # 1) the headline: BC < AWAC on success AND AWAC nearer the target, every seed.
    for s, bs, as_, bd, ad in zip(SEEDS, BC_SUCCESS, AWAC_SUCCESS, BC_DIST, AWAC_DIST):
        if not (bs < as_):
            fail.append(f"seed{s}: BC success {bs} !< AWAC {as_}")
        if not (ad < bd):
            fail.append(f"seed{s}: AWAC dist {ad} !< BC dist {bd}")
    if not (mean(BC_SUCCESS) < mean(AWAC_SUCCESS)):
        fail.append("mean BC success !< mean AWAC success")
    # 2) significance: every difference CI excludes 0 (lo > 0), matching meta.
    for s, d, (mlo, mhi) in zip(SEEDS, diff, META_DIFF_CI):
        if d[0] <= 0:
            fail.append(f"seed{s}: diff CI does not exclude 0 ({d[0]:+.3f}..{d[1]:+.3f})")
        if abs(round(d[0], 2) - mlo) > TOL_CI or abs(round(d[1], 2) - mhi) > TOL_CI:
            fail.append(f"seed{s}: diff CI {d[0]:.3f}..{d[1]:.3f} drifted from meta {mlo}..{mhi}")
    # 3) honest MODEST framing: AWAC still FAR from the expert ceiling.
    if not (mean(AWAC_DIST) > 10 * EXPERT_DIST):
        fail.append("AWAC dist not honestly far from expert — the modest framing broke")
    if not (mean(AWAC_DIST) < RANDOM_DIST):
        fail.append("AWAC dist not below the random baseline")
    # 4) the Break-It: naive |Q| inflates on NARROW data; AWAC stays bounded.
    for s, q in zip(SEEDS, NAIVE_NARROW_ABS_Q):
        if not (q > 3 * AWAC_NARROW_ABS_Q):
            fail.append(f"seed{s}: naive|Q| {q} did not inflate vs AWAC {AWAC_NARROW_ABS_Q}")
    if inflation < 4.0:
        fail.append(f"naive/AWAC |Q| inflation {inflation:.1f}x below the ~7x signal")

    if fail:
        print("\nSTOP — regenerated vizdata does NOT match meta.yaml reference_run:")
        for f in fail:
            print("  x " + f)
        return 1

    # ------------------------------------------------------------------- pack JSON
    def ci2(c: tuple[float, float]) -> list[float]:
        return [round(c[0], 4), round(c[1], 4)]

    data = {
        "provenance": {
            "source": "curriculum/phase4_capstone/ch4_offline_primer/offline.py",
            "generator": "site/scripts/vizdata/ch4_offline_primer.py",
            "seeds": SEEDS,
            "device": "cpu",
            "config": "offline.py defaults (episodes 200, expert_frac 0.3, steps 8000, "
                      "n_seeds 5 x eval_episodes 20 = 100 pooled rollouts); Break-It: "
                      "--naive --expert_frac 1.0 (narrow, expert-only)",
            "stack": "torch 2.10.0 / mujoco 3.10.0 / numpy 2.4.6",
            "measured": "2026-07-07",
            "note": "Success counts, distances, and naive |Q| are TRANSCRIBED from "
                    "meta.yaml's MEASURED reference_run (re-running all seeds + the "
                    "narrow Break-It is minutes of CPU and buys nothing — seed-0 CPU is "
                    "bitwise reproducible). The Wilson success CIs and the Newcombe "
                    "difference CIs are NOT re-implemented here: they are computed by "
                    "offline.py's OWN wilson_ci + diff_ci (the ch1.6 idiom the chapter "
                    "grades with), lifted by exec-ing offline.py's prefix. Honest MODEST "
                    "framing: the AWAC win over BC is real, significant (diff CI excludes "
                    "0 every seed) and seed-robust, but AWAC still reaches only ~0.09-0.11 "
                    "m — far from the scripted expert's ~0.0001 m. BC is a FAIR clone; the "
                    "win is the reward-aware extraction (weight exp(A/beta)). The Break-It: "
                    "naive maximize-Q with no data constraint inflates |Q| ~7x on NARROW "
                    "(expert-only) data (extrapolation error / OOD actions) while AWAC stays "
                    "bounded ~1.1; on the BROAD expert+random mix coverage keeps naive honest "
                    "— coverage-dependence is the lesson, and why 4.3's narrow correction "
                    "data needs the advantage constraint.",
        },
        "demo": "offline_bc_vs_awac",
        "seeds": SEEDS,
        "n_pool": N_POOL,
        "beta": BETA,
        "expert_frac": EXPERT_FRAC,
        "behavior_return": BEHAVIOR_RETURN,
        "baselines": {"random_dist": RANDOM_DIST, "expert_dist": EXPERT_DIST},
        # PANEL 1 — the headline: BC vs AWAC success on the SAME fixed dataset.
        "headline": {
            "bc": {
                "k": bc_k,
                "success": BC_SUCCESS,
                "ci": [ci2(c) for c in bc_ci],
                "dist": BC_DIST,
                "mean_success": round(mean(BC_SUCCESS), 4),
                "mean_dist": round(mean(BC_DIST), 4),
            },
            "awac": {
                "k": awac_k,
                "success": AWAC_SUCCESS,
                "ci": [ci2(c) for c in awac_ci],
                "dist": AWAC_DIST,
                "mean_success": round(mean(AWAC_SUCCESS), 4),
                "mean_dist": round(mean(AWAC_DIST), 4),
            },
            # AWAC - BC difference CI per seed; every one excludes 0 (lo > 0).
            "diff_ci": [ci2(c) for c in diff],
        },
        # PANEL 2 — the Break-It: naive maximize-Q vs AWAC on NARROW data.
        "naive": {
            "expert_frac": NAIVE_EXPERT_FRAC,
            "naive_abs_q": NAIVE_NARROW_ABS_Q,
            "naive_mean": round(naive_mean, 3),
            "awac_abs_q": AWAC_NARROW_ABS_Q,
            "inflation": round(inflation, 2),
            "broad_note": "On the BROAD expert+random mix, the random half COVERS the "
                          "action space, so the critic stays honest and even naive-maxQ "
                          "survives. The damage is coverage-dependent — narrow (expert-only) "
                          "correction data is exactly where the advantage constraint earns "
                          "its keep.",
        },
    }

    OUT_JSON.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    kb = OUT_JSON.stat().st_size / 1024
    print(f"\nwrote {OUT_JSON}  ({kb:.1f} KB)")
    print("OK — matches meta.yaml; BC < AWAC every seed, diff CI excludes 0 every seed, "
          f"naive|Q| inflates {inflation:.1f}x on narrow data (AWAC still far from expert).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
