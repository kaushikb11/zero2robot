"""zero2robot 5.5 — FAST: Turning Torques into Tokens (DCT -> Quantize -> BPE).

A modern VLA can speak actions two ways. ch1.5's FLOW head decodes a whole action
chunk CONTINUOUSLY, in a few Euler steps. The other way — pi0-FAST, OpenVLA — turns
the chunk into a short string of DISCRETE TOKENS and lets a language model emit them
autoregressively, so a robot's motion rides the SAME cross-entropy machinery as text.
This chapter builds that tokenizer from scratch. No policy, no env rollout — the
subject is the CODEC, in the ch1.7 mold: how an H x act_dim action chunk becomes a
handful of tokens, and how you invert it.

The pipeline, every piece from scratch: (1) DCT-II along the TIME axis (an orthonormal
cosine transform, built as a matrix) — a smooth trajectory's energy concentrates in a few
LOW-frequency coefficients; (2) QUANTIZE the coefficients to integers (round to a step) —
the tiny high-frequency ones round to exactly 0; (3) BPE the flattened integers (the
minbpe move, on ACTION tokens not text) — losslessly crushing the zero-runs into single
tokens. Then INVERT it (BPE-decode -> dequantize -> inverse DCT) and MEASURE error vs count.

HEADLINE (a deterministic codec claim): DCT->quantize->BPE encodes a chunk in a FRACTION of
the tokens naive per-step-per-dim binning needs, at comparable reconstruction error. The
DCT is orthonormal, so quantizing coefficients injects the same error ENERGY as quantizing
raw samples (Parseval) — but in the frequency domain the error concentrates, coefficients
round to 0, and BPE merges the zero-runs. Measured ~2x on these real robot chunks (only
PARTLY smooth — scripted experts switch phases; the smooth synthetic chunk compresses more).

Break it (--break time_domain): spend the budget in the TIME domain — keep every Nth action
and hold it — instead of dropping high frequencies. On a trajectory whose information is
spread across every timestep, that shortcut CRATERS the reconstruction (RMSE ~15x worse) and
makes the error JERKY (~200x): the DCT basis, not the BPE, is what rebuilds the motion.

Run it:      python curriculum/phase5_practitioner/ch5.5_fast/fast.py --seed 0
Break it:    python curriculum/phase5_practitioner/ch5.5_fast/fast.py --seed 0 --break time_domain
CI smoke:    python curriculum/phase5_practitioner/ch5.5_fast/fast.py --smoke --seed 0 --no-rerun
"""

# --- region: setup ---
import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Chapter artifacts run as loose scripts from the repo root; put the root on
# sys.path so `curriculum.common` resolves (same pattern as the other chapters).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from curriculum.common.device import banner  # noqa: E402
from curriculum.common.envs.aloha_cube.aloha_cube_env import AlohaCubeEnv  # noqa: E402
from curriculum.common.envs.aloha_cube.scripted_expert import ScriptedExpert as AlohaExpert  # noqa: E402
from curriculum.common.envs.pusht.pusht_env import PushTEnv  # noqa: E402
from curriculum.common.envs.pusht.scripted_expert import ScriptedExpert as PushtExpert  # noqa: E402
from curriculum.common.seeding import set_seed  # noqa: E402

FLOW_DECODE_STEPS = 5  # ch1.5's flow head reaches the chunk in ~this many Euler steps (the fork; see report)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out", type=Path, default=Path("outputs/ch5.5-fast"))
parser.add_argument("--horizon", type=int, default=24, help="H: timesteps per action chunk (the DCT runs along this axis; kept <= the shortest ALOHA demo)")
parser.add_argument("--episodes_per_task", type=int, default=8)  # cpu-laptop: 8 | smoke: 2
parser.add_argument("--q_scale", type=float, default=0.05,
                    help="quantization step on the [-1,1]-normalized coefficients; RMSE ~ q_scale/sqrt(12). smaller = finer = more tokens")
parser.add_argument("--q_max", type=int, default=127, help="clip quantized integers to [-q_max, q_max] (int8-ish alphabet)")
parser.add_argument("--keep_coeffs", type=int, default=None,
                    help="keep the K lowest-frequency DCT coeffs per dim, zero the rest (default: keep all H — quantization already zeros the highs)")
parser.add_argument("--num_merges", type=int, default=256, help="BPE merges to learn on the action-token corpus. T4/laptop: 256 | smoke: 32")
parser.add_argument("--seed", type=int, default=0, help="seeds the demos and the synthetic toy trajectory (the codec itself is deterministic)")
parser.add_argument("--break", dest="break_mode", choices=("time_domain",), default=None,
                    help="time_domain = spend the budget in the TIME domain (downsample+hold) instead of dropping high frequencies: reconstruction craters and goes jerky (measured)")
parser.add_argument("--downsample", type=int, default=2, help="--break time_domain: keep every Nth action and hold (zero-order) — the time-domain analog of a low-pass")
parser.add_argument("--smoke", action="store_true",
                    help="tiny hermetic CPU run for CI; two runs must produce byte-identical metrics.json")
parser.add_argument("--rerun", dest="rerun", action="store_true", default=True)
parser.add_argument("--no-rerun", dest="rerun", action="store_false", help="skip .rrd recording (CI smoke)")
args = parser.parse_args()

rng = set_seed(args.seed)  # numpy Generator for the synthetic toy; the DCT/quantize/BPE codec draws no randomness
if args.smoke:  # smoke pins everything the CI byte-compare depends on
    args.episodes_per_task, args.num_merges, args.horizon = 2, 32, 16
H = args.horizon
KEEP = args.keep_coeffs if args.keep_coeffs is not None else H
USE_DCT = args.break_mode != "time_domain"  # --break time_domain spends its budget in the time domain instead
banner("ch5.5-fast", device="cpu")  # a pure-numpy codec: no torch model, no GPU tier
args.out.mkdir(parents=True, exist_ok=True)
if args.rerun:
    import rerun as rr
    rr.init("zero2robot/ch5.5-fast", spawn=False)
    rr.save(str(args.out / "fast.rrd"))
# --- endregion ---

# --- region: dct ---
# The DCT-II as a plain matrix — no scipy, no FFT trick, just the definition. Row k of D is
# the k-th cosine basis vector (k=0 = DC average, higher k = faster wiggles). ORTHONORMAL
# scaling makes D @ D.T = I, so the inverse is just D.T and Parseval preserves L2 energy.
def dct_matrix(n: int) -> np.ndarray:
    """(n, n) orthonormal DCT-II matrix. coeffs = D @ signal; signal = D.T @ coeffs."""
    k = np.arange(n)[:, None]           # frequency index (rows)
    t = np.arange(n)[None, :]           # time index (cols)
    d = np.cos(np.pi * (2 * t + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    d[0] *= 1.0 / np.sqrt(2.0)          # DC row scaled so the basis is orthonormal
    return d


DCT = dct_matrix(H)


def dct(chunk: np.ndarray) -> np.ndarray:
    """(H, act_dim) trajectory -> coefficients, each dim's time series independently. Row 0
    is the average pose; rows toward H-1 are the fast wiggles."""
    return DCT @ chunk


def idct(coeffs: np.ndarray) -> np.ndarray:
    """Inverse: coefficients -> trajectory. Just the transpose (D is orthonormal) — no scaling
    to undo, no schedule (contrast ch1.4's DDPM)."""
    return DCT.T @ coeffs
# --- endregion ---

# --- region: quantize ---
# Turn real coefficients into small integers. round(x / step) is the whole quantizer.
# Two things make the grid mostly ZEROS (what BPE compresses): a smooth trajectory's
# high-frequency coeffs are ~0 and round to 0, plus an optional low-pass keeping only the
# K lowest frequencies per dim. In --break time_domain we quantize raw samples — no zeros.
def quantize(coeffs: np.ndarray, keep: int, q_scale: float, q_max: int) -> np.ndarray:
    q = np.clip(np.round(coeffs / q_scale), -q_max, q_max).astype(np.int64)
    if keep < len(q):
        q[keep:] = 0  # drop the high-frequency rows (a low-pass; only meaningful in the DCT domain)
    return q


def dequantize(q: np.ndarray, q_scale: float) -> np.ndarray:
    return q.astype(np.float64) * q_scale


def flatten_grid(q: np.ndarray) -> list[int]:
    """(H, act_dim) grid -> 1-D token list, DIM-MAJOR: each dim's H coeffs run low->high
    frequency and land contiguous, so every dim ends in a long zero-run — what BPE eats."""
    return q.T.reshape(-1).tolist()
# --- endregion ---

# --- region: bpe ---
# Byte-pair encoding, the minbpe move, on ACTION tokens. Repeatedly find the most frequent
# adjacent PAIR across the corpus and merge it into one new token id. Runs of zeros collapse
# first (0,0)->A, (A,A)->B, ... — a chunk's 20-zero tail becomes one or two tokens. BPE is
# LOSSLESS entropy coding on top of the lossy DCT+quantize: it changes the token COUNT, never
# the reconstruction. Ties broken by pair value -> deterministic.
def merge_seq(seq: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    out, i = [], 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(seq[i])
            i += 1
    return out


def train_bpe(seqs: list[list[int]], num_merges: int) -> dict[tuple[int, int], int]:
    """Learn merges. Returns an ordered dict (a,b)->new_id; order IS the apply order (encode
    replays it). Base ids are already 0..V-1 (see remap below)."""
    seqs = [list(s) for s in seqs]
    next_id = max((max(s) for s in seqs if s), default=-1) + 1
    merges: dict[tuple[int, int], int] = {}
    for _ in range(num_merges):
        counts: dict[tuple[int, int], int] = {}
        for s in seqs:
            for pair in zip(s, s[1:]):
                counts[pair] = counts.get(pair, 0) + 1
        if not counts:
            break
        best = max(counts, key=lambda p: (counts[p], p))  # most frequent; tie -> largest pair (deterministic)
        if counts[best] < 2:
            break  # nothing repeats: no merge would shorten the corpus
        merges[best] = next_id
        seqs = [merge_seq(s, best, next_id) for s in seqs]
        next_id += 1
    return merges


def bpe_encode(seq: list[int], merges: dict[tuple[int, int], int]) -> list[int]:
    for pair, new_id in merges.items():  # replay merges in learned order
        seq = merge_seq(seq, pair, new_id)
    return seq


def bpe_decode(seq: list[int], merges: dict[tuple[int, int], int]) -> list[int]:
    inv = {new_id: pair for pair, new_id in merges.items()}
    changed = True
    while changed:  # expand merged tokens back to their pairs until only base ids remain
        changed, out = False, []
        for t in seq:
            if t in inv:
                out.extend(inv[t])
                changed = True
            else:
                out.append(t)
        seq = out
    return seq
# --- endregion ---

# --- region: data ---
# Real robot ACTION CHUNKS — the same PushT + ALOHA scripted demos ch1.7's VLA dataset is
# built from, replayed in-process (deterministic on CPU, no rendering). We only need the
# ACTIONS, sliced into non-overlapping H-chunks.
TASKS = [
    {"name": "pusht", "env": PushTEnv, "expert": PushtExpert, "act_dim": PushTEnv.ACT_DIM},
    {"name": "aloha", "env": AlohaCubeEnv, "expert": AlohaExpert, "act_dim": AlohaCubeEnv.ACT_DIM},
]


def collect_chunks(task: dict, episodes: int, seed: int, horizon: int) -> list[np.ndarray]:
    env = task["env"]()
    chunks = []
    for e in range(episodes):
        env.reset(seed + e)
        expert = task["expert"](noise=0.0, seed=seed + e)
        acts, done = [], False
        while not done:
            action = expert.action(env)
            acts.append(np.asarray(action, np.float64)[: task["act_dim"]])
            _, _, done, _ = env.step(action)
        acts = np.stack(acts)
        for i in range(0, len(acts) - horizon + 1, horizon):  # non-overlapping chunks; drop the remainder
            chunks.append(acts[i : i + horizon])
    return chunks


def normalize_chunks(chunks: list[np.ndarray]) -> list[np.ndarray]:
    """Per-dim min-max to [-1,1] so one quantization step means the same thing across dims/tasks."""
    arr = np.stack(chunks)                                   # (N, H, act_dim)
    lo, hi = arr.min((0, 1)), arr.max((0, 1))
    span = np.where(hi - lo < 1e-6, 1.0, hi - lo)            # constant dims -> span 1 (no divide-by-zero)
    return list(2.0 * (arr - lo) / span - 1.0)


# The toy's SYNTHETIC chunk: two low-frequency sinusoids per dim + a whisper of noise — a
# self-contained SMOOTH trajectory (unlike the phase-switching scripted demos) whose DCT
# energy truly concentrates, so the spectrum viz is legible and truncation is cheap.
def synthetic_chunk(horizon: int, act_dim: int, gen: np.random.Generator) -> np.ndarray:
    t = np.linspace(0.0, 1.0, horizon)[:, None]
    f1, p1 = gen.uniform(0.4, 1.6, (1, act_dim)), gen.uniform(0.0, 2 * np.pi, (1, act_dim))
    f2, p2 = gen.uniform(0.4, 1.6, (1, act_dim)), gen.uniform(0.0, 2 * np.pi, (1, act_dim))
    traj = 0.7 * np.sin(2 * np.pi * f1 * t + p1) + 0.3 * np.sin(2 * np.pi * f2 * t + p2)
    traj += 0.015 * gen.standard_normal((horizon, act_dim))  # a whisper of high-frequency noise
    return np.clip(traj, -1.0, 1.0)


per_task = {t["name"]: normalize_chunks(collect_chunks(t, args.episodes_per_task, args.seed, H)) for t in TASKS}
chunks = [c for name in per_task for c in per_task[name]]   # pooled corpus (mixed embodiments share the integer alphabet)
n_elems = sum(c.size for c in chunks)
print(f"corpus: {len(chunks)} chunks (H={H}) [{len(per_task['pusht'])} pusht / {len(per_task['aloha'])} aloha], {n_elems} action numbers")
# --- endregion ---

# --- region: codec ---
# The whole FAST codec, one chunk at a time. encode: (DCT ->) quantize -> flatten. The
# reconstruction comes straight from the quantized integers (BPE is lossless). --break
# time_domain skips the DCT and spends the budget in TIME: keep every Nth action and
# zero-order-HOLD it, then quantize the staircase — no basis in which energy concentrates.
def downsample_hold(chunk_norm: np.ndarray, factor: int) -> np.ndarray:
    return np.repeat(chunk_norm[::factor], factor, axis=0)[: len(chunk_norm)]  # ZOH back to H rows


def encode_chunk(chunk_norm: np.ndarray) -> tuple[list[int], np.ndarray]:
    if USE_DCT:
        q = quantize(dct(chunk_norm), KEEP, args.q_scale, args.q_max)
    else:  # --break time_domain: quantize the held (downsampled) samples directly
        q = quantize(downsample_hold(chunk_norm, args.downsample), H, args.q_scale, args.q_max)
    return flatten_grid(q), q


def decode_chunk(q: np.ndarray) -> np.ndarray:
    grid = dequantize(q, args.q_scale)
    return idct(grid) if USE_DCT else grid  # break path is already a time-domain staircase


def error_jerk(recon: np.ndarray, target: np.ndarray) -> float:
    """Mean squared 2nd difference of the reconstruction ERROR — a smoothness meter for what
    the codec ADDED. The --break time_domain staircase inflates it ~200x (jerky residual)."""
    return float((np.diff(recon - target, n=2, axis=0) ** 2).mean())


# Encode every chunk, learn ONE BPE over the pooled action-token corpus, and remap the
# raw integers to a dense 0..V-1 alphabet first (so merged-token ids never collide).
raw_seqs, quants = zip(*(encode_chunk(c) for c in chunks))
alphabet = sorted({v for s in raw_seqs for v in s})
to_id = {v: i for i, v in enumerate(alphabet)}
base_seqs = [[to_id[v] for v in s] for s in raw_seqs]
merges = train_bpe(base_seqs, args.num_merges)
encoded = [bpe_encode(s, merges) for s in base_seqs]
assert all(bpe_decode(e, merges) == b for e, b in zip(encoded, base_seqs)), "BPE must round-trip (it is lossless)"

fast_tokens = sum(len(e) for e in encoded)                 # DCT->quantize->BPE: the method
naive_tokens = n_elems                                     # per-step-per-dim binning: one token per action number
compression_ratio = naive_tokens / fast_tokens
coeff_zero_frac = float(np.mean([(q == 0).mean() for q in quants]))  # why BPE wins: the share of quantized coeffs that are 0
# --- endregion ---

# --- region: measure ---
# Reconstruction error (normalized action units) and error smoothness, per chunk. The naive
# baseline quantizes raw samples at the SAME step; by Parseval the orthonormal DCT adds no
# error of its own, so FAST's and naive's error energies are comparable. Compression is the win.
sq_err_fast = sq_err_naive = 0.0
ejerk_fast = ejerk_naive = 0.0
for chunk_norm, q in zip(chunks, quants):
    recon = decode_chunk(q)
    q_naive = np.clip(np.round(chunk_norm / args.q_scale), -args.q_max, args.q_max)
    recon_naive = q_naive * args.q_scale                   # per-element time-domain dequant
    sq_err_fast += float(((recon - chunk_norm) ** 2).sum())
    sq_err_naive += float(((recon_naive - chunk_norm) ** 2).sum())
    ejerk_fast += error_jerk(recon, chunk_norm)            # smoothness of FAST's residual
    ejerk_naive += error_jerk(recon_naive, chunk_norm)     # smoothness of the white time-domain residual

fast_rmse = (sq_err_fast / n_elems) ** 0.5
naive_rmse = (sq_err_naive / n_elems) ** 0.5
ejerk_fast, ejerk_naive = ejerk_fast / len(chunks), ejerk_naive / len(chunks)
# The FORK (ch1.5 flow vs FAST): FAST decodes AUTOREGRESSIVELY, one LM step per token
# (~fast_tokens/len(chunks) sequential steps); flow decodes the whole chunk in
# FLOW_DECODE_STEPS Euler steps, flat. Tokens buy LM reuse + exact likelihood; flow buys a
# short constant decode.
fast_ar_steps = fast_tokens / len(chunks)
print(f"FAST : {fast_tokens} tokens  |  naive per-step binning: {naive_tokens} tokens  "
      f"-> {compression_ratio:.2f}x fewer  ({coeff_zero_frac:.0%} of quantized coeffs are 0)  "
      f"[break={args.break_mode or 'none'}]")
print(f"recon RMSE (normalized): FAST {fast_rmse:.4f}  |  naive per-step {naive_rmse:.4f}  (comparable — Parseval)")
print(f"error jerk (residual smoothness): FAST {ejerk_fast:.6f}  |  naive per-step {ejerk_naive:.6f}")
print(f"decode fork: FAST ~{fast_ar_steps:.1f} autoregressive steps/chunk  vs  "
      f"flow {FLOW_DECODE_STEPS} Euler steps (constant)")
# --- endregion ---

# --- region: report ---
metrics = {
    "break_mode": args.break_mode or "none",
    "coeff_zero_frac": round(coeff_zero_frac, 6),          # why BPE wins: fraction of quantized coeffs that are 0
    "compression_ratio": round(compression_ratio, 6),      # HEADLINE: naive_tokens / fast_tokens (> 1 = fewer FAST tokens)
    "fast_ar_steps_per_chunk": round(fast_ar_steps, 6),
    "fast_error_jerk": round(ejerk_fast, 8),               # break time_domain inflates this ~200x (jerky staircase)
    "fast_recon_rmse": round(fast_rmse, 6),                # break time_domain inflates this ~15x (motion falls apart)
    "fast_tokens": int(fast_tokens),
    "flow_decode_steps": int(FLOW_DECODE_STEPS),
    "horizon": int(H),
    "keep_coeffs": int(KEEP),
    "naive_error_jerk": round(ejerk_naive, 8),             # per-step fine binning reference
    "naive_recon_rmse": round(naive_rmse, 6),
    "naive_tokens": int(naive_tokens),
    "num_chunks": int(len(chunks)),
    "num_merges_learned": int(len(merges)),
    "q_scale": float(args.q_scale),
    "seed": int(args.seed),
    "smoke": bool(args.smoke),
    "vocab_base": int(len(alphabet)),
}
(args.out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

# demo/vizdata.json — the fast_codec visualizer's precomputed grid (static, like ch5.1: no
# live compute in the browser). ONE smooth synthetic chunk; for each (keep_coeffs, q_scale)
# the sliders expose the reconstruction, its token count (DCT+quant+BPE), and its RMSE.
toy = synthetic_chunk(H, 3, rng)                            # 3 dims: legible spectrum + overlay
toy_coeffs = dct(toy)
keep_grid = sorted({k for k in (H // 8, H // 4, H // 2, H) if 1 <= k <= H})
scale_grid = [args.q_scale * 0.5, args.q_scale, args.q_scale * 2.0]
settings = []
for keep in keep_grid:
    for scale in scale_grid:
        q = quantize(toy_coeffs, keep, scale, args.q_max)
        recon = idct(dequantize(q, scale))
        seq = [to_id.get(v, 0) for v in flatten_grid(q)]     # reuse the learned alphabet+merges for a live count
        tok = len(bpe_encode(seq, merges))
        rmse = float(((recon - toy) ** 2).mean() ** 0.5)
        settings.append({"keep_coeffs": int(keep), "q_scale": round(scale, 5),
                         "tokens": int(tok), "rmse": round(rmse, 5),
                         "recon": recon.round(4).tolist()})
vizdata = {
    "horizon": int(H), "act_dim": 3, "q_max": int(args.q_max),
    "naive_tokens": int(toy.size),                           # per-step-per-dim baseline for the toy chunk
    "trajectory": toy.round(4).tolist(),                    # (H, 3) original
    "dct_coeffs": toy_coeffs.round(4).tolist(),             # (H, 3) spectrum (rows = frequency)
    "keep_grid": [int(k) for k in keep_grid],
    "scale_grid": [round(s, 5) for s in scale_grid],
    "settings": settings,                                   # len(keep_grid) x len(scale_grid) precomputed reconstructions
}
(args.out / "demo").mkdir(parents=True, exist_ok=True)
(args.out / "demo" / "vizdata.json").write_text(json.dumps(vizdata) + "\n")

if args.rerun:
    for name, val in (("compression_ratio", compression_ratio), ("tokens/fast", fast_tokens),
                      ("tokens/naive", naive_tokens), ("rmse/fast", fast_rmse),
                      ("rmse/naive", naive_rmse), ("error_jerk/fast", ejerk_fast),
                      ("error_jerk/naive", ejerk_naive)):
        rr.log(f"codec/{name}", rr.Scalars([float(val)]), static=True)
    rr.log("toy/spectrum", rr.BarChart(np.abs(toy_coeffs[:, 0])), static=True)  # dim-0 DCT magnitudes
    toy_recon = idct(dequantize(quantize(toy_coeffs, KEEP, args.q_scale, args.q_max), args.q_scale))
    for k in range(H):  # the toy trajectory + its FAST reconstruction (default settings), dim 0
        rr.set_time("t", sequence=k)
        rr.log("toy/original", rr.Scalars([float(toy[k, 0])]))
        rr.log("toy/reconstruction", rr.Scalars([float(toy_recon[k, 0])]))

print(f"wrote {args.out / 'metrics.json'} + {args.out / 'demo' / 'vizdata.json'}")
if args.rerun:
    print(f"recording: {args.out / 'fast.rrd'} — open it with: rerun {args.out / 'fast.rrd'}")
# --- endregion ---
