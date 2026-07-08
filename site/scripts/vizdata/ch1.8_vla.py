#!/usr/bin/env python3
"""Generate the ch1.8 "VLA rollout replay + fusion attention" toy's vizdata.

The site toy (site/src/components/toys/VlaRolloutToy.tsx) is an HONEST RECORDED
REPLAY, not live inference. It has two panels:

  1. THE ROLLOUT REPLAY: one recorded PushT episode of the trained tiny VLA,
     played back through the same top-down canvas the ch1.1 toy draws with
     (pusher pushing the T toward the target, step by step; play/pause/scrub).
     It LOOKS live but is a deterministic recording — a LIVE in-browser VLA is
     future work, BLOCKED on a sampler-aware contract-v2 runtime (the flow ODE
     sampler, ch1.5) PLUS offscreen RGB rendering to featurize frames, which
     MuJoCo-WASM does not expose, PLUS a tokenizer port (see demo/embed.yaml).
  2. THE FUSION ATTENTION: the CLS token's attention over the fused sequence
     [CLS, vision, state, tok_0..tok_15] — vla.py's `fusion/cls_attention` — as a
     bar viz: what the VLA "looks at". Honest caption: since --break blind ==
     sighted (meta.yaml), the random-init vision channel is NOT load-bearing.

PROVENANCE / honesty (the whole point of a generator over hand-typed JSON):
every number comes from the chapter artifact
`curriculum/phase1_imitation/ch1.8_vla/vla.py` run at seed 0 on CPU with the
DEFAULT config — the exact reference run recorded in that chapter's meta.yaml. We
do NOT modify vla.py (444 LOC cap): we import it as a module (with argv pinned to
the reference config and outputs pointed at a throwaway temp dir), which trains
the policy and writes its metrics.json, then we REUSE its trained `policy`,
frozen `encoder`, `sample_action`, and `encode_instruction` to replay one eval
episode (reproduced byte-for-byte from that episode's seed) while recording the
per-step pusher/tee poses, and to read the CLS fusion attention exactly as the
chapter logs it. We assert the freshly measured metrics match meta.yaml's
reference_run before writing; a mismatch aborts (never fabricate).

NO binaries are written or committed: the regenerated dataset .npz + the trained
checkpoint + any .rrd live in a temp dir deleted on exit, and the 96x96 camera
FRAMES are never persisted — the replay is pure top-down vector geometry
(pusher/tee x,y,yaw), exactly what the ch1.1 canvas already draws. vizdata.json
is small TEXT (subsampled frames).

Run:  .venv/bin/python site/scripts/vizdata/ch1.8_vla.py
      (default config regenerates ch1.7's data at 60 episodes/task then trains +
       evals the VLA; several minutes on CPU. --device cpu pins the deterministic
       tier so the recorded episode reproduces the chapter.)
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths
HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[3]  # site/scripts/vizdata/<file> -> repo root
CHAPTER = REPO_ROOT / "curriculum" / "phase1_imitation" / "ch1.8_vla"
ARTIFACT = CHAPTER / "vla.py"
OUT_JSON = CHAPTER / "demo" / "vizdata.json"

SEED = 0
WORLD_HALF_EXTENT_M = 0.45  # the PushT table half-size (playground viewport constant)
MAX_FRAMES = 90             # subsample the replay to at most this many kept frames

# ---------------------------------------------------------------- reference (meta.yaml)
# The chapter's recorded reference_run (seed 0, cpu, default config). We assert the
# freshly measured run reproduces these before writing — the honesty gate. CPU
# training is deterministic on a fixed machine, so seed-0 reproduces exactly.
REFERENCE = {
    "num_examples": 2984,
    "feature_dim": 64,
    "final_train_loss": 0.28249,
    "pusht_success_rate": 0.583333,
    "baseline_pusht_success_rate": 0.0,
    "aloha_success_rate": 0.0,
    "pusht_success_ci_lo": 0.319511,
    "pusht_success_ci_hi": 0.80674,
}
# CPU runs are deterministic on a fixed machine but torch build differences across
# machines can nudge the flow-mse a hair; the success RATES (the toy's honest claim)
# are integers-over-N and must land exactly on the reference.
LOSS_TOL = 5e-3
RATE_TOL = 1e-6


def load_vla_module(data_dir: Path, out_dir: Path):
    """Import vla.py as a module WITHOUT editing it: pin argv to the reference
    config (seed 0, cpu, no rerun) and point its outputs at throwaway temp dirs,
    so import-time execution trains + evals the reference policy. Returns the
    executed module namespace (mod.policy, mod.encoder, mod.sample_action, ...)."""
    argv_saved = sys.argv[:]
    sys.argv = [
        str(ARTIFACT),
        "--seed", str(SEED),
        "--device", "cpu",
        "--no-rerun",
        "--data", str(data_dir),
        "--out", str(out_dir),
    ]
    try:
        spec = importlib.util.spec_from_file_location("ch18_vla_artifact", ARTIFACT)
        mod = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        # Register under its name BEFORE exec so vla.py's module-scope
        # torch.save(policy, ...) (a full-object pickle of TinyVLA) can resolve the
        # class by reference — pickle re-imports the module to verify class identity,
        # which fails unless the synthetic module is discoverable in sys.modules.
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)  # trains + evals + writes out_dir/metrics.json
        return mod
    finally:
        sys.argv = argv_saved


def record_episode(mod, ep: int):
    """Replay ONE PushT eval episode of the trained policy, reproduced byte-for-byte
    from that episode's seed (mod.rollout reseeds env + sampler per episode, so this
    is order-independent and identical to what the chapter's eval measured), while
    recording the per-step pusher/tee poses. Mirrors mod.rollout — reusing mod's
    trained policy, frozen encoder, and ODE sampler — but captures geometry."""
    torch = __import__("torch")
    args, device = mod.args, mod.device
    ep_seed = 10_000 + args.seed + ep
    instruction = mod.manifest["tasks"][0]["templates"][0]  # the fixed PushT instruction
    tok_row = mod.encode_instruction(instruction)

    env = mod.PushTEnv()
    obs = env.reset(ep_seed)
    mod.gen.manual_seed(ep_seed)  # seed the sampler from the episode (as mod.rollout does)
    tok = torch.from_numpy(tok_row).to(device).unsqueeze(0)

    frames = []  # [pusher_x, pusher_y, tee_x, tee_y, tee_yaw] per env step

    def snap():
        px, py = env.pusher_pos
        tx, ty, tyaw = env.tee_pose
        frames.append([float(px), float(py), float(tx), float(ty), float(tyaw)])

    snap()  # the reset pose (frame 0)
    done, info, ret = False, {}, 0.0
    with torch.no_grad():
        while not done:
            feat = mod.encoder(
                torch.from_numpy(env.render_frame(mod.IMG_HW, mod.IMG_HW)[None]).to(device)
            )
            cond = mod.policy.fuse(tok, feat, torch.from_numpy(obs[None]).to(device))
            x = mod.sample_action(mod.policy, cond, args.flow_steps)
            action = (x * mod.act_std_t + mod.act_mean_t)[0, :2].cpu().numpy().clip(-1.0, 1.0)
            obs, reward, done, info = env.step(action)
            ret += reward
            snap()
    pos_err0 = float(np.hypot(frames[0][2], frames[0][3]))
    pos_err_final = float(np.hypot(frames[-1][2], frames[-1][3]))
    return {
        "episode": ep,
        "ep_seed": ep_seed,
        "instruction": instruction,
        "success": bool(info["success"]),
        "mean_return": round(ret, 4),
        "steps": len(frames) - 1,
        "pos_err0": round(pos_err0, 4),
        "pos_err_final": round(pos_err_final, 4),
        "frames_full": frames,
    }


def subsample(frames: list[list[float]], keep: int) -> list[list[float]]:
    """Evenly subsample to at most `keep` frames, ALWAYS keeping first and last so
    the replay starts at the reset pose and ends where the episode ended."""
    n = len(frames)
    if n <= keep:
        idxs = list(range(n))
    else:
        idxs = sorted({round(i * (n - 1) / (keep - 1)) for i in range(keep)})
    return [[round(v, 4) for v in frames[i]] for i in idxs]


def read_cls_attention(mod):
    """Read the CLS token's fused attention EXACTLY as the chapter logs
    `fusion/cls_attention`: fuse the PushT instruction with the first training
    example's frozen feature + state, then take the last block's CLS->sequence
    attention (mean over heads). Sequence layout is [CLS, vision, state, tok_0..15];
    softmax => sums to 1. Returns raw weights + labels + grouped totals."""
    torch = __import__("torch")
    tok0 = torch.from_numpy(
        mod.encode_instruction(mod.manifest["tasks"][0]["templates"][0])
    ).to(mod.device).unsqueeze(0)
    with torch.no_grad():
        mod.policy.fuse(tok0, mod.feats_t[:1], mod.states_t[:1])
    attn = mod.policy.blocks[-1].last_attn[0].cpu().double().numpy()  # (3 + MAX_TOKENS,)
    ids = tok0[0].cpu().numpy().tolist()
    vocab = mod.vocab
    pad_id = mod.PAD_ID
    # sequence labels: CLS, vision, state, then one per instruction token id
    tok_labels = [vocab[i] for i in ids]
    labels = ["CLS", "vision", "state"] + tok_labels
    weights = [round(float(w), 6) for w in attn.tolist()]
    # grouped totals (the honest comparison the panel leads with)
    cls_w = float(attn[0])
    vision_w = float(attn[1])
    state_w = float(attn[2])
    lang_slice = attn[3:]
    language_w = float(lang_slice.sum())
    # per-word language weights, real (non-pad) tokens only, for the word-level bars
    tokens = [
        {"word": vocab[ids[j]], "weight": round(float(lang_slice[j]), 6)}
        for j in range(len(ids))
        if ids[j] != pad_id
    ]
    return {
        "labels": labels,
        "weights": weights,
        "sum": round(float(attn.sum()), 6),
        "cls": round(cls_w, 6),
        "vision": round(vision_w, 6),
        "state": round(state_w, 6),
        "language": round(language_w, 6),
        "tokens": tokens,
        "n_language_tokens": len(tokens),
    }


def approx(got: float, want: float, tol: float) -> bool:
    return abs(float(got) - float(want)) <= tol


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="ch1.8-vizdata-") as tmp:
        tmp = Path(tmp)
        data_dir, out_dir = tmp / "data", tmp / "out"
        print(f"[ch1.8 vizdata] importing {ARTIFACT.name} at reference config "
              f"(seed 0, cpu, default) — this trains + evals; several minutes …")
        mod = load_vla_module(data_dir, out_dir)

        metrics = json.loads((out_dir / "metrics.json").read_text())

        # -------- honesty gate: the freshly measured run MUST match meta.yaml --------
        if metrics.get("smoke"):
            print("[ch1.8 vizdata] ABORT: ran in smoke config, not the reference default",
                  file=sys.stderr)
            return 1
        rate_keys = ("pusht_success_rate", "baseline_pusht_success_rate", "aloha_success_rate")
        for key in rate_keys:
            if not approx(metrics[key], REFERENCE[key], RATE_TOL):
                print(f"[ch1.8 vizdata] ABORT: {key} = {metrics[key]!r} != meta reference "
                      f"{REFERENCE[key]!r} (CPU seed-0 should reproduce exactly)", file=sys.stderr)
                return 1
        for key in ("num_examples", "feature_dim"):
            if int(metrics[key]) != int(REFERENCE[key]):
                print(f"[ch1.8 vizdata] ABORT: {key} = {metrics[key]!r} != meta reference "
                      f"{REFERENCE[key]!r}", file=sys.stderr)
                return 1
        if not approx(metrics["final_train_loss"], REFERENCE["final_train_loss"], LOSS_TOL):
            print(f"[ch1.8 vizdata] ABORT: final_train_loss = {metrics['final_train_loss']!r} "
                  f"!= meta reference {REFERENCE['final_train_loss']!r} (tol {LOSS_TOL})",
                  file=sys.stderr)
            return 1
        # the trained policy must actually beat the untrained reference (the lesson's rock)
        if not (metrics["pusht_success_rate"] > metrics["baseline_pusht_success_rate"] + 0.15):
            print("[ch1.8 vizdata] ABORT: trained PushT does not beat untrained by the "
                  "expected margin", file=sys.stderr)
            return 1
        print(f"[ch1.8 vizdata] reference match OK (PushT trained "
              f"{metrics['pusht_success_rate']:.4f} vs untrained "
              f"{metrics['baseline_pusht_success_rate']:.4f}; ALOHA "
              f"{metrics['aloha_success_rate']:.4f}; loss {metrics['final_train_loss']:.5f}; "
              f"{metrics['num_examples']} examples)")

        # -------- pick a representative SUCCESSFUL eval episode to replay --------
        # Scan the 12 eval episodes in order (each reproduced from its own seed);
        # take the first success so the replay shows the policy actually solving the
        # task. Deterministic. If none succeed (shouldn't, at 0.58), fall back to the
        # episode with the smallest final position error.
        n_eval = int(metrics["eval_episodes"])
        recorded = []
        chosen = None
        for ep in range(n_eval):
            r = record_episode(mod, ep)
            recorded.append(r)
            print(f"[ch1.8 vizdata]   ep {ep:2d} (seed {r['ep_seed']}): "
                  f"success={r['success']} steps={r['steps']} "
                  f"pos_err {r['pos_err0']:.3f} -> {r['pos_err_final']:.3f}")
            if r["success"]:
                chosen = r
                break
        if chosen is None:
            chosen = min(recorded, key=lambda r: r["pos_err_final"])
            print(f"[ch1.8 vizdata] no success in {n_eval} eps; replaying best "
                  f"(ep {chosen['episode']}, final pos_err {chosen['pos_err_final']:.3f})")

        frames = subsample(chosen["frames_full"], MAX_FRAMES)

        # -------- the fusion CLS attention (the money viz) --------
        attention = read_cls_attention(mod)
        if not approx(attention["sum"], 1.0, 1e-4):
            print(f"[ch1.8 vizdata] ABORT: CLS attention sums to {attention['sum']} (expected 1.0)",
                  file=sys.stderr)
            return 1
        print(f"[ch1.8 vizdata] CLS attention: vision {attention['vision']:.3f} · "
              f"state {attention['state']:.3f} · language {attention['language']:.3f} · "
              f"cls {attention['cls']:.3f} (sum {attention['sum']:.4f})")

        vizdata = {
            "provenance": (
                "curriculum/phase1_imitation/ch1.8_vla/vla.py, seed 0, --device cpu "
                "(deterministic tier), DEFAULT config (model_dim 64, layers 2, heads 4, "
                "hidden 128, flow_steps 6, epochs 200, episodes_per_task 60, "
                "eval_episodes 12). vla.py imported UNMODIFIED; its trained policy + frozen "
                "encoder + ODE sampler replay one eval episode (reproduced from that "
                "episode's seed) recording pusher/tee poses, and the CLS fusion attention "
                "is read exactly as the chapter's fusion/cls_attention. All success rates "
                "match this chapter's meta.yaml reference_run (measured 2026-07-06). This is "
                "a RECORDED REPLAY, not live inference: a live in-browser VLA is BLOCKED on a "
                "sampler-aware contract-v2 runtime (the ch1.5 flow ODE sampler) PLUS offscreen "
                "RGB rendering to featurize frames (MuJoCo-WASM exposes none) PLUS a tokenizer "
                "port (see demo/embed.yaml). NO frame/image binaries anywhere — the replay is "
                "vector geometry only. Regenerate: "
                ".venv/bin/python site/scripts/vizdata/ch1.8_vla.py"
            ),
            "seed": SEED,
            "live_blocked": (
                "This is a recorded replay, not live inference. A live in-browser VLA needs a "
                "sampler-aware contract-v2 runtime (an ODE flow sampler, strictly more than "
                "ch1.5's), offscreen RGB rendering to featurize each frame (MuJoCo-WASM does "
                "not expose it), and a tokenizer port — future work."
            ),
            "world_half_extent_m": WORLD_HALF_EXTENT_M,
            "meta": {
                "pusht_success_rate": metrics["pusht_success_rate"],
                "pusht_success_ci_lo": metrics["pusht_success_ci_lo"],
                "pusht_success_ci_hi": metrics["pusht_success_ci_hi"],
                "baseline_pusht_success_rate": metrics["baseline_pusht_success_rate"],
                "aloha_success_rate": metrics["aloha_success_rate"],
                # from meta.yaml reference_run: --break blind is UNCHANGED vs sighted
                # (0.583333 == 0.583333) — vision is NOT load-bearing. Not re-measured here.
                "break_blind_pusht_success_rate": 0.583333,
                "final_train_loss": metrics["final_train_loss"],
                "flow_steps": metrics["flow_steps"],
                "model_dim": metrics["model_dim"],
                "num_examples": metrics["num_examples"],
                "eval_episodes": metrics["eval_episodes"],
                "feature_dim": metrics["feature_dim"],
            },
            # the T-block half-extents (metres), so the toy draws the exact PushT geometry
            "tee": {
                "bar_half": [0.06, 0.015],
                "stem_half": [0.015, 0.045],
                "stem_offset_y": -0.06,
            },
            "target": {"x": 0.0, "y": 0.0, "yaw": 0.0},
            "rollout": {
                "episode": chosen["episode"],
                "ep_seed": chosen["ep_seed"],
                "instruction": chosen["instruction"],
                "success": chosen["success"],
                "mean_return": chosen["mean_return"],
                "steps": chosen["steps"],
                "kept": len(frames),
                "pos_err0": chosen["pos_err0"],
                "pos_err_final": chosen["pos_err_final"],
                # [pusher_x, pusher_y, tee_x, tee_y, tee_yaw] per kept frame (metres/rad)
                "frames": frames,
            },
            "attention": attention,
        }

        OUT_JSON.write_text(json.dumps(vizdata, indent=2) + "\n")
        size_kb = OUT_JSON.stat().st_size / 1024
        print(f"[ch1.8 vizdata] wrote {OUT_JSON.relative_to(REPO_ROOT)} "
              f"(episode {chosen['episode']}, {len(frames)} frames, "
              f"{size_kb:.1f} KB, text only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
