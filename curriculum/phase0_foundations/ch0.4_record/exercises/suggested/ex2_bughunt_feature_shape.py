"""SUGGESTED exercise candidate (humans promote) — bug-hunt, ch0.4.

This is `build_features` — the schema every recorded frame is written against.
It is supposed to declare the SAME features as gen_demos.build_features and
pusht_env.py: `observation.state` is float32[10], `action` is float32[2]. It
doesn't. Exactly one shape is wrong, and it's the kind of wrong that doesn't
blow up here — it blows up three chapters later, when chapter 1.1's behavior
cloning tries to train on rows that are the wrong width, or (worse) trains
quietly on the wrong columns and never works.

Before you correct the number, write one sentence: if this nine-wide schema
shipped, what exactly breaks in chapter 1.1's training loop — and why wouldn't
it break here, where you wrote it?

Find the wrong number, fix it, and re-run checks.py until the schema matches the
training-data contract again. The observation layout is documented in
pusht_env.py and the chapter's Features region.

Run:  python ex2_bughunt_feature_shape.py
Estimated learner time: 15 minutes.
"""

import numpy as np

METADATA = {
    "type": "bug-hunt",
    "chapter": "ch0.4-record",
}

OBS_DIM = 10
ACT_DIM = 2
STATE_NAMES = [
    "pusher_x", "pusher_y", "tee_x", "tee_y", "sin_tee_yaw", "cos_tee_yaw",
    "target_x", "target_y", "sin_target_yaw", "cos_target_yaw",
]


def build_features() -> dict:
    """The dataset schema. observation.state MUST be float32[10] (ten names),
    action MUST be float32[2] — this is the contract chapter 1.1 trains on."""
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (9,),
            "names": STATE_NAMES,
        },
        "action": {
            "dtype": "float32",
            "shape": (ACT_DIM,),
            "names": ["pusher_vx", "pusher_vy"],
        },
    }


def state_shape() -> tuple:
    return tuple(build_features()["observation.state"]["shape"])


def action_shape() -> tuple:
    return tuple(build_features()["action"]["shape"])


if __name__ == "__main__":
    features = build_features()
    state = features["observation.state"]
    print(f"observation.state: dtype={state['dtype']} shape={state['shape']} "
          f"names={len(state['names'])}")
    ok = state_shape() == (OBS_DIM,) and len(state["names"]) == OBS_DIM
    # A real recorded observation has OBS_DIM entries; the schema must match it.
    sample = np.zeros(OBS_DIM, dtype=np.float32)
    print(f"a real observation has {sample.shape[0]} numbers; schema declares {state_shape()[0]} "
          f"-> {'MATCH' if ok else 'MISMATCH (fix build_features)'}")
