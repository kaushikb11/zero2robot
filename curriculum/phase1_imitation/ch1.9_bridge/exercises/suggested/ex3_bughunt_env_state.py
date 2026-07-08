"""SUGGESTED exercise candidate (humans promote) — bug-hunt (fast), ch1.9.

THE bridging bug, isolated. bridge.py maps our recorder's `observation.state`
onto ACT's state-only schema, which requires the key `observation.environment_
state`. The dangerous part is not the KEY — it is the TYPE. lerobot classifies a
feature by its `FeatureType`, and ACT's `validate_features()` accepts a state-only
policy ONLY if it finds a feature typed ENV (or an image). Rename the key but
leave the type STATE and one of two things happens: the policy is rejected, or
(worse, in looser setups) the environment-state token is silently never built
and the model trains on a truncated input — the "silently trains on garbage" trap.

Features here are the tiny (type, shape) tuples lerobot's `dataset_to_policy_
features` produces, so this gate is fast and self-contained (no torch, no lerobot).

Before you read the fix, write one sentence: the key is now
`observation.environment_state` and every shape still checks out — so why does
ACT still reject the policy when only the FeatureType is wrong?

FIND THE BUG in `bridge_state_feature` below, then fix it so the bridged feature
is typed ENV. `checks.py` gates on the signature (the bridged feature is still
STATE) and then verifies your fix. Estimated learner time: 15 minutes.
"""

METADATA = {"type": "bug-hunt", "chapter": "ch1.9-bridge", "fast": True}

# A feature is (feature_type, shape) — exactly what dataset_to_policy_features
# emits, minus the torch dependency. ACT wants the env state typed "ENV".
ENV_STATE_KEY = "observation.environment_state"


def bridge_state_feature(features: dict[str, tuple[str, tuple[int, ...]]]) -> dict:
    """Re-key our proprioceptive `observation.state` as ACT's `observation.
    environment_state`. Returns a NEW features dict; the action feature is
    untouched.

    BUG: this renames the key but carries the old feature UNCHANGED — so the
    bridged feature is still typed "STATE". ACT's validate_features() will not
    see an env-state input and rejects the policy (or trains on nothing).
    The fix: emit the feature typed "ENV" (keep the shape).
    """
    bridged = dict(features)
    state = bridged.pop("observation.state")           # (type, shape)
    bridged[ENV_STATE_KEY] = state
    return bridged


if __name__ == "__main__":
    feats = {"observation.state": ("STATE", (10,)), "action": ("ACTION", (6,))}
    out = bridge_state_feature(feats)
    print("bridged features:", out)
    kind = out[ENV_STATE_KEY][0]
    print(f"environment_state type: {kind}  (must be 'ENV' for ACT to accept it)")
    print("FIXED" if kind == "ENV" else "STILL BROKEN — ACT.validate_features() would reject this")
