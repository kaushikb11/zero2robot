"""SUGGESTED exercise candidate (humans promote) — reading/investigation, ch3.8.

This is the "you can now read any robot-learning paper" exercise: a guided READING of
the real frontier, not a code run. Nothing to execute — the deliverable is that you can
open a released VLA and recognize the pieces you built.

THE READING (STUDY tier; the chapter's read-the-real-thing block points at the pinned
sources): read the architecture description of NVIDIA's GR00T N1 (its "dual-system"
design) alongside Physical Intelligence's pi0. As you read, map each part onto what you
built: pi0's action head is the ch1.5 flow-matching head; its conditioning is the
ch1.7/1.8 VLM fusion. GR00T splits the policy into two systems running at two rates.

THE QUESTION. In GR00T N1's dual-system architecture, what is the relationship between
"System 2" and "System 1"?
  A) System 1 is a slow, high-level planner and System 2 is a fast reflex; System 1
     overrides System 2's actions frame by frame.
  B) They are two independent policies trained separately and averaged at inference,
     like an ensemble.
  C) System 2 is a slow vision-language model that produces a latent/representation
     which CONDITIONS System 1, a fast action head running at control rate — the same
     "fused representation -> action head" boundary probe.py hooks and probes.

Record your answer in PREDICTION. `checks.py` verifies your recorded choice (like every
predict-then-run gate, it only checks that you committed a reading, not your reasoning).

Then, in a sentence for yourself: if you had GR00T's real checkpoint, WHICH tensor
boundary would you hook to probe "what does System 2 tell System 1?" — and why is that
the same hook probe.py places on `policy.norm`? Estimated learner time: 30 minutes
(mostly reading).
"""

PREDICTION = None  # <- set to "A", "B", or "C" after the guided reading

METADATA = {"type": "reading-investigation", "chapter": "ch3.8-frontier",
            "choices": ["A", "B", "C"], "gate_before_run": True}


if __name__ == "__main__":
    if PREDICTION not in ("A", "B", "C"):
        raise SystemExit("Do the guided reading, then set PREDICTION to 'A', 'B', or 'C'.")
    print(f"(your recorded reading: {PREDICTION})")
    print("Check it: pytest curriculum/phase3_advanced/ch3.8_frontier/exercises/suggested/checks.py -k ex4")
