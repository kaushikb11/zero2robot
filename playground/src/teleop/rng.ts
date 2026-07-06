// Small deterministic PRNG for browser episode reseeding. Same seed -> same
// start state, every time (so "reset" is reproducible and a learner can re-run
// a seed). mulberry32 is a compact, well-distributed 32-bit generator.
//
// HONEST LIMITATION: this is NOT numpy's PCG64. reset(seed) here reproduces the
// SAME SAMPLING PROCEDURE as pusht_env.reset (annulus block placement + uniform
// yaw + rejection-sampled pusher), but the exact positions for a given seed
// differ from the Python env. That is fine for teleop and for Phase-3 policy
// inference (a policy trained on pusht_env does not require the browser to spawn
// the identical start state — only that obs CONSTRUCTION matches, which is
// asserted separately by the obs-parity check). The Python env remains the
// authority for training-data determinism.
export class Rng {
  private s: number;

  constructor(seed: number) {
    // Mix the seed so small seeds (0, 1, 2) still produce well-separated streams.
    this.s = (seed ^ 0x9e3779b9) >>> 0;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    let t = (this.s += 0x6d2b79f5) >>> 0;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform float in [lo, hi). */
  uniform(lo: number, hi: number): number {
    return lo + (hi - lo) * this.next();
  }
}
