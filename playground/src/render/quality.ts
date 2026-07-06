// Quality-tier auto-degrade: a frame-time monitor that watches for sustained low
// FPS and drops the render tier so low-end devices stay interactive. This is the
// MIDDLE rung of the graceful-degradation ladder in playground/CLAUDE.md
// (full sim -> reduced render -> video fallback + Colab link).
//
// It NEVER touches physics, observations, recording, or the policy — only how
// the DISPLAY canvas is drawn: its backing-store resolution and how often it
// repaints. So it cannot affect the obs[10]/action[2] contract or determinism;
// the 96x96 recording canvas and the sim step are entirely independent of it.
//
// Pure + DOM-free (it consumes frame deltas and emits state via a callback), so
// it unit-tests under Node with a synthetic frame stream — no browser needed.

export type QualityTier = 'full' | 'reduced' | 'minimal';

export interface TierSpec {
  tier: QualityTier;
  /** Display-canvas backing-store resolution multiplier vs the base size. The
   *  on-screen (CSS) size is held constant; only the pixel count drops. */
  renderScale: number;
  /** Repaint every (frameSkip + 1)th frame. 0 = repaint every frame. */
  frameSkip: number;
}

/** Ordered best -> worst. The controller only ever steps one rung at a time. */
export const TIERS: readonly TierSpec[] = [
  { tier: 'full', renderScale: 1.0, frameSkip: 0 },
  { tier: 'reduced', renderScale: 0.66, frameSkip: 1 },
  { tier: 'minimal', renderScale: 0.5, frameSkip: 2 },
];

export interface QualityConfig {
  /** Below this smoothed FPS, sustained, the tier degrades one rung. */
  degradeFps: number;
  /** Above this smoothed FPS, sustained, the tier recovers one rung. */
  recoverFps: number;
  /** Consecutive sub-degrade-threshold frames required before a degrade. */
  degradeFrames: number;
  /** Consecutive above-recover-threshold frames required before a recover
   *  (kept well above degradeFrames as hysteresis, so the tier never flaps). */
  recoverFrames: number;
  /** Ignore the first N samples (JIT + WASM/asset warmup skews early frames). */
  warmupFrames: number;
  /** EMA factor for the frame-time estimate (0..1; higher = more responsive). */
  smoothing: number;
}

export const DEFAULT_QUALITY_CONFIG: QualityConfig = {
  degradeFps: 24,
  recoverFps: 55,
  degradeFrames: 90, // ~a couple seconds of sustained slowness before reacting
  recoverFrames: 240, // recover conservatively (hysteresis vs flapping)
  warmupFrames: 30,
  smoothing: 0.1,
};

export interface QualityState extends TierSpec {
  /** Current smoothed FPS estimate. */
  fps: number;
}

export class QualityController {
  private cfg: QualityConfig;
  private idx = 0; // index into TIERS (0 = full)
  private emaMs = 0;
  private samples = 0;
  private lowStreak = 0;
  private highStreak = 0;
  private onChange?: (s: QualityState) => void;

  constructor(
    opts: { config?: Partial<QualityConfig>; onChange?: (s: QualityState) => void } = {},
  ) {
    this.cfg = { ...DEFAULT_QUALITY_CONFIG, ...opts.config };
    this.onChange = opts.onChange;
  }

  get fps(): number {
    return this.emaMs > 0 ? 1000 / this.emaMs : 0;
  }

  get tier(): QualityTier {
    return TIERS[this.idx].tier;
  }

  get state(): QualityState {
    return { ...TIERS[this.idx], fps: this.fps };
  }

  /** Feed one frame delta (ms). Returns true iff the tier changed this sample;
   *  a change also fires the onChange callback with the new state. */
  sample(dtMs: number): boolean {
    // Clamp absurd deltas (tab-switch / debugger pause) so a single 5 s stall
    // cannot swing the EMA or the streak counters.
    const dt = Math.max(1, Math.min(dtMs, 1000));
    this.emaMs = this.emaMs === 0 ? dt : this.emaMs + this.cfg.smoothing * (dt - this.emaMs);
    this.samples += 1;
    if (this.samples <= this.cfg.warmupFrames) return false;

    const fps = this.fps;
    if (fps < this.cfg.degradeFps) {
      this.lowStreak += 1;
      this.highStreak = 0;
    } else if (fps > this.cfg.recoverFps) {
      this.highStreak += 1;
      this.lowStreak = 0;
    } else {
      this.lowStreak = 0;
      this.highStreak = 0;
    }

    if (this.lowStreak >= this.cfg.degradeFrames && this.idx < TIERS.length - 1) {
      this.idx += 1;
      this.lowStreak = 0;
      this.onChange?.(this.state);
      return true;
    }
    if (this.highStreak >= this.cfg.recoverFrames && this.idx > 0) {
      this.idx -= 1;
      this.highStreak = 0;
      this.onChange?.(this.state);
      return true;
    }
    return false;
  }
}
