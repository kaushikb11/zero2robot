// On-screen FPS meter (rolling 60-frame average) and physics-steps-per-second
// counter. Values are also mirrored onto window.__spikeMetrics so headless
// measurement (browse/playwright) can read them without OCR.
const WINDOW_FRAMES = 60;

declare global {
  interface Window {
    __spikeMetrics?: {
      fps: number;
      stepsPerSec: number;
      frames: number;
      totalSteps: number;
      simTime: number;
    };
  }
}

export class Meters {
  private frameTimes: number[] = [];
  private lastFrame = performance.now();
  private stepEvents: Array<{ t: number; n: number }> = [];
  private frames = 0;
  private totalSteps = 0;

  constructor(private el: HTMLElement) {}

  /** Call once per rAF frame with the number of physics steps taken. */
  tick(stepsThisFrame: number, simTime: number): void {
    const now = performance.now();
    this.frameTimes.push(now - this.lastFrame);
    this.lastFrame = now;
    if (this.frameTimes.length > WINDOW_FRAMES) this.frameTimes.shift();

    this.frames += 1;
    this.totalSteps += stepsThisFrame;
    this.stepEvents.push({ t: now, n: stepsThisFrame });
    while (this.stepEvents.length > 0 && now - this.stepEvents[0].t > 1000) {
      this.stepEvents.shift();
    }

    const meanFrameMs =
      this.frameTimes.reduce((a, b) => a + b, 0) / this.frameTimes.length;
    const fps = meanFrameMs > 0 ? 1000 / meanFrameMs : 0;
    const stepsPerSec = this.stepEvents.reduce((a, e) => a + e.n, 0);

    this.el.textContent =
      `fps          ${fps.toFixed(1)} (60-frame avg)\n` +
      `physics      ${stepsPerSec} steps/s\n` +
      `sim time     ${simTime.toFixed(1)} s`;

    window.__spikeMetrics = {
      fps: Math.round(fps * 10) / 10,
      stepsPerSec,
      frames: this.frames,
      totalSteps: this.totalSteps,
      simTime,
    };
  }
}
